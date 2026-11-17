# Copyright 2026, The FlagOS Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""SGLang MoE top-k gating (fused softmax + top-k) in Triton — GCU rewrite.

In-place ABI (sgl_kernel.topk_softmax): softmax(gating_output) in fp32, then
the k largest experts per token row are written into the preallocated
topk_weights (fp32) and topk_ids (int32) buffers, ordered by descending
weight.  run() returns (topk_weights, topk_ids).

Why this diverges from the generic implementation
--------------------------------------------------
The generic kernel transcribes ATen CUDA's gatherTopK + bitonicSortKVInPlace
schedule and uses ``tl.gather`` (7 sites) for the in-register butterfly.  The
enflame GCU backend does not implement ``tl.gather``: a minimal kernel using
it fails in the Triton->GCU-IR pass pipeline ("Pipeline run failed: PassManager
execution failed"), and the full-size generic kernel dies with SIGSEGV (rc=-11)
inside make_gcuir — the baseline ABORT signature of this op.

Selection contract
------------------
Each of the K passes takes the row max of the remaining experts and masks
that expert to -inf.  Ties resolve to the LOWEST expert index — exactly ATen
gatherTopK's fill rule ([p > kth, ascending index] ++ [p == kth, ascending
index], truncated to k), which is also torch.topk's tie-break on the
reference device.  The tests compare ids and weights after sorting, so only
the selected expert set and the weight values must match.

Performance structure (L600 / GCU400, 48 SP)
--------------------------------------------
* The k-pass selection uses the hand-written leftmost-argmax double
  reduction (``tl.max`` + ``tl.where(sel == mv)`` + ``tl.min``).  The
  ``tl.max(..., return_indices=True)`` tuple-reduce lowering measures
  3.2-4.3x slower across the k-selection family on this backend, in the 1D
  row form (moe_fused_gate / gate_topk precedents, 2026-11-08) and in the
  2D row-batch form (re-measured 2026-11-12: 66.6 vs 20.9us at m=512).
* Selection runs on the unnormalised exponentials e = exp(x - row_max):
  softmax is strictly monotonic and dividing by the positive row sum cannot
  collapse distinct bf16 inputs onto equal fp32 probabilities, so
  argmax(e) selects the same experts as argmax(e / s).  The selected
  weights are produced by one [rows, K] division by the row sum after the
  loop — bitwise-equal to the per-lane division e/s at the selected
  positions, but 32x fewer division lanes for E=256/K=8.
* Two launch arms, dispatched on the row count (measured on L600, 48 SP,
  E=256/k=8 fp32 gating, kernel-mode do_bench median):
  - M <= 192: one row per program, 1-D [BLOCK_E] vectors, num_warps=1
    (4.76/4.86/8.94/17.8us at M=1/8/64/192).  The 2-D tile carries a
    ~17.5us fixed per-tile cost nearly independent of BLOCK_M in [2, 16],
    so 1-D wins below the ~192-row crossover (2026-11-12 sweep: 2-D at
    M=64 measures 17.6-18.7us vs 8.9us 1-D).
  - 192 < M <= 768: row batches [16, E], num_warps=4, row-masked
    (14.3/14.8us at M=512/768);
  - 768 < M <= 1536: [32, E], num_warps=4 (21.0/22.0us at M=1024/1536);
  - M > 1536: [64, E], num_warps=4 (33.5/58.9us at M=2048/4096), capped
    back to [32, E] when BLOCK_M * BLOCK_E > 16384 (DSM budget).
* Row-batch store structure (2026-11-12 round 2): per-k [BLOCK_M] vector
  stores replace the [BLOCK_M, BK] accumulator + 2-D masked store
  (-32% at M=512: 20.9 -> 14.3us; vllm topk_softmax round-2 precedent).
  Renormalisation divides the stored raw maxima by their own sum (the row
  sum s cancels: (mv/s) / sum(mv/s) == mv / sum(mv)) through a K-pass
  reload, so no [BLOCK_M, BK] accumulator is ever materialised.
* The row-batch arm is row-masked (no anchored slide): the reload-based
  renorm reads back what the same program stored, which is only race-free
  when no row is double-covered — the old slide's tail overlap would let
  a second program re-divide already-normalised weights.
* BLOCK_E floor of 128 on both arms: width-64 (and below) reductions are
  pathologically slow on this backend (vllm topk_softmax probe 2026-11-12:
  E=64 at BLOCK_E=64 is ~2.5x slower than padded to 128; here: E=64 M=512
  row-batch 17.2 -> 9.0us).  Eligible pow-2 E < 128 load through a column
  mask; E=256/512 keep the maskless EVEN path.
* The row-batch arm is adopted per configuration only after a first-call
  cross-check against the row kernel (exact ids + allclose weights); a
  compile failure or mismatch permanently falls back to the row kernel for
  that configuration in this process — defence against silent miscompiles
  of the 2-D reduction family (compiler_problems P10) and against
  environment-level regressions.
"""

import re

import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# GCU dialect shim (ported verbatim from _enflame/ops/bmm_chunk.py /
# extend_attention.py, production-verified on the vllm side at d1a7005).
#
# The 2026-11-07 20:22 FlagTree wheel emits `memref_ext.*` / `gcu.matmul`
# ops with trailing zero operand segments that the external
# `gcu-compiler-opt` (2026-10-20) rejects on fresh JIT compiles
# (`slice_start/slice_pad_start ... size mismatch in attribute
# conversion`).  Carrying the shim inside this module keeps the 2-D row
# batch arm self-contained instead of relying on a sibling op file being
# imported first.  It is idempotent and safe alongside the copies in other
# modules.
# ---------------------------------------------------------------------------

_GCU_SENTINEL = "-9223372036854775808"


def _fix_gcu_attr(attr_match, line, op_name):
    inner = attr_match.group(1)
    seg_match = re.search(r"operandSegmentSizes = array<i32: ([^>]*)>", inner)
    if op_name == "memref_ext.memset_start":
        return ""
    if op_name.startswith("memref_ext.") or op_name == "gcu.matmul":
        if seg_match:
            segs = [s.strip() for s in seg_match.group(1).split(",")]
            while len(segs) > 1 and segs[-1] == "0":
                segs.pop()
            new_seg = "array<i32: " + ", ".join(segs) + ">"
            rest = re.sub(r"operandSegmentSizes = array<i32:[^>]*>", "", inner)
            rest = rest.strip().strip(",").strip()
            if rest:
                return f" <{{operandSegmentSizes = {new_seg}, {rest}}}>"
            return f" <{{operandSegmentSizes = {new_seg}}}>"
        return attr_match.group(0)
    if (
        op_name in ("memref.reinterpret_cast", "memref.subview")
        and not seg_match
    ):
        opers = re.search(r"\(([^()]*)\)\s*<\{", line)
        n_ops = (
            len([x for x in opers.group(1).split(",") if x.strip()])
            if opers
            else 1
        )
        offs = re.search(r"static_offsets = array<i64: ([^>]*)>", line)
        sizes = re.search(r"static_sizes = array<i64: ([^>]*)>", line)
        strides = re.search(r"static_strides = array<i64: ([^>]*)>", line)
        n_off = offs.group(1).count(_GCU_SENTINEL) if offs else 0
        n_size = sizes.group(1).count(_GCU_SENTINEL) if sizes else 0
        n_stride = strides.group(1).count(_GCU_SENTINEL) if strides else 0
        if 1 + n_off + n_size + n_stride == n_ops:
            seg = f"array<i32: 1, {n_off}, {n_size}, {n_stride}"
            rest = inner.strip()
            if rest:
                return f" <{{operandSegmentSizes = {seg}, {rest}}}>"
            return f" <{{operandSegmentSizes = {seg}}}>"
    return attr_match.group(0)


def _fix_gcu_dialect_line(line):
    op_match = re.search(r'"([a-z0-9_.]+)"\(', line)
    if not op_match:
        return line
    op_name = op_match.group(1)
    if not (
        op_name.startswith("memref_ext.")
        or op_name.startswith("gcu.")
        or op_name in ("memref.reinterpret_cast", "memref.subview")
    ):
        return line
    return re.sub(
        r" <\{([^}]*)\}>",
        lambda m: _fix_gcu_attr(m, line, op_name),
        line,
        count=1,
    )


def _patch_gcu_compiler_opt():
    import triton.backends.enflame.toolkit as _toolkit

    if getattr(_toolkit.gcu_compiler_opt, "_flaggems_dialect_shim", False):
        return
    _orig = _toolkit.gcu_compiler_opt

    def _shim(content, *args):
        if isinstance(content, str) and "operandSegmentSizes" in content:
            content = "\n".join(
                _fix_gcu_dialect_line(line) for line in content.split("\n")
            )
        return _orig(content, *args)

    _shim._flaggems_dialect_shim = True
    _toolkit.gcu_compiler_opt = _shim


# _patch_gcu_compiler_opt()  # DISABLED 2026-11-09: 私有 GCUIR shim 实测引入回归,详见 commit b333be6


# ---------------------------------------------------------------------------
# Row kernel: one token row per program (1-D [BLOCK_E] vectors).
# ---------------------------------------------------------------------------


@triton.jit
def _topk_softmax_row_kernel(
    gating_ptr,
    w_ptr,
    id_ptr,
    E,
    stride_gm,
    stride_ge,
    stride_wm,
    stride_wk,
    stride_im,
    stride_ik,
    K: tl.constexpr,
    BK: tl.constexpr,
    BLOCK_E: tl.constexpr,
    EVEN_E: tl.constexpr,
    RENORM: tl.constexpr,
):
    row = tl.program_id(0)
    offs_e = tl.arange(0, BLOCK_E)
    if EVEN_E:
        x = tl.load(gating_ptr + row * stride_gm + offs_e * stride_ge).to(
            tl.float32
        )
        row_max = tl.max(x, axis=0)
        e = tl.exp(x - row_max)
        sel = e
    else:
        e_mask = offs_e < E
        x = tl.load(
            gating_ptr + row * stride_gm + offs_e * stride_ge,
            mask=e_mask,
            other=float("-inf"),
        ).to(tl.float32)
        row_max = tl.max(x, axis=0)
        e = tl.where(e_mask, tl.exp(x - row_max), 0.0)
        sel = tl.where(e_mask, e, float("-inf"))

    s = tl.sum(e, axis=0)
    s = tl.where(s > 0.0, s, 1.0)

    offs_k = tl.arange(0, BK)
    acc_w = tl.zeros([BK], dtype=tl.float32)
    acc_i = tl.zeros([BK], dtype=tl.int32)
    for r in tl.static_range(K):
        mv = tl.max(sel, axis=0)
        mi = tl.min(tl.where(sel == mv, offs_e, BLOCK_E), axis=0)
        acc_w = tl.where(offs_k == r, mv, acc_w)
        acc_i = tl.where(offs_k == r, mi, acc_i)
        sel = tl.where(offs_e == mi, float("-inf"), sel)

    acc_w = acc_w / s
    if RENORM:
        acc_w = acc_w / (tl.sum(acc_w, axis=0) + 1e-20)

    k_mask = offs_k < K
    tl.store(w_ptr + row * stride_wm + offs_k * stride_wk, acc_w, mask=k_mask)
    tl.store(id_ptr + row * stride_im + offs_k * stride_ik, acc_i, mask=k_mask)


# ---------------------------------------------------------------------------
# Row-batch kernel: [BLOCK_M, BLOCK_E] tiles, row-masked (no slide), per-k
# [BLOCK_M] vector stores with reload-based renormalisation.
# ---------------------------------------------------------------------------


@triton.jit
def _topk_softmax_rows_kernel(
    gating_ptr,
    w_ptr,
    id_ptr,
    M,
    E,
    stride_gm,
    stride_ge,
    stride_wm,
    stride_wk,
    stride_im,
    stride_ik,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_E: tl.constexpr,
    MASK_COLS: tl.constexpr,
    RENORM: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    rmask = rows < M
    cols = tl.arange(0, BLOCK_E)
    if MASK_COLS:
        # BLOCK_E is BLOCK_E-floor-padded above E: pad columns load -inf and
        # never win a round, exactly like the row kernel's masked path.
        lmask = rmask[:, None] & (cols < E)[None, :]
        x = tl.load(
            gating_ptr + rows[:, None] * stride_gm + cols[None, :] * stride_ge,
            mask=lmask,
            other=float("-inf"),
        ).to(tl.float32)
    else:
        x = tl.load(
            gating_ptr + rows[:, None] * stride_gm + cols[None, :] * stride_ge,
            mask=rmask[:, None],
            other=float("-inf"),
        ).to(tl.float32)

    row_max = tl.max(x, axis=1)
    e = tl.exp(x - row_max[:, None])
    if MASK_COLS:
        e = tl.where((cols < E)[None, :], e, 0.0)
    s = tl.sum(e, axis=1)
    s = tl.where(s > 0.0, s, 1.0)

    sel = e
    if MASK_COLS:
        sel = tl.where((cols < E)[None, :], sel, float("-inf"))
    mvs = tl.zeros([BLOCK_M], dtype=tl.float32)
    for r in tl.static_range(K):
        mv = tl.max(sel, axis=1)
        mi = tl.min(
            tl.where(sel == mv[:, None], cols[None, :], BLOCK_E), axis=1
        )
        tl.store(w_ptr + rows * stride_wm + r * stride_wk, mv, mask=rmask)
        tl.store(id_ptr + rows * stride_im + r * stride_ik, mi, mask=rmask)
        if RENORM:
            mvs += mv
        sel = tl.where(cols[None, :] == mi[:, None], float("-inf"), sel)

    # Renormalisation divides the stored raw maxima by their own sum; the
    # softmax denominator s cancels ((mv/s) / sum(mv/s) == mv / sum(mv)), so
    # the loop stores unnormalised maxima and this pass applies one divide.
    # Reading back what this program just wrote is race-free because rows
    # are never double-covered (no anchored slide).
    if RENORM:
        denom = tl.where(mvs > 0.0, mvs, 1.0)
    else:
        denom = s
    for r in tl.static_range(K):
        idx = rows * stride_wm + r * stride_wk
        val = tl.load(w_ptr + idx, mask=rmask, other=0.0)
        tl.store(w_ptr + idx, val / denom, mask=rmask)


# ---------------------------------------------------------------------------
# Host dispatch
# ---------------------------------------------------------------------------

_MAX_GRID_X = 32768

# Per-process first-call validation state for the row-batch kernel:
# key -> True (validated, use directly) / False (fall back to the row kernel).
_ROWS_KERNEL_STATE = {}


def _rows_kernel_eligible(num_experts):
    # E must be a power of two (>= 16, <= 512 to keep the fp32 tile set
    # inside a comfortable DSM budget at BLOCK_M=32).  Powers of two below
    # the 128-wide BLOCK_E floor load through the column mask.
    return 16 <= num_experts <= 512 and num_experts == triton.next_power_of_2(
        num_experts
    )


# Width-64-and-below reductions are pathologically slow on this backend, so
# both arms pad BLOCK_E up to 128 (vllm topk_softmax probe 2026-11-12; here:
# E=64 row-batch 17.2 -> 9.0us).
_MIN_BLOCK_E = 128


def _launch_rows_kernel(
    gating,
    weights,
    ids,
    rows,
    num_experts,
    k,
    block_m,
    renorm,
    num_warps,
):
    stride_gm = gating.stride(0)
    stride_ge = gating.stride(1)
    stride_wm = weights.stride(0)
    stride_wk = weights.stride(1)
    stride_im = ids.stride(0)
    stride_ik = ids.stride(1)
    block_e = max(triton.next_power_of_2(num_experts), _MIN_BLOCK_E)
    mask_cols = 1 if block_e != num_experts else 0
    key = (
        rows,
        num_experts,
        k,
        renorm,
        block_m,
        block_e,
        gating.dtype,
        stride_gm,
        stride_ge,
        stride_wm,
        stride_wk,
        stride_im,
        stride_ik,
    )
    grid = (triton.cdiv(rows, block_m),)
    state = _ROWS_KERNEL_STATE.get(key)
    if state is True:
        _topk_softmax_rows_kernel[grid](
            gating,
            weights,
            ids,
            rows,
            num_experts,
            stride_gm,
            stride_ge,
            stride_wm,
            stride_wk,
            stride_im,
            stride_ik,
            K=k,
            BLOCK_M=block_m,
            BLOCK_E=block_e,
            MASK_COLS=mask_cols,
            RENORM=renorm,
            num_warps=num_warps,
            num_stages=1,
        )
        return True
    if state is False:
        return False

    # First call for this configuration: compile+run the row-batch kernel
    # into stride-matched scratch buffers and cross-check it against the
    # row kernel (generic-semantics anchor) before adopting it.  A compile
    # failure (e.g. an environment-level backend regression) or a value
    # mismatch permanently disables the row-batch path for this
    # configuration in this process.  The user buffers already hold the
    # row kernel's validated output, so no copy-back is needed on success.
    try:
        w_ref = torch.empty_strided(
            weights.shape,
            weights.stride(),
            dtype=weights.dtype,
            device=weights.device,
        )
        i_ref = torch.empty_strided(
            ids.shape, ids.stride(), dtype=ids.dtype, device=ids.device
        )
        _topk_softmax_rows_kernel[grid](
            gating,
            w_ref,
            i_ref,
            rows,
            num_experts,
            stride_gm,
            stride_ge,
            stride_wm,
            stride_wk,
            stride_im,
            stride_ik,
            K=k,
            BLOCK_M=block_m,
            BLOCK_E=block_e,
            MASK_COLS=mask_cols,
            RENORM=renorm,
            num_warps=num_warps,
            num_stages=1,
        )
    except Exception:
        _ROWS_KERNEL_STATE[key] = False
        return False
    _topk_softmax_row_kernel[(rows,)](
        gating,
        weights,
        ids,
        num_experts,
        stride_gm,
        stride_ge,
        stride_wm,
        stride_wk,
        stride_im,
        stride_ik,
        K=k,
        BK=max(8, triton.next_power_of_2(k)),
        BLOCK_E=max(16, triton.next_power_of_2(num_experts)),
        EVEN_E=1,
        RENORM=renorm,
        num_warps=1,
        num_stages=1,
    )
    ok = torch.equal(i_ref, ids) and torch.allclose(
        w_ref, weights, atol=1e-4, rtol=1e-4
    )
    _ROWS_KERNEL_STATE[key] = bool(ok)
    return bool(ok)


def topk_softmax(topk_weights, topk_ids, gating_output, renormalize=False):
    k = int(topk_ids.shape[-1])
    M = int(gating_output.shape[0])
    E = int(gating_output.shape[1])
    if M == 0 or E == 0 or k == 0:
        return topk_weights, topk_ids

    block_e = triton.next_power_of_2(E)
    if E >= 16:
        # Width-64-and-below reductions are pathologically slow; pad up to
        # the same 128 floor the row-batch arm uses (masked EVEN_E=0 path).
        block_e = max(block_e, _MIN_BLOCK_E)
    else:
        block_e = max(block_e, 16)
    block_k = max(8, triton.next_power_of_2(k))
    even_e = 1 if E == block_e else 0
    renorm = 1 if renormalize else 0
    rows_eligible = _rows_kernel_eligible(E)

    for row_start in range(0, M, _MAX_GRID_X):
        rows = min(_MAX_GRID_X, M - row_start)
        gating = gating_output[row_start : row_start + rows]
        weights = topk_weights[row_start : row_start + rows]
        ids = topk_ids[row_start : row_start + rows]

        launched = False
        if rows_eligible and rows > 192:
            # Measured on L600 (48 SP), E=256/k=8 fp32 gating (kernel-mode
            # do_bench median, 2026-11-12): the row kernel wins through
            # m=192 (8.9/17.8us at m=64/192; the 2-D tile fixed cost is
            # ~17.5us); row batches win from m=256 on.  Band bounds keep
            # cdiv(rows, BLOCK_M) inside one 48-program dispatch wave where
            # possible (16 -> <=48 tiles at m<=768; 32 -> <=48 tiles at
            # m<=1536; 64 -> 1.33 waves at m=4096, still fastest).
            if rows <= 768:
                block_m, rows_warps = 16, 4
            elif rows <= 1536:
                block_m, rows_warps = 32, 4
            else:
                block_m, rows_warps = 64, 4
            tile_e = max(triton.next_power_of_2(E), _MIN_BLOCK_E)
            if block_m * tile_e > 16384:
                # Keep the fp32 tile set (x/e/sel + reduction temporaries)
                # inside the ~448KB dual-program DSM budget at BLOCK_E=512.
                block_m, rows_warps = 32, 4
            launched = _launch_rows_kernel(
                gating,
                weights,
                ids,
                rows,
                E,
                k,
                block_m,
                renorm,
                rows_warps,
            )
        if launched:
            continue

        _topk_softmax_row_kernel[(rows,)](
            gating,
            weights,
            ids,
            E,
            gating.stride(0),
            gating.stride(1),
            weights.stride(0),
            weights.stride(1),
            ids.stride(0),
            ids.stride(1),
            K=k,
            BK=block_k,
            BLOCK_E=block_e,
            EVEN_E=even_e,
            RENORM=renorm,
            num_warps=1 if block_e <= 512 else 4,
            num_stages=1,
        )
    return topk_weights, topk_ids


__all__ = ["topk_softmax"]
