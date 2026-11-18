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

"""Gated delta-rule linear attention recurrence (fused recurrent GDN), Triton.

One program per (batch, value-head, V-block). The fp32 state tile
[BLOCK_V, BLOCK_K] lives in registers for the whole sequential T walk, so the
state is touched in global memory exactly twice: the prologue load and the
epilogue store. The per-step loads of k, q, v, g and beta are issued one
timestep ahead so their memory latency overlaps the previous step's reduction
chain, which is the dominant stall on the serial T recurrence.

Enflame GCU adaptations (generic semantics preserved):

1. The flat 1-D grid folds the former (cdiv(V, BLOCK_V), B * HV) launch into
   grid.x <= 65535 / grid.y <= 255 / grid.z <= 255 (the generic form hits the
   hardware grid.y limit of 255 as soon as B * HV >= 256, e.g. B=8 decode).
   The kernel recombines the flat program id and guards ``pid < num_programs``.
2. Wave-aware BLOCK_V widening: the serial per-sequence recurrence makes wall
   clock ~ ceil(programs / 48) waves x per-program op chain, so multi-wave
   contiguous shapes widen the state tile (up to 128 rows, <= 16384 fp32
   elements). The form inside the widened branch is T-dependent: T == 1 keeps
   the two [BLOCK_V, BLOCK_K] product-sum trees (faster per program on a
   single step), while T > 1 folds them into one ``tl.dot`` against a
   [BLOCK_K, 16] operand whose column 0 holds k and column 1 holds
   q * scale; the output follows the algebraically identical
   ``out = <state_gated, qs> + vcorr * <k, qs>``. Bench-core kernel-mode
   medians vs the grid-fixed original: decode B=1 45->8.9us, B=8 312->46.6us,
   B=64 2496->340us, prefill B=1/T=2048 66.8->11.6ms (5.1-7.3x). Single-wave,
   strided, narrow-K and dot-unavailable multi-token shapes keep the original
   narrow tree path. All dispatch is host-side from metadata only.
3. Toolchain-compat shim (2026-11-07 20:22 build skew): first compile of
   canonical 2-D tile accesses (``memref_ext.slice_*``) and ``tl.dot``
   (``gcu.matmul``) fails on this build; a scoped retry shim rewrites those
   encodings back to the old form and retries (the delivered
   chunk_local_cumsum_vector pattern). Fallback-only and scoped to this
   module's kernel names. Additionally, a zeros-initialized state tile fed to
   ``tl.dot`` fails the in-process make_gcuir pipeline entirely (memref.alloc
   symbol error), so the None-state dot path seeds the tile from a cached
   zero buffer read as a live 2-D load (values identical to zeros). Remove
   both workarounds once the toolchain is re-synced.
4. T == 1 with no initial state collapses the whole state machine: the state
   is zero before the single step, so ``o = beta * v * <k, q * scale>`` (and
   the optional final state is ``outer(beta * v, k)``). A dedicated constexpr
   arm computes this with pure vector work -- the [BLOCK_V, BLOCK_K] tile is
   never materialized, removing the dominant per-program cost of the decode
   family (bench-core kernel-mode medians, 2026-11-12: decode B=8
   46.4->21.9us, B=64 340->145.7us, B=256 1.35ms->573.5us, B=1 9.1->~6us).
   T == 1 with an initial state keeps the tree path (state is non-zero).
5. Dispatch hardening (2026-11-12): shapes that pass the wave guard but take
   neither inner branch (T > 1 with BLOCK_K < 128) previously reached the
   launch with ``num_stages`` unbound (UnboundLocalError, probe-found on
   (2,64,2,8,64,64)); the default is now set before the guard so they keep
   the narrow tile with num_stages = 3 as documented.
"""

import re

import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# Toolchain-compat shim (2026-11-07 20:22 skew): retry-path GCUIR downgrade
# scoped to this module's kernels. See compiler_issues/vllm_moe_sum.md for the
# family analysis and chunk_local_cumsum_vector.py for the delivered pattern.
# ---------------------------------------------------------------------------
_COMPAT_SCOPED_KERNELS = ("_gdn_fwd_kernel",)


def _downgrade_gcuir_for_external_tool(text):
    """Rewrite new-format GCUIR op encodings back to the old external-tool form."""

    def _strip_trailing_zero_segments(match):
        entries = [entry.strip() for entry in match.group(2).split(",")]
        while len(entries) > 1 and entries[-1] == "0":
            entries.pop()
        return match.group(1) + ", ".join(entries) + match.group(3)

    out_lines = []
    for line in text.splitlines():
        if '"memref_ext.slice_' in line or '"gcu.matmul"' in line:
            if "operandSegmentSizes = array<i32:" in line:
                line = re.sub(
                    r"(operandSegmentSizes = array<i32: )"
                    r"((?:\s*\d+\s*,\s*)*\s*\d+\s*)(>)",
                    _strip_trailing_zero_segments,
                    line,
                )
            if '"gcu.matmul"' in line:
                # Legacy gcu.matmul keeps one empty variadic segment entry.
                line = re.sub(
                    r"operandSegmentSizes = array<i32: (\d+(?:, \d+){3})>",
                    r"operandSegmentSizes = array<i32: \1, 0>",
                    line,
                )
                line = re.sub(r"inputPrecision = \d+ : i32, ", "", line)
                line = re.sub(r", inputPrecision = \d+ : i32", "", line)
        if '"memref_ext.memset_start"' in line:
            line = re.sub(
                r" <\{operandSegmentSizes = array<i32: [0-9, ]+>\}>", "", line
            )
        out_lines.append(line)
    return "\n".join(out_lines) + "\n"


def _install_gcuir_text_compat():
    """Wrap enflame toolkit.gcu_compiler_opt with a scoped downgrade retry."""
    try:
        from triton.backends.enflame import toolkit
    except ImportError:
        return
    if getattr(toolkit.gcu_compiler_opt, "_gdn_fwd_gcu_compat", False):
        return
    original = toolkit.gcu_compiler_opt

    def gcu_compiler_opt_compat(content, *args):
        try:
            return original(content, *args)
        except Exception as first_error:
            text = content if isinstance(content, str) else str(content)
            if not any(name in text for name in _COMPAT_SCOPED_KERNELS):
                raise first_error
            try:
                return toolkit._run_command(
                    "gcu-compiler-opt",
                    _downgrade_gcuir_for_external_tool(text),
                    "-mlir-print-op-generic",
                    *args,
                )
            except Exception:
                raise first_error

    gcu_compiler_opt_compat._gdn_fwd_gcu_compat = True
    toolkit.gcu_compiler_opt = gcu_compiler_opt_compat


# _install_gcuir_text_compat()  # DISABLED 2026-11-09: 私有 GCUIR shim 实测引入回归(降级重试路径
# 搞挂合法 kernel,同 conftest shim 事故);详见 commit b333be6/e1acdf6 与
# compiler_issues/vllm_memref_slice_pad_start.md。编译器修复后统一处置。


# GCU launch limits for the folded flat grid (compiler_issues/grid_hard_limits.md).
_GRID_X_MAX = 65535
_GRID_YZ_MAX = 255

# Cached zero tile for the dot path's load-origin state init (see below).
# [rows, cols] sized for the largest widened tile (BLOCK_V <= 128 rows,
# BLOCK_K <= 256 cols); every program reads the same region (L2-resident).
_ZERO_TILE_ROWS = 128
_ZERO_TILE_COLS = 256
_ZERO_TILE_CACHE = {}


def _zero_state_tile(device):
    """One zero fp32 tile, cached per device.

    torch_gcu's current toolchain fails the in-process make_gcuir pipeline
    when a tl.dot consumes a zeros-initialized state tile (memref.alloc
    symbol error; probe evidence in this module's job tmp). A state tile
    that originates from a live 2-D load compiles, so when initial_state is
    None and the dot path is taken, the kernel loads this zero tile with
    strides (0, 0, cols, 1): every program reads the same zeros, which is
    semantically identical to a zeros init. The load must be a genuine 2-D
    tile access: a stride-0 broadcast row aborts on device (probe W2), so
    the buffer is a real [128, 256] tile. It is zero-filled once per
    process and never written by the kernel.
    """
    key = str(device)
    buf = _ZERO_TILE_CACHE.get(key)
    if buf is None:
        buf = torch.zeros(
            _ZERO_TILE_ROWS,
            _ZERO_TILE_COLS,
            dtype=torch.float32,
            device=device,
        )
        _ZERO_TILE_CACHE[key] = buf
    return buf


def _gdn_grid(total):
    """Fold ``total`` logical programs into a GCU-safe 3-D grid.

    Any decomposition with ``grid_x * grid_y * grid_z >= total`` is correct;
    the kernel recombines the flat program id via ``grid_x``/``grid_xy``
    runtime arguments and guards ``pid < num_programs``. Covers up to
    65535 * 255 * 255 ~= 4.26e9 programs.
    """
    if total <= _GRID_X_MAX:
        return (total, 1, 1)
    grid_y = triton.cdiv(total, _GRID_X_MAX)
    if grid_y <= _GRID_YZ_MAX:
        return (triton.cdiv(total, grid_y), grid_y, 1)
    grid_z = triton.cdiv(total, _GRID_X_MAX * _GRID_YZ_MAX)
    grid_y = triton.cdiv(total, _GRID_X_MAX * grid_z)
    return (triton.cdiv(total, grid_y * grid_z), grid_y, grid_z)


@triton.jit
def _gdn_fwd_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    g_ptr,
    beta_ptr,
    h0_ptr,
    o_ptr,
    ht_ptr,
    T,
    H,
    HV,
    K,
    V,
    scale,
    RATIO,
    sqb,
    sqt,
    sqh,
    sqk,
    skb,
    skt,
    skh,
    skk,
    svb,
    svt,
    svh,
    svv,
    sgb,
    sgt,
    sgh,
    sbb,
    sbt,
    sbh,
    sbv,
    sob,
    sot,
    soh,
    sov,
    ihb,
    ihh,
    ihv,
    ihk,
    thb,
    thh,
    thv,
    thk,
    grid_x,
    grid_xy,
    num_programs,
    BLOCK_K: tl.constexpr,
    BLOCK_V: tl.constexpr,
    NV: tl.constexpr,
    HAS_INIT: tl.constexpr,
    OUT_FINAL: tl.constexpr,
    USE_L2NORM: tl.constexpr,
    BETA_CHANNEL: tl.constexpr,
    MASK_K: tl.constexpr,
    MASK_V: tl.constexpr,
    DOT: tl.constexpr,
    T1_NOINIT: tl.constexpr,
):
    pid = (
        tl.program_id(0)
        + tl.program_id(1) * grid_x
        + tl.program_id(2) * grid_xy
    )
    if pid >= num_programs:
        return
    pid_v = pid % NV
    pid_bh = pid // NV
    hv = pid_bh % HV
    b = pid_bh // HV
    h = hv // RATIO

    offs_k = tl.arange(0, BLOCK_K)
    offs_v = pid_v * BLOCK_V + tl.arange(0, BLOCK_V)

    if HAS_INIT:
        if MASK_K or MASK_V:
            st_mask = (offs_v[:, None] < V) & (offs_k[None, :] < K)
            state = tl.load(
                h0_ptr
                + b * ihb
                + hv * ihh
                + offs_v[:, None] * ihv
                + offs_k[None, :] * ihk,
                mask=st_mask,
                other=0.0,
            )
        else:
            state = tl.load(
                h0_ptr
                + b * ihb
                + hv * ihh
                + offs_v[:, None] * ihv
                + offs_k[None, :] * ihk,
            )
    else:
        state = tl.zeros((BLOCK_V, BLOCK_K), dtype=tl.float32)

    k_base = k_ptr + b * skb + h * skh
    q_base = q_ptr + b * sqb + h * sqh
    v_base = v_ptr + b * svb + hv * svh
    o_base = o_ptr + b * sob + hv * soh
    g_base = g_ptr + b * sgb + hv * sgh
    b_base = beta_ptr + b * sbb + hv * sbh

    if MASK_K:
        kmask = offs_k < K
    if MASK_V:
        vmask = offs_v < V

    # Column group for the fused dual inner product (DOT path): col 0 holds
    # k, col 1 holds q * scale. One tl.dot(state, w) yields <state, k> and
    # <state, qs> in a single MMA, replacing the two [BLOCK_V, BLOCK_K]
    # product-sum trees whose loop-carried product tiles materialize across
    # the token walk at BLOCK_V >= 64. The output then follows the
    # algebraically identical out = <state_gated, qs> + vcorr * <k, qs>
    # instead of reducing the updated state again.
    dcols = tl.arange(0, 16)

    # Prime the one-step-ahead load registers with timestep 0.
    cur = 0
    if MASK_K:
        k_t = tl.load(k_base + offs_k * skk, mask=kmask, other=0.0)
        q_t = tl.load(q_base + offs_k * sqk, mask=kmask, other=0.0)
    else:
        k_t = tl.load(k_base + offs_k * skk)
        q_t = tl.load(q_base + offs_k * sqk)
    if MASK_V:
        v_t = tl.load(v_base + offs_v * svv, mask=vmask, other=0.0)
    else:
        v_t = tl.load(v_base + offs_v * svv)
    gval = tl.load(g_base)
    if BETA_CHANNEL:
        if MASK_V:
            bt = tl.load(b_base + offs_v * sbv, mask=vmask, other=0.0).to(
                tl.float32
            )
        else:
            bt = tl.load(b_base + offs_v * sbv).to(tl.float32)
    else:
        bt = tl.load(b_base).to(tl.float32)

    if T1_NOINIT:
        # T == 1 with no initial state: the state is zero before the single
        # step, so the whole [BLOCK_V, BLOCK_K] state machine collapses
        # algebraically to pure vector work (the tile is never materialized):
        #   pred = 0, vcorr = beta * v, state = outer(vcorr, k)
        #   o    = <state, q * scale> = beta * v * <k, q * scale>
        # (k/q are the L2-normalized vectors when USE_L2NORM is set, matching
        # the reference which normalizes before the recurrence.)
        kf = k_t.to(tl.float32)
        qf = q_t.to(tl.float32)
        if USE_L2NORM:
            # Vector division here: the scalar-reciprocal form measured 3-5%
            # slower on the decode family (the scalar div serializes on this
            # short chain) while helping the T walk, so the two paths keep
            # different forms.
            kf = kf / tl.sqrt(tl.sum(kf * kf, axis=0) + 1e-6)
            qf = qf / tl.sqrt(tl.sum(qf * qf, axis=0) + 1e-6)
        s = tl.sum(kf * (qf * scale), axis=0)
        vf = v_t.to(tl.float32)
        vb = bt * vf
        o_t = vb * s
        if MASK_V:
            tl.store(
                o_base + offs_v * sov,
                o_t.to(o_ptr.dtype.element_ty),
                mask=vmask,
            )
        else:
            tl.store(o_base + offs_v * sov, o_t.to(o_ptr.dtype.element_ty))
        if OUT_FINAL:
            if MASK_K or MASK_V:
                fin_mask = (offs_v[:, None] < V) & (offs_k[None, :] < K)
                tl.store(
                    ht_ptr
                    + b * thb
                    + hv * thh
                    + offs_v[:, None] * thv
                    + offs_k[None, :] * thk,
                    vb[:, None] * kf[None, :],
                    mask=fin_mask,
                )
            else:
                tl.store(
                    ht_ptr
                    + b * thb
                    + hv * thh
                    + offs_v[:, None] * thv
                    + offs_k[None, :] * thk,
                    vb[:, None] * kf[None, :],
                )
        return

    for t in range(0, T):
        # Issue the next timestep's loads before consuming the current ones.
        # `cur` is clamped so the final prefetch stays in bounds.
        cur = tl.minimum(t + 1, T - 1)
        if MASK_K:
            k_n = tl.load(
                k_base + cur * skt + offs_k * skk, mask=kmask, other=0.0
            )
            q_n = tl.load(
                q_base + cur * sqt + offs_k * sqk, mask=kmask, other=0.0
            )
        else:
            k_n = tl.load(k_base + cur * skt + offs_k * skk)
            q_n = tl.load(q_base + cur * sqt + offs_k * sqk)
        if MASK_V:
            v_n = tl.load(
                v_base + cur * svt + offs_v * svv, mask=vmask, other=0.0
            )
        else:
            v_n = tl.load(v_base + cur * svt + offs_v * svv)
        g_n = tl.load(g_base + cur * sgt)
        if BETA_CHANNEL:
            if MASK_V:
                bt_n = tl.load(
                    b_base + cur * sbt + offs_v * sbv, mask=vmask, other=0.0
                ).to(tl.float32)
            else:
                bt_n = tl.load(b_base + cur * sbt + offs_v * sbv).to(
                    tl.float32
                )
        else:
            bt_n = tl.load(b_base + cur * sbt).to(tl.float32)

        kf = k_t.to(tl.float32)
        qf = q_t.to(tl.float32)
        if USE_L2NORM:
            # Scalar reciprocal + vector multiply: measured -7% on the serial
            # T=2048 dot walk (the division sits on the loop-carried chain).
            kf = kf * (1.0 / tl.sqrt(tl.sum(kf * kf, axis=0) + 1e-6))
            qf = qf * (1.0 / tl.sqrt(tl.sum(qf * qf, axis=0) + 1e-6))
        qs = qf * scale

        d = tl.exp(gval)

        if DOT:
            # The decay scales the dot OUTPUT instead of the state operand:
            # (S * d) @ w == d * (S @ w), so the [BLOCK_V, BLOCK_K] scale
            # pass leaves the serial chain and only the [BLOCK_V, 16] dot
            # result and the fused state update below touch the full tile.
            w = tl.where(
                dcols[None, :] == 1,
                qs[:, None],
                tl.where(dcols[None, :] == 0, kf[:, None], 0.0),
            )
            dqa = tl.dot(state, w)
            pred = d * tl.sum(tl.where(dcols[None, :] == 0, dqa, 0.0), axis=1)
        else:
            pred = d * tl.sum(state * kf[None, :], axis=1)

        vf = v_t.to(tl.float32)
        vcorr = (vf - pred) * bt

        if DOT:
            state = state * d + vcorr[:, None] * kf[None, :]
            o_t = d * tl.sum(tl.where(dcols[None, :] == 1, dqa, 0.0), axis=1)
            o_t += vcorr * tl.sum(kf * qs, axis=0)
        else:
            state = state * d + vcorr[:, None] * kf[None, :]
            o_t = tl.sum(state * qs[None, :], axis=1)

        if MASK_V:
            tl.store(
                o_base + t * sot + offs_v * sov,
                o_t.to(o_ptr.dtype.element_ty),
                mask=vmask,
            )
        else:
            tl.store(
                o_base + t * sot + offs_v * sov, o_t.to(o_ptr.dtype.element_ty)
            )

        k_t = k_n
        q_t = q_n
        v_t = v_n
        gval = g_n
        bt = bt_n

    if OUT_FINAL:
        if MASK_K or MASK_V:
            fin_mask = (offs_v[:, None] < V) & (offs_k[None, :] < K)
            tl.store(
                ht_ptr
                + b * thb
                + hv * thh
                + offs_v[:, None] * thv
                + offs_k[None, :] * thk,
                state,
                mask=fin_mask,
            )
        else:
            tl.store(
                ht_ptr
                + b * thb
                + hv * thh
                + offs_v[:, None] * thv
                + offs_k[None, :] * thk,
                state,
            )


def fused_recurrent_gated_delta_rule(
    q,
    k,
    v,
    g,
    beta,
    scale,
    initial_state,
    output_final_state,
    use_qk_l2norm_in_kernel=False,
):
    B, T, H, K = q.shape
    HV = v.shape[2]
    V = v.shape[-1]
    ratio = HV // H
    beta_channel = beta.dim() == v.dim()

    BLOCK_K = max(16, triton.next_power_of_2(K))
    # The register-resident fp32 state tile [BLOCK_V, BLOCK_K] drives the whole
    # cost model. An interleaved A/B over all ten timing shapes (five repeats
    # each, min-vs-median spread under 1% for the retained configs) showed a
    # narrow tile is better everywhere than the wide one this kernel shipped
    # with: 32x128 at 4 warps beat 64x128 at 8 warps by 4-23% on the K=V=128
    # shapes, and 16x64 at 2 warps beat 64x64 at 4 warps by 27% on bench0 and 6%
    # on bench1. Two warps carry a 2048-element tile (32 fp32 per thread) and
    # four warps carry 4096 without spilling, so the tile is sized to that budget
    # and the extra V-blocks add independent programs instead of only widening
    # each program.
    BLOCK_V = max(
        16, min(16 if BLOCK_K <= 64 else 32, triton.next_power_of_2(V))
    )
    num_warps = 2 if BLOCK_V * BLOCK_K <= 2048 else 4

    o = torch.empty((B, T, HV, V), dtype=v.dtype, device=v.device)
    if output_final_state:
        final_state = torch.empty(
            (B, HV, V, K), dtype=torch.float32, device=v.device
        )
        thb, thh, thv, thk = final_state.stride()
        ht = final_state
    else:
        final_state = None
        thb = thh = thv = thk = 0
        ht = o

    if initial_state is not None:
        h0 = initial_state
        ihb, ihh, ihv, ihk = h0.stride()
        has_init = True
    else:
        h0 = q
        ihb = ihh = ihv = ihk = 0
        has_init = False

    sqb, sqt, sqh, sqk = q.stride()
    skb, skt, skh, skk = k.stride()
    svb, svt, svh, svv = v.stride()
    sgb, sgt, sgh = g.stride()
    sbb, sbt, sbh = beta.stride()[0], beta.stride()[1], beta.stride()[2]
    sbv = beta.stride()[3] if beta_channel else 0
    sob, sot, soh, sov = o.stride()

    # Wave-aware BV dispatch (host metadata only, no device reads): programs
    # = B * HV * cdiv(V, BLOCK_V) run a serial per-sequence recurrence, so
    # wall clock ~ ceil(programs / 48) waves x per-program op chain. Widening
    # BLOCK_V folds more V rows into one program: per-program cost grows
    # sub-linearly while the program count drops proportionally. Measured
    # here on the bench family: decode B=8 tree-wide 312->47us (-85%),
    # decode B=1 dot-wide 45->9.6us, prefill T=2048 dot-wide 66.8->11.6ms.
    # Guards:
    #   * multi-wave at the narrow tile (B * HV * nv > 48): single-wave shapes
    #     already fit the machine, nothing to gain from widening;
    #   * innermost strides must be 1: strided (gather) q/k/v/state access is
    #     priced per element, so widening buys nothing;
    #   * the state tile stays <= 16384 fp32 elements (64 KB, the measured
    #     DSM-healthy domain; 48-way residency verified by perf).
    # Form choice inside the widened branch (measured, see below): T == 1
    # with no initial state takes the collapsed vector arm (see the kernel's
    # T1_NOINIT branch); T == 1 with an initial state keeps the product-sum
    # trees (faster per program on a single step); T > 1 switches to the dot
    # fusion (kills the loop-carried product-tile materialization that
    # dominates the serial walk). Without the dot (BLOCK_K outside
    # [128, 256]) multi-token shapes keep the narrow tile.
    # 2026-11-12 additions on top of the widened/dot forms (bench core,
    # kernel-mode medians, two fresh runs each): the T1_NOINIT arm took
    # decode B=8 46.4->21.9us / B=64 340->145.7us / B=256 1.35ms->573.5us /
    # B=1 9.1->~6us; folding the decay out of the pre-dot state scaling plus
    # the reciprocal-form l2norm took the T=2048 dot walk 11.76->10.33ms
    # (-12.2%). Hyperparameter rescan after the arm landed confirmed
    # BLOCK_V=128 / num_warps=1 / num_stages=1 (nw=2: +23-26%; BLOCK_V=64:
    # ~2x worse -- the k/q load + l2norm chain is per-program fixed cost).
    nv = triton.cdiv(V, BLOCK_V)
    use_dot = False
    num_stages = 3
    if (
        V > 16
        and sqk == 1
        and skk == 1
        and svv == 1
        and sov == 1
        and (not has_init or ihk == 1)
        and (not output_final_state or thk == 1)
        and B * HV * nv > 48
    ):
        # The two [BLOCK_V, BLOCK_K] product-sum trees fold into one tl.dot
        # against a [BLOCK_K, 16] operand (col 0 = k, col 1 = q * scale) with
        # the output following out = <state_gated, qs> + vcorr * <k, qs>;
        # below BLOCK_K = 128 the MMA's fixed overhead exceeds the tree cost.
        # Measured on this kernel (bench core, kernel-mode medians): on
        # single-step decode the tree is the faster per-program form
        # (B=8: 46.7 vs 50.2us, B=64: 340 vs 356us), while on the serial
        # T>1 walk the dot removes the loop-carried product-tile
        # materialization and wins big (B=1, T=2048: 66.8 vs 11.6ms), so the
        # dot carries only the multi-token shapes. Without the dot (BLOCK_K
        # out of [128, 256]) multi-token shapes keep the narrow tile: the
        # wide tree regresses on T > 1 (loop-carried product materialization).
        # Shapes that pass the wave guard but take neither inner branch
        # (T > 1 with BLOCK_K < 128) keep the narrow tile and the default
        # num_stages/num_warps set above.
        dot_ok = 128 <= BLOCK_K <= _ZERO_TILE_COLS
        if T == 1 or dot_ok:
            BLOCK_V = min(
                triton.next_power_of_2(V), 128, max(16, 16384 // BLOCK_K)
            )
            nv = triton.cdiv(V, BLOCK_V)
            use_dot = dot_ok and T > 1
            # A zeros-initialized state tile breaks the in-process make_gcuir
            # pipeline on this toolchain, so the None-state case swaps in the
            # cached zero tile as a live 2-D load (load-origin layout, values
            # identical to zeros; see _zero_state_tile).
            if use_dot and not has_init:
                h0 = _zero_state_tile(v.device)
                ihb = ihh = 0
                ihv = _ZERO_TILE_COLS
                ihk = 1
                has_init = True
            num_warps = 1
            num_stages = 1

    total = B * HV * nv
    grid = _gdn_grid(total)
    grid_xy = grid[0] * grid[1]

    _gdn_fwd_kernel[grid](
        q,
        k,
        v,
        g,
        beta,
        h0,
        o,
        ht,
        T,
        H,
        HV,
        K,
        V,
        float(scale),
        ratio,
        sqb,
        sqt,
        sqh,
        sqk,
        skb,
        skt,
        skh,
        skk,
        svb,
        svt,
        svh,
        svv,
        sgb,
        sgt,
        sgh,
        sbb,
        sbt,
        sbh,
        sbv,
        sob,
        sot,
        soh,
        sov,
        ihb,
        ihh,
        ihv,
        ihk,
        thb,
        thh,
        thv,
        thk,
        grid[0],
        grid_xy,
        total,
        BLOCK_K=BLOCK_K,
        BLOCK_V=BLOCK_V,
        NV=nv,
        HAS_INIT=has_init,
        OUT_FINAL=bool(output_final_state),
        USE_L2NORM=bool(use_qk_l2norm_in_kernel),
        BETA_CHANNEL=beta_channel,
        MASK_K=(K % BLOCK_K != 0),
        MASK_V=(V % BLOCK_V != 0),
        DOT=use_dot,
        T1_NOINIT=(T == 1 and not has_init),
        num_warps=num_warps,
        num_stages=num_stages,
        enable_fp_fusion=not use_dot,
    )

    return o, final_state


__all__ = ["fused_recurrent_gated_delta_rule"]
