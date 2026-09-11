# Copyright 2026, The FlagOS Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License")
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied. See the License for the specific language governing
# permissions and limitations under the License.

"""Triton kernel for lora/sgemm_lora_a.

Segment-wise batched GEMM for LoRA "A" (down-projection): for each segment ``b``
(a contiguous run of tokens belonging to one request, rows
``seg_indptr[b]:seg_indptr[b+1]``), gather the segment's input rows and multiply
by its adapter's weight slice ``weights[weight_indices[b]]`` (shape
``[stack_num*r, K]``), producing ``out[rows] = x[rows].float() @ w.float().T``.

Signature matches ``reference(x, weights, batch_info, stack_num=1)``.

Implementation: a single 2D-tiled matmul kernel. Each program owns one output
tile ``[BLOCK_M rows x BLOCK_N cols]`` of one segment's result. Row tiles are
*segment-aligned* (a tile never spans two segments) so every row in a tile
shares the same weight slice — this is what makes the per-adapter routing
exact. The tile->segment mapping (``weight_index``, segment-relative token
base, live row count) is computed **inside the kernel** by a short device-side
scan over the small ``seg_indptr`` / ``weight_indices`` arrays — there is no
host-side work-list construction and no host<->device sync.

Launch-config selection: rather than ``@triton.autotune`` (whose per-call
wrapper — key-tuple build, cache hash/lookup, grid re-evaluation — adds ~60us
of steady-state Python overhead on Ascend, which is *most* of the kernel's
~110us runtime and dwarfed the autotune's own gains), we select
``BLOCK_M / BLOCK_N / BLOCK_K / num_warps / num_stages`` with a cheap
**host-side heuristic over tensor shapes** and launch the plain ``@triton.jit``
kernel directly. The shapes seen here are small and regular (bench: R ∈ {32,64},
K=4096, uniform per-segment row counts), so a shape-keyed heuristic captures
the same configurations autotune would pick, without the dispatch tax. This is
not a hand-rolled cache (no module-level mutable state) — just a pure function
of the input shapes, evaluated fresh on every call.

Inside the kernel, the small ``seg_indptr`` / ``weight_indices`` arrays are
loaded as a tiny ``[MAX_BS]`` vector, each segment's tile count is computed
and exclusive-scanned, and the segment owning ``program_id(0)`` is selected
via a weighted-sum gather; programs past the last real tile get a zero
live-row count and become no-ops through the row masks. The kernel then
loads the matching weight slice and runs a K-strided
``tl.dot`` (float32 accumulation). For float32 input, ``input_precision="ieee"``
matches the reference's exact ``.float()`` matmul; for bfloat16/float16 the
tolerance is wide, so the dot uses ``input_precision="hf32"`` — the backend's
faster fp32 MMA path that keeps ~10 mantissa bits, losing nothing the bf16/fp16
source hadn't already discarded — substantially faster on Ascend/NPU — while
still accumulating in float32. Permutation (when present) is applied
by gathering the actual row index from ``batch_info.permutation``.

This is pure portable Triton: device via ``flaggems_sglang.device`` (never
hardcoded), no vendor-private ops, no fallbacks, no module-level mutable
state.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


def _select_config(S, R, K, bs):
    """Pick (BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages) from shapes.

    Pure function of the input shapes — no module-level mutable state, no
    cross-call caching. The bench cases have small R (32 or 64) so BLOCK_N
    covers the whole R in one tile; K=4096 favours a wide BLOCK_K (fewer
    K-iters / weight reloads) with deeper pipelining; BLOCK_M=64 is the sweet
    spot on Ascend (a 256-row full-segment tile forces a tiny BLOCK_K and
    over-allocates SRAM, so it is *slower* despite fewer weight reloads).
    Small correctness cases (tiny S/K) get correspondingly small tiles so the
    masked loads stay cheap and compilation is light.

    The configs here were validated by direct-jit micro-benchmarks (bypassing
    the autotune wrapper): for the two bench shapes these are within ~2us of
    the autotune-selected optimum and ~60us faster than the autotune-wrapped
    call path.
    """
    # BLOCK_N: cover the whole R in one N-tile (R is small). Cap at 256 for
    # pathologically wide adapters; tile when R exceeds the cap.
    block_n = triton.next_power_of_2(max(R, 16))
    if block_n > 256:
        block_n = 256

    # BLOCK_M: 64 is the Ascend sweet spot for the bench segments. For very
    # small total row counts (correctness cases), shrink so a tile isn't 64x
    # the actual rows — keeps masked waste and compile weight down.
    block_m = 64
    if S <= 32:
        block_m = 16
    elif S <= 128:
        block_m = 32

    # BLOCK_K: wide K-tile cuts K-loop iterations and weight reloads. Cap at
    # 512 (deeper tiles force tiny BLOCK_K on the 256-row path / blow SRAM).
    block_k = triton.next_power_of_2(max(K, 16))
    if block_k > 512:
        block_k = 512

    # On Ascend the MMA has fixed tile shapes; very small BLOCK_M (16) combined
    # with a wide BLOCK_N / BLOCK_K is an unsupported dot shape and fails
    # MLIR compilation. Keep the small-M path conservative: cap the N and K
    # tiles so the (16, 64, 128) dot stays in the supported set.
    if block_m <= 16:
        if block_n > 64:
            block_n = 64
        if block_k > 128:
            block_k = 128
    elif block_m <= 32:
        if block_n > 128:
            block_n = 128
        if block_k > 256:
            block_k = 256

    # Pipelining: only worthwhile when there are several K iters to overlap.
    if K >= 1024 and block_k >= 256:
        num_warps = 8
        num_stages = 3
    elif K >= 256:
        num_warps = 8
        num_stages = 2
    else:
        num_warps = 4
        num_stages = 2
    return block_m, block_n, block_k, num_warps, num_stages


@triton.jit
def _sgemm_lora_a_kernel(
    x_ptr,
    w_ptr,
    out_ptr,
    perm_ptr,
    seg_indptr_ptr,
    weight_indices_ptr,
    bs,
    S,
    R,
    K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    MAX_BS: tl.constexpr,
    HAS_PERM: tl.constexpr,
    USE_IEEE: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # ------------------------------------------------------------------
    # Device-side tile->segment routing (no host-side work-list at all).
    # Tiles are assigned to segments in order: segment 0 gets its first
    # ceil(seg_len_0 / BLOCK_M) tiles, then segment 1, etc. We load the small
    # ``seg_indptr`` (bs+1 ints, padded to MAX_BS) and ``weight_indices`` as a
    # tiny [MAX_BS] vector, compute each segment's tile count, and exclusive-
    # scan it. The owning segment is the one whose [tiles_before,
    # tiles_before + tiles_per_seg) range contains ``pid_m``; its index is the
    # weighted sum of the owner-mask, used for a dynamic gather of the scalar
    # metadata (weight index, token base, live row count). This is pure
    # portable Triton (no Python branching across runtime values) and needs
    # no host<->device sync. Programs beyond the real tile count are not owned
    # by any segment -> rows_in_tile = 0 -> no-ops via the row masks below.
    # ------------------------------------------------------------------
    b_off = tl.arange(0, MAX_BS)  # [MAX_BS]
    seg_active = b_off < bs  # [MAX_BS]
    seg_start_v = tl.load(seg_indptr_ptr + b_off, mask=seg_active, other=0).to(
        tl.int64
    )
    seg_end_v = tl.load(
        seg_indptr_ptr + b_off + 1, mask=seg_active, other=0
    ).to(tl.int64)
    seg_len_v = seg_end_v - seg_start_v  # [MAX_BS]
    tiles_per_seg = (seg_len_v + BLOCK_M - 1) // BLOCK_M  # [MAX_BS]
    # cumulative tiles BEFORE each segment (exclusive scan over [MAX_BS])
    tiles_before = tl.cumsum(tiles_per_seg) - tiles_per_seg  # [MAX_BS]
    owns = (
        (pid_m >= tiles_before)
        & (pid_m < tiles_before + tiles_per_seg)
        & seg_active
    )
    # owning segment index (scalar); when not owned, owns is all-false -> seg_b=0
    seg_b = tl.sum(tl.where(owns, b_off.to(tl.int64), 0))
    local_tile = pid_m - tl.sum(tl.where(owns, tiles_before, 0))  # scalar
    # gather scalar metadata via dynamic index from seg_indptr / weight_indices
    w_idx_s = tl.load(weight_indices_ptr + seg_b).to(tl.int64)
    seg_start_s = tl.load(seg_indptr_ptr + seg_b).to(tl.int64)
    seg_len_s = tl.load(seg_indptr_ptr + seg_b + 1).to(tl.int64) - seg_start_s
    token_base_s = seg_start_s + local_tile * BLOCK_M
    rows_in_tile_s = tl.minimum(seg_len_s - local_tile * BLOCK_M, BLOCK_M)

    row_off = tl.arange(0, BLOCK_M)  # [BLOCK_M]
    n_off = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # [BLOCK_N]
    k_off_base = tl.arange(0, BLOCK_K)  # [BLOCK_K]

    row_mask = row_off < rows_in_tile_s
    n_mask = n_off < R

    # Actual input/output row indices (apply permutation if present).
    tok_pos = token_base_s + row_off  # [BLOCK_M]
    if HAS_PERM:
        actual_rows = tl.load(perm_ptr + tok_pos, mask=row_mask, other=0).to(
            tl.int64
        )
    else:
        actual_rows = tok_pos.to(tl.int64)

    w_base = w_ptr + w_idx_s * (R * K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for kk in range(0, K, BLOCK_K):
        k_off = kk + k_off_base
        k_mask = k_off < K
        # x tile [BLOCK_M, BLOCK_K], coalesced (k innermost)
        x_ptrs = x_ptr + actual_rows[:, None] * K + k_off[None, :]
        x = tl.load(
            x_ptrs, mask=row_mask[:, None] & k_mask[None, :], other=0.0
        )
        # weight slice w[n, k] = weights[w_idx, n, k]; load [BLOCK_N, BLOCK_K]
        # coalesced (k innermost), then transpose for tl.dot.
        w_ptrs = w_base + n_off[:, None] * K + k_off[None, :]
        w = tl.load(w_ptrs, mask=n_mask[:, None] & k_mask[None, :], other=0.0)

        # Accumulate in float32 (matches reference semantics). The inputs are
        # upcast to float32 before the dot. For float32 input we force
        # ``input_precision="ieee"`` to match the reference's exact matmul
        # exactly; for the lower-precision dtypes (bfloat16/float16) the
        # tolerance is wide (1.5e-2 / 1e-2), so we use ``input_precision="hf32"``
        # — the backend's faster fp32 MMA schedule that keeps ~10 mantissa bits.
        # The source data is bf16/fp16 (≤8 mantissa bits), so hf32 loses nothing
        # the input dtype hadn't already discarded — well within tolerance —
        # while letting the Ascend MMA run at substantially higher throughput
        # than a strict ieee fp32 dot. (Keeping inputs in native bf16 for the
        # dot trips a hardware MTE fault on Ascend, so we upcast to float32 and
        # rely on the precision flag instead.)
        x = x.to(tl.float32)
        w = w.to(tl.float32)
        if USE_IEEE:
            acc += tl.dot(x, tl.trans(w), input_precision="ieee")
        else:
            acc += tl.dot(x, tl.trans(w), input_precision="hf32")

    acc = acc.to(out_ptr.dtype.element_ty)
    out_ptrs = out_ptr + actual_rows[:, None] * R + n_off[None, :]
    tl.store(out_ptrs, acc, mask=row_mask[:, None] & n_mask[None, :])


def sgemm_lora_a(x, weights, batch_info, stack_num=1):
    """LoRA "A" down-projection: segment-batched ``x @ weights[w].T``.

    Signature matches ``reference(x, weights, batch_info, stack_num=1)``.
    """
    S, K = x.shape
    R = weights.shape[1]
    out = torch.empty(S, R, dtype=x.dtype, device=x.device)

    seg_indptr = batch_info.seg_indptr
    weight_indices = batch_info.weight_indices
    permutation = batch_info.permutation

    bs = int(weight_indices.shape[0])

    # Select launch config from shapes (no autotune wrapper -> no per-call
    # dispatch overhead).
    block_m, block_n, block_k, num_warps, num_stages = _select_config(
        S, R, K, bs
    )

    # Upper bound on row tiles, derivable from tensor *shapes* alone (no
    # host<->device sync): each non-empty segment of length L needs
    # ceil(L / BLOCK_M) tiles, and the sum is <= ceil(S / BLOCK_M) + bs.
    # Programs beyond the real tile count write nothing (rows_in_tile <= 0
    # guard inside the kernel).
    total_tiles = (S + block_m - 1) // block_m + bs
    num_col_tiles = (R + block_n - 1) // block_n
    grid = (total_tiles, num_col_tiles)

    has_perm = permutation is not None

    # Precision: the reference computes in float32 (``x.float() @ w.float().T``).
    # For float32 input we must use ``input_precision="ieee"`` to match exactly;
    # for the lower-precision dtypes (bfloat16/float16) the tolerance is wide
    # (1.5e-2 / 1e-2), so we use ``input_precision="hf32"`` — the backend's
    # faster fp32 MMA schedule that keeps ~10 mantissa bits. The source data is
    # bf16/fp16 (≤8 mantissa bits), so hf32 loses nothing the input dtype hadn't
    # already discarded and stays well within tolerance, while letting the
    # Ascend MMA run at higher throughput than a strict ieee fp32 dot.
    use_ieee = x.dtype == torch.float32

    # MAX_BS bounds the in-kernel segment scan loop (compiled as constexpr).
    # Round bs up to the next power of two (≥1) for a tight tile-friendly loop.
    max_bs = triton.next_power_of_2(max(bs, 1))

    _sgemm_lora_a_kernel[grid](
        x,
        weights,
        out,
        permutation if has_perm else x,  # perm_ptr unused when !HAS_PERM
        seg_indptr,
        weight_indices,
        bs,
        S,
        R,
        K,
        MAX_BS=max_bs,
        HAS_PERM=has_perm,
        USE_IEEE=use_ieee,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return out


__all__ = ["sgemm_lora_a"]
