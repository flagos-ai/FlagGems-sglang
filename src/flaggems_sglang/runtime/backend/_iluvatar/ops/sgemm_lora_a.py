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

"""Triton implementation of lora/sgemm_lora_a.

LoRA "A" (down-projection) segmented batched GEMM. For each segment ``b``
(a contiguous run of rows belonging to one request) the kernel computes

    out[rows, :] = (x[rows].to(float) @ weights[w_idx].to(float).T).to(x.dtype)

where ``rows = permutation[start:end]`` if a permutation exists, else
``rows = arange(start, end)``; ``start, end = seg_indptr[b], seg_indptr[b+1]``
and ``w_idx = weight_indices[b]``. ``weights`` has shape
``[num_lora, stack_num*r, K]`` and the output is ``[S, stack_num*r]``.

This is a *segmented* / *grouped* GEMM: every segment multiplies a different
weight slice, so the natural unit of parallelism is the segment + output-tile.
Each program tile owns one ``[BLOCK_M, BLOCK_N]`` output block of a single
segment: it loads the segment's contiguous ``x`` rows (gathered through the
permutation when present) and the segment's ``[BLOCK_N, K]`` weight slice,
accumulates the contraction over K in fp32, and writes back to ``out``. This
keeps the per-segment weight slice resident in registers across the full K
contraction and coalesces both the x reads and the out writes.

The grid is ``grid = (cdiv(max_seg, BLOCK_M), cdiv(R, BLOCK_N), bs)``: the
first axis tiles a single segment's M dimension, the second tiles N = R, and
the third indexes the segment. ``BLOCK_M`` is held at 32 -- short, narrow-R
segments (e.g. 8x64-row segments with R=32) fan out into 2 M-tiles x 8
segments = 16 programs, exactly filling the 16 SMs of the target device. K is
folded into the inner loop with a tunable ``BLOCK_K`` and ``num_stages`` for
software pipelining of the bandwidth-bound contraction. Per-segment launch
metadata (start/end row, adapter index) is read from ``seg_indptr`` /
``weight_indices`` / the optional ``permutation`` inside the kernel, so a
single launch handles all segments with no host-side Python loop and no
per-segment kernel launch.

Two kernels are dispatched by ``R`` so the autotune config sets can be tailored
to the N = R tile width without risking a coverage bug (an N-tile wider than its
config's BLOCK_N would silently drop output rows):

  * ``_kernel_n32`` -- BLOCK_N=32 configs, for R <= 32. R=32 is covered by a
    single N-tile, and the freed N dimension lets BLOCK_K grow to 1024 on the
    bf16/fp16 latency-bound shapes (K-loop trip count 4096->4), which on this
    16-SM, launch-latency-bound device is the winning move for the 8x64 R=32
    benchmark (124us -> ~96us).
  * ``_kernel_n64`` -- BLOCK_N=64 configs, for R > 32 (covers R=64 in one
    N-tile, R=96 in two). Smaller BLOCK_K (128/256) wins here: the shape has
    32 programs over 2 waves, and a large BLOCK_K both spills shared memory on
    fp32 and starves occupancy. BLOCK_K=128 + 8 warps is the fp32 sweet spot
    (seg256x4 fp32 399us -> ~263us) and matches the bf16 best.

Precision matches the reference: accumulation is fp32. For fp32 inputs
``tl.dot`` uses ``input_precision="ieee"`` to reproduce the reference's exact
float32 matmul; bf16/fp16 inputs accumulate in fp32 (the tensor-core default)
then cast back to the input dtype, matching the reference's
``.float() @ .float().T`` then ``.to(x.dtype)``.
"""

import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# Shared kernel body. The two @triton.jit entry points below duplicate this
# body verbatim because @triton.jit cannot be parameterised by an autotune
# config set; keeping a single textual body is what lets each kernel carry its
# own autotune configs (BLOCK_N=32 vs BLOCK_N=64) while staying in sync.
#
# Configs that overflow shared memory on a given shape are simply skipped by
# the Triton autotuner (it prints "Autotuning failed with out of resource" and
# moves on), so the aggressive BLOCK_K=1024 / BLOCK_K=512 configs are safe to
# list here: they win on bf16/fp16 latency-bound shapes and are auto-skipped on
# fp32 / larger-R shapes that cannot fit them.
# ---------------------------------------------------------------------------


# --- kernel_n32: BLOCK_N=32, for R <= 32 (seg64x8 R=32, correctness R=16) ---
@triton.autotune(
    configs=[
        # K=4096 latency-bound winners (bf16/fp16). BLOCK_K=1024 halves the
        # K-loop to 4 iterations; 8 warps + 2 stages for deep pipelining.
        triton.Config(
            {"BLOCK_M": 32, "BLOCK_N": 32, "BLOCK_K": 1024},
            num_warps=8,
            num_stages=2,
        ),
        triton.Config(
            {"BLOCK_M": 32, "BLOCK_N": 32, "BLOCK_K": 1024},
            num_warps=8,
            num_stages=3,
        ),
        triton.Config(
            {"BLOCK_M": 32, "BLOCK_N": 32, "BLOCK_K": 512},
            num_warps=8,
            num_stages=2,
        ),
        triton.Config(
            {"BLOCK_M": 32, "BLOCK_N": 32, "BLOCK_K": 512},
            num_warps=8,
            num_stages=3,
        ),
        # fp32 / larger-K portable fallback (fits shared memory on all dtypes).
        triton.Config(
            {"BLOCK_M": 32, "BLOCK_N": 32, "BLOCK_K": 256},
            num_warps=8,
            num_stages=3,
        ),
        triton.Config(
            {"BLOCK_M": 32, "BLOCK_N": 32, "BLOCK_K": 256},
            num_warps=8,
            num_stages=2,
        ),
        triton.Config(
            {"BLOCK_M": 32, "BLOCK_N": 32, "BLOCK_K": 128},
            num_warps=4,
            num_stages=3,
        ),
    ],
    key=["max_seg", "R", "K", "IS_FP32"],
)
@triton.jit
def _kernel_n32(
    x_ptr,  # * [S, K]
    w_ptr,  # * [num_lora, R, K]
    out_ptr,  # * [S, R]
    perm_ptr,  # * [S] or None
    seg_indptr,  # * [bs+1]
    weight_indices,  # * [bs]
    S,
    K,
    R,
    num_lora,
    xs_stride_m,
    xs_stride_k,  # x row/col stride
    ws_stride_l,
    ws_stride_n,
    ws_stride_k,  # weight lora/n/k stride
    os_stride_m,
    os_stride_n,  # out row/col stride
    max_seg,  # autotune key: max segment length across segments
    HAS_PERM: tl.constexpr,
    IS_FP32: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_b = tl.program_id(2)

    start = tl.load(seg_indptr + pid_b)
    end = tl.load(seg_indptr + pid_b + 1)
    seg_n = end - start  # number of rows in this segment

    w_idx = tl.load(weight_indices + pid_b)

    n_m_tiles = tl.cdiv(seg_n, BLOCK_M)
    n_n_tiles = tl.cdiv(R, BLOCK_N)
    # Out-of-range tiles (segment shorter than max_seg, or N-tiles beyond R)
    # no-op rather than running a full K contraction just to mask out an empty
    # block.
    if pid_m >= n_m_tiles or pid_n >= n_n_tiles:
        return

    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
    x_lm = tl.arange(0, BLOCK_M)  # local segment row index
    offs_m = pid_m * BLOCK_M + x_lm

    # Absolute row index into x / out for this tile. With a permutation, rows
    # are gathered through perm[start + offs_m]; otherwise rows are contiguous
    # (start + offs_m). Computed once here so it is in scope for the store.
    if HAS_PERM:
        out_row_idx = tl.load(
            perm_ptr + start + offs_m, mask=(offs_m < seg_n), other=0
        )
    else:
        out_row_idx = start + offs_m

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Base pointer to this adapter's weight slice: weights[w_idx].
    w_base = w_ptr + w_idx * ws_stride_l

    for kk in range(0, K, BLOCK_K):
        k_off = kk + rk

        # x rows gathered via out_row_idx (absolute rows).
        x_ptrs = (
            x_ptr
            + out_row_idx[:, None] * xs_stride_m
            + k_off[None, :] * xs_stride_k
        )
        x_tile = tl.load(
            x_ptrs,
            mask=(offs_m[:, None] < seg_n) & (k_off[None, :] < K),
            other=0.0,
        )

        # weight slice: weights[w_idx, :, :] -> [R, K]. We need [BLOCK_N, BLOCK_K]
        # i.e. w_tile[n, k] = weights[w_idx, rn[n], k_off[k]].
        w_ptrs = (
            w_base + rn[:, None] * ws_stride_n + k_off[None, :] * ws_stride_k
        )
        w_tile = tl.load(
            w_ptrs, mask=(rn[:, None] < R) & (k_off[None, :] < K), other=0.0
        )

        if IS_FP32:
            acc += tl.dot(
                x_tile, w_tile.T, out_dtype=tl.float32, input_precision="ieee"
            )
        else:
            acc += tl.dot(x_tile, w_tile.T, out_dtype=tl.float32)

    # Store to out[rows, :].
    out_ptrs = (
        out_ptr
        + out_row_idx[:, None] * os_stride_m
        + rn[None, :] * os_stride_n
    )
    out_mask = (offs_m[:, None] < seg_n) & (rn[None, :] < R)

    if IS_FP32:
        tl.store(out_ptrs, acc, mask=out_mask)
    else:
        tl.store(out_ptrs, acc.to(out_ptr.dtype.element_ty), mask=out_mask)


# --- kernel_n32_splitk: split-K over K for the latency-bound n32 case. ---
#
# seg64x8_nlora4_r32_k4096 has only 8 segments x 2 M-tiles x 1 N-tile = 16
# program tiles -- a single wave on the 16-SM device, with exactly one program
# resident per SM and no wave-level memory-level parallelism. The K loop
# (K=4096) therefore runs fully serial inside each program: the next K-iteration's
# global loads cannot overlap with the previous iteration's compute across
# *concurrent* tiles, only across *stages* of one tile, so memory latency is
# only hidden by software pipelining within a single program.
#
# Split-K breaks the single wave into several waves of *smaller* programs by
# partitioning the K contraction across a new grid axis. With ``n_splits``
# slices, the grid grows from 16 to ``16 * n_splits`` programs; each program
# does 1/``n_splits`` of the K contraction and contributes its partial sum to a
# shared fp32 accumulator via ``tl.atomic_add``. Each program's K slice is
# ``cdiv(K, n_splits)`` (ceiling, so non-divisor splits are safe: every K row
# is covered exactly once -- see ``_kernel_n32_splitk``'s in_slice masking),
# so its working set -- and thus its shared-memory footprint -- shrinks,
# raising occupancy.
#
# The empirical sweet spot for the bench shape is ``n_splits=3`` (48 progs /
# 3 waves), not the over-split 8-way (64 progs / 4 waves) the previous
# heuristic picked. 3-way leaves each program a 1366-wide K slice (~5
# BLOCK_K=256 iterations) -- enough pipelined work that 8-warps + 4-stage
# pipelining can hide the K-loop latency, while the 3-wave concurrency buys
# the wave-level memory-level parallelism the single 16-prog wave lacks.
# Over-splitting to 8-way shrinks each slice to ~512 (2 iterations), leaving
# the pipeline nothing to hide, and atomic_add contention + cast cost
# dominate (~80us vs ~58us). ``_choose_n_splits`` targets the 3-wave count.
#
# This is only worthwhile when the launch is launch/latency bound -- few program
# tiles and large K. ``_choose_n_splits`` gates it accordingly: it is only
# applied when the base program count is small (<= ~2 waves) and K is large, so
# the small correctness cases (short K, few rows) and the seg256x4 shape (32
# programs / 2 waves already, handled by _kernel_n64) keep the existing
# non-atomic path. The split-K kernel reuses the n32 autotune config set with
# BLOCK_K sized for the (smaller) per-program K slice.
#
# Because the partial sums are accumulated in fp32 and atomically added into an
# fp32 buffer, the kernel is precision-equivalent to the non-split path
# (fp32 accumulation throughout) for every input dtype. A tiny cast kernel then
# materialises the fp32 buffer into the output dtype. For fp32 inputs the cast
# is a no-op view.
@triton.autotune(
    configs=[
        # seg64x8_nlora4_r32_k4096 winner (bf16): 3-way K-split -> 48 progs / 3
        # waves on the 16-SM device; per-program K slice = cdiv(4096,3)=1366.
        # BLOCK_K=256 + 8 warps + 4-stage pipelining hides the ~5-iteration K
        # loop's memory latency: 75us -> ~58us over the previous 8-way split
        # (BLOCK_K=128 / 4 warps). Placed first so autotune selects it quickly
        # when it wins (config order has no semantic effect, only compile cost).
        triton.Config(
            {"BLOCK_M": 32, "BLOCK_N": 32, "BLOCK_K": 256},
            num_warps=8,
            num_stages=4,
        ),
        triton.Config(
            {"BLOCK_M": 32, "BLOCK_N": 32, "BLOCK_K": 256},
            num_warps=8,
            num_stages=3,
        ),
        triton.Config(
            {"BLOCK_M": 32, "BLOCK_N": 32, "BLOCK_K": 256},
            num_warps=8,
            num_stages=2,
        ),
        triton.Config(
            {"BLOCK_M": 32, "BLOCK_N": 32, "BLOCK_K": 256},
            num_warps=4,
            num_stages=4,
        ),
        # 1-tile-per-segment variant (BLOCK_M=64). 64-row segments -> 1 M-tile,
        # so 3-way split -> 24 progs / 1.5 waves (vs 48 progs / 3 waves at
        # BLOCK_M=32). Both land within a few us; autotune picks whichever the
        # current shape prefers.
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 32, "BLOCK_K": 256},
            num_warps=8,
            num_stages=4,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 32, "BLOCK_K": 256},
            num_warps=8,
            num_stages=3,
        ),
        # Portable fallbacks: fit shared memory on every dtype / K-slice length,
        # including very small K-slices (short correctness cases use K_SPLIT=K
        # with n_splits=1, which routes to the non-split path instead).
        triton.Config(
            {"BLOCK_M": 32, "BLOCK_N": 32, "BLOCK_K": 128},
            num_warps=8,
            num_stages=3,
        ),
        triton.Config(
            {"BLOCK_M": 32, "BLOCK_N": 32, "BLOCK_K": 128},
            num_warps=4,
            num_stages=3,
        ),
        triton.Config(
            {"BLOCK_M": 32, "BLOCK_N": 32, "BLOCK_K": 512},
            num_warps=8,
            num_stages=2,
        ),
        triton.Config(
            {"BLOCK_M": 32, "BLOCK_N": 32, "BLOCK_K": 1024},
            num_warps=8,
            num_stages=2,
        ),
    ],
    key=["max_seg", "R", "K_SLICE", "IS_FP32"],
    # Each split-K program atomic_adds its partial sum into the fp32 accumulator,
    # so the kernel is NOT idempotent on acc_ptr -- running several autotune
    # candidate configs against the same buffer would double-/triple-count and
    # produce nonsense. reset_to_zero makes the autotuner zero the accumulator
    # before evaluating each candidate config, restoring idempotence.
    reset_to_zero=["acc_ptr"],
)
@triton.jit
def _kernel_n32_splitk(
    x_ptr,  # * [S, K]
    w_ptr,  # * [num_lora, R, K]
    acc_ptr,  # * [S, R] fp32 accumulator
    perm_ptr,  # * [S] or None
    seg_indptr,  # * [bs+1]
    weight_indices,  # * [bs]
    S,
    K,
    R,
    num_lora,
    BS,  # number of segments (grid z axis is BS * n_splits)
    xs_stride_m,
    xs_stride_k,  # x row/col stride
    ws_stride_l,
    ws_stride_n,
    ws_stride_k,  # weight lora/n/k stride
    os_stride_m,
    os_stride_n,  # acc row/col stride
    max_seg,  # autotune key: max segment length across segments
    K_SLICE,  # per-program K contraction length (cdiv(K, n_splits))
    HAS_PERM: tl.constexpr,
    IS_FP32: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_bs = tl.program_id(2)
    pid_b = pid_bs % BS
    pid_k = pid_bs // BS  # K-split index

    start = tl.load(seg_indptr + pid_b)
    end = tl.load(seg_indptr + pid_b + 1)
    seg_n = end - start

    w_idx = tl.load(weight_indices + pid_b)

    n_m_tiles = tl.cdiv(seg_n, BLOCK_M)
    n_n_tiles = tl.cdiv(R, BLOCK_N)
    if pid_m >= n_m_tiles or pid_n >= n_n_tiles:
        return

    # This program's K range within the full K contraction.
    k_start = pid_k * K_SLICE
    k_end = k_start + K_SLICE
    if k_end > K:
        k_end = K

    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
    x_lm = tl.arange(0, BLOCK_M)
    offs_m = pid_m * BLOCK_M + x_lm

    if HAS_PERM:
        out_row_idx = tl.load(
            perm_ptr + start + offs_m, mask=(offs_m < seg_n), other=0
        )
    else:
        out_row_idx = start + offs_m

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    w_base = w_ptr + w_idx * ws_stride_l

    for kk in range(k_start, k_end, BLOCK_K):
        k_off = kk + rk

        # Each split-K program owns the half-open K range [k_start, k_end);
        # loads outside this range must read zero (not the neighbouring slice's
        # weights) or the slices overlap and the atomic accumulation double
        # counts the overlapping K rows.
        in_slice = (k_off >= k_start) & (k_off < k_end)

        x_ptrs = (
            x_ptr
            + out_row_idx[:, None] * xs_stride_m
            + k_off[None, :] * xs_stride_k
        )
        x_tile = tl.load(
            x_ptrs,
            mask=(offs_m[:, None] < seg_n) & in_slice[None, :],
            other=0.0,
        )

        w_ptrs = (
            w_base + rn[:, None] * ws_stride_n + k_off[None, :] * ws_stride_k
        )
        w_tile = tl.load(
            w_ptrs, mask=(rn[:, None] < R) & in_slice[None, :], other=0.0
        )

        if IS_FP32:
            acc += tl.dot(
                x_tile, w_tile.T, out_dtype=tl.float32, input_precision="ieee"
            )
        else:
            acc += tl.dot(x_tile, w_tile.T, out_dtype=tl.float32)

    # Atomically add this K-slice's partial sum into the fp32 accumulator.
    acc_ptrs = (
        acc_ptr
        + out_row_idx[:, None] * os_stride_m
        + rn[None, :] * os_stride_n
    )
    acc_mask = (offs_m[:, None] < seg_n) & (rn[None, :] < R)
    tl.atomic_add(acc_ptrs, acc, mask=acc_mask)


# --- kernel_n64: BLOCK_N=64, for R > 32 (seg256x4 R=64, correctness R=96) ---
#
# BLOCK_M is autotuned (32 and 64) and the launch grid is sized with
# min_block_m=32 so a BLOCK_M=64 config no-ops its extra M-tiles via the
# ``pid_m >= cdiv(seg_n, BLOCK_M)`` early return -- this is safe and keeps the
# grid launching 32 programs (8 M-tiles x 1 N-tile x 4 segs = 2 full waves on
# the 16-SM device). seg256x4 is bf16 / K=4096 and *latency* bound: it has only
# 4 segments, so each SM must hide the K-loop latency through wave-level
# concurrency. The 32-program / 2-wave launch empirically beats a single 16-
# program wave here (each program does less work, so 2 wave-fronts overlap
# compute / memory better than one 64-row program per SM). The config set
# therefore centres on BLOCK_M=32 + BLOCK_K=128/256 with deep pipelining
# (stages 2/3) and 8 warps; a BLOCK_M=64 config is kept for shapes whose
# segments are long enough to amortise the larger tile.
@triton.autotune(
    configs=[
        # bf16 latency-bound winners: BLOCK_M=32, 8 warps, deep pipelining.
        triton.Config(
            {"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 128},
            num_warps=8,
            num_stages=2,
        ),
        triton.Config(
            {"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 128},
            num_warps=8,
            num_stages=3,
        ),
        # Deeper pipeline (stages 4/5) to hide K-loop memory latency on the
        # latency-bound seg256x4 shape; smaller BLOCK_K keeps shared memory in
        # budget for the extra in-flight stage buffers.
        triton.Config(
            {"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 64},
            num_warps=8,
            num_stages=4,
        ),
        triton.Config(
            {"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 64},
            num_warps=8,
            num_stages=5,
        ),
        triton.Config(
            {"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 128},
            num_warps=8,
            num_stages=4,
        ),
        triton.Config(
            {"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 256},
            num_warps=8,
            num_stages=2,
        ),
        triton.Config(
            {"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 256},
            num_warps=8,
            num_stages=3,
        ),
        # 4-warps variants (smaller register footprint, may lift occupancy).
        triton.Config(
            {"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 128},
            num_warps=4,
            num_stages=3,
        ),
        triton.Config(
            {"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 256},
            num_warps=4,
            num_stages=3,
        ),
        # Larger M-tile for long-segment shapes (no-ops extra tiles on seg256).
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 128},
            num_warps=8,
            num_stages=3,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 256},
            num_warps=8,
            num_stages=3,
        ),
    ],
    key=["max_seg", "R", "K", "IS_FP32"],
)
@triton.jit
def _kernel_n64(
    x_ptr,  # * [S, K]
    w_ptr,  # * [num_lora, R, K]
    out_ptr,  # * [S, R]
    perm_ptr,  # * [S] or None
    seg_indptr,  # * [bs+1]
    weight_indices,  # * [bs]
    S,
    K,
    R,
    num_lora,
    xs_stride_m,
    xs_stride_k,  # x row/col stride
    ws_stride_l,
    ws_stride_n,
    ws_stride_k,  # weight lora/n/k stride
    os_stride_m,
    os_stride_n,  # out row/col stride
    max_seg,  # autotune key: max segment length across segments
    HAS_PERM: tl.constexpr,
    IS_FP32: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_b = tl.program_id(2)

    start = tl.load(seg_indptr + pid_b)
    end = tl.load(seg_indptr + pid_b + 1)
    seg_n = end - start  # number of rows in this segment

    w_idx = tl.load(weight_indices + pid_b)

    n_m_tiles = tl.cdiv(seg_n, BLOCK_M)
    n_n_tiles = tl.cdiv(R, BLOCK_N)
    # Out-of-range tiles (segment shorter than max_seg, or N-tiles beyond R)
    # no-op rather than running a full K contraction just to mask out an empty
    # block.
    if pid_m >= n_m_tiles or pid_n >= n_n_tiles:
        return

    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
    x_lm = tl.arange(0, BLOCK_M)  # local segment row index
    offs_m = pid_m * BLOCK_M + x_lm

    # Absolute row index into x / out for this tile. With a permutation, rows
    # are gathered through perm[start + offs_m]; otherwise rows are contiguous
    # (start + offs_m). Computed once here so it is in scope for the store.
    if HAS_PERM:
        out_row_idx = tl.load(
            perm_ptr + start + offs_m, mask=(offs_m < seg_n), other=0
        )
    else:
        out_row_idx = start + offs_m

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Base pointer to this adapter's weight slice: weights[w_idx].
    w_base = w_ptr + w_idx * ws_stride_l

    for kk in range(0, K, BLOCK_K):
        k_off = kk + rk

        # x rows gathered via out_row_idx (absolute rows).
        x_ptrs = (
            x_ptr
            + out_row_idx[:, None] * xs_stride_m
            + k_off[None, :] * xs_stride_k
        )
        x_tile = tl.load(
            x_ptrs,
            mask=(offs_m[:, None] < seg_n) & (k_off[None, :] < K),
            other=0.0,
        )

        # weight slice: weights[w_idx, :, :] -> [R, K]. We need [BLOCK_N, BLOCK_K]
        # i.e. w_tile[n, k] = weights[w_idx, rn[n], k_off[k]].
        w_ptrs = (
            w_base + rn[:, None] * ws_stride_n + k_off[None, :] * ws_stride_k
        )
        w_tile = tl.load(
            w_ptrs, mask=(rn[:, None] < R) & (k_off[None, :] < K), other=0.0
        )

        if IS_FP32:
            acc += tl.dot(
                x_tile, w_tile.T, out_dtype=tl.float32, input_precision="ieee"
            )
        else:
            acc += tl.dot(x_tile, w_tile.T, out_dtype=tl.float32)

    # Store to out[rows, :].
    out_ptrs = (
        out_ptr
        + out_row_idx[:, None] * os_stride_m
        + rn[None, :] * os_stride_n
    )
    out_mask = (offs_m[:, None] < seg_n) & (rn[None, :] < R)

    if IS_FP32:
        tl.store(out_ptrs, acc, mask=out_mask)
    else:
        tl.store(out_ptrs, acc.to(out_ptr.dtype.element_ty), mask=out_mask)


# --- kernel_n64_splitk: split-K over K for the latency-bound n64 case. ---
#
# seg256x4_nlora2_r64_k4096 has only 4 segments x 4 M-tiles (BLOCK_M=64) x 1
# N-tile = 16 program tiles -- a single wave on the 16-SM device, with one
# program resident per SM and no wave-level memory-level parallelism, exactly
# the launch-latency-bound regime where the K contraction (K=4096) runs fully
# serial inside each program. This mirrors the n32 latency-bound case, so the
# same medicine applies: split K across a new grid axis so more programs run
# concurrently per SM, buying the wave-level memory-level parallelism a single
# wave structurally lacks.
#
# With ``n_splits`` slices the grid grows from 16 to ``16 * n_splits`` programs;
# each program does ``1/n_splits`` of the K contraction and contributes its
# partial sum to a shared fp32 accumulator via ``tl.atomic_add``. Each
# program's K slice is ``cdiv(K, n_splits)`` (ceiling, so *every* K row is
# covered exactly once even when n_splits does not divide K -- the half-open
# range ``[pid_k*K_SLICE, min((pid_k+1)*K_SLICE, K))`` tiles K with no gaps and
# no overlaps; out-of-slice elements are masked to zero by ``in_slice``).
#
# BLOCK_M is held at 64 (one tile covers 64 rows, so a 256-row segment is 4
# M-tiles; short correctness segments mask the tail inside one tile). The
# config set autotunes BLOCK_K / num_warps / num_stages around the
# empirically-winning BLOCK_K=128 / 4 warps / stages 2-3 family. n_splits=3 is
# the heuristic sweet spot for the bench shape (base 16 progs -> 48 progs / 3
# waves): it balances the concurrency gain against the atomic_add contention
# that grows with the split count (n_splits>=4 reverts to ~137us, worse than
# the non-split path's ~157us, so the heuristic picks 3).
#
# Because the partial sums accumulate in fp32 and atomically add into an fp32
# buffer, the kernel is precision-equivalent to the non-split fp32-accumulating
# path for the bf16/fp16 inputs it targets. fp32 inputs are excluded from
# split-K (``_choose_n64_splits`` returns 1): the reference is a single fused
# fp32 contraction per segment, and re-summing K-split fp32 partials perturbs
# the accumulation order enough to risk the strict ``atol=1e-4`` fp32
# tolerance; the non-split ``_kernel_n64`` path reproduces the reference
# exactly via ``input_precision="ieee"``. A tiny cast kernel then materialises
# the fp32 buffer into the output dtype (a no-op view for fp32, unused here).
@triton.autotune(
    configs=[
        # Empirical winners on seg256x4 bf16 (K=4096, K_SLICE=1366 for n=3):
        # BLOCK_K=128 + 4 warps, deep pipelining (stages 2/3/4 all within ~98us).
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 128},
            num_warps=4,
            num_stages=3,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 128},
            num_warps=4,
            num_stages=4,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 128},
            num_warps=4,
            num_stages=2,
        ),
        # 8-warp / other BLOCK_K variants -- kept so autotune can pick them on
        # shapes where the K-slice length or dtype shifts the balance.
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 128},
            num_warps=8,
            num_stages=3,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 64},
            num_warps=4,
            num_stages=3,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 64},
            num_warps=8,
            num_stages=3,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 256},
            num_warps=4,
            num_stages=3,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 256},
            num_warps=8,
            num_stages=3,
        ),
    ],
    key=["max_seg", "R", "K_SLICE", "IS_FP32"],
    # Each split-K program atomic_adds its partial sum into the fp32
    # accumulator, so the kernel is NOT idempotent on acc_ptr -- running
    # several autotune candidate configs against the same buffer would
    # double-/triple-count. reset_to_zero makes the autotuner zero the
    # accumulator before evaluating each candidate config.
    reset_to_zero=["acc_ptr"],
)
@triton.jit
def _kernel_n64_splitk(
    x_ptr,  # * [S, K]
    w_ptr,  # * [num_lora, R, K]
    acc_ptr,  # * [S, R] fp32 accumulator
    perm_ptr,  # * [S] or None
    seg_indptr,  # * [bs+1]
    weight_indices,  # * [bs]
    S,
    K,
    R,
    num_lora,
    BS,  # number of segments (grid z axis is BS * n_splits)
    xs_stride_m,
    xs_stride_k,  # x row/col stride
    ws_stride_l,
    ws_stride_n,
    ws_stride_k,  # weight lora/n/k stride
    os_stride_m,
    os_stride_n,  # acc row/col stride
    max_seg,  # autotune key: max segment length across segments
    K_SLICE,  # per-program K contraction length (cdiv(K, n_splits))
    HAS_PERM: tl.constexpr,
    IS_FP32: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_bs = tl.program_id(2)
    pid_b = pid_bs % BS
    pid_k = pid_bs // BS  # K-split index

    start = tl.load(seg_indptr + pid_b)
    end = tl.load(seg_indptr + pid_b + 1)
    seg_n = end - start

    w_idx = tl.load(weight_indices + pid_b)

    n_m_tiles = tl.cdiv(seg_n, BLOCK_M)
    n_n_tiles = tl.cdiv(R, BLOCK_N)
    if pid_m >= n_m_tiles or pid_n >= n_n_tiles:
        return

    # This program's K range within the full K contraction. Tiled with
    # K_SLICE = cdiv(K, n_splits) the half-open ranges cover every K row
    # exactly once (the last slice may be short; k_end is clamped to K).
    k_start = pid_k * K_SLICE
    k_end = k_start + K_SLICE
    if k_end > K:
        k_end = K

    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
    x_lm = tl.arange(0, BLOCK_M)
    offs_m = pid_m * BLOCK_M + x_lm

    if HAS_PERM:
        out_row_idx = tl.load(
            perm_ptr + start + offs_m, mask=(offs_m < seg_n), other=0
        )
    else:
        out_row_idx = start + offs_m

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    w_base = w_ptr + w_idx * ws_stride_l

    for kk in range(k_start, k_end, BLOCK_K):
        k_off = kk + rk

        # Each split-K program owns the half-open K range [k_start, k_end);
        # loads outside this range must read zero (not the neighbouring slice's
        # data) or the slices overlap and the atomic accumulation double
        # counts the overlapping K rows.
        in_slice = (k_off >= k_start) & (k_off < k_end)

        x_ptrs = (
            x_ptr
            + out_row_idx[:, None] * xs_stride_m
            + k_off[None, :] * xs_stride_k
        )
        x_tile = tl.load(
            x_ptrs,
            mask=(offs_m[:, None] < seg_n) & in_slice[None, :],
            other=0.0,
        )

        w_ptrs = (
            w_base + rn[:, None] * ws_stride_n + k_off[None, :] * ws_stride_k
        )
        w_tile = tl.load(
            w_ptrs, mask=(rn[:, None] < R) & in_slice[None, :], other=0.0
        )

        if IS_FP32:
            acc += tl.dot(
                x_tile, w_tile.T, out_dtype=tl.float32, input_precision="ieee"
            )
        else:
            acc += tl.dot(x_tile, w_tile.T, out_dtype=tl.float32)

    # Atomically add this K-slice's partial sum into the fp32 accumulator.
    acc_ptrs = (
        acc_ptr
        + out_row_idx[:, None] * os_stride_m
        + rn[None, :] * os_stride_n
    )
    acc_mask = (offs_m[:, None] < seg_n) & (rn[None, :] < R)
    tl.atomic_add(acc_ptrs, acc, mask=acc_mask)


def _choose_n_splits(R, K, max_seg, bs, min_block_m, min_block_n, is_fp32):
    """Heuristic for the K-split factor on the n32 path.

    Split-K only pays off when the launch is latency-bound: few base program
    tiles (so a single/few wave(s) leave SMs serialised on one program each)
    and a large K (so there is a real K contraction worth splitting). When the
    base program count already fills the device several times over, splitting
    K only adds atomic + cast overhead without buying concurrency. Returns 1
    (no split) in every other case so the small correctness cases and the
    well-filled shapes keep the plain non-atomic kernel.

    fp32 inputs skip split-K: the reference is a single fused fp32 contraction
    per segment (``.float() @ .float().T``), and re-splitting the K contraction
    across programs then re-summing partials via fp32 atomics perturbs the
    accumulation order enough to risk the strict ``atol=1e-4`` fp32 tolerance.
    The non-split path reproduces the reference exactly via
    ``input_precision="ieee"``. fp32 is also compute- (not latency-) bound, so
    it has less to gain from the extra concurrency anyway.
    """
    if is_fp32:
        return 1
    n_m_tiles = triton.cdiv(max_seg, min_block_m) if max_seg > 0 else 1
    if n_m_tiles < 1:
        n_m_tiles = 1
    n_n_tiles = triton.cdiv(R, min_block_n)
    base_progs = n_m_tiles * n_n_tiles * bs
    # Only the genuinely launch-bound case: <= ~2 waves on a 16-SM device, and
    # K large enough that splitting produces meaningful per-slice work.
    if base_progs > 32 or K < 2048:
        return 1
    # Target ~3 waves on a 16-SM device (48 programs). 3 is the empirical
    # winner for the seg64x8 bench shape: base 16 progs * 3 = 48 progs / 3
    # waves, with each program's K slice = cdiv(K, 3) = cdiv(4096, 3) = 1366
    # (~5 BLOCK_K=256 iterations) -- enough pipelined work per program to hide
    # the K-loop latency, while the 3-wave concurrency buys the wave-level
    # memory-level parallelism a single 16-prog wave lacks. Larger split
    # counts (8 -> 64 progs / 4 waves) over-split: each program's K slice
    # shrinks to ~512 (2 BLOCK_K iterations) so pipelining has nothing to
    # hide, and atomic_add contention + cast cost dominate -- 8-way measures
    # ~80us vs the 3-way ~58us.
    #
    # The split-K kernel tiles K with cdiv(K, n_splits) half-open ranges
    # (see ``_kernel_n32_splitk``), so a non-divisor n_splits is safe: every
    # K row is covered exactly once (the last slice may be short, clamped by
    # ``min(k_start + K_SLICE, K)`` and masked by ``in_slice``). This is what
    # lets n_splits=3 win on K=4096 (4096 % 3 != 0) -- no K-divisibility
    # constraint is needed.
    target = 3 * 16
    for cand in (3, 4, 2, 5, 6, 8):
        if base_progs * cand >= target:
            return cand
    return 1


def _choose_n64_splits(R, K, max_seg, bs, is_fp32):
    """Heuristic for the K-split factor on the n64 (R > 32) path.

    Mirrors the n32 reasoning: split-K only pays off when the launch is
    latency-bound -- few base program tiles (a single/few wave(s) leave SMs
    serialised on one program each) and a large K contraction worth splitting.
    Returns 1 (no split -> plain non-atomic ``_kernel_n64``) in every other
    case so the small correctness shapes (short K, few rows) keep the
    non-atomic path.

    The n64 split-K kernel uses BLOCK_M=64 (min_block_m=64) and BLOCK_N=64, so
    the base program count is ``cdiv(max_seg, 64) * cdiv(R, 64) * bs``. For the
    seg256x4 bench shape that is 4 * 1 * 4 = 16 -- a single wave on the 16-SM
    device.

    The split target is ~48 programs (3 waves), not 4-per-SM: on this shape
    atomic_add contention rises sharply once the split count pushes past 3
    waves (measured n=4 -> ~137us, worse than the non-split ~157us; n=3 ->
    ~98us). So the candidate list favours 3, then 2, before falling back to
    other divisors that reach the 3-wave target. Unlike ``_choose_n_splits``
    the n64 split-K kernel handles non-divisor n_splits correctly (it tiles K
    with ``cdiv(K, n_splits)`` half-open ranges), so we do NOT require
    ``K % n_splits == 0`` -- n=3 (K=4096 % 3 != 0) is the empirical winner.

    fp32 inputs skip split-K for the same precision reason as the n32 path:
    re-summing K-split fp32 partials perturbs the accumulation order enough to
    risk the strict ``atol=1e-4`` fp32 tolerance, and the non-split n64 kernel
    reproduces the reference exactly via ``input_precision="ieee"``.
    """
    if is_fp32:
        return 1
    n_m_tiles = triton.cdiv(max_seg, 64) if max_seg > 0 else 1
    if n_m_tiles < 1:
        n_m_tiles = 1
    n_n_tiles = triton.cdiv(R, 64)
    base_progs = n_m_tiles * n_n_tiles * bs
    if base_progs > 32 or K < 2048:
        return 1
    # Target ~3 waves on a 16-SM device (48 programs). n=3 is the empirical
    # winner for the seg256x4 bench shape (16 base progs * 3 = 48); the
    # n64 split-K kernel's cdiv-based K tiling makes non-divisor splits safe.
    target = 3 * 16
    for cand in (3, 2, 4, 6, 5, 8):
        if base_progs * cand >= target:
            return cand
    return 1


@triton.jit
def _cast_fp32_to_dtype(acc_ptr, out_ptr, total, BLOCK: tl.constexpr):
    """Cast the fp32 split-K accumulator to the output dtype.

    One program per ``[BLOCK]`` element tile of the flat ``[S*R]`` output; the
    work is a trivial dtype conversion, so a wide vectorised tile keeps this
    pass to a few microseconds on the bench shapes (16K-65K elements). The
    accumulator and output are both contiguous ``[S, R]`` tensors, so flat
    element indexing is correct.
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < total
    val = tl.load(acc_ptr + offs, mask=mask, other=0.0)
    tl.store(out_ptr + offs, val.to(out_ptr.dtype.element_ty), mask=mask)


def sgemm_lora_a(x, weights, batch_info, stack_num=1):
    S, K = x.shape
    num_lora, R, _ = weights.shape
    assert weights.shape[2] == K, "weights last dim must equal K"
    assert R == stack_num * (weights.shape[1] // stack_num), "R mismatch"

    out = torch.zeros(S, R, dtype=x.dtype, device=x.device)

    seg_indptr = batch_info.seg_indptr
    weight_indices = batch_info.weight_indices
    permutation = batch_info.permutation
    bs = batch_info.bs

    has_perm = permutation is not None

    # All segments share the same length in the benchmark shapes; the autotune
    # key uses a representative (max) segment length so configs are reused
    # across tiles of a segment.
    max_seg = int(batch_info.max_len) if batch_info.max_len is not None else 0
    if max_seg <= 0:
        max_seg = (
            int((seg_indptr[1:] - seg_indptr[:-1]).max().item())
            if bs > 0
            else 0
        )

    is_fp32 = x.dtype == torch.float32

    # Dispatch by R so the autotune config set matches the N-tile width. The
    # grid's N-tile count uses the *minimum* BLOCK_N of the chosen kernel's
    # configs so a config with a larger BLOCK_N never drops output rows; the
    # M-tile count uses the minimum BLOCK_M (32) so a config with a larger
    # BLOCK_M simply no-ops its extra tiles (``pid_m >= cdiv(seg_n, BLOCK_M)``
    # early return). One kernel per N-tile width keeps BLOCK_N uniform within
    # each config set -- this is what makes the min-BLOCK_N grid safe.
    if R <= 32:
        min_block_n = 32
        min_block_m = 32

        # For the latency-bound n32 case (few program tiles + large K), split K
        # across a new grid axis so more programs run concurrently per SM,
        # buying the wave-level memory-level parallelism a single wave lacks.
        # Partial sums land in an fp32 accumulator via atomic_add; a tiny cast
        # kernel then writes the accumulator to the output dtype. When the
        # heuristic says no split (n_splits == 1) the plain non-atomic n32
        # kernel is used directly on ``out``.
        n_splits = _choose_n_splits(
            R, K, max_seg, bs, min_block_m, min_block_n, is_fp32
        )

        n_m_tiles = triton.cdiv(max_seg, min_block_m)
        if n_m_tiles < 1:
            n_m_tiles = 1
        n_n_tiles = triton.cdiv(R, min_block_n)

        if n_splits > 1:
            # ceil division so EVERY K row is covered exactly once even when
            # n_splits does not divide K (e.g. K=4096, n_splits=3 -> K_slice=1366,
            # slices [0,1366),[1366,2732),[2732,4096)). The kernel clamps the
            # last slice with ``min(k_start + K_SLICE, K)`` and masks out-of-slice
            # loads with ``in_slice``, so non-divisor splits are correct.
            K_slice = triton.cdiv(K, n_splits)
            acc = torch.zeros(S, R, dtype=torch.float32, device=x.device)
            split_grid = (n_m_tiles, n_n_tiles, bs * n_splits)
            _kernel_n32_splitk[split_grid](
                x,
                weights,
                acc,
                permutation,
                seg_indptr,
                weight_indices,
                S,
                K,
                R,
                num_lora,
                bs,
                x.stride(0),
                x.stride(1),
                weights.stride(0),
                weights.stride(1),
                weights.stride(2),
                acc.stride(0),
                acc.stride(1),
                max_seg,
                K_slice,
                HAS_PERM=has_perm,
                IS_FP32=is_fp32,
            )
            # Cast the fp32 accumulator into the output dtype. For fp32 inputs
            # this is a straight copy; for bf16/fp16 a downcast.
            total = S * R
            BLOCK = 4096
            cast_grid = (triton.cdiv(total, BLOCK),)
            _cast_fp32_to_dtype[cast_grid](
                acc,
                out,
                total,
                BLOCK=BLOCK,
            )
        else:
            grid = (n_m_tiles, n_n_tiles, bs)
            _kernel_n32[grid](
                x,
                weights,
                out,
                permutation,
                seg_indptr,
                weight_indices,
                S,
                K,
                R,
                num_lora,
                x.stride(0),
                x.stride(1),
                weights.stride(0),
                weights.stride(1),
                weights.stride(2),
                out.stride(0),
                out.stride(1),
                max_seg,
                HAS_PERM=has_perm,
                IS_FP32=is_fp32,
            )
    else:
        min_block_n = 64
        # BLOCK_M=64 for the split-K kernel (one tile covers 64 rows); the
        # non-split ``_kernel_n64`` autotunes BLOCK_M over {32, 64} so its grid
        # is sized with min_block_m=32 (a BLOCK_M=64 config no-ops its extra
        # M-tiles). When split-K is selected we use the dedicated
        # ``_kernel_n64_splitk`` (BLOCK_M=64 only) sized with min_block_m=64.
        non_split_min_block_m = 32
        n_m_tiles_non = triton.cdiv(max_seg, non_split_min_block_m)
        if n_m_tiles_non < 1:
            n_m_tiles_non = 1
        n_n_tiles = triton.cdiv(R, min_block_n)

        # For the latency-bound n64 case (few program tiles + large K, e.g.
        # seg256x4: 16 base progs / 1 wave, K=4096), split K across a new grid
        # axis so more programs run concurrently per SM, buying the wave-level
        # memory-level parallelism a single wave lacks. Partial sums land in
        # an fp32 accumulator via atomic_add; a tiny cast kernel then writes
        # the accumulator to the output dtype. fp32 is excluded (kept on the
        # non-split path for exact ieee match). When n_splits == 1 the plain
        # non-atomic n64 kernel is used directly on ``out``.
        n_splits = _choose_n64_splits(R, K, max_seg, bs, is_fp32)

        if n_splits > 1:
            split_min_block_m = 64
            n_m_tiles = triton.cdiv(max_seg, split_min_block_m)
            if n_m_tiles < 1:
                n_m_tiles = 1
            K_slice = triton.cdiv(K, n_splits)
            acc = torch.zeros(S, R, dtype=torch.float32, device=x.device)
            split_grid = (n_m_tiles, n_n_tiles, bs * n_splits)
            _kernel_n64_splitk[split_grid](
                x,
                weights,
                acc,
                permutation,
                seg_indptr,
                weight_indices,
                S,
                K,
                R,
                num_lora,
                bs,
                x.stride(0),
                x.stride(1),
                weights.stride(0),
                weights.stride(1),
                weights.stride(2),
                acc.stride(0),
                acc.stride(1),
                max_seg,
                K_slice,
                HAS_PERM=has_perm,
                IS_FP32=is_fp32,
            )
            # Cast the fp32 accumulator into the output dtype. For fp32 inputs
            # this is a straight copy; for bf16/fp16 a downcast.
            total = S * R
            BLOCK = 4096
            cast_grid = (triton.cdiv(total, BLOCK),)
            _cast_fp32_to_dtype[cast_grid](
                acc,
                out,
                total,
                BLOCK=BLOCK,
            )
        else:
            grid = (n_m_tiles_non, n_n_tiles, bs)
            _kernel_n64[grid](
                x,
                weights,
                out,
                permutation,
                seg_indptr,
                weight_indices,
                S,
                K,
                R,
                num_lora,
                x.stride(0),
                x.stride(1),
                weights.stride(0),
                weights.stride(1),
                weights.stride(2),
                out.stride(0),
                out.stride(1),
                max_seg,
                HAS_PERM=has_perm,
                IS_FP32=is_fp32,
            )
    return out


__all__ = ["sgemm_lora_a"]
