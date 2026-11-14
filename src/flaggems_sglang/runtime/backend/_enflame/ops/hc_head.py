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

"""Triton implementation of the fused DSV4 ``hc_head`` LM-head mixer.

The op fuses: RMSNorm over the flattened ``[T, hc_mult * hidden_size]`` view,
a linear mix ``mixes = x_flat @ hc_fn.T * rsqrt``, a sigmoid gate
``pre = sigmoid(mixes * hc_scale + hc_base) + hc_eps``, and a weighted
reduction ``y = sum_j pre[:, j] * x[:, j, :]`` collapsing the ``hc_mult`` axis
to ``[T, hidden_size]``.

Strategy (v8)
-------------
  * **hc_mult <= 4 -- single fused kernel (``_hc_head_kernel_v8``), one launch
    for every T.** Pass 1 walks the flattened row in ``BLOCK_K`` tiles with
    *per-lane delayed-reduction accumulators* (one ``[BLOCK_K]`` fp32 vector
    per (token, reduction) pair, textually unrolled): the k-loop body is only
    loads + vector FMAs -- reduction trees run exactly 5 times per token at
    end-of-pass-1. The 4 ``hc_fn`` row slices are loaded once per k-tile and
    shared by all ``BLOCK_T`` tokens in the program, amortising the 448KB
    weight re-read across the token block (the dominant traffic at large T).
    ``BLOCK_T`` tokens per program are likewise textually unrolled so every
    accumulator stays a plain 1D vector (2D loop-carried accumulator tiles
    are both slow and miscompiled on this backend). The tail token block is
    anchored (``t0 = min(pid * BLOCK_T, T - BLOCK_T)``), so the overlapping
    re-computation writes identical values (idempotent) and no token masks
    are needed. ``K``/``HIDDEN`` are constexpr (static trip counts, masks
    fold away when divisible) and the RMSNorm mean uses a host-computed
    reciprocal (``sq * inv_k``). Pass 2 re-streams the hidden axis per token
    with ``y[t, d] = sum_j pre_j * x[t, j, d]``.

    Dispatch (GCU400/Libra, benchmarked on (T, 4, 7168) bf16):
    ``BLOCK_T = 1 if T <= 48 else 2`` (single-wave chains win below the
    machine-saturation point; the shared-weight amortisation wins above),
    ``BLOCK_K = min(2048, next_pow2(K))``, ``BLOCK_D = min(4096,
    next_pow2(hidden))``, ``num_warps = 8``. One exception (2026-11-12
    rescan): for very small token counts (``T <= 24``) a fat
    ``BLOCK_K = min(8192, next_pow2(K))`` tile wins on the BT=1 arm
    (14.5 vs 15.4us at T=1, 13.8 vs 15.3us at T=4..24; measured
    ``gcu.dsm_memory_size`` 176128B still fits the 2-programs/SP line and
    T <= 24 programs cannot oversubscribe the SPs). The crossover sits
    between T=24 (BK=8192 wins by 1.1us) and T=28 (BK=8192 loses by
    8.1us), so the arm is capped at 24.

  * **hc_mult in 5..8 -- legacy v7 paths unchanged**: the k-split two-kernel
    partial path for ``T <= 256`` and the tree-per-tile fused kernel for
    larger ``T`` (see below). These shapes never occur in the benchmark or
    test suite; the code is kept for compatibility.

All math is fp32 (matching the reference's ``.float()``); the output is cast
back to ``x.dtype``. Portability: plain loads/stores and fp32 arithmetic only
-- no atomics, no vendor extensions, no dtype-specific hacks.
"""

import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# launch config search spaces
# ---------------------------------------------------------------------------


def _fused_configs():
    # Large-T fused kernel: the op is memory-bound so the main levers are tile
    # sizes (occupancy / latency hiding), not warp count. BLOCK_K walks the
    # flattened row; BLOCK_D walks the hidden axis in pass 2. BLOCK_T groups
    # tokens per program. The benchmark row length is K = 4 * 7168 = 28672, so
    # BLOCK_K values that align one k-tile to one hc_mult slice maximise
    # coalesced reuse of the xk tile across the hc_mult dot rows.
    cfgs = []
    for block_t in (1, 8, 32):
        for block_k in (1024, 2048, 4096, 8192):
            for block_d in (256, 1024, 4096):
                cfgs.append(
                    triton.Config(
                        {
                            "BLOCK_T": block_t,
                            "BLOCK_K": block_k,
                            "BLOCK_D": block_d,
                        },
                        num_warps=4,
                        num_stages=3,
                    )
                )
    # Low-occupancy configs: fewer warps so a single program fills a SM's
    # lanes without wasting them when there are very few tokens.
    for block_t in (1, 4):
        for block_d in (256, 1024, 4096):
            cfgs.append(
                triton.Config(
                    {"BLOCK_T": block_t, "BLOCK_K": 2048, "BLOCK_D": block_d},
                    num_warps=2,
                    num_stages=3,
                )
            )
            cfgs.append(
                triton.Config(
                    {"BLOCK_T": block_t, "BLOCK_K": 4096, "BLOCK_D": block_d},
                    num_warps=2,
                    num_stages=2,
                )
            )
            cfgs.append(
                triton.Config(
                    {"BLOCK_T": block_t, "BLOCK_K": 8192, "BLOCK_D": block_d},
                    num_warps=4,
                    num_stages=2,
                )
            )
    # Higher warp count option for large-T occupancy hiding.
    for block_t in (8, 32):
        for block_d in (1024, 4096):
            cfgs.append(
                triton.Config(
                    {"BLOCK_T": block_t, "BLOCK_K": 4096, "BLOCK_D": block_d},
                    num_warps=8,
                    num_stages=2,
                )
            )
            cfgs.append(
                triton.Config(
                    {"BLOCK_T": block_t, "BLOCK_K": 8192, "BLOCK_D": block_d},
                    num_warps=8,
                    num_stages=2,
                )
            )
    # Few-programs / large-tile configs for the latency-bound regime:
    # maximise work per program so launch + loop overhead amortises.
    for block_t in (4, 8):
        for block_d in (2048, 4096):
            cfgs.append(
                triton.Config(
                    {"BLOCK_T": block_t, "BLOCK_K": 16384, "BLOCK_D": block_d},
                    num_warps=8,
                    num_stages=2,
                )
            )
    # Persistent configs: BLOCK_K spanning the whole flattened row (single
    # k-tile) so the k-loop collapses entirely; high warp counts feed the
    # wide tile.
    for block_k in (8192, 16384, 32768):
        for warps in (8, 16):
            cfgs.append(
                triton.Config(
                    {"BLOCK_T": 1, "BLOCK_K": block_k, "BLOCK_D": 4096},
                    num_warps=warps,
                    num_stages=2,
                )
            )
    return cfgs


def _partial_a_configs():
    # Kernel A: pass-1 partials (sqsum + mix dots) over a k-chunk.
    # 2D grid (token-blocks x KSPLIT). KSPLIT is a plain runtime arg (not in
    # the config dict, and not autotuned -- see below): the grid is
    # (cdiv(T, BLOCK_T) x KSPLIT) and the scratch is sized by it.
    #
    # NOTE: KSPLIT is *not* autotuned -- it is chosen by a Python-side
    # heuristic in ``hc_head`` because the k-split scratch buffer is sized by
    # it and every slot must be overwritten on every launch (autotune reruns
    # with different KSPLIT would leave stale slots in a shared buffer).
    cfgs = []
    for block_t in (1, 4):
        for block_k in (2048, 4096):
            for warps, stages in ((4, 3), (2, 2)):
                cfgs.append(
                    triton.Config(
                        {"BLOCK_T": block_t, "BLOCK_K": block_k},
                        num_warps=warps,
                        num_stages=stages,
                    )
                )
    return cfgs


def _finish_configs():
    # Kernel B: reduce partials + gate + weighted sum. 2D grid over
    # (token-block x hidden-block). BLOCK_D controls how many hidden columns
    # a program owns; small BLOCK_D exposes more programs so small-T fills
    # the SMs.
    cfgs = []
    for block_t in (1, 2, 4):
        for block_d in (256, 512, 1024, 2048, 4096):
            cfgs.append(
                triton.Config(
                    {"BLOCK_T": block_t, "BLOCK_D": block_d},
                    num_warps=4,
                    num_stages=3,
                )
            )
    for block_t in (1, 4):
        for block_d in (512, 1024, 2048):
            cfgs.append(
                triton.Config(
                    {"BLOCK_T": block_t, "BLOCK_D": block_d},
                    num_warps=2,
                    num_stages=2,
                )
            )
            cfgs.append(
                triton.Config(
                    {"BLOCK_T": block_t, "BLOCK_D": block_d},
                    num_warps=8,
                    num_stages=2,
                )
            )
    return cfgs


# ---------------------------------------------------------------------------
# k-split path, kernel A: partial sqsum + partial mix dots over a k-chunk
# ---------------------------------------------------------------------------


@triton.autotune(
    configs=_partial_a_configs(),
    key=["T", "HC_MULT", "HIDDEN", "K", "X_STRIDE_T"],
)
@triton.jit
def _hc_head_partial_kernel(
    x_ptr,
    hc_fn_ptr,
    part_ptr,
    T,
    HC_MULT: tl.constexpr,
    K,  # HC_MULT * HIDDEN (flattened row length)
    X_STRIDE_T,  # stride between tokens in x: HC_MULT * HIDDEN
    HC_FN_ROW_STRIDE,  # stride between rows of hc_fn: HC_MULT * HIDDEN
    PART_STRIDE_T,  # stride between tokens in the scratch: KSPLIT*(HC_MULT+1)
    PART_STRIDE_K,  # stride between k-partials in the scratch: HC_MULT+1
    KSPLIT,  # number of k-chunks the row was split into
    BLOCK_T: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_t = tl.program_id(0)  # token-block index
    pid_k = tl.program_id(1)  # k-chunk index (grid axis 1 == KSPLIT)
    offs_t = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)  # [BLOCK_T]
    t_mask = offs_t < T  # [BLOCK_T]

    # this program owns flat-row range [k_start, k_end)
    chunk = tl.cdiv(K, KSPLIT)
    k_start = pid_k.to(tl.int64) * chunk
    k_end = tl.minimum(k_start + chunk, K)

    # sqsum accumulator (shared across all hc_mult rows) and one scalar-per-token
    # dot accumulator per hc_mult row -- unrolled named accumulators keep the
    # working set in registers (no [BLOCK_T, HC_MULT, BLOCK_K] intermediate).
    sqacc = tl.zeros((BLOCK_T,), dtype=tl.float32)
    mix0 = tl.zeros((BLOCK_T,), dtype=tl.float32)
    mix1 = tl.zeros((BLOCK_T,), dtype=tl.float32)
    mix2 = tl.zeros((BLOCK_T,), dtype=tl.float32)
    mix3 = tl.zeros((BLOCK_T,), dtype=tl.float32)
    mix4 = tl.zeros((BLOCK_T,), dtype=tl.float32)
    mix5 = tl.zeros((BLOCK_T,), dtype=tl.float32)
    mix6 = tl.zeros((BLOCK_T,), dtype=tl.float32)
    mix7 = tl.zeros((BLOCK_T,), dtype=tl.float32)

    k_idx = tl.arange(0, BLOCK_K)  # [BLOCK_K]
    x_base_t = offs_t.to(tl.int64) * X_STRIDE_T  # [BLOCK_T]

    for k0 in range(0, chunk, BLOCK_K):
        k_off = k_start + k0 + k_idx  # [BLOCK_K]
        k_mask = k_off < k_end  # [BLOCK_K]

        # x_flat[t, k] = x[t, k // HIDDEN, k % HIDDEN]; offsets are contiguous
        # in the flattened view so a single coalesced load per token.
        x_off = x_base_t[:, None] + k_off[None, :]  # [BLOCK_T, BLOCK_K]
        xk = tl.load(
            x_ptr + x_off,
            mask=t_mask[:, None] & k_mask[None, :],
            other=0.0,
        ).to(
            tl.float32
        )  # [BLOCK_T, BLOCK_K]

        sqacc += tl.sum(xk * xk, axis=1)  # [BLOCK_T]

        for j in tl.static_range(0, HC_MULT):
            hck_j = tl.load(
                hc_fn_ptr + j * HC_FN_ROW_STRIDE + k_off,
                mask=k_mask,
                other=0.0,
            ).to(
                tl.float32
            )  # [BLOCK_K]
            partial = tl.sum(xk * hck_j[None, :], axis=1)  # [BLOCK_T]
            if j == 0:
                mix0 = mix0 + partial
            elif j == 1:
                mix1 = mix1 + partial
            elif j == 2:
                mix2 = mix2 + partial
            elif j == 3:
                mix3 = mix3 + partial
            elif j == 4:
                mix4 = mix4 + partial
            elif j == 5:
                mix5 = mix5 + partial
            elif j == 6:
                mix6 = mix6 + partial
            elif j == 7:
                mix7 = mix7 + partial

    # store this chunk's partials; slot HC_MULT holds the sqsum. The scratch
    # is sized exactly (T, KSPLIT, HC_MULT+1) and every slot < KSPLIT is
    # overwritten on each launch (KSPLIT is fixed by the caller, not
    # autotuned), so torch.empty suffices -- no stale-slot hazard.
    part_base = (
        offs_t.to(tl.int64)[:, None] * PART_STRIDE_T
        + pid_k.to(tl.int64) * PART_STRIDE_K
    )  # [BLOCK_T, 1]

    sq_off = part_base + HC_MULT  # [BLOCK_T, 1]
    tl.store(part_ptr + sq_off, sqacc[:, None], mask=t_mask[:, None])

    for j in tl.static_range(0, HC_MULT):
        if j == 0:
            mixj = mix0
        elif j == 1:
            mixj = mix1
        elif j == 2:
            mixj = mix2
        elif j == 3:
            mixj = mix3
        elif j == 4:
            mixj = mix4
        elif j == 5:
            mixj = mix5
        elif j == 6:
            mixj = mix6
        elif j == 7:
            mixj = mix7
        tl.store(
            part_ptr + part_base + j,
            mixj[:, None],
            mask=t_mask[:, None],
        )


# ---------------------------------------------------------------------------
# k-split path, kernel B: reduce partials -> rsqrt + gate -> weighted sum
# ---------------------------------------------------------------------------


@triton.autotune(
    configs=_finish_configs(),
    key=["T", "HC_MULT", "HIDDEN", "K", "X_STRIDE_T"],
)
@triton.jit
def _hc_head_finish_kernel(
    x_ptr,
    part_ptr,
    hc_scale_ptr,
    hc_base_ptr,
    y_ptr,
    norm_eps,
    hc_eps,
    T,
    HC_MULT: tl.constexpr,
    HIDDEN,
    K,
    X_STRIDE_T,
    X_STRIDE_M,  # stride between hc_mult slices in x: HIDDEN
    Y_STRIDE_T,  # stride between tokens in y: HIDDEN
    PART_STRIDE_T,
    PART_STRIDE_K,
    KSPLIT,  # number of k-chunks the row was split into
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_t = tl.program_id(0)
    pid_d = tl.program_id(1)
    offs_t = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)  # [BLOCK_T]
    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)  # [BLOCK_D]
    t_mask = offs_t < T  # [BLOCK_T]
    d_mask = offs_d < HIDDEN  # [BLOCK_D]

    # ---- reduce the k-split partials (exactly KSPLIT live slots) ----
    # Plain 1D loads (one scalar slot per token per k-chunk) so every read
    # slot is one kernel A wrote; no atomics, no padding slots.
    part_base_t = offs_t.to(tl.int64) * PART_STRIDE_T  # [BLOCK_T]

    sqacc = tl.zeros((BLOCK_T,), dtype=tl.float32)
    mix0 = tl.zeros((BLOCK_T,), dtype=tl.float32)
    mix1 = tl.zeros((BLOCK_T,), dtype=tl.float32)
    mix2 = tl.zeros((BLOCK_T,), dtype=tl.float32)
    mix3 = tl.zeros((BLOCK_T,), dtype=tl.float32)
    mix4 = tl.zeros((BLOCK_T,), dtype=tl.float32)
    mix5 = tl.zeros((BLOCK_T,), dtype=tl.float32)
    mix6 = tl.zeros((BLOCK_T,), dtype=tl.float32)
    mix7 = tl.zeros((BLOCK_T,), dtype=tl.float32)

    for p in range(0, KSPLIT):
        p_off = part_base_t + p.to(tl.int64) * PART_STRIDE_K  # [BLOCK_T]
        sqacc += tl.load(part_ptr + p_off + HC_MULT, mask=t_mask, other=0.0)
        if HC_MULT > 0:
            mix0 = mix0 + tl.load(part_ptr + p_off + 0, mask=t_mask, other=0.0)
        if HC_MULT > 1:
            mix1 = mix1 + tl.load(part_ptr + p_off + 1, mask=t_mask, other=0.0)
        if HC_MULT > 2:
            mix2 = mix2 + tl.load(part_ptr + p_off + 2, mask=t_mask, other=0.0)
        if HC_MULT > 3:
            mix3 = mix3 + tl.load(part_ptr + p_off + 3, mask=t_mask, other=0.0)
        if HC_MULT > 4:
            mix4 = mix4 + tl.load(part_ptr + p_off + 4, mask=t_mask, other=0.0)
        if HC_MULT > 5:
            mix5 = mix5 + tl.load(part_ptr + p_off + 5, mask=t_mask, other=0.0)
        if HC_MULT > 6:
            mix6 = mix6 + tl.load(part_ptr + p_off + 6, mask=t_mask, other=0.0)
        if HC_MULT > 7:
            mix7 = mix7 + tl.load(part_ptr + p_off + 7, mask=t_mask, other=0.0)

    rsqrt = tl.rsqrt(sqacc / K.to(tl.float32) + norm_eps)  # [BLOCK_T]
    hc_scale = tl.load(hc_scale_ptr).to(tl.float32)  # scalar

    # ---- weighted sum y[t, d] = sum_j pre[t,j] * x[t,j,d] ----
    x_base_t = offs_t.to(tl.int64) * X_STRIDE_T  # [BLOCK_T]

    for j in tl.static_range(0, HC_MULT):
        if j == 0:
            mixj = mix0
        elif j == 1:
            mixj = mix1
        elif j == 2:
            mixj = mix2
        elif j == 3:
            mixj = mix3
        elif j == 4:
            mixj = mix4
        elif j == 5:
            mixj = mix5
        elif j == 6:
            mixj = mix6
        elif j == 7:
            mixj = mix7
        pre_j = (
            tl.sigmoid(
                mixj * rsqrt * hc_scale
                + tl.load(hc_base_ptr + j).to(tl.float32)
            )
            + hc_eps
        )  # [BLOCK_T]

        xoff = x_base_t[:, None] + j * X_STRIDE_M + offs_d[None, :]
        xd = tl.load(
            x_ptr + xoff,
            mask=t_mask[:, None] & d_mask[None, :],
            other=0.0,
        ).to(
            tl.float32
        )  # [BLOCK_T, BLOCK_D]

        if j == 0:
            yd = pre_j[:, None] * xd  # [BLOCK_T, BLOCK_D]
        else:
            yd += pre_j[:, None] * xd

    y_store_off = offs_t.to(tl.int64)[:, None] * Y_STRIDE_T + offs_d[None, :]
    tl.store(
        y_ptr + y_store_off,
        yd.to(y_ptr.dtype.element_ty),
        mask=t_mask[:, None] & d_mask[None, :],
    )


# ---------------------------------------------------------------------------
# fused single kernel (large T): two serial in-kernel passes
# ---------------------------------------------------------------------------


@triton.autotune(
    configs=_fused_configs(),
    key=["T", "HC_MULT", "HIDDEN", "X_STRIDE_T", "X_STRIDE_M"],
)
@triton.jit
def _hc_head_kernel(
    x_ptr,
    hc_fn_ptr,
    hc_scale_ptr,
    hc_base_ptr,
    y_ptr,
    norm_eps,
    hc_eps,
    T,
    HC_MULT: tl.constexpr,
    HIDDEN,
    K,  # HC_MULT * HIDDEN (flattened row length)
    X_STRIDE_T,  # stride between tokens in x: HC_MULT * HIDDEN
    X_STRIDE_M,  # stride between hc_mult slices in x: HIDDEN
    Y_STRIDE_T,  # stride between tokens in y: HIDDEN
    HC_FN_ROW_STRIDE,  # stride between rows of hc_fn: HC_MULT * HIDDEN
    BLOCK_T: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_t = tl.program_id(0)  # token-block index
    offs_t = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)  # [BLOCK_T]
    t_mask = offs_t < T  # [BLOCK_T]

    # ---- pass 1: norm (sqsum) + linear mix (dot into hc_fn) ----
    sqacc = tl.zeros((BLOCK_T,), dtype=tl.float32)  # sum x^2 per token
    mix0 = tl.zeros((BLOCK_T,), dtype=tl.float32)
    mix1 = tl.zeros((BLOCK_T,), dtype=tl.float32)
    mix2 = tl.zeros((BLOCK_T,), dtype=tl.float32)
    mix3 = tl.zeros((BLOCK_T,), dtype=tl.float32)
    mix4 = tl.zeros((BLOCK_T,), dtype=tl.float32)
    mix5 = tl.zeros((BLOCK_T,), dtype=tl.float32)
    mix6 = tl.zeros((BLOCK_T,), dtype=tl.float32)
    mix7 = tl.zeros((BLOCK_T,), dtype=tl.float32)

    num_k_tiles = tl.cdiv(K, BLOCK_K)
    k_idx = tl.arange(0, BLOCK_K)  # [BLOCK_K]

    x_base_t = offs_t.to(tl.int64) * X_STRIDE_T  # [BLOCK_T]

    for k0 in range(0, num_k_tiles):
        k_off = k0 * BLOCK_K + k_idx  # [BLOCK_K]
        k_mask = k_off < K  # [BLOCK_K]

        x_off = x_base_t[:, None] + k_off[None, :]  # [BLOCK_T, BLOCK_K]
        xk = tl.load(
            x_ptr + x_off,
            mask=t_mask[:, None] & k_mask[None, :],
            other=0.0,
        ).to(
            tl.float32
        )  # [BLOCK_T, BLOCK_K]

        sqacc += tl.sum(xk * xk, axis=1)  # [BLOCK_T]

        for j in tl.static_range(0, HC_MULT):
            hck_j = tl.load(
                hc_fn_ptr + j * HC_FN_ROW_STRIDE + k_off,
                mask=k_mask,
                other=0.0,
            ).to(
                tl.float32
            )  # [BLOCK_K]
            partial = tl.sum(xk * hck_j[None, :], axis=1)  # [BLOCK_T]
            if j == 0:
                mix0 = mix0 + partial
            elif j == 1:
                mix1 = mix1 + partial
            elif j == 2:
                mix2 = mix2 + partial
            elif j == 3:
                mix3 = mix3 + partial
            elif j == 4:
                mix4 = mix4 + partial
            elif j == 5:
                mix5 = mix5 + partial
            elif j == 6:
                mix6 = mix6 + partial
            elif j == 7:
                mix7 = mix7 + partial

    # rsqrt and finalize mixes
    rsqrt = tl.rsqrt(sqacc / K.to(tl.float32) + norm_eps)  # [BLOCK_T]

    hc_scale = tl.load(hc_scale_ptr).to(tl.float32)  # scalar

    # ---- pass 2: weighted sum y[t, d] = sum_j pre[t,j] * x[t,j,d] ----
    num_d_tiles = tl.cdiv(HIDDEN, BLOCK_D)
    d_idx = tl.arange(0, BLOCK_D)

    for d0 in range(0, num_d_tiles):
        d_off = d0 * BLOCK_D + d_idx  # [BLOCK_D]
        d_mask = d_off < HIDDEN  # [BLOCK_D]

        yd = tl.zeros(
            (BLOCK_T, BLOCK_D), dtype=tl.float32
        )  # [BLOCK_T, BLOCK_D]

        for j in tl.static_range(0, HC_MULT):
            if j == 0:
                mixj = mix0
            elif j == 1:
                mixj = mix1
            elif j == 2:
                mixj = mix2
            elif j == 3:
                mixj = mix3
            elif j == 4:
                mixj = mix4
            elif j == 5:
                mixj = mix5
            elif j == 6:
                mixj = mix6
            elif j == 7:
                mixj = mix7
            pre_j = (
                tl.sigmoid(
                    mixj * rsqrt * hc_scale
                    + tl.load(hc_base_ptr + j).to(tl.float32)
                )
                + hc_eps
            )  # [BLOCK_T]

            xoff = (
                x_base_t[:, None] + j * X_STRIDE_M + d_off[None, :]
            )  # [BLOCK_T, BLOCK_D]
            xd = tl.load(
                x_ptr + xoff,
                mask=t_mask[:, None] & d_mask[None, :],
                other=0.0,
            ).to(
                tl.float32
            )  # [BLOCK_T, BLOCK_D]

            yd += pre_j[:, None] * xd  # [BLOCK_T, BLOCK_D]

        y_store_off = (
            offs_t.to(tl.int64)[:, None] * Y_STRIDE_T + d_off[None, :]
        )
        tl.store(
            y_ptr + y_store_off,
            yd.to(y_ptr.dtype.element_ty),
            mask=t_mask[:, None] & d_mask[None, :],
        )


# ---------------------------------------------------------------------------
# v8 fused kernel (hc_mult <= 4): per-lane delayed reduction + BLOCK_T
# textually-unrolled token batching, single launch for every T.
# ---------------------------------------------------------------------------


@triton.jit
def _hc_head_kernel_v8(
    x_ptr,
    hc_fn_ptr,
    hc_scale_ptr,
    hc_base_ptr,
    y_ptr,
    norm_eps,
    hc_eps,
    inv_k,  # host-computed 1/K (mean via multiply-reciprocal)
    T,
    HC_MULT: tl.constexpr,
    K: tl.constexpr,  # HC_MULT * HIDDEN (flattened row length)
    HIDDEN: tl.constexpr,
    BLOCK_T: tl.constexpr,  # tokens per program (1, 2 or 4)
    BLOCK_K: tl.constexpr,  # pass-1 flat-row tile
    BLOCK_D: tl.constexpr,  # pass-2 hidden-axis tile
):
    pid = tl.program_id(0)
    t0 = pid * BLOCK_T
    t0 = tl.minimum(t0, T - BLOCK_T)  # anchor tail block (host: T >= BLOCK_T)
    xb0 = t0.to(tl.int64) * K
    yb0 = t0.to(tl.int64) * HIDDEN
    if BLOCK_T > 1:
        t1 = t0 + 1
        xb1 = t1.to(tl.int64) * K
        yb1 = t1.to(tl.int64) * HIDDEN
    if BLOCK_T > 2:
        t2 = t0 + 2
        xb2 = t2.to(tl.int64) * K
        yb2 = t2.to(tl.int64) * HIDDEN
        t3 = t0 + 3
        xb3 = t3.to(tl.int64) * K
        yb3 = t3.to(tl.int64) * HIDDEN

    # ---- pass 1: per-lane 1D accumulators, one set per token ----
    sq0 = tl.zeros((BLOCK_K,), tl.float32)
    m00 = tl.zeros((BLOCK_K,), tl.float32)
    m10 = tl.zeros((BLOCK_K,), tl.float32)
    m20 = tl.zeros((BLOCK_K,), tl.float32)
    m30 = tl.zeros((BLOCK_K,), tl.float32)
    if BLOCK_T > 1:
        sq1 = tl.zeros((BLOCK_K,), tl.float32)
        m01 = tl.zeros((BLOCK_K,), tl.float32)
        m11 = tl.zeros((BLOCK_K,), tl.float32)
        m21 = tl.zeros((BLOCK_K,), tl.float32)
        m31 = tl.zeros((BLOCK_K,), tl.float32)
    if BLOCK_T > 2:
        sq2 = tl.zeros((BLOCK_K,), tl.float32)
        m02 = tl.zeros((BLOCK_K,), tl.float32)
        m12 = tl.zeros((BLOCK_K,), tl.float32)
        m22 = tl.zeros((BLOCK_K,), tl.float32)
        m32 = tl.zeros((BLOCK_K,), tl.float32)
        sq3 = tl.zeros((BLOCK_K,), tl.float32)
        m03 = tl.zeros((BLOCK_K,), tl.float32)
        m13 = tl.zeros((BLOCK_K,), tl.float32)
        m23 = tl.zeros((BLOCK_K,), tl.float32)
        m33 = tl.zeros((BLOCK_K,), tl.float32)

    for k0 in range(0, K, BLOCK_K):
        k_off = k0 + tl.arange(0, BLOCK_K)
        k_mask = k_off < K
        w0 = tl.load(hc_fn_ptr + 0 * K + k_off, mask=k_mask, other=0.0).to(
            tl.float32
        )
        if HC_MULT > 1:
            w1 = tl.load(hc_fn_ptr + 1 * K + k_off, mask=k_mask, other=0.0).to(
                tl.float32
            )
        if HC_MULT > 2:
            w2 = tl.load(hc_fn_ptr + 2 * K + k_off, mask=k_mask, other=0.0).to(
                tl.float32
            )
        if HC_MULT > 3:
            w3 = tl.load(hc_fn_ptr + 3 * K + k_off, mask=k_mask, other=0.0).to(
                tl.float32
            )

        xk0 = tl.load(x_ptr + xb0 + k_off, mask=k_mask, other=0.0).to(
            tl.float32
        )
        sq0 += xk0 * xk0
        m00 += xk0 * w0
        if HC_MULT > 1:
            m10 += xk0 * w1
        if HC_MULT > 2:
            m20 += xk0 * w2
        if HC_MULT > 3:
            m30 += xk0 * w3
        if BLOCK_T > 1:
            xk1 = tl.load(x_ptr + xb1 + k_off, mask=k_mask, other=0.0).to(
                tl.float32
            )
            sq1 += xk1 * xk1
            m01 += xk1 * w0
            if HC_MULT > 1:
                m11 += xk1 * w1
            if HC_MULT > 2:
                m21 += xk1 * w2
            if HC_MULT > 3:
                m31 += xk1 * w3
        if BLOCK_T > 2:
            xk2 = tl.load(x_ptr + xb2 + k_off, mask=k_mask, other=0.0).to(
                tl.float32
            )
            sq2 += xk2 * xk2
            m02 += xk2 * w0
            if HC_MULT > 1:
                m12 += xk2 * w1
            if HC_MULT > 2:
                m22 += xk2 * w2
            if HC_MULT > 3:
                m32 += xk2 * w3
            xk3 = tl.load(x_ptr + xb3 + k_off, mask=k_mask, other=0.0).to(
                tl.float32
            )
            sq3 += xk3 * xk3
            m03 += xk3 * w0
            if HC_MULT > 1:
                m13 += xk3 * w1
            if HC_MULT > 2:
                m23 += xk3 * w2
            if HC_MULT > 3:
                m33 += xk3 * w3

    # ---- gate: 5 trees per token, then sigmoid ----
    hc_scale = tl.load(hc_scale_ptr).to(tl.float32)
    r0 = tl.rsqrt(tl.sum(sq0, 0) * inv_k + norm_eps)
    pre00 = (
        tl.sigmoid(
            tl.sum(m00, 0) * r0 * hc_scale
            + tl.load(hc_base_ptr + 0).to(tl.float32)
        )
        + hc_eps
    )
    if HC_MULT > 1:
        pre10 = (
            tl.sigmoid(
                tl.sum(m10, 0) * r0 * hc_scale
                + tl.load(hc_base_ptr + 1).to(tl.float32)
            )
            + hc_eps
        )
    if HC_MULT > 2:
        pre20 = (
            tl.sigmoid(
                tl.sum(m20, 0) * r0 * hc_scale
                + tl.load(hc_base_ptr + 2).to(tl.float32)
            )
            + hc_eps
        )
    if HC_MULT > 3:
        pre30 = (
            tl.sigmoid(
                tl.sum(m30, 0) * r0 * hc_scale
                + tl.load(hc_base_ptr + 3).to(tl.float32)
            )
            + hc_eps
        )
    if BLOCK_T > 1:
        r1 = tl.rsqrt(tl.sum(sq1, 0) * inv_k + norm_eps)
        pre01 = (
            tl.sigmoid(
                tl.sum(m01, 0) * r1 * hc_scale
                + tl.load(hc_base_ptr + 0).to(tl.float32)
            )
            + hc_eps
        )
        if HC_MULT > 1:
            pre11 = (
                tl.sigmoid(
                    tl.sum(m11, 0) * r1 * hc_scale
                    + tl.load(hc_base_ptr + 1).to(tl.float32)
                )
                + hc_eps
            )
        if HC_MULT > 2:
            pre21 = (
                tl.sigmoid(
                    tl.sum(m21, 0) * r1 * hc_scale
                    + tl.load(hc_base_ptr + 2).to(tl.float32)
                )
                + hc_eps
            )
        if HC_MULT > 3:
            pre31 = (
                tl.sigmoid(
                    tl.sum(m31, 0) * r1 * hc_scale
                    + tl.load(hc_base_ptr + 3).to(tl.float32)
                )
                + hc_eps
            )
    if BLOCK_T > 2:
        r2 = tl.rsqrt(tl.sum(sq2, 0) * inv_k + norm_eps)
        pre02 = (
            tl.sigmoid(
                tl.sum(m02, 0) * r2 * hc_scale
                + tl.load(hc_base_ptr + 0).to(tl.float32)
            )
            + hc_eps
        )
        if HC_MULT > 1:
            pre12 = (
                tl.sigmoid(
                    tl.sum(m12, 0) * r2 * hc_scale
                    + tl.load(hc_base_ptr + 1).to(tl.float32)
                )
                + hc_eps
            )
        if HC_MULT > 2:
            pre22 = (
                tl.sigmoid(
                    tl.sum(m22, 0) * r2 * hc_scale
                    + tl.load(hc_base_ptr + 2).to(tl.float32)
                )
                + hc_eps
            )
        if HC_MULT > 3:
            pre32 = (
                tl.sigmoid(
                    tl.sum(m32, 0) * r2 * hc_scale
                    + tl.load(hc_base_ptr + 3).to(tl.float32)
                )
                + hc_eps
            )
        r3 = tl.rsqrt(tl.sum(sq3, 0) * inv_k + norm_eps)
        pre03 = (
            tl.sigmoid(
                tl.sum(m03, 0) * r3 * hc_scale
                + tl.load(hc_base_ptr + 0).to(tl.float32)
            )
            + hc_eps
        )
        if HC_MULT > 1:
            pre13 = (
                tl.sigmoid(
                    tl.sum(m13, 0) * r3 * hc_scale
                    + tl.load(hc_base_ptr + 1).to(tl.float32)
                )
                + hc_eps
            )
        if HC_MULT > 2:
            pre23 = (
                tl.sigmoid(
                    tl.sum(m23, 0) * r3 * hc_scale
                    + tl.load(hc_base_ptr + 2).to(tl.float32)
                )
                + hc_eps
            )
        if HC_MULT > 3:
            pre33 = (
                tl.sigmoid(
                    tl.sum(m33, 0) * r3 * hc_scale
                    + tl.load(hc_base_ptr + 3).to(tl.float32)
                )
                + hc_eps
            )

    # ---- pass 2: y[t, d] = sum_j pre_j * x[t, j, d] ----
    for d0 in range(0, HIDDEN, BLOCK_D):
        d_off = d0 + tl.arange(0, BLOCK_D)
        d_mask = d_off < HIDDEN
        xd00 = tl.load(
            x_ptr + xb0 + 0 * HIDDEN + d_off, mask=d_mask, other=0.0
        ).to(tl.float32)
        yd0 = pre00 * xd00
        if HC_MULT > 1:
            xd10 = tl.load(
                x_ptr + xb0 + 1 * HIDDEN + d_off, mask=d_mask, other=0.0
            ).to(tl.float32)
            yd0 += pre10 * xd10
        if HC_MULT > 2:
            xd20 = tl.load(
                x_ptr + xb0 + 2 * HIDDEN + d_off, mask=d_mask, other=0.0
            ).to(tl.float32)
            yd0 += pre20 * xd20
        if HC_MULT > 3:
            xd30 = tl.load(
                x_ptr + xb0 + 3 * HIDDEN + d_off, mask=d_mask, other=0.0
            ).to(tl.float32)
            yd0 += pre30 * xd30
        tl.store(
            y_ptr + yb0 + d_off,
            yd0.to(y_ptr.dtype.element_ty),
            mask=d_mask,
        )
        if BLOCK_T > 1:
            xd01 = tl.load(
                x_ptr + xb1 + 0 * HIDDEN + d_off, mask=d_mask, other=0.0
            ).to(tl.float32)
            yd1 = pre01 * xd01
            if HC_MULT > 1:
                xd11 = tl.load(
                    x_ptr + xb1 + 1 * HIDDEN + d_off, mask=d_mask, other=0.0
                ).to(tl.float32)
                yd1 += pre11 * xd11
            if HC_MULT > 2:
                xd21 = tl.load(
                    x_ptr + xb1 + 2 * HIDDEN + d_off, mask=d_mask, other=0.0
                ).to(tl.float32)
                yd1 += pre21 * xd21
            if HC_MULT > 3:
                xd31 = tl.load(
                    x_ptr + xb1 + 3 * HIDDEN + d_off, mask=d_mask, other=0.0
                ).to(tl.float32)
                yd1 += pre31 * xd31
            tl.store(
                y_ptr + yb1 + d_off,
                yd1.to(y_ptr.dtype.element_ty),
                mask=d_mask,
            )
        if BLOCK_T > 2:
            xd02 = tl.load(
                x_ptr + xb2 + 0 * HIDDEN + d_off, mask=d_mask, other=0.0
            ).to(tl.float32)
            yd2 = pre02 * xd02
            if HC_MULT > 1:
                xd12 = tl.load(
                    x_ptr + xb2 + 1 * HIDDEN + d_off, mask=d_mask, other=0.0
                ).to(tl.float32)
                yd2 += pre12 * xd12
            if HC_MULT > 2:
                xd22 = tl.load(
                    x_ptr + xb2 + 2 * HIDDEN + d_off, mask=d_mask, other=0.0
                ).to(tl.float32)
                yd2 += pre22 * xd22
            if HC_MULT > 3:
                xd32 = tl.load(
                    x_ptr + xb2 + 3 * HIDDEN + d_off, mask=d_mask, other=0.0
                ).to(tl.float32)
                yd2 += pre32 * xd32
            tl.store(
                y_ptr + yb2 + d_off,
                yd2.to(y_ptr.dtype.element_ty),
                mask=d_mask,
            )
            xd03 = tl.load(
                x_ptr + xb3 + 0 * HIDDEN + d_off, mask=d_mask, other=0.0
            ).to(tl.float32)
            yd3 = pre03 * xd03
            if HC_MULT > 1:
                xd13 = tl.load(
                    x_ptr + xb3 + 1 * HIDDEN + d_off, mask=d_mask, other=0.0
                ).to(tl.float32)
                yd3 += pre13 * xd13
            if HC_MULT > 2:
                xd23 = tl.load(
                    x_ptr + xb3 + 2 * HIDDEN + d_off, mask=d_mask, other=0.0
                ).to(tl.float32)
                yd3 += pre23 * xd23
            if HC_MULT > 3:
                xd33 = tl.load(
                    x_ptr + xb3 + 3 * HIDDEN + d_off, mask=d_mask, other=0.0
                ).to(tl.float32)
                yd3 += pre33 * xd33
            tl.store(
                y_ptr + yb3 + d_off,
                yd3.to(y_ptr.dtype.element_ty),
                mask=d_mask,
            )


def hc_head(x, hc_fn, hc_scale, hc_base, norm_eps, hc_eps):
    shape, dtype = x.size(), x.dtype
    T, hc_mult, hidden = shape
    K = hc_mult * hidden
    x = x.contiguous()
    hc_fn = hc_fn.contiguous()
    hc_scale = hc_scale.contiguous()
    hc_base = hc_base.contiguous()

    y = torch.empty((T, hidden), dtype=dtype, device=x.device)

    if T == 0:
        return y

    if hc_mult <= 4:
        # v8 single-launch fused kernel (per-lane delayed reduction +
        # textually-unrolled BLOCK_T token batching). Dispatch anchors:
        # GCU400/Libra, (T, 4, 7168) bf16, do_bench median -- T<=48 the
        # single-wave BT=1 chain wins (15.2 vs 19.4us @ T=2/4/8), T>=64 the
        # shared-hc_fn BT=2 amortisation wins (25.1 vs 29.5us @ T=64).
        # Small-T BK rescan (2026-11-12): on the BT=1 arm a fat BK=8192
        # k-tile wins while T<=24 (single-digit-SP regime, DSM 176128B in
        # the 2-programs/SP line): 14.5/13.8us vs 15.4/15.3us at T=1/16.
        # It collapses at T>=28 (24.2 vs 16.1us @ T=28), so the arm stops
        # at 24 and requires next_pow2(K) >= 8192 (smaller K keeps the
        # exact-fit BK).
        if T <= 24 and triton.next_power_of_2(K) >= 8192:
            block_k = 8192
        else:
            block_k = min(2048, triton.next_power_of_2(K))
        block_t = 1 if T <= 48 else 2
        block_t = min(block_t, T)
        block_d = min(4096, triton.next_power_of_2(hidden))
        _hc_head_kernel_v8[(triton.cdiv(T, block_t),)](
            x,
            hc_fn,
            hc_scale,
            hc_base,
            y,
            float(norm_eps),
            float(hc_eps),
            1.0 / K if K > 0 else 0.0,
            T,
            hc_mult,
            K,
            hidden,
            BLOCK_T=block_t,
            BLOCK_K=block_k,
            BLOCK_D=block_d,
            num_warps=8,
            num_stages=2,
        )
        return y

    # ---- legacy v7 paths below: hc_mult in 5..8 only (v8 handles <= 4) ----
    if T <= 256:
        # k-split partial path: kernel A computes per-k-chunk partials into an
        # fp32 scratch (every slot is overwritten by its owning program on
        # every launch -- KSPLIT is fixed below, not autotuned), kernel B
        # reduces, gates and does the weighted sum. The k-split parallelises
        # the hc_fn stream at tiny T (where the whole 458KB weight matrix is
        # otherwise walked by one program) and divides the per-token-block
        # hc_fn re-read traffic by KSPLIT at mid T.
        #
        # KSPLIT heuristic: K-chunks should leave enough token-block programs
        # to fill the SMs. t1 -> 16 chunks (16 programs); t128 -> 4; t256 ->
        # 2 (2*32/BLOCK_T = 64+ programs already).
        if T <= 4:
            ks = 16
        elif T <= 32:
            ks = 8
        elif T <= 128:
            ks = 4
        else:
            ks = 2
        part = torch.empty(
            (T, ks, hc_mult + 1), dtype=torch.float32, device=x.device
        )

        grid_a = lambda meta: (
            triton.cdiv(T, meta["BLOCK_T"]),
            ks,
        )
        _hc_head_partial_kernel[grid_a](
            x,
            hc_fn,
            part,
            T,
            hc_mult,
            K,
            x.stride(0),
            hc_fn.stride(0),
            part.stride(0),
            part.stride(1),
            ks,
        )

        grid_b = lambda meta: (
            triton.cdiv(T, meta["BLOCK_T"]),
            triton.cdiv(hidden, meta["BLOCK_D"]),
        )
        _hc_head_finish_kernel[grid_b](
            x,
            part,
            hc_scale,
            hc_base,
            y,
            float(norm_eps),
            float(hc_eps),
            T,
            hc_mult,
            hidden,
            K,
            x.stride(0),
            x.stride(1),
            y.stride(0),
            part.stride(0),
            part.stride(1),
            ks,
        )
        return y

    # large T: v5 fused single kernel (bandwidth-bound; a second launch would
    # only add overhead).
    grid = lambda meta: (triton.cdiv(T, meta["BLOCK_T"]),)
    _hc_head_kernel[grid](
        x,
        hc_fn,
        hc_scale,
        hc_base,
        y,
        float(norm_eps),
        float(hc_eps),
        T,
        hc_mult,
        hidden,
        K,
        x.stride(0),
        x.stride(1),
        y.stride(0),
        hc_fn.stride(0),
    )
    return y


__all__ = ["hc_head"]
