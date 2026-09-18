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

Strategy (v7)
-------------
  * **T <= 256 -- k-split partial path (two kernels, no atomics).**
    The pass-1 work (sqsum + the ``hc_mult`` mix dots) is split along the
    flattened-row axis K across a 2D grid (token-blocks x KSPLIT):

      - Kernel A (``_hc_head_partial_kernel``) computes *partial* sqsum and
        mix accumulators for its k-chunk and stores them to a small fp32
        scratch buffer ``part[T, 32, HC_MULT+1]``. At T=1 this replaces the
        fused kernel's single serial program (which streamed the whole
        458KB hc_fn alone, pure latency) with KSPLIT parallel programs each
        touching a 1/KSPLIT slice. At T=128 it cuts the hc_fn L2 re-read
        traffic from ``(T/BLOCK_T) * 458KB`` to ``(T/BLOCK_T) * 458KB /
        KSPLIT`` (57MB -> ~7MB), since each token-block only walks its own
        k-chunk of each hc_fn row.

      - Kernel B (``_hc_head_finish_kernel``) reduces the KSPLIT partials
        (plain fp32 scalar loop over the live slots -- no atomics), applies
        rsqrt + sigmoid gate, and does the (token x hidden) weighted sum.
        KSPLIT is picked by a Python heuristic (NOT autotuned) so the
        scratch is sized exactly and every slot is overwritten each launch
        (no stale-slot hazard when Triton reruns the kernel).

  * **T > 256 -- fused single kernel** (v5): two serial in-kernel passes;
    bandwidth-bound at these sizes so a second launch would only add
    overhead. The v6 fp32 hc_fn pre-cast is gone: hc_fn is already fp32 at
    every call site, so that cast kernel was a wasted launch.

All math is fp32 (matching the reference's ``.float()``); the output is cast
back to ``x.dtype``. ``hc_mult <= 8`` is treated as a compile-time constant
with unrolled per-row scalar accumulators (no ``[T, HC_MULT, K]``
intermediate). Portability: plain loads/stores and fp32 arithmetic only --
no atomics, no vendor extensions, no dtype-specific hacks.
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
