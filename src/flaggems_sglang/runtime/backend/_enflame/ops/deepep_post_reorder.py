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

"""DeepEP MoE post-reorder epilogue -- Enflame GCU specialization.
"""

import triton
import triton.language as tl

# Program count for the 1-token kernel grid clamp. Re-tuned in v9 for the
# interleaved body: 24 saturates the gather streams (32/48/96 measure worse at
# every hot T, both for the t1 and t2 bodies).
_T1_PROGRAMS = 24

# Program count for the 2-token-blocked kernel (topk == 8, T >= _T2_MIN_TOKENS).
# Swept 12/16/24/32/48 at T=2048/512: 24 wins — more resident programs
# serialize the gather streams on this backend; 12/16 starve it.
_T2_PROGRAMS = 24

# Minimum token count before the 2-token-blocked kernel beats the 1-token one.
# t1 wins at T=32/64/128 (42.6 vs 52.5 us at T=64 on the v9 bodies), ties at
# T=192 (84.1 vs 84.0 us), and t2 clearly wins from T=192 up (T=512: 180.6 vs
# 195.9 us) — the crossover sits at ~T=192, so t2 only takes over from here.
_T2_MIN_TOKENS = 192

# At very large token counts the round-robin pair walk loses to a contiguous
# block partition: each program owns a ceil-sized span of consecutive token
# PAIRS, so its output stores sweep a contiguous [2*pairs, H] stripe instead of
# striding across the whole output. Same-process A/B on the eval device
# (T=2048: 638 vs 654 us; T=4096: ~1264 vs ~1440 us) — at T <= 1536 the
# round-robin walk still wins or ties, so the block form only takes over from
# this threshold up.
_T2_BLK_MIN_TOKENS = 2048

# Upper bound on concurrently resident programs for the rare wide-hidden-size
# fallback kernel.
_MAX_PROGRAMS = 96

# Max column tile; next_pow2(hidden_size) is capped at this (8192 = 2 * H of
# the production H=7168 shape; wider column tiles beat splitting the columns
# across programs or chunks on this device). Hidden sizes above the cap take
# the chunked wide kernel.
_MAX_BLOCK_H = 8192


@triton.jit
def _post_reorder_kernel(
    down_ptr,  # [num_tokens * topk, hidden_size] expert outputs
    out_ptr,  # [num_tokens, hidden_size] merged output (written in place)
    src2dst_ptr,  # [num_tokens, topk] int32 destination rows (-1 = skip)
    w_ptr,  # [num_tokens, topk] routing weights
    num_tokens,
    scale,
    stride_down_m,
    stride_out_m,
    stride_s2d_t,
    stride_w_t,
    TOPK: tl.constexpr,
    BLOCK_H: tl.constexpr,
    H: tl.constexpr,
):
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    cols = tl.arange(0, BLOCK_H)
    cmask = cols < H
    kis = tl.arange(0, TOPK)
    for t in range(pid, num_tokens, nprog):
        # Token's (dst, weight) vectors, hoisted above the gathers. Invalid
        # slots (dst < 0) get weight 0 up front; the weight load stays fused
        # into this tl.where (splitting it out costs ~6% on this backend).
        dstv = tl.load(src2dst_ptr + t * stride_s2d_t + kis)
        wv = tl.where(
            dstv >= 0, tl.load(w_ptr + t * stride_w_t + kis) * scale, 0.0
        ).to(out_ptr.dtype.element_ty)
        # Clamp destinations so the wide row loads carry only the column mask;
        # invalid slots read row 0 and are nullified by their zero weight.
        dstc = tl.maximum(dstv, 0)
        if TOPK == 8:
            # Hot production topk: extract slot i's scalars and immediately
            # issue that slot's row load before extracting slot i+1, hiding
            # the extraction dependency chain behind gather memory latency.
            d0 = tl.sum(tl.where(kis == 0, dstc, 0))
            w0 = tl.sum(tl.where(kis == 0, wv, 0.0))
            r0 = tl.load(
                down_ptr + d0 * stride_down_m + cols, mask=cmask, other=0.0
            )
            d1 = tl.sum(tl.where(kis == 1, dstc, 0))
            w1 = tl.sum(tl.where(kis == 1, wv, 0.0))
            r1 = tl.load(
                down_ptr + d1 * stride_down_m + cols, mask=cmask, other=0.0
            )
            d2 = tl.sum(tl.where(kis == 2, dstc, 0))
            w2 = tl.sum(tl.where(kis == 2, wv, 0.0))
            r2 = tl.load(
                down_ptr + d2 * stride_down_m + cols, mask=cmask, other=0.0
            )
            d3 = tl.sum(tl.where(kis == 3, dstc, 0))
            w3 = tl.sum(tl.where(kis == 3, wv, 0.0))
            r3 = tl.load(
                down_ptr + d3 * stride_down_m + cols, mask=cmask, other=0.0
            )
            d4 = tl.sum(tl.where(kis == 4, dstc, 0))
            w4 = tl.sum(tl.where(kis == 4, wv, 0.0))
            r4 = tl.load(
                down_ptr + d4 * stride_down_m + cols, mask=cmask, other=0.0
            )
            d5 = tl.sum(tl.where(kis == 5, dstc, 0))
            w5 = tl.sum(tl.where(kis == 5, wv, 0.0))
            r5 = tl.load(
                down_ptr + d5 * stride_down_m + cols, mask=cmask, other=0.0
            )
            d6 = tl.sum(tl.where(kis == 6, dstc, 0))
            w6 = tl.sum(tl.where(kis == 6, wv, 0.0))
            r6 = tl.load(
                down_ptr + d6 * stride_down_m + cols, mask=cmask, other=0.0
            )
            d7 = tl.sum(tl.where(kis == 7, dstc, 0))
            w7 = tl.sum(tl.where(kis == 7, wv, 0.0))
            r7 = tl.load(
                down_ptr + d7 * stride_down_m + cols, mask=cmask, other=0.0
            )
            acc = ((r0 * w0 + r1 * w1) + (r2 * w2 + r3 * w3)) + (
                (r4 * w4 + r5 * w5) + (r6 * w6 + r7 * w7)
            )
        else:
            if TOPK == 1:
                d0 = tl.sum(tl.where(kis == 0, dstc, 0))
                w0 = tl.sum(tl.where(kis == 0, wv, 0.0))
                r0 = tl.load(
                    down_ptr + d0 * stride_down_m + cols, mask=cmask, other=0.0
                )
                acc = r0 * w0
            else:
                if TOPK == 2:
                    d0 = tl.sum(tl.where(kis == 0, dstc, 0))
                    w0 = tl.sum(tl.where(kis == 0, wv, 0.0))
                    r0 = tl.load(
                        down_ptr + d0 * stride_down_m + cols,
                        mask=cmask,
                        other=0.0,
                    )
                    d1 = tl.sum(tl.where(kis == 1, dstc, 0))
                    w1 = tl.sum(tl.where(kis == 1, wv, 0.0))
                    r1 = tl.load(
                        down_ptr + d1 * stride_down_m + cols,
                        mask=cmask,
                        other=0.0,
                    )
                    acc = r0 * w0 + r1 * w1
                else:
                    if TOPK == 4:
                        # Slot-by-slot interleave (same pattern as TOPK==8):
                        # extract slot i's scalars, immediately issue its row
                        # load, so the extraction chain hides behind gather
                        # latency instead of serializing ahead of all loads.
                        d0 = tl.sum(tl.where(kis == 0, dstc, 0))
                        w0 = tl.sum(tl.where(kis == 0, wv, 0.0))
                        r0 = tl.load(
                            down_ptr + d0 * stride_down_m + cols,
                            mask=cmask,
                            other=0.0,
                        )
                        d1 = tl.sum(tl.where(kis == 1, dstc, 0))
                        w1 = tl.sum(tl.where(kis == 1, wv, 0.0))
                        r1 = tl.load(
                            down_ptr + d1 * stride_down_m + cols,
                            mask=cmask,
                            other=0.0,
                        )
                        d2 = tl.sum(tl.where(kis == 2, dstc, 0))
                        w2 = tl.sum(tl.where(kis == 2, wv, 0.0))
                        r2 = tl.load(
                            down_ptr + d2 * stride_down_m + cols,
                            mask=cmask,
                            other=0.0,
                        )
                        d3 = tl.sum(tl.where(kis == 3, dstc, 0))
                        w3 = tl.sum(tl.where(kis == 3, wv, 0.0))
                        r3 = tl.load(
                            down_ptr + d3 * stride_down_m + cols,
                            mask=cmask,
                            other=0.0,
                        )
                        acc = (r0 * w0 + r1 * w1) + (r2 * w2 + r3 * w3)
                    else:
                        # Generic power-of-two topk > 8: serial combine; all
                        # lanes are real so the clamped-row trick holds.
                        acc = tl.zeros(
                            [BLOCK_H], dtype=out_ptr.dtype.element_ty
                        )
                        for i in tl.static_range(TOPK):
                            di = tl.sum(tl.where(kis == i, dstc, 0))
                            wi = tl.sum(tl.where(kis == i, wv, 0.0))
                            row = tl.load(
                                down_ptr + di * stride_down_m + cols,
                                mask=cmask,
                                other=0.0,
                            )
                            acc += row * wi
        tl.store(out_ptr + t * stride_out_m + cols, acc, mask=cmask)


@triton.jit
def _post_reorder_kernel_t2(
    down_ptr,
    out_ptr,
    src2dst_ptr,
    w_ptr,
    num_tokens,
    scale,
    stride_down_m,
    stride_out_m,
    stride_s2d_t,
    stride_w_t,
    BLOCK_H: tl.constexpr,
    H: tl.constexpr,
):
    """Hot path for topk == 8: two tokens per iteration.

    Slot-by-slot interleave: for each slot i, extract both tokens' dst/weight
    scalars and immediately issue both row loads before moving to slot i+1,
    hiding the scalar-extraction dependency chain behind gather memory
    latency. The second token of a trailing odd pair parks on token 0 (its
    weight is forced to 0 and its store is skipped). Same bf16 balanced-tree
    combine as the 1-token kernel.
    """
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    cols = tl.arange(0, BLOCK_H)
    cmask = cols < H
    kis = tl.arange(0, 8)
    for t0 in range(pid * 2, num_tokens, nprog * 2):
        ts1 = t0
        ts2 = t0 + 1
        v2 = ts2 < num_tokens
        ts2s = tl.where(v2, ts2, 0)
        dstv1 = tl.load(src2dst_ptr + ts1 * stride_s2d_t + kis)
        dstv2 = tl.load(src2dst_ptr + ts2s * stride_s2d_t + kis)
        dstc1 = tl.maximum(dstv1, 0)
        dstc2 = tl.maximum(dstv2, 0)
        wv1 = tl.where(
            dstv1 >= 0, tl.load(w_ptr + ts1 * stride_w_t + kis) * scale, 0.0
        ).to(out_ptr.dtype.element_ty)
        wv2 = tl.where(
            v2 & (dstv2 >= 0),
            tl.load(w_ptr + ts2s * stride_w_t + kis) * scale,
            0.0,
        ).to(out_ptr.dtype.element_ty)
        d10 = tl.sum(tl.where(kis == 0, dstc1, 0))
        w10 = tl.sum(tl.where(kis == 0, wv1, 0.0))
        d20 = tl.sum(tl.where(kis == 0, dstc2, 0))
        w20 = tl.sum(tl.where(kis == 0, wv2, 0.0))
        r10 = tl.load(
            down_ptr + d10 * stride_down_m + cols, mask=cmask, other=0.0
        )
        r20 = tl.load(
            down_ptr + d20 * stride_down_m + cols, mask=cmask, other=0.0
        )
        d11 = tl.sum(tl.where(kis == 1, dstc1, 0))
        w11 = tl.sum(tl.where(kis == 1, wv1, 0.0))
        d21 = tl.sum(tl.where(kis == 1, dstc2, 0))
        w21 = tl.sum(tl.where(kis == 1, wv2, 0.0))
        r11 = tl.load(
            down_ptr + d11 * stride_down_m + cols, mask=cmask, other=0.0
        )
        r21 = tl.load(
            down_ptr + d21 * stride_down_m + cols, mask=cmask, other=0.0
        )
        d12 = tl.sum(tl.where(kis == 2, dstc1, 0))
        w12 = tl.sum(tl.where(kis == 2, wv1, 0.0))
        d22 = tl.sum(tl.where(kis == 2, dstc2, 0))
        w22 = tl.sum(tl.where(kis == 2, wv2, 0.0))
        r12 = tl.load(
            down_ptr + d12 * stride_down_m + cols, mask=cmask, other=0.0
        )
        r22 = tl.load(
            down_ptr + d22 * stride_down_m + cols, mask=cmask, other=0.0
        )
        d13 = tl.sum(tl.where(kis == 3, dstc1, 0))
        w13 = tl.sum(tl.where(kis == 3, wv1, 0.0))
        d23 = tl.sum(tl.where(kis == 3, dstc2, 0))
        w23 = tl.sum(tl.where(kis == 3, wv2, 0.0))
        r13 = tl.load(
            down_ptr + d13 * stride_down_m + cols, mask=cmask, other=0.0
        )
        r23 = tl.load(
            down_ptr + d23 * stride_down_m + cols, mask=cmask, other=0.0
        )
        d14 = tl.sum(tl.where(kis == 4, dstc1, 0))
        w14 = tl.sum(tl.where(kis == 4, wv1, 0.0))
        d24 = tl.sum(tl.where(kis == 4, dstc2, 0))
        w24 = tl.sum(tl.where(kis == 4, wv2, 0.0))
        r14 = tl.load(
            down_ptr + d14 * stride_down_m + cols, mask=cmask, other=0.0
        )
        r24 = tl.load(
            down_ptr + d24 * stride_down_m + cols, mask=cmask, other=0.0
        )
        d15 = tl.sum(tl.where(kis == 5, dstc1, 0))
        w15 = tl.sum(tl.where(kis == 5, wv1, 0.0))
        d25 = tl.sum(tl.where(kis == 5, dstc2, 0))
        w25 = tl.sum(tl.where(kis == 5, wv2, 0.0))
        r15 = tl.load(
            down_ptr + d15 * stride_down_m + cols, mask=cmask, other=0.0
        )
        r25 = tl.load(
            down_ptr + d25 * stride_down_m + cols, mask=cmask, other=0.0
        )
        d16 = tl.sum(tl.where(kis == 6, dstc1, 0))
        w16 = tl.sum(tl.where(kis == 6, wv1, 0.0))
        d26 = tl.sum(tl.where(kis == 6, dstc2, 0))
        w26 = tl.sum(tl.where(kis == 6, wv2, 0.0))
        r16 = tl.load(
            down_ptr + d16 * stride_down_m + cols, mask=cmask, other=0.0
        )
        r26 = tl.load(
            down_ptr + d26 * stride_down_m + cols, mask=cmask, other=0.0
        )
        d17 = tl.sum(tl.where(kis == 7, dstc1, 0))
        w17 = tl.sum(tl.where(kis == 7, wv1, 0.0))
        d27 = tl.sum(tl.where(kis == 7, dstc2, 0))
        w27 = tl.sum(tl.where(kis == 7, wv2, 0.0))
        r17 = tl.load(
            down_ptr + d17 * stride_down_m + cols, mask=cmask, other=0.0
        )
        r27 = tl.load(
            down_ptr + d27 * stride_down_m + cols, mask=cmask, other=0.0
        )
        acc1 = ((r10 * w10 + r11 * w11) + (r12 * w12 + r13 * w13)) + (
            (r14 * w14 + r15 * w15) + (r16 * w16 + r17 * w17)
        )
        acc2 = ((r20 * w20 + r21 * w21) + (r22 * w22 + r23 * w23)) + (
            (r24 * w24 + r25 * w25) + (r26 * w26 + r27 * w27)
        )
        tl.store(out_ptr + ts1 * stride_out_m + cols, acc1, mask=cmask)
        if v2:
            tl.store(out_ptr + ts2 * stride_out_m + cols, acc2, mask=cmask)


@triton.jit
def _post_reorder_kernel_t2_blk(
    down_ptr,
    out_ptr,
    src2dst_ptr,
    w_ptr,
    num_tokens,
    scale,
    stride_down_m,
    stride_out_m,
    stride_s2d_t,
    stride_w_t,
    BLOCK_H: tl.constexpr,
    H: tl.constexpr,
):
    """Hot path for topk == 8 at very large token counts: block-partitioned.

    Identical body to ``_post_reorder_kernel_t2`` (slot-by-slot interleave of
    extraction and row loads, bf16 balanced-tree combine) but the pair walk is
    a contiguous block partition instead of round-robin: program ``p`` owns
    tokens ``[start, end)`` where each program gets
    ``ceil(num_tokens / (2 * nprog))`` pairs. The per-program output stores
    then sweep a contiguous [end - start, H] stripe rather than striding
    across the full output, which measures consistently faster once the stripe
    is large (T >= 2048) and slower or tied below it.
    """
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    cols = tl.arange(0, BLOCK_H)
    cmask = cols < H
    kis = tl.arange(0, 8)
    # ceil-partition so every token is covered for any (num_tokens, nprog)
    pairs_per = (num_tokens + 2 * nprog - 1) // (2 * nprog)
    start = pid * pairs_per * 2
    end = tl.minimum(start + pairs_per * 2, num_tokens)
    for t0 in range(start, end, 2):
        ts1 = t0
        ts2 = t0 + 1
        v2 = ts2 < num_tokens
        ts2s = tl.where(v2, ts2, 0)
        dstv1 = tl.load(src2dst_ptr + ts1 * stride_s2d_t + kis)
        dstv2 = tl.load(src2dst_ptr + ts2s * stride_s2d_t + kis)
        dstc1 = tl.maximum(dstv1, 0)
        dstc2 = tl.maximum(dstv2, 0)
        wv1 = tl.where(
            dstv1 >= 0, tl.load(w_ptr + ts1 * stride_w_t + kis) * scale, 0.0
        ).to(out_ptr.dtype.element_ty)
        wv2 = tl.where(
            v2 & (dstv2 >= 0),
            tl.load(w_ptr + ts2s * stride_w_t + kis) * scale,
            0.0,
        ).to(out_ptr.dtype.element_ty)
        d10 = tl.sum(tl.where(kis == 0, dstc1, 0))
        w10 = tl.sum(tl.where(kis == 0, wv1, 0.0))
        d20 = tl.sum(tl.where(kis == 0, dstc2, 0))
        w20 = tl.sum(tl.where(kis == 0, wv2, 0.0))
        r10 = tl.load(
            down_ptr + d10 * stride_down_m + cols, mask=cmask, other=0.0
        )
        r20 = tl.load(
            down_ptr + d20 * stride_down_m + cols, mask=cmask, other=0.0
        )
        d11 = tl.sum(tl.where(kis == 1, dstc1, 0))
        w11 = tl.sum(tl.where(kis == 1, wv1, 0.0))
        d21 = tl.sum(tl.where(kis == 1, dstc2, 0))
        w21 = tl.sum(tl.where(kis == 1, wv2, 0.0))
        r11 = tl.load(
            down_ptr + d11 * stride_down_m + cols, mask=cmask, other=0.0
        )
        r21 = tl.load(
            down_ptr + d21 * stride_down_m + cols, mask=cmask, other=0.0
        )
        d12 = tl.sum(tl.where(kis == 2, dstc1, 0))
        w12 = tl.sum(tl.where(kis == 2, wv1, 0.0))
        d22 = tl.sum(tl.where(kis == 2, dstc2, 0))
        w22 = tl.sum(tl.where(kis == 2, wv2, 0.0))
        r12 = tl.load(
            down_ptr + d12 * stride_down_m + cols, mask=cmask, other=0.0
        )
        r22 = tl.load(
            down_ptr + d22 * stride_down_m + cols, mask=cmask, other=0.0
        )
        d13 = tl.sum(tl.where(kis == 3, dstc1, 0))
        w13 = tl.sum(tl.where(kis == 3, wv1, 0.0))
        d23 = tl.sum(tl.where(kis == 3, dstc2, 0))
        w23 = tl.sum(tl.where(kis == 3, wv2, 0.0))
        r13 = tl.load(
            down_ptr + d13 * stride_down_m + cols, mask=cmask, other=0.0
        )
        r23 = tl.load(
            down_ptr + d23 * stride_down_m + cols, mask=cmask, other=0.0
        )
        d14 = tl.sum(tl.where(kis == 4, dstc1, 0))
        w14 = tl.sum(tl.where(kis == 4, wv1, 0.0))
        d24 = tl.sum(tl.where(kis == 4, dstc2, 0))
        w24 = tl.sum(tl.where(kis == 4, wv2, 0.0))
        r14 = tl.load(
            down_ptr + d14 * stride_down_m + cols, mask=cmask, other=0.0
        )
        r24 = tl.load(
            down_ptr + d24 * stride_down_m + cols, mask=cmask, other=0.0
        )
        d15 = tl.sum(tl.where(kis == 5, dstc1, 0))
        w15 = tl.sum(tl.where(kis == 5, wv1, 0.0))
        d25 = tl.sum(tl.where(kis == 5, dstc2, 0))
        w25 = tl.sum(tl.where(kis == 5, wv2, 0.0))
        r15 = tl.load(
            down_ptr + d15 * stride_down_m + cols, mask=cmask, other=0.0
        )
        r25 = tl.load(
            down_ptr + d25 * stride_down_m + cols, mask=cmask, other=0.0
        )
        d16 = tl.sum(tl.where(kis == 6, dstc1, 0))
        w16 = tl.sum(tl.where(kis == 6, wv1, 0.0))
        d26 = tl.sum(tl.where(kis == 6, dstc2, 0))
        w26 = tl.sum(tl.where(kis == 6, wv2, 0.0))
        r16 = tl.load(
            down_ptr + d16 * stride_down_m + cols, mask=cmask, other=0.0
        )
        r26 = tl.load(
            down_ptr + d26 * stride_down_m + cols, mask=cmask, other=0.0
        )
        d17 = tl.sum(tl.where(kis == 7, dstc1, 0))
        w17 = tl.sum(tl.where(kis == 7, wv1, 0.0))
        d27 = tl.sum(tl.where(kis == 7, dstc2, 0))
        w27 = tl.sum(tl.where(kis == 7, wv2, 0.0))
        r17 = tl.load(
            down_ptr + d17 * stride_down_m + cols, mask=cmask, other=0.0
        )
        r27 = tl.load(
            down_ptr + d27 * stride_down_m + cols, mask=cmask, other=0.0
        )
        acc1 = ((r10 * w10 + r11 * w11) + (r12 * w12 + r13 * w13)) + (
            (r14 * w14 + r15 * w15) + (r16 * w16 + r17 * w17)
        )
        acc2 = ((r20 * w20 + r21 * w21) + (r22 * w22 + r23 * w23)) + (
            (r24 * w24 + r25 * w25) + (r26 * w26 + r27 * w27)
        )
        tl.store(out_ptr + ts1 * stride_out_m + cols, acc1, mask=cmask)
        if v2:
            tl.store(out_ptr + ts2 * stride_out_m + cols, acc2, mask=cmask)


@triton.jit
def _post_reorder_kernel_padded(
    down_ptr,
    out_ptr,
    src2dst_ptr,
    w_ptr,
    num_tokens,
    topk,
    scale,
    stride_down_m,
    stride_out_m,
    stride_s2d_t,
    stride_w_t,
    TOPK: tl.constexpr,
    BLOCK_H: tl.constexpr,
    H: tl.constexpr,
):
    """Generic fallback for topk values that are not powers of two.

    Lanes >= topk park on dst = -1 / weight 0; each slot's row load keeps the
    ``dst >= 0`` predicate so a padded lane reads nothing (the clamped-row
    trick cannot fold a padded lane through a shared row 0 read).
    """
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    cols = tl.arange(0, BLOCK_H)
    cmask = cols < H
    kis = tl.arange(0, TOPK)
    for t in range(pid, num_tokens, nprog):
        dstv = tl.load(
            src2dst_ptr + t * stride_s2d_t + kis, mask=kis < topk, other=-1
        )
        wv = tl.where(
            dstv >= 0,
            tl.load(w_ptr + t * stride_w_t + kis, mask=kis < topk, other=0.0)
            * scale,
            0.0,
        ).to(out_ptr.dtype.element_ty)
        acc = tl.zeros([BLOCK_H], dtype=out_ptr.dtype.element_ty)
        for i in tl.static_range(TOPK):
            di = tl.sum(tl.where(kis == i, dstv, 0))
            wi = tl.sum(tl.where(kis == i, wv, 0.0))
            row = tl.load(
                down_ptr + di * stride_down_m + cols,
                mask=cmask & (di >= 0),
                other=0.0,
            )
            acc += row * wi
        tl.store(out_ptr + t * stride_out_m + cols, acc, mask=cmask)


@triton.jit
def _post_reorder_kernel_wide(
    down_ptr,
    out_ptr,
    src2dst_ptr,
    w_ptr,
    hidden_size,
    num_tokens,
    scale,
    stride_down_m,
    stride_out_m,
    stride_s2d_t,
    stride_w_t,
    TOPK: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    """Rare fallback for hidden_size > _MAX_BLOCK_H: walk the columns in
    runtime-masked BLOCK_H-sized chunks with a serial combine."""
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    kis = tl.arange(0, TOPK)
    for t in range(pid, num_tokens, nprog):
        dstv = tl.load(src2dst_ptr + t * stride_s2d_t + kis)
        wv = tl.where(
            dstv >= 0, tl.load(w_ptr + t * stride_w_t + kis) * scale, 0.0
        ).to(out_ptr.dtype.element_ty)
        dstc = tl.maximum(dstv, 0)
        kis2 = tl.arange(0, TOPK)
        for c0 in range(0, hidden_size, BLOCK_H):
            cols = c0 + tl.arange(0, BLOCK_H)
            cmask = cols < hidden_size
            acc = tl.zeros([BLOCK_H], dtype=out_ptr.dtype.element_ty)
            for i in tl.static_range(TOPK):
                di = tl.sum(tl.where(kis2 == i, dstc, 0))
                wi = tl.sum(tl.where(kis2 == i, wv, 0.0))
                acc += (
                    tl.load(
                        down_ptr + di * stride_down_m + cols,
                        mask=cmask,
                        other=0.0,
                    )
                    * wi
                )
            tl.store(out_ptr + t * stride_out_m + cols, acc, mask=cmask)


def deepep_post_reorder(
    down_output,
    output,
    src2dst,
    topk_ids,
    topk_weights,
    topk,
    hidden_size,
    routed_scaling_factor,
):
    num_tokens = output.shape[0]

    topk_pad = max(triton.next_power_of_2(topk), 1)
    scale = float(routed_scaling_factor)
    strides = (
        down_output.stride(0),
        output.stride(0),
        src2dst.stride(0),
        topk_weights.stride(0),
    )

    if hidden_size > _MAX_BLOCK_H:
        # Rare: hidden wider than the max tile — chunk the columns.
        grid = (min(num_tokens, _MAX_PROGRAMS),)
        _post_reorder_kernel_wide[grid](
            down_output,
            output,
            src2dst,
            topk_weights,
            hidden_size,
            num_tokens,
            scale,
            *strides,
            TOPK=topk_pad,
            BLOCK_H=_MAX_BLOCK_H,
            num_warps=1,
            num_stages=1,
        )
        return output

    block_h = min(_MAX_BLOCK_H, triton.next_power_of_2(hidden_size))

    if topk == 8 and num_tokens >= _T2_MIN_TOKENS:
        # Hot production shape: 2-token blocked body. Round-robin pair walk up
        # to _T2_BLK_MIN_TOKENS; contiguous block partition beyond it (its
        # per-program output stripe gets long enough to win there).
        nprog = _T2_PROGRAMS
        grid = (min((num_tokens + 1) // 2, nprog),)
        if num_tokens >= _T2_BLK_MIN_TOKENS:
            _post_reorder_kernel_t2_blk[grid](
                down_output,
                output,
                src2dst,
                topk_weights,
                num_tokens,
                scale,
                *strides,
                BLOCK_H=block_h,
                H=hidden_size,
                num_warps=1,
                num_stages=1,
            )
            return output
        _post_reorder_kernel_t2[grid](
            down_output,
            output,
            src2dst,
            topk_weights,
            num_tokens,
            scale,
            *strides,
            BLOCK_H=block_h,
            H=hidden_size,
            num_warps=1,
            num_stages=1,
        )
        return output

    grid = (min(num_tokens, _T1_PROGRAMS),)
    if topk_pad == topk:
        _post_reorder_kernel[grid](
            down_output,
            output,
            src2dst,
            topk_weights,
            num_tokens,
            scale,
            *strides,
            TOPK=topk_pad,
            BLOCK_H=block_h,
            H=hidden_size,
            num_warps=1,
            num_stages=1,
        )
    else:
        # Rare path: pad topk to the next power of two; padded lanes park on
        # dst = -1 and contribute nothing.
        _post_reorder_kernel_padded[grid](
            down_output,
            output,
            src2dst,
            topk_weights,
            num_tokens,
            topk,
            scale,
            *strides,
            TOPK=topk_pad,
            BLOCK_H=block_h,
            H=hidden_size,
            num_warps=1,
            num_stages=1,
        )
    return output


__all__ = ["deepep_post_reorder"]
