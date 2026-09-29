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

"""DeepEP MoE post-reorder epilogue -- MetaX specialization.
"""

import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_H": 64}, num_warps=1, num_stages=2),
        triton.Config({"BLOCK_H": 128}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK_H": 512}, num_warps=2, num_stages=2),
        # Stage-count variants of the two winning shapes: the gather loop's
        # pipelining depth was never swept before v8, and s=3 edges out s=2 at
        # the large (2048-token) regime (~188.0 vs ~188.4us, HBM-wall bound).
        triton.Config({"BLOCK_H": 512}, num_warps=2, num_stages=3),
        triton.Config({"BLOCK_H": 1024}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_H": 1024}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_H": 1024}, num_warps=4, num_stages=4),
    ],
    key=["hidden_size", "num_tokens"],
)
@triton.jit
def _post_reorder_kernel(
    down_ptr,  # [num_tokens * topk, hidden]  (any float dtype)
    out_ptr,  # [num_tokens, hidden]
    src2dst_ptr,  # [num_tokens, topk] int32, -1 = skip
    w_ptr,  # [num_tokens, topk] float32
    scaling,
    num_tokens,
    TOPK: tl.constexpr,
    TOPK_P2: tl.constexpr,
    HIDDEN: tl.constexpr,
    WIDE_INDEX: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    # Both operands are constexpr, so this resolves at compile time: the tail
    # mask disappears entirely whenever BLOCK_H divides the hidden dim for
    # this specialization (e.g. BLOCK_H=512 with hidden=7168).
    MASK_FREE: tl.constexpr = HIDDEN % BLOCK_H == 0

    pid_h = tl.program_id(0)
    pid_t = tl.program_id(1)
    offs = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)

    ki = tl.arange(0, TOPK_P2)
    # Vector load of the whole topk row. The mask only guards padded slots
    # (ki >= TOPK) on the last token, where an unmasked load would run past
    # the end of the tensor; these tiny TOPK_P2-wide loads are negligible
    # next to the hidden-block gathers below. -1 padding keeps invalid slots
    # safe through the +1-shift extraction.
    kmask = ki < TOPK
    dsts = tl.load(src2dst_ptr + pid_t * TOPK + ki, mask=kmask, other=-1)
    ws = tl.load(w_ptr + pid_t * TOPK + ki, mask=kmask, other=0.0)

    elem_ty = down_ptr.dtype.element_ty

    # Static unroll over the padded topk axis: one hidden-block of one source
    # row per iteration, fully predicated (no branches), so the working set
    # stays in L1 and the topk gathers pipeline without register blow-up.
    acc = tl.zeros([BLOCK_H], dtype=tl.float32)
    for i in tl.static_range(TOPK_P2):
        # Extract slot i from the vectors (+1 shift makes the padding sum to
        # exactly -1, i.e. an invalid slot, for every padded iteration).
        d = tl.sum(tl.where(ki == i, dsts + 1, 0), axis=0) - 1
        wv = tl.sum(tl.where(ki == i, ws, 0.0), axis=0)
        # Invalid slots (d == -1) get weight 0; their row index is clamped to
        # 0 so the unmasked gather stays in bounds and the loaded row is
        # multiplied by 0. Padded slots (ki >= TOPK) read element 0 of the
        # weight vector, which is forced to 0 the same way.
        wv = tl.where(d >= 0, wv, 0.0)
        wv = wv.to(elem_ty).to(tl.float32) * scaling
        # The reference routes the weight through down_output's dtype first.
        safe_d = tl.maximum(d, 0)
        if WIDE_INDEX:
            row_ptr = down_ptr + safe_d.to(tl.int64) * HIDDEN + offs
        else:
            # num_tokens*topk*hidden_size < 2^31: offsets fit int32.
            row_ptr = down_ptr + safe_d * HIDDEN + offs
        if MASK_FREE:
            row = tl.load(row_ptr, eviction_policy="evict_first").to(
                tl.float32
            )
        else:
            row = tl.load(
                row_ptr,
                mask=offs < HIDDEN,
                other=0.0,
                eviction_policy="evict_first",
            ).to(tl.float32)
        acc += row * wv

    if WIDE_INDEX:
        out_row = out_ptr + pid_t.to(tl.int64) * HIDDEN + offs
    else:
        out_row = out_ptr + pid_t * HIDDEN + offs
    if MASK_FREE:
        tl.store(out_row, acc.to(out_ptr.dtype.element_ty))
    else:
        tl.store(out_row, acc.to(out_ptr.dtype.element_ty), mask=offs < HIDDEN)


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
    # Row-offset arithmetic stays on the int32 fast path unless the down_output
    # tensor itself can exceed 2^31 elements.
    wide = down_output.numel() >= 2**31 or output.numel() >= 2**31

    def grid(meta):
        return (triton.cdiv(hidden_size, meta["BLOCK_H"]), num_tokens)

    _post_reorder_kernel[grid](
        down_output,
        output,
        src2dst,
        topk_weights,
        routed_scaling_factor,
        num_tokens,
        topk,
        triton.next_power_of_2(topk),
        hidden_size,
        wide,
    )
    return output


__all__ = ["deepep_post_reorder"]
