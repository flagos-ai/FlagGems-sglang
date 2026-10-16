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

import triton
import triton.language as tl

# Below this many output elements, 32-bit offsets are safe (offsets are
# ``dst * hidden_size + col < n * hidden_size``).
_INT32_MAX_ELEMS = 1 << 31

# Launch-table bounds (deterministic host-side heuristic, no autotune):
# tokens at or below this run in the launch-bound regime -> narrow tiles.
_SMALL_T = 8


@triton.jit
def _pre_reorder_scatter_kernel(
    input_ptr,  # [num_tokens, hidden] token rows to broadcast
    out_ptr,  # [n, hidden] gateup_input, written in place
    dst_ptr,  # [num_tokens, topk] int32 permutation of [0, n)
    ids_ptr,  # [num_tokens, topk] int32 expert ids
    scale_ptr,  # [1] fp32 (or any tensor when HAS_SCALE is False)
    HAS_SCALE: tl.constexpr,
    HIDDEN: tl.constexpr,  # hidden_size as compile-time constant
    NLE: tl.constexpr,  # num_local_experts as compile-time constant
    TOPK: tl.constexpr,
    KP2: tl.constexpr,  # next_power_of_2(TOPK) for the slot vector
    POW2_TOPK: tl.constexpr,  # TOPK == KP2: slot loads need no k mask
    INT64: tl.constexpr,  # use 64-bit row offsets (huge outputs only)
    BLOCK_H: tl.constexpr,
):
    EVEN_H: tl.constexpr = HIDDEN % BLOCK_H == 0  # column masks elided

    # Token-major grid: token id on the minor axis so consecutive CTAs share
    # the same token's index loads and scaled row through L2.
    t = tl.program_id(1)

    # All topk slots of this token in one vectorized load each.
    k = tl.arange(0, KP2)
    if POW2_TOPK:
        ids_v = tl.load(ids_ptr + t * TOPK + k)
        dst_v = tl.load(dst_ptr + t * TOPK + k)
    else:
        kmask = k < TOPK
        ids_v = tl.load(ids_ptr + t * TOPK + k, mask=kmask, other=NLE)
        dst_v = tl.load(dst_ptr + t * TOPK + k, mask=kmask, other=0)
    if INT64:
        dst_v = dst_v.to(tl.int64)
    # Slots routed off this EP rank keep their original gateup row: the
    # store is simply predicated off for them (no gather, no tl.where).
    live = ids_v != NLE

    hb = tl.program_id(0)
    cols = hb * BLOCK_H + tl.arange(0, BLOCK_H)
    # The column tile is a contiguous, BLOCK_H-aligned slab of the row: tell
    # the compiler so loads/stores vectorize into wide ops.
    cols = tl.max_contiguous(tl.multiple_of(cols, BLOCK_H), BLOCK_H)
    if INT64:
        off = dst_v[:, None] * HIDDEN + cols.to(tl.int64)[None, :]
    else:
        off = dst_v[:, None] * HIDDEN + cols[None, :]

    if EVEN_H:
        # Token row, dequantized once and reused for all of the token's slots.
        x = tl.load(input_ptr + t * HIDDEN + cols)
        x = x.to(tl.float32)
        if HAS_SCALE:
            x = x * (1.0 / tl.load(scale_ptr))
        x = x.to(out_ptr.dtype.element_ty)
        tl.store(out_ptr + off, x[None, :], mask=live[:, None])
    else:
        cmask = cols < HIDDEN
        x = tl.load(input_ptr + t * HIDDEN + cols, mask=cmask, other=0.0)
        x = x.to(tl.float32)
        if HAS_SCALE:
            x = x * (1.0 / tl.load(scale_ptr))
        x = x.to(out_ptr.dtype.element_ty)
        tl.store(
            out_ptr + off, x[None, :], mask=live[:, None] & cmask[None, :]
        )


def pre_reorder_cutlass(
    input,
    gateup_input,
    src2dst,
    topk_ids,
    a1_scales,
    num_local_experts,
    topk,
    num_tokens,
    hidden_size,
):
    out = gateup_input
    if topk_ids.numel() == 0:
        return out

    # The kernel indexes with flat row-major offsets; make that assumption
    # explicit (host-side, free for the already-contiguous benchmark cases).
    if not input.is_contiguous():
        input = input.contiguous()
    if not out.is_contiguous():
        out = out.contiguous()
    if not src2dst.is_contiguous():
        src2dst = src2dst.contiguous()
    if not topk_ids.is_contiguous():
        topk_ids = topk_ids.contiguous()

    if num_tokens <= _SMALL_T:
        block_h, num_warps = 128, 1
    else:
        block_h, num_warps = 512, 2
    if hidden_size < block_h:
        block_h = max(triton.next_power_of_2(hidden_size), 16)
        num_warps = min(num_warps, max(block_h * 2 // 128, 1))

    grid = (triton.cdiv(hidden_size, block_h), num_tokens)
    _pre_reorder_scatter_kernel[grid](
        input,
        out,
        src2dst,
        topk_ids,
        a1_scales if a1_scales is not None else input,
        HAS_SCALE=a1_scales is not None,
        INT64=out.numel() >= _INT32_MAX_ELEMS,
        HIDDEN=hidden_size,
        NLE=num_local_experts,
        TOPK=topk,
        KP2=triton.next_power_of_2(topk),
        POW2_TOPK=(topk == triton.next_power_of_2(topk)),
        BLOCK_H=block_h,
        num_warps=num_warps,
    )
    return out


__all__ = ["pre_reorder_cutlass"]
