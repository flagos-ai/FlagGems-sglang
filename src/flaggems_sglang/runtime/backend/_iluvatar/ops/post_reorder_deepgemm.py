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

_TINY_BLOCK_H = 1024
_TINY_NUM_WARPS = 8
_DRAM_BLOCK_H = 128
_DRAM_NUM_WARPS = 1
_TINY_PROGRAM_LIMIT = 512


@triton.jit
def _post_reorder_deepgemm_kernel(
    down_ptr,
    out_ptr,
    src2dst_ptr,
    ids_ptr,
    w_ptr,
    topk,
    num_tokens,
    hidden_size,
    routed_scaling_factor,
    TOPK: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    pid_h = tl.program_id(0)
    pid_t = tl.program_id(1)

    offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = offs_h < hidden_size

    k_base = pid_t.to(tl.int64) * topk

    acc = tl.zeros((BLOCK_H,), dtype=tl.float32)
    for i in tl.static_range(TOPK):
        kid = tl.load(ids_ptr + k_base + i)
        d = tl.load(src2dst_ptr + k_base + i).to(tl.int64)
        wv = tl.load(w_ptr + k_base + i).to(tl.float32)
        wv = tl.where(kid >= 0, wv, 0.0)
        row = tl.load(
            down_ptr + d * hidden_size + offs_h, mask=mask_h, other=0.0
        ).to(tl.float32)
        acc += row * wv

    res = (acc * routed_scaling_factor).to(out_ptr.dtype.element_ty)
    out_offs = pid_t.to(tl.int64) * hidden_size + offs_h
    tl.store(out_ptr + out_offs, res, mask=mask_h)


def post_reorder_deepgemm(
    down_output,
    output,
    src2dst,
    topk_ids,
    topk_weights,
    topk,
    num_tokens,
    hidden_size,
    routed_scaling_factor,
):
    TOPK = triton.next_power_of_2(topk)

    if (
        triton.cdiv(hidden_size, _DRAM_BLOCK_H) * num_tokens
        < _TINY_PROGRAM_LIMIT
    ):
        block_h, num_warps = _TINY_BLOCK_H, _TINY_NUM_WARPS
    else:
        block_h, num_warps = _DRAM_BLOCK_H, _DRAM_NUM_WARPS

    grid = (triton.cdiv(hidden_size, block_h), num_tokens)
    _post_reorder_deepgemm_kernel[grid](
        down_output,
        output,
        src2dst,
        topk_ids,
        topk_weights,
        topk,
        num_tokens,
        hidden_size,
        routed_scaling_factor,
        TOPK=TOPK,
        BLOCK_H=block_h,
        num_warps=num_warps,
    )
    return output


__all__ = ["post_reorder_deepgemm"]
