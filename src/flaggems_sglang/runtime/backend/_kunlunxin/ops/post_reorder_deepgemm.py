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

"""MoE DeepGEMM post-reorder epilogue in pure Triton."""

import triton
import triton.language as tl


@triton.jit
def _post_reorder_kernel(
    down_output_ptr,
    output_ptr,
    src2dst_ptr,
    topk_ids_ptr,
    topk_weights_ptr,
    stride_down_row,
    stride_out_row,
    stride_s2d_row,
    hidden_size,
    topk,
    routed_scaling_factor,
    BLOCK_N: tl.constexpr,
):
    pid_t = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < hidden_size

    s2d_row = src2dst_ptr + pid_t * stride_s2d_row
    ids_row = topk_ids_ptr + pid_t * topk
    w_row = topk_weights_ptr + pid_t * topk

    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for i in range(topk):
        eid = tl.load(ids_row + i)
        if eid >= 0:
            row = tl.load(s2d_row + i)
            w = tl.load(w_row + i)
            vals = tl.load(
                down_output_ptr + row.to(tl.int64) * stride_down_row + offs_n,
                mask=mask_n,
                other=0.0,
            ).to(tl.float32)
            acc += vals * w

    acc = acc * routed_scaling_factor
    out_ptrs = output_ptr + pid_t.to(tl.int64) * stride_out_row + offs_n
    tl.store(out_ptrs, acc.to(output_ptr.dtype.element_ty), mask=mask_n)


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
    grid = (num_tokens, triton.cdiv(hidden_size, 1024))
    _post_reorder_kernel[grid](
        down_output,
        output,
        src2dst,
        topk_ids,
        topk_weights,
        down_output.stride(0),
        output.stride(0),
        src2dst.stride(0),
        hidden_size,
        topk,
        routed_scaling_factor,
        BLOCK_N=1024,
        num_warps=4,
        num_stages=2,
    )
    return output


__all__ = ["post_reorder_deepgemm"]
