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


@triton.jit
def _post_reorder_unrolled_kernel(
    down_output_ptr,
    output_ptr,
    src2dst_ptr,
    topk_ids_ptr,
    topk_weights_ptr,
    stride_down_row,
    stride_out_row,
    stride_s2d_row,
    stride_ids_row,
    stride_w_row,
    hidden_size,
    num_tokens,
    routed_scaling_factor,
    TOPK: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_t = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_t = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_t = offs_t < num_tokens
    mask_n = offs_n < hidden_size
    mask2d = mask_t[:, None] & mask_n[None, :]

    acc = tl.zeros((BLOCK_T, BLOCK_N), dtype=tl.float32)
    for k in tl.static_range(TOPK):
        dst = tl.load(
            src2dst_ptr + offs_t * stride_s2d_row + k, mask=mask_t, other=0
        ).to(tl.int64)
        eid = tl.load(
            topk_ids_ptr + offs_t * stride_ids_row + k, mask=mask_t, other=-1
        )
        w = tl.load(
            topk_weights_ptr + offs_t * stride_w_row + k,
            mask=mask_t,
            other=0.0,
        )
        w = tl.where(eid >= 0, w, 0.0)

        vals = tl.load(
            down_output_ptr + dst[:, None] * stride_down_row + offs_n[None, :],
            mask=mask2d,
            other=0.0,
        ).to(tl.float32)
        acc += vals * w[:, None]

    acc = acc * routed_scaling_factor
    out_ptrs = (
        output_ptr
        + offs_t.to(tl.int64)[:, None] * stride_out_row
        + offs_n[None, :]
    )
    tl.store(out_ptrs, acc.to(output_ptr.dtype.element_ty), mask=mask2d)


@triton.jit
def _post_reorder_loop_kernel(
    down_output_ptr,
    output_ptr,
    src2dst_ptr,
    topk_ids_ptr,
    topk_weights_ptr,
    stride_down_row,
    stride_out_row,
    stride_s2d_row,
    stride_ids_row,
    stride_w_row,
    hidden_size,
    num_tokens,
    topk,
    routed_scaling_factor,
    BLOCK_T: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_t = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_t = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_t = offs_t < num_tokens
    mask_n = offs_n < hidden_size
    mask2d = mask_t[:, None] & mask_n[None, :]

    acc = tl.zeros((BLOCK_T, BLOCK_N), dtype=tl.float32)
    for k in range(topk):
        dst = tl.load(
            src2dst_ptr + offs_t * stride_s2d_row + k, mask=mask_t, other=0
        ).to(tl.int64)
        eid = tl.load(
            topk_ids_ptr + offs_t * stride_ids_row + k, mask=mask_t, other=-1
        )
        w = tl.load(
            topk_weights_ptr + offs_t * stride_w_row + k,
            mask=mask_t,
            other=0.0,
        )
        w = tl.where(eid >= 0, w, 0.0)

        vals = tl.load(
            down_output_ptr + dst[:, None] * stride_down_row + offs_n[None, :],
            mask=mask2d,
            other=0.0,
        ).to(tl.float32)
        acc += vals * w[:, None]

    acc = acc * routed_scaling_factor
    out_ptrs = (
        output_ptr
        + offs_t.to(tl.int64)[:, None] * stride_out_row
        + offs_n[None, :]
    )
    tl.store(out_ptrs, acc.to(output_ptr.dtype.element_ty), mask=mask2d)


_UNROLL_MAX_TOKENS = 128
_UNROLL_MAX_TOPK = 8
_UNROLL_BLOCK_T = 2
_UNROLL_BLOCK_N = 1024
_UNROLL_NUM_WARPS = 4
_LOOP_BLOCK_T = 2
_LOOP_BLOCK_N = 1024
_LOOP_NUM_WARPS = 4


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
    stride_down_row = down_output.stride(0)
    stride_out_row = output.stride(0)
    stride_s2d_row = src2dst.stride(0)
    stride_ids_row = topk_ids.stride(0)
    stride_w_row = topk_weights.stride(0)

    if topk <= _UNROLL_MAX_TOPK and num_tokens <= _UNROLL_MAX_TOKENS:
        grid = (
            triton.cdiv(num_tokens, _UNROLL_BLOCK_T),
            triton.cdiv(hidden_size, _UNROLL_BLOCK_N),
        )
        _post_reorder_unrolled_kernel[grid](
            down_output,
            output,
            src2dst,
            topk_ids,
            topk_weights,
            stride_down_row,
            stride_out_row,
            stride_s2d_row,
            stride_ids_row,
            stride_w_row,
            hidden_size,
            num_tokens,
            routed_scaling_factor,
            TOPK=topk,
            BLOCK_T=_UNROLL_BLOCK_T,
            BLOCK_N=_UNROLL_BLOCK_N,
            num_warps=_UNROLL_NUM_WARPS,
            num_stages=2,
        )
    else:
        grid = (
            triton.cdiv(num_tokens, _LOOP_BLOCK_T),
            triton.cdiv(hidden_size, _LOOP_BLOCK_N),
        )
        _post_reorder_loop_kernel[grid](
            down_output,
            output,
            src2dst,
            topk_ids,
            topk_weights,
            stride_down_row,
            stride_out_row,
            stride_s2d_row,
            stride_ids_row,
            stride_w_row,
            hidden_size,
            num_tokens,
            topk,
            routed_scaling_factor,
            BLOCK_T=_LOOP_BLOCK_T,
            BLOCK_N=_LOOP_BLOCK_N,
            num_warps=_LOOP_NUM_WARPS,
            num_stages=2,
        )
    return output


__all__ = ["post_reorder_deepgemm"]
