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

_GATHER_MIN_TOKENS = 256


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_H": 256}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK_H": 512}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_H": 512}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_H": 1024}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_H": 1024}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_H": 2048}, num_warps=8, num_stages=2),
    ],
    key=["num_tokens", "hidden_size"],
)
@triton.jit
def _post_reorder_deepgemm_kernel(
    down_ptr,
    out_ptr,
    src2dst_ptr,
    ids_ptr,
    w_ptr,
    scale,
    num_tokens,
    hidden_size,
    down_row_stride,
    src2dst_row_stride,
    ids_row_stride,
    w_row_stride,
    TOPK: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    pid_t = tl.program_id(0)
    pid_h = tl.program_id(1)
    offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = offs_h < hidden_size

    src_base = src2dst_ptr + pid_t * src2dst_row_stride
    ids_base = ids_ptr + pid_t * ids_row_stride
    w_base = w_ptr + pid_t * w_row_stride

    acc = tl.zeros([BLOCK_H], dtype=tl.float32)
    for i in tl.static_range(TOPK):
        dst = tl.load(src_base + i).to(tl.int64)
        eid = tl.load(ids_base + i)
        w = tl.load(w_base + i)
        valid = tl.where(eid >= 0, w, 0.0)
        dst = tl.where(eid >= 0, dst, 0)
        row = tl.load(
            down_ptr + dst * down_row_stride + offs_h,
            mask=mask_h,
            other=0.0,
        ).to(tl.float32)
        acc += row * valid

    acc = acc * scale
    tl.store(
        out_ptr + pid_t.to(tl.int64) * hidden_size + offs_h,
        acc.to(out_ptr.dtype.element_ty),
        mask=mask_h,
    )


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_H": 256}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK_H": 512}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_H": 512}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_H": 1024}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_H": 1024}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_H": 2048}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_H": 2048}, num_warps=8, num_stages=2),
    ],
    key=["num_tokens", "hidden_size"],
)
@triton.jit
def _post_reorder_deepgemm_gather_kernel(
    down_ptr,
    out_ptr,
    src2dst_ptr,
    ids_ptr,
    w_ptr,
    scale,
    num_tokens,
    hidden_size,
    down_row_stride,
    src2dst_row_stride,
    ids_row_stride,
    w_row_stride,
    TOPK: tl.constexpr,
    K_PAD: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    pid_t = tl.program_id(0)
    pid_h = tl.program_id(1)
    offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = offs_h < hidden_size
    offs_k = tl.arange(0, K_PAD)
    mask_k = offs_k < TOPK

    src_base = src2dst_ptr + pid_t * src2dst_row_stride
    ids_base = ids_ptr + pid_t * ids_row_stride
    w_base = w_ptr + pid_t * w_row_stride
    dst = tl.load(src_base + offs_k, mask=mask_k, other=0)
    ids = tl.load(ids_base + offs_k, mask=mask_k, other=-1)
    w = tl.load(w_base + offs_k, mask=mask_k, other=0.0)

    valid = ids >= 0
    w = tl.where(valid, w, 0.0)

    rows = tl.load(
        down_ptr
        + dst[:, None].to(tl.int64) * down_row_stride
        + offs_h[None, :],
        mask=valid[:, None] & mask_h[None, :],
        other=0.0,
    ).to(tl.float32)

    acc = tl.sum(rows * w[:, None], axis=0) * scale
    tl.store(
        out_ptr + pid_t.to(tl.int64) * hidden_size + offs_h,
        acc.to(out_ptr.dtype.element_ty),
        mask=mask_h,
    )


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
    if num_tokens == 0:
        return output

    def grid(meta):
        return (num_tokens, triton.cdiv(hidden_size, meta["BLOCK_H"]))

    common = dict(
        num_tokens=num_tokens,
        hidden_size=hidden_size,
        down_row_stride=down_output.stride(0),
        src2dst_row_stride=src2dst.stride(0),
        ids_row_stride=topk_ids.stride(0),
        w_row_stride=topk_weights.stride(0),
    )
    if num_tokens >= _GATHER_MIN_TOKENS:
        _post_reorder_deepgemm_gather_kernel[grid](
            down_output,
            output,
            src2dst,
            topk_ids,
            topk_weights,
            float(routed_scaling_factor),
            TOPK=topk,
            K_PAD=triton.next_power_of_2(topk),
            **common,
        )
    else:
        _post_reorder_deepgemm_kernel[grid](
            down_output,
            output,
            src2dst,
            topk_ids,
            topk_weights,
            float(routed_scaling_factor),
            TOPK=topk,
            **common,
        )
    return output


__all__ = ["post_reorder_deepgemm"]
