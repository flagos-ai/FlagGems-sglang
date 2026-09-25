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

"""DeepEP MoE post-reorder epilogue -- Hygon DCU specialization.
"""

import triton
import triton.language as tl


@triton.autotune(
    configs=[
        # latency-oriented (few programs, wide blocks)
        triton.Config({"BLOCK_H": 2048}, num_warps=16, num_stages=1),
        triton.Config({"BLOCK_H": 4096}, num_warps=16, num_stages=1),
        triton.Config({"BLOCK_H": 4096}, num_warps=32, num_stages=1),
        triton.Config({"BLOCK_H": 8192}, num_warps=16, num_stages=1),
        # bandwidth-oriented
        triton.Config({"BLOCK_H": 512}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_H": 1024}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_H": 1024}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_H": 2048}, num_warps=8, num_stages=2),
    ],
    key=["hidden_size", "topk", "num_tokens"],
)
@triton.jit
def _post_reorder_kernel(
    down_ptr,  # [num_tokens * topk, hidden_size]
    out_ptr,  # [num_tokens, hidden_size]
    src2dst_ptr,  # [num_tokens, topk] int32
    weights_ptr,  # [num_tokens, topk] fp32
    num_tokens,
    hidden_size,
    topk: tl.constexpr,
    scaling,
    BLOCK_H: tl.constexpr,
):
    pid = tl.program_id(0)
    num_h = tl.cdiv(hidden_size, BLOCK_H)
    token = pid // num_h
    hb = pid % num_h

    offs_h = hb * BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = offs_h < hidden_size

    src_row = src2dst_ptr + token * topk
    w_row = weights_ptr + token * topk

    acc = tl.zeros([BLOCK_H], dtype=tl.float32)
    if topk % 4 == 0:
        # 4-wide unroll: batch the index/weight scalar loads before the row
        # gathers so more memory loads are in flight.
        for i in tl.static_range(0, topk, 4):
            d0 = tl.load(src_row + i + 0)
            d1 = tl.load(src_row + i + 1)
            d2 = tl.load(src_row + i + 2)
            d3 = tl.load(src_row + i + 3)
            # Match the reference: weight is rounded through the input dtype
            # before the fp32 multiply.
            w0 = (
                tl.load(w_row + i + 0)
                .to(down_ptr.dtype.element_ty)
                .to(tl.float32)
                * scaling
            )
            w1 = (
                tl.load(w_row + i + 1)
                .to(down_ptr.dtype.element_ty)
                .to(tl.float32)
                * scaling
            )
            w2 = (
                tl.load(w_row + i + 2)
                .to(down_ptr.dtype.element_ty)
                .to(tl.float32)
                * scaling
            )
            w3 = (
                tl.load(w_row + i + 3)
                .to(down_ptr.dtype.element_ty)
                .to(tl.float32)
                * scaling
            )
            v0 = tl.where(d0 >= 0, 1.0, 0.0)
            v1 = tl.where(d1 >= 0, 1.0, 0.0)
            v2 = tl.where(d2 >= 0, 1.0, 0.0)
            v3 = tl.where(d3 >= 0, 1.0, 0.0)
            d0s = tl.maximum(d0, 0)
            d1s = tl.maximum(d1, 0)
            d2s = tl.maximum(d2, 0)
            d3s = tl.maximum(d3, 0)
            r0 = tl.load(
                down_ptr + d0s * hidden_size + offs_h, mask=mask_h, other=0.0
            )
            r1 = tl.load(
                down_ptr + d1s * hidden_size + offs_h, mask=mask_h, other=0.0
            )
            r2 = tl.load(
                down_ptr + d2s * hidden_size + offs_h, mask=mask_h, other=0.0
            )
            r3 = tl.load(
                down_ptr + d3s * hidden_size + offs_h, mask=mask_h, other=0.0
            )
            acc += (
                r0.to(tl.float32) * (w0 * v0)
                + r1.to(tl.float32) * (w1 * v1)
                + r2.to(tl.float32) * (w2 * v2)
                + r3.to(tl.float32) * (w3 * v3)
            )
    else:
        for i in tl.static_range(topk):
            dst = tl.load(src_row + i)
            w = tl.load(w_row + i)
            w = w.to(down_ptr.dtype.element_ty).to(tl.float32) * scaling
            valid = tl.where(dst >= 0, 1.0, 0.0)
            dst_safe = tl.maximum(dst, 0)
            row = tl.load(
                down_ptr + dst_safe * hidden_size + offs_h,
                mask=mask_h,
                other=0.0,
            )
            acc += row.to(tl.float32) * (w * valid)

    tl.store(
        out_ptr + token * hidden_size + offs_h,
        acc.to(out_ptr.dtype.element_ty),
        mask=mask_h,
    )


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
    grid = lambda meta: (
        num_tokens * ((hidden_size + meta["BLOCK_H"] - 1) // meta["BLOCK_H"]),
    )
    _post_reorder_kernel[grid](
        down_output,
        output,
        src2dst,
        topk_weights,
        num_tokens,
        hidden_size,
        topk=topk,
        scaling=routed_scaling_factor,
    )
    return output


__all__ = ["deepep_post_reorder"]
