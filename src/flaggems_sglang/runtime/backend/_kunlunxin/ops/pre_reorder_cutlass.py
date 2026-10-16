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

"""Task102 KunlunXin: small-batch route owners, large-batch token."""

import triton
import triton.language as tl


@triton.jit
def _pre_reorder_xpu_route_owner(
    input_ptr,
    output_ptr,
    src2dst_ptr,
    topk_ids_ptr,
    scale_ptr,
    stride_i_t: tl.constexpr,
    stride_i_d: tl.constexpr,
    stride_o_t: tl.constexpr,
    stride_o_d: tl.constexpr,
    stride_map_t: tl.constexpr,
    stride_map_k: tl.constexpr,
    stride_id_t: tl.constexpr,
    stride_id_k: tl.constexpr,
    num_local_experts: tl.constexpr,
    topk: tl.constexpr,
    hidden_size: tl.constexpr,
    HAS_SCALE: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    route = tl.program_id(0)
    token = route // topk
    slot = route - token * topk
    cols = tl.arange(0, BLOCK_D)
    mask = cols < hidden_size
    expert = tl.load(topk_ids_ptr + token * stride_id_t + slot * stride_id_k)
    if expert != num_local_experts:
        destination = tl.load(
            src2dst_ptr + token * stride_map_t + slot * stride_map_k
        ).to(tl.int64)
        values = tl.load(
            input_ptr + token * stride_i_t + cols * stride_i_d,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        if HAS_SCALE:
            values *= 1.0 / tl.load(scale_ptr).to(tl.float32)
        tl.store(
            output_ptr + destination * stride_o_t + cols * stride_o_d,
            values.to(output_ptr.dtype.element_ty),
            mask=mask,
        )


@triton.jit
def _pre_reorder_xpu_token_owner(
    input_ptr,
    output_ptr,
    src2dst_ptr,
    topk_ids_ptr,
    scale_ptr,
    stride_i_t: tl.constexpr,
    stride_i_d: tl.constexpr,
    stride_o_t: tl.constexpr,
    stride_o_d: tl.constexpr,
    stride_map_t: tl.constexpr,
    stride_map_k: tl.constexpr,
    stride_id_t: tl.constexpr,
    stride_id_k: tl.constexpr,
    num_local_experts: tl.constexpr,
    topk: tl.constexpr,
    hidden_size: tl.constexpr,
    HAS_SCALE: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    token = tl.program_id(0)
    cols = tl.arange(0, BLOCK_D)
    mask = cols < hidden_size
    values = tl.load(
        input_ptr + token * stride_i_t + cols * stride_i_d,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    if HAS_SCALE:
        values *= 1.0 / tl.load(scale_ptr).to(tl.float32)
    values = values.to(output_ptr.dtype.element_ty)
    for slot in range(topk):
        expert = tl.load(
            topk_ids_ptr + token * stride_id_t + slot * stride_id_k
        )
        if expert != num_local_experts:
            destination = tl.load(
                src2dst_ptr + token * stride_map_t + slot * stride_map_k
            ).to(tl.int64)
            tl.store(
                output_ptr + destination * stride_o_t + cols * stride_o_d,
                values,
                mask=mask,
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
    block_d = (
        8192 if hidden_size >= 4096 else triton.next_power_of_2(hidden_size)
    )
    common = (
        input,
        gateup_input,
        src2dst,
        topk_ids,
        a1_scales,
        input.stride(0),
        input.stride(1),
        gateup_input.stride(0),
        gateup_input.stride(1),
        src2dst.stride(0),
        src2dst.stride(1),
        topk_ids.stride(0),
        topk_ids.stride(1),
        num_local_experts,
        topk,
        hidden_size,
    )
    if num_tokens <= 16:
        _pre_reorder_xpu_route_owner[(num_tokens * topk,)](
            *common,
            HAS_SCALE=a1_scales is not None,
            BLOCK_D=block_d,
            num_warps=1,
            num_stages=1,
        )
    else:
        _pre_reorder_xpu_token_owner[(num_tokens,)](
            *common,
            HAS_SCALE=a1_scales is not None,
            BLOCK_D=block_d,
            num_warps=1,
            num_stages=1,
        )
    return gateup_input


__all__ = ["pre_reorder_cutlass"]
