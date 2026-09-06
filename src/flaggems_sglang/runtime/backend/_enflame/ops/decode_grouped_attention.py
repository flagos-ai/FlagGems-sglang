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

import torch
import triton
import triton.language as tl


@triton.jit
def _row_kernel(
    q,
    k_buffer,
    v_buffer,
    kv_indptr,
    kv_indices,
    output,
    sm_scale,
    H_Q: tl.constexpr,
    H_KV: tl.constexpr,
    D: tl.constexpr,
    D_V: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_DV: tl.constexpr,
):
    batch_id = tl.program_id(0)
    query_head = tl.program_id(1)
    kv_head = query_head // GROUP_SIZE
    d_offsets = tl.arange(0, BLOCK_D)
    dv_offsets = tl.arange(0, BLOCK_DV)
    q_values = tl.load(
        q + (batch_id * H_Q + query_head) * D + d_offsets,
        mask=d_offsets < D,
        other=0.0,
    ).to(tl.float32)
    start = tl.load(kv_indptr + batch_id)
    end = tl.load(kv_indptr + batch_id + 1)
    running_max = -float("inf")
    running_sum = 0.0
    accumulator = tl.zeros((BLOCK_DV,), dtype=tl.float32)
    for position in tl.range(start, end):
        page = tl.load(kv_indices + position)
        k_values = tl.load(
            k_buffer + (page * H_KV + kv_head) * D + d_offsets,
            mask=d_offsets < D,
            other=0.0,
        ).to(tl.float32)
        score = tl.sum(q_values * k_values, axis=0) * sm_scale
        next_max = tl.maximum(running_max, score)
        old_scale = tl.exp(running_max - next_max)
        probability = tl.exp(score - next_max)
        v_values = tl.load(
            v_buffer + (page * H_KV + kv_head) * D_V + dv_offsets,
            mask=dv_offsets < D_V,
            other=0.0,
        ).to(tl.float32)
        accumulator = accumulator * old_scale + probability * v_values
        running_sum = running_sum * old_scale + probability
        running_max = next_max
    output_offsets = (batch_id * H_Q + query_head) * D_V + dv_offsets
    tl.store(
        output + output_offsets,
        accumulator / running_sum,
        mask=dv_offsets < D_V,
    )


@triton.jit
def _flash_kernel(
    q,
    k_buffer,
    v_buffer,
    kv_indptr,
    kv_indices,
    output,
    sm_scale,
    H_Q: tl.constexpr,
    H_KV: tl.constexpr,
    D: tl.constexpr,
    D_V: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_DV: tl.constexpr,
):
    batch_id = tl.program_id(0)
    query_block = tl.program_id(1)
    kv_head = tl.program_id(2)
    query_offsets = (
        kv_head * GROUP_SIZE + query_block * BLOCK_H + tl.arange(0, BLOCK_H)
    )
    d_offsets = tl.arange(0, BLOCK_D)
    dv_offsets = tl.arange(0, BLOCK_DV)
    query_mask = query_offsets < (kv_head + 1) * GROUP_SIZE
    q_values = tl.load(
        q + (batch_id * H_Q + query_offsets[:, None]) * D + d_offsets[None, :],
        mask=query_mask[:, None] & (d_offsets[None, :] < D),
        other=0.0,
    )
    running_max = tl.full((BLOCK_H,), -float("inf"), dtype=tl.float32)
    running_sum = tl.zeros((BLOCK_H,), dtype=tl.float32)
    accumulator = tl.zeros((BLOCK_H, BLOCK_DV), dtype=tl.float32)
    start = tl.load(kv_indptr + batch_id)
    end = tl.load(kv_indptr + batch_id + 1)
    for block_start in tl.range(start, end, BLOCK_N):
        sequence_offsets = block_start + tl.arange(0, BLOCK_N)
        sequence_mask = sequence_offsets < end
        pages = tl.load(
            kv_indices + sequence_offsets, mask=sequence_mask, other=0
        )
        k_values = tl.load(
            k_buffer
            + (pages[:, None] * H_KV + kv_head) * D
            + d_offsets[None, :],
            mask=sequence_mask[:, None] & (d_offsets[None, :] < D),
            other=0.0,
        )
        scores = tl.dot(q_values, tl.trans(k_values)) * sm_scale
        scores = tl.where(
            query_mask[:, None] & sequence_mask[None, :],
            scores,
            -float("inf"),
        )
        block_max = tl.max(scores, axis=1)
        next_max = tl.where(
            query_mask, tl.maximum(running_max, block_max), 0.0
        )
        old_scale = tl.where(
            query_mask,
            tl.exp2((running_max - next_max) * 1.4426950408889634),
            0.0,
        )
        probabilities = tl.where(
            query_mask[:, None] & sequence_mask[None, :],
            tl.exp2((scores - next_max[:, None]) * 1.4426950408889634),
            0.0,
        )
        v_values = tl.load(
            v_buffer
            + (pages[:, None] * H_KV + kv_head) * D_V
            + dv_offsets[None, :],
            mask=sequence_mask[:, None] & (dv_offsets[None, :] < D_V),
            other=0.0,
        )
        accumulator = accumulator * old_scale[:, None] + tl.dot(
            probabilities.to(tl.bfloat16), v_values
        )
        running_sum = running_sum * old_scale + tl.sum(probabilities, axis=1)
        running_max = next_max
    output_offsets = (
        batch_id * H_Q + query_offsets[:, None]
    ) * D_V + dv_offsets[None, :]
    tl.store(
        output + output_offsets,
        accumulator / running_sum[:, None],
        mask=query_mask[:, None] & (dv_offsets[None, :] < D_V),
    )


def decode_grouped_attention(
    q, k_buffer, v_buffer, kv_indptr, kv_indices, sm_scale
):
    batch_size, query_heads, head_size = q.shape
    kv_heads = k_buffer.shape[1]
    value_size = v_buffer.shape[2]
    group_size = query_heads // kv_heads
    output = torch.empty(
        (batch_size, query_heads, value_size),
        dtype=torch.float32,
        device=q.device,
    )
    fast_path = (
        q.dtype == torch.bfloat16
        and k_buffer.dtype == torch.bfloat16
        and v_buffer.dtype == torch.bfloat16
        and query_heads == 128
        and kv_heads == 1
        and head_size == 128
        and value_size == 128
    )
    if fast_path:
        block_h, block_n, num_warps = 64, 128, 8
        _flash_kernel[
            (batch_size, triton.cdiv(group_size, block_h), kv_heads)
        ](
            q,
            k_buffer,
            v_buffer,
            kv_indptr,
            kv_indices,
            output,
            sm_scale,
            query_heads,
            kv_heads,
            head_size,
            value_size,
            group_size,
            block_h,
            block_n,
            triton.next_power_of_2(head_size),
            triton.next_power_of_2(value_size),
            num_warps=num_warps,
            num_stages=2,
        )
    else:
        _row_kernel[(batch_size, query_heads)](
            q,
            k_buffer,
            v_buffer,
            kv_indptr,
            kv_indices,
            output,
            sm_scale,
            query_heads,
            kv_heads,
            head_size,
            value_size,
            group_size,
            triton.next_power_of_2(head_size),
            triton.next_power_of_2(value_size),
            num_warps=8 if max(head_size, value_size) >= 256 else 4,
            num_stages=1,
        )
    return output


__all__ = ["decode_grouped_attention"]
