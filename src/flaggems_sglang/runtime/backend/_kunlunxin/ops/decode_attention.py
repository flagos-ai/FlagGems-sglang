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
def _decode_attention_serial_kernel(
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
    BLOCK_D: tl.constexpr,
):
    batch_head = tl.program_id(0)
    batch = batch_head // H_Q
    query_head = batch_head - batch * H_Q
    kv_head = query_head // (H_Q // H_KV)
    start = tl.load(kv_indptr + batch)
    end = tl.load(kv_indptr + batch + 1)
    sequence_length = end - start

    dim_offsets = tl.arange(0, BLOCK_D)
    dim_mask = dim_offsets < D
    query = tl.load(
        q + batch_head * D + dim_offsets,
        mask=dim_mask,
        other=0.0,
    )
    accumulator = tl.zeros((BLOCK_D,), dtype=tl.float32)
    running_max = float("-inf")
    running_sum = 0.0

    for token_offset in range(sequence_length):
        token_index = tl.load(kv_indices + start + token_offset)
        kv_base = (token_index * H_KV + kv_head) * D
        key = tl.load(
            k_buffer + kv_base + dim_offsets,
            mask=dim_mask,
            other=0.0,
        )
        score = tl.sum(query * key) * sm_scale
        next_max = tl.maximum(running_max, score)
        correction = tl.exp(running_max - next_max)
        probability = tl.exp(score - next_max)
        value = tl.load(
            v_buffer + kv_base + dim_offsets,
            mask=dim_mask,
            other=0.0,
        ).to(tl.float32)
        accumulator = accumulator * correction + probability * value
        running_sum = running_sum * correction + probability
        running_max = next_max

    tl.store(
        output + batch_head * D + dim_offsets,
        accumulator / running_sum,
        mask=dim_mask,
    )


@triton.jit
def _decode_attention_group4_kernel(
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
    BLOCK_D: tl.constexpr,
):
    batch_kv = tl.program_id(0)
    batch = batch_kv // H_KV
    kv_head = batch_kv - batch * H_KV
    first_query_head = kv_head * 4
    start = tl.load(kv_indptr + batch)
    end = tl.load(kv_indptr + batch + 1)
    sequence_length = end - start
    dim_lanes = tl.arange(0, BLOCK_D)
    dim_mask = dim_lanes < D
    query_base = (batch * H_Q + first_query_head) * D
    query0 = tl.load(q + query_base + dim_lanes, mask=dim_mask, other=0.0)
    query1 = tl.load(q + query_base + D + dim_lanes, mask=dim_mask, other=0.0)
    query2 = tl.load(
        q + query_base + 2 * D + dim_lanes, mask=dim_mask, other=0.0
    )
    query3 = tl.load(
        q + query_base + 3 * D + dim_lanes, mask=dim_mask, other=0.0
    )
    accumulator0 = tl.zeros((BLOCK_D,), dtype=tl.float32)
    accumulator1 = tl.zeros((BLOCK_D,), dtype=tl.float32)
    accumulator2 = tl.zeros((BLOCK_D,), dtype=tl.float32)
    accumulator3 = tl.zeros((BLOCK_D,), dtype=tl.float32)
    maximum0 = float("-inf")
    maximum1 = float("-inf")
    maximum2 = float("-inf")
    maximum3 = float("-inf")
    denominator0 = 0.0
    denominator1 = 0.0
    denominator2 = 0.0
    denominator3 = 0.0
    for token_offset in range(sequence_length):
        token_index = tl.load(kv_indices + start + token_offset)
        kv_base = (token_index * H_KV + kv_head) * D
        key = tl.load(k_buffer + kv_base + dim_lanes, mask=dim_mask, other=0.0)
        value = tl.load(
            v_buffer + kv_base + dim_lanes,
            mask=dim_mask,
            other=0.0,
        ).to(tl.float32)
        score0 = tl.sum(query0 * key) * sm_scale
        next_maximum0 = tl.maximum(maximum0, score0)
        correction0 = tl.exp(maximum0 - next_maximum0)
        probability0 = tl.exp(score0 - next_maximum0)
        accumulator0 = accumulator0 * correction0 + probability0 * value
        denominator0 = denominator0 * correction0 + probability0
        maximum0 = next_maximum0
        score1 = tl.sum(query1 * key) * sm_scale
        next_maximum1 = tl.maximum(maximum1, score1)
        correction1 = tl.exp(maximum1 - next_maximum1)
        probability1 = tl.exp(score1 - next_maximum1)
        accumulator1 = accumulator1 * correction1 + probability1 * value
        denominator1 = denominator1 * correction1 + probability1
        maximum1 = next_maximum1
        score2 = tl.sum(query2 * key) * sm_scale
        next_maximum2 = tl.maximum(maximum2, score2)
        correction2 = tl.exp(maximum2 - next_maximum2)
        probability2 = tl.exp(score2 - next_maximum2)
        accumulator2 = accumulator2 * correction2 + probability2 * value
        denominator2 = denominator2 * correction2 + probability2
        maximum2 = next_maximum2
        score3 = tl.sum(query3 * key) * sm_scale
        next_maximum3 = tl.maximum(maximum3, score3)
        correction3 = tl.exp(maximum3 - next_maximum3)
        probability3 = tl.exp(score3 - next_maximum3)
        accumulator3 = accumulator3 * correction3 + probability3 * value
        denominator3 = denominator3 * correction3 + probability3
        maximum3 = next_maximum3
    output_base = (batch * H_Q + first_query_head) * D
    tl.store(
        output + output_base + dim_lanes,
        accumulator0 / denominator0,
        mask=dim_mask,
    )
    tl.store(
        output + output_base + D + dim_lanes,
        accumulator1 / denominator1,
        mask=dim_mask,
    )
    tl.store(
        output + output_base + 2 * D + dim_lanes,
        accumulator2 / denominator2,
        mask=dim_mask,
    )
    tl.store(
        output + output_base + 3 * D + dim_lanes,
        accumulator3 / denominator3,
        mask=dim_mask,
    )


def decode_attention(q, k_buffer, v_buffer, kv_indptr, kv_indices, sm_scale):
    batch_size, query_heads, head_dim = q.shape
    kv_heads = k_buffer.shape[1]
    block_dim = triton.next_power_of_2(head_dim)
    output = torch.empty(
        (batch_size, query_heads, v_buffer.shape[-1]),
        dtype=torch.float32,
        device=q.device,
    )
    if query_heads == 32 and kv_heads == 8 and head_dim == 128:
        _decode_attention_group4_kernel[(batch_size * kv_heads,)](
            q,
            k_buffer,
            v_buffer,
            kv_indptr,
            kv_indices,
            output,
            sm_scale,
            query_heads,
            kv_heads,
            head_dim,
            block_dim,
            num_warps=2,
            num_stages=2,
        )
        return output
    _decode_attention_serial_kernel[(batch_size * query_heads,)](
        q,
        k_buffer,
        v_buffer,
        kv_indptr,
        kv_indices,
        output,
        sm_scale,
        query_heads,
        kv_heads,
        head_dim,
        block_dim,
        num_warps=2,
        num_stages=2,
    )
    return output


__all__ = ["decode_attention"]
