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
def _direct_kernel(
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
    MAX_SEQ_LEN: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    query_head = tl.program_id(0)
    batch = tl.program_id(1)
    batch_head = batch * H_Q + query_head
    kv_head = query_head // (H_Q // H_KV)
    start = tl.load(kv_indptr + batch)
    end = tl.load(kv_indptr + batch + 1)
    sequence_length = end - start
    dim_lanes = tl.arange(0, BLOCK_D)
    dim_mask = dim_lanes < D
    query = tl.load(
        q + batch_head * D + dim_lanes,
        mask=dim_mask,
        other=0.0,
    ).to(tl.float32)
    running_max = -float("inf")
    running_sum = 0.0
    accumulator = tl.zeros((BLOCK_D,), dtype=tl.float32)
    token_lanes = tl.arange(0, BLOCK_N)
    for block_start in tl.range(0, MAX_SEQ_LEN, BLOCK_N):
        token_offsets = block_start + token_lanes
        token_mask = token_offsets < sequence_length
        token_indices = tl.load(
            kv_indices + start + token_offsets,
            mask=token_mask,
            other=0,
        )
        kv_offsets = (token_indices[:, None] * H_KV + kv_head) * D + dim_lanes[
            None, :
        ]
        keys = tl.load(
            k_buffer + kv_offsets,
            mask=token_mask[:, None] & dim_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        logits = tl.sum(keys * query[None, :], axis=1) * sm_scale
        logits = tl.where(token_mask, logits, -float("inf"))
        block_max = tl.max(logits, axis=0)
        next_max = tl.maximum(running_max, block_max)
        correction = tl.exp(running_max - next_max)
        probabilities = tl.exp(logits - next_max)
        running_sum = running_sum * correction + tl.sum(probabilities, axis=0)
        values = tl.load(
            v_buffer + kv_offsets,
            mask=token_mask[:, None] & dim_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        accumulator = accumulator * correction + tl.sum(
            probabilities[:, None] * values,
            axis=0,
        )
        running_max = next_max
    tl.store(
        output + batch_head * D + dim_lanes,
        accumulator / running_sum,
        mask=dim_mask,
    )


@triton.jit
def _dot_kernel(
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
    GROUP_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    kv_head = tl.program_id(0)
    batch = tl.program_id(1)
    start = tl.load(kv_indptr + batch)
    end = tl.load(kv_indptr + batch + 1)
    sequence_length = end - start
    head_lanes = tl.arange(0, BLOCK_M)
    dim_lanes = tl.arange(0, BLOCK_D)
    token_lanes = tl.arange(0, BLOCK_N)
    query_heads = kv_head * GROUP_SIZE + head_lanes
    query_mask = head_lanes < GROUP_SIZE
    dim_mask = dim_lanes < D
    query = tl.load(
        q + (batch * H_Q + query_heads[:, None]) * D + dim_lanes[None, :],
        mask=query_mask[:, None] & dim_mask[None, :],
        other=0.0,
    )
    running_max = tl.full((BLOCK_M,), -float("inf"), tl.float32)
    running_sum = tl.zeros((BLOCK_M,), dtype=tl.float32)
    accumulator = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
    log2e: tl.constexpr = 1.4426950408889634
    for block_start in range(0, sequence_length, BLOCK_N):
        token_offsets = block_start + token_lanes
        token_mask = token_offsets < sequence_length
        token_indices = tl.load(
            kv_indices + start + token_offsets,
            mask=token_mask,
            other=0,
        )
        keys = tl.load(
            k_buffer
            + (token_indices[None, :] * H_KV + kv_head) * D
            + dim_lanes[:, None],
            mask=dim_mask[:, None] & token_mask[None, :],
            other=0.0,
        )
        logits = tl.dot(query, keys, allow_tf32=False)
        logits *= sm_scale * log2e
        logits = tl.where(token_mask[None, :], logits, -float("inf"))
        next_max = tl.maximum(running_max, tl.max(logits, axis=1))
        correction = tl.math.exp2(running_max - next_max)
        probabilities = tl.math.exp2(logits - next_max[:, None])
        running_sum = running_sum * correction + tl.sum(probabilities, axis=1)
        accumulator *= correction[:, None]
        values = tl.load(
            v_buffer
            + (token_indices[:, None] * H_KV + kv_head) * D
            + dim_lanes[None, :],
            mask=token_mask[:, None] & dim_mask[None, :],
            other=0.0,
        )
        accumulator = tl.dot(
            probabilities.to(values.dtype),
            values,
            accumulator,
            allow_tf32=False,
        )
        running_max = next_max
    tl.store(
        output + (batch * H_Q + query_heads[:, None]) * D + dim_lanes[None, :],
        accumulator / running_sum[:, None],
        mask=query_mask[:, None] & dim_mask[None, :],
    )


def _launch_direct(q, k_buffer, v_buffer, kv_indptr, kv_indices, sm_scale):
    batch_size, query_heads, head_dim = q.shape
    kv_heads = k_buffer.shape[1]
    block_dim = triton.next_power_of_2(head_dim)
    block_tokens = 16 if block_dim >= 512 else 32 if block_dim >= 128 else 64
    output = torch.empty(
        (batch_size, query_heads, v_buffer.shape[-1]),
        dtype=torch.float32,
        device=q.device,
    )
    _direct_kernel[(query_heads, batch_size)](
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
        k_buffer.shape[0] // batch_size,
        block_dim,
        block_tokens,
        num_warps=4,
        num_stages=2,
    )
    return output


def _launch_dot(
    q,
    k_buffer,
    v_buffer,
    kv_indptr,
    kv_indices,
    sm_scale,
    block_tokens,
    num_warps,
):
    batch_size, query_heads, head_dim = q.shape
    kv_heads = k_buffer.shape[1]
    group_size = query_heads // kv_heads
    block_heads = max(16, triton.next_power_of_2(group_size))
    block_dim = triton.next_power_of_2(head_dim)
    output = torch.empty(
        (batch_size, query_heads, v_buffer.shape[-1]),
        dtype=torch.float32,
        device=q.device,
    )
    _dot_kernel[(kv_heads, batch_size)](
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
        group_size,
        block_heads,
        block_dim,
        block_tokens,
        num_warps=num_warps,
        num_stages=2,
    )
    return output


def decode_attention(q, k_buffer, v_buffer, kv_indptr, kv_indices, sm_scale):
    batch_size, query_heads, head_dim = q.shape
    if query_heads == 32 and k_buffer.shape[1] == 8 and head_dim == 128:
        block_tokens = 64 if batch_size == 1 else 32
        num_warps = 8 if batch_size == 1 else 4
        return _launch_dot(
            q,
            k_buffer,
            v_buffer,
            kv_indptr,
            kv_indices,
            sm_scale,
            block_tokens,
            num_warps,
        )
    return _launch_direct(
        q, k_buffer, v_buffer, kv_indptr, kv_indices, sm_scale
    )


__all__ = ["decode_attention"]
