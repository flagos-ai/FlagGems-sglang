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
def _gather_kv_kernel(
    k_buffer,
    v_buffer,
    kv_indices,
    gathered_k,
    gathered_v,
    TOTAL_TOKENS: tl.constexpr,
    NUM_PROGRAMS: tl.constexpr,
    H_KV: tl.constexpr,
    D: tl.constexpr,
    D_V: tl.constexpr,
):
    dim_offsets = tl.arange(0, D)
    for token_id in range(tl.program_id(0), TOTAL_TOKENS, NUM_PROGRAMS):
        page = tl.load(kv_indices + token_id)
        keys = tl.load(k_buffer + page * H_KV * D + dim_offsets)
        values = tl.load(v_buffer + page * H_KV * D_V + dim_offsets)
        tl.store(gathered_k + token_id * D + dim_offsets, keys)
        tl.store(gathered_v + token_id * D_V + dim_offsets, values)


@triton.jit
def _scores_dot_kernel(
    q,
    gathered_k,
    scores,
    sm_scale,
    H_Q: tl.constexpr,
    D: tl.constexpr,
    SEQUENCE_LENGTH: tl.constexpr,
    NUM_TOKEN_BLOCKS: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    batch_token_block = tl.program_id(0)
    head_block = tl.program_id(1)
    batch_id = batch_token_block // NUM_TOKEN_BLOCKS
    token_block = batch_token_block - batch_id * NUM_TOKEN_BLOCKS
    head_offsets = head_block * BLOCK_H + tl.arange(0, BLOCK_H)
    token_offsets = token_block * BLOCK_N + tl.arange(0, BLOCK_N)
    dim_offsets = tl.arange(0, D)
    query_values = tl.load(
        q
        + (batch_id * H_Q + head_offsets[:, None]) * D
        + dim_offsets[None, :],
        mask=head_offsets[:, None] < H_Q,
        other=0.0,
    ).to(tl.bfloat16)
    key_values = tl.load(
        gathered_k
        + (batch_id * SEQUENCE_LENGTH + token_offsets[None, :]) * D
        + dim_offsets[:, None],
        mask=token_offsets[None, :] < SEQUENCE_LENGTH,
        other=0.0,
    ).to(tl.bfloat16)
    score_values = tl.dot(query_values, key_values) * sm_scale
    score_offsets = (
        batch_id * H_Q + head_offsets[:, None]
    ) * SEQUENCE_LENGTH + token_offsets[None, :]
    tl.store(
        scores + score_offsets,
        score_values,
        mask=(head_offsets[:, None] < H_Q)
        & (token_offsets[None, :] < SEQUENCE_LENGTH),
    )


@triton.jit
def _softmax_kernel(
    scores,
    H_Q: tl.constexpr,
    SEQUENCE_LENGTH: tl.constexpr,
):
    head_id = tl.program_id(0)
    batch_id = tl.program_id(1)
    token_offsets = tl.arange(0, SEQUENCE_LENGTH)
    offsets = (batch_id * H_Q + head_id) * SEQUENCE_LENGTH + token_offsets
    values = tl.load(scores + offsets)
    values -= tl.max(values, axis=0)
    probabilities = tl.exp(values)
    probabilities /= tl.sum(probabilities, axis=0)
    tl.store(scores + offsets, probabilities)


@triton.jit
def _output_dot_kernel(
    gathered_v,
    scores,
    output,
    H_Q: tl.constexpr,
    D_V: tl.constexpr,
    SEQUENCE_LENGTH: tl.constexpr,
    NUM_DIM_BLOCKS: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    tile_id = tl.program_id(0)
    batch_id = tl.program_id(1)
    head_block = tile_id // NUM_DIM_BLOCKS
    dim_block = tile_id - head_block * NUM_DIM_BLOCKS
    head_offsets = head_block * BLOCK_H + tl.arange(0, BLOCK_H)
    dim_offsets = dim_block * BLOCK_D + tl.arange(0, BLOCK_D)
    accumulator = tl.zeros((BLOCK_H, BLOCK_D), tl.float32)
    for token_start in range(0, SEQUENCE_LENGTH, BLOCK_N):
        token_offsets = token_start + tl.arange(0, BLOCK_N)
        probabilities = tl.load(
            scores
            + (batch_id * H_Q + head_offsets[:, None]) * SEQUENCE_LENGTH
            + token_offsets[None, :],
            mask=(head_offsets[:, None] < H_Q)
            & (token_offsets[None, :] < SEQUENCE_LENGTH),
            other=0.0,
        ).to(tl.bfloat16)
        values = tl.load(
            gathered_v
            + (batch_id * SEQUENCE_LENGTH + token_offsets[:, None]) * D_V
            + dim_offsets[None, :],
            mask=(token_offsets[:, None] < SEQUENCE_LENGTH)
            & (dim_offsets[None, :] < D_V),
            other=0.0,
        ).to(tl.bfloat16)
        accumulator += tl.dot(probabilities, values)
    output_offsets = (
        batch_id * H_Q + head_offsets[:, None]
    ) * D_V + dim_offsets[None, :]
    tl.store(
        output + output_offsets,
        accumulator,
        mask=(head_offsets[:, None] < H_Q) & (dim_offsets[None, :] < D_V),
    )


def decode_grouped_attention(
    q, k_buffer, v_buffer, kv_indptr, kv_indices, sm_scale
):
    batch_size, query_heads, head_size = q.shape
    kv_heads = k_buffer.shape[1]
    value_size = v_buffer.shape[2]
    output = torch.empty(
        (batch_size, query_heads, value_size),
        dtype=torch.float32,
        device=q.device,
    )
    matrix_path = (
        q.dtype == torch.bfloat16
        and k_buffer.dtype == torch.bfloat16
        and v_buffer.dtype == torch.bfloat16
        and query_heads == 128
        and kv_heads == 1
        and head_size == 128
        and value_size == 128
    )
    if matrix_path:
        block_h = 32
        block_n = 32
        block_d = 32
        total_tokens = kv_indices.numel()
        sequence_length = total_tokens // batch_size
        gathered_k = torch.empty_like(k_buffer)
        gathered_v = torch.empty_like(v_buffer)
        scores = torch.empty(
            (batch_size, query_heads, sequence_length),
            dtype=torch.float32,
            device=q.device,
        )
        num_gather_programs = min(total_tokens, 32768)
        _gather_kv_kernel[(num_gather_programs,)](
            k_buffer,
            v_buffer,
            kv_indices,
            gathered_k,
            gathered_v,
            total_tokens,
            num_gather_programs,
            kv_heads,
            head_size,
            value_size,
            num_warps=4,
        )
        num_token_blocks = triton.cdiv(sequence_length, block_n)
        num_head_blocks = triton.cdiv(query_heads, block_h)
        _scores_dot_kernel[(batch_size * num_token_blocks, num_head_blocks)](
            q,
            gathered_k,
            scores,
            sm_scale,
            query_heads,
            head_size,
            sequence_length,
            num_token_blocks,
            block_h,
            block_n,
            num_warps=4,
            num_stages=1,
        )
        _softmax_kernel[(query_heads, batch_size)](
            scores,
            query_heads,
            sequence_length,
            num_warps=4,
        )
        num_dim_blocks = triton.cdiv(value_size, block_d)
        _output_dot_kernel[(num_head_blocks * num_dim_blocks, batch_size)](
            gathered_v,
            scores,
            output,
            query_heads,
            value_size,
            sequence_length,
            num_dim_blocks,
            block_h,
            block_n,
            block_d,
            num_warps=4,
            num_stages=1,
        )
    else:
        group_size = query_heads // kv_heads
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
