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
def _decode_attention_kernel(
    q,
    k_buffer,
    v_buffer,
    kv_indptr,
    kv_indices,
    output,
    sm_scale,
    total_work,
    H_Q: tl.constexpr,
    H_KV: tl.constexpr,
    D: tl.constexpr,
    MAX_SEQ_LEN: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    program_id = tl.program_id(0)
    program_count = tl.num_programs(0)
    dim_lanes = tl.arange(0, BLOCK_D)
    dim_mask = dim_lanes < D
    token_lanes = tl.arange(0, BLOCK_N)
    for batch_head in tl.range(program_id, total_work, program_count):
        batch = batch_head // H_Q
        query_head = batch_head - batch * H_Q
        kv_head = query_head // (H_Q // H_KV)
        start = tl.load(kv_indptr + batch)
        end = tl.load(kv_indptr + batch + 1)
        sequence_length = end - start
        query = tl.load(
            q + batch_head * D + dim_lanes,
            mask=dim_mask,
            other=0.0,
        ).to(tl.float32)
        running_max = -float("inf")
        running_sum = 0.0
        accumulator = tl.zeros((BLOCK_D,), dtype=tl.float32)
        for block_start in tl.range(0, MAX_SEQ_LEN, BLOCK_N):
            token_offsets = block_start + token_lanes
            token_mask = token_offsets < sequence_length
            token_indices = tl.load(
                kv_indices + start + token_offsets,
                mask=token_mask,
                other=0,
            )
            kv_offsets = (
                token_indices[:, None] * H_KV + kv_head
            ) * D + dim_lanes[None, :]
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
            running_sum = running_sum * correction + tl.sum(
                probabilities, axis=0
            )
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


def decode_attention(q, k_buffer, v_buffer, kv_indptr, kv_indices, sm_scale):
    batch_size, query_heads, head_dim = q.shape
    kv_heads = k_buffer.shape[1]
    max_sequence_length = k_buffer.shape[0] // batch_size
    block_dim = triton.next_power_of_2(head_dim)
    block_tokens = 16 if block_dim >= 512 else 32
    total_work = batch_size * query_heads
    output = torch.empty(
        (batch_size, query_heads, v_buffer.shape[-1]),
        dtype=torch.float32,
        device=q.device,
    )
    _decode_attention_kernel[(min(total_work, 65535),)](
        q,
        k_buffer,
        v_buffer,
        kv_indptr,
        kv_indices,
        output,
        sm_scale,
        total_work,
        query_heads,
        kv_heads,
        head_dim,
        max_sequence_length,
        block_dim,
        block_tokens,
        num_warps=4,
        num_stages=2,
    )
    return output


__all__ = ["decode_attention"]
