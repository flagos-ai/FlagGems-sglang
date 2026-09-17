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
def _segment_kernel(
    input_ids,
    weights,
    seg_indptr,
    weight_indices,
    lora_ranks,
    extra_embeddings,
    output,
    vocab_size,
    RANK: tl.constexpr,
    WORK_PER_SEGMENT: tl.constexpr,
    RANK_BLOCKS: tl.constexpr,
    EXTRA_COUNT: tl.constexpr,
    HAS_EXTRA: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    program_id = tl.program_id(0)
    segment = program_id // WORK_PER_SEGMENT
    inner = program_id - segment * WORK_PER_SEGMENT
    token_block = inner // RANK_BLOCKS
    rank_block = inner - token_block * RANK_BLOCKS
    start = tl.load(seg_indptr + segment)
    end = tl.load(seg_indptr + segment + 1)
    weight_index = tl.load(weight_indices + segment)
    active_rank = tl.load(lora_ranks + weight_index)
    rows = start + token_block * BLOCK_T + tl.arange(0, BLOCK_T)
    ranks = rank_block * BLOCK_R + tl.arange(0, BLOCK_R)
    row_mask = rows < end
    rank_mask = ranks < RANK
    valid = row_mask[:, None] & rank_mask[None, :]
    tokens = tl.load(input_ids + rows, mask=row_mask, other=0)
    is_extra = tokens >= vocab_size
    clamped = tl.minimum(tokens, vocab_size - 1)
    offsets = (weight_index * RANK + ranks[None, :]) * vocab_size + clamped[
        :, None
    ]
    values = tl.load(weights + offsets, mask=valid, other=0.0)
    if HAS_EXTRA:
        extra_indices = tl.maximum(tokens - vocab_size, 0)
        extra_offsets = (
            weight_index * EXTRA_COUNT + extra_indices[:, None]
        ) * RANK + ranks[None, :]
        extra_values = tl.load(
            extra_embeddings + extra_offsets,
            mask=valid & is_extra[:, None],
            other=0.0,
        )
        values = tl.where(is_extra[:, None], extra_values, values)
    values = tl.where(ranks[None, :] < active_rank, values, 0.0)
    tl.store(
        output + rows[:, None] * RANK + ranks[None, :], values, mask=valid
    )


@triton.jit
def _uniform_kernel(
    input_ids,
    weights,
    weight_indices,
    lora_ranks,
    output,
    vocab_size,
    token_count,
    RANK: tl.constexpr,
    SEGMENT_LEN: tl.constexpr,
    RANK_BLOCKS: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    program_id = tl.program_id(0)
    token_block = program_id // RANK_BLOCKS
    rank_block = program_id - token_block * RANK_BLOCKS
    rows = token_block * BLOCK_T + tl.arange(0, BLOCK_T)
    ranks = rank_block * BLOCK_R + tl.arange(0, BLOCK_R)
    row_mask = rows < token_count
    rank_mask = ranks[None, :] < RANK
    segments = rows // SEGMENT_LEN
    weight_index = tl.load(weight_indices + segments, mask=row_mask, other=0)
    active_rank = tl.load(lora_ranks + weight_index, mask=row_mask, other=0)
    tokens = tl.load(input_ids + rows, mask=row_mask, other=0)
    offsets = (
        weight_index[:, None] * RANK + ranks[None, :]
    ) * vocab_size + tokens[:, None]
    valid = row_mask[:, None] & rank_mask
    values = tl.load(weights + offsets, mask=valid, other=0.0)
    values = tl.where(ranks[None, :] < active_rank[:, None], values, 0.0)
    tl.store(
        output + rows[:, None] * RANK + ranks[None, :], values, mask=valid
    )


@triton.jit
def _flat_kernel(
    input_ids,
    weights,
    weight_indices,
    lora_ranks,
    output,
    vocab_size,
    element_count,
    RANK: tl.constexpr,
    SEGMENT_LEN: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < element_count
    rows = offsets // RANK
    ranks = offsets - rows * RANK
    segments = rows // SEGMENT_LEN
    weight_index = tl.load(weight_indices + segments, mask=mask, other=0)
    active_rank = tl.load(lora_ranks + weight_index, mask=mask, other=0)
    tokens = tl.load(input_ids + rows, mask=mask, other=0)
    weight_offsets = (weight_index * RANK + ranks) * vocab_size + tokens
    values = tl.load(weights + weight_offsets, mask=mask, other=0.0)
    values = tl.where(ranks < active_rank, values, 0.0)
    tl.store(output + offsets, values, mask=mask)


def _segment_launch(
    input_ids,
    weights,
    batch_info,
    vocab_size,
    extra_embeddings,
    output,
    block_t,
    block_r,
    num_warps,
):
    rank = weights.shape[1]
    token_blocks = triton.cdiv(batch_info.max_len, block_t)
    rank_blocks = triton.cdiv(rank, block_r)
    extra_count = 0 if extra_embeddings is None else extra_embeddings.shape[1]
    extra_pointer = weights if extra_embeddings is None else extra_embeddings
    _segment_kernel[(batch_info.bs * token_blocks * rank_blocks,)](
        input_ids,
        weights,
        batch_info.seg_indptr,
        batch_info.weight_indices,
        batch_info.lora_ranks,
        extra_pointer,
        output,
        vocab_size,
        rank,
        token_blocks * rank_blocks,
        rank_blocks,
        extra_count,
        extra_embeddings is not None,
        block_t,
        block_r,
        num_warps=num_warps,
        num_stages=2,
    )


def embedding_lora_a(
    input_ids, weights, batch_info, vocab_size, extra_embeddings=None
):
    token_count = input_ids.shape[0]
    rank = weights.shape[1]
    output = torch.empty(
        (token_count, rank), dtype=weights.dtype, device=weights.device
    )
    if token_count == 0:
        return output
    uniform = (
        extra_embeddings is None
        and token_count == batch_info.bs * batch_info.max_len
    )
    if rank == 32 and uniform:
        block_t, block_r = 64, 8
        rank_blocks = triton.cdiv(rank, block_r)
        _uniform_kernel[(triton.cdiv(token_count, block_t) * rank_blocks,)](
            input_ids,
            weights,
            batch_info.weight_indices,
            batch_info.lora_ranks,
            output,
            vocab_size,
            token_count,
            rank,
            batch_info.max_len,
            rank_blocks,
            block_t,
            block_r,
            num_warps=1,
            num_stages=2,
        )
    elif rank == 64 and uniform:
        block_size = 1024
        element_count = output.numel()
        _flat_kernel[(triton.cdiv(element_count, block_size),)](
            input_ids,
            weights,
            batch_info.weight_indices,
            batch_info.lora_ranks,
            output,
            vocab_size,
            element_count,
            rank,
            batch_info.max_len,
            block_size,
            num_warps=1,
            num_stages=2,
        )
    else:
        _segment_launch(
            input_ids,
            weights,
            batch_info,
            vocab_size,
            extra_embeddings,
            output,
            16,
            16,
            4,
        )
    return output


__all__ = ["embedding_lora_a"]
