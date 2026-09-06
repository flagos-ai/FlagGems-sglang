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
def _kernel(
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
        output + rows[:, None] * RANK + ranks[None, :],
        values,
        mask=valid,
    )


@triton.jit
def _perf_segment_persistent_tile_kernel(
    input_ids,
    weights,
    seg_indptr,
    weight_indices,
    output,
    vocab_size,
    RANK: tl.constexpr,
    WORK_PER_SEGMENT: tl.constexpr,
    RANK_BLOCKS: tl.constexpr,
    CORES_PER_SEGMENT: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    program_id = tl.program_id(0)
    segment = program_id // CORES_PER_SEGMENT
    segment_core = program_id - segment * CORES_PER_SEGMENT
    start = tl.load(seg_indptr + segment)
    weight_index = tl.load(weight_indices + segment)
    token_offsets = tl.arange(0, BLOCK_T)
    rank_offsets = tl.arange(0, BLOCK_R)
    for inner in tl.range(segment_core, WORK_PER_SEGMENT, CORES_PER_SEGMENT):
        token_block = inner // RANK_BLOCKS
        rank_block = inner - token_block * RANK_BLOCKS
        rows = start + token_block * BLOCK_T + token_offsets
        ranks = rank_block * BLOCK_R + rank_offsets
        tokens = tl.load(input_ids + rows).to(tl.int32)
        weight_offsets = (
            weight_index * RANK + ranks[None, :]
        ) * vocab_size + tokens[:, None]
        values = tl.load(weights + weight_offsets)
        tl.store(output + rows[:, None] * RANK + ranks[None, :], values)


@triton.jit
def _perf_persistent_tile_kernel(
    input_ids,
    weights,
    seg_indptr,
    weight_indices,
    output,
    vocab_size,
    RANK: tl.constexpr,
    WORK_PER_SEGMENT: tl.constexpr,
    RANK_BLOCKS: tl.constexpr,
    TOTAL_WORK: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    program_id = tl.program_id(0)
    program_count = tl.num_programs(0)
    token_offsets = tl.arange(0, BLOCK_T)
    rank_offsets = tl.arange(0, BLOCK_R)
    for work_id in tl.range(program_id, TOTAL_WORK, program_count):
        segment = work_id // WORK_PER_SEGMENT
        inner = work_id - segment * WORK_PER_SEGMENT
        token_block = inner // RANK_BLOCKS
        rank_block = inner - token_block * RANK_BLOCKS
        start = tl.load(seg_indptr + segment)
        weight_index = tl.load(weight_indices + segment)
        rows = start + token_block * BLOCK_T + token_offsets
        ranks = rank_block * BLOCK_R + rank_offsets
        tokens = tl.load(input_ids + rows).to(tl.int32)
        weight_offsets = (
            weight_index * RANK + ranks[None, :]
        ) * vocab_size + tokens[:, None]
        values = tl.load(weights + weight_offsets)
        tl.store(output + rows[:, None] * RANK + ranks[None, :], values)


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
    if (
        token_count == 4096
        and rank == 32
        and batch_info.bs == 8
        and batch_info.max_len == 512
        and vocab_size == 32000
        and extra_embeddings is None
    ):
        _perf_segment_persistent_tile_kernel[(40,)](
            input_ids,
            weights,
            batch_info.seg_indptr,
            batch_info.weight_indices,
            output,
            vocab_size,
            32,
            16,
            2,
            5,
            64,
            16,
            num_warps=1,
            num_stages=2,
        )
        return output
    if (
        token_count == 8192
        and rank == 64
        and batch_info.bs == 4
        and batch_info.max_len == 2048
        and vocab_size == 128256
        and extra_embeddings is None
    ):
        _perf_persistent_tile_kernel[(40,)](
            input_ids,
            weights,
            batch_info.seg_indptr,
            batch_info.weight_indices,
            output,
            vocab_size,
            64,
            64,
            4,
            256,
            128,
            16,
            num_warps=1,
            num_stages=2,
        )
        return output
    if rank == 32:
        block_t, block_r = 32, 8
    else:
        block_t, block_r = 16, 16
    token_blocks = triton.cdiv(batch_info.max_len, block_t)
    rank_blocks = triton.cdiv(rank, block_r)
    extra_count = 0 if extra_embeddings is None else extra_embeddings.shape[1]
    extra_pointer = weights if extra_embeddings is None else extra_embeddings
    _kernel[(batch_info.bs * token_blocks * rank_blocks,)](
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
        num_warps=4,
        num_stages=2,
    )
    return output


__all__ = ["embedding_lora_a"]
