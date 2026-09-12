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
    EXTRA_COUNT: tl.constexpr,
    HAS_EXTRA: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    program_id = tl.program_id(0)
    segment = program_id // WORK_PER_SEGMENT
    inner = program_id - segment * WORK_PER_SEGMENT
    token_block = inner // RANK
    rank_offset = inner - token_block * RANK
    start = tl.load(seg_indptr + segment)
    end = tl.load(seg_indptr + segment + 1)
    weight_index = tl.load(weight_indices + segment)
    active_rank = tl.load(lora_ranks + weight_index)
    rows = start + token_block * BLOCK_T + tl.arange(0, BLOCK_T)
    row_mask = rows < end
    tokens = tl.load(input_ids + rows, mask=row_mask, other=0)
    is_extra = tokens >= vocab_size
    clamped = tl.minimum(tokens, vocab_size - 1)
    offsets = (weight_index * RANK + rank_offset) * vocab_size + clamped
    values = tl.load(weights + offsets, mask=row_mask, other=0.0)
    if HAS_EXTRA:
        extra_indices = tl.maximum(tokens - vocab_size, 0)
        extra_offsets = (
            weight_index * EXTRA_COUNT + extra_indices
        ) * RANK + rank_offset
        extra_values = tl.load(
            extra_embeddings + extra_offsets,
            mask=row_mask & is_extra,
            other=0.0,
        )
        values = tl.where(is_extra, extra_values, values)
    values = tl.where(rank_offset < active_rank, values, 0.0)
    tl.store(output + rows * RANK + rank_offset, values, mask=row_mask)


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
    block_t = 512 if rank == 32 else 2048
    num_warps = 8 if rank == 32 else 1
    token_blocks = triton.cdiv(batch_info.max_len, block_t)
    work_per_segment = token_blocks * rank
    extra_count = 0 if extra_embeddings is None else extra_embeddings.shape[1]
    extra_pointer = weights if extra_embeddings is None else extra_embeddings
    _kernel[(batch_info.bs * work_per_segment,)](
        input_ids,
        weights,
        batch_info.seg_indptr,
        batch_info.weight_indices,
        batch_info.lora_ranks,
        extra_pointer,
        output,
        vocab_size,
        rank,
        work_per_segment,
        extra_count,
        extra_embeddings is not None,
        block_t,
        num_warps=num_warps,
        num_stages=2,
    )
    return output


__all__ = ["embedding_lora_a"]
