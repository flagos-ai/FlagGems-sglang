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
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import triton
import triton.language as tl

_XPU_CLUSTER_COUNT = 12
_MAX_VOCAB_TILE = 8192
_INT64_DRAFT_IS_INT32_PAIRS = True


@triton.jit
def _draft_topk1_argmax_2d_kernel(
    logits_ptr,
    topk_index_ptr,
    stride_logits_batch,
    stride_logits_vocab,
    BATCH_SIZE: tl.constexpr,
    VOCAB_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    program = tl.program_id(0).to(tl.int64)
    row = program * BLOCK_M + tl.arange(0, BLOCK_M)
    row_mask = row < BATCH_SIZE
    safe_row = tl.minimum(row, BATCH_SIZE - 1)
    max_values = tl.full([BLOCK_M], -float("inf"), tl.float32)
    argmax_values = tl.full([BLOCK_M], 0, tl.int64)

    for start_n in range(0, VOCAB_SIZE, BLOCK_N):
        column = start_n + tl.arange(0, BLOCK_N)
        column_mask = column < VOCAB_SIZE
        safe_column = tl.minimum(column, VOCAB_SIZE - 1)
        values = tl.load(
            logits_ptr + safe_row[:, None] * VOCAB_SIZE + safe_column[None, :],
        )
        values = tl.where(
            row_mask[:, None] & column_mask[None, :],
            values,
            -float("inf"),
        )
        local_max, local_argmax = tl.max(
            values,
            axis=1,
            return_indices=True,
            return_indices_tie_break_left=True,
        )
        update = local_max > max_values
        max_values = tl.where(update, local_max, max_values)
        argmax_values = tl.where(
            update,
            start_n + local_argmax,
            argmax_values,
        )

    store_row = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    tl.store(
        topk_index_ptr + store_row,
        argmax_values,
        mask=store_row < BATCH_SIZE,
    )


@triton.jit
def _draft_topk1_partial_2d_kernel(
    logits_ptr,
    partial_values_ptr,
    partial_indices_ptr,
    stride_logits_batch,
    stride_logits_vocab,
    BATCH_SIZE: tl.constexpr,
    VOCAB_SIZE: tl.constexpr,
    PARTITION_COUNT: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    program = tl.program_id(0)
    partition = program % PARTITION_COUNT
    row_group = program // PARTITION_COUNT
    row = row_group.to(tl.int64) * BLOCK_M + tl.arange(0, BLOCK_M)
    column = partition * BLOCK_N + tl.arange(0, BLOCK_N)
    row_mask = row < BATCH_SIZE
    column_mask = column < VOCAB_SIZE
    safe_row = tl.minimum(row, BATCH_SIZE - 1)
    safe_column = tl.minimum(column, VOCAB_SIZE - 1)
    values = tl.load(
        logits_ptr + safe_row[:, None] * VOCAB_SIZE + safe_column[None, :],
    )
    values = tl.where(
        row_mask[:, None] & column_mask[None, :],
        values,
        -float("inf"),
    )
    local_max, local_argmax = tl.max(
        values,
        axis=1,
        return_indices=True,
        return_indices_tie_break_left=True,
    )
    selected_index = partition * BLOCK_N + local_argmax

    store_program = tl.program_id(0)
    store_partition = store_program % PARTITION_COUNT
    store_row_group = store_program // PARTITION_COUNT
    store_row = store_row_group * BLOCK_M + tl.arange(0, BLOCK_M)
    output_offset = store_row * PARTITION_COUNT + store_partition
    store_mask = store_row < BATCH_SIZE
    tl.store(partial_values_ptr + output_offset, local_max, mask=store_mask)
    tl.store(
        partial_indices_ptr + output_offset,
        selected_index,
        mask=store_mask,
    )


@triton.jit
def _draft_topk1_finalize_partitions_kernel(
    partial_values_ptr,
    partial_indices_ptr,
    topk_index_ptr,
    BATCH_SIZE: tl.constexpr,
    PARTITION_COUNT: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    row = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    row_mask = row < BATCH_SIZE
    safe_row = tl.minimum(row, BATCH_SIZE - 1)
    max_value = tl.full([BLOCK_M], -float("inf"), tl.float32)
    max_index = tl.full([BLOCK_M], 0, tl.int64)
    for partition in range(0, PARTITION_COUNT):
        offset = safe_row * PARTITION_COUNT + partition
        value = tl.load(partial_values_ptr + offset)
        index = tl.load(partial_indices_ptr + offset)
        update = value > max_value
        max_value = tl.where(update, value, max_value)
        max_index = tl.where(update, index, max_index)
    tl.store(topk_index_ptr + row, max_index, mask=row_mask)


@triton.jit
def _draft_topk1_write_outputs_kernel(
    topk_index_words_ptr,
    positions_ptr,
    draft_tokens_ptr,
    topk_p_ptr,
    out_positions_ptr,
    out_draft_tokens_ptr,
    stride_positions,
    stride_draft_batch,
    stride_draft_column,
    stride_out_draft_batch,
    stride_out_draft_column,
    BATCH_SIZE: tl.constexpr,
    DRAFT_SIZE: tl.constexpr,
    BLOCK_DRAFT: tl.constexpr,
    DRAFT_TOKEN_COLUMN: tl.constexpr,
    HAS_DRAFT_TOKENS: tl.constexpr,
    INT64_DRAFT_IS_INT32_PAIRS: tl.constexpr,
):
    row = tl.program_id(0)
    selected_low = tl.load(topk_index_words_ptr + row * 2)
    tl.store(topk_p_ptr + row, 1.0)
    position = tl.load(positions_ptr + row * stride_positions)
    tl.store(out_positions_ptr + row, position + 1)

    if HAS_DRAFT_TOKENS:
        column = tl.arange(0, BLOCK_DRAFT)
        mask = column < DRAFT_SIZE
        safe_column = tl.minimum(column, DRAFT_SIZE - 1)
        input_offset = (
            row * stride_draft_batch + safe_column * stride_draft_column
        )
        output_offset = (
            row * stride_out_draft_batch + column * stride_out_draft_column
        )
        if INT64_DRAFT_IS_INT32_PAIRS:
            low_word = tl.load(draft_tokens_ptr + input_offset)
            high_word = tl.load(draft_tokens_ptr + input_offset + 1)
            selected = column == DRAFT_TOKEN_COLUMN
            low_word = tl.where(selected, selected_low, low_word)
            high_word = tl.where(selected, 0, high_word)
            tl.store(
                out_draft_tokens_ptr + output_offset,
                low_word,
                mask=mask,
            )
            tl.store(
                out_draft_tokens_ptr + output_offset + 1,
                high_word,
                mask=mask,
            )
        else:
            values = tl.load(draft_tokens_ptr + input_offset)
            values = tl.where(
                column == DRAFT_TOKEN_COLUMN,
                selected_low,
                values,
            )
            tl.store(
                out_draft_tokens_ptr + output_offset,
                values,
                mask=mask,
            )


def draft_topk1(
    next_token_logits,
    positions,
    draft_tokens=None,
    draft_token_column=0,
):
    batch_size, vocab_size = next_token_logits.shape
    topk_p = torch.empty(
        (batch_size, 1),
        dtype=torch.float32,
        device=next_token_logits.device,
    )
    topk_index = torch.empty(
        (batch_size, 1),
        dtype=torch.int64,
        device=next_token_logits.device,
    )
    out_positions = torch.empty_like(positions)

    has_draft_tokens = draft_tokens is not None
    if has_draft_tokens:
        draft_size = draft_tokens.shape[1]
        normalized_column = draft_token_column
        if normalized_column < 0:
            normalized_column += draft_size
        out_draft_tokens = torch.empty_like(draft_tokens)
        if _INT64_DRAFT_IS_INT32_PAIRS and draft_tokens.dtype == torch.int64:
            draft_tokens_ptr = draft_tokens.view(torch.int32).reshape(-1)
            out_draft_tokens_ptr = out_draft_tokens.view(torch.int32).reshape(
                -1
            )
            stride_draft_batch = draft_tokens.stride(0) * 2
            stride_draft_column = draft_tokens.stride(1) * 2
            stride_out_draft_batch = out_draft_tokens.stride(0) * 2
            stride_out_draft_column = out_draft_tokens.stride(1) * 2
        else:
            draft_tokens_ptr = draft_tokens
            out_draft_tokens_ptr = out_draft_tokens
            stride_draft_batch = draft_tokens.stride(0)
            stride_draft_column = draft_tokens.stride(1)
            stride_out_draft_batch = out_draft_tokens.stride(0)
            stride_out_draft_column = out_draft_tokens.stride(1)
        block_draft = triton.next_power_of_2(draft_size)
    else:
        draft_size = 0
        normalized_column = 0
        out_draft_tokens = None
        draft_tokens_ptr = positions
        out_draft_tokens_ptr = out_positions
        stride_draft_batch = 0
        stride_draft_column = 0
        stride_out_draft_batch = 0
        stride_out_draft_column = 0
        block_draft = 1

    block_m = triton.next_power_of_2(
        triton.cdiv(batch_size, _XPU_CLUSTER_COUNT)
    )
    row_group_count = triton.cdiv(batch_size, block_m)
    if vocab_size <= _MAX_VOCAB_TILE:
        block_n = triton.next_power_of_2(vocab_size)
        _draft_topk1_argmax_2d_kernel[(row_group_count,)](
            next_token_logits,
            topk_index,
            next_token_logits.stride(0),
            next_token_logits.stride(1),
            BATCH_SIZE=batch_size,
            VOCAB_SIZE=vocab_size,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
        )
    else:
        partition_count = triton.cdiv(vocab_size, _MAX_VOCAB_TILE)
        partial_values = torch.empty(
            (batch_size, partition_count),
            dtype=torch.float32,
            device=next_token_logits.device,
        )
        partial_indices = torch.empty(
            (batch_size, partition_count),
            dtype=torch.int64,
            device=next_token_logits.device,
        )
        _draft_topk1_partial_2d_kernel[(row_group_count * partition_count,)](
            next_token_logits,
            partial_values,
            partial_indices,
            next_token_logits.stride(0),
            next_token_logits.stride(1),
            BATCH_SIZE=batch_size,
            VOCAB_SIZE=vocab_size,
            PARTITION_COUNT=partition_count,
            BLOCK_M=block_m,
            BLOCK_N=_MAX_VOCAB_TILE,
        )
        _draft_topk1_finalize_partitions_kernel[(row_group_count,)](
            partial_values,
            partial_indices,
            topk_index,
            BATCH_SIZE=batch_size,
            PARTITION_COUNT=partition_count,
            BLOCK_M=block_m,
        )

    topk_index_words = topk_index.view(torch.int32).reshape(-1)
    _draft_topk1_write_outputs_kernel[(batch_size,)](
        topk_index_words,
        positions,
        draft_tokens_ptr,
        topk_p,
        out_positions,
        out_draft_tokens_ptr,
        positions.stride(0),
        stride_draft_batch,
        stride_draft_column,
        stride_out_draft_batch,
        stride_out_draft_column,
        BATCH_SIZE=batch_size,
        DRAFT_SIZE=draft_size,
        BLOCK_DRAFT=block_draft,
        DRAFT_TOKEN_COLUMN=normalized_column,
        HAS_DRAFT_TOKENS=has_draft_tokens,
        INT64_DRAFT_IS_INT32_PAIRS=(
            _INT64_DRAFT_IS_INT32_PAIRS
            and has_draft_tokens
            and draft_tokens.dtype == torch.int64
        ),
    )

    return topk_p, topk_index, out_positions, out_draft_tokens


__all__ = ["draft_topk1"]
