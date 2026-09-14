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

_PARTITION_SIZE = 4096
_STREAMING_LIMIT = 32768
_STREAMING_TILE = 4096
_INT64_STORAGE_IS_INT32 = False


@triton.jit
def _write_outputs(
    row,
    selected_index,
    positions_ptr,
    draft_tokens_ptr,
    topk_p_ptr,
    topk_index_ptr,
    out_positions_ptr,
    out_draft_tokens_ptr,
    stride_positions,
    stride_draft_batch,
    stride_draft_column,
    stride_out_draft_batch,
    stride_out_draft_column,
    DRAFT_SIZE: tl.constexpr,
    BLOCK_DRAFT: tl.constexpr,
    DRAFT_TOKEN_COLUMN: tl.constexpr,
    HAS_DRAFT_TOKENS: tl.constexpr,
):
    tl.store(topk_p_ptr + row, 1.0)
    tl.store(topk_index_ptr + row, selected_index)
    position = tl.load(positions_ptr + row * stride_positions)
    tl.store(out_positions_ptr + row, position + 1)

    if HAS_DRAFT_TOKENS:
        columns = tl.arange(0, BLOCK_DRAFT)
        mask = columns < DRAFT_SIZE
        values = tl.load(
            draft_tokens_ptr
            + row * stride_draft_batch
            + columns * stride_draft_column,
            mask=mask,
            other=0,
        )
        values = tl.where(
            columns == DRAFT_TOKEN_COLUMN,
            selected_index,
            values,
        )
        tl.store(
            out_draft_tokens_ptr
            + row * stride_out_draft_batch
            + columns * stride_out_draft_column,
            values,
            mask=mask,
        )


@triton.jit
def _draft_topk1_streaming_kernel(
    logits_ptr,
    positions_ptr,
    draft_tokens_ptr,
    topk_p_ptr,
    topk_index_ptr,
    out_positions_ptr,
    out_draft_tokens_ptr,
    stride_positions,
    stride_draft_batch,
    stride_draft_column,
    stride_out_draft_batch,
    stride_out_draft_column,
    stride_logits_batch,
    stride_logits_vocab,
    VOCAB_SIZE: tl.constexpr,
    BLOCK_N: tl.constexpr,
    DRAFT_SIZE: tl.constexpr,
    BLOCK_DRAFT: tl.constexpr,
    DRAFT_TOKEN_COLUMN: tl.constexpr,
    HAS_DRAFT_TOKENS: tl.constexpr,
):
    row = tl.program_id(0)
    maximum = tl.full((), -float("inf"), tl.float32)
    selected_index = tl.zeros((), tl.int64)
    for start_n in range(0, VOCAB_SIZE, BLOCK_N):
        column = start_n + tl.arange(0, BLOCK_N)
        values = tl.load(
            logits_ptr
            + row * stride_logits_batch
            + column * stride_logits_vocab,
            mask=column < VOCAB_SIZE,
            other=-float("inf"),
        ).to(tl.float32)
        local_maximum, local_index = tl.max(
            values,
            axis=0,
            return_indices=True,
            return_indices_tie_break_left=True,
        )
        update = local_maximum > maximum
        maximum = tl.where(update, local_maximum, maximum)
        selected_index = tl.where(
            update,
            start_n + local_index,
            selected_index,
        )
    _write_outputs(
        row,
        selected_index,
        positions_ptr,
        draft_tokens_ptr,
        topk_p_ptr,
        topk_index_ptr,
        out_positions_ptr,
        out_draft_tokens_ptr,
        stride_positions,
        stride_draft_batch,
        stride_draft_column,
        stride_out_draft_batch,
        stride_out_draft_column,
        DRAFT_SIZE,
        BLOCK_DRAFT,
        DRAFT_TOKEN_COLUMN,
        HAS_DRAFT_TOKENS,
    )


@triton.jit
def _draft_topk1_direct_kernel(
    logits_ptr,
    positions_ptr,
    draft_tokens_ptr,
    topk_p_ptr,
    topk_index_ptr,
    out_positions_ptr,
    out_draft_tokens_ptr,
    stride_positions,
    stride_draft_batch,
    stride_draft_column,
    stride_out_draft_batch,
    stride_out_draft_column,
    stride_logits_batch,
    stride_logits_vocab,
    VOCAB_SIZE: tl.constexpr,
    BLOCK_VOCAB: tl.constexpr,
    DRAFT_SIZE: tl.constexpr,
    BLOCK_DRAFT: tl.constexpr,
    DRAFT_TOKEN_COLUMN: tl.constexpr,
    HAS_DRAFT_TOKENS: tl.constexpr,
):
    row = tl.program_id(0)
    columns = tl.arange(0, BLOCK_VOCAB)
    logits = tl.load(
        logits_ptr + row * stride_logits_batch + columns * stride_logits_vocab,
        mask=columns < VOCAB_SIZE,
        other=-float("inf"),
    ).to(tl.float32)
    selected_index = tl.argmax(
        logits,
        axis=0,
        tie_break_left=True,
    )
    _write_outputs(
        row,
        selected_index,
        positions_ptr,
        draft_tokens_ptr,
        topk_p_ptr,
        topk_index_ptr,
        out_positions_ptr,
        out_draft_tokens_ptr,
        stride_positions,
        stride_draft_batch,
        stride_draft_column,
        stride_out_draft_batch,
        stride_out_draft_column,
        DRAFT_SIZE,
        BLOCK_DRAFT,
        DRAFT_TOKEN_COLUMN,
        HAS_DRAFT_TOKENS,
    )


@triton.jit
def _draft_topk1_partial_kernel(
    logits_ptr,
    partial_values_ptr,
    partial_indices_ptr,
    stride_logits_batch,
    stride_logits_vocab,
    VOCAB_SIZE: tl.constexpr,
    PARTITION_COUNT: tl.constexpr,
    BLOCK_VOCAB: tl.constexpr,
):
    program = tl.program_id(0)
    row = program // PARTITION_COUNT
    partition = program - row * PARTITION_COUNT
    columns = partition * BLOCK_VOCAB + tl.arange(0, BLOCK_VOCAB)
    mask = columns < VOCAB_SIZE
    logits = tl.load(
        logits_ptr + row * stride_logits_batch + columns * stride_logits_vocab,
        mask=mask,
        other=-float("inf"),
    ).to(tl.float32)
    local_index = tl.argmax(
        logits,
        axis=0,
        tie_break_left=True,
    )
    selected_index = partition * BLOCK_VOCAB + local_index
    selected_value = tl.load(
        logits_ptr
        + row * stride_logits_batch
        + selected_index * stride_logits_vocab,
    ).to(tl.float32)
    partial_offset = row * PARTITION_COUNT + partition
    tl.store(partial_values_ptr + partial_offset, selected_value)
    tl.store(partial_indices_ptr + partial_offset, selected_index)


@triton.jit
def _draft_topk1_finalize_kernel(
    partial_values_ptr,
    partial_indices_ptr,
    positions_ptr,
    draft_tokens_ptr,
    topk_p_ptr,
    topk_index_ptr,
    out_positions_ptr,
    out_draft_tokens_ptr,
    stride_positions,
    stride_draft_batch,
    stride_draft_column,
    stride_out_draft_batch,
    stride_out_draft_column,
    PARTITION_COUNT: tl.constexpr,
    BLOCK_PARTITIONS: tl.constexpr,
    DRAFT_SIZE: tl.constexpr,
    BLOCK_DRAFT: tl.constexpr,
    DRAFT_TOKEN_COLUMN: tl.constexpr,
    HAS_DRAFT_TOKENS: tl.constexpr,
):
    row = tl.program_id(0)
    partitions = tl.arange(0, BLOCK_PARTITIONS)
    partial_values = tl.load(
        partial_values_ptr + row * PARTITION_COUNT + partitions,
        mask=partitions < PARTITION_COUNT,
        other=-float("inf"),
    )
    selected_partition = tl.argmax(
        partial_values,
        axis=0,
        tie_break_left=True,
    )
    selected_index = tl.load(
        partial_indices_ptr + row * PARTITION_COUNT + selected_partition,
    )
    _write_outputs(
        row,
        selected_index,
        positions_ptr,
        draft_tokens_ptr,
        topk_p_ptr,
        topk_index_ptr,
        out_positions_ptr,
        out_draft_tokens_ptr,
        stride_positions,
        stride_draft_batch,
        stride_draft_column,
        stride_out_draft_batch,
        stride_out_draft_column,
        DRAFT_SIZE,
        BLOCK_DRAFT,
        DRAFT_TOKEN_COLUMN,
        HAS_DRAFT_TOKENS,
    )


def chunk_topk1_outputs(
    next_token_logits, positions, draft_tokens, draft_token_column
):
    batch_size, vocab_size = next_token_logits.shape
    topk_p = torch.empty(
        (batch_size, 1),
        dtype=torch.float32,
        device=next_token_logits.device,
    )
    integer_outputs = torch.empty(
        (2, batch_size),
        dtype=torch.int64,
        device=next_token_logits.device,
    )
    topk_index = integer_outputs[0].view(batch_size, 1)
    out_positions = integer_outputs[1]
    positions_ptr = positions
    topk_index_ptr = topk_index
    out_positions_ptr = out_positions
    stride_positions = positions.stride(0)
    if _INT64_STORAGE_IS_INT32:
        positions_ptr = positions.view(torch.int32).reshape(-1)
        integer_kernel_storage = integer_outputs.view(torch.int32).reshape(-1)
        topk_index_ptr = integer_kernel_storage[:batch_size]
        out_positions_ptr = integer_kernel_storage[batch_size : 2 * batch_size]
        stride_positions = 1

    has_draft_tokens = draft_tokens is not None
    if has_draft_tokens:
        draft_size = draft_tokens.shape[1]
        normalized_column = draft_token_column
        if normalized_column < 0:
            normalized_column += draft_size
        out_draft_tokens = torch.empty_like(draft_tokens)
        if _INT64_STORAGE_IS_INT32 and draft_tokens.dtype == torch.int64:
            draft_tokens_ptr = draft_tokens.view(torch.int32).reshape(-1)
            out_draft_tokens_ptr = out_draft_tokens.view(torch.int32).reshape(
                -1
            )
            stride_draft_batch = draft_size
            stride_draft_column = 1
            stride_out_draft_batch = draft_size
            stride_out_draft_column = 1
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
        draft_tokens_ptr = positions_ptr
        out_draft_tokens_ptr = out_positions_ptr
        stride_draft_batch = 0
        stride_draft_column = 0
        stride_out_draft_batch = 0
        stride_out_draft_column = 0
        block_draft = 1

    output_args = (
        positions_ptr,
        draft_tokens_ptr,
        topk_p,
        topk_index_ptr,
        out_positions_ptr,
        out_draft_tokens_ptr,
        stride_positions,
        stride_draft_batch,
        stride_draft_column,
        stride_out_draft_batch,
        stride_out_draft_column,
    )
    output_meta = {
        "DRAFT_SIZE": draft_size,
        "BLOCK_DRAFT": block_draft,
        "DRAFT_TOKEN_COLUMN": normalized_column,
        "HAS_DRAFT_TOKENS": has_draft_tokens,
    }
    return (
        topk_p,
        topk_index,
        out_positions,
        out_draft_tokens,
        output_args,
        output_meta,
        batch_size,
        vocab_size,
    )


def draft_topk1(
    next_token_logits,
    positions,
    draft_tokens=None,
    draft_token_column=0,
):
    (
        topk_p,
        topk_index,
        out_positions,
        out_draft_tokens,
        output_args,
        output_meta,
        batch_size,
        vocab_size,
    ) = chunk_topk1_outputs(
        next_token_logits,
        positions,
        draft_tokens,
        draft_token_column,
    )

    if vocab_size <= _STREAMING_LIMIT:
        block_n = min(
            _STREAMING_TILE,
            triton.next_power_of_2(vocab_size),
        )
        _draft_topk1_streaming_kernel[(batch_size,)](
            next_token_logits,
            *output_args,
            next_token_logits.stride(0),
            next_token_logits.stride(1),
            VOCAB_SIZE=vocab_size,
            BLOCK_N=block_n,
            **output_meta,
            num_warps=8 if block_n >= 2048 else 4,
            num_stages=1,
        )
        return topk_p, topk_index, out_positions, out_draft_tokens

    if vocab_size <= _PARTITION_SIZE:
        block_vocab = triton.next_power_of_2(vocab_size)
        _draft_topk1_direct_kernel[(batch_size,)](
            next_token_logits,
            *output_args,
            next_token_logits.stride(0),
            next_token_logits.stride(1),
            VOCAB_SIZE=vocab_size,
            BLOCK_VOCAB=block_vocab,
            **output_meta,
            num_warps=4,
            num_stages=1,
        )
    else:
        partition_count = triton.cdiv(vocab_size, _PARTITION_SIZE)
        partial_values = torch.empty(
            (batch_size, partition_count),
            dtype=torch.float32,
            device=next_token_logits.device,
        )
        partial_indices = torch.empty(
            (batch_size, partition_count),
            dtype=torch.int32,
            device=next_token_logits.device,
        )
        _draft_topk1_partial_kernel[(batch_size * partition_count,)](
            next_token_logits,
            partial_values,
            partial_indices,
            next_token_logits.stride(0),
            next_token_logits.stride(1),
            VOCAB_SIZE=vocab_size,
            PARTITION_COUNT=partition_count,
            BLOCK_VOCAB=_PARTITION_SIZE,
            num_warps=4,
            num_stages=1,
        )
        _draft_topk1_finalize_kernel[(batch_size,)](
            partial_values,
            partial_indices,
            *output_args,
            PARTITION_COUNT=partition_count,
            BLOCK_PARTITIONS=triton.next_power_of_2(partition_count),
            **output_meta,
            num_warps=1,
            num_stages=1,
        )

    return topk_p, topk_index, out_positions, out_draft_tokens


__all__ = ["draft_topk1"]
