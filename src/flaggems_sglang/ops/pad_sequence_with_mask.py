# Copyright 2026, The FlagOS Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Triton implementation of attention/pad_sequence_with_mask."""

import torch
import triton
import triton.language as tl

__all__ = ["pad_sequence_with_mask"]

# A row tile of 64 and a hidden tile of 128 keep one program's working set at
# 8K elements. Upstream sizes its tile as ``next_power_of_2(max_len)`` by
# ``next_power_of_2(hidden)``, which at max_len=2048, hidden=4096 is 8.4M
# elements in a single program -- far past what can stay in registers.
BLOCK_M = 64
BLOCK_D = 128


@triton.jit
def _pad_sequence_with_mask_kernel(
    input_ptr,
    offsets_ptr,
    lengths_ptr,
    out_ptr,
    mask_ptr,
    max_len,
    hidden,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_m = tl.program_id(0)
    batch = tl.program_id(1)
    pid_d = tl.program_id(2)

    start = tl.load(offsets_ptr + batch)
    length = tl.load(lengths_ptr + batch)

    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)

    row_ok = rows < max_len
    col_ok = cols < hidden
    gather = row_ok[:, None] & (rows < length)[:, None] & col_ok[None, :]

    src = input_ptr + (start + rows)[:, None] * hidden + cols[None, :]
    dst = out_ptr + (batch * max_len + rows)[:, None] * hidden + cols[None, :]

    values = tl.load(src, mask=gather, other=0.0)
    # Writes every in-range element, so the pad band is zeroed rather than
    # left as allocator garbage -- the output is a fresh tensor.
    tl.store(dst, values, mask=row_ok[:, None] & col_ok[None, :])

    # One hidden tile per row tile owns the mask write, which covers the whole
    # padded row exactly once.
    if pid_d == 0:
        tl.store(mask_ptr + batch * max_len + rows, rows < length, mask=row_ok)


def pad_sequence_with_mask(input_emb, offsets, lengths, max_len):
    """Gather per-sequence rows from ``input_emb``, zero-padding to ``max_len``.

    Args:
        input_emb: ``(total_tokens, hidden)`` packed token rows.
        offsets: ``(batch,)`` index of each sequence's first row.
        lengths: ``(batch,)`` number of valid rows in each sequence.
        max_len: padded sequence length.

    Returns:
        ``(batch_size, padded, valid_mask)`` where ``padded`` is
        ``(batch, max_len, hidden)`` and ``valid_mask`` is a flattened
        ``(batch * max_len,)`` boolean.
    """
    if input_emb.ndim != 2:
        raise ValueError("input_emb must be 2D (total_tokens, hidden)")
    if offsets.ndim != 1 or lengths.ndim != 1:
        raise ValueError("offsets and lengths must be 1D")
    if offsets.shape[0] != lengths.shape[0]:
        raise ValueError("offsets and lengths must have the same length")
    if offsets.dtype not in (torch.int32, torch.int64):
        raise ValueError("offsets must be int32 or int64")
    if lengths.dtype not in (torch.int32, torch.int64):
        raise ValueError("lengths must be int32 or int64")
    if (
        offsets.device != input_emb.device
        or lengths.device != input_emb.device
    ):
        raise ValueError("all inputs must be on the same device")
    if max_len < 0:
        raise ValueError("max_len must be non-negative")

    batch_size = offsets.shape[0]
    hidden = input_emb.shape[1]

    padded = torch.empty(
        (batch_size, max_len, hidden),
        dtype=input_emb.dtype,
        device=input_emb.device,
    )
    valid_mask = torch.empty(
        (batch_size * max_len,),
        dtype=torch.bool,
        device=input_emb.device,
    )

    # An empty grid dimension would launch nothing and leave the outputs
    # uninitialised, so short-circuit. ``hidden == 0`` still has a populated
    # mask -- the sequences are as long as they are, there is just no
    # hidden data to carry -- so it cannot share the empty-output path.
    if batch_size == 0 or max_len == 0:
        return batch_size, padded, valid_mask

    if hidden == 0:
        positions = torch.arange(max_len, device=input_emb.device)
        valid_mask = (positions.unsqueeze(0) < lengths.unsqueeze(1)).reshape(
            -1
        )
        return batch_size, padded, valid_mask

    grid = (
        triton.cdiv(max_len, BLOCK_M),
        batch_size,
        triton.cdiv(hidden, BLOCK_D),
    )
    _pad_sequence_with_mask_kernel[grid](
        input_emb,
        offsets,
        lengths,
        padded,
        valid_mask,
        max_len,
        hidden,
        BLOCK_M=BLOCK_M,
        BLOCK_D=BLOCK_D,
    )

    return batch_size, padded, valid_mask
