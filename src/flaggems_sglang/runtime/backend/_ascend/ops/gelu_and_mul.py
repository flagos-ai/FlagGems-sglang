# Copyright 2026, The FlagOS Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License")
# you may not use this file except in compliance with the License
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Ascend isolated launch-policy probe for FlagOS Task 29.

This source is byte-derived from the 8/8-correct T29-CAND-007 Ascend route.
T29-CAND-027 changes only the maximum tile area from 512 elements to 1024 elements; exact-erf arithmetic, runtime-stride
addressing, shape-aware row packing, persistent grid-stride coverage, and the
one-argument wrapper remain unchanged. This local probe has not been compiled
or timed on Ascend hardware.

Copyright 2023-2024 SGLang Team
Copyright 2026 FlagOS Contributors
SPDX-License-Identifier: Apache-2.0
"""

import torch
import triton
import triton.language as tl

__all__ = ["gelu_and_mul"]


@triton.jit
def _gelu_and_mul_kernel(
    hidden_states_ptr,
    output_ptr,
    hidden_size,
    batch_size,
    input_stride_batch,
    input_stride_hidden,
    output_stride_batch,
    output_stride_hidden,
    num_ctas,
    tiles_per_cta,
    num_column_tiles,
    total_tiles,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_COLUMNS: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    lanes = tl.arange(0, BLOCK_ROWS * BLOCK_COLUMNS)
    rows_in_tile = lanes // BLOCK_COLUMNS
    columns_in_tile = lanes - rows_in_tile * BLOCK_COLUMNS

    for iteration in range(0, tiles_per_cta):
        tile_id = pid + iteration * num_ctas
        batch_tile = tile_id // num_column_tiles
        column_tile = tile_id - batch_tile * num_column_tiles
        batch_index = batch_tile * BLOCK_ROWS + rows_in_tile
        columns = column_tile * BLOCK_COLUMNS + columns_in_tile
        mask = (
            (tile_id < total_tiles)
            & (batch_index < batch_size)
            & (columns < hidden_size)
        )

        input_row = batch_index * input_stride_batch
        gate = tl.load(
            hidden_states_ptr + input_row + columns * input_stride_hidden,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        up = tl.load(
            hidden_states_ptr
            + input_row
            + (hidden_size + columns) * input_stride_hidden,
            mask=mask,
            other=0.0,
        ).to(tl.float32)

        activated = gate * (0.5 * (1.0 + tl.erf(gate * 0.7071067811865475)))
        output_offsets = (
            batch_index * output_stride_batch + columns * output_stride_hidden
        )
        tl.store(output_ptr + output_offsets, activated * up, mask=mask)


def gelu_and_mul(hidden_states):
    batch_size, doubled_hidden_size = hidden_states.shape
    hidden_size = doubled_hidden_size // 2
    output = torch.empty(
        (batch_size, hidden_size),
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )
    block_columns = min(1024, triton.next_power_of_2(hidden_size))
    block_rows = min(1024 // block_columns, triton.next_power_of_2(batch_size))
    num_column_tiles = triton.cdiv(hidden_size, block_columns)
    num_batch_tiles = triton.cdiv(batch_size, block_rows)
    num_tiles = num_batch_tiles * num_column_tiles
    num_ctas = min(num_tiles, 48)
    tiles_per_cta = triton.cdiv(num_tiles, num_ctas)
    grid = (num_ctas,)
    _gelu_and_mul_kernel[grid](
        hidden_states,
        output,
        hidden_size,
        batch_size,
        hidden_states.stride(0),
        hidden_states.stride(1),
        output.stride(0),
        output.stride(1),
        num_ctas,
        tiles_per_cta,
        num_column_tiles,
        num_tiles,
        block_rows,
        block_columns,
        num_warps=4,
    )
    return output
