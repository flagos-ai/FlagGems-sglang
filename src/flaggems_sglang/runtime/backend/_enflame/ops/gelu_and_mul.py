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

"""Enflame cap-12 CTA diagnostic for FlagOS Task 29.

This draft keeps the official-evaluated C009 GCU400 block-pointer/DMA kernel,
65,536-element right-axis-first tile, four-warp launch, exact-erf arithmetic,
and runtime-stride order.  Its only executable scheduling change is the
physical CTA cap: 48 becomes 12, the default Enflame GCU300 cap published by
FlagGems at commit bca7a6994a1750c04177dfeb8c24f900ffbd7d4a.

This is an architecture-policy diagnostic, not a claim that the unpublished
FlagOS Enflame evaluator is GCU300.  It has not been compiled or measured on
Enflame hardware.

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
    INPUT_ORDER_0: tl.constexpr,
    INPUT_ORDER_1: tl.constexpr,
    OUTPUT_ORDER_0: tl.constexpr,
    OUTPUT_ORDER_1: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_COLUMNS: tl.constexpr,
):
    pid = tl.program_id(axis=0)

    for iteration in range(0, tiles_per_cta):
        tile_id = pid + iteration * num_ctas
        batch_tile = tile_id // num_column_tiles
        column_tile = tile_id - batch_tile * num_column_tiles
        batch_offset = (batch_tile * BLOCK_ROWS).to(tl.int32)
        column_offset = (column_tile * BLOCK_COLUMNS).to(tl.int32)

        gate_block_ptr = tl.make_block_ptr(
            base=hidden_states_ptr,
            shape=(batch_size, hidden_size),
            strides=(input_stride_batch, input_stride_hidden),
            offsets=(batch_offset, column_offset),
            block_shape=(BLOCK_ROWS, BLOCK_COLUMNS),
            order=(INPUT_ORDER_0, INPUT_ORDER_1),
        )
        up_block_ptr = tl.make_block_ptr(
            base=hidden_states_ptr + hidden_size * input_stride_hidden,
            shape=(batch_size, hidden_size),
            strides=(input_stride_batch, input_stride_hidden),
            offsets=(batch_offset, column_offset),
            block_shape=(BLOCK_ROWS, BLOCK_COLUMNS),
            order=(INPUT_ORDER_0, INPUT_ORDER_1),
        )
        output_block_ptr = tl.make_block_ptr(
            base=output_ptr,
            shape=(batch_size, hidden_size),
            strides=(output_stride_batch, output_stride_hidden),
            offsets=(batch_offset, column_offset),
            block_shape=(BLOCK_ROWS, BLOCK_COLUMNS),
            order=(OUTPUT_ORDER_0, OUTPUT_ORDER_1),
        )

        gate = tl.load(
            gate_block_ptr,
            boundary_check=(INPUT_ORDER_0, INPUT_ORDER_1),
        ).to(tl.float32)
        up = tl.load(
            up_block_ptr,
            boundary_check=(INPUT_ORDER_0, INPUT_ORDER_1),
        ).to(tl.float32)
        activated = gate * (0.5 * (1.0 + tl.erf(gate * 0.7071067811865475)))
        tl.store(
            output_block_ptr,
            (activated * up).to(output_block_ptr.type.element_ty),
            boundary_check=(OUTPUT_ORDER_0, OUTPUT_ORDER_1),
        )


def gelu_and_mul(hidden_states):
    batch_size, doubled_hidden_size = hidden_states.shape
    hidden_size = doubled_hidden_size // 2
    output = torch.empty(
        (batch_size, hidden_size),
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )
    input_strides = (
        hidden_states.stride(0),
        hidden_states.stride(1),
    )
    output_strides = (
        output.stride(0),
        output.stride(1),
    )
    input_stride_order = tuple(
        sorted(range(2), key=lambda axis: abs(input_strides[axis]))
    )
    output_stride_order = tuple(
        sorted(range(2), key=lambda axis: abs(output_strides[axis]))
    )
    block_columns = min(65536, triton.next_power_of_2(hidden_size))
    block_rows = min(
        65536 // block_columns, triton.next_power_of_2(batch_size)
    )
    num_column_tiles = triton.cdiv(hidden_size, block_columns)
    num_batch_tiles = triton.cdiv(batch_size, block_rows)
    num_tiles = num_batch_tiles * num_column_tiles
    num_ctas = min(num_tiles, 12)
    tiles_per_cta = triton.cdiv(num_tiles, num_ctas)
    grid = (num_ctas,)
    _gelu_and_mul_kernel[grid](
        hidden_states,
        output,
        hidden_size,
        batch_size,
        input_strides[0],
        input_strides[1],
        output_strides[0],
        output_strides[1],
        num_ctas,
        tiles_per_cta,
        num_column_tiles,
        input_stride_order[0],
        input_stride_order[1],
        output_stride_order[0],
        output_stride_order[1],
        block_rows,
        block_columns,
        num_warps=4,
    )
    return output
