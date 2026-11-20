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
def _grouped_gemma_rmsnorm_rows_kernel(
    x_ptr,
    weight_ptr,
    out_ptr,
    GROUP_SIZE: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    EPS: tl.constexpr,
    NUM_UNITS: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
):
    cols = tl.arange(0, GROUP_SIZE)
    row_offsets = tl.arange(0, BLOCK_ROWS)
    group_offsets = row_offsets[:, None] * GROUP_SIZE + cols[None, :]
    weight_values = tl.load(
        weight_ptr
        + (row_offsets % NUM_GROUPS)[:, None] * GROUP_SIZE
        + cols[None, :]
    ).to(tl.float32)
    for block_index in tl.range(
        tl.program_id(0),
        NUM_UNITS // BLOCK_ROWS,
        tl.num_programs(0),
        num_stages=2,
    ):
        base = block_index.to(tl.int64) * (BLOCK_ROWS * GROUP_SIZE)
        x = tl.load(
            x_ptr + base + group_offsets, offset_state_policy="continuous"
        ).to(tl.float32)
        rstd = tl.rsqrt(tl.sum(x * x, 1) / GROUP_SIZE + EPS)
        tl.store(
            out_ptr + base + group_offsets,
            x * rstd[:, None] * (1.0 + weight_values),
            offset_state_policy="continuous",
        )


@triton.jit
def _grouped_gemma_rmsnorm_chunked_rows_kernel(
    x_ptr,
    weight_ptr,
    out_ptr,
    CHUNK_SIZE: tl.constexpr,
    NUM_CHUNKS: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    EPS: tl.constexpr,
    NUM_UNITS: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
):
    cols = tl.arange(0, CHUNK_SIZE)
    row_offsets = tl.arange(0, BLOCK_ROWS)
    group_offsets = (
        row_offsets[:, None] * (CHUNK_SIZE * NUM_CHUNKS) + cols[None, :]
    )
    weight_offsets = (row_offsets % NUM_GROUPS)[:, None] * (
        CHUNK_SIZE * NUM_CHUNKS
    ) + cols[None, :]
    for block_index in tl.range(
        tl.program_id(0),
        NUM_UNITS // BLOCK_ROWS,
        tl.num_programs(0),
        num_stages=2,
    ):
        base = block_index.to(tl.int64) * (
            BLOCK_ROWS * CHUNK_SIZE * NUM_CHUNKS
        )
        accumulator = tl.zeros((BLOCK_ROWS, CHUNK_SIZE), tl.float32)
        for index in tl.static_range(NUM_CHUNKS):
            value = tl.load(
                x_ptr + base + group_offsets + index * CHUNK_SIZE
            ).to(tl.float32)
            accumulator += value * value
        rstd = tl.rsqrt(
            tl.sum(accumulator, 1) / (CHUNK_SIZE * NUM_CHUNKS) + EPS
        )
        for index in tl.static_range(NUM_CHUNKS):
            value = tl.load(
                x_ptr + base + group_offsets + index * CHUNK_SIZE
            ).to(tl.float32)
            weight_values = tl.load(
                weight_ptr + weight_offsets + index * CHUNK_SIZE
            ).to(tl.float32)
            tl.store(
                out_ptr + base + group_offsets + index * CHUNK_SIZE,
                value * rstd[:, None] * (1.0 + weight_values),
            )


@triton.jit
def _grouped_gemma_rmsnorm_single_unit_kernel(
    x_ptr,
    weight_ptr,
    out_ptr,
    GROUP_SIZE: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    EPS: tl.constexpr,
    NUM_UNITS: tl.constexpr,
):
    cols = tl.arange(0, GROUP_SIZE)
    for unit_index in tl.range(
        tl.program_id(0), NUM_UNITS, tl.num_programs(0), num_stages=2
    ):
        base = unit_index.to(tl.int64) * GROUP_SIZE
        x = tl.load(x_ptr + base + cols, offset_state_policy="continuous").to(
            tl.float32
        )
        weight_values = tl.load(
            weight_ptr + (unit_index % NUM_GROUPS) * GROUP_SIZE + cols,
            offset_state_policy="continuous",
        ).to(tl.float32)
        rstd = tl.rsqrt(tl.sum(x * x, 0) / GROUP_SIZE + EPS)
        tl.store(
            out_ptr + base + cols,
            x * rstd * (1.0 + weight_values),
            offset_state_policy="continuous",
        )


@triton.jit
def _grouped_gemma_rmsnorm_single_row_chunks_kernel(
    x_ptr,
    weight_ptr,
    out_ptr,
    CHUNK_SIZE: tl.constexpr,
    NUM_CHUNKS: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    EPS: tl.constexpr,
    NUM_UNITS: tl.constexpr,
):
    cols = tl.arange(0, CHUNK_SIZE)
    for unit_index in tl.range(
        tl.program_id(0), NUM_UNITS, tl.num_programs(0), num_stages=2
    ):
        base = unit_index.to(tl.int64) * (CHUNK_SIZE * NUM_CHUNKS)
        group_offset = (unit_index % NUM_GROUPS) * (CHUNK_SIZE * NUM_CHUNKS)
        accumulator = tl.zeros((CHUNK_SIZE,), tl.float32)
        for index in tl.static_range(NUM_CHUNKS):
            value = tl.load(
                x_ptr + base + index * CHUNK_SIZE + cols,
                offset_state_policy="continuous",
            ).to(tl.float32)
            accumulator += value * value
        rstd = tl.rsqrt(
            tl.sum(accumulator, 0) / (CHUNK_SIZE * NUM_CHUNKS) + EPS
        )
        for index in tl.static_range(NUM_CHUNKS):
            value = tl.load(
                x_ptr + base + index * CHUNK_SIZE + cols,
                offset_state_policy="continuous",
            ).to(tl.float32)
            weight_values = tl.load(
                weight_ptr + group_offset + index * CHUNK_SIZE + cols,
                offset_state_policy="continuous",
            ).to(tl.float32)
            tl.store(
                out_ptr + base + index * CHUNK_SIZE + cols,
                value * rstd * (1.0 + weight_values),
                offset_state_policy="continuous",
            )


def grouped_gemma_rmsnorm(
    x,
    weight,
    group_size,
    eps=1e-6,
):
    out = torch.empty_like(x)
    units = x.numel() // group_size
    groups = x.shape[-1] // group_size
    chunk = group_size & -group_size
    parts = group_size // chunk
    budget = max(1, 131072 // (group_size * x.element_size()))
    span = 1 << budget.bit_length() - 1
    if parts == 1:
        cap = min(64, span)
    elif x.dtype == torch.float16:
        cap = 64
    else:
        cap = min(16, span)
    rows = min(cap, units & -units)
    aligned = rows > 1 and not groups & groups - 1 and groups <= rows
    if parts == 1:
        if aligned:
            _grouped_gemma_rmsnorm_rows_kernel.run(
                x,
                weight,
                out,
                group_size,
                groups,
                eps,
                units,
                rows,
                groups_per_cluster=rows,
                isCloseUnrollControl=False,
                enable_fp_fusion=False,
                isCloseMemoryAsync=False,
                grid=(min(12, units // rows),),
                warmup=False,
            )
            return out
        _grouped_gemma_rmsnorm_single_unit_kernel.run(
            x,
            weight,
            out,
            group_size,
            groups,
            eps,
            units,
            enable_fp_fusion=False,
            isCloseMemoryAsync=False,
            grid=(min(12, units),),
            warmup=False,
        )
        return out
    if aligned:
        _grouped_gemma_rmsnorm_chunked_rows_kernel.run(
            x,
            weight,
            out,
            chunk,
            parts,
            groups,
            eps,
            units,
            rows,
            groups_per_cluster=rows,
            isCloseUnrollControl=False,
            enable_fp_fusion=False,
            isCloseMemoryAsync=False,
            grid=(min(12, units // rows),),
            warmup=False,
        )
        return out
    _grouped_gemma_rmsnorm_single_row_chunks_kernel.run(
        x,
        weight,
        out,
        chunk,
        parts,
        groups,
        eps,
        units,
        enable_fp_fusion=False,
        isCloseMemoryAsync=False,
        grid=(min(12, units),),
        warmup=False,
    )
    return out


__all__ = ["grouped_gemma_rmsnorm"]
