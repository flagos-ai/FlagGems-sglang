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
def _fused_dot_kernel(
    x,
    weights,
    seg_indptr,
    weight_indices,
    lora_ranks,
    scalings,
    permutation,
    base_output,
    output,
    rank: tl.constexpr,
    output_dim: tl.constexpr,
    max_segment_length: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    block_k: tl.constexpr,
    has_permutation: tl.constexpr,
):
    segment_tile = tl.program_id(0)
    output_tile = tl.program_id(1)
    row_tiles: tl.constexpr = tl.cdiv(max_segment_length, block_m)
    segment = segment_tile // row_tiles
    row_tile = segment_tile - segment * row_tiles
    segment_start = tl.load(seg_indptr + segment)
    segment_end = tl.load(seg_indptr + segment + 1)
    adapter = tl.load(weight_indices + segment)
    positions = segment_start + row_tile * block_m + tl.arange(0, block_m)
    if has_permutation:
        rows = tl.load(
            permutation + positions, mask=positions < segment_end, other=0
        )
    else:
        rows = positions
    local_columns = output_tile * block_n + tl.arange(0, block_n)
    mask_m = positions < segment_end
    mask_n = local_columns < output_dim
    gate_accumulator = tl.zeros((block_m, block_n), dtype=tl.float32)
    up_accumulator = tl.zeros((block_m, block_n), dtype=tl.float32)
    for offset_k in range(0, rank, block_k):
        offsets_k = offset_k + tl.arange(0, block_k)
        mask_k = offsets_k < rank
        gate_x = tl.load(
            x + rows[:, None] * (2 * rank) + offsets_k[None, :],
            mask=mask_m[:, None] & mask_k[None, :],
            other=0.0,
        )
        up_x = tl.load(
            x + rows[:, None] * (2 * rank) + rank + offsets_k[None, :],
            mask=mask_m[:, None] & mask_k[None, :],
            other=0.0,
        )
        gate_weights = tl.load(
            weights
            + adapter * (2 * output_dim * rank)
            + local_columns[None, :] * rank
            + offsets_k[:, None],
            mask=mask_k[:, None] & mask_n[None, :],
            other=0.0,
        )
        up_weights = tl.load(
            weights
            + adapter * (2 * output_dim * rank)
            + (output_dim + local_columns[None, :]) * rank
            + offsets_k[:, None],
            mask=mask_k[:, None] & mask_n[None, :],
            other=0.0,
        )
        gate_accumulator += tl.dot(gate_x, gate_weights)
        up_accumulator += tl.dot(up_x, up_weights)
    active = tl.load(lora_ranks + adapter) != 0
    scaling = tl.where(active, tl.load(scalings + adapter), 0.0)
    gate_offsets = rows[:, None] * (2 * output_dim) + local_columns[None, :]
    up_offsets = gate_offsets + output_dim
    output_mask = mask_m[:, None] & mask_n[None, :]
    gate_base = tl.load(
        base_output + gate_offsets, mask=output_mask, other=0.0
    ).to(tl.float32)
    up_base = tl.load(
        base_output + up_offsets, mask=output_mask, other=0.0
    ).to(tl.float32)
    tl.store(
        output + gate_offsets,
        gate_base + scaling * gate_accumulator,
        mask=output_mask,
    )
    tl.store(
        output + up_offsets,
        up_base + scaling * up_accumulator,
        mask=output_mask,
    )


@triton.jit
def _serial_kernel(
    x,
    weights,
    seg_indptr,
    weight_indices,
    lora_ranks,
    scalings,
    permutation,
    base_output,
    output,
    rank: tl.constexpr,
    output_dim: tl.constexpr,
    max_segment_length: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    has_permutation: tl.constexpr,
):
    segment_tile = tl.program_id(0)
    output_tile = tl.program_id(1)
    row_tiles: tl.constexpr = tl.cdiv(max_segment_length, block_m)
    output_tiles: tl.constexpr = tl.cdiv(output_dim, block_n)
    segment = segment_tile // row_tiles
    row_tile = segment_tile - segment * row_tiles
    slice_index = output_tile // output_tiles
    local_output_tile = output_tile - slice_index * output_tiles
    segment_start = tl.load(seg_indptr + segment)
    segment_end = tl.load(seg_indptr + segment + 1)
    adapter = tl.load(weight_indices + segment)
    active = tl.load(lora_ranks + adapter) != 0
    scaling = tl.where(active, tl.load(scalings + adapter), 0.0)
    positions = segment_start + row_tile * block_m + tl.arange(0, block_m)
    local_columns = local_output_tile * block_n + tl.arange(0, block_n)
    columns = slice_index * output_dim + local_columns
    mask_m = positions < segment_end
    compute_m = mask_m & active
    mask_n = local_columns < output_dim
    if has_permutation:
        rows = tl.load(permutation + positions, mask=mask_m, other=0)
    else:
        rows = positions
    accumulator = tl.zeros((block_m, block_n), dtype=tl.float32)
    for rank_offset in tl.static_range(0, rank):
        x_values = tl.load(
            x + rows * (2 * rank) + slice_index * rank + rank_offset,
            mask=compute_m,
            other=0.0,
        ).to(tl.float32)
        weight_values = tl.load(
            weights
            + adapter * (2 * output_dim * rank)
            + columns * rank
            + rank_offset,
            mask=mask_n,
            other=0.0,
        ).to(tl.float32)
        accumulator += x_values[:, None] * weight_values[None, :]
    output_offsets = rows[:, None] * (2 * output_dim) + columns[None, :]
    output_mask = mask_m[:, None] & mask_n[None, :]
    base = tl.load(
        base_output + output_offsets, mask=output_mask, other=0.0
    ).to(tl.float32)
    tl.store(
        output + output_offsets, base + scaling * accumulator, mask=output_mask
    )


def _launch_dot(
    x,
    weights,
    batch_info,
    output_dim,
    base_output,
    block_m,
    block_n,
    num_stages,
):
    rank = weights.shape[-1]
    max_segment_length = max(1, batch_info.max_len)
    output = torch.empty_like(base_output)
    permutation = batch_info.permutation
    _fused_dot_kernel[
        (
            batch_info.bs * triton.cdiv(max_segment_length, block_m),
            triton.cdiv(output_dim, block_n),
        )
    ](
        x,
        weights,
        batch_info.seg_indptr,
        batch_info.weight_indices,
        batch_info.lora_ranks,
        batch_info.scalings,
        x if permutation is None else permutation,
        base_output,
        output,
        rank,
        output_dim,
        max_segment_length,
        block_m,
        block_n,
        32,
        has_permutation=permutation is not None,
        num_warps=4,
        num_stages=num_stages,
    )
    return output


def _launch_serial(x, weights, batch_info, output_dim, base_output):
    rank = weights.shape[-1]
    max_segment_length = max(1, batch_info.max_len)
    block_m = 16
    block_n = 64
    row_tiles = triton.cdiv(max_segment_length, block_m)
    output_tiles = triton.cdiv(output_dim, block_n)
    output = torch.empty_like(base_output)
    permutation = batch_info.permutation
    _serial_kernel[(batch_info.bs * row_tiles, 2 * output_tiles)](
        x,
        weights,
        batch_info.seg_indptr,
        batch_info.weight_indices,
        batch_info.lora_ranks,
        batch_info.scalings,
        x if permutation is None else permutation,
        base_output,
        output,
        rank,
        output_dim,
        max_segment_length,
        block_m,
        block_n,
        has_permutation=permutation is not None,
        num_warps=4,
        num_stages=1,
    )
    return output


def gate_up_lora_b(x, gate_up_lora_b, batch_info, output_dim, base_output):
    rank = gate_up_lora_b.shape[-1]
    if (
        x.element_size() == 2
        and (rank in (32, 64) or (rank == 16 and x.dtype == torch.float16))
        and batch_info.permutation is None
    ):
        max_segment_length = batch_info.max_len
        if rank == 16:
            if max_segment_length <= 16:
                block_n = 64 if output_dim <= 64 else 128
                return _launch_dot(
                    x,
                    gate_up_lora_b,
                    batch_info,
                    output_dim,
                    base_output,
                    16,
                    block_n,
                    1,
                )
            if max_segment_length <= 64:
                return _launch_dot(
                    x,
                    gate_up_lora_b,
                    batch_info,
                    output_dim,
                    base_output,
                    64,
                    256,
                    2,
                )
            return _launch_dot(
                x,
                gate_up_lora_b,
                batch_info,
                output_dim,
                base_output,
                128,
                128,
                2,
            )
        if rank == 32:
            if max_segment_length <= 64:
                return _launch_dot(
                    x,
                    gate_up_lora_b,
                    batch_info,
                    output_dim,
                    base_output,
                    64,
                    512,
                    2,
                )
            if max_segment_length <= 128:
                return _launch_dot(
                    x,
                    gate_up_lora_b,
                    batch_info,
                    output_dim,
                    base_output,
                    128,
                    256,
                    2,
                )
            return _launch_dot(
                x,
                gate_up_lora_b,
                batch_info,
                output_dim,
                base_output,
                256,
                128,
                2,
            )
        if max_segment_length <= 64:
            return _launch_dot(
                x,
                gate_up_lora_b,
                batch_info,
                output_dim,
                base_output,
                64,
                256,
                2,
            )
        if max_segment_length <= 128:
            return _launch_dot(
                x,
                gate_up_lora_b,
                batch_info,
                output_dim,
                base_output,
                128,
                128,
                2,
            )
        return _launch_dot(
            x,
            gate_up_lora_b,
            batch_info,
            output_dim,
            base_output,
            256,
            128,
            2,
        )
    return _launch_serial(
        x, gate_up_lora_b, batch_info, output_dim, base_output
    )


__all__ = ["gate_up_lora_b"]
