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
def _xpu_prepare_x(
    x,
    seg_indptr,
    weight_indices,
    lora_ranks,
    expanded_x,
    total_elements,
    rank: tl.constexpr,
    max_segment_length: tl.constexpr,
    block_size: tl.constexpr,
):
    offsets = tl.program_id(0) * block_size + tl.arange(0, block_size)
    mask = offsets < total_elements
    k = offsets % rank
    segment_position = offsets // rank
    local_row = segment_position % max_segment_length
    segment = segment_position // max_segment_length
    start = tl.load(seg_indptr + segment, mask=mask, other=0)
    end = tl.load(seg_indptr + segment + 1, mask=mask, other=0)
    adapter = tl.load(weight_indices + segment, mask=mask, other=0)
    active = tl.load(lora_ranks + adapter, mask=mask, other=0) != 0
    row = start + local_row
    valid = mask & (row < end) & active
    values = tl.load(x + row * rank + k, mask=valid, other=0.0)
    tl.store(expanded_x + offsets, values.to(tl.float16), mask=mask)


@triton.jit
def _xpu_prepare_weights(
    weights,
    weight_indices,
    lora_ranks,
    scalings,
    expanded_weights,
    total_elements,
    rank: tl.constexpr,
    output_size: tl.constexpr,
    stride_weight_adapter: tl.constexpr,
    stride_weight_output: tl.constexpr,
    block_size: tl.constexpr,
):
    offsets = tl.program_id(0) * block_size + tl.arange(0, block_size)
    mask = offsets < total_elements
    k = offsets % rank
    segment_output = offsets // rank
    output_column = segment_output % output_size
    segment = segment_output // output_size
    adapter = tl.load(weight_indices + segment, mask=mask, other=0)
    active = tl.load(lora_ranks + adapter, mask=mask, other=0) != 0
    scaling = tl.load(scalings + adapter, mask=mask, other=0.0)
    values = tl.load(
        weights
        + adapter * stride_weight_adapter
        + output_column * stride_weight_output
        + k,
        mask=mask & active,
        other=0.0,
    ).to(tl.float32)
    tl.store(
        expanded_weights + offsets,
        (values * scaling).to(tl.float16),
        mask=mask,
    )


@triton.jit
def _xpu_group_dot(
    expanded_x,
    expanded_weights,
    delta,
    rank: tl.constexpr,
    output_size: tl.constexpr,
    max_segment_length: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    block_k: tl.constexpr,
):
    program = tl.program_id(0)
    output_tiles: tl.constexpr = tl.cdiv(output_size, block_n)
    row_tiles: tl.constexpr = tl.cdiv(max_segment_length, block_m)
    segment_row_tile = program // output_tiles
    output_tile = program - segment_row_tile * output_tiles
    segment = segment_row_tile // row_tiles
    row_tile = segment_row_tile - segment * row_tiles
    offsets_m = row_tile * block_m + tl.arange(0, block_m)
    offsets_n = output_tile * block_n + tl.arange(0, block_n)
    mask_m = offsets_m < max_segment_length
    mask_n = offsets_n < output_size
    accumulator = tl.zeros((block_m, block_n), dtype=tl.float32)
    x_segment = expanded_x + segment * max_segment_length * rank
    weight_segment = expanded_weights + segment * output_size * rank
    for offset_k in range(0, rank, block_k):
        offsets_k = offset_k + tl.arange(0, block_k)
        mask_k = offsets_k < rank
        x_tile = tl.load(
            x_segment + offsets_m[:, None] * rank + offsets_k[None, :],
            mask=mask_m[:, None] & mask_k[None, :],
            other=0.0,
        )
        weight_tile = tl.load(
            weight_segment + offsets_n[:, None] * rank + offsets_k[None, :],
            mask=mask_n[:, None] & mask_k[None, :],
            other=0.0,
        )
        accumulator += tl.dot(x_tile, tl.trans(weight_tile))
    delta_offsets = (
        segment * max_segment_length * output_size
        + offsets_m[:, None] * output_size
        + offsets_n[None, :]
    )
    tl.store(
        delta + delta_offsets,
        accumulator,
        mask=mask_m[:, None] & mask_n[None, :],
    )


@triton.jit
def _xpu_add_base(
    delta,
    seg_indptr,
    base_output,
    output,
    total_elements,
    output_size: tl.constexpr,
    max_segment_length: tl.constexpr,
    block_size: tl.constexpr,
):
    offsets = tl.program_id(0) * block_size + tl.arange(0, block_size)
    mask = offsets < total_elements
    output_column = offsets % output_size
    segment_position = offsets // output_size
    local_row = segment_position % max_segment_length
    segment = segment_position // max_segment_length
    start = tl.load(seg_indptr + segment, mask=mask, other=0)
    end = tl.load(seg_indptr + segment + 1, mask=mask, other=0)
    row = start + local_row
    valid = mask & (row < end)
    base = tl.load(
        base_output + row * output_size + output_column, mask=valid, other=0.0
    )
    update = tl.load(delta + offsets, mask=valid, other=0.0)
    tl.store(
        output + row * output_size + output_column, base + update, mask=valid
    )


def _launch_xpu_pipeline(
    x, weights, batch_info, base_output, block_m, block_n
):
    _, rank = x.shape
    output_size = weights.shape[1]
    batch_size = batch_info.bs
    max_segment_length = max(1, batch_info.max_len)
    expanded_x = torch.empty(
        (batch_size, max_segment_length, rank),
        device=x.device,
        dtype=torch.float16,
    )
    expanded_weights = torch.empty(
        (batch_size, output_size, rank), device=x.device, dtype=torch.float16
    )
    delta = torch.empty(
        (batch_size, max_segment_length, output_size),
        device=x.device,
        dtype=torch.float32,
    )
    output = torch.empty_like(base_output)
    x_elements = expanded_x.numel()
    weight_elements = expanded_weights.numel()
    delta_elements = delta.numel()
    _xpu_prepare_x[(triton.cdiv(x_elements, 8192),)](
        x,
        batch_info.seg_indptr,
        batch_info.weight_indices,
        batch_info.lora_ranks,
        expanded_x,
        x_elements,
        rank,
        max_segment_length,
        8192,
        num_warps=4,
    )
    _xpu_prepare_weights[(triton.cdiv(weight_elements, 8192),)](
        weights,
        batch_info.weight_indices,
        batch_info.lora_ranks,
        batch_info.scalings,
        expanded_weights,
        weight_elements,
        rank,
        output_size,
        weights.stride(0),
        weights.stride(1),
        8192,
        num_warps=4,
    )
    torch.cuda.synchronize()
    dot_grid = (
        batch_size
        * triton.cdiv(max_segment_length, block_m)
        * triton.cdiv(output_size, block_n),
    )
    _xpu_group_dot[dot_grid](
        expanded_x,
        expanded_weights,
        delta,
        rank,
        output_size,
        max_segment_length,
        block_m,
        block_n,
        64,
        num_warps=4,
        num_stages=2,
    )
    torch.cuda.synchronize()
    _xpu_add_base[(triton.cdiv(delta_elements, 8192),)](
        delta,
        batch_info.seg_indptr,
        base_output,
        output,
        delta_elements,
        output_size,
        max_segment_length,
        8192,
        num_warps=4,
    )
    output._task23_buffers = (expanded_x, expanded_weights, delta)
    return output


@triton.jit
def _sgemm_lora_b_kernel(
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
    output_size: tl.constexpr,
    max_segment_length: tl.constexpr,
    stride_x_row: tl.constexpr,
    stride_weight_adapter: tl.constexpr,
    stride_weight_output: tl.constexpr,
    stride_base_row: tl.constexpr,
    has_permutation: tl.constexpr,
    use_dot: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    block_k: tl.constexpr,
):
    segment_tile = tl.program_id(0)
    output_tile = tl.program_id(1)
    tiles_per_segment: tl.constexpr = tl.cdiv(max_segment_length, block_m)
    segment = segment_tile // tiles_per_segment
    tile_m = segment_tile - segment * tiles_per_segment

    start = tl.load(seg_indptr + segment)
    end = tl.load(seg_indptr + segment + 1)
    adapter = tl.load(weight_indices + segment)
    active = tl.load(lora_ranks + adapter) != 0
    scaling = tl.load(scalings + adapter)

    offsets_m = tile_m * block_m + tl.arange(0, block_m)
    offsets_n = output_tile * block_n + tl.arange(0, block_n)
    sequence_positions = start + offsets_m
    mask_m = sequence_positions < end
    compute_m = mask_m & active
    mask_n = offsets_n < output_size
    if has_permutation:
        rows = tl.load(permutation + sequence_positions, mask=mask_m, other=0)
    else:
        rows = sequence_positions

    accumulator = tl.zeros((block_m, block_n), dtype=tl.float32)
    if use_dot:
        for offset_k in range(0, rank, block_k):
            offsets_k = offset_k + tl.arange(0, block_k)
            mask_k = offsets_k < rank
            x_tile = tl.load(
                x + rows[:, None] * stride_x_row + offsets_k[None, :],
                mask=compute_m[:, None] & mask_k[None, :],
                other=0.0,
            )
            weight_tile = tl.load(
                weights
                + adapter * stride_weight_adapter
                + offsets_k[:, None]
                + offsets_n[None, :] * stride_weight_output,
                mask=mask_k[:, None] & mask_n[None, :],
                other=0.0,
            )
            accumulator += tl.dot(x_tile, weight_tile)
    else:
        for offset_k in range(0, rank):
            x_values = tl.load(
                x + rows * stride_x_row + offset_k,
                mask=compute_m,
                other=0.0,
            ).to(tl.float32)
            weight_values = tl.load(
                weights
                + adapter * stride_weight_adapter
                + offsets_n * stride_weight_output
                + offset_k,
                mask=mask_n,
                other=0.0,
            ).to(tl.float32)
            accumulator += x_values[:, None] * weight_values[None, :]

    output_offsets = rows[:, None] * stride_base_row + offsets_n[None, :]
    base = tl.load(
        base_output + output_offsets,
        mask=mask_m[:, None] & mask_n[None, :],
        other=0.0,
    ).to(tl.float32)
    tl.store(
        output + output_offsets,
        base + accumulator * scaling,
        mask=mask_m[:, None] & mask_n[None, :],
    )


@triton.jit
def _serial_kernel(
    x,
    weights,
    seg_indptr,
    weight_indices,
    lora_ranks,
    scalings,
    base_output,
    output,
    rank: tl.constexpr,
    output_size: tl.constexpr,
    max_segment_length: tl.constexpr,
    stride_x_row: tl.constexpr,
    stride_weight_adapter: tl.constexpr,
    stride_weight_output: tl.constexpr,
    stride_base_row: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
):
    segment_tile = tl.program_id(0)
    output_tile = tl.program_id(1)
    tiles_per_segment: tl.constexpr = tl.cdiv(max_segment_length, block_m)
    segment = segment_tile // tiles_per_segment
    tile_m = segment_tile - segment * tiles_per_segment
    start = tl.load(seg_indptr + segment)
    end = tl.load(seg_indptr + segment + 1)
    adapter = tl.load(weight_indices + segment)
    active = tl.load(lora_ranks + adapter) != 0
    scaling = tl.load(scalings + adapter)
    offsets_m = tile_m * block_m + tl.arange(0, block_m)
    offsets_n = output_tile * block_n + tl.arange(0, block_n)
    rows = start + offsets_m
    mask_m = rows < end
    compute_m = mask_m & active
    mask_n = offsets_n < output_size
    accumulator = tl.zeros((block_m, block_n), dtype=tl.float32)
    for offset_k in tl.static_range(0, rank):
        x_values = tl.load(
            x + rows * stride_x_row + offset_k,
            mask=compute_m,
            other=0.0,
        ).to(tl.float32)
        weight_values = tl.load(
            weights
            + adapter * stride_weight_adapter
            + offsets_n * stride_weight_output
            + offset_k,
            mask=mask_n,
            other=0.0,
        ).to(tl.float32)
        accumulator += x_values[:, None] * weight_values[None, :]
    output_offsets = rows[:, None] * stride_base_row + offsets_n[None, :]
    base = tl.load(
        base_output + output_offsets,
        mask=mask_m[:, None] & mask_n[None, :],
        other=0.0,
    ).to(tl.float32)
    tl.store(
        output + output_offsets,
        base + accumulator * scaling,
        mask=mask_m[:, None] & mask_n[None, :],
    )


@triton.jit
def _serial_permuted_kernel(
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
    output_size: tl.constexpr,
    max_segment_length: tl.constexpr,
    stride_x_row: tl.constexpr,
    stride_weight_adapter: tl.constexpr,
    stride_weight_output: tl.constexpr,
    stride_base_row: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
):
    segment_tile = tl.program_id(0)
    output_tile = tl.program_id(1)
    tiles_per_segment: tl.constexpr = tl.cdiv(max_segment_length, block_m)
    segment = segment_tile // tiles_per_segment
    tile_m = segment_tile - segment * tiles_per_segment
    start = tl.load(seg_indptr + segment)
    end = tl.load(seg_indptr + segment + 1)
    adapter = tl.load(weight_indices + segment)
    active = tl.load(lora_ranks + adapter) != 0
    scaling = tl.load(scalings + adapter)
    offsets_m = tile_m * block_m + tl.arange(0, block_m)
    offsets_n = output_tile * block_n + tl.arange(0, block_n)
    sequence_positions = start + offsets_m
    mask_m = sequence_positions < end
    compute_m = mask_m & active
    mask_n = offsets_n < output_size
    rows = tl.load(permutation + sequence_positions, mask=mask_m, other=0)
    accumulator = tl.zeros((block_m, block_n), dtype=tl.float32)
    for offset_k in tl.static_range(0, rank):
        x_values = tl.load(
            x + rows * stride_x_row + offset_k,
            mask=compute_m,
            other=0.0,
        ).to(tl.float32)
        weight_values = tl.load(
            weights
            + adapter * stride_weight_adapter
            + offsets_n * stride_weight_output
            + offset_k,
            mask=mask_n,
            other=0.0,
        ).to(tl.float32)
        accumulator += x_values[:, None] * weight_values[None, :]
    output_offsets = rows[:, None] * stride_base_row + offsets_n[None, :]
    base = tl.load(
        base_output + output_offsets,
        mask=mask_m[:, None] & mask_n[None, :],
        other=0.0,
    ).to(tl.float32)
    tl.store(
        output + output_offsets,
        base + accumulator * scaling,
        mask=mask_m[:, None] & mask_n[None, :],
    )


def _launch_config(
    x,
    weights,
    batch_info,
    base_output,
    block_m,
    block_n,
    block_k,
    num_warps,
    num_stages,
):
    _, rank = x.shape
    output_size = weights.shape[1]
    batch_size = batch_info.bs
    max_segment_length = max(1, batch_info.max_len)
    output = torch.empty_like(base_output)
    grid = (
        batch_size * triton.cdiv(max_segment_length, block_m),
        triton.cdiv(output_size, block_n),
    )
    permutation = batch_info.permutation
    _sgemm_lora_b_kernel[grid](
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
        output_size,
        max_segment_length,
        x.stride(0),
        weights.stride(0),
        weights.stride(1),
        base_output.stride(0),
        permutation is not None,
        x.dtype != torch.float32,
        block_m,
        block_n,
        block_k,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return output


def _launch_serial(x, weights, batch_info, base_output, block_m, block_n):
    _, rank = x.shape
    output_size = weights.shape[1]
    max_segment_length = max(1, batch_info.max_len)
    output = torch.empty_like(base_output)
    grid = (
        batch_info.bs * triton.cdiv(max_segment_length, block_m),
        triton.cdiv(output_size, block_n),
    )
    _serial_kernel[grid](
        x,
        weights,
        batch_info.seg_indptr,
        batch_info.weight_indices,
        batch_info.lora_ranks,
        batch_info.scalings,
        base_output,
        output,
        rank,
        output_size,
        max_segment_length,
        x.stride(0),
        weights.stride(0),
        weights.stride(1),
        base_output.stride(0),
        block_m,
        block_n,
        num_warps=4,
        num_stages=1,
    )
    return output


def _launch_serial_permuted(x, weights, batch_info, base_output):
    _, rank = x.shape
    output_size = weights.shape[1]
    max_segment_length = max(1, batch_info.max_len)
    block_m = 16
    block_n = 64
    output = torch.empty_like(base_output)
    grid = (
        batch_info.bs * triton.cdiv(max_segment_length, block_m),
        triton.cdiv(output_size, block_n),
    )
    _serial_permuted_kernel[grid](
        x,
        weights,
        batch_info.seg_indptr,
        batch_info.weight_indices,
        batch_info.lora_ranks,
        batch_info.scalings,
        batch_info.permutation,
        base_output,
        output,
        rank,
        output_size,
        max_segment_length,
        x.stride(0),
        weights.stride(0),
        weights.stride(1),
        base_output.stride(0),
        block_m,
        block_n,
        num_warps=4,
        num_stages=1,
    )
    return output


def sgemm_lora_b(x, weights, batch_info, base_output):
    if batch_info.permutation is not None:
        return _launch_serial_permuted(x, weights, batch_info, base_output)
    if weights.shape[1] == 4096 and x.shape[1] in (32, 64):
        return _launch_xpu_pipeline(
            x, weights, batch_info, base_output, 16, 32
        )
    if x.shape[1] == 32:
        return _launch_serial(x, weights, batch_info, base_output, 16, 128)
    if x.shape[1] >= 64:
        return _launch_serial(x, weights, batch_info, base_output, 32, 64)
    return _launch_serial(x, weights, batch_info, base_output, 16, 64)


__all__ = ["sgemm_lora_b"]
