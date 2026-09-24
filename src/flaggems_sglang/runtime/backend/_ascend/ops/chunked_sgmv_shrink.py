# Copyright 2026 FlagOS Contributors
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

import torch
import triton
import triton.language as tl

__all__ = ["chunked_sgmv_shrink"]


@triton.jit
def _sgemm_lora_a_kernel(
    x_ptr,
    weights_ptr,
    seg_indptr_ptr,
    weight_indices_ptr,
    permutation_ptr,
    output_ptr,
    hidden_size: tl.constexpr,
    output_size: tl.constexpr,
    max_segment_length: tl.constexpr,
    stride_x_row: tl.constexpr,
    stride_x_hidden: tl.constexpr,
    stride_weight_adapter: tl.constexpr,
    stride_weight_output: tl.constexpr,
    stride_weight_hidden: tl.constexpr,
    stride_output_row: tl.constexpr,
    stride_output_rank: tl.constexpr,
    has_permutation: tl.constexpr,
    use_tf32: tl.constexpr,
    round_inputs: tl.constexpr,
    use_default_dot: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    block_k: tl.constexpr,
):
    segment_tile = tl.program_id(0)
    output_tile = tl.program_id(1)
    tiles_per_segment: tl.constexpr = tl.cdiv(max_segment_length, block_m)
    segment = segment_tile // tiles_per_segment
    tile_m = segment_tile - segment * tiles_per_segment
    start = tl.load(seg_indptr_ptr + segment)
    end = tl.load(seg_indptr_ptr + segment + 1)
    adapter = tl.load(weight_indices_ptr + segment)
    positions = start + tile_m * block_m + tl.arange(0, block_m)
    mask_m = positions < end
    if has_permutation:
        rows = tl.load(permutation_ptr + positions, mask=mask_m, other=0)
    else:
        rows = positions
    offsets_n = output_tile * block_n + tl.arange(0, block_n)
    mask_n = offsets_n < output_size
    accumulator = tl.zeros((block_m, block_n), dtype=tl.float32)
    for offset_k in tl.range(0, hidden_size, block_k):
        offsets_k = offset_k + tl.arange(0, block_k)
        mask_k = offsets_k < hidden_size
        x_values = tl.load(
            x_ptr
            + rows[:, None] * stride_x_row
            + offsets_k[None, :] * stride_x_hidden,
            mask=mask_m[:, None] & mask_k[None, :],
            other=0.0,
        )
        weight_values = tl.load(
            weights_ptr
            + adapter * stride_weight_adapter
            + offsets_k[:, None] * stride_weight_hidden
            + offsets_n[None, :] * stride_weight_output,
            mask=mask_k[:, None] & mask_n[None, :],
            other=0.0,
        )
        if round_inputs:
            x_bits = x_values.to(tl.int32, bitcast=True)
            weight_bits = weight_values.to(tl.int32, bitcast=True)
            rounded_x = ((x_bits + 0x1000) & -8192).to(
                tl.float32, bitcast=True
            )
            rounded_weight = ((weight_bits + 0x1000) & -8192).to(
                tl.float32, bitcast=True
            )
            x_values = rounded_x
            weight_values = rounded_weight
            accumulator += tl.dot(x_values, weight_values)
        elif use_default_dot:
            accumulator += tl.dot(x_values, weight_values)
        elif use_tf32:
            accumulator += tl.dot(
                x_values, weight_values, input_precision="tf32"
            )
        else:
            accumulator += tl.dot(
                x_values, weight_values, input_precision="ieee"
            )
    output_offsets = (
        rows[:, None] * stride_output_row
        + offsets_n[None, :] * stride_output_rank
    )
    tl.store(
        output_ptr + output_offsets,
        accumulator,
        mask=mask_m[:, None] & mask_n[None, :],
    )


@triton.jit
def _sgemm_lora_a_serial_kernel(
    x_ptr,
    weights_ptr,
    seg_indptr_ptr,
    weight_indices_ptr,
    permutation_ptr,
    output_ptr,
    hidden_size: tl.constexpr,
    output_size: tl.constexpr,
    max_segment_length: tl.constexpr,
    stride_x_row: tl.constexpr,
    stride_x_hidden: tl.constexpr,
    stride_weight_adapter: tl.constexpr,
    stride_weight_output: tl.constexpr,
    stride_weight_hidden: tl.constexpr,
    stride_output_row: tl.constexpr,
    stride_output_rank: tl.constexpr,
    has_permutation: tl.constexpr,
    round_mode: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
):
    segment_tile = tl.program_id(0)
    output_tile = tl.program_id(1)
    tiles_per_segment: tl.constexpr = tl.cdiv(max_segment_length, block_m)
    segment = segment_tile // tiles_per_segment
    tile_m = segment_tile - segment * tiles_per_segment
    start = tl.load(seg_indptr_ptr + segment)
    end = tl.load(seg_indptr_ptr + segment + 1)
    adapter = tl.load(weight_indices_ptr + segment)
    positions = start + tile_m * block_m + tl.arange(0, block_m)
    mask_m = positions < end
    if has_permutation:
        rows = tl.load(permutation_ptr + positions, mask=mask_m, other=0)
    else:
        rows = positions
    offsets_n = output_tile * block_n + tl.arange(0, block_n)
    mask_n = offsets_n < output_size
    accumulator = tl.zeros((block_m, block_n), dtype=tl.float32)
    for offset_k in tl.static_range(0, hidden_size):
        x_values = tl.load(
            x_ptr + rows * stride_x_row + offset_k * stride_x_hidden,
            mask=mask_m,
            other=0.0,
        ).to(tl.float32)
        weight_values = tl.load(
            weights_ptr
            + adapter * stride_weight_adapter
            + offsets_n * stride_weight_output
            + offset_k * stride_weight_hidden,
            mask=mask_n,
            other=0.0,
        ).to(tl.float32)
        if round_mode == 1 or round_mode == 3:
            x_bits = x_values.to(tl.int32, bitcast=True)
            weight_bits = weight_values.to(tl.int32, bitcast=True)
            rounded_x = ((x_bits + 0x0FFF + ((x_bits >> 13) & 1)) & -8192).to(
                tl.float32, bitcast=True
            )
            rounded_weight = (
                (weight_bits + 0x0FFF + ((weight_bits >> 13) & 1)) & -8192
            ).to(tl.float32, bitcast=True)
            if round_mode == 1:
                x_values = rounded_x
                weight_values = rounded_weight
            else:
                use_rounded = end - start > 1
                x_values = tl.where(use_rounded, rounded_x, x_values)
                weight_values = tl.where(
                    use_rounded, rounded_weight, weight_values
                )
        elif round_mode == 2 or round_mode == 4:
            x_bits = x_values.to(tl.int32, bitcast=True)
            weight_bits = weight_values.to(tl.int32, bitcast=True)
            truncated_x = (x_bits & -8192).to(tl.float32, bitcast=True)
            truncated_weight = (weight_bits & -8192).to(
                tl.float32, bitcast=True
            )
            if round_mode == 2:
                x_values = truncated_x
                weight_values = truncated_weight
            else:
                use_truncated = end - start >= 16
                x_values = tl.where(use_truncated, truncated_x, x_values)
                weight_values = tl.where(
                    use_truncated, truncated_weight, weight_values
                )
        elif round_mode == 5:
            x_bits = x_values.to(tl.int32, bitcast=True)
            weight_bits = weight_values.to(tl.int32, bitcast=True)
            rounded_x = ((x_bits + 0x1000) & -8192).to(
                tl.float32, bitcast=True
            )
            rounded_weight = ((weight_bits + 0x1000) & -8192).to(
                tl.float32, bitcast=True
            )
            x_values = rounded_x
            weight_values = rounded_weight
        accumulator += x_values[:, None] * weight_values[None, :]
    output_offsets = (
        rows[:, None] * stride_output_row
        + offsets_n[None, :] * stride_output_rank
    )
    tl.store(
        output_ptr + output_offsets,
        accumulator,
        mask=mask_m[:, None] & mask_n[None, :],
    )


@triton.jit
def _sgemm_lora_a_split_kernel(
    x_ptr,
    weights_ptr,
    seg_indptr_ptr,
    weight_indices_ptr,
    partial_ptr,
    hidden_size: tl.constexpr,
    output_size: tl.constexpr,
    max_segment_length: tl.constexpr,
    stride_x_row: tl.constexpr,
    stride_x_hidden: tl.constexpr,
    stride_weight_adapter: tl.constexpr,
    stride_weight_output: tl.constexpr,
    stride_weight_hidden: tl.constexpr,
    stride_partial_split: tl.constexpr,
    stride_partial_row: tl.constexpr,
    stride_partial_rank: tl.constexpr,
    split_count: tl.constexpr,
    split_size: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    block_k: tl.constexpr,
):
    combined_tile = tl.program_id(0)
    output_tile = tl.program_id(1)
    split = combined_tile % split_count
    segment_tile = combined_tile // split_count
    tiles_per_segment: tl.constexpr = tl.cdiv(max_segment_length, block_m)
    segment = segment_tile // tiles_per_segment
    tile_m = segment_tile - segment * tiles_per_segment
    start = tl.load(seg_indptr_ptr + segment)
    end = tl.load(seg_indptr_ptr + segment + 1)
    adapter = tl.load(weight_indices_ptr + segment)
    rows = start + tile_m * block_m + tl.arange(0, block_m)
    mask_m = rows < end
    offsets_n = output_tile * block_n + tl.arange(0, block_n)
    mask_n = offsets_n < output_size
    accumulator = tl.zeros((block_m, block_n), dtype=tl.float32)
    split_start = split * split_size
    for local_k in tl.static_range(0, split_size, block_k):
        offsets_k = split_start + local_k + tl.arange(0, block_k)
        mask_k = offsets_k < hidden_size
        x_values = tl.load(
            x_ptr
            + rows[:, None] * stride_x_row
            + offsets_k[None, :] * stride_x_hidden,
            mask=mask_m[:, None] & mask_k[None, :],
            other=0.0,
        )
        weight_values = tl.load(
            weights_ptr
            + adapter * stride_weight_adapter
            + offsets_k[:, None] * stride_weight_hidden
            + offsets_n[None, :] * stride_weight_output,
            mask=mask_k[:, None] & mask_n[None, :],
            other=0.0,
        )
        accumulator += tl.dot(x_values, weight_values)
    partial_offsets = (
        split * stride_partial_split
        + rows[:, None] * stride_partial_row
        + offsets_n[None, :] * stride_partial_rank
    )
    tl.store(
        partial_ptr + partial_offsets,
        accumulator,
        mask=mask_m[:, None] & mask_n[None, :],
    )


@triton.jit
def _sgemm_lora_a_split_2d_kernel(
    x_ptr,
    weights_ptr,
    seg_indptr_ptr,
    weight_indices_ptr,
    partial_ptr,
    hidden_size: tl.constexpr,
    output_size: tl.constexpr,
    max_segment_length: tl.constexpr,
    stride_x_row: tl.constexpr,
    stride_x_hidden: tl.constexpr,
    stride_weight_adapter: tl.constexpr,
    stride_weight_output: tl.constexpr,
    stride_weight_hidden: tl.constexpr,
    stride_partial_split: tl.constexpr,
    stride_partial_row: tl.constexpr,
    stride_partial_rank: tl.constexpr,
    split_size: tl.constexpr,
    output_tiles: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    block_k: tl.constexpr,
):
    split = tl.program_id(0)
    segment_output_tile = tl.program_id(1)
    segment_tile = segment_output_tile // output_tiles
    output_tile = segment_output_tile - segment_tile * output_tiles
    tiles_per_segment: tl.constexpr = tl.cdiv(max_segment_length, block_m)
    segment = segment_tile // tiles_per_segment
    tile_m = segment_tile - segment * tiles_per_segment
    start = tl.load(seg_indptr_ptr + segment)
    end = tl.load(seg_indptr_ptr + segment + 1)
    adapter = tl.load(weight_indices_ptr + segment)
    rows = start + tile_m * block_m + tl.arange(0, block_m)
    mask_m = rows < end
    offsets_n = output_tile * block_n + tl.arange(0, block_n)
    mask_n = offsets_n < output_size
    accumulator = tl.zeros((block_m, block_n), dtype=tl.float32)
    split_start = split * split_size
    for local_k in tl.static_range(0, split_size, block_k):
        offsets_k = split_start + local_k + tl.arange(0, block_k)
        mask_k = offsets_k < hidden_size
        x_values = tl.load(
            x_ptr
            + rows[:, None] * stride_x_row
            + offsets_k[None, :] * stride_x_hidden,
            mask=mask_m[:, None] & mask_k[None, :],
            other=0.0,
        )
        weight_values = tl.load(
            weights_ptr
            + adapter * stride_weight_adapter
            + offsets_k[:, None] * stride_weight_hidden
            + offsets_n[None, :] * stride_weight_output,
            mask=mask_k[:, None] & mask_n[None, :],
            other=0.0,
        )
        accumulator += tl.dot(x_values, weight_values)
    partial_offsets = (
        split * stride_partial_split
        + rows[:, None] * stride_partial_row
        + offsets_n[None, :] * stride_partial_rank
    )
    tl.store(
        partial_ptr + partial_offsets,
        accumulator,
        mask=mask_m[:, None] & mask_n[None, :],
    )


@triton.jit
def _sgemm_lora_a_split_once_kernel(
    x_ptr,
    weights_ptr,
    seg_indptr_ptr,
    weight_indices_ptr,
    partial_ptr,
    hidden_size: tl.constexpr,
    output_size: tl.constexpr,
    max_segment_length: tl.constexpr,
    stride_x_row: tl.constexpr,
    stride_x_hidden: tl.constexpr,
    stride_weight_adapter: tl.constexpr,
    stride_weight_output: tl.constexpr,
    stride_weight_hidden: tl.constexpr,
    stride_partial_split: tl.constexpr,
    stride_partial_row: tl.constexpr,
    stride_partial_rank: tl.constexpr,
    split_size: tl.constexpr,
    output_tiles: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
):
    split = tl.program_id(0)
    segment_output_tile = tl.program_id(1)
    segment_tile = segment_output_tile // output_tiles
    output_tile = segment_output_tile - segment_tile * output_tiles
    tiles_per_segment: tl.constexpr = tl.cdiv(max_segment_length, block_m)
    segment = segment_tile // tiles_per_segment
    tile_m = segment_tile - segment * tiles_per_segment
    start = tl.load(seg_indptr_ptr + segment)
    end = tl.load(seg_indptr_ptr + segment + 1)
    adapter = tl.load(weight_indices_ptr + segment)
    rows = start + tile_m * block_m + tl.arange(0, block_m)
    offsets_n = output_tile * block_n + tl.arange(0, block_n)
    offsets_k = split * split_size + tl.arange(0, split_size)
    mask_m = rows < end
    mask_n = offsets_n < output_size
    mask_k = offsets_k < hidden_size
    x_values = tl.load(
        x_ptr
        + rows[:, None] * stride_x_row
        + offsets_k[None, :] * stride_x_hidden,
        mask=mask_m[:, None] & mask_k[None, :],
        other=0.0,
    )
    weight_values = tl.load(
        weights_ptr
        + adapter * stride_weight_adapter
        + offsets_k[:, None] * stride_weight_hidden
        + offsets_n[None, :] * stride_weight_output,
        mask=mask_k[:, None] & mask_n[None, :],
        other=0.0,
    )
    accumulator = tl.dot(x_values, weight_values)
    partial_offsets = (
        split * stride_partial_split
        + rows[:, None] * stride_partial_row
        + offsets_n[None, :] * stride_partial_rank
    )
    tl.store(
        partial_ptr + partial_offsets,
        accumulator,
        mask=mask_m[:, None] & mask_n[None, :],
    )


@triton.jit
def _sgemm_lora_a_reduce_kernel(
    partial_ptr,
    output_ptr,
    token_count: tl.constexpr,
    output_size: tl.constexpr,
    stride_partial_split: tl.constexpr,
    stride_partial_row: tl.constexpr,
    stride_partial_rank: tl.constexpr,
    stride_output_row: tl.constexpr,
    stride_output_rank: tl.constexpr,
    split_count: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
):
    rows = tl.program_id(0) * block_m + tl.arange(0, block_m)
    offsets_n = tl.program_id(1) * block_n + tl.arange(0, block_n)
    mask = (rows[:, None] < token_count) & (offsets_n[None, :] < output_size)
    accumulator = tl.zeros((block_m, block_n), dtype=tl.float32)
    for split in tl.static_range(0, split_count):
        partial_offsets = (
            split * stride_partial_split
            + rows[:, None] * stride_partial_row
            + offsets_n[None, :] * stride_partial_rank
        )
        accumulator += tl.load(
            partial_ptr + partial_offsets, mask=mask, other=0.0
        )
    output_offsets = (
        rows[:, None] * stride_output_row
        + offsets_n[None, :] * stride_output_rank
    )
    tl.store(output_ptr + output_offsets, accumulator, mask=mask)


@triton.jit
def _gather_permuted_kernel(
    x_ptr,
    permutation_ptr,
    packed_ptr,
    element_count,
    hidden_size: tl.constexpr,
    block_size: tl.constexpr,
):
    offsets = tl.program_id(0) * block_size + tl.arange(0, block_size)
    mask = offsets < element_count
    positions = offsets // hidden_size
    hidden_offsets = offsets - positions * hidden_size
    rows = tl.load(permutation_ptr + positions, mask=mask, other=0)
    values = tl.load(
        x_ptr + rows * hidden_size + hidden_offsets, mask=mask, other=0.0
    )
    tl.store(packed_ptr + offsets, values, mask=mask)


@triton.jit
def _scatter_permuted_kernel(
    packed_ptr,
    permutation_ptr,
    output_ptr,
    element_count,
    output_size: tl.constexpr,
    block_size: tl.constexpr,
):
    offsets = tl.program_id(0) * block_size + tl.arange(0, block_size)
    mask = offsets < element_count
    positions = offsets // output_size
    output_offsets = offsets - positions * output_size
    rows = tl.load(permutation_ptr + positions, mask=mask, other=0)
    values = tl.load(packed_ptr + offsets, mask=mask, other=0.0)
    tl.store(
        output_ptr + rows * output_size + output_offsets, values, mask=mask
    )


def _launch(
    x,
    weights,
    batch_info,
    stack_num,
    block_m,
    block_n,
    block_k,
    num_warps,
    num_stages,
    use_tf32=False,
    round_inputs=False,
    use_default_dot=False,
):
    token_count, hidden_size = x.shape
    output_size = weights.shape[1]
    max_segment_length = max(1, batch_info.max_len)
    output = torch.zeros(
        (token_count, output_size), dtype=x.dtype, device=x.device
    )
    grid = (
        batch_info.bs * triton.cdiv(max_segment_length, block_m),
        triton.cdiv(output_size, block_n),
    )
    permutation = batch_info.permutation
    _sgemm_lora_a_kernel[grid](
        x,
        weights,
        batch_info.seg_indptr,
        batch_info.weight_indices,
        x if permutation is None else permutation,
        output,
        hidden_size,
        output_size,
        max_segment_length,
        x.stride(0),
        x.stride(1),
        weights.stride(0),
        weights.stride(1),
        weights.stride(2),
        output.stride(0),
        output.stride(1),
        permutation is not None,
        use_tf32,
        round_inputs,
        use_default_dot,
        block_m,
        block_n,
        block_k,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return output


def _launch_serial(x, weights, batch_info, stack_num, round_mode=0):
    token_count, hidden_size = x.shape
    output_size = weights.shape[1]
    max_segment_length = max(1, batch_info.max_len)
    block_m = 8
    block_n = 16
    output = torch.zeros(
        (token_count, output_size), dtype=x.dtype, device=x.device
    )
    grid = (
        batch_info.bs * triton.cdiv(max_segment_length, block_m),
        triton.cdiv(output_size, block_n),
    )
    permutation = batch_info.permutation
    _sgemm_lora_a_serial_kernel[grid](
        x,
        weights,
        batch_info.seg_indptr,
        batch_info.weight_indices,
        x if permutation is None else permutation,
        output,
        hidden_size,
        output_size,
        max_segment_length,
        x.stride(0),
        x.stride(1),
        weights.stride(0),
        weights.stride(1),
        weights.stride(2),
        output.stride(0),
        output.stride(1),
        permutation is not None,
        round_mode,
        block_m,
        block_n,
        num_warps=4,
        num_stages=1,
    )
    return output


def _launch_split(
    x,
    weights,
    batch_info,
    stack_num,
    split_count,
    block_m,
    block_n,
    block_k,
    gcu_options=False,
):
    token_count, hidden_size = x.shape
    output_size = weights.shape[1]
    max_segment_length = max(1, batch_info.max_len)
    split_size = triton.cdiv(hidden_size, split_count * block_k) * block_k
    partial = torch.empty(
        (split_count, token_count, output_size),
        dtype=torch.float32,
        device=x.device,
    )
    output = torch.zeros(
        (token_count, output_size), dtype=x.dtype, device=x.device
    )
    split_grid = (
        batch_info.bs * triton.cdiv(max_segment_length, block_m) * split_count,
        triton.cdiv(output_size, block_n),
    )
    split_arguments = (
        x,
        weights,
        batch_info.seg_indptr,
        batch_info.weight_indices,
        partial,
        hidden_size,
        output_size,
        max_segment_length,
        x.stride(0),
        x.stride(1),
        weights.stride(0),
        weights.stride(1),
        weights.stride(2),
        partial.stride(0),
        partial.stride(1),
        partial.stride(2),
        split_count,
        split_size,
        block_m,
        block_n,
        block_k,
    )
    if gcu_options:
        _sgemm_lora_a_split_kernel[split_grid](
            *split_arguments,
            num_warps=1,
            num_stages=1,
            enable_fp_fusion=True,
        )
    else:
        _sgemm_lora_a_split_kernel[split_grid](
            *split_arguments,
            num_warps=4,
            num_stages=2,
        )
    reduce_grid = (
        triton.cdiv(token_count, block_m),
        triton.cdiv(output_size, block_n),
    )
    reduce_arguments = (
        partial,
        output,
        token_count,
        output_size,
        partial.stride(0),
        partial.stride(1),
        partial.stride(2),
        output.stride(0),
        output.stride(1),
        split_count,
        block_m,
        block_n,
    )
    if gcu_options:
        _sgemm_lora_a_reduce_kernel[reduce_grid](
            *reduce_arguments,
            num_warps=1,
            num_stages=1,
            enable_fp_fusion=True,
        )
    else:
        _sgemm_lora_a_reduce_kernel[reduce_grid](
            *reduce_arguments,
            num_warps=4,
            num_stages=1,
        )
    return output


def _launch_split_2d(
    x,
    weights,
    batch_info,
    stack_num,
    split_count,
    block_m,
    block_n,
    block_k,
):
    token_count, hidden_size = x.shape
    output_size = weights.shape[1]
    max_segment_length = max(1, batch_info.max_len)
    split_size = triton.cdiv(hidden_size, split_count * block_k) * block_k
    partial = torch.empty(
        (split_count, token_count, output_size),
        dtype=torch.float32,
        device=x.device,
    )
    output = torch.zeros(
        (token_count, output_size), dtype=x.dtype, device=x.device
    )
    output_tiles = triton.cdiv(output_size, block_n)
    split_grid = (
        split_count,
        batch_info.bs
        * triton.cdiv(max_segment_length, block_m)
        * output_tiles,
    )
    _sgemm_lora_a_split_2d_kernel[split_grid](
        x,
        weights,
        batch_info.seg_indptr,
        batch_info.weight_indices,
        partial,
        hidden_size,
        output_size,
        max_segment_length,
        x.stride(0),
        x.stride(1),
        weights.stride(0),
        weights.stride(1),
        weights.stride(2),
        partial.stride(0),
        partial.stride(1),
        partial.stride(2),
        split_size,
        output_tiles,
        block_m,
        block_n,
        block_k,
        num_warps=4,
        num_stages=2,
    )
    reduce_grid = (
        triton.cdiv(token_count, block_m),
        triton.cdiv(output_size, block_n),
    )
    _sgemm_lora_a_reduce_kernel[reduce_grid](
        partial,
        output,
        token_count,
        output_size,
        partial.stride(0),
        partial.stride(1),
        partial.stride(2),
        output.stride(0),
        output.stride(1),
        split_count,
        block_m,
        block_n,
        num_warps=4,
        num_stages=1,
    )
    return output


def _launch_split_once(
    x,
    weights,
    batch_info,
    stack_num,
    split_count,
    block_m,
    block_n,
):
    token_count, hidden_size = x.shape
    output_size = weights.shape[1]
    max_segment_length = max(1, batch_info.max_len)
    split_size = triton.next_power_of_2(triton.cdiv(hidden_size, split_count))
    partial = torch.empty(
        (split_count, token_count, output_size),
        dtype=torch.float32,
        device=x.device,
    )
    output = torch.zeros(
        (token_count, output_size), dtype=x.dtype, device=x.device
    )
    output_tiles = triton.cdiv(output_size, block_n)
    split_grid = (
        split_count,
        batch_info.bs
        * triton.cdiv(max_segment_length, block_m)
        * output_tiles,
    )
    _sgemm_lora_a_split_once_kernel[split_grid](
        x,
        weights,
        batch_info.seg_indptr,
        batch_info.weight_indices,
        partial,
        hidden_size,
        output_size,
        max_segment_length,
        x.stride(0),
        x.stride(1),
        weights.stride(0),
        weights.stride(1),
        weights.stride(2),
        partial.stride(0),
        partial.stride(1),
        partial.stride(2),
        split_size,
        output_tiles,
        block_m,
        block_n,
        num_warps=8,
        num_stages=1,
    )
    reduce_grid = (
        triton.cdiv(token_count, block_m),
        triton.cdiv(output_size, block_n),
    )
    _sgemm_lora_a_reduce_kernel[reduce_grid](
        partial,
        output,
        token_count,
        output_size,
        partial.stride(0),
        partial.stride(1),
        partial.stride(2),
        output.stride(0),
        output.stride(1),
        split_count,
        block_m,
        block_n,
        num_warps=4,
        num_stages=1,
    )
    return output


def _launch_split_permuted(
    x,
    weights,
    batch_info,
    stack_num,
    split_count,
    block_m,
    block_n,
    block_k,
):
    token_count, hidden_size = x.shape
    output_size = weights.shape[1]
    packed_x = torch.empty_like(x)
    input_elements = token_count * hidden_size
    _gather_permuted_kernel[(triton.cdiv(input_elements, 256),)](
        x,
        batch_info.permutation,
        packed_x,
        input_elements,
        hidden_size,
        256,
        num_warps=4,
        num_stages=1,
    )
    packed_output = _launch_split(
        packed_x,
        weights,
        batch_info,
        stack_num,
        split_count,
        block_m,
        block_n,
        block_k,
    )
    output = torch.zeros_like(packed_output)
    output_elements = token_count * output_size
    _scatter_permuted_kernel[(triton.cdiv(output_elements, 256),)](
        packed_output,
        batch_info.permutation,
        output,
        output_elements,
        output_size,
        256,
        num_warps=4,
        num_stages=1,
    )
    return output


def m16n16k32w4s2(x, weights, batch_info, stack_num=1):
    return _launch(x, weights, batch_info, stack_num, 16, 16, 32, 4, 2)


def m16n32k32w4s2(x, weights, batch_info, stack_num=1):
    return _launch(x, weights, batch_info, stack_num, 16, 32, 32, 4, 2)


def m16n32k32w4s2_tf32(x, weights, batch_info, stack_num=1):
    return _launch(x, weights, batch_info, stack_num, 16, 32, 32, 4, 2, True)


def m16n32k32w4s2_round(x, weights, batch_info, stack_num=1):
    return _launch(
        x, weights, batch_info, stack_num, 16, 32, 32, 4, 2, False, True
    )


def m16n32k64w4s2_round(x, weights, batch_info, stack_num=1):
    return _launch(
        x, weights, batch_info, stack_num, 16, 32, 64, 4, 2, False, True
    )


def m32n64k32w4s2_round(x, weights, batch_info, stack_num=1):
    return _launch(
        x, weights, batch_info, stack_num, 32, 64, 32, 4, 2, False, True
    )


def m32n64k64w4s2_round(x, weights, batch_info, stack_num=1):
    return _launch(
        x, weights, batch_info, stack_num, 32, 64, 64, 4, 2, False, True
    )


def m64n64k64w4s2_round(x, weights, batch_info, stack_num=1):
    return _launch(
        x, weights, batch_info, stack_num, 64, 64, 64, 4, 2, False, True
    )


def m64n64k128w4s2_round(x, weights, batch_info, stack_num=1):
    return _launch(
        x, weights, batch_info, stack_num, 64, 64, 128, 4, 2, False, True
    )


def m64n64k256w4s2_round(x, weights, batch_info, stack_num=1):
    return _launch(
        x, weights, batch_info, stack_num, 64, 64, 256, 4, 2, False, True
    )


def native_m32n32k64(x, weights, batch_info, stack_num=1):
    return _launch(
        x, weights, batch_info, stack_num, 32, 32, 64, 4, 2, False, False, True
    )


def native_m64n32k64(x, weights, batch_info, stack_num=1):
    return _launch(
        x, weights, batch_info, stack_num, 64, 32, 64, 4, 2, False, False, True
    )


def native_m64n64k64(x, weights, batch_info, stack_num=1):
    return _launch(
        x, weights, batch_info, stack_num, 64, 64, 64, 4, 2, False, False, True
    )


def native_m128n64k64(x, weights, batch_info, stack_num=1):
    return _launch(
        x,
        weights,
        batch_info,
        stack_num,
        128,
        64,
        64,
        4,
        2,
        False,
        False,
        True,
    )


def native_m32n32k128w4s2(x, weights, batch_info, stack_num=1):
    return _launch(
        x,
        weights,
        batch_info,
        stack_num,
        32,
        32,
        128,
        4,
        2,
        False,
        False,
        True,
    )


def native_m32n32k256w4s2(x, weights, batch_info, stack_num=1):
    return _launch(
        x,
        weights,
        batch_info,
        stack_num,
        32,
        32,
        256,
        4,
        2,
        False,
        False,
        True,
    )


def native_m32n32k128w1s1(x, weights, batch_info, stack_num=1):
    return _launch(
        x,
        weights,
        batch_info,
        stack_num,
        32,
        32,
        128,
        1,
        1,
        False,
        False,
        True,
    )


def native_m32n32k128w2s1(x, weights, batch_info, stack_num=1):
    return _launch(
        x,
        weights,
        batch_info,
        stack_num,
        32,
        32,
        128,
        2,
        1,
        False,
        False,
        True,
    )


def native_m128n64k128w4s2(x, weights, batch_info, stack_num=1):
    return _launch(
        x,
        weights,
        batch_info,
        stack_num,
        128,
        64,
        128,
        4,
        2,
        False,
        False,
        True,
    )


def native_m128n64k256w4s2(x, weights, batch_info, stack_num=1):
    return _launch(
        x,
        weights,
        batch_info,
        stack_num,
        128,
        64,
        256,
        4,
        2,
        False,
        False,
        True,
    )


def native_m64n64k128w4s2(x, weights, batch_info, stack_num=1):
    return _launch(
        x,
        weights,
        batch_info,
        stack_num,
        64,
        64,
        128,
        4,
        2,
        False,
        False,
        True,
    )


def native_m64n64k256w4s2(x, weights, batch_info, stack_num=1):
    return _launch(
        x,
        weights,
        batch_info,
        stack_num,
        64,
        64,
        256,
        4,
        2,
        False,
        False,
        True,
    )


def native_m64n64k512w4s2(x, weights, batch_info, stack_num=1):
    return _launch(
        x,
        weights,
        batch_info,
        stack_num,
        64,
        64,
        512,
        4,
        2,
        False,
        False,
        True,
    )


def native_m128n64k512w4s2(x, weights, batch_info, stack_num=1):
    return _launch(
        x,
        weights,
        batch_info,
        stack_num,
        128,
        64,
        512,
        4,
        2,
        False,
        False,
        True,
    )


def native_m256n64k128w4s2(x, weights, batch_info, stack_num=1):
    return _launch(
        x,
        weights,
        batch_info,
        stack_num,
        256,
        64,
        128,
        4,
        2,
        False,
        False,
        True,
    )


def m16n64k32w4s2(x, weights, batch_info, stack_num=1):
    return _launch(x, weights, batch_info, stack_num, 16, 64, 32, 4, 2)


def m32n16k32w4s2(x, weights, batch_info, stack_num=1):
    return _launch(x, weights, batch_info, stack_num, 32, 16, 32, 4, 2)


def m32n32k32w4s2(x, weights, batch_info, stack_num=1):
    return _launch(x, weights, batch_info, stack_num, 32, 32, 32, 4, 2)


def m32n64k32w4s2(x, weights, batch_info, stack_num=1):
    return _launch(x, weights, batch_info, stack_num, 32, 64, 32, 4, 2)


def m64n16k32w4s2(x, weights, batch_info, stack_num=1):
    return _launch(x, weights, batch_info, stack_num, 64, 16, 32, 4, 2)


def m64n32k32w4s2(x, weights, batch_info, stack_num=1):
    return _launch(x, weights, batch_info, stack_num, 64, 32, 32, 4, 2)


def m64n64k32w4s2(x, weights, batch_info, stack_num=1):
    return _launch(x, weights, batch_info, stack_num, 64, 64, 32, 4, 2)


def m128n32k32w4s2(x, weights, batch_info, stack_num=1):
    return _launch(x, weights, batch_info, stack_num, 128, 32, 32, 4, 2)


def m128n64k32w4s2(x, weights, batch_info, stack_num=1):
    return _launch(x, weights, batch_info, stack_num, 128, 64, 32, 4, 2)


def m256n64k32w4s2(x, weights, batch_info, stack_num=1):
    return _launch(x, weights, batch_info, stack_num, 256, 64, 32, 4, 2)


def m32n32k64w4s2(x, weights, batch_info, stack_num=1):
    return _launch(x, weights, batch_info, stack_num, 32, 32, 64, 4, 2)


def m64n32k64w4s2(x, weights, batch_info, stack_num=1):
    return _launch(x, weights, batch_info, stack_num, 64, 32, 64, 4, 2)


def m64n64k64w4s2(x, weights, batch_info, stack_num=1):
    return _launch(x, weights, batch_info, stack_num, 64, 64, 64, 4, 2)


def m128n64k64w4s2(x, weights, batch_info, stack_num=1):
    return _launch(x, weights, batch_info, stack_num, 128, 64, 64, 4, 2)


def split4_m64n32k64(x, weights, batch_info, stack_num=1):
    return _launch_split(x, weights, batch_info, stack_num, 4, 64, 32, 64)


def split1_m64n32k64(x, weights, batch_info, stack_num=1):
    return _launch_split(x, weights, batch_info, stack_num, 1, 64, 32, 64)


def split8_m64n32k64(x, weights, batch_info, stack_num=1):
    return _launch_split(x, weights, batch_info, stack_num, 8, 64, 32, 64)


def split16_m64n32k64(x, weights, batch_info, stack_num=1):
    return _launch_split(x, weights, batch_info, stack_num, 16, 64, 32, 64)


def split32_m64n32k64(x, weights, batch_info, stack_num=1):
    return _launch_split(x, weights, batch_info, stack_num, 32, 64, 32, 64)


def split64_m64n32k64(x, weights, batch_info, stack_num=1):
    return _launch_split(x, weights, batch_info, stack_num, 64, 64, 32, 64)


def split8_m32n32k64(x, weights, batch_info, stack_num=1):
    return _launch_split(x, weights, batch_info, stack_num, 8, 32, 32, 64)


def split4_m256n64k64(x, weights, batch_info, stack_num=1):
    return _launch_split(x, weights, batch_info, stack_num, 4, 256, 64, 64)


def split8_m256n64k64(x, weights, batch_info, stack_num=1):
    return _launch_split(x, weights, batch_info, stack_num, 8, 256, 64, 64)


def split16_m256n64k64(x, weights, batch_info, stack_num=1):
    return _launch_split(x, weights, batch_info, stack_num, 16, 256, 64, 64)


def split8_m128n64k64(x, weights, batch_info, stack_num=1):
    return _launch_split(x, weights, batch_info, stack_num, 8, 128, 64, 64)


def gcu_split4_m64n32k64(x, weights, batch_info, stack_num=1):
    return _launch_split(
        x, weights, batch_info, stack_num, 4, 64, 32, 64, True
    )


def gcu_split8_m64n32k64(x, weights, batch_info, stack_num=1):
    return _launch_split(
        x, weights, batch_info, stack_num, 8, 64, 32, 64, True
    )


def gcu_split16_m64n32k64(x, weights, batch_info, stack_num=1):
    return _launch_split(
        x, weights, batch_info, stack_num, 16, 64, 32, 64, True
    )


def gcu_split32_m64n32k64(x, weights, batch_info, stack_num=1):
    return _launch_split(
        x, weights, batch_info, stack_num, 32, 64, 32, 64, True
    )


def gcu_split8_m64n64k64(x, weights, batch_info, stack_num=1):
    return _launch_split(
        x, weights, batch_info, stack_num, 8, 64, 64, 64, True
    )


def gcu_split8_m128n32k64(x, weights, batch_info, stack_num=1):
    return _launch_split(
        x, weights, batch_info, stack_num, 8, 128, 32, 64, True
    )


def gcu_split8_m256n32k64(x, weights, batch_info, stack_num=1):
    return _launch_split(
        x, weights, batch_info, stack_num, 8, 256, 32, 64, True
    )


def gcu_split8_m256n64k64(x, weights, batch_info, stack_num=1):
    return _launch_split(
        x, weights, batch_info, stack_num, 8, 256, 64, 64, True
    )


def gcu_split8_m64n32k128(x, weights, batch_info, stack_num=1):
    return _launch_split(
        x, weights, batch_info, stack_num, 8, 64, 32, 128, True
    )


def gcu_split4_m64n64k128(x, weights, batch_info, stack_num=1):
    return _launch_split(
        x, weights, batch_info, stack_num, 4, 64, 64, 128, True
    )


def gcu_split8_m64n64k128(x, weights, batch_info, stack_num=1):
    return _launch_split(
        x, weights, batch_info, stack_num, 8, 64, 64, 128, True
    )


def gcu_split16_m64n64k128(x, weights, batch_info, stack_num=1):
    return _launch_split(
        x, weights, batch_info, stack_num, 16, 64, 64, 128, True
    )


def gcu_split4_m256n64k64(x, weights, batch_info, stack_num=1):
    return _launch_split(
        x, weights, batch_info, stack_num, 4, 256, 64, 64, True
    )


def gcu_split16_m256n64k64(x, weights, batch_info, stack_num=1):
    return _launch_split(
        x, weights, batch_info, stack_num, 16, 256, 64, 64, True
    )


def gcu_split8_m256n64k128(x, weights, batch_info, stack_num=1):
    return _launch_split(
        x, weights, batch_info, stack_num, 8, 256, 64, 128, True
    )


def gcu_split16_m64n32k128(x, weights, batch_info, stack_num=1):
    return _launch_split(
        x, weights, batch_info, stack_num, 16, 64, 32, 128, True
    )


def gcu_split8_m64n32k256(x, weights, batch_info, stack_num=1):
    return _launch_split(
        x, weights, batch_info, stack_num, 8, 64, 32, 256, True
    )


def gcu_split16_m256n64k128(x, weights, batch_info, stack_num=1):
    return _launch_split(
        x, weights, batch_info, stack_num, 16, 256, 64, 128, True
    )


def gcu_split8_m256n64k256(x, weights, batch_info, stack_num=1):
    return _launch_split(
        x, weights, batch_info, stack_num, 8, 256, 64, 256, True
    )


def gcu_split8_m64n32k512(x, weights, batch_info, stack_num=1):
    return _launch_split(
        x, weights, batch_info, stack_num, 8, 64, 32, 512, True
    )


def gcu_split8_m256n64k512(x, weights, batch_info, stack_num=1):
    return _launch_split(
        x, weights, batch_info, stack_num, 8, 256, 64, 512, True
    )


def serial_round_tf32(x, weights, batch_info, stack_num=1):
    return _launch_serial(x, weights, batch_info, stack_num, 1)


def serial_truncate_tf32(x, weights, batch_info, stack_num=1):
    return _launch_serial(x, weights, batch_info, stack_num, 2)


def serial_round_tf32_matrix_only(x, weights, batch_info, stack_num=1):
    return _launch_serial(x, weights, batch_info, stack_num, 3)


def serial_ieee(x, weights, batch_info, stack_num=1):
    return _launch_serial(x, weights, batch_info, stack_num, 0)


def serial_truncate_tf32_matrix_only(x, weights, batch_info, stack_num=1):
    return _launch_serial(x, weights, batch_info, stack_num, 4)


def serial_round_half_up_tf32_matrix_only(x, weights, batch_info, stack_num=1):
    return _launch_serial(x, weights, batch_info, stack_num, 5)


def split8_2d_m64n32k64(x, weights, batch_info, stack_num=1):
    return _launch_split_2d(x, weights, batch_info, stack_num, 8, 64, 32, 64)


def split16_2d_m64n32k64(x, weights, batch_info, stack_num=1):
    return _launch_split_2d(x, weights, batch_info, stack_num, 16, 64, 32, 64)


def split16_2d_m64n32k256(x, weights, batch_info, stack_num=1):
    return _launch_split_2d(x, weights, batch_info, stack_num, 16, 64, 32, 256)


def split16_once_m64n32(x, weights, batch_info, stack_num=1):
    return _launch_split_once(x, weights, batch_info, stack_num, 16, 64, 32)


def split8_once_m64n32(x, weights, batch_info, stack_num=1):
    return _launch_split_once(x, weights, batch_info, stack_num, 8, 64, 32)


def split32_once_m64n32(x, weights, batch_info, stack_num=1):
    return _launch_split_once(x, weights, batch_info, stack_num, 32, 64, 32)


def split64_once_m64n32(x, weights, batch_info, stack_num=1):
    return _launch_split_once(x, weights, batch_info, stack_num, 64, 64, 32)


def chunked_sgmv_shrink(x, weights, batch_info, num_slices=1):
    if x.shape[0] == 0 or batch_info.bs == 0:
        return x.new_zeros((x.shape[0], weights.shape[1]))
    stack_num = num_slices
    expected_tokens = getattr(batch_info, "expected_tokens", None)
    if (
        expected_tokens is not None
        and expected_tokens < x.shape[0]
        and x.dtype != torch.float32
    ):
        return _launch_serial(x, weights, batch_info, stack_num, 0)
    if x.dtype == torch.float32:
        return _launch_serial(x, weights, batch_info, stack_num)
    if x.shape[1] < 1024:
        return _launch(
            x,
            weights,
            batch_info,
            stack_num,
            16,
            32,
            32,
            4,
            2,
            use_default_dot=True,
        )
    if batch_info.max_len <= 8 and weights.shape[1] <= 16:
        return _launch(
            x,
            weights,
            batch_info,
            stack_num,
            16,
            16,
            256,
            4,
            2,
            use_default_dot=True,
        )
    if batch_info.max_len >= 128 and weights.shape[1] <= 64:
        return _launch(
            x,
            weights,
            batch_info,
            stack_num,
            64,
            64,
            256,
            4,
            2,
            use_default_dot=True,
        )
    return _launch(
        x,
        weights,
        batch_info,
        stack_num,
        32,
        32,
        256,
        4,
        2,
        use_default_dot=True,
    )


def sgemm_lora_a_split8(x, weights, batch_info, stack_num=1):
    if x.dtype == torch.float32:
        return m16n32k32w4s2_tf32(x, weights, batch_info, stack_num)
    if batch_info.permutation is not None:
        split_count = 8 if x.shape[1] >= 1024 else 1
        return _launch_split_permuted(
            x,
            weights,
            batch_info,
            stack_num,
            split_count,
            64,
            32,
            64,
        )
    if x.shape[1] >= 1024:
        return split8_m64n32k64(x, weights, batch_info, stack_num)
    return split1_m64n32k64(x, weights, batch_info, stack_num)


class _EvalBatchInfo:
    pass


def run(
    x,
    weights,
    seg_indptr,
    weight_indices,
    permutation,
    bs,
    max_len,
    expected_tokens,
    num_slices=1,
):
    """Local evaluator adapter; the submission entrypoint remains chunked_sgmv_shrink."""
    info = _EvalBatchInfo()
    info.seg_indptr = seg_indptr
    info.weight_indices = weight_indices
    info.permutation = permutation
    info.bs = int(bs)
    info.max_len = int(max_len)
    info.expected_tokens = int(expected_tokens)
    return chunked_sgmv_shrink(x, weights, info, num_slices)
