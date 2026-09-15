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
def _gate_up_lora_b_kernel(
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
    blocks_m: tl.constexpr,
    blocks_n: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    block_k: tl.constexpr,
    has_permutation: tl.constexpr,
    use_dot: tl.constexpr,
):
    segment_block = tl.program_id(0)
    segment = segment_block // blocks_m
    row_block = segment_block - segment * blocks_m
    slice_block = tl.program_id(1)
    slice_index = slice_block // blocks_n
    local_block = slice_block - slice_index * blocks_n
    segment_start = tl.load(seg_indptr + segment)
    segment_end = tl.load(seg_indptr + segment + 1)
    positions = segment_start + row_block * block_m + tl.arange(0, block_m)
    mask_m = positions < segment_end
    if has_permutation:
        rows = tl.load(permutation + positions, mask=mask_m, other=0)
    else:
        rows = positions
    offsets_n = local_block * block_n + tl.arange(0, block_n)
    mask_n = offsets_n < output_dim
    columns = slice_index * output_dim + offsets_n
    offsets_k = tl.arange(0, block_k)
    adapter = tl.load(weight_indices + segment)
    active_rank = tl.load(lora_ranks + adapter)
    scaling = tl.where(active_rank == 0, 0.0, tl.load(scalings + adapter))
    if use_dot:
        x_values = tl.load(
            x
            + rows[:, None] * (2 * rank)
            + slice_index * rank
            + offsets_k[None, :],
            mask=mask_m[:, None] & (offsets_k[None, :] < rank),
            other=0.0,
        )
        weight_values = tl.load(
            weights
            + adapter * (2 * output_dim * rank)
            + columns[None, :] * rank
            + offsets_k[:, None],
            mask=(offsets_k[:, None] < rank) & mask_n[None, :],
            other=0.0,
        )
        update = tl.dot(
            x_values,
            weight_values,
            input_precision="ieee",
            out_dtype=tl.float32,
        )
    else:
        update = tl.zeros((block_m, block_n), dtype=tl.float32)
        for rank_offset in tl.static_range(0, rank):
            x_values = tl.load(
                x + rows * (2 * rank) + slice_index * rank + rank_offset,
                mask=mask_m,
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
            update += x_values[:, None] * weight_values[None, :]
    output_offsets = rows[:, None] * (2 * output_dim) + columns[None, :]
    output_mask = mask_m[:, None] & mask_n[None, :]
    base = tl.load(
        base_output + output_offsets, mask=output_mask, other=0.0
    ).to(tl.float32)
    tl.store(
        output + output_offsets, base + scaling * update, mask=output_mask
    )


def _launch(
    x,
    weights,
    batch_info,
    output_dim,
    base_output,
    block_m,
    block_n,
    num_warps,
):
    rank = weights.shape[-1]
    max_segment_length = max(1, batch_info.max_len)
    blocks_m = triton.cdiv(max_segment_length, block_m)
    blocks_n = triton.cdiv(output_dim, block_n)
    output = torch.empty_like(base_output)
    permutation = batch_info.permutation
    _gate_up_lora_b_kernel[(batch_info.bs * blocks_m, 2 * blocks_n)](
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
        blocks_m,
        blocks_n,
        block_m,
        block_n,
        triton.next_power_of_2(rank),
        has_permutation=permutation is not None,
        use_dot=x.element_size() == 2,
        num_warps=num_warps,
        num_stages=1,
    )
    return output


def gate_up_lora_b(x, gate_up_lora_b, batch_info, output_dim, base_output):
    if batch_info.max_len >= 128:
        return _launch(
            x, gate_up_lora_b, batch_info, output_dim, base_output, 128, 256, 8
        )
    if (
        output_dim == 4096
        and gate_up_lora_b.shape[-1] == 32
        and x.element_size() == 2
        and batch_info.permutation is None
    ):
        return _launch(
            x, gate_up_lora_b, batch_info, output_dim, base_output, 64, 64, 8
        )
    return _launch(
        x, gate_up_lora_b, batch_info, output_dim, base_output, 64, 128, 8
    )


__all__ = ["gate_up_lora_b"]
