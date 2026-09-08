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


@triton.jit
def _qkv_lora_b_safe_kernel(
    x_ptr,
    weights_ptr,
    seg_indptr_ptr,
    weight_indices_ptr,
    lora_ranks_ptr,
    scalings_ptr,
    permutation_ptr,
    output_offsets_ptr,
    output_ptr,
    STRIDE_X_ROW: tl.constexpr,
    STRIDE_X_COL: tl.constexpr,
    STRIDE_W_ADAPTER: tl.constexpr,
    STRIDE_W_OUT: tl.constexpr,
    STRIDE_W_RANK: tl.constexpr,
    STRIDE_OUT_ROW: tl.constexpr,
    STRIDE_OUT_COL: tl.constexpr,
    STRIDE_SEG: tl.constexpr,
    STRIDE_WIDX: tl.constexpr,
    STRIDE_RANKS: tl.constexpr,
    STRIDE_PERM: tl.constexpr,
    STRIDE_OFFSETS: tl.constexpr,
    S: tl.constexpr,
    R: tl.constexpr,
    NUM_SLICES: tl.constexpr,
    N_TILES: tl.constexpr,
    HAS_PERMUTATION: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    batch_slice_id = tl.program_id(1)
    batch_id = batch_slice_id // NUM_SLICES
    slice_id = batch_slice_id % NUM_SLICES
    tile_s = tl.program_id(0) // N_TILES
    tile_n = tl.program_id(0) % N_TILES

    segment_start = tl.load(seg_indptr_ptr + batch_id * STRIDE_SEG)
    segment_end = tl.load(seg_indptr_ptr + (batch_id + 1) * STRIDE_SEG)
    weight_index = tl.load(weight_indices_ptr + batch_id * STRIDE_WIDX)
    rank = tl.load(lora_ranks_ptr + weight_index * STRIDE_RANKS)
    scaling = tl.load(scalings_ptr + weight_index)
    output_start = tl.load(output_offsets_ptr + slice_id * STRIDE_OFFSETS)
    output_end = tl.load(output_offsets_ptr + (slice_id + 1) * STRIDE_OFFSETS)
    output_size = output_end - output_start

    logical_rows = segment_start + tile_s * BLOCK_S + tl.arange(0, BLOCK_S)
    live = (logical_rows < segment_end) & (logical_rows < S) & (rank > 0)
    if HAS_PERMUTATION:
        rows = tl.load(
            permutation_ptr + logical_rows * STRIDE_PERM,
            mask=live,
            other=0,
        )
    else:
        rows = logical_rows

    offsets_n = tile_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offsets_r = tl.arange(0, BLOCK_R)
    in_n = offsets_n < output_size
    in_r = offsets_r < R

    x_tile = tl.load(
        x_ptr
        + rows[:, None] * STRIDE_X_ROW
        + (slice_id * R + offsets_r[None, :]) * STRIDE_X_COL,
        mask=live[:, None] & in_r[None, :],
        other=0.0,
    )
    weight_tile = tl.load(
        weights_ptr
        + weight_index * STRIDE_W_ADAPTER
        + offsets_r[:, None] * STRIDE_W_RANK
        + (output_start + offsets_n[None, :]) * STRIDE_W_OUT,
        mask=in_r[:, None] & in_n[None, :],
        other=0.0,
    )
    accumulator = tl.dot(x_tile, weight_tile, input_precision="ieee") * scaling

    output_addresses = (
        output_ptr
        + rows[:, None] * STRIDE_OUT_ROW
        + (output_start + offsets_n[None, :]) * STRIDE_OUT_COL
    )
    output_mask = live[:, None] & in_n[None, :]
    base = tl.load(output_addresses, mask=output_mask, other=0.0).to(
        tl.float32
    )
    tl.store(output_addresses, base + accumulator, mask=output_mask)


def _as_index(tensor):
    if tensor.dtype != torch.int64:
        return tensor, tensor.stride(0)
    tensor = tensor.to(torch.int32)
    return tensor, tensor.stride(0)


def _block_r_for(rank):
    if rank <= 16:
        return 16
    if rank <= 32:
        return 32
    if rank <= 64:
        return 64
    if rank <= 128:
        return 128
    return 256


def qkv_lora_b(
    x, qkv_lora_b, batch_info, output_offset, max_qkv_out_dim, base_output
):
    seg_indptr, stride_seg = _as_index(batch_info.seg_indptr)
    weight_indices, stride_weight_indices = _as_index(
        batch_info.weight_indices
    )
    lora_ranks, stride_lora_ranks = _as_index(batch_info.lora_ranks)
    output_offsets, stride_output_offsets = _as_index(output_offset)
    if batch_info.permutation is None:
        permutation = seg_indptr
        stride_permutation = stride_seg
    else:
        permutation, stride_permutation = _as_index(batch_info.permutation)

    output = base_output.clone()
    sequence_length = x.shape[0]
    rank = qkv_lora_b.shape[-1]
    batch_size = int(batch_info.bs)
    num_slices = output_offset.numel() - 1
    segment_lengths = seg_indptr[1 : batch_size + 1] - seg_indptr[:batch_size]
    max_segment_length = int(segment_lengths.max()) if batch_size > 0 else 0
    block_s = 16
    block_n = 64
    block_r = _block_r_for(rank)
    tiles_s = -(-max_segment_length // block_s)
    tiles_n = -(-int(max_qkv_out_dim) // block_n)
    grid = (max(tiles_s * tiles_n, 1), max(batch_size * num_slices, 1))
    stride_x = x.stride()
    stride_weights = qkv_lora_b.stride()
    stride_output = output.stride()

    _qkv_lora_b_safe_kernel[grid](
        x,
        qkv_lora_b,
        seg_indptr,
        weight_indices,
        lora_ranks,
        batch_info.scalings,
        permutation,
        output_offsets,
        output,
        stride_x[0],
        stride_x[1],
        stride_weights[0],
        stride_weights[1],
        stride_weights[2],
        stride_output[0],
        stride_output[1],
        stride_seg,
        stride_weight_indices,
        stride_lora_ranks,
        stride_permutation,
        stride_output_offsets,
        sequence_length,
        rank,
        num_slices,
        tiles_n,
        batch_info.permutation is not None,
        block_s,
        block_n,
        block_r,
        num_warps=4,
        num_stages=1,
    )
    return output


__all__ = ["qkv_lora_b"]
