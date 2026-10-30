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


@triton.jit
def _expand_tile(
    x_ptr,
    weights_ptr,
    base_ptr,
    output_ptr,
    scaling_ptr,
    rows,
    token_mask,
    out_start,
    output_size,
    output_offsets,
    weight_index,
    rank_is_zero,
    slice_id,
    x_stride_token,
    x_stride_rank,
    weight_stride_lora,
    weight_stride_output,
    weight_stride_rank,
    base_stride_token,
    base_stride_col,
    output_stride_token,
    output_stride_col,
    RANK: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    USE_DOT: tl.constexpr,
):
    output_mask = output_offsets < output_size
    mask = token_mask[:, None] & output_mask[None, :]
    columns = out_start + output_offsets
    base = tl.load(
        base_ptr
        + rows[:, None] * base_stride_token
        + columns[None, :] * base_stride_col,
        mask=mask,
        other=0.0,
    )
    output_ptrs = (
        output_ptr
        + rows[:, None] * output_stride_token
        + columns[None, :] * output_stride_col
    )
    if rank_is_zero:
        tl.store(output_ptrs, base, mask=mask)
    else:
        accumulator = tl.zeros((BLOCK_S, BLOCK_N), dtype=tl.float32)
        if USE_DOT:
            k_offsets = tl.arange(0, BLOCK_K)
            for k_start in range(0, RANK, BLOCK_K):
                k = k_start + k_offsets
                k_mask = k < RANK
                x = tl.load(
                    x_ptr
                    + rows[:, None] * x_stride_token
                    + (slice_id * RANK + k[None, :]) * x_stride_rank,
                    mask=token_mask[:, None] & k_mask[None, :],
                    other=0.0,
                )
                weights = tl.load(
                    weights_ptr
                    + weight_index * weight_stride_lora
                    + columns[None, :] * weight_stride_output
                    + k[:, None] * weight_stride_rank,
                    mask=k_mask[:, None] & output_mask[None, :],
                    other=0.0,
                )
                accumulator += tl.dot(x, weights, input_precision="ieee")
            # Without this round trip the AMD backend loads base_output and
            # stores the result in the WMMA layout, two bytes a lane;
            # splitting and joining the accumulator makes it convert the
            # accumulator first. The values are unchanged.
            low, high = tl.split(
                tl.reshape(accumulator, (BLOCK_S, BLOCK_N // 2, 2))
            )
            accumulator = tl.reshape(tl.join(low, high), (BLOCK_S, BLOCK_N))
            scaling = tl.load(scaling_ptr).to(tl.float32)
            result = tl.fma(accumulator, scaling, base.to(tl.float32))
        else:
            for k in range(0, RANK):
                x = tl.load(
                    x_ptr
                    + rows * x_stride_token
                    + (slice_id * RANK + k) * x_stride_rank,
                    mask=token_mask,
                    other=0.0,
                ).to(tl.float32)
                weights = tl.load(
                    weights_ptr
                    + weight_index * weight_stride_lora
                    + columns * weight_stride_output
                    + k * weight_stride_rank,
                    mask=output_mask,
                    other=0.0,
                ).to(tl.float32)
                accumulator = tl.fma(x[:, None], weights[None, :], accumulator)
            scaling = tl.load(scaling_ptr).to(tl.float32)
            result = base.to(tl.float32) + accumulator * scaling
        tl.store(
            output_ptrs, result.to(output_ptr.dtype.element_ty), mask=mask
        )


@triton.jit
def _chunked_sgmv_expand_kernel(
    x_ptr,
    weights_ptr,
    base_ptr,
    output_ptr,
    seg_indptr_ptr,
    weight_indices_ptr,
    lora_ranks_ptr,
    scalings_ptr,
    permutation_ptr,
    slice_offsets_ptr,
    batch_size,
    n_rows,
    out_dim,
    max_out_dim,
    x_stride_token,
    x_stride_rank,
    weight_stride_lora,
    weight_stride_output,
    weight_stride_rank,
    base_stride_token,
    base_stride_col,
    output_stride_token,
    output_stride_col,
    seg_indptr_stride,
    weight_indices_stride,
    lora_ranks_stride,
    scalings_stride,
    permutation_stride,
    slice_offsets_stride,
    N_SLICES: tl.constexpr,
    RANK: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    USE_DOT: tl.constexpr,
):
    x_stride_token = tl.cast(x_stride_token, tl.int64)
    x_stride_rank = tl.cast(x_stride_rank, tl.int64)
    weight_stride_lora = tl.cast(weight_stride_lora, tl.int64)
    weight_stride_output = tl.cast(weight_stride_output, tl.int64)
    weight_stride_rank = tl.cast(weight_stride_rank, tl.int64)
    base_stride_token = tl.cast(base_stride_token, tl.int64)
    base_stride_col = tl.cast(base_stride_col, tl.int64)
    output_stride_token = tl.cast(output_stride_token, tl.int64)
    output_stride_col = tl.cast(output_stride_col, tl.int64)

    batch_id = tl.program_id(2)
    slice_id = tl.program_id(1)
    matrix_pid = tl.program_id(0)
    if batch_id < batch_size:
        segment_start = tl.load(seg_indptr_ptr + batch_id * seg_indptr_stride)
        segment_length = (
            tl.load(seg_indptr_ptr + (batch_id + 1) * seg_indptr_stride)
            - segment_start
        )
        out_start = tl.load(
            slice_offsets_ptr + slice_id * slice_offsets_stride
        )
        output_size = (
            tl.load(slice_offsets_ptr + (slice_id + 1) * slice_offsets_stride)
            - out_start
        )
        num_output_blocks = tl.cdiv(max_out_dim, BLOCK_N)
        token_block = matrix_pid // num_output_blocks
        output_block = matrix_pid - token_block * num_output_blocks
        if token_block * BLOCK_S >= segment_length:
            return
        if output_block * BLOCK_N >= output_size:
            return
        weight_index = tl.load(
            weight_indices_ptr + batch_id * weight_indices_stride
        )
        rank_is_zero = (
            tl.load(lora_ranks_ptr + weight_index * lora_ranks_stride) == 0
        )
        scaling_ptr = scalings_ptr + weight_index * scalings_stride
        token_offsets = token_block * BLOCK_S + tl.arange(0, BLOCK_S)
        output_offsets = output_block * BLOCK_N + tl.arange(0, BLOCK_N)
        token_mask = token_offsets < segment_length
        rows = tl.load(
            permutation_ptr
            + (segment_start + token_offsets) * permutation_stride,
            mask=token_mask,
            other=0,
        )
        if (out_start % 8 == 0) & (output_size % 8 == 0):
            # Slices that start and end on multiples of 8 columns: the values
            # are recomputed as such so that the compiler knows it here and
            # vectorizes the base_output loads and the stores (tl.multiple_of
            # would also mark the other branch).
            _expand_tile(
                x_ptr,
                weights_ptr,
                base_ptr,
                output_ptr,
                scaling_ptr,
                rows,
                token_mask,
                out_start // 8 * 8,
                output_size // 8 * 8,
                output_offsets,
                weight_index,
                rank_is_zero,
                slice_id,
                x_stride_token,
                x_stride_rank,
                weight_stride_lora,
                weight_stride_output,
                weight_stride_rank,
                base_stride_token,
                base_stride_col,
                output_stride_token,
                output_stride_col,
                RANK,
                BLOCK_S,
                BLOCK_N,
                BLOCK_K,
                USE_DOT,
            )
        else:
            _expand_tile(
                x_ptr,
                weights_ptr,
                base_ptr,
                output_ptr,
                scaling_ptr,
                rows,
                token_mask,
                out_start,
                output_size,
                output_offsets,
                weight_index,
                rank_is_zero,
                slice_id,
                x_stride_token,
                x_stride_rank,
                weight_stride_lora,
                weight_stride_output,
                weight_stride_rank,
                base_stride_token,
                base_stride_col,
                output_stride_token,
                output_stride_col,
                RANK,
                BLOCK_S,
                BLOCK_N,
                BLOCK_K,
                USE_DOT,
            )
    else:
        # The last row of the grid copies what no segment and slice covers:
        # rows after the last segment and columns outside the slices. This
        # relies on the permutation mapping the segment positions onto rows
        # [0, seg_indptr[batch_size]), as a sort of the batch's tokens does.
        covered_rows = tl.load(seg_indptr_ptr + batch_size * seg_indptr_stride)
        first_col = tl.load(slice_offsets_ptr)
        last_col = tl.load(slice_offsets_ptr + N_SLICES * slice_offsets_stride)
        if (covered_rows >= n_rows) & (first_col <= 0) & (last_col >= out_dim):
            return
        row_tiles = tl.cdiv(n_rows, BLOCK_S)
        col_tiles = tl.cdiv(out_dim, BLOCK_N)
        for tile in range(
            matrix_pid * N_SLICES + slice_id,
            row_tiles * col_tiles,
            tl.num_programs(0) * N_SLICES,
        ):
            r = (tile // col_tiles) * BLOCK_S + tl.arange(0, BLOCK_S)
            c = (tile % col_tiles) * BLOCK_N + tl.arange(0, BLOCK_N)
            rest = (r >= covered_rows)[:, None] | (
                (c < first_col) | (c >= last_col)
            )[None, :]
            mask = (r < n_rows)[:, None] & (c < out_dim)[None, :] & rest
            r64 = r.to(tl.int64)
            values = tl.load(
                base_ptr
                + r64[:, None] * base_stride_token
                + c[None, :] * base_stride_col,
                mask=mask,
            )
            tl.store(
                output_ptr
                + r64[:, None] * output_stride_token
                + c[None, :] * output_stride_col,
                values,
                mask=mask,
            )


def chunked_sgmv_expand(
    x, weights, batch_info, slice_offsets, max_slice_size, base_output
):
    n_slices = slice_offsets.numel() - 1
    rank = weights.shape[-1]
    if x.shape[1] != n_slices * rank:
        raise ValueError("x width must equal n_slices * rank")
    if (
        base_output.numel() == 0
        or n_slices <= 0
        or batch_info.bs == 0
        or x.shape[0] == 0
    ):
        return base_output.clone()

    # The task's batch_info lists no max_len hint; use it when the
    # harness object provides one, otherwise pay a single host sync.
    seg_indptr = batch_info.seg_indptr
    max_len = getattr(batch_info, "max_len", None)
    if max_len is None:
        max_len = int((seg_indptr[1:] - seg_indptr[:-1]).max().item())
    if max_len == 0:
        return base_output.clone()

    # The kernel writes every output element instead of adding into a clone
    # of base_output. gfx1100: 16-bit inputs take 8 warps; fp32 runs the
    # rank loop as FMAs on smaller tiles. Without FP contraction, the kernel
    # rounds as the generic op compiles on gfx1100 whatever the shapes: one
    # FMA in the 16-bit epilogue, FMAs in the fp32 rank loop and a multiply
    # and an add in the fp32 epilogue.
    use_dot = x.dtype != torch.float32
    if use_dot:
        block_s = 32 if max_len <= 64 else 64
        block_n, num_warps = 128, 8
    elif max_len <= 32:
        block_s, block_n, num_warps = 16, 64, 16
    else:
        block_s = 32 if max_len <= 64 else 64
        block_n, num_warps = 128, 16 if max_len <= 64 else 8
    output = torch.empty_like(base_output)
    grid = (
        triton.cdiv(max_len, block_s)
        * triton.cdiv(int(max_slice_size), block_n),
        n_slices,
        batch_info.bs + 1,
    )
    _chunked_sgmv_expand_kernel[grid](
        x,
        weights,
        base_output,
        output,
        seg_indptr,
        batch_info.weight_indices,
        batch_info.lora_ranks,
        batch_info.scalings,
        batch_info.permutation,
        slice_offsets,
        batch_info.bs,
        base_output.shape[0],
        base_output.shape[1],
        int(max_slice_size),
        *x.stride(),
        *weights.stride(),
        *base_output.stride(),
        *output.stride(),
        seg_indptr.stride(0),
        batch_info.weight_indices.stride(0),
        batch_info.lora_ranks.stride(0),
        batch_info.scalings.stride(0),
        batch_info.permutation.stride(0),
        slice_offsets.stride(0),
        N_SLICES=n_slices,
        RANK=rank,
        BLOCK_S=block_s,
        BLOCK_N=block_n,
        BLOCK_K=32,
        USE_DOT=use_dot,
        num_warps=num_warps,
        num_stages=2,
        enable_fp_fusion=False,
    )
    return output


__all__ = ["chunked_sgmv_expand"]
