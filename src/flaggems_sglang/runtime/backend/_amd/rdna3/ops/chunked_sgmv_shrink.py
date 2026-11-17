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

"""chunked_sgmv_shrink for RDNA3 (gfx1100) (Triton).

``output[rows] = x[rows] @ weights[adapter].T`` for every segment of the
batch, with a rank of a few dozen columns and a long hidden dimension.

For 16-bit inputs, a hidden dimension of up to 1024 takes one kernel whose
last programs zero the rows no segment covers. A longer one is split over
several programs and a second kernel adds the partial sums, as in the generic
op, but the split follows the batch (``_split_config``) instead of a fixed 16
splits and the output is not zero-filled first when the second kernel writes
all of it. The shapes are runtime arguments, so that a new ``max_len`` does
not compile new kernels.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _shrink_split_kernel(
    x_ptr,
    w_ptr,
    seg_indptr_ptr,
    weight_indices_ptr,
    permutation_ptr,
    partial_ptr,
    hidden_size,
    output_size,
    split_size,
    tiles_per_segment,
    stride_x_row,
    stride_w_adapter,
    stride_w_output,
    stride_partial_split,
    stride_partial_row,
    SPLIT_COUNT: tl.constexpr,
    HAS_PERMUTATION: tl.constexpr,
    EVEN_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    output_tile = tl.program_id(1)
    split = pid % SPLIT_COUNT
    segment_tile = pid // SPLIT_COUNT
    segment = segment_tile // tiles_per_segment
    tile_m = segment_tile - segment * tiles_per_segment
    start = tl.load(seg_indptr_ptr + segment)
    end = tl.load(seg_indptr_ptr + segment + 1)
    if start + tile_m * BLOCK_M >= end:
        return
    adapter = tl.load(weight_indices_ptr + segment).to(tl.int64)
    positions = start + tile_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = positions < end
    if HAS_PERMUTATION:
        rows = tl.load(permutation_ptr + positions, mask=mask_m, other=0)
    else:
        rows = positions
    offsets_n = output_tile * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offsets_n < output_size
    offsets_k = tl.arange(0, BLOCK_K)
    k_begin = split * split_size
    k_end = tl.minimum(k_begin + split_size, hidden_size)
    x_ptrs = (
        x_ptr + rows.to(tl.int64)[:, None] * stride_x_row + offsets_k[None, :]
    )
    w_ptrs = (
        w_ptr
        + adapter * stride_w_adapter
        + offsets_n[None, :] * stride_w_output
        + offsets_k[:, None]
    )
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(k_begin, k_end, BLOCK_K):
        if EVEN_K:
            x = tl.load(x_ptrs + k, mask=mask_m[:, None], other=0.0)
            w = tl.load(w_ptrs + k, mask=mask_n[None, :], other=0.0)
        else:
            mask_k = offsets_k + k < k_end
            x = tl.load(
                x_ptrs + k, mask=mask_m[:, None] & mask_k[None, :], other=0.0
            )
            w = tl.load(
                w_ptrs + k, mask=mask_k[:, None] & mask_n[None, :], other=0.0
            )
        accumulator = tl.dot(x, w, accumulator)
    partial_ptrs = (
        partial_ptr
        + split * stride_partial_split
        + positions[:, None] * stride_partial_row
        + offsets_n[None, :]
    )
    tl.store(partial_ptrs, accumulator, mask=mask_m[:, None] & mask_n[None, :])


@triton.jit
def _shrink_reduce_kernel(
    partial_ptr,
    output_ptr,
    seg_indptr_ptr,
    permutation_ptr,
    token_count,
    output_size,
    batch_size,
    stride_partial_split,
    stride_partial_row,
    stride_output_row,
    SPLIT_COUNT: tl.constexpr,
    HAS_PERMUTATION: tl.constexpr,
    WRITE_ALL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    covered = rows < tl.load(seg_indptr_ptr + batch_size)
    mask_n = cols < output_size
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for split in tl.static_range(SPLIT_COUNT):
        accumulator += tl.load(
            partial_ptr
            + split * stride_partial_split
            + rows[:, None] * stride_partial_row
            + cols[None, :],
            mask=covered[:, None] & mask_n[None, :],
            other=0.0,
        )
    # With WRITE_ALL every row of the output is written, the uncovered ones
    # with zeros, so the output needs no zero fill first.
    if WRITE_ALL:
        write = rows < token_count
    else:
        write = covered
    if HAS_PERMUTATION:
        out_rows = tl.load(permutation_ptr + rows, mask=write, other=0)
    else:
        out_rows = rows
    tl.store(
        output_ptr
        + out_rows.to(tl.int64)[:, None] * stride_output_row
        + cols[None, :],
        accumulator.to(output_ptr.dtype.element_ty),
        mask=write[:, None] & mask_n[None, :],
    )


@triton.jit
def _shrink_single_kernel(
    x_ptr,
    w_ptr,
    seg_indptr_ptr,
    weight_indices_ptr,
    permutation_ptr,
    output_ptr,
    hidden_size,
    output_size,
    token_count,
    batch_size,
    tiles_per_segment,
    stride_x_row,
    stride_w_adapter,
    stride_w_output,
    stride_output_row,
    HAS_PERMUTATION: tl.constexpr,
    EVEN_K: tl.constexpr,
    ZERO_PROGRAMS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    output_tile = tl.program_id(1)
    offsets_n = output_tile * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offsets_n < output_size
    segment_tiles = batch_size * tiles_per_segment
    if ZERO_PROGRAMS > 0:
        if pid >= segment_tiles:
            # The last ZERO_PROGRAMS programs zero the rows no segment covers.
            covered = tl.load(seg_indptr_ptr + batch_size)
            zeros = tl.zeros(
                (BLOCK_M, BLOCK_N), dtype=output_ptr.dtype.element_ty
            )
            for start in range(
                covered + (pid - segment_tiles) * BLOCK_M,
                token_count,
                ZERO_PROGRAMS * BLOCK_M,
            ):
                positions = start + tl.arange(0, BLOCK_M)
                mask_m = positions < token_count
                if HAS_PERMUTATION:
                    rows = tl.load(permutation_ptr + positions, mask=mask_m)
                else:
                    rows = positions
                tl.store(
                    output_ptr
                    + rows.to(tl.int64)[:, None] * stride_output_row
                    + offsets_n[None, :],
                    zeros,
                    mask=mask_m[:, None] & mask_n[None, :],
                )
            return
    segment = pid // tiles_per_segment
    tile_m = pid - segment * tiles_per_segment
    start = tl.load(seg_indptr_ptr + segment)
    end = tl.load(seg_indptr_ptr + segment + 1)
    if start + tile_m * BLOCK_M >= end:
        return
    adapter = tl.load(weight_indices_ptr + segment).to(tl.int64)
    positions = start + tile_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = positions < end
    if HAS_PERMUTATION:
        rows = tl.load(permutation_ptr + positions, mask=mask_m, other=0)
    else:
        rows = positions
    rows = rows.to(tl.int64)
    offsets_k = tl.arange(0, BLOCK_K)
    x_ptrs = x_ptr + rows[:, None] * stride_x_row + offsets_k[None, :]
    w_ptrs = (
        w_ptr
        + adapter * stride_w_adapter
        + offsets_n[None, :] * stride_w_output
        + offsets_k[:, None]
    )
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, hidden_size, BLOCK_K):
        if EVEN_K:
            x = tl.load(x_ptrs + k, mask=mask_m[:, None], other=0.0)
            w = tl.load(w_ptrs + k, mask=mask_n[None, :], other=0.0)
        else:
            mask_k = offsets_k + k < hidden_size
            x = tl.load(
                x_ptrs + k, mask=mask_m[:, None] & mask_k[None, :], other=0.0
            )
            w = tl.load(
                w_ptrs + k, mask=mask_k[:, None] & mask_n[None, :], other=0.0
            )
        accumulator = tl.dot(x, w, accumulator)
    tl.store(
        output_ptr + rows[:, None] * stride_output_row + offsets_n[None, :],
        accumulator.to(output_ptr.dtype.element_ty),
        mask=mask_m[:, None] & mask_n[None, :],
    )


@triton.jit
def _shrink_direct_kernel(
    x_ptr,
    w_ptr,
    seg_indptr_ptr,
    weight_indices_ptr,
    permutation_ptr,
    output_ptr,
    hidden_size,
    output_size,
    tiles_per_segment,
    stride_x_row,
    stride_w_adapter,
    stride_w_output,
    stride_output_row,
    HAS_PERMUTATION: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    segment_tile = tl.program_id(0)
    output_tile = tl.program_id(1)
    segment = segment_tile // tiles_per_segment
    tile_m = segment_tile - segment * tiles_per_segment
    start = tl.load(seg_indptr_ptr + segment)
    end = tl.load(seg_indptr_ptr + segment + 1)
    if start + tile_m * BLOCK_M >= end:
        return
    adapter = tl.load(weight_indices_ptr + segment).to(tl.int64)
    positions = start + tile_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = positions < end
    if HAS_PERMUTATION:
        rows = tl.load(permutation_ptr + positions, mask=mask_m, other=0)
    else:
        rows = positions
    rows = rows.to(tl.int64)
    offsets_n = output_tile * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offsets_n < output_size
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, hidden_size, BLOCK_K):
        offsets_k = k + tl.arange(0, BLOCK_K)
        mask_k = offsets_k < hidden_size
        x = tl.load(
            x_ptr + rows[:, None] * stride_x_row + offsets_k[None, :],
            mask=mask_m[:, None] & mask_k[None, :],
            other=0.0,
        )
        w = tl.load(
            w_ptr
            + adapter * stride_w_adapter
            + offsets_n[None, :] * stride_w_output
            + offsets_k[:, None],
            mask=mask_k[:, None] & mask_n[None, :],
            other=0.0,
        )
        accumulator = tl.dot(x, w, accumulator, input_precision="ieee")
    tl.store(
        output_ptr + rows[:, None] * stride_output_row + offsets_n[None, :],
        accumulator.to(output_ptr.dtype.element_ty),
        mask=mask_m[:, None] & mask_n[None, :],
    )


def _single_config(token_count, max_len, bs, hidden_size, output_size):
    """(BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages) of the one-kernel
    path, or None for the split path."""
    # A short hidden dimension is a few K steps a program, and one launch
    # instead of two is most of the time at these sizes. A long one with few
    # tokens leaves too few programs walking too many K steps.
    if hidden_size > 1024:
        return None
    block_n = 32 if output_size <= 32 else 64
    block_k = min(256, max(16, triton.next_power_of_2(hidden_size)))
    return 16, block_n, block_k, 4, 2


def _split_config(token_count, max_len, bs, hidden_size, output_size):
    """(K blocks a split, BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages)."""
    block_m = 16
    block_n = 32 if output_size <= 32 else 64
    block_k = 128 if hidden_size % 128 == 0 else 64
    k_blocks = triton.cdiv(hidden_size, block_k)
    tiles = (
        bs * triton.cdiv(max_len, block_m) * triton.cdiv(output_size, block_n)
    )
    if tiles >= 512:
        blocks = max(3, triton.cdiv(k_blocks, 2))
    elif tiles >= 192:
        blocks = 5
    else:
        blocks = 3
    if output_size > 96:
        # Every split writes output_size float32 partial sums a row.
        blocks = max(blocks, triton.cdiv(k_blocks, 8))
    # Splits that start a power of two of K blocks apart ran up to 3x slower
    # on gfx1100 (e.g. 4 or 8 splits of K = 4096).
    if blocks & (blocks - 1) == 0:
        blocks += 1
    return blocks, block_m, block_n, block_k, 4, 1


def chunked_sgmv_shrink(x, weights, batch_info, num_slices=1):
    token_count, hidden_size = x.shape
    output_size = weights.shape[1]
    if token_count == 0 or batch_info.bs == 0:
        return x.new_zeros((token_count, output_size))
    if x.stride(1) != 1:
        x = x.contiguous()
    if weights.stride(2) != 1:
        weights = weights.contiguous()
    max_len = max(1, batch_info.max_len)
    permutation = batch_info.permutation
    has_permutation = permutation is not None
    if x.dtype == torch.float32:
        # float32 products run in IEEE mode on the FMA units, as in the
        # generic op.
        block_m, block_n, block_k = 16, 32, 32
        output = torch.zeros(
            (token_count, output_size), dtype=x.dtype, device=x.device
        )
        tiles = triton.cdiv(max_len, block_m)
        _shrink_direct_kernel[
            (batch_info.bs * tiles, triton.cdiv(output_size, block_n))
        ](
            x,
            weights,
            batch_info.seg_indptr,
            batch_info.weight_indices,
            permutation if has_permutation else batch_info.seg_indptr,
            output,
            hidden_size,
            output_size,
            tiles,
            x.stride(0),
            weights.stride(0),
            weights.stride(1),
            output.stride(0),
            HAS_PERMUTATION=has_permutation,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_K=block_k,
            num_warps=4,
            num_stages=2,
        )
        return output

    write_all = not has_permutation or permutation.numel() == token_count
    single = _single_config(
        token_count, max_len, batch_info.bs, hidden_size, output_size
    )
    if single is not None:
        block_m, block_n, block_k, warps, stages = single
        output = (torch.empty if write_all else torch.zeros)(
            (token_count, output_size), dtype=x.dtype, device=x.device
        )
        tiles = triton.cdiv(max_len, block_m)
        zero_programs = 4 if write_all else 0
        _shrink_single_kernel[
            (
                batch_info.bs * tiles + zero_programs,
                triton.cdiv(output_size, block_n),
            )
        ](
            x,
            weights,
            batch_info.seg_indptr,
            batch_info.weight_indices,
            permutation if has_permutation else batch_info.seg_indptr,
            output,
            hidden_size,
            output_size,
            token_count,
            batch_info.bs,
            tiles,
            x.stride(0),
            weights.stride(0),
            weights.stride(1),
            output.stride(0),
            HAS_PERMUTATION=has_permutation,
            EVEN_K=hidden_size % block_k == 0,
            ZERO_PROGRAMS=zero_programs,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_K=block_k,
            num_warps=warps,
            num_stages=stages,
        )
        return output
    blocks, block_m, block_n, block_k, warps, stages = _split_config(
        token_count, max_len, batch_info.bs, hidden_size, output_size
    )
    split_size = blocks * block_k
    split_count = triton.cdiv(hidden_size, split_size)
    output = (torch.empty if write_all else torch.zeros)(
        (token_count, output_size), dtype=x.dtype, device=x.device
    )
    partial = torch.empty(
        (split_count, token_count, output_size),
        dtype=torch.float32,
        device=x.device,
    )
    tiles = triton.cdiv(max_len, block_m)
    _shrink_split_kernel[
        (
            batch_info.bs * tiles * split_count,
            triton.cdiv(output_size, block_n),
        )
    ](
        x,
        weights,
        batch_info.seg_indptr,
        batch_info.weight_indices,
        permutation if has_permutation else batch_info.seg_indptr,
        partial,
        hidden_size,
        output_size,
        split_size,
        tiles,
        x.stride(0),
        weights.stride(0),
        weights.stride(1),
        partial.stride(0),
        partial.stride(1),
        SPLIT_COUNT=split_count,
        HAS_PERMUTATION=has_permutation,
        EVEN_K=hidden_size % block_k == 0,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_warps=warps,
        num_stages=stages,
    )
    reduce_m, reduce_n = 32, triton.next_power_of_2(min(output_size, 64))
    _shrink_reduce_kernel[
        (
            triton.cdiv(token_count, reduce_m),
            triton.cdiv(output_size, reduce_n),
        )
    ](
        partial,
        output,
        batch_info.seg_indptr,
        permutation if has_permutation else batch_info.seg_indptr,
        token_count,
        output_size,
        batch_info.bs,
        partial.stride(0),
        partial.stride(1),
        output.stride(0),
        SPLIT_COUNT=split_count,
        HAS_PERMUTATION=has_permutation,
        WRITE_ALL=write_all,
        BLOCK_M=reduce_m,
        BLOCK_N=reduce_n,
        num_warps=4,
        num_stages=1,
    )
    return output


__all__ = ["chunked_sgmv_shrink"]
