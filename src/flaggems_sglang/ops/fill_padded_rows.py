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

"""moe/fill_padded_rows: set every padded row to a
constant fill value, so x[row, :] = fill_value for row >= num_token_non_padded.
The pad count is read from device memory inside the kernel and the grid is
static, so the whole op stays CUDA-graph capturable. Copy and fill are fused
into one launch over a fresh empty_like buffer: contiguous inputs use a flat 1D
pass split at the device-read scalar, row-strided inputs use 2D tiles with
independent strides."""

import torch
import triton
import triton.language as tl

# Flat-pass launch configs chosen by a static total-size heuristic
# (block, num_warps), picked per size class — measured on-device:
# - tiny tensors (total <= 1024 elements) sit at the launch floor and are
#   indifferent to shape;
# - small/mid ones want few, fat programs (2048 elements across 32 warps);
# - around 16-64K elements a single 4096-wide 32-warp program wave covers the
#   whole tensor in one FILL-wave, beating a multi-wave grid;
# - beyond one wave of occupancy the widest program loses residency, and
#   2048x32 wins again.
_CFG_TINY = (256, 4)
_CFG_SMALL = (2048, 32)
_CFG_WAVE = (4096, 32)
_CFG_BIG = (2048, 32)


@triton.jit
def _fill_pad_rows_flat_kernel(
    x_ptr,
    out_ptr,
    n_pad_ptr,
    total,
    n_cols,
    fill_value,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    n = tl.load(n_pad_ptr)
    split = n * n_cols  # flat index of the first padded element
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    in_bounds = offs < total
    is_pad = offs >= split
    fill = tl.full([BLOCK], fill_value, tl.float32).to(x_ptr.dtype.element_ty)
    src = tl.load(x_ptr + offs, mask=in_bounds & (~is_pad), other=0.0)
    val = tl.where(is_pad, fill, src)
    tl.store(out_ptr + offs, val, mask=in_bounds)


@triton.jit
def _fill_pad_rows_flat_kernel_i64(
    x_ptr,
    out_ptr,
    n_pad_ptr,
    total,
    n_cols,
    fill_value,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    n = tl.load(n_pad_ptr).to(tl.int64)
    split = n * n_cols
    offs = pid * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    in_bounds = offs < total
    is_pad = offs >= split
    fill = tl.full([BLOCK], fill_value, tl.float32).to(x_ptr.dtype.element_ty)
    src = tl.load(x_ptr + offs, mask=in_bounds & (~is_pad), other=0.0)
    val = tl.where(is_pad, fill, src)
    tl.store(out_ptr + offs, val, mask=in_bounds)


@triton.jit
def _fill_pad_rows_strided_kernel(
    x_ptr,
    out_ptr,
    n_pad_ptr,
    n_rows,
    n_cols,
    x_row_stride,
    out_row_stride,
    fill_value,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_COLS: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    n = tl.load(n_pad_ptr).to(tl.int64)
    rows = pid * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS).to(tl.int64)
    cols = tl.arange(0, BLOCK_COLS).to(tl.int64)
    mask = (rows[:, None] < n_rows) & (cols[None, :] < n_cols)
    is_pad = rows[:, None] >= n
    fill = tl.full([BLOCK_ROWS, BLOCK_COLS], fill_value, tl.float32).to(
        x_ptr.dtype.element_ty
    )
    src = tl.load(
        x_ptr + rows[:, None] * x_row_stride + cols[None, :],
        mask=mask & (~is_pad),
        other=0.0,
    )
    val = tl.where(is_pad, fill, src)
    tl.store(
        out_ptr + rows[:, None] * out_row_stride + cols[None, :],
        val,
        mask=mask,
    )


def fill_padded_rows(x, num_token_non_padded, fill_value):
    out = torch.empty_like(x)
    n_rows, n_cols = out.shape
    total = out.numel()
    if total == 0:
        return out

    if x.is_contiguous() and out.is_contiguous():
        kernel = (
            _fill_pad_rows_flat_kernel
            if total <= 0x7FFFFFFF
            else _fill_pad_rows_flat_kernel_i64
        )
        if total <= 1024:
            block, num_warps = _CFG_TINY
        elif total <= 8192:
            block, num_warps = _CFG_SMALL
        elif total <= 65536:
            block, num_warps = _CFG_WAVE
        else:
            block, num_warps = _CFG_BIG
        grid = (triton.cdiv(total, block),)
        kernel[grid](
            x,
            out,
            num_token_non_padded,
            total,
            n_cols,
            fill_value,
            BLOCK=block,
            num_warps=num_warps,
        )
    else:
        # Row-strided (stride(1) == 1): 2D row/col tiles, independent strides.
        block_cols = triton.next_power_of_2(max(min(n_cols, 1024), 1))
        grid = (triton.cdiv(n_rows, 32), triton.cdiv(n_cols, block_cols))
        _fill_pad_rows_strided_kernel[grid](
            x,
            out,
            num_token_non_padded,
            n_rows,
            n_cols,
            x.stride(0),
            out.stride(0),
            fill_value,
            BLOCK_ROWS=32,
            BLOCK_COLS=block_cols,
            num_warps=4,
        )
    return out


__all__ = ["fill_padded_rows"]
