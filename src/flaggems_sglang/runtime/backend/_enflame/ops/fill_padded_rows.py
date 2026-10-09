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
def _fill_padded_rows_masked_kernel(
    x_ptr,
    out_ptr,
    n_ptr,
    total,
    fill_value,
    NCOLS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    in_bounds = offs < total
    # Threshold over the flat index space: elements at or beyond row
    # ``n`` (read from device memory) become fill_value. Masking the load
    # with the same predicate and seeding ``other`` with fill_value means
    # padded lanes are never fetched and need no separate select.
    thresh = tl.load(n_ptr) * NCOLS
    val = tl.load(
        x_ptr + offs, mask=in_bounds & (offs < thresh), other=fill_value
    )
    tl.store(out_ptr + offs, val, mask=in_bounds)


@triton.jit
def _fill_padded_rows_even_kernel(
    x_ptr,
    out_ptr,
    n_ptr,
    fill_value,
    NCOLS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # total == grid_size * BLOCK: every program covers a full block, no
    # tail predicate and no ``total`` argument needed. The load stays
    # masked by the (device-read) pad threshold so padded lanes still
    # avoid the memory fetch.
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    thresh = tl.load(n_ptr) * NCOLS
    val = tl.load(x_ptr + offs, mask=offs < thresh, other=fill_value)
    tl.store(out_ptr + offs, val)


@triton.jit
def _fill_padded_rows_2d_kernel(
    x_ptr,
    out_ptr,
    n_ptr,
    n_rows,
    n_cols,
    fill_value,
    BLOCK_R: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid_r = tl.program_id(0)
    pid_c = tl.program_id(1)
    offs_r = pid_r * BLOCK_R + tl.arange(0, BLOCK_R)
    offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    in_bounds = (offs_r[:, None] < n_rows) & (offs_c[None, :] < n_cols)
    offs = offs_r[:, None] * n_cols + offs_c[None, :]
    n = tl.load(n_ptr)
    val = tl.load(
        x_ptr + offs,
        mask=in_bounds & (offs_r[:, None] < n),
        other=fill_value,
    )
    tl.store(out_ptr + offs, val, mask=in_bounds)


def fill_padded_rows(x, num_token_non_padded, fill_value):
    out = torch.empty_like(x)
    n_rows, n_cols = x.shape

    if torch.is_tensor(num_token_non_padded):
        n_dev = num_token_non_padded
    else:
        # Scalar fallback: keep the device-tensor interface so the kernel
        # always reads the count from memory.
        n_dev = torch.tensor(
            [num_token_non_padded], dtype=torch.int32, device=x.device
        )

    if not x.is_contiguous():
        # Strided rows (stride(1) == 1 per op contract): use the 2D
        # address space so the flat-suffix trick is not applied to a
        # non-contiguous layout.
        _fill_padded_rows_2d_kernel[
            (triton.cdiv(n_rows, 128), triton.cdiv(n_cols, 64))
        ](
            x,
            out,
            n_dev,
            n_rows,
            n_cols,
            fill_value,
            BLOCK_R=128,
            BLOCK_C=64,
            num_warps=4,
        )
        return out

    total = n_rows * n_cols
    if total <= 131072:
        block = triton.next_power_of_2(total)
        num_warps = 4 if block <= 4096 else 8
        grid = (1,)
    else:
        block = 131072
        num_warps = 8
        grid = (triton.cdiv(total, block),)

    if total % block == 0:
        _fill_padded_rows_even_kernel[grid](
            x,
            out,
            n_dev,
            fill_value,
            NCOLS=n_cols,
            BLOCK=block,
            num_warps=num_warps,
        )
        return out

    _fill_padded_rows_masked_kernel[grid](
        x,
        out,
        n_dev,
        total,
        fill_value,
        NCOLS=n_cols,
        BLOCK=block,
        num_warps=num_warps,
    )
    return out


__all__ = ["fill_padded_rows"]
