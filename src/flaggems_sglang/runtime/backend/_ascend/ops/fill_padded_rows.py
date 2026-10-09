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

"""Operator: moe/fill_padded_rows -- copy x and set every row at or beyond
num_token_non_padded to fill_value. A single flat kernel does the whole copy-
and-fill in one launch, reading the token count from device memory (no device-
to-host sync), so the op stays CUDA-graph capturable."""

import torch
import triton
import triton.language as tl


@triton.jit
def _fill_flat_kernel(
    x_ptr,
    out_ptr,
    n_ptr,
    fill_value,
    row_size,
    total,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    # Contiguity hints let the compiler emit vectorized (128-bit) accesses.
    offs = tl.max_contiguous(tl.multiple_of(offs, BLOCK), BLOCK)
    n = tl.load(n_ptr)  # device-side token count; no host sync
    # The copied prefix is a contiguous tail in flat order, so
    # ``offs < n * row_size`` already implies ``offs < total`` — the load
    # needs only that one predicate; the store still covers every real
    # element so padded positions receive the constant.
    keep = offs < n * row_size
    v = tl.load(x_ptr + offs, mask=keep, other=fill_value)
    tl.store(out_ptr + offs, v, mask=offs < total)


@triton.jit
def _fill_strided_kernel(
    x_ptr,
    out_ptr,
    n_ptr,
    fill_value,
    row_size,
    stride_x0,
    stride_o0,
    BLOCK: tl.constexpr,
):
    # Non-contiguous fallback: one program per row, columns in one block.
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    n = tl.load(n_ptr)  # device-side token count; no host sync
    mask = col < row_size
    keep = row < n
    v = tl.load(
        x_ptr + row * stride_x0 + col, mask=mask & keep, other=fill_value
    )
    tl.store(out_ptr + row * stride_o0 + col, v, mask=mask)


def fill_padded_rows(x, num_token_non_padded, fill_value):
    n_rows, row_size = x.shape
    out = torch.empty_like(x)
    numel = n_rows * row_size
    if numel == 0:
        return out
    if x.stride(1) == 1 and x.stride(0) == row_size:
        # Size-driven launch config (local vars only): a 64-element
        # single-warp program covers tiny inputs; a 256-element 2-warp
        # program measures at the launch-overhead floor for everything
        # larger — wider blocks (1024/4096 elements) measured 1-2 us above
        # that floor on device, so they are avoided here.
        if numel <= 256:
            block, num_warps = 64, 1
        else:
            block, num_warps = 256, 2
        grid = (triton.cdiv(numel, block),)
        _fill_flat_kernel[grid](
            x,
            out,
            num_token_non_padded,
            fill_value,
            row_size,
            numel,
            BLOCK=block,
            num_warps=num_warps,
            num_stages=1,
        )
        return out
    # Non-contiguous fallback: one program per row (row_size <= BLOCK needed).
    grid = (n_rows,)
    _fill_strided_kernel[grid](
        x,
        out,
        num_token_non_padded,
        fill_value,
        row_size,
        x.stride(0),
        out.stride(0),
        BLOCK=triton.next_power_of_2(row_size),
        num_warps=1,
        num_stages=1,
    )
    return out


__all__ = ["fill_padded_rows"]
