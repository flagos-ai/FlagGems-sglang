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

"""fill_padded_rows for the sglang MoE decode path: set every row row >=
num_token_non_padded of x to fill_value, returning a new tensor. Copy and fill
are fused into one launch that reads the pad count from device memory (no sync,
static grid, so the op is CUDA-graph capturable); each program covers
BLOCK_ROWS consecutive rows, making the pad test a vector rows >= n compare."""

import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["n_rows", "row_stride"])
def _fill_padded_rows_kernel(
    x_ptr,
    out_ptr,
    n_ptr,
    n_rows,
    fill_value,
    row_stride,
    COLS: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = pid * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    rows = tl.max_contiguous(tl.multiple_of(rows, BLOCK_ROWS), BLOCK_ROWS)
    # num_token_non_padded lives in device memory; read it in-kernel so the
    # grid stays static and the call is CUDA-graph capturable.
    n = tl.load(n_ptr)
    row_ok = rows < n_rows
    pad = rows >= n
    cols = tl.arange(0, COLS)
    cols = tl.max_contiguous(tl.multiple_of(cols, COLS), COLS)
    offs = rows[:, None] * row_stride + cols[None, :]
    # Pad rows never read x; their value comes from fill_value below.
    x = tl.load(x_ptr + offs, mask=(row_ok & ~pad)[:, None], other=0)
    val = tl.where(pad[:, None], fill_value, x)
    tl.store(out_ptr + offs, val, mask=row_ok[:, None])


def fill_padded_rows(x, num_token_non_padded, fill_value):
    """Return a copy of ``x`` with padded rows set to ``fill_value``."""
    out = torch.empty_like(x)
    n_rows, n_cols = x.shape
    if n_rows == 0 or n_cols == 0:
        return out
    # One kernel for every layout: rows at/after the device-side boundary take
    # fill_value (their input values are never read); rows below are copied.
    # BLOCK_ROWS=64 with 1 warp won the interleaved launch sweep on the
    # target device (x8 cols: 64/128 are within noise of each other; 64
    # keeps more programs in flight for the tiny rows-per-program tails).
    _fill_padded_rows_kernel[(triton.cdiv(n_rows, 64),)](
        x,
        out,
        num_token_non_padded,
        n_rows,
        fill_value,
        x.stride(0),
        COLS=triton.next_power_of_2(n_cols),
        BLOCK_ROWS=64,
        num_warps=1,
    )
    return out


__all__ = ["fill_padded_rows"]
