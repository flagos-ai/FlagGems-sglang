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

"""fill_padded_rows: set every padding row to a constant,
x[row, :] = fill_value for row >= num_token_non_padded. The row count lives in
device memory and the grid is static, so the step stays CUDA-graph
capturable; one flat 1D kernel fuses the reference's clone() + fill into a
single launch."""

import torch
import triton
import triton.language as tl


@triton.jit
def _fill_padded_rows_kernel(
    x_ptr,  # [..., cols] input, stride(1) == 1
    out_ptr,  # same shape/layout as x
    n_ptr,  # [1] device scalar: num_token_non_padded
    stride_row,  # x.stride(0) in elements
    rows,  # x.shape[0]
    numel,  # rows * stride_row (flat extent of the storage span)
    fill,  # scalar fill value
    BLOCK: tl.constexpr,
):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)

    # First padded element offset, computed entirely on device.
    n = tl.load(n_ptr)
    n = tl.minimum(tl.maximum(n, 0), rows)
    start = n * stride_row

    keep = offs < start
    inb = offs < numel
    # Load with a pure bounds mask; the copy-vs-fill choice is a separate
    # ``tl.where`` on ``keep``.  (On some backends folding ``fill`` into the
    # load's ``other=`` under a combined predicate mis-selects lanes, so the
    # two concerns are kept independent.)
    v = tl.load(x_ptr + offs, mask=inb, other=0.0)
    v = tl.where(keep, v, fill)
    tl.store(out_ptr + offs, v, mask=inb)


def fill_padded_rows(x, num_token_non_padded, fill_value):
    rows = x.shape[0]
    stride_row = x.stride(0)
    out = torch.empty_like(x)
    numel = rows * stride_row
    if numel == 0:
        return out

    if numel <= 512:
        block, warps = 512, 1
    elif numel <= 4096:
        block, warps = 2048, 2
    elif numel <= 8192:
        block, warps = 4096, 4
    elif numel <= 65536:
        block, warps = 8192, 8
    else:
        block, warps = 16384, 8

    _fill_padded_rows_kernel[(triton.cdiv(numel, block),)](
        x,
        out,
        num_token_non_padded,
        stride_row,
        rows,
        numel,
        fill_value,
        BLOCK=block,
        num_warps=warps,
        num_stages=1,
    )
    return out


__all__ = ["fill_padded_rows"]
