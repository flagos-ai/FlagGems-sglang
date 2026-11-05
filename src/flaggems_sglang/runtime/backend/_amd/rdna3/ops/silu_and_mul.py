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
def _silu_and_mul_kernel(
    x_ptr,
    out_ptr,
    d: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    col_block_idx = tl.program_id(1)

    col_offs = col_block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = col_offs < d

    # int64 scalar row base (overflow-safe), int32 lane offsets (fast).
    base = x_ptr + row_idx.to(tl.int64) * (2 * d)
    x1 = tl.load(base + col_offs, mask=mask, other=0.0).to(tl.float32)
    x2 = tl.load(base + d + col_offs, mask=mask, other=0.0).to(tl.float32)

    silu_x1 = tl.fdiv(x1, 1.0 + tl.exp(-x1))

    out_base = out_ptr + row_idx.to(tl.int64) * d
    tl.store(out_base + col_offs, silu_x1 * x2, mask=mask)


def silu_and_mul(hidden_states: torch.Tensor) -> torch.Tensor:
    shape = hidden_states.shape
    d = shape[-1] // 2
    numel = hidden_states.numel()
    out = hidden_states.new_empty(shape[:-1] + (d,))
    if d == 0 or numel == 0:
        return out
    x = hidden_states.contiguous()
    n_rows = numel // (2 * d)
    # gfx1100: for up to 64 rows and from 2048 rows on, 16 warps a program,
    # on 1024-column blocks when d is a power of two up to 8192 and on one
    # block of the padded row (up to 8192 columns) otherwise; in between, the
    # generic one-block-per-row launch is as fast or faster.
    block_size = min(8192, triton.next_power_of_2(d))
    num_warps = 4
    if n_rows <= 64 or n_rows >= 2048:
        num_warps = 16
        if d <= 8192 and d & (d - 1) == 0:
            block_size = min(1024, d)
    _silu_and_mul_kernel[(n_rows, triton.cdiv(d, block_size))](
        x, out, d=d, BLOCK_SIZE=block_size, num_warps=num_warps
    )
    return out


__all__ = ["silu_and_mul"]
