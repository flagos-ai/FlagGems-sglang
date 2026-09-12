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

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _fused_rmsnorm_kernel(
    x_ptr,
    weight_ptr,
    out_ptr,
    eps,
    row_width: tl.constexpr,
    block_size: tl.constexpr,
):
    row = tl.program_id(0)
    columns = tl.arange(0, block_size)
    mask = columns < row_width
    row_offsets = row * row_width + columns
    x = tl.load(x_ptr + row_offsets, mask=mask, other=0.0).to(tl.float32)
    mean_square = tl.sum(x * x, axis=0) / row_width
    inv_rms = 1.0 / tl.sqrt(mean_square + eps)
    weight = tl.load(weight_ptr + columns, mask=mask, other=0.0).to(tl.float32)
    tl.store(out_ptr + row_offsets, x * inv_rms * weight, mask=mask)


def fused_rmsnorm(
    x: torch.Tensor, weight: torch.Tensor, eps: float
) -> torch.Tensor:
    x = x.contiguous()
    weight = weight.contiguous()
    if x.ndim == 0:
        raise ValueError("x must have at least one dimension")
    if weight.ndim != 1 or weight.numel() != x.shape[-1]:
        raise ValueError("weight must match the last dimension of x")
    if x.numel() == 0:
        raise ValueError("x must be non-empty")
    row_width = x.shape[-1]
    output = torch.empty_like(x)
    row_count = x.numel() // row_width
    block_size = triton.next_power_of_2(row_width)
    _fused_rmsnorm_kernel[(row_count,)](
        x,
        weight,
        output,
        eps,
        row_width=row_width,
        block_size=block_size,
        num_warps=2,
    )
    return output


__all__ = ["fused_rmsnorm"]
