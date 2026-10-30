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

"""gemma_rmsnorm_residual_scalar implementation for NVIDIA."""

import torch
import triton
import triton.language as tl


@triton.jit
def _gemma_rmsnorm_residual_scalar_kernel(
    x_ptr,
    weight_ptr,
    residual_ptr,
    scalar_ptr,
    out_ptr,
    X_ROW_STRIDE: tl.constexpr,
    RESIDUAL_STRIDES: tl.constexpr,
    WEIGHT_STRIDE: tl.constexpr,
    D: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_D)
    x = tl.load(
        x_ptr + row * X_ROW_STRIDE + cols,
        cols < D,
        other=0,
        cache_modifier=".cg",
    ).to(tl.float32)
    weight = tl.load(weight_ptr + cols * WEIGHT_STRIDE, cols < D, other=0).to(
        tl.float32
    )
    residual = tl.load(
        residual_ptr + row * RESIDUAL_STRIDES[0] + cols * RESIDUAL_STRIDES[1],
        cols < D,
        other=0,
        cache_modifier=".cg",
    ).to(tl.float32)
    scalar = tl.load(scalar_ptr).to(tl.float32)
    mean_square = tl.sum(x * x, 0) / D
    y = (x * tl.rsqrt(mean_square + EPS) * weight + residual) * scalar
    tl.store(out_ptr + row * D + cols, y, cols < D, cache_modifier=".cs")


def gemma_rmsnorm_residual_scalar(x, weight, residual, scalar, eps):
    (num_rows, hidden_size) = x.shape
    out = torch.empty(x.shape, dtype=x.dtype, device=x.device)
    if num_rows and hidden_size:
        block_d = triton.next_power_of_2(hidden_size)
        _gemma_rmsnorm_residual_scalar_kernel[num_rows,](
            x,
            weight,
            residual,
            scalar,
            out,
            x.stride(0),
            tuple(residual.stride()),
            weight.stride(0),
            hidden_size,
            eps,
            block_d,
            num_warps=4 if block_d <= 4096 else 8,
            enable_fp_fusion=False,
        )
    return out


__all__ = ["gemma_rmsnorm_residual_scalar"]
