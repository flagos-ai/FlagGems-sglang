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

"""gemma_dual_rmsnorm_residual_scalar implementation for NVIDIA."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

MAX_HIDDEN_SIZE = 8192


@triton.jit
def _gemma_dual_rmsnorm_residual_scalar_kernel(
    out_ptr,
    x1_ptr,
    x2_ptr,
    residual_ptr,
    weight1_ptr,
    weight2_ptr,
    weight3_ptr,
    scalar_ptr,
    x1_row_stride,
    x2_row_stride,
    residual_row_stride,
    D: tl.constexpr,
    eps1,
    eps2,
    eps3,
    BLOCK_D: tl.constexpr,
    STREAM_INPUTS: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_D)
    mask = offsets < D
    if STREAM_INPUTS:
        x1 = tl.load(
            x1_ptr + row * x1_row_stride + offsets,
            mask=mask,
            other=0.0,
            cache_modifier=".cg",
        ).to(tl.float32)
        x2 = tl.load(
            x2_ptr + row * x2_row_stride + offsets,
            mask=mask,
            other=0.0,
            cache_modifier=".cg",
        ).to(tl.float32)
        residual = tl.load(
            residual_ptr + row * residual_row_stride + offsets,
            mask=mask,
            other=0.0,
            cache_modifier=".cg",
        ).to(tl.float32)
    else:
        x1 = tl.load(
            x1_ptr + row * x1_row_stride + offsets, mask=mask, other=0.0
        ).to(tl.float32)
        x2 = tl.load(
            x2_ptr + row * x2_row_stride + offsets, mask=mask, other=0.0
        ).to(tl.float32)
        residual = tl.load(
            residual_ptr + row * residual_row_stride + offsets,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
    w1 = tl.load(weight1_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    w2 = tl.load(weight2_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    w3 = tl.load(weight3_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    scalar = tl.load(scalar_ptr).to(tl.float32)
    x1_norm = x1 * tl.rsqrt(tl.sum(x1 * x1, axis=0) / D + eps1) * w1
    x2_norm = x2 * tl.rsqrt(tl.sum(x2 * x2, axis=0) / D + eps2) * w2
    combined = x1_norm + x2_norm
    out_norm = (
        combined
        * tl.rsqrt(tl.sum(combined * combined, axis=0) / D + eps3)
        * w3
    )
    out = (out_norm + residual) * scalar
    tl.store(
        out_ptr + row * D + offsets,
        out.to(out_ptr.dtype.element_ty),
        mask=mask,
    )


def _gemma_dual_rmsnorm_residual_scalar_row_stride(tensor):
    if tensor.dim() < 2 or tensor.stride(-1) != 1:
        return None
    row_stride = tensor.stride(-2)
    expected = row_stride * tensor.shape[-2]
    for dim in range(tensor.dim() - 3, -1, -1):
        if tensor.stride(dim) != expected:
            return None
        expected *= tensor.shape[dim]
    return row_stride


def gemma_dual_rmsnorm_residual_scalar(
    x1,
    weight1,
    x2,
    weight2,
    weight3,
    residual,
    scalar,
    eps1,
    eps2,
    eps3,
):
    dtypes = (torch.float32, torch.bfloat16, torch.float16)
    if (
        x1.dtype not in dtypes
        or x2.dtype != x1.dtype
        or residual.dtype != x1.dtype
    ):
        return None
    if x1.dim() < 2 or x2.shape != x1.shape or residual.shape != x1.shape:
        return None
    hidden_size = x1.shape[-1]
    if hidden_size <= 0 or hidden_size > MAX_HIDDEN_SIZE:
        return None
    for weight in (weight1, weight2, weight3):
        if (
            weight.dim() != 1
            or weight.shape[0] != hidden_size
            or (not weight.is_contiguous())
        ):
            return None
    if scalar.numel() != 1:
        return None
    x1_row_stride = _gemma_dual_rmsnorm_residual_scalar_row_stride(x1)
    x2_row_stride = _gemma_dual_rmsnorm_residual_scalar_row_stride(x2)
    residual_row_stride = _gemma_dual_rmsnorm_residual_scalar_row_stride(
        residual
    )
    if (
        x1_row_stride is None
        or x2_row_stride is None
        or residual_row_stride is None
    ):
        return None
    num_rows = x1.numel() // hidden_size
    out = torch.empty(x1.shape, dtype=x1.dtype, device=x1.device)
    stream_inputs = (
        hidden_size >= 1024 and x1.numel() * x1.element_size() >= 1048576
    )
    with torch.get_device_module().device(x1.device):
        _gemma_dual_rmsnorm_residual_scalar_kernel[num_rows,](
            out,
            x1,
            x2,
            residual,
            weight1,
            weight2,
            weight3,
            scalar,
            x1_row_stride,
            x2_row_stride,
            residual_row_stride,
            hidden_size,
            eps1,
            eps2,
            eps3,
            BLOCK_D=triton.next_power_of_2(hidden_size),
            STREAM_INPUTS=stream_inputs,
            num_warps=8,
        )
    return out


__all__ = ["gemma_dual_rmsnorm_residual_scalar"]
