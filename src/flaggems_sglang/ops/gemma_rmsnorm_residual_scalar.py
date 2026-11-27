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

"""Triton kernel for activation_norm/gemma_rmsnorm_residual_scalar."""

from __future__ import annotations

import contextlib

import torch
import triton
import triton.language as tl


def _maybe_device_ctx(device: torch.device):
    if device.type == "cuda":
        if device.index is None or device.index == torch.cuda.current_device():
            return contextlib.nullcontext()
        return torch.cuda.device(device)
    if device.type == "cpu":
        return contextlib.nullcontext()
    return torch.get_device_module(device).device(device)


def _as_2d_view(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim == 2:
        return tensor
    d = tensor.shape[-1]
    try:
        return tensor.view(-1, d)
    except RuntimeError:
        return tensor.contiguous().view(-1, d)


def _select_num_warps(block_d: int) -> int:
    if block_d <= 512:
        return 2
    if block_d <= 2048:
        return 4
    if block_d <= 8192:
        return 8
    return 16


@triton.jit
def _gemma_rmsnorm_contig_kernel(
    x_ptr,
    weight_ptr,
    residual_ptr,
    scalar_ptr,
    out_ptr,
    eps,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
    EVEN_D: tl.constexpr,
    USE_CG: tl.constexpr,
    USE_CS: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK_D)
    row_off = row * D
    x_row = x_ptr + row_off
    res_row = residual_ptr + row_off
    out_row = out_ptr + row_off

    if EVEN_D:
        if USE_CG:
            x = tl.load(x_row + cols, cache_modifier=".cg").to(tl.float32)
            res = tl.load(res_row + cols, cache_modifier=".cg").to(tl.float32)
        else:
            x = tl.load(x_row + cols).to(tl.float32)
            res = tl.load(res_row + cols).to(tl.float32)
        w = tl.load(weight_ptr + cols, cache_modifier=".ca").to(tl.float32)
    else:
        mask = cols < D
        if USE_CG:
            x = tl.load(
                x_row + cols, mask=mask, other=0.0, cache_modifier=".cg"
            ).to(tl.float32)
            res = tl.load(
                res_row + cols, mask=mask, other=0.0, cache_modifier=".cg"
            ).to(tl.float32)
        else:
            x = tl.load(x_row + cols, mask=mask, other=0.0).to(tl.float32)
            res = tl.load(res_row + cols, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(
            weight_ptr + cols, mask=mask, other=0.0, cache_modifier=".ca"
        ).to(tl.float32)

    scalar = tl.load(scalar_ptr, cache_modifier=".ca").to(tl.float32)
    rstd = tl.rsqrt(tl.sum(x * x, axis=0) * (1.0 / D) + eps)
    out = ((x * rstd * w + res) * scalar).to(out_ptr.dtype.element_ty)

    if EVEN_D:
        if USE_CS:
            tl.store(out_row + cols, out, cache_modifier=".cs")
        else:
            tl.store(out_row + cols, out)
    else:
        mask = cols < D
        if USE_CS:
            tl.store(out_row + cols, out, mask=mask, cache_modifier=".cs")
        else:
            tl.store(out_row + cols, out, mask=mask)


@triton.jit
def _gemma_rmsnorm_strided_kernel(
    x_ptr,
    weight_ptr,
    residual_ptr,
    scalar_ptr,
    out_ptr,
    stride_xm,
    stride_xd,
    stride_rm,
    stride_rd,
    stride_w,
    stride_om,
    eps,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    row64 = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK_D).to(tl.int64)
    mask = cols < D

    x = tl.load(
        x_ptr + row64 * stride_xm + cols * stride_xd, mask=mask, other=0.0
    ).to(tl.float32)
    res = tl.load(
        residual_ptr + row64 * stride_rm + cols * stride_rd,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    w = tl.load(
        weight_ptr + cols * stride_w,
        mask=mask,
        other=0.0,
        cache_modifier=".ca",
    ).to(tl.float32)
    scalar = tl.load(scalar_ptr, cache_modifier=".ca").to(tl.float32)

    rstd = tl.rsqrt(tl.sum(x * x, axis=0) * (1.0 / D) + eps)
    out = ((x * rstd * w + res) * scalar).to(out_ptr.dtype.element_ty)
    tl.store(out_ptr + row64 * stride_om + cols, out, mask=mask)


@triton.jit
def _gemma_rmsnorm_large_d_kernel(
    x_ptr,
    weight_ptr,
    residual_ptr,
    scalar_ptr,
    out_ptr,
    stride_xm,
    stride_xd,
    stride_rm,
    stride_rd,
    stride_w,
    stride_om,
    eps,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    row64 = tl.program_id(0).to(tl.int64)
    inv_d = 1.0 / D
    scalar = tl.load(scalar_ptr).to(tl.float32)

    sum_sq = tl.zeros((), dtype=tl.float32)
    for start in range(0, D, BLOCK_D):
        cols = start + tl.arange(0, BLOCK_D).to(tl.int64)
        mask = cols < D
        x = tl.load(
            x_ptr + row64 * stride_xm + cols * stride_xd,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        sum_sq += tl.sum(x * x, axis=0)

    rstd = tl.rsqrt(sum_sq * inv_d + eps)
    out_ty = out_ptr.dtype.element_ty
    for start in range(0, D, BLOCK_D):
        cols = start + tl.arange(0, BLOCK_D).to(tl.int64)
        mask = cols < D
        x = tl.load(
            x_ptr + row64 * stride_xm + cols * stride_xd,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        res = tl.load(
            residual_ptr + row64 * stride_rm + cols * stride_rd,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        w = tl.load(weight_ptr + cols * stride_w, mask=mask, other=0.0).to(
            tl.float32
        )
        out = (x * rstd * w + res) * scalar
        tl.store(out_ptr + row64 * stride_om + cols, out.to(out_ty), mask=mask)


def gemma_rmsnorm_residual_scalar(
    x: torch.Tensor,
    weight: torch.Tensor,
    residual: torch.Tensor,
    scalar: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    if x.ndim < 2 or residual.shape != x.shape:
        raise ValueError("x and residual must share shape with ndim >= 2")

    orig_shape = x.shape
    hidden_size = orig_shape[-1]
    out = torch.empty_like(x)
    if x.numel() == 0:
        return out

    x_2d = x if x.ndim == 2 else _as_2d_view(x)
    res_2d = residual if residual.ndim == 2 else _as_2d_view(residual)
    out_2d = out if out.ndim == 2 else out.view(-1, hidden_size)
    num_rows = x_2d.shape[0]

    stride_xm = x_2d.stride(0)
    stride_xd = x_2d.stride(1)
    stride_rm = res_2d.stride(0)
    stride_rd = res_2d.stride(1)
    stride_w = weight.stride(0)

    with _maybe_device_ctx(x.device):
        if hidden_size <= 16384:
            block_d = 1 << (hidden_size - 1).bit_length()
            num_warps = _select_num_warps(block_d)
            contiguous = (
                stride_xd == 1
                and stride_rd == 1
                and stride_w == 1
                and stride_xm == hidden_size
                and stride_rm == hidden_size
            )
            if contiguous:
                _gemma_rmsnorm_contig_kernel[(num_rows,)](
                    x_2d,
                    weight,
                    res_2d,
                    scalar,
                    out_2d,
                    float(eps),
                    D=hidden_size,
                    BLOCK_D=block_d,
                    EVEN_D=(hidden_size == block_d),
                    USE_CG=(num_rows >= 4),
                    USE_CS=(
                        num_rows * hidden_size * x.element_size() >= 4194304
                    ),
                    num_warps=num_warps,
                )
            else:
                _gemma_rmsnorm_strided_kernel[(num_rows,)](
                    x_2d,
                    weight,
                    res_2d,
                    scalar,
                    out_2d,
                    stride_xm,
                    stride_xd,
                    stride_rm,
                    stride_rd,
                    stride_w,
                    out_2d.stride(0),
                    float(eps),
                    D=hidden_size,
                    BLOCK_D=block_d,
                    num_warps=num_warps,
                )
        else:
            _gemma_rmsnorm_large_d_kernel[(num_rows,)](
                x_2d,
                weight,
                res_2d,
                scalar,
                out_2d,
                stride_xm,
                stride_xd,
                stride_rm,
                stride_rd,
                stride_w,
                out_2d.stride(0),
                float(eps),
                D=hidden_size,
                BLOCK_D=4096,
                num_warps=8,
            )
    return out


__all__ = ["gemma_rmsnorm_residual_scalar"]
