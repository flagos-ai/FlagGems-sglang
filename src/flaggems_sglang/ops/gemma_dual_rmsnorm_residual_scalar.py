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

"""Triton kernel for gemma_dual_rmsnorm_residual_scalar."""

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
    if block_d <= 4096:
        return 8
    return 16


@triton.jit
def _gemma_dual_rmsnorm_contig_kernel(
    out_ptr,
    x1_ptr,
    x2_ptr,
    residual_ptr,
    weight1_ptr,
    weight2_ptr,
    weight3_ptr,
    scalar_ptr,
    eps1,
    eps2,
    eps3,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
    EVEN_D: tl.constexpr,
    USE_CG: tl.constexpr,
    USE_CS: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK_D)
    row_off = row * D
    x1_row = x1_ptr + row_off
    x2_row = x2_ptr + row_off
    res_row = residual_ptr + row_off
    out_row = out_ptr + row_off
    inv_d = 1.0 / D

    if EVEN_D:
        if USE_CG:
            x1 = tl.load(x1_row + cols, cache_modifier=".cg").to(tl.float32)
            x2 = tl.load(x2_row + cols, cache_modifier=".cg").to(tl.float32)
            res = tl.load(res_row + cols, cache_modifier=".cg").to(tl.float32)
        else:
            x1 = tl.load(x1_row + cols).to(tl.float32)
            x2 = tl.load(x2_row + cols).to(tl.float32)
            res = tl.load(res_row + cols).to(tl.float32)
        w1 = tl.load(weight1_ptr + cols, cache_modifier=".ca").to(tl.float32)
        w2 = tl.load(weight2_ptr + cols, cache_modifier=".ca").to(tl.float32)
        w3 = tl.load(weight3_ptr + cols, cache_modifier=".ca").to(tl.float32)
    else:
        mask = cols < D
        if USE_CG:
            x1 = tl.load(
                x1_row + cols, mask=mask, other=0.0, cache_modifier=".cg"
            ).to(tl.float32)
            x2 = tl.load(
                x2_row + cols, mask=mask, other=0.0, cache_modifier=".cg"
            ).to(tl.float32)
            res = tl.load(
                res_row + cols, mask=mask, other=0.0, cache_modifier=".cg"
            ).to(tl.float32)
        else:
            x1 = tl.load(x1_row + cols, mask=mask, other=0.0).to(tl.float32)
            x2 = tl.load(x2_row + cols, mask=mask, other=0.0).to(tl.float32)
            res = tl.load(res_row + cols, mask=mask, other=0.0).to(tl.float32)
        w1 = tl.load(
            weight1_ptr + cols, mask=mask, other=0.0, cache_modifier=".ca"
        ).to(tl.float32)
        w2 = tl.load(
            weight2_ptr + cols, mask=mask, other=0.0, cache_modifier=".ca"
        ).to(tl.float32)
        w3 = tl.load(
            weight3_ptr + cols, mask=mask, other=0.0, cache_modifier=".ca"
        ).to(tl.float32)

    scalar = tl.load(scalar_ptr, cache_modifier=".ca").to(tl.float32)
    n1 = x1 * tl.rsqrt(tl.sum(x1 * x1, axis=0) * inv_d + eps1) * w1
    n2 = x2 * tl.rsqrt(tl.sum(x2 * x2, axis=0) * inv_d + eps2) * w2
    comb = n1 + n2
    n3 = comb * tl.rsqrt(tl.sum(comb * comb, axis=0) * inv_d + eps3) * w3
    out = ((n3 + res) * scalar).to(out_ptr.dtype.element_ty)

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
def _gemma_dual_rmsnorm_strided_kernel(
    out_ptr,
    x1_ptr,
    x2_ptr,
    residual_ptr,
    weight1_ptr,
    weight2_ptr,
    weight3_ptr,
    scalar_ptr,
    stride_x1m,
    stride_x1d,
    stride_x2m,
    stride_x2d,
    stride_rm,
    stride_rd,
    stride_w1,
    stride_w2,
    stride_w3,
    stride_om,
    eps1,
    eps2,
    eps3,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    row64 = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK_D).to(tl.int64)
    mask = cols < D
    inv_d = 1.0 / D

    x1 = tl.load(
        x1_ptr + row64 * stride_x1m + cols * stride_x1d, mask=mask, other=0.0
    ).to(tl.float32)
    x2 = tl.load(
        x2_ptr + row64 * stride_x2m + cols * stride_x2d, mask=mask, other=0.0
    ).to(tl.float32)
    res = tl.load(
        residual_ptr + row64 * stride_rm + cols * stride_rd,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    w1 = tl.load(
        weight1_ptr + cols * stride_w1,
        mask=mask,
        other=0.0,
        cache_modifier=".ca",
    ).to(tl.float32)
    w2 = tl.load(
        weight2_ptr + cols * stride_w2,
        mask=mask,
        other=0.0,
        cache_modifier=".ca",
    ).to(tl.float32)
    w3 = tl.load(
        weight3_ptr + cols * stride_w3,
        mask=mask,
        other=0.0,
        cache_modifier=".ca",
    ).to(tl.float32)
    scalar = tl.load(scalar_ptr, cache_modifier=".ca").to(tl.float32)

    n1 = x1 * tl.rsqrt(tl.sum(x1 * x1, axis=0) * inv_d + eps1) * w1
    n2 = x2 * tl.rsqrt(tl.sum(x2 * x2, axis=0) * inv_d + eps2) * w2
    comb = n1 + n2
    n3 = comb * tl.rsqrt(tl.sum(comb * comb, axis=0) * inv_d + eps3) * w3
    out = ((n3 + res) * scalar).to(out_ptr.dtype.element_ty)
    tl.store(out_ptr + row64 * stride_om + cols, out, mask=mask)


@triton.jit
def _gemma_dual_rmsnorm_large_d_kernel(
    out_ptr,
    x1_ptr,
    x2_ptr,
    residual_ptr,
    weight1_ptr,
    weight2_ptr,
    weight3_ptr,
    scalar_ptr,
    stride_x1m,
    stride_x1d,
    stride_x2m,
    stride_x2d,
    stride_rm,
    stride_rd,
    stride_w1,
    stride_w2,
    stride_w3,
    stride_om,
    eps1,
    eps2,
    eps3,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    row64 = tl.program_id(0).to(tl.int64)
    inv_d = 1.0 / D
    scalar = tl.load(scalar_ptr).to(tl.float32)

    sum_sq1 = tl.zeros((), dtype=tl.float32)
    sum_sq2 = tl.zeros((), dtype=tl.float32)
    for start in range(0, D, BLOCK_D):
        cols = start + tl.arange(0, BLOCK_D).to(tl.int64)
        mask = cols < D
        x1 = tl.load(
            x1_ptr + row64 * stride_x1m + cols * stride_x1d,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        x2 = tl.load(
            x2_ptr + row64 * stride_x2m + cols * stride_x2d,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        sum_sq1 += tl.sum(x1 * x1, axis=0)
        sum_sq2 += tl.sum(x2 * x2, axis=0)

    rstd1 = tl.rsqrt(sum_sq1 * inv_d + eps1)
    rstd2 = tl.rsqrt(sum_sq2 * inv_d + eps2)

    sum_sq3 = tl.zeros((), dtype=tl.float32)
    for start in range(0, D, BLOCK_D):
        cols = start + tl.arange(0, BLOCK_D).to(tl.int64)
        mask = cols < D
        x1 = tl.load(
            x1_ptr + row64 * stride_x1m + cols * stride_x1d,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        x2 = tl.load(
            x2_ptr + row64 * stride_x2m + cols * stride_x2d,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        w1 = tl.load(weight1_ptr + cols * stride_w1, mask=mask, other=0.0).to(
            tl.float32
        )
        w2 = tl.load(weight2_ptr + cols * stride_w2, mask=mask, other=0.0).to(
            tl.float32
        )
        comb = x1 * rstd1 * w1 + x2 * rstd2 * w2
        sum_sq3 += tl.sum(comb * comb, axis=0)

    rstd3 = tl.rsqrt(sum_sq3 * inv_d + eps3)
    out_ty = out_ptr.dtype.element_ty
    for start in range(0, D, BLOCK_D):
        cols = start + tl.arange(0, BLOCK_D).to(tl.int64)
        mask = cols < D
        x1 = tl.load(
            x1_ptr + row64 * stride_x1m + cols * stride_x1d,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        x2 = tl.load(
            x2_ptr + row64 * stride_x2m + cols * stride_x2d,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        res = tl.load(
            residual_ptr + row64 * stride_rm + cols * stride_rd,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        w1 = tl.load(weight1_ptr + cols * stride_w1, mask=mask, other=0.0).to(
            tl.float32
        )
        w2 = tl.load(weight2_ptr + cols * stride_w2, mask=mask, other=0.0).to(
            tl.float32
        )
        w3 = tl.load(weight3_ptr + cols * stride_w3, mask=mask, other=0.0).to(
            tl.float32
        )
        comb = x1 * rstd1 * w1 + x2 * rstd2 * w2
        out = (comb * rstd3 * w3 + res) * scalar
        tl.store(out_ptr + row64 * stride_om + cols, out.to(out_ty), mask=mask)


def gemma_dual_rmsnorm_residual_scalar(
    x1: torch.Tensor,
    weight1: torch.Tensor,
    x2: torch.Tensor,
    weight2: torch.Tensor,
    weight3: torch.Tensor,
    residual: torch.Tensor,
    scalar: torch.Tensor,
    eps1: float = 1e-6,
    eps2: float = 1e-6,
    eps3: float = 1e-6,
) -> torch.Tensor:
    if x1.ndim < 2 or x2.shape != x1.shape or residual.shape != x1.shape:
        raise ValueError(
            "x1, x2, and residual must share shape with ndim >= 2"
        )

    orig_shape = x1.shape
    hidden_size = orig_shape[-1]
    out = torch.empty_like(x1)
    if x1.numel() == 0:
        return out

    x1_2d = x1 if x1.ndim == 2 else _as_2d_view(x1)
    x2_2d = x2 if x2.ndim == 2 else _as_2d_view(x2)
    res_2d = residual if residual.ndim == 2 else _as_2d_view(residual)
    out_2d = out if out.ndim == 2 else out.view(-1, hidden_size)
    num_rows = x1_2d.shape[0]

    stride_x1m = x1_2d.stride(0)
    stride_x1d = x1_2d.stride(1)
    stride_x2m = x2_2d.stride(0)
    stride_x2d = x2_2d.stride(1)
    stride_rm = res_2d.stride(0)
    stride_rd = res_2d.stride(1)
    stride_w1 = weight1.stride(0)
    stride_w2 = weight2.stride(0)
    stride_w3 = weight3.stride(0)

    with _maybe_device_ctx(x1.device):
        if hidden_size <= 16384:
            block_d = 1 << (hidden_size - 1).bit_length()
            num_warps = _select_num_warps(block_d)
            contiguous = (
                stride_x1d == 1
                and stride_x2d == 1
                and stride_rd == 1
                and stride_w1 == 1
                and stride_w2 == 1
                and stride_w3 == 1
                and stride_x1m == hidden_size
                and stride_x2m == hidden_size
                and stride_rm == hidden_size
            )
            if contiguous:
                _gemma_dual_rmsnorm_contig_kernel[(num_rows,)](
                    out_2d,
                    x1_2d,
                    x2_2d,
                    res_2d,
                    weight1,
                    weight2,
                    weight3,
                    scalar,
                    float(eps1),
                    float(eps2),
                    float(eps3),
                    D=hidden_size,
                    BLOCK_D=block_d,
                    EVEN_D=(hidden_size == block_d),
                    USE_CG=(num_rows >= 4),
                    USE_CS=(
                        num_rows * hidden_size * x1.element_size() >= 8388608
                    ),
                    num_warps=num_warps,
                )
            else:
                _gemma_dual_rmsnorm_strided_kernel[(num_rows,)](
                    out_2d,
                    x1_2d,
                    x2_2d,
                    res_2d,
                    weight1,
                    weight2,
                    weight3,
                    scalar,
                    stride_x1m,
                    stride_x1d,
                    stride_x2m,
                    stride_x2d,
                    stride_rm,
                    stride_rd,
                    stride_w1,
                    stride_w2,
                    stride_w3,
                    out_2d.stride(0),
                    float(eps1),
                    float(eps2),
                    float(eps3),
                    D=hidden_size,
                    BLOCK_D=block_d,
                    num_warps=num_warps,
                )
        else:
            _gemma_dual_rmsnorm_large_d_kernel[(num_rows,)](
                out_2d,
                x1_2d,
                x2_2d,
                res_2d,
                weight1,
                weight2,
                weight3,
                scalar,
                stride_x1m,
                stride_x1d,
                stride_x2m,
                stride_x2d,
                stride_rm,
                stride_rd,
                stride_w1,
                stride_w2,
                stride_w3,
                out_2d.stride(0),
                float(eps1),
                float(eps2),
                float(eps3),
                D=hidden_size,
                BLOCK_D=4096,
                num_warps=8,
            )
    return out


__all__ = ["gemma_dual_rmsnorm_residual_scalar"]
