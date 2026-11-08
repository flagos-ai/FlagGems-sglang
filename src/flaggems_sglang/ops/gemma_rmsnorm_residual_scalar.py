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

"""High-performance Triton kernel for activation_norm/gemma_rmsnorm_residual_scalar."""

from __future__ import annotations

import contextlib

import torch
import triton
import triton.language as tl


def _device_ctx(device: torch.device):
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


def _select_config(num_rows: int, block_d: int) -> tuple[int, int]:
    """Deterministic multi-row CTA & warp heuristic (reuses weight & scalar in registers)."""
    if block_d <= 2048 and num_rows >= 64:
        rows_per_cta = 4
    elif block_d <= 4096 and num_rows >= 32:
        rows_per_cta = 2
    else:
        rows_per_cta = 1

    if block_d <= 256:
        num_warps = 2
    elif block_d <= 1024:
        num_warps = 4
    elif block_d <= 4096:
        num_warps = 8
    else:
        num_warps = 16
    return rows_per_cta, num_warps


@triton.jit
def _gemma_rmsnorm_residual_scalar_kernel(
    x_ptr,
    weight_ptr,
    residual_ptr,
    scalar_ptr,
    out_ptr,
    num_rows,
    stride_xm,
    stride_xd,
    stride_rm,
    stride_rd,
    stride_w,
    stride_om,
    eps,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
    ROWS_PER_CTA: tl.constexpr,
    CONTIGUOUS_INNER: tl.constexpr,
    STREAM_INPUTS: tl.constexpr,
):
    pid = tl.program_id(0)
    cols = tl.arange(0, BLOCK_D).to(tl.int64)
    mask = cols < D
    inv_d = 1.0 / D

    w = tl.load(
        weight_ptr + cols * stride_w,
        mask=mask,
        other=0.0,
        eviction_policy="evict_last",
    ).to(tl.float32)
    scalar = tl.load(scalar_ptr, eviction_policy="evict_last").to(tl.float32)

    stride_xm64 = tl.full((), stride_xm, tl.int64)
    stride_xd64 = tl.full((), stride_xd, tl.int64)
    stride_rm64 = tl.full((), stride_rm, tl.int64)
    stride_rd64 = tl.full((), stride_rd, tl.int64)
    stride_om64 = tl.full((), stride_om, tl.int64)
    eps_f = tl.full((), eps, tl.float32)
    out_ty = out_ptr.dtype.element_ty

    base_row = pid * ROWS_PER_CTA
    for r in tl.static_range(0, ROWS_PER_CTA):
        row = base_row + r
        if row < num_rows:
            row64 = tl.full((), row, tl.int64)
            if CONTIGUOUS_INNER:
                p_x = x_ptr + row64 * stride_xm64 + cols
                p_res = residual_ptr + row64 * stride_rm64 + cols
            else:
                p_x = x_ptr + row64 * stride_xm64 + cols * stride_xd64
                p_res = residual_ptr + row64 * stride_rm64 + cols * stride_rd64

            if STREAM_INPUTS:
                x = tl.load(
                    p_x, mask=mask, other=0.0, eviction_policy="evict_first"
                ).to(tl.float32)
                res = tl.load(
                    p_res, mask=mask, other=0.0, eviction_policy="evict_first"
                ).to(tl.float32)
            else:
                x = tl.load(p_x, mask=mask, other=0.0).to(tl.float32)
                res = tl.load(p_res, mask=mask, other=0.0).to(tl.float32)

            normed = x * tl.rsqrt(tl.sum(x * x, axis=0) * inv_d + eps_f) * w
            out = (normed + res) * scalar
            tl.store(out_ptr + row64 * stride_om64 + cols, out.to(out_ty), mask=mask)


@triton.jit
def _gemma_rmsnorm_residual_scalar_large_d_kernel(
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
        cols = (start + tl.arange(0, BLOCK_D)).to(tl.int64)
        mask = cols < D
        x = tl.load(
            x_ptr + row64 * stride_xm + cols * stride_xd, mask=mask, other=0.0
        ).to(tl.float32)
        sum_sq += tl.sum(x * x, axis=0)

    rstd = tl.rsqrt(sum_sq * inv_d + eps)
    out_ty = out_ptr.dtype.element_ty
    for start in range(0, D, BLOCK_D):
        cols = (start + tl.arange(0, BLOCK_D)).to(tl.int64)
        mask = cols < D
        x = tl.load(
            x_ptr + row64 * stride_xm + cols * stride_xd, mask=mask, other=0.0
        ).to(tl.float32)
        res = tl.load(
            residual_ptr + row64 * stride_rm + cols * stride_rd, mask=mask, other=0.0
        ).to(tl.float32)
        w = tl.load(weight_ptr + cols * stride_w, mask=mask, other=0.0).to(tl.float32)
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
    out = torch.empty(orig_shape, dtype=x.dtype, device=x.device)
    if x.numel() == 0:
        return out

    x_2d = _as_2d_view(x)
    res_2d = _as_2d_view(residual)
    out_2d = out.view(-1, hidden_size)
    num_rows = x_2d.shape[0]

    with _device_ctx(x.device):
        if hidden_size <= 16384:
            block_d = triton.next_power_of_2(hidden_size)
            rows_per_cta, num_warps = _select_config(num_rows, block_d)
            contiguous_inner = x_2d.stride(1) == 1 and res_2d.stride(1) == 1
            stream_inputs = (
                hidden_size >= 1024 and x.numel() * x.element_size() >= 1048576
            )
            grid = (triton.cdiv(num_rows, rows_per_cta),)
            _gemma_rmsnorm_residual_scalar_kernel[grid](
                x_2d,
                weight,
                res_2d,
                scalar,
                out_2d,
                num_rows,
                x_2d.stride(0),
                x_2d.stride(1),
                res_2d.stride(0),
                res_2d.stride(1),
                weight.stride(0),
                out_2d.stride(0),
                float(eps),
                D=hidden_size,
                BLOCK_D=block_d,
                ROWS_PER_CTA=rows_per_cta,
                CONTIGUOUS_INNER=contiguous_inner,
                STREAM_INPUTS=stream_inputs,
                num_warps=num_warps,
            )
        else:
            _gemma_rmsnorm_residual_scalar_large_d_kernel[(num_rows,)](
                x_2d,
                weight,
                res_2d,
                scalar,
                out_2d,
                x_2d.stride(0),
                x_2d.stride(1),
                res_2d.stride(0),
                res_2d.stride(1),
                weight.stride(0),
                out_2d.stride(0),
                float(eps),
                D=hidden_size,
                BLOCK_D=4096,
                num_warps=8,
            )
    return out


__all__ = ["gemma_rmsnorm_residual_scalar"]
