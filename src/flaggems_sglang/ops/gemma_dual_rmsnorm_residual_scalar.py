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

"""High-performance Triton kernel for activation_norm/gemma_dual_rmsnorm_residual_scalar."""

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
    """Deterministic multi-row CTA & warp heuristic (reuses w1/w2/w3/scalar in registers)."""
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
def _gemma_dual_rmsnorm_residual_scalar_kernel(
    out_ptr,
    x1_ptr,
    x2_ptr,
    residual_ptr,
    weight1_ptr,
    weight2_ptr,
    weight3_ptr,
    scalar_ptr,
    num_rows,
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
    ROWS_PER_CTA: tl.constexpr,
    CONTIGUOUS_INNER: tl.constexpr,
    STREAM_INPUTS: tl.constexpr,
):
    pid = tl.program_id(0)
    cols = tl.arange(0, BLOCK_D).to(tl.int64)
    mask = cols < D
    inv_d = 1.0 / D

    w1 = tl.load(
        weight1_ptr + cols * stride_w1,
        mask=mask,
        other=0.0,
        eviction_policy="evict_last",
    ).to(tl.float32)
    w2 = tl.load(
        weight2_ptr + cols * stride_w2,
        mask=mask,
        other=0.0,
        eviction_policy="evict_last",
    ).to(tl.float32)
    w3 = tl.load(
        weight3_ptr + cols * stride_w3,
        mask=mask,
        other=0.0,
        eviction_policy="evict_last",
    ).to(tl.float32)
    scalar = tl.load(scalar_ptr, eviction_policy="evict_last").to(tl.float32)

    stride_x1m64 = tl.full((), stride_x1m, tl.int64)
    stride_x2m64 = tl.full((), stride_x2m, tl.int64)
    stride_rm64 = tl.full((), stride_rm, tl.int64)
    stride_om64 = tl.full((), stride_om, tl.int64)
    stride_x1d64 = tl.full((), stride_x1d, tl.int64)
    stride_x2d64 = tl.full((), stride_x2d, tl.int64)
    stride_rd64 = tl.full((), stride_rd, tl.int64)

    eps1_f = tl.full((), eps1, tl.float32)
    eps2_f = tl.full((), eps2, tl.float32)
    eps3_f = tl.full((), eps3, tl.float32)
    out_ty = out_ptr.dtype.element_ty

    base_row = pid * ROWS_PER_CTA
    for r in tl.static_range(0, ROWS_PER_CTA):
        row = base_row + r
        if row < num_rows:
            row64 = tl.full((), row, tl.int64)
            if CONTIGUOUS_INNER:
                p_x1 = x1_ptr + row64 * stride_x1m64 + cols
                p_x2 = x2_ptr + row64 * stride_x2m64 + cols
                p_res = residual_ptr + row64 * stride_rm64 + cols
            else:
                p_x1 = x1_ptr + row64 * stride_x1m64 + cols * stride_x1d64
                p_x2 = x2_ptr + row64 * stride_x2m64 + cols * stride_x2d64
                p_res = residual_ptr + row64 * stride_rm64 + cols * stride_rd64

            if STREAM_INPUTS:
                x1 = tl.load(
                    p_x1, mask=mask, other=0.0, eviction_policy="evict_first"
                ).to(tl.float32)
                x2 = tl.load(
                    p_x2, mask=mask, other=0.0, eviction_policy="evict_first"
                ).to(tl.float32)
                res = tl.load(
                    p_res, mask=mask, other=0.0, eviction_policy="evict_first"
                ).to(tl.float32)
            else:
                x1 = tl.load(p_x1, mask=mask, other=0.0).to(tl.float32)
                x2 = tl.load(p_x2, mask=mask, other=0.0).to(tl.float32)
                res = tl.load(p_res, mask=mask, other=0.0).to(tl.float32)

            n1 = x1 * tl.rsqrt(tl.sum(x1 * x1, axis=0) * inv_d + eps1_f) * w1
            n2 = x2 * tl.rsqrt(tl.sum(x2 * x2, axis=0) * inv_d + eps2_f) * w2
            comb = n1 + n2
            n3 = (
                comb
                * tl.rsqrt(tl.sum(comb * comb, axis=0) * inv_d + eps3_f)
                * w3
            )
            out = (n3 + res) * scalar
            tl.store(
                out_ptr + row64 * stride_om64 + cols, out.to(out_ty), mask=mask
            )


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
        cols = (start + tl.arange(0, BLOCK_D)).to(tl.int64)
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
        cols = (start + tl.arange(0, BLOCK_D)).to(tl.int64)
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
        cols = (start + tl.arange(0, BLOCK_D)).to(tl.int64)
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
    out = torch.empty(orig_shape, dtype=x1.dtype, device=x1.device)
    if x1.numel() == 0:
        return out

    x1_2d = _as_2d_view(x1)
    x2_2d = _as_2d_view(x2)
    res_2d = _as_2d_view(residual)
    out_2d = out.view(-1, hidden_size)
    num_rows = x1_2d.shape[0]

    with _device_ctx(x1.device):
        if hidden_size <= 16384:
            block_d = triton.next_power_of_2(hidden_size)
            rows_per_cta, num_warps = _select_config(num_rows, block_d)
            contiguous_inner = (
                x1_2d.stride(1) == 1
                and x2_2d.stride(1) == 1
                and res_2d.stride(1) == 1
            )
            stream_inputs = (
                hidden_size >= 1024
                and x1.numel() * x1.element_size() >= 1048576
            )
            grid = (triton.cdiv(num_rows, rows_per_cta),)
            _gemma_dual_rmsnorm_residual_scalar_kernel[grid](
                out_2d,
                x1_2d,
                x2_2d,
                res_2d,
                weight1,
                weight2,
                weight3,
                scalar,
                num_rows,
                x1_2d.stride(0),
                x1_2d.stride(1),
                x2_2d.stride(0),
                x2_2d.stride(1),
                res_2d.stride(0),
                res_2d.stride(1),
                weight1.stride(0),
                weight2.stride(0),
                weight3.stride(0),
                out_2d.stride(0),
                float(eps1),
                float(eps2),
                float(eps3),
                D=hidden_size,
                BLOCK_D=block_d,
                ROWS_PER_CTA=rows_per_cta,
                CONTIGUOUS_INNER=contiguous_inner,
                STREAM_INPUTS=stream_inputs,
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
                x1_2d.stride(0),
                x1_2d.stride(1),
                x2_2d.stride(0),
                x2_2d.stride(1),
                res_2d.stride(0),
                res_2d.stride(1),
                weight1.stride(0),
                weight2.stride(0),
                weight3.stride(0),
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
