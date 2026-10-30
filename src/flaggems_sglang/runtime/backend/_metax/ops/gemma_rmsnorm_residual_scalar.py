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

"""Operator: activation_norm/gemma_rmsnorm_residual_scalar"""

import torch
import triton
import triton.language as tl

_INT32_MAX_ELEMS = 2**31 - 1


@triton.jit
def _kernel_even(
    x_ptr,
    w_ptr,
    res_ptr,
    scalar_ptr,
    out_ptr,
    eps,
    BLOCK: tl.constexpr,
    WIDE: tl.constexpr,
    ORDER: tl.constexpr,
):
    # Hoist the single-element scalar load ahead of the row loads so it is
    # not a dependent global load on the epilogue's critical path.
    scalar = tl.load(scalar_ptr).to(tl.float32)

    # One program per row; n_cols == BLOCK so the row stride is a constant.
    row = tl.program_id(0)
    if WIDE:
        row = row.to(tl.int64)
    cols = tl.arange(0, BLOCK)
    cols = tl.max_contiguous(tl.multiple_of(cols, BLOCK), BLOCK)
    off = row * BLOCK + cols

    x = tl.load(x_ptr + off).to(tl.float32)
    if ORDER == 1:
        res = tl.load(res_ptr + off).to(tl.float32)
        w = tl.load(w_ptr + cols).to(tl.float32)
    rstd = tl.rsqrt(tl.sum(x * x, axis=0) * (1.0 / BLOCK) + eps)
    if ORDER == 0:
        w = tl.load(w_ptr + cols).to(tl.float32)
        res = tl.load(res_ptr + off).to(tl.float32)
    out = (x * rstd * w + res) * scalar
    tl.store(out_ptr + off, out.to(out_ptr.dtype.element_ty))


@triton.jit
def _kernel_general(
    x_ptr,
    w_ptr,
    res_ptr,
    scalar_ptr,
    out_ptr,
    n_cols,
    eps,
    BLOCK: tl.constexpr,
    EVEN: tl.constexpr,
    WIDE: tl.constexpr,
    ORDER: tl.constexpr,
):
    scalar = tl.load(scalar_ptr).to(tl.float32)

    row = tl.program_id(0)
    if WIDE:
        row = row.to(tl.int64)
    cols = tl.arange(0, BLOCK)
    off = row * n_cols + cols

    if EVEN:
        x = tl.load(x_ptr + off).to(tl.float32)
        if ORDER == 1:
            res = tl.load(res_ptr + off).to(tl.float32)
            w = tl.load(w_ptr + cols).to(tl.float32)
        rstd = tl.rsqrt(tl.sum(x * x, axis=0) * (1.0 / BLOCK) + eps)
        if ORDER == 0:
            w = tl.load(w_ptr + cols).to(tl.float32)
            res = tl.load(res_ptr + off).to(tl.float32)
        out = (x * rstd * w + res) * scalar
        tl.store(out_ptr + off, out.to(out_ptr.dtype.element_ty))
    else:
        mask = cols < n_cols
        x = tl.load(x_ptr + off, mask=mask, other=0.0).to(tl.float32)
        if ORDER == 1:
            res = tl.load(res_ptr + off, mask=mask, other=0.0).to(tl.float32)
            w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        rstd = tl.rsqrt(tl.sum(x * x, axis=0) / n_cols + eps)
        if ORDER == 0:
            w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
            res = tl.load(res_ptr + off, mask=mask, other=0.0).to(tl.float32)
        out = (x * rstd * w + res) * scalar
        tl.store(out_ptr + off, out.to(out_ptr.dtype.element_ty), mask=mask)


def _launch_config(block, n_rows):
    """Static (num_warps, ORDER) for a given BLOCK size and row count."""
    if n_rows < 512:
        return (8 if block >= 8192 else 4), 1
    if block <= 1024:
        return 1, 0
    if block <= 4096:
        return 4, 0
    return 8, 0


def gemma_rmsnorm_residual_scalar(x, weight, residual, scalar, eps):
    if not x.is_contiguous():
        x = x.contiguous()
    if not residual.is_contiguous():
        residual = residual.contiguous()
    if not weight.is_contiguous():
        weight = weight.contiguous()

    m, n = x.shape
    out = torch.empty_like(x)
    if m == 0 or n == 0:
        return out

    block = triton.next_power_of_2(n)
    num_warps, order = _launch_config(block, m)
    wide = m * n >= _INT32_MAX_ELEMS
    if n == block:
        # Exact power-of-two rows: n_cols folds into the constexpr BLOCK.
        _kernel_even[(m,)](
            x,
            weight,
            residual,
            scalar,
            out,
            eps,
            BLOCK=block,
            WIDE=wide,
            ORDER=order,
            num_warps=num_warps,
        )
    else:
        _kernel_general[(m,)](
            x,
            weight,
            residual,
            scalar,
            out,
            n,
            eps,
            BLOCK=block,
            EVEN=False,
            WIDE=wide,
            ORDER=order,
            num_warps=num_warps,
        )
    return out


__all__ = ["gemma_rmsnorm_residual_scalar"]
