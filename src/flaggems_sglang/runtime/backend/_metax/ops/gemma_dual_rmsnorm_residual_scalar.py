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

"""Operator: activation_norm/gemma_dual_rmsnorm_residual_scalar"""

import torch
import triton
import triton.language as tl

_INT32_MAX_ELEMS = 2**31 - 1


@triton.jit
def _gemma_dual_rmsnorm_residual_scalar_kernel(
    x1_ptr,
    w1_ptr,
    x2_ptr,
    w2_ptr,
    w3_ptr,
    res_ptr,
    scalar_ptr,
    out_ptr,
    n_cols,
    eps1,
    eps2,
    eps3,
    BLOCK: tl.constexpr,
    EVEN: tl.constexpr,
    WIDE: tl.constexpr,
):
    row = tl.program_id(0)
    if WIDE:
        row = row.to(tl.int64)
    cols = tl.arange(0, BLOCK)
    cols = tl.max_contiguous(tl.multiple_of(cols, BLOCK), BLOCK)
    off = row * n_cols + cols

    if EVEN:
        x1 = tl.load(x1_ptr + off).to(tl.float32)
        x2 = tl.load(x2_ptr + off).to(tl.float32)
    else:
        mask = cols < n_cols
        x1 = tl.load(x1_ptr + off, mask=mask, other=0.0).to(tl.float32)
        x2 = tl.load(x2_ptr + off, mask=mask, other=0.0).to(tl.float32)

    # rmsnorm(x1, w1) and rmsnorm(x2, w2). For pow2 N the mean is the exact
    # reciprocal multiply; the masked path divides by the runtime n_cols.
    if EVEN:
        rstd1 = tl.rsqrt(tl.sum(x1 * x1, axis=0) * (1.0 / BLOCK) + eps1)
        rstd2 = tl.rsqrt(tl.sum(x2 * x2, axis=0) * (1.0 / BLOCK) + eps2)
    else:
        rstd1 = tl.rsqrt(tl.sum(x1 * x1, axis=0) / n_cols + eps1)
        rstd2 = tl.rsqrt(tl.sum(x2 * x2, axis=0) / n_cols + eps2)

    if EVEN:
        w1 = tl.load(w1_ptr + cols).to(tl.float32)
        w2 = tl.load(w2_ptr + cols).to(tl.float32)
    else:
        w1 = tl.load(w1_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        w2 = tl.load(w2_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    h = x1 * rstd1 * w1 + x2 * rstd2 * w2

    # rmsnorm(n1 + n2, w3)
    if EVEN:
        rstd3 = tl.rsqrt(tl.sum(h * h, axis=0) * (1.0 / BLOCK) + eps3)
    else:
        rstd3 = tl.rsqrt(tl.sum(h * h, axis=0) / n_cols + eps3)
    if EVEN:
        w3 = tl.load(w3_ptr + cols).to(tl.float32)
        res = tl.load(res_ptr + off).to(tl.float32)
    else:
        w3 = tl.load(w3_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        res = tl.load(res_ptr + off, mask=mask, other=0.0).to(tl.float32)
    out = (h * rstd3 * w3 + res) * tl.load(scalar_ptr).to(tl.float32)

    out = out.to(out_ptr.dtype.element_ty)
    if EVEN:
        tl.store(out_ptr + off, out)
    else:
        tl.store(out_ptr + off, out, mask=mask)


def _launch_config(block, n_rows):
    """Static (num_warps, num_stages) for a given BLOCK size and row."""
    if block <= 1024:
        num_warps = 1
    elif block <= 4096:
        num_warps = 8 if n_rows < 512 else 4
    else:
        num_warps = 8
    return num_warps, 1


def gemma_dual_rmsnorm_residual_scalar(
    x1, weight1, x2, weight2, weight3, residual, scalar, eps1, eps2, eps3
):
    if not x1.is_contiguous():
        x1 = x1.contiguous()
    if not x2.is_contiguous():
        x2 = x2.contiguous()
    if not residual.is_contiguous():
        residual = residual.contiguous()
    if not weight1.is_contiguous():
        weight1 = weight1.contiguous()
    if not weight2.is_contiguous():
        weight2 = weight2.contiguous()
    if not weight3.is_contiguous():
        weight3 = weight3.contiguous()

    m, n = x1.shape
    out = torch.empty_like(x1)
    if m == 0 or n == 0:
        return out

    block = triton.next_power_of_2(n)
    num_warps, num_stages = _launch_config(block, m)
    _gemma_dual_rmsnorm_residual_scalar_kernel[(m,)](
        x1,
        weight1,
        x2,
        weight2,
        weight3,
        residual,
        scalar,
        out,
        n,
        eps1,
        eps2,
        eps3,
        BLOCK=block,
        EVEN=(n == block),
        WIDE=(m * n >= _INT32_MAX_ELEMS),
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return out


__all__ = ["gemma_dual_rmsnorm_residual_scalar"]
