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

# Rows at or below this width are reduced by a single register-resident
# program; wider rows fall back to a looping multi-pass variant.
_SINGLE_BLOCK_MAX = 16384


@triton.autotune(
    configs=[
        triton.Config({"ROW_MOD": m}, num_warps=nw)
        for nw in (2, 4, 8, 16)
        for m in ("", ".cg")
    ],
    key=["M", "N", "BLOCK_N"],
)
@triton.jit
def _gemma_dual_rmsnorm_residual_scalar_kernel(
    x1_ptr,
    x2_ptr,
    w1_ptr,
    w2_ptr,
    w3_ptr,
    res_ptr,
    scalar_ptr,
    out_ptr,
    M,
    stride_row,
    N,
    eps1,
    eps2,
    eps3,
    BLOCK_N: tl.constexpr,
    EVEN_N: tl.constexpr,
    ROW_MOD: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N
    base = row * stride_row

    if EVEN_N:
        x1 = tl.load(x1_ptr + base + cols, cache_modifier=ROW_MOD).to(
            tl.float32
        )
        x2 = tl.load(x2_ptr + base + cols, cache_modifier=ROW_MOD).to(
            tl.float32
        )
        res = tl.load(res_ptr + base + cols, cache_modifier=ROW_MOD).to(
            tl.float32
        )

        w1 = tl.load(w1_ptr + cols).to(tl.float32)
        w2 = tl.load(w2_ptr + cols).to(tl.float32)
        w3 = tl.load(w3_ptr + cols).to(tl.float32)
    else:
        x1 = tl.load(
            x1_ptr + base + cols, mask=mask, other=0.0, cache_modifier=ROW_MOD
        ).to(tl.float32)
        x2 = tl.load(
            x2_ptr + base + cols, mask=mask, other=0.0, cache_modifier=ROW_MOD
        ).to(tl.float32)
        res = tl.load(
            res_ptr + base + cols, mask=mask, other=0.0, cache_modifier=ROW_MOD
        ).to(tl.float32)
        w1 = tl.load(w1_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        w2 = tl.load(w2_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        w3 = tl.load(w3_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    scalar = tl.load(scalar_ptr).to(tl.float32)

    rstd1 = tl.rsqrt(tl.sum(x1 * x1, axis=0) / N + eps1)
    rstd2 = tl.rsqrt(tl.sum(x2 * x2, axis=0) / N + eps2)

    # t = n1 + n2; keep it live for the third norm and the epilogue.
    t = x1 * rstd1 * w1 + x2 * rstd2 * w2

    rstd3 = tl.rsqrt(tl.sum(t * t, axis=0) / N + eps3)

    out = (t * rstd3 * w3 + res) * scalar
    if EVEN_N:
        tl.store(out_ptr + base + cols, out.to(out_ptr.dtype.element_ty))
    else:
        tl.store(
            out_ptr + base + cols, out.to(out_ptr.dtype.element_ty), mask=mask
        )


@triton.jit
def _gemma_dual_rmsnorm_residual_scalar_kernel_loop(
    x1_ptr,
    x2_ptr,
    w1_ptr,
    w2_ptr,
    w3_ptr,
    res_ptr,
    scalar_ptr,
    out_ptr,
    stride_row,
    N,
    eps1,
    eps2,
    eps3,
    BLOCK_N: tl.constexpr,
):
    """Two-pass variant for rows larger than a single register-."""
    row = tl.program_id(0).to(tl.int64)
    base = row * stride_row
    scalar = tl.load(scalar_ptr).to(tl.float32)

    # Pass 1: sum of squares of x1 and x2.
    sumsq1 = 0.0
    sumsq2 = 0.0
    for start in tl.range(0, N, BLOCK_N):
        cols = start + tl.arange(0, BLOCK_N)
        mask = cols < N
        x1 = tl.load(x1_ptr + base + cols, mask=mask, other=0.0).to(tl.float32)
        x2 = tl.load(x2_ptr + base + cols, mask=mask, other=0.0).to(tl.float32)
        sumsq1 += tl.sum(x1 * x1, axis=0)
        sumsq2 += tl.sum(x2 * x2, axis=0)
    rstd1 = tl.rsqrt(sumsq1 / N + eps1)
    rstd2 = tl.rsqrt(sumsq2 / N + eps2)

    # Pass 2: t = n1 + n2, accumulate its sum of squares.
    sumsq3 = 0.0
    for start in tl.range(0, N, BLOCK_N):
        cols = start + tl.arange(0, BLOCK_N)
        mask = cols < N
        x1 = tl.load(x1_ptr + base + cols, mask=mask, other=0.0).to(tl.float32)
        x2 = tl.load(x2_ptr + base + cols, mask=mask, other=0.0).to(tl.float32)
        w1 = tl.load(w1_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        w2 = tl.load(w2_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        t = x1 * rstd1 * w1 + x2 * rstd2 * w2
        sumsq3 += tl.sum(t * t, axis=0)
    rstd3 = tl.rsqrt(sumsq3 / N + eps3)

    # Pass 3: epilogue (inputs re-read, L2-resident).
    for start in tl.range(0, N, BLOCK_N):
        cols = start + tl.arange(0, BLOCK_N)
        mask = cols < N
        x1 = tl.load(x1_ptr + base + cols, mask=mask, other=0.0).to(tl.float32)
        x2 = tl.load(x2_ptr + base + cols, mask=mask, other=0.0).to(tl.float32)
        w1 = tl.load(w1_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        w2 = tl.load(w2_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        w3 = tl.load(w3_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        res = tl.load(res_ptr + base + cols, mask=mask, other=0.0).to(
            tl.float32
        )
        t = x1 * rstd1 * w1 + x2 * rstd2 * w2
        out = (t * rstd3 * w3 + res) * scalar
        tl.store(
            out_ptr + base + cols, out.to(out_ptr.dtype.element_ty), mask=mask
        )


def gemma_dual_rmsnorm_residual_scalar(
    x1, weight1, x2, weight2, weight3, residual, scalar, eps1, eps2, eps3
):
    if not x1.is_contiguous():
        x1 = x1.contiguous()
    if not x2.is_contiguous():
        x2 = x2.contiguous()
    if not residual.is_contiguous():
        residual = residual.contiguous()

    M, N = x1.shape
    out = torch.empty_like(x1)

    if N <= _SINGLE_BLOCK_MAX:
        BLOCK_N = triton.next_power_of_2(N)

        if BLOCK_N == N:
            _gemma_dual_rmsnorm_residual_scalar_kernel[(M,)](
                x1,
                x2,
                weight1,
                weight2,
                weight3,
                residual,
                scalar,
                out,
                M,
                x1.stride(0),
                N,
                eps1,
                eps2,
                eps3,
                BLOCK_N=BLOCK_N,
                EVEN_N=True,
            )
        else:
            _gemma_dual_rmsnorm_residual_scalar_kernel.fn[(M,)](
                x1,
                x2,
                weight1,
                weight2,
                weight3,
                residual,
                scalar,
                out,
                M,
                x1.stride(0),
                N,
                eps1,
                eps2,
                eps3,
                BLOCK_N=BLOCK_N,
                EVEN_N=False,
                ROW_MOD="",
                num_warps=4,
            )
    else:
        _gemma_dual_rmsnorm_residual_scalar_kernel_loop[(M,)](
            x1,
            x2,
            weight1,
            weight2,
            weight3,
            residual,
            scalar,
            out,
            x1.stride(0),
            N,
            eps1,
            eps2,
            eps3,
            BLOCK_N=4096,
            num_warps=4,
        )
    return out


__all__ = ["gemma_dual_rmsnorm_residual_scalar"]
