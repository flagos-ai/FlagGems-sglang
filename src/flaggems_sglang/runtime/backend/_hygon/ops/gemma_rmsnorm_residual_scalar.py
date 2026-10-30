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

# Rows at or below this width are reduced by a single register-resident
# program; wider rows fall back to a looping two-pass variant.
_SINGLE_BLOCK_MAX = 16384


@triton.autotune(
    configs=[
        triton.Config({"ROW_MOD": m, "ROW_EVICT": ev}, num_warps=nw)
        for nw in (2, 4, 8, 16)
        for m in ("", ".cg")
        for ev in (False, True)
    ],
    key=["M", "N", "BLOCK_N"],
)
@triton.jit
def _gemma_rmsnorm_residual_scalar_kernel(
    x_ptr,
    w_ptr,
    res_ptr,
    scalar_ptr,
    out_ptr,
    M,
    stride_row,
    N,
    eps,
    BLOCK_N: tl.constexpr,
    EVEN_N: tl.constexpr,
    ROW_MOD: tl.constexpr,
    ROW_EVICT: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N
    base = row * stride_row

    if ROW_EVICT:
        # Streaming rows: hint that each line can be dropped from cache right
        # after this CTA's single read (identical results, cache hint only).
        if EVEN_N:
            x = tl.load(
                x_ptr + base + cols,
                cache_modifier=ROW_MOD,
                eviction_policy="evict_first",
            ).to(tl.float32)
        else:
            x = tl.load(
                x_ptr + base + cols,
                mask=mask,
                other=0.0,
                cache_modifier=ROW_MOD,
                eviction_policy="evict_first",
            ).to(tl.float32)
    else:
        if EVEN_N:
            x = tl.load(x_ptr + base + cols, cache_modifier=ROW_MOD).to(
                tl.float32
            )
        else:
            x = tl.load(
                x_ptr + base + cols,
                mask=mask,
                other=0.0,
                cache_modifier=ROW_MOD,
            ).to(tl.float32)
    w = (
        tl.load(w_ptr + cols, mask=mask, other=0.0)
        if not EVEN_N
        else tl.load(w_ptr + cols)
    )
    w = w.to(tl.float32)
    if ROW_EVICT:
        if EVEN_N:
            res = tl.load(
                res_ptr + base + cols,
                cache_modifier=ROW_MOD,
                eviction_policy="evict_first",
            ).to(tl.float32)
        else:
            res = tl.load(
                res_ptr + base + cols,
                mask=mask,
                other=0.0,
                cache_modifier=ROW_MOD,
                eviction_policy="evict_first",
            ).to(tl.float32)
    else:
        if EVEN_N:
            res = tl.load(res_ptr + base + cols, cache_modifier=ROW_MOD).to(
                tl.float32
            )
        else:
            res = tl.load(
                res_ptr + base + cols,
                mask=mask,
                other=0.0,
                cache_modifier=ROW_MOD,
            ).to(tl.float32)
    scalar = tl.load(scalar_ptr).to(tl.float32)

    rstd = tl.rsqrt(tl.sum(x * x, axis=0) / N + eps)

    out = (x * rstd * w + res) * scalar
    if EVEN_N:
        tl.store(out_ptr + base + cols, out.to(out_ptr.dtype.element_ty))
    else:
        tl.store(
            out_ptr + base + cols, out.to(out_ptr.dtype.element_ty), mask=mask
        )


@triton.jit
def _gemma_rmsnorm_residual_scalar_kernel_loop(
    x_ptr,
    w_ptr,
    res_ptr,
    scalar_ptr,
    out_ptr,
    stride_row,
    N,
    eps,
    BLOCK_N: tl.constexpr,
):
    """Two-pass variant for rows larger than a single register-."""
    row = tl.program_id(0).to(tl.int64)
    base = row * stride_row
    scalar = tl.load(scalar_ptr).to(tl.float32)

    # Pass 1: sum of squares of x.
    sumsq = 0.0
    for start in tl.range(0, N, BLOCK_N):
        cols = start + tl.arange(0, BLOCK_N)
        mask = cols < N
        x = tl.load(x_ptr + base + cols, mask=mask, other=0.0).to(tl.float32)
        sumsq += tl.sum(x * x, axis=0)
    rstd = tl.rsqrt(sumsq / N + eps)

    for start in tl.range(0, N, BLOCK_N):
        cols = start + tl.arange(0, BLOCK_N)
        mask = cols < N
        x = tl.load(x_ptr + base + cols, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        res = tl.load(res_ptr + base + cols, mask=mask, other=0.0).to(
            tl.float32
        )
        out = (x * rstd * w + res) * scalar
        tl.store(
            out_ptr + base + cols, out.to(out_ptr.dtype.element_ty), mask=mask
        )


def gemma_rmsnorm_residual_scalar(x, weight, residual, scalar, eps):
    if not x.is_contiguous():
        x = x.contiguous()
    if not residual.is_contiguous():
        residual = residual.contiguous()

    M, N = x.shape
    out = torch.empty_like(x)

    if N <= _SINGLE_BLOCK_MAX:
        BLOCK_N = triton.next_power_of_2(N)

        if BLOCK_N == N:
            _gemma_rmsnorm_residual_scalar_kernel[(M,)](
                x,
                weight,
                residual,
                scalar,
                out,
                M,
                x.stride(0),
                N,
                eps,
                BLOCK_N=BLOCK_N,
                EVEN_N=True,
            )
        else:
            _gemma_rmsnorm_residual_scalar_kernel.fn[(M,)](
                x,
                weight,
                residual,
                scalar,
                out,
                M,
                x.stride(0),
                N,
                eps,
                BLOCK_N=BLOCK_N,
                EVEN_N=False,
                ROW_MOD="",
                ROW_EVICT=False,
                num_warps=4,
            )
    else:
        _gemma_rmsnorm_residual_scalar_kernel_loop[(M,)](
            x,
            weight,
            residual,
            scalar,
            out,
            x.stride(0),
            N,
            eps,
            BLOCK_N=4096,
            num_warps=4,
        )
    return out


__all__ = ["gemma_rmsnorm_residual_scalar"]
