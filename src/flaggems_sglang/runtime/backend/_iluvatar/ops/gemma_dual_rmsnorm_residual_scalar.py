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


@triton.jit
def _dual_rmsnorm_residual_scalar_kernel(
    x1_ptr,
    w1_ptr,
    x2_ptr,
    w2_ptr,
    w3_ptr,
    res_ptr,
    scalar_ptr,
    out_ptr,
    eps1,
    eps2,
    eps3,
    N: tl.constexpr,
    BLOCK_N: tl.constexpr,
    STREAM_ROW: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    offs = row * N + cols
    mask = cols < N

    if STREAM_ROW:
        x1 = tl.load(
            x1_ptr + offs, mask=mask, other=0.0, cache_modifier=".cv"
        ).to(tl.float32)
        x2 = tl.load(
            x2_ptr + offs, mask=mask, other=0.0, cache_modifier=".cv"
        ).to(tl.float32)
        res = tl.load(
            res_ptr + offs, mask=mask, other=0.0, cache_modifier=".cv"
        ).to(tl.float32)
    else:
        x1 = tl.load(x1_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        x2 = tl.load(x2_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        res = tl.load(res_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    w1 = tl.load(w1_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    w2 = tl.load(w2_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    w3 = tl.load(w3_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    scalar = tl.load(scalar_ptr).to(tl.float32)

    inv1 = tl.rsqrt(tl.sum(x1 * x1, axis=0) / N + eps1)
    inv2 = tl.rsqrt(tl.sum(x2 * x2, axis=0) / N + eps2)
    s = x1 * inv1 * w1 + x2 * inv2 * w2
    inv3 = tl.rsqrt(tl.sum(s * s, axis=0) / N + eps3)

    out = (s * inv3 * w3 + res) * scalar
    tl.store(out_ptr + offs, out.to(out_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _dual_rmsnorm_residual_scalar_chunked_kernel(
    x1_ptr,
    w1_ptr,
    x2_ptr,
    w2_ptr,
    w3_ptr,
    res_ptr,
    scalar_ptr,
    out_ptr,
    eps1,
    eps2,
    eps3,
    N: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    base = row * N

    # Sweep 1: the two input second moments.
    acc1 = tl.zeros((BLOCK_N,), dtype=tl.float32)
    acc2 = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for off in tl.static_range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)
        x1 = tl.load(x1_ptr + base + cols).to(tl.float32)
        x2 = tl.load(x2_ptr + base + cols).to(tl.float32)
        acc1 += x1 * x1
        acc2 += x2 * x2
    inv1 = tl.rsqrt(tl.sum(acc1, axis=0) / N + eps1)
    inv2 = tl.rsqrt(tl.sum(acc2, axis=0) / N + eps2)

    # Sweep 2: the mixed-norm second moment, from the recomputed sum.
    acc3 = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for off in tl.static_range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)
        x1 = tl.load(x1_ptr + base + cols).to(tl.float32)
        x2 = tl.load(x2_ptr + base + cols).to(tl.float32)
        w1 = tl.load(w1_ptr + cols).to(tl.float32)
        w2 = tl.load(w2_ptr + cols).to(tl.float32)
        s = x1 * inv1 * w1 + x2 * inv2 * w2
        acc3 += s * s
    inv3 = tl.rsqrt(tl.sum(acc3, axis=0) / N + eps3)

    # Sweep 3: emit the output row.
    scalar = tl.load(scalar_ptr).to(tl.float32)
    for off in tl.static_range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)
        x1 = tl.load(x1_ptr + base + cols).to(tl.float32)
        x2 = tl.load(x2_ptr + base + cols).to(tl.float32)
        w1 = tl.load(w1_ptr + cols).to(tl.float32)
        w2 = tl.load(w2_ptr + cols).to(tl.float32)
        w3 = tl.load(w3_ptr + cols).to(tl.float32)
        res = tl.load(res_ptr + base + cols).to(tl.float32)
        s = x1 * inv1 * w1 + x2 * inv2 * w2
        out = (s * inv3 * w3 + res) * scalar
        tl.store(out_ptr + base + cols, out.to(out_ptr.dtype.element_ty))


def gemma_dual_rmsnorm_residual_scalar(
    x1, weight1, x2, weight2, weight3, residual, scalar, eps1, eps2, eps3
):
    m, n = x1.shape
    out = torch.empty_like(x1)
    if m == 0:
        return out
    block_n = triton.next_power_of_2(n)
    chunked = (
        n % 2048 == 0
        and n >= 4096
        and (
            (m >= 4096)
            or (n == 4096 and 64 <= m < 512)
            or (n == 8192 and m >= 64)
        )
    )
    if chunked:
        _dual_rmsnorm_residual_scalar_chunked_kernel[(m,)](
            x1,
            weight1,
            x2,
            weight2,
            weight3,
            residual,
            scalar,
            out,
            eps1,
            eps2,
            eps3,
            N=n,
            BLOCK_N=2048,
            num_warps=16,
        )
        return out
    # Single-pass regime: warps per (rows, N) band + .cv row loads when the
    # row count is large enough that the streamed rows cannot stay in L2.
    if m >= 4096:
        num_warps = 8 if n <= 1024 else (16 if n <= 4096 else 32)
        stream = n <= 1024
    elif m >= 512:
        num_warps = 8 if n <= 1024 else 32
        stream = n != 8192
    else:
        num_warps = 8 if n <= 1024 else (16 if n <= 4096 else 32)
        stream = False
    _dual_rmsnorm_residual_scalar_kernel[(m,)](
        x1,
        weight1,
        x2,
        weight2,
        weight3,
        residual,
        scalar,
        out,
        eps1,
        eps2,
        eps3,
        N=n,
        BLOCK_N=block_n,
        STREAM_ROW=stream,
        num_warps=num_warps,
    )
    return out


__all__ = ["gemma_dual_rmsnorm_residual_scalar"]
