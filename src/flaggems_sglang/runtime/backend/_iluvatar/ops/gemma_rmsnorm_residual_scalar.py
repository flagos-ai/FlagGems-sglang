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


@triton.jit
def _grrs_1pass_kernel(
    X,
    W,
    R,
    S,
    OUT,
    row_stride_x,
    row_stride_r,
    N,
    eps,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    xb = tl.load(X + row * row_stride_x + cols, mask=mask, other=0.0)
    wb = tl.load(W + cols, mask=mask, other=0.0)
    rb = tl.load(R + row * row_stride_r + cols, mask=mask, other=0.0)
    s = tl.load(S).to(tl.float32)

    x32 = xb.to(tl.float32)
    var = tl.sum(x32 * x32, axis=0) / N
    rrms = tl.math.rsqrt(var + eps)

    y = (x32 * rrms * wb.to(tl.float32) + rb.to(tl.float32)) * s
    tl.store(OUT + row * N + cols, y.to(OUT.dtype.element_ty), mask=mask)


@triton.jit
def _grrs_1pass_exact_kernel(
    X,
    W,
    R,
    S,
    OUT,
    row_stride_x,
    row_stride_r,
    eps,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)

    xb = tl.load(X + row * row_stride_x + cols)
    wb = tl.load(W + cols)
    rb = tl.load(R + row * row_stride_r + cols)
    s = tl.load(S).to(tl.float32)

    x32 = xb.to(tl.float32)
    var = tl.sum(x32 * x32, axis=0) / BLOCK_N
    rrms = tl.math.rsqrt(var + eps)

    y = (x32 * rrms * wb.to(tl.float32) + rb.to(tl.float32)) * s
    tl.store(OUT + row * BLOCK_N + cols, y.to(OUT.dtype.element_ty))


@triton.jit
def _grrs_rows_kernel(
    X,
    W,
    R,
    S,
    OUT,
    row_stride_x,
    row_stride_r,
    M,
    N,
    eps,
    BLOCK_N: tl.constexpr,
    ROWS: tl.constexpr,
):
    # ROWS rows per program: fills the device when m is smaller than the SM
    # count (one row per program would leave most SMs idle).
    row0 = tl.program_id(0) * ROWS
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N
    rows = row0 + tl.arange(0, ROWS)
    rmask = rows < M
    m2 = rmask[:, None] & mask[None, :]

    xb = tl.load(
        X + rows[:, None] * row_stride_x + cols[None, :], mask=m2, other=0.0
    )
    wb = tl.load(W + cols, mask=mask, other=0.0)
    rb = tl.load(
        R + rows[:, None] * row_stride_r + cols[None, :], mask=m2, other=0.0
    )
    s = tl.load(S).to(tl.float32)

    x32 = xb.to(tl.float32)
    var = tl.sum(x32 * x32, axis=1) / N
    rrms = tl.math.rsqrt(var + eps)

    y = (
        x32 * rrms[:, None] * wb.to(tl.float32)[None, :] + rb.to(tl.float32)
    ) * s
    tl.store(
        OUT + rows[:, None] * N + cols[None, :],
        y.to(OUT.dtype.element_ty),
        mask=m2,
    )


@triton.jit
def _grrs_2pass_kernel(
    X,
    W,
    R,
    S,
    OUT,
    row_stride_x,
    row_stride_r,
    N,
    eps,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    x_row = X + row * row_stride_x

    # Hoist the scalar load above the streaming loop: it is independent of the
    # reduction and its latency would otherwise serialize behind the barrier.
    s = tl.load(S).to(tl.float32)

    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for off in tl.range(0, N, BLOCK_N, loop_unroll_factor=2):
        cols = off + tl.arange(0, BLOCK_N)
        mask = cols < N
        xv = tl.load(x_row + cols, mask=mask, other=0.0).to(tl.float32)
        acc += xv * xv
    var = tl.sum(acc, axis=0) / N
    rrms = tl.math.rsqrt(var + eps)

    # Pass 2: epilogue.
    r_row = R + row * row_stride_r
    for off in tl.range(0, N, BLOCK_N, loop_unroll_factor=2):
        cols = off + tl.arange(0, BLOCK_N)
        mask = cols < N
        xv = tl.load(x_row + cols, mask=mask, other=0.0).to(tl.float32)
        wv = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
        rv = tl.load(r_row + cols, mask=mask, other=0.0).to(tl.float32)
        y = (xv * rrms * wv + rv) * s
        tl.store(OUT + row * N + cols, y.to(OUT.dtype.element_ty), mask=mask)


@triton.jit
def _grrs_2pass_exact_kernel(
    X,
    W,
    R,
    S,
    OUT,
    row_stride_x,
    row_stride_r,
    N,
    eps,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    x_row = X + row * row_stride_x

    s = tl.load(S).to(tl.float32)

    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for off in tl.range(0, N, BLOCK_N, loop_unroll_factor=2):
        cols = off + tl.arange(0, BLOCK_N)
        xv = tl.load(x_row + cols).to(tl.float32)
        acc += xv * xv
    var = tl.sum(acc, axis=0) / N
    rrms = tl.math.rsqrt(var + eps)

    r_row = R + row * row_stride_r
    for off in tl.range(0, N, BLOCK_N, loop_unroll_factor=2):
        cols = off + tl.arange(0, BLOCK_N)
        xv = tl.load(x_row + cols).to(tl.float32)
        wv = tl.load(W + cols).to(tl.float32)
        rv = tl.load(r_row + cols).to(tl.float32)
        y = (xv * rrms * wv + rv) * s
        tl.store(OUT + row * N + cols, y.to(OUT.dtype.element_ty))


def _launch_2pass(
    x,
    weight,
    residual,
    scalar,
    out,
    m,
    n,
    row_stride_x,
    row_stride_r,
    eps,
    block_n=2048,
):
    # Streaming 2pass: mask-free exact tiles when BLOCK_N divides N.
    num_warps = 16 if block_n == 2048 else 32
    if n % block_n == 0:
        _grrs_2pass_exact_kernel[(m,)](
            x,
            weight,
            residual,
            scalar,
            out,
            row_stride_x,
            row_stride_r,
            n,
            eps,
            BLOCK_N=block_n,
            num_warps=num_warps,
            num_stages=2,
        )
    else:
        _grrs_2pass_kernel[(m,)](
            x,
            weight,
            residual,
            scalar,
            out,
            row_stride_x,
            row_stride_r,
            n,
            eps,
            BLOCK_N=block_n,
            num_warps=num_warps,
            num_stages=2,
        )


def gemma_rmsnorm_residual_scalar(x, weight, residual, scalar, eps):
    """``out = (x * rsqrt(mean(x^2,-1) + eps) * weight + residual)."""
    n = x.shape[-1]
    m = x.numel() // n

    # x / residual may be strided views whose last dim is contiguous;
    # stride(-2)
    # is then the row stride of the reshaped 2-D tensor.
    if x.stride(-1) != 1:
        x = x.contiguous()
    if residual.stride(-1) != 1:
        residual = residual.contiguous()
    if weight.stride(0) != 1:
        weight = weight.contiguous()
    row_stride_x = x.stride(-2) if x.dim() >= 2 else n
    row_stride_r = residual.stride(-2) if residual.dim() >= 2 else n

    out = torch.empty((m, n), dtype=x.dtype, device=x.device)

    def _launch_1pass(nw):
        block_n = triton.next_power_of_2(n)
        if block_n == n:
            _grrs_1pass_exact_kernel[(m,)](
                x,
                weight,
                residual,
                scalar,
                out,
                row_stride_x,
                row_stride_r,
                eps,
                BLOCK_N=block_n,
                num_warps=nw,
            )
        else:
            _grrs_1pass_kernel[(m,)](
                x,
                weight,
                residual,
                scalar,
                out,
                row_stride_x,
                row_stride_r,
                n,
                eps,
                BLOCK_N=block_n,
                num_warps=nw,
            )

    if n <= 2048:
        # Short rows always fit a register tile — single read of x,
        # always best.
        if 2 <= m < 16 and n <= 1024:
            _grrs_rows_kernel[(triton.cdiv(m, 2),)](
                x,
                weight,
                residual,
                scalar,
                out,
                row_stride_x,
                row_stride_r,
                m,
                n,
                eps,
                BLOCK_N=triton.next_power_of_2(n),
                ROWS=2,
                num_warps=16,
            )
        else:
            _launch_1pass(8 if n <= 1024 else 16)
    elif n <= 4096:
        # A 4096-wide row fits registers for moderate row counts; only a very
        # large batch makes the resident tile spill and 2pass worthwhile.
        if m <= 1024:
            _launch_1pass(32)
        else:
            _launch_2pass(
                x,
                weight,
                residual,
                scalar,
                out,
                m,
                n,
                row_stride_x,
                row_stride_r,
                eps,
            )
    elif m <= 16:
        _launch_1pass(64)
    else:
        _launch_2pass(
            x,
            weight,
            residual,
            scalar,
            out,
            m,
            n,
            row_stride_x,
            row_stride_r,
            eps,
            block_n=4096,
        )
    return out.view(x.shape)


__all__ = ["gemma_rmsnorm_residual_scalar"]
