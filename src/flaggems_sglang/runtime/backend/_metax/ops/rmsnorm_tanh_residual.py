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

"""Fused RMS normalization, tanh gating, and residual addition."""

import torch
import triton
import triton.language as tl

# Widest row the one-pass register-resident path will allocate.  Beyond this
# the chunked two-pass kernel takes over (never hit by the benchmark widths).
_MAX_REG_WIDTH = 8192

# Address arithmetic stays int32 while N*D fits 2^31.
_MAX_I32_ELEMENTS = 2**31 - 1

_LATENCY_BOUND_N = 256


def _launch_cfg(n_rows, dim):
    """Return (PER, num_warps, WS, CS) for a flattened (n_rows,."""
    if n_rows <= _LATENCY_BOUND_N:
        return 1, 8, 0, 0
    if dim >= 4096:
        return 1, 4 if n_rows >= 2048 else 8, 1, 1
    return 1, (2 if n_rows >= 2048 else 4), 1, 1


def _next_pow2(n):
    return max(1, triton.next_power_of_2(n))


@triton.jit
def _rmsnorm_tanh_residual_kernel(
    x_ptr,
    g_ptr,
    r_ptr,
    w_ptr,
    y_ptr,
    eps,
    N,  # total rows (for the tail row mask)
    D: tl.constexpr,  # hidden width
    BLOCK_D: tl.constexpr,  # next_pow2(D) register tile
    GATE_REPEAT: tl.constexpr,  # x rows per gate row (row broadcast factor)
    PER: tl.constexpr,  # rows per program
    EXACT_ROWS: tl.constexpr,  # N % PER == 0 -> no row mask needed
    I32_ADDRS: tl.constexpr,
    WS: tl.constexpr,  # g/r/w load schedule selector
    CS: tl.constexpr,  # streaming-store selector
):
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK_D)
    EXACT: tl.constexpr = D == BLOCK_D
    mcol = offs < D
    OUT_DTYPE: tl.constexpr = y_ptr.dtype.element_ty

    for i in tl.static_range(PER):
        row = pid * PER + i
        if I32_ADDRS:
            xrow = row * D
            grow = row // GATE_REPEAT * D
        else:
            r64 = row.to(tl.int64)
            xrow = r64 * D
            grow = r64 // GATE_REPEAT * D

        if EXACT_ROWS:
            if EXACT:
                x = tl.load(
                    x_ptr + xrow + offs, eviction_policy="evict_first"
                ).to(tl.float32)
            else:
                x = tl.load(
                    x_ptr + xrow + offs,
                    mask=mcol,
                    other=0.0,
                    eviction_policy="evict_first",
                ).to(tl.float32)
        else:
            mrow = row < N
            x = tl.load(
                x_ptr + xrow + offs,
                mask=mrow if EXACT else mcol & mrow,
                other=0.0,
                eviction_policy="evict_first",
            ).to(tl.float32)

        if WS:
            rstd = tl.rsqrt(tl.sum(x * x, axis=0) / D + eps)
            if EXACT_ROWS:
                if EXACT:
                    g = tl.load(g_ptr + grow + offs).to(tl.float32)
                    res = tl.load(r_ptr + xrow + offs).to(tl.float32)
                    w = tl.load(w_ptr + offs).to(tl.float32)
                else:
                    g = tl.load(g_ptr + grow + offs, mask=mcol, other=0.0).to(
                        tl.float32
                    )
                    res = tl.load(
                        r_ptr + xrow + offs, mask=mcol, other=0.0
                    ).to(tl.float32)
                    w = tl.load(w_ptr + offs, mask=mcol, other=0.0).to(
                        tl.float32
                    )
            else:
                mrow = row < N
                g = tl.load(
                    g_ptr + grow + offs,
                    mask=mrow if EXACT else mcol & mrow,
                    other=0.0,
                ).to(tl.float32)
                res = tl.load(
                    r_ptr + xrow + offs,
                    mask=mrow if EXACT else mcol & mrow,
                    other=0.0,
                ).to(tl.float32)
                w = tl.load(
                    w_ptr + offs, mask=None if EXACT else mcol, other=0.0
                ).to(tl.float32)
        else:
            if EXACT_ROWS:
                if EXACT:
                    g = tl.load(g_ptr + grow + offs).to(tl.float32)
                    res = tl.load(r_ptr + xrow + offs).to(tl.float32)
                    w = tl.load(w_ptr + offs).to(tl.float32)
                else:
                    g = tl.load(g_ptr + grow + offs, mask=mcol, other=0.0).to(
                        tl.float32
                    )
                    res = tl.load(
                        r_ptr + xrow + offs, mask=mcol, other=0.0
                    ).to(tl.float32)
                    w = tl.load(w_ptr + offs, mask=mcol, other=0.0).to(
                        tl.float32
                    )
            else:
                mrow = row < N
                g = tl.load(
                    g_ptr + grow + offs,
                    mask=mrow if EXACT else mcol & mrow,
                    other=0.0,
                ).to(tl.float32)
                res = tl.load(
                    r_ptr + xrow + offs,
                    mask=mrow if EXACT else mcol & mrow,
                    other=0.0,
                ).to(tl.float32)
                w = tl.load(
                    w_ptr + offs, mask=None if EXACT else mcol, other=0.0
                ).to(tl.float32)
            rstd = tl.rsqrt(tl.sum(x * x, axis=0) / D + eps)

        normed = (x * rstd).to(OUT_DTYPE).to(tl.float32) * w
        e = tl.math.exp2(2.8853900817779268 * g)
        gated = (
            (1.0 - tl.math.exp2(1.0 - tl.math.log2(e + 1.0)))
            .to(OUT_DTYPE)
            .to(tl.float32)
        )
        y = res + gated * normed

        if EXACT_ROWS:
            tl.store(
                y_ptr + xrow + offs,
                y.to(OUT_DTYPE),
                mask=None if EXACT else mcol,
                cache_modifier=None if not CS else ".cs",
            )
        else:
            mrow = row < N
            tl.store(
                y_ptr + xrow + offs,
                y.to(OUT_DTYPE),
                mask=mrow if EXACT else mcol & mrow,
                cache_modifier=None if not CS else ".cs",
            )


@triton.jit
def _rmsnorm_tanh_residual_chunked_kernel(
    x_ptr,
    g_ptr,
    r_ptr,
    w_ptr,
    y_ptr,
    eps,
    D: tl.constexpr,
    CHUNK: tl.constexpr,
    GATE_REPEAT: tl.constexpr,
    I32_ADDRS: tl.constexpr,
):
    """Two-pass fallback for rows wider than the register-resident tile."""
    row = tl.program_id(0)
    if I32_ADDRS:
        xrow = row * D
    else:
        xrow = row.to(tl.int64) * D

    NUM_CHUNKS: tl.constexpr = (D + CHUNK - 1) // CHUNK
    ssq = 0.0
    for i in tl.static_range(NUM_CHUNKS):
        offs = i * CHUNK + tl.arange(0, CHUNK)
        m = offs < D
        x = tl.load(x_ptr + xrow + offs, mask=m, other=0.0).to(tl.float32)
        ssq += tl.sum(x * x, axis=0)

    rstd = tl.rsqrt(ssq / D + eps)
    if I32_ADDRS:
        grow = row // GATE_REPEAT * D
    else:
        grow = row.to(tl.int64) // GATE_REPEAT * D
    for i in tl.static_range(NUM_CHUNKS):
        offs = i * CHUNK + tl.arange(0, CHUNK)
        m = offs < D
        g = tl.load(g_ptr + grow + offs, mask=m, other=0.0).to(tl.float32)
        res = tl.load(r_ptr + xrow + offs, mask=m, other=0.0).to(tl.float32)
        w = tl.load(w_ptr + offs, mask=m, other=0.0).to(tl.float32)
        x = tl.load(x_ptr + xrow + offs, mask=m, other=0.0).to(tl.float32)
        normed = (x * rstd).to(y_ptr.dtype.element_ty).to(tl.float32) * w
        e = tl.math.exp2(2.8853900817779268 * g)
        gated = (
            (1.0 - tl.math.exp2(1.0 - tl.math.log2(e + 1.0)))
            .to(y_ptr.dtype.element_ty)
            .to(tl.float32)
        )
        y = res + gated * normed
        tl.store(y_ptr + xrow + offs, y.to(y_ptr.dtype.element_ty), mask=m)


def rmsnorm_tanh_residual(x, gate, residual, weight, eps):
    dim = x.shape[-1]
    x2 = x.reshape(-1, dim)
    g2 = gate.reshape(-1, dim)
    r2 = residual.reshape(-1, dim)
    N = x2.shape[0]
    out = torch.empty_like(x, memory_format=torch.contiguous_format)
    if N == 0:
        return out

    # Row strides of the flattened views (flattening a trailing-dim slice
    # makes every view row-contiguous, so one stride drives each of them).
    x_stride = x2.stride(0)
    g_stride = g2.stride(0)
    r_stride = r2.stride(0)
    x_contig = (
        x2.stride(-1) == 1 and x_stride == dim and x2.storage_offset() == 0
    )
    g_contig = (
        g2.stride(-1) == 1 and g_stride == dim and g2.storage_offset() == 0
    )
    r_contig = (
        r2.stride(-1) == 1 and r_stride == dim and r2.storage_offset() == 0
    )

    x_flat = x2.contiguous() if not x_contig else x2
    g_flat = g2.contiguous() if not g_contig else g2
    r_flat = r2.contiguous() if not r_contig else r2

    repeat = N // g2.shape[0]
    i32 = x_flat.numel() < _MAX_I32_ELEMENTS

    if dim <= _MAX_REG_WIDTH:
        block_d = _next_pow2(dim)
        per, num_warps, ws, cs = _launch_cfg(N, dim)
        grid = (triton.cdiv(N, per),)
        _rmsnorm_tanh_residual_kernel[grid](
            x_flat,
            g_flat,
            r_flat,
            weight,
            out,
            eps,
            N,
            D=dim,
            BLOCK_D=block_d,
            GATE_REPEAT=repeat,
            PER=per,
            EXACT_ROWS=(N % per == 0),
            I32_ADDRS=i32,
            WS=ws,
            CS=cs,
            num_warps=num_warps,
            num_stages=1,
        )
    else:
        _rmsnorm_tanh_residual_chunked_kernel[(N,)](
            x_flat,
            g_flat,
            r_flat,
            weight,
            out,
            eps,
            D=dim,
            CHUNK=4096,
            GATE_REPEAT=repeat,
            I32_ADDRS=i32,
            num_warps=8,
            num_stages=1,
        )
    return out


__all__ = ["rmsnorm_tanh_residual"]
