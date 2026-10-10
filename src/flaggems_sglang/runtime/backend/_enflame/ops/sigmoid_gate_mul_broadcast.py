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

import torch
import triton
import triton.language as tl

_TILE_MAX_D = 8192


@triton.jit
def _sgmb_row1(x_ptr, g_ptr, out_ptr, BLOCK_D: tl.constexpr):
    cols = tl.arange(0, BLOCK_D)
    x = tl.load(x_ptr + cols)
    sig = tl.sigmoid(tl.load(g_ptr).to(tl.float32)).to(x_ptr.dtype.element_ty)
    tl.store(out_ptr + cols, x * sig)


@triton.jit
def _sgmb_row1_masked(x_ptr, g_ptr, out_ptr, D, BLOCK_D: tl.constexpr):
    cols = tl.arange(0, BLOCK_D)
    mask = cols < D
    x = tl.load(x_ptr + cols, mask=mask, other=0.0)
    sig = tl.sigmoid(tl.load(g_ptr).to(tl.float32)).to(x_ptr.dtype.element_ty)
    tl.store(out_ptr + cols, x * sig, mask=mask)


@triton.jit
def _sgmb_tile_exact(
    x_ptr, g_ptr, out_ptr, D, BLOCK_M: tl.constexpr, BLOCK_D: tl.constexpr
):
    pid = tl.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    g = tl.load(g_ptr + rows).to(tl.float32)
    sig = tl.sigmoid(g).to(x_ptr.dtype.element_ty)
    offs = rows[:, None] * BLOCK_D + tl.arange(0, BLOCK_D)[None, :]
    x = tl.load(x_ptr + offs)
    y = x * sig[:, None]
    tl.store(out_ptr + offs, y)


@triton.jit
def _sgmb_tile_exact_i64(
    x_ptr, g_ptr, out_ptr, D, BLOCK_M: tl.constexpr, BLOCK_D: tl.constexpr
):
    pid = tl.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    g = tl.load(g_ptr + rows).to(tl.float32)
    sig = tl.sigmoid(g).to(x_ptr.dtype.element_ty)
    offs = (
        rows[:, None].to(tl.int64) * BLOCK_D + tl.arange(0, BLOCK_D)[None, :]
    )
    x = tl.load(x_ptr + offs)
    y = x * sig[:, None]
    tl.store(out_ptr + offs, y)


@triton.jit
def _sgmb_tile_masked(
    x_ptr, g_ptr, out_ptr, N, D, BLOCK_M: tl.constexpr, BLOCK_D: tl.constexpr
):
    pid = tl.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    rmask = rows < N
    g = tl.load(g_ptr + rows, mask=rmask, other=0.0).to(tl.float32)
    sig = tl.sigmoid(g).to(x_ptr.dtype.element_ty)
    cols = tl.arange(0, BLOCK_D)
    mask = rmask[:, None] & (cols < D)[None, :]
    offs = rows[:, None] * D + cols[None, :]
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    y = x * sig[:, None]
    tl.store(out_ptr + offs, y, mask=mask)


@triton.jit
def _sgmb_tile_masked_i64(
    x_ptr, g_ptr, out_ptr, N, D, BLOCK_M: tl.constexpr, BLOCK_D: tl.constexpr
):
    pid = tl.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    rmask = rows < N
    g = tl.load(g_ptr + rows, mask=rmask, other=0.0).to(tl.float32)
    sig = tl.sigmoid(g).to(x_ptr.dtype.element_ty)
    cols = tl.arange(0, BLOCK_D)
    mask = rmask[:, None] & (cols < D)[None, :]
    offs = rows[:, None].to(tl.int64) * D + cols[None, :]
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    y = x * sig[:, None]
    tl.store(out_ptr + offs, y, mask=mask)


@triton.jit
def _sgmb_loop(
    x_ptr, g_ptr, out_ptr, N, D, BLOCK_M: tl.constexpr, BLOCK_H: tl.constexpr
):
    pid = tl.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    rmask = rows < N
    g = tl.load(g_ptr + rows, mask=rmask, other=0.0).to(tl.float32)
    sig = tl.sigmoid(g).to(x_ptr.dtype.element_ty)
    for h0 in range(0, D, BLOCK_H):
        cols = h0 + tl.arange(0, BLOCK_H)
        mask = rmask[:, None] & (cols < D)[None, :]
        offs = rows[:, None] * D + cols[None, :]
        x = tl.load(x_ptr + offs, mask=mask, other=0.0)
        y = x * sig[:, None]
        tl.store(out_ptr + offs, y, mask=mask)


def _next_pow2(v):
    return 1 << (max(v, 1) - 1).bit_length()


def _pick_tile(N, D):
    if N <= 1:
        return 1, (1 if D <= 2048 else 4)
    if N <= 8:
        return (8, 1) if D <= 2048 else (8, 8)
    if N <= 64:
        if D <= 2048:
            return 64, 4
        if D <= 4096:
            return 64, 8
        return 32, 8
    if N <= 512:
        if D <= 2048:
            return 128, 4
        if D <= 4096:
            return 64, 2
        return 16, 2
    if D <= 2048:
        return 128, 2
    if D <= 4096:
        return 64, 2
    return 16, 2


def sigmoid_gate_mul_broadcast(x, gate):
    xc = x if x.is_contiguous() else x.contiguous()
    gc = gate if gate.is_contiguous() else gate.contiguous()
    out = torch.empty_like(xc)
    if xc.numel() == 0:
        return out

    D = xc.shape[-1] if xc.dim() >= 1 else 1
    N = xc.numel() // max(D, 1)

    if D <= _TILE_MAX_D:
        block_d = _next_pow2(D)
        if N == 1:
            if block_d == D:
                _sgmb_row1[(1,)](xc, gc, out, BLOCK_D=block_d, num_warps=2)
            else:
                _sgmb_row1_masked[(1,)](
                    xc, gc, out, D, BLOCK_D=block_d, num_warps=2
                )
            return out
        block_m, num_warps = _pick_tile(N, D)
        if block_d == D and N % block_m == 0:
            if xc.numel() < 2**31:
                _sgmb_tile_exact[(N // block_m,)](
                    xc,
                    gc,
                    out,
                    D,
                    BLOCK_M=block_m,
                    BLOCK_D=block_d,
                    num_warps=num_warps,
                )
            else:
                _sgmb_loop[(triton.cdiv(N, block_m),)](
                    xc,
                    gc,
                    out,
                    N,
                    D,
                    BLOCK_M=block_m,
                    BLOCK_H=block_d,
                    num_warps=num_warps,
                )
        elif xc.numel() < 2**31:
            _sgmb_tile_masked[(triton.cdiv(N, block_m),)](
                xc,
                gc,
                out,
                N,
                D,
                BLOCK_M=block_m,
                BLOCK_D=block_d,
                num_warps=num_warps,
            )
        else:
            _sgmb_tile_masked_i64[(triton.cdiv(N, block_m),)](
                xc,
                gc,
                out,
                N,
                D,
                BLOCK_M=block_m,
                BLOCK_D=block_d,
                num_warps=num_warps,
            )
    else:
        block_m, num_warps = _pick_tile(N, min(D, 8192))
        _sgmb_loop[(triton.cdiv(N, block_m),)](
            xc,
            gc,
            out,
            N,
            D,
            BLOCK_M=block_m,
            BLOCK_H=4096,
            num_warps=num_warps,
        )
    return out


__all__ = ["sigmoid_gate_mul_broadcast"]
