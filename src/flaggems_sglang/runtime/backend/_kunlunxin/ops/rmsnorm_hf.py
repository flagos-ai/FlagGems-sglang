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

"""Operator: activation_norm/rmsnorm_hf (pure Triton)."""

import torch
import triton
import triton.language as tl


@triton.jit
def _rmsnorm_hf_tile(
    x_ptr,
    w_ptr,
    y_ptr,
    n_rows,
    n_cols,
    eps,
    stride_xm,
    stride_ym,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.arange(0, BLOCK_N)
    mask = (rows[:, None] < n_rows) & (cols[None, :] < n_cols)
    x = tl.load(
        x_ptr + rows[:, None] * stride_xm + cols[None, :], mask=mask, other=0.0
    ).to(tl.float32)
    ss = tl.sum(x * x, axis=1)
    rrms = tl.math.rsqrt(ss / n_cols + eps)
    y = (x * rrms[:, None]).to(y_ptr.dtype.element_ty)
    w = tl.load(w_ptr + cols, mask=cols < n_cols, other=0.0)
    tl.store(
        y_ptr + rows[:, None] * stride_ym + cols[None, :],
        w[None, :] * y,
        mask=mask,
    )


@triton.jit
def _rmsnorm_hf_loop(
    x_ptr,
    w_ptr,
    y_ptr,
    n_cols,
    eps,
    stride_xm,
    stride_ym,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    x_row = x_ptr + row * stride_xm
    ss = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for start in range(0, n_cols, BLOCK_N):
        cols = start + tl.arange(0, BLOCK_N)
        x = tl.load(x_row + cols, mask=cols < n_cols, other=0.0).to(tl.float32)
        ss += x * x
    rrms = tl.math.rsqrt(tl.sum(ss, axis=0) / n_cols + eps)
    y_row = y_ptr + row * stride_ym
    for start in range(0, n_cols, BLOCK_N):
        cols = start + tl.arange(0, BLOCK_N)
        cmask = cols < n_cols
        x = tl.load(x_row + cols, mask=cmask, other=0.0).to(tl.float32)
        y = (x * rrms).to(y_ptr.dtype.element_ty)
        w = tl.load(w_ptr + cols, mask=cmask, other=0.0)
        tl.store(y_row + cols, w * y, mask=cmask)


def rmsnorm_hf(input, weight, eps):
    x = input.contiguous()
    w = weight.contiguous()
    n_rows = x.numel() // x.shape[-1]
    n_cols = x.shape[-1]
    out = torch.empty_like(x)

    # Per-shape launch heuristic (local variables only; no cross-call cache).
    block_n = triton.next_power_of_2(n_cols)
    max_tile_cols = 8192
    if block_n <= 256:
        # Narrow rows: pack many rows per program, 16 or 32 of them.
        block_m = 16 if n_rows <= 256 else 32
        num_warps = 8
        grid = (triton.cdiv(n_rows, block_m),)
        _rmsnorm_hf_tile[grid](
            x,
            w,
            out,
            n_rows,
            n_cols,
            eps,
            x.stride(0),
            out.stride(0),
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            num_warps=num_warps,
        )
    elif n_cols <= max_tile_cols and n_rows > 8:
        # Wide rows with enough batch: one tile covers whole rows. Keep a
        # handful of rows per program so the weight vector is loaded once.
        if n_rows <= 16:
            block_m = 1
        elif n_rows <= 128:
            block_m = 2
        elif n_cols % 1024 == 0:
            block_m = 4
        else:
            block_m = 2
        grid = (triton.cdiv(n_rows, block_m),)
        _rmsnorm_hf_tile[grid](
            x,
            w,
            out,
            n_rows,
            n_cols,
            eps,
            x.stride(0),
            out.stride(0),
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            num_warps=8,
        )
    else:
        # Tiny batch (or hidden wider than a single tile): one row per
        # program, column-blocked two-pass, to keep enough programs busy.
        loop_block = min(4096, block_n)
        grid = (n_rows,)
        _rmsnorm_hf_loop[grid](
            x,
            w,
            out,
            n_cols,
            eps,
            x.stride(0),
            out.stride(0),
            BLOCK_N=loop_block,
            num_warps=8,
        )
    return out


__all__ = ["rmsnorm_hf"]
