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

# Largest 1-D grid every backend accepts (Ascend rtKernelLaunch coreDim).
_MAX_GRID = 65535


@triton.jit
def _qk_rmsnorm_h128(
    x_ptr,
    w_ptr,
    out_ptr,
    BLOCK_S: tl.constexpr,
    EPS: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = pid * BLOCK_S + tl.arange(0, BLOCK_S)
    cols = tl.arange(0, 128)
    offs = rows[:, None] * 128 + cols[None, :]

    x = tl.load(x_ptr + offs).to(tl.float32)
    rstd = tl.rsqrt(tl.sum(x * x, axis=1) / 128 + EPS)
    w = tl.load(w_ptr + cols).to(tl.float32)

    tl.store(
        out_ptr + offs,
        ((x * rstd[:, None]) * w[None, :]).to(out_ptr.dtype.element_ty),
    )


@triton.jit
def _qk_rmsnorm_h128_masked(
    x_ptr,
    w_ptr,
    out_ptr,
    n_rows,
    eps,
    BLOCK_S: tl.constexpr,
    TILES: tl.constexpr,
):
    pid = tl.program_id(0)
    cols = tl.arange(0, 128)
    w = tl.load(w_ptr + cols).to(tl.float32)

    for i in tl.static_range(TILES):
        rows = (pid * TILES + i) * BLOCK_S + tl.arange(0, BLOCK_S)
        row_ok = rows < n_rows
        offs = rows[:, None] * 128 + cols[None, :]

        x = tl.load(x_ptr + offs, mask=row_ok[:, None], other=0.0).to(
            tl.float32
        )
        rstd = tl.rsqrt(tl.sum(x * x, axis=1) / 128 + eps)
        y = (x * rstd[:, None]) * w[None, :]
        tl.store(
            out_ptr + offs,
            y.to(out_ptr.dtype.element_ty),
            mask=row_ok[:, None],
        )


@triton.jit
def _qk_rmsnorm_flat(
    x_ptr,
    w_ptr,
    out_ptr,
    D: tl.constexpr,
    BLOCK_S: tl.constexpr,
    EPS: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = pid * BLOCK_S + tl.arange(0, BLOCK_S)
    cols = tl.arange(0, D)
    offs = rows[:, None] * D + cols[None, :]

    x = tl.load(x_ptr + offs).to(tl.float32)
    rstd = tl.rsqrt(tl.sum(x * x, axis=1) / D + EPS)
    w = tl.load(w_ptr + cols).to(tl.float32)

    tl.store(
        out_ptr + offs,
        ((x * rstd[:, None]) * w[None, :]).to(out_ptr.dtype.element_ty),
    )


@triton.jit
def _qk_rmsnorm_flat_masked(
    x_ptr,
    w_ptr,
    out_ptr,
    n_rows,
    eps,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_S: tl.constexpr,
    TILES: tl.constexpr,
):
    pid = tl.program_id(0)
    cols = tl.arange(0, BLOCK_D)
    col_ok = cols < D
    w = tl.load(w_ptr + cols, mask=col_ok, other=0.0).to(tl.float32)

    for i in tl.static_range(TILES):
        rows = (pid * TILES + i) * BLOCK_S + tl.arange(0, BLOCK_S)
        row_ok = rows < n_rows
        mask = row_ok[:, None] & col_ok[None, :]
        offs = rows[:, None] * D + cols[None, :]

        x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        rstd = tl.rsqrt(tl.sum(x * x, axis=1) / D + eps)

        y = (x * rstd[:, None]) * w[None, :]
        tl.store(out_ptr + offs, y.to(out_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _qk_rmsnorm_strided(
    x_ptr,
    w_ptr,
    out_ptr,
    S,
    sb,
    ss,
    n_rows,
    eps,
    H: tl.constexpr,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = pid * BLOCK_R + tl.arange(0, BLOCK_R)
    cols = tl.arange(0, BLOCK_D)
    row_ok = rows < n_rows
    col_ok = cols < D
    mask = row_ok[:, None] & col_ok[None, :]

    t = rows // H
    tok_off = (t // S) * sb + (t % S) * ss
    in_offs = (tok_off + (rows % H) * D)[:, None] + cols[None, :]

    x = tl.load(x_ptr + in_offs, mask=mask, other=0.0).to(tl.float32)
    rstd = tl.rsqrt(tl.sum(x * x, axis=1) / D + eps)
    w = tl.load(w_ptr + cols, mask=col_ok, other=0.0).to(tl.float32)

    y = (x * rstd[:, None]) * w[None, :]
    # Output is always fresh and contiguous: D-wide rows in (token, head)
    # order.
    out_offs = (rows * D)[:, None] + cols[None, :]
    tl.store(out_ptr + out_offs, y.to(out_ptr.dtype.element_ty), mask=mask)


def _next_pow2(n):
    return 1 << (n - 1).bit_length() if n > 1 else 1


def _h128_config(n_rows, itemsize):
    if n_rows >= 131072:
        return (64, 8) if itemsize > 2 else (128, 8)
    if n_rows >= 8192:
        return 32, 2
    return 8, 4


def _flat_config(n_rows, itemsize):
    if n_rows >= 8192:
        # fp32 tiles hold twice the bytes; keep the on-chip buffer comfortable.
        return (64, 8) if itemsize > 2 else (128, 8)
    return 16, 2


def zimage_qk_rmsnorm(x, weight, eps):
    shape = x.shape
    dim = shape[-1]
    dtype = x.dtype

    out = (
        torch.empty_like(x)
        if x.is_contiguous()
        else torch.empty(shape, dtype=dtype, device=x.device)
    )

    n_rows = x.numel() // dim if dim > 0 else 0
    if n_rows == 0 or dim == 0:
        return out

    w = weight if weight.is_contiguous() else weight.contiguous()

    itemsize = x.element_size()
    heads = shape[-2]

    if x.is_contiguous():
        if dim == 128:
            # Dedicated power-of-two fast path: no column masks anywhere.
            block_s, num_warps = _h128_config(n_rows, itemsize)
            if n_rows % block_s == 0:
                grid = n_rows // block_s
                if grid <= _MAX_GRID:
                    _qk_rmsnorm_h128[(grid,)](
                        x,
                        w,
                        out,
                        block_s,
                        eps,
                        num_warps=num_warps,
                    )
                    return out
            # Row count not tile-aligned (e.g. S = 4360): row mask only,
            # tiles inside the kernel keep the grid within launch limits.
            tiles_total = (n_rows + block_s - 1) // block_s
            tiles = (tiles_total + _MAX_GRID - 1) // _MAX_GRID
            grid = (tiles_total + tiles - 1) // tiles
            _qk_rmsnorm_h128_masked[grid](
                x,
                w,
                out,
                n_rows,
                eps,
                block_s,
                tiles,
                num_warps=num_warps,
            )
            return out
        block_d = _next_pow2(dim)
        block_s, num_warps = _flat_config(n_rows, itemsize)
        if dim == block_d and n_rows % block_s == 0:
            grid = n_rows // block_s
            if grid <= _MAX_GRID:
                _qk_rmsnorm_flat[(grid,)](
                    x,
                    w,
                    out,
                    dim,
                    block_s,
                    eps,
                    num_warps=num_warps,
                )
                return out
        # Ragged tail (dim not a power of two or rows not tile-aligned), or
        # more tiles than one program apiece: loop tiles inside the kernel.
        tiles_total = (n_rows + block_s - 1) // block_s
        tiles = (tiles_total + _MAX_GRID - 1) // _MAX_GRID
        grid = (tiles_total + tiles - 1) // tiles
        _qk_rmsnorm_flat_masked[grid](
            x,
            w,
            out,
            n_rows,
            eps,
            dim,
            block_d,
            block_s,
            tiles,
            num_warps=num_warps,
        )
        return out

    # Fused-QKV style slice: last dim contiguous, heads packed. Read in place.
    x_strides = x.stride()
    if x_strides[-1] == 1 and x_strides[-2] == dim:
        n3 = len(shape)
        sb = x_strides[0] if n3 > 3 else heads * dim
        ss = x_strides[1] if n3 > 2 else heads * dim
        if n3 == 3:
            sb = ss  # [S, H, D]: batch and sequence collapse into tokens
        block_d = _next_pow2(dim)
        block_r, num_warps = _flat_config(n_rows, itemsize)
        grid = ((n_rows + block_r - 1) // block_r,)
        if grid[0] <= _MAX_GRID:
            _qk_rmsnorm_strided[grid](
                x,
                w,
                out,
                shape[1] if n3 > 2 else 1,
                sb,
                ss,
                n_rows,
                eps,
                heads,
                dim,
                block_d,
                block_r,
                num_warps=num_warps,
            )
            return out

    xc = x.contiguous()
    block_d = _next_pow2(dim)
    block_s, num_warps = _flat_config(n_rows, itemsize)
    tiles_total = (n_rows + block_s - 1) // block_s
    tiles = (tiles_total + _MAX_GRID - 1) // _MAX_GRID
    grid = (tiles_total + tiles - 1) // tiles
    _qk_rmsnorm_flat_masked[grid](
        xc,
        w,
        out,
        n_rows,
        eps,
        dim,
        block_d,
        block_s,
        tiles,
        num_warps=num_warps,
    )
    return out


__all__ = ["zimage_qk_rmsnorm"]
