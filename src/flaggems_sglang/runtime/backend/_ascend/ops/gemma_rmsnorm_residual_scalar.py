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

"""Operator: activation_norm/gemma_rmsnorm_residual_scalar — portable."""

import triton
import triton.language as tl

# Largest 1-D grid every backend accepts (Ascend rtKernelLaunch coreDim).
_MAX_GRID = 65535

_TILE_ELEMS = 8192

# Rows packed into one program, capped so the row-strip width never dwarfs N
# and the budget above still bounds ROWS * BLOCK_N.
_MAX_ROWS = 4

# Tile-loop length bounds.  The loop is what hoists the weight load out of the
# per-row work; more tiles help until the grid thins out.
_MAX_TILES = 8
_MIN_PROGRAMS = 256

# Tiny grids are launch-latency-bound; ``num_warps`` measured flat elsewhere.
_SMALL_GRID = 8


@triton.jit
def _gemma_rmsnorm_residual_scalar_dense_kernel(
    x_ptr,
    r_ptr,
    w_ptr,
    s_ptr,
    out_ptr,
    eps,
    N: tl.constexpr,
    ROWS: tl.constexpr,
    TILES: tl.constexpr,
):
    pid = tl.program_id(0)
    cols = tl.arange(0, N)
    w = tl.load(w_ptr + cols).to(tl.float32)
    scalar = tl.load(s_ptr).to(tl.float32)
    base = pid * (ROWS * TILES)
    for i in tl.static_range(TILES):
        rows = base + i * ROWS + tl.arange(0, ROWS)
        off = rows[:, None] * N + cols[None, :]

        x = tl.load(x_ptr + off).to(tl.float32)
        inv = tl.rsqrt(tl.sum(x * x, axis=1) / N + eps)
        res = tl.load(r_ptr + off).to(tl.float32)

        y = (x * inv[:, None] * w[None, :] + res) * scalar
        # Stores cast fp32 -> the output pointer's element type implicitly.
        tl.store(out_ptr + off, y)


@triton.jit
def _gemma_rmsnorm_residual_scalar_masked_kernel(
    x_ptr,
    r_ptr,
    w_ptr,
    s_ptr,
    out_ptr,
    sx,
    sxr,
    eps,
    n_rows,
    N: tl.constexpr,
    BLOCK_N: tl.constexpr,
    ROWS: tl.constexpr,
    TILES: tl.constexpr,
):
    pid = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    col_mask = cols < N
    # ``w`` lives in the (masked) weight vector; load it once per program.
    w = tl.load(w_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)
    scalar = tl.load(s_ptr).to(tl.float32)
    for i in tl.static_range(TILES):
        rows = (pid * TILES + i) * ROWS + tl.arange(0, ROWS)
        mask = (rows < n_rows)[:, None] & col_mask[None, :]
        off = rows[:, None] * sx + cols[None, :]

        x = tl.load(x_ptr + off, mask=mask, other=0.0).to(tl.float32)
        inv = tl.rsqrt(tl.sum(x * x, axis=1) / N + eps)
        res = tl.load(
            r_ptr + rows[:, None] * sxr + cols[None, :], mask=mask, other=0.0
        ).to(tl.float32)
        y = (x * inv[:, None] * w[None, :] + res) * scalar
        tl.store(out_ptr + rows[:, None] * N + cols[None, :], y, mask=mask)


def _dense_geometry(m, n):
    """(ROWS, TILES) for the mask-free path, or None if it cannot be."""
    if n > _TILE_ELEMS:
        return None
    rows = _MAX_ROWS if n * _MAX_ROWS <= 2 * _TILE_ELEMS else 1
    while rows > 1 and m % rows:
        rows >>= 1
    tiles = _MAX_TILES
    while tiles > 1:
        if m % (rows * tiles) == 0 and m // (rows * tiles) >= _MIN_PROGRAMS:
            break
        tiles >>= 1
    return rows, tiles


def gemma_rmsnorm_residual_scalar(x, weight, residual, scalar, eps):
    n = x.shape[-1]
    m = x.numel() // n
    # x.new_empty skips the torch factory dispatch (dtype/device resolution)
    # that torch.empty pays on this backend; the buffer is fresh every call.
    out = x.new_empty(x.shape)
    if m == 0:
        return out

    if x.stride(-1) != 1:
        x = x.contiguous()
    if residual.stride(-1) != 1:
        residual = residual.contiguous()
    if x.dim() == 2:
        sx = x.stride(0)
    elif x.is_contiguous():
        sx = n
    else:
        x = x.contiguous()
        sx = n
    sxr = (
        residual.stride(0)
        if residual.dim() == 2
        else (
            n if residual.is_contiguous() else residual.contiguous().stride(0)
        )
    )

    if sx == n and sxr == n and (n & (n - 1)) == 0:
        geom = _dense_geometry(m, n)
        if geom is not None:
            rows, tiles = geom
            grid = m // (rows * tiles)
            if grid <= _MAX_GRID:
                _gemma_rmsnorm_residual_scalar_dense_kernel[(grid,)](
                    x,
                    residual,
                    weight,
                    scalar,
                    out,
                    eps,
                    n,
                    rows,
                    tiles,
                    num_warps=1 if grid <= _SMALL_GRID else 4,
                )
                return out

    block_n = 1 << (n - 1).bit_length() if n > 1 else 1
    rows = min(_MAX_ROWS, max(1, _TILE_ELEMS // block_n), max(1, m))
    tiles_total = (m + rows - 1) // rows
    tiles = (tiles_total + _MAX_GRID - 1) // _MAX_GRID
    grid = ((tiles_total + tiles - 1) // tiles,)
    _gemma_rmsnorm_residual_scalar_masked_kernel[grid](
        x,
        residual,
        weight,
        scalar,
        out,
        sx,
        sxr,
        eps,
        m,
        n,
        block_n,
        rows,
        tiles,
        num_warps=1 if grid[0] <= _SMALL_GRID else 4,
    )
    return out


__all__ = ["gemma_rmsnorm_residual_scalar"]
