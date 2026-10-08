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
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import triton
import triton.language as tl


@triton.jit
def _row_full(X, T, Y, INNER: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < INNER
    tau = tl.load(T + row).to(tl.float32)
    x = tl.load(X + row * INNER + cols, mask=mask, other=0.0).to(tl.float32)
    tl.store(
        Y + row * INNER + cols, (x * tau).to(Y.dtype.element_ty), mask=mask
    )


@triton.jit
def _persist_rows(
    X,
    T,
    Y,
    rows,
    INNER: tl.constexpr,
    BLOCK: tl.constexpr,
    NPROG: tl.constexpr,
):
    pid = tl.program_id(0)
    tiles_per_row = tl.cdiv(INNER, BLOCK)
    for tile in tl.range(pid, rows * tiles_per_row, NPROG, num_stages=1):
        row = tile // tiles_per_row
        cols = tile % tiles_per_row * BLOCK + tl.arange(0, BLOCK)
        mask = cols < INNER
        tau = tl.load(T + row).to(tl.float32)
        x = tl.load(X + row * INNER + cols, mask=mask, other=0.0).to(
            tl.float32
        )
        tl.store(
            Y + row * INNER + cols, (x * tau).to(Y.dtype.element_ty), mask=mask
        )


@triton.jit
def _vector_tiles(
    X,
    T,
    Y,
    ROWS: tl.constexpr,
    INNER: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    PROGRAMS: tl.constexpr,
):
    columns = tl.cdiv(INNER, BLOCK_N)
    tiles = tl.cdiv(ROWS, BLOCK_M) * columns
    for tile in tl.range(tl.program_id(0), tiles, PROGRAMS, num_stages=1):
        rows = tile // columns * BLOCK_M + tl.arange(0, BLOCK_M)
        cols = tile % columns * BLOCK_N + tl.arange(0, BLOCK_N)
        mask = (rows[:, None] < ROWS) & (cols[None, :] < INNER)
        scale = tl.load(T + rows, mask=rows < ROWS, other=0).to(tl.float32)
        values = tl.load(
            X + rows[:, None] * INNER + cols[None, :], mask=mask, other=0
        ).to(tl.float32)
        tl.store(
            Y + rows[:, None] * INNER + cols[None, :],
            (values * scale[:, None]).to(Y.dtype.element_ty),
            mask=mask,
        )


def _block(n):
    size = 1
    while size < n:
        size *= 2
    return size


def _entry(x, tau, out, rows, inner):
    block = 1024 if inner > 8192 else _block(inner)
    if rows >= 128:
        block_n = min(2048, _block(inner))
        block_m = max(1, min(8, 16384 // block_n))
        programs = min(
            40, triton.cdiv(rows, block_m) * triton.cdiv(inner, block_n)
        )
        grid = (programs, 1, 1)
        _vector_tiles[grid](
            x,
            tau,
            out,
            ROWS=rows,
            INNER=inner,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            PROGRAMS=programs,
            num_warps=8,
            num_stages=1,
        )
    elif inner > 8192:
        programs = 40
        grid = (programs, 1, 1)
        _persist_rows[grid](
            x,
            tau,
            out,
            rows,
            INNER=inner,
            BLOCK=block,
            NPROG=programs,
            num_warps=8,
            num_stages=1,
        )
    else:
        warps = 8 if block >= 256 else 4 if block >= 64 else 2
        grid = (rows, 1, 1)
        _row_full[grid](
            x,
            tau,
            out,
            INNER=inner,
            BLOCK=block,
            num_warps=warps,
            num_stages=1,
        )


def log_scaling_tau(x, tau):
    source = x.contiguous()
    out = torch.empty_like(source)
    n = source.numel()
    if n == 0:
        return out
    rows = source.shape[0]
    inner = n // rows
    scale = tau.contiguous()
    _entry(source, scale, out, rows, inner)
    return out


__all__ = ["log_scaling_tau"]
