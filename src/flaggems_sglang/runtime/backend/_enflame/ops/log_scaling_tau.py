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

MAX_BLOCK = 32768
MAX_PROGRAMS = 12


@triton.jit
def _row_block(X, T, Y, INNER: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    tau = tl.load(T + row).to(tl.float32)
    x_block = tl.make_block_ptr(
        X + row * INNER,
        shape=(INNER,),
        strides=(1,),
        offsets=(0,),
        block_shape=(BLOCK,),
        order=(0,),
    )
    y_block = tl.make_block_ptr(
        Y + row * INNER,
        shape=(INNER,),
        strides=(1,),
        offsets=(0,),
        block_shape=(BLOCK,),
        order=(0,),
    )
    x = tl.load(x_block, boundary_check=(0,)).to(tl.float32)
    tl.store(y_block, (x * tau).to(Y.dtype.element_ty), boundary_check=(0,))


@triton.jit
def _multirow(
    X,
    T,
    Y,
    rows,
    INNER: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    num_n = tl.cdiv(INNER, BLOCK_N)
    tile_m = pid // num_n
    tile_n = pid - tile_m * num_n
    row = tile_m * BLOCK_M + tl.arange(0, BLOCK_M)
    col = tile_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (row[:, None] < rows) & (col[None, :] < INNER)
    tau = tl.load(T + row, mask=row < rows, other=0.0).to(tl.float32)
    x = tl.load(
        X + row[:, None] * INNER + col[None, :], mask=mask, other=0.0
    ).to(tl.float32)
    tl.store(
        Y + row[:, None] * INNER + col[None, :],
        (x * tau[:, None]).to(Y.dtype.element_ty),
        mask=mask,
    )


@triton.jit
def _persist(X, T, Y, n_elements, INNER, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    offs0 = tl.arange(0, BLOCK)
    nblocks = tl.cdiv(n_elements, BLOCK)
    for bid in tl.range(pid, nblocks, nprog, num_stages=1):
        offs = bid * BLOCK + offs0
        mask = offs < n_elements
        x = tl.load(X + offs, mask=mask, other=0.0).to(tl.float32)
        tau = tl.load(T + offs // INNER, mask=mask, other=0.0).to(tl.float32)
        tl.store(Y + offs, (x * tau).to(Y.dtype.element_ty), mask=mask)


def _entry(source, scale, out, rows, inner):
    n = rows * inner
    if inner <= MAX_BLOCK:
        block = triton.next_power_of_2(max(1, inner))
        if block > MAX_BLOCK:
            block = MAX_BLOCK
        if block >= inner:
            if rows >= 128:
                warps = 1 if rows >= 4096 else 2 if rows >= 2048 else 4
                target_rows = 64 if rows >= 2048 else 32
                element_limit = (
                    131072 if source.dtype == torch.float32 else 262144
                )
                if warps == 1:
                    element_limit //= 2
                block_m = min(target_rows, max(1, element_limit // block))
                grid = (triton.cdiv(rows, block_m), 1, 1)
                _multirow[grid](
                    source,
                    scale,
                    out,
                    rows,
                    INNER=inner,
                    BLOCK_M=block_m,
                    BLOCK_N=block,
                    num_warps=warps,
                    num_stages=1,
                )
                return
            grid = (rows, 1, 1)
            _row_block[grid](
                source,
                scale,
                out,
                INNER=inner,
                BLOCK=block,
                num_warps=1 if block <= 256 else 2,
                num_stages=1,
            )
            return
    block = MAX_BLOCK
    nblocks = triton.cdiv(n, block)
    programs = nblocks if nblocks <= MAX_PROGRAMS else MAX_PROGRAMS
    grid = (programs, 1, 1)
    _persist[grid](
        source,
        scale,
        out,
        n,
        inner,
        BLOCK=block,
        num_warps=2,
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
