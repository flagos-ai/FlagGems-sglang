# Copyright 2026, The FlagOS Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
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
ROW_GROUP = 8
MAX_PROGRAMS_SCALED = 48


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
def _row_scaled(
    X,
    T,
    Y,
    rows,
    INNER: tl.constexpr,
    BLOCK_N: tl.constexpr,
    EVEN_N: tl.constexpr,
    U: tl.constexpr,
):
    # Persistent grid-stride over groups of U rows. Per group, the U scalar
    # tau loads and U row loads are emitted first, then the U stores, so the
    # row loads share one issue window. The final group is anchored at
    # rows - U so every row is covered exactly or idempotently: no row mask,
    # no out-of-bounds access (dispatch guarantees rows >= U).
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    col = tl.arange(0, BLOCK_N)
    ngroups = tl.cdiv(rows, U)
    anchor = rows - U
    for g in tl.range(pid, ngroups, nprog):
        r0 = tl.minimum(g * U, anchor)
        if U == 8 and EVEN_N:
            t0 = tl.load(T + r0 + 0)
            t1 = tl.load(T + r0 + 1)
            t2 = tl.load(T + r0 + 2)
            t3 = tl.load(T + r0 + 3)
            t4 = tl.load(T + r0 + 4)
            t5 = tl.load(T + r0 + 5)
            t6 = tl.load(T + r0 + 6)
            t7 = tl.load(T + r0 + 7)
            b0 = (r0 + 0) * INNER + col
            b1 = (r0 + 1) * INNER + col
            b2 = (r0 + 2) * INNER + col
            b3 = (r0 + 3) * INNER + col
            b4 = (r0 + 4) * INNER + col
            b5 = (r0 + 5) * INNER + col
            b6 = (r0 + 6) * INNER + col
            b7 = (r0 + 7) * INNER + col
            x0 = tl.load(X + b0)
            x1 = tl.load(X + b1)
            x2 = tl.load(X + b2)
            x3 = tl.load(X + b3)
            x4 = tl.load(X + b4)
            x5 = tl.load(X + b5)
            x6 = tl.load(X + b6)
            x7 = tl.load(X + b7)
            tl.store(Y + b0, (x0 * t0).to(Y.dtype.element_ty))
            tl.store(Y + b1, (x1 * t1).to(Y.dtype.element_ty))
            tl.store(Y + b2, (x2 * t2).to(Y.dtype.element_ty))
            tl.store(Y + b3, (x3 * t3).to(Y.dtype.element_ty))
            tl.store(Y + b4, (x4 * t4).to(Y.dtype.element_ty))
            tl.store(Y + b5, (x5 * t5).to(Y.dtype.element_ty))
            tl.store(Y + b6, (x6 * t6).to(Y.dtype.element_ty))
            tl.store(Y + b7, (x7 * t7).to(Y.dtype.element_ty))
        else:
            for j in tl.static_range(0, U):
                r = r0 + j
                tau = tl.load(T + r)
                base = r * INNER + col
                if EVEN_N:
                    x = tl.load(X + base)
                    tl.store(
                        Y + base, (x * tau).to(Y.dtype.element_ty)
                    )
                else:
                    m = col < INNER
                    x = tl.load(X + base, mask=m, other=0.0)
                    tl.store(
                        Y + base, (x * tau).to(Y.dtype.element_ty), mask=m
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
        block_n = triton.next_power_of_2(max(1, inner))
        itemsize = source.element_size()
        # Per-program live set ~= 2 * U * BLOCK_N * itemsize bytes
        # (x tiles + y tiles); keep it <= 128KB to stay inside the
        # 448KB dual-program DSM line.
        budget = 16384 if itemsize > 2 else 32768
        u = ROW_GROUP
        while u > 1 and u * block_n > budget:
            u //= 2
        if rows >= u:
            grid = (min(MAX_PROGRAMS_SCALED, triton.cdiv(rows, u)), 1, 1)
            # 2026-11-12 GCU400 实测：U 批发射臂 block_n∈{1024,2048} 时 nw4 比
            # nw2 快 5~7%（8192 行 fp16 -6.8% / bf16 -7.0% / 1024 块 -5.4%）；
            # block_n<=512 反向（+14~19%），fp32 仅弱信号 -1.9% 故不并入。
            num_warps = max(2, min(8, block_n // 1024))
            if itemsize <= 2 and block_n >= 1024:
                num_warps = max(4, num_warps)
            _row_scaled[grid](
                source,
                scale,
                out,
                rows,
                INNER=inner,
                BLOCK_N=block_n,
                EVEN_N=(block_n == inner),
                U=u,
                num_warps=num_warps,
                num_stages=1,
            )
            return
        grid = (rows, 1, 1)
        _row_block[grid](
            source,
            scale,
            out,
            INNER=inner,
            BLOCK=block_n,
            num_warps=max(1, min(8, block_n * itemsize // 8192)),
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
        num_warps=max(2, min(8, block * source.element_size() // 8192)),
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
