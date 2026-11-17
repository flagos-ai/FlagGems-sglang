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

_FP8_DTYPE = torch.float8_e4m3fn
_RECIP_448 = tl.constexpr(0.0022321429569274187)


@triton.jit
def _scale_from_bits(mm, eps):
    ax = mm.to(tl.float32, bitcast=True)
    return tl.where(
        mm > 0x7F800000, float("nan"), tl.maximum(ax, eps) * _RECIP_448
    )


@triton.jit
def _f8(v):
    u = v.to(tl.int32, bitcast=True)
    a = u & 0x7FFFFFFF
    nb = ((a + 0x7FFFF + ((a >> 20) & 1)) >> 20) - 960
    nb = tl.minimum(nb, 126)
    ns = (a.to(tl.float32, bitcast=True) + 16384.0).to(
        tl.int32, bitcast=True
    ) & 0x7FFFFF
    b = tl.where(a < 0x3C800000, ns, nb)
    b = tl.where(a > 0x7F800000, 127, b)
    s = (u >> 24) & 0x80
    return (b | s).to(tl.int8).to(tl.float8e4nv, bitcast=True)


@triton.jit
def _f8_mw(v):
    u = v.to(tl.int32, bitcast=True)
    a = u & 0x7FFFFFFF
    nb = ((a + 0x7FFFF + ((a >> 20) & 1)) >> 20) - 960
    nb = tl.minimum(nb, 126)
    ns = (a.to(tl.float32, bitcast=True) + 16384.0).to(
        tl.int32, bitcast=True
    ) & 0x7FFFFF
    b = tl.where(a < 0x3C800000, ns, nb)
    b = b + (a > 0x7F800000).to(tl.int32)
    s = (u >> 24) & 0x80
    return (b | s).to(tl.int8).to(tl.float8e4nv, bitcast=True)


@triton.jit
def _absmax_bits(v):
    return v.to(tl.int32, bitcast=True) & 0x7FFFFFFF


@triton.jit
def k_fused(x_ptr, y_ptr, s_ptr, eps, N, BLK: tl.constexpr):
    off = tl.arange(0, BLK)
    m = off < N
    v = tl.load(x_ptr + off, mask=m, other=0.0).to(tl.float32)
    scale = _scale_from_bits(tl.max(_absmax_bits(v)), eps)
    tl.store(s_ptr, scale)
    tl.store(y_ptr + off, _f8(v * (1.0 / scale)), mask=m)


@triton.jit
def k_onepass(
    x_ptr, y_ptr, s_ptr, eps, N, CHUNK, BLK: tl.constexpr, ABLK: tl.constexpr
):
    pid = tl.program_id(0)
    acc = tl.zeros((ABLK,), dtype=tl.int32)
    aoffs = tl.arange(0, ABLK)
    for base in tl.range(0, N, ABLK):
        off = base + aoffs
        m = off < N
        v = tl.load(x_ptr + off, mask=m, other=0.0).to(tl.float32)
        acc = tl.maximum(acc, _absmax_bits(v))
    scale = _scale_from_bits(tl.max(acc), eps)
    tl.store(s_ptr, scale)
    inv = 1.0 / scale
    offs = tl.arange(0, BLK)
    my_lo = pid * CHUNK
    my_hi = tl.minimum(my_lo + CHUNK, N)
    for base in tl.range(my_lo, my_hi, BLK):
        off = base + offs
        m = off < N
        v = tl.load(x_ptr + off, mask=m, other=0.0).to(tl.float32)
        tl.store(y_ptr + off, _f8(v * inv), mask=m)


@triton.jit
def k_absmax(x_ptr, part_ptr, N, BLK: tl.constexpr, ITERS: tl.constexpr):
    pid = tl.program_id(0)
    base = pid * (BLK * ITERS)
    acc = tl.zeros((BLK,), dtype=tl.int32)
    for i in tl.range(0, ITERS):
        off = base + i * BLK + tl.arange(0, BLK)
        m = off < N
        v = tl.load(x_ptr + off, mask=m, other=0.0).to(tl.float32)
        acc = tl.maximum(acc, _absmax_bits(v))
    bits = tl.max(acc)
    tl.store(part_ptr + pid, bits.to(tl.float32, bitcast=True))


@triton.jit
def k_quant(
    x_ptr,
    y_ptr,
    part_ptr,
    eps,
    N,
    NP1,
    NPB: tl.constexpr,
    BLK: tl.constexpr,
    ITERS: tl.constexpr,
):
    pid = tl.program_id(0)
    po = tl.arange(0, NPB)
    p = tl.load(part_ptr + po, mask=po < NP1, other=0.0)
    mm = tl.max(p.to(tl.int32, bitcast=True))
    scale = _scale_from_bits(mm, eps)
    tl.store(part_ptr + NP1, scale)
    inv = 1.0 / scale
    base = pid * (BLK * ITERS)
    for i in tl.range(0, ITERS):
        off = base + i * BLK + tl.arange(0, BLK)
        m = off < N
        v = tl.load(x_ptr + off, mask=m, other=0.0).to(tl.float32)
        tl.store(y_ptr + off, _f8_mw(v * inv), mask=m)


@triton.jit
def _elem_off(off, D, S, sd, st, sh):
    d = off % D
    r = off // D
    t = r % S
    h = r // S
    return h * sh + t * st + d * sd


@triton.jit
def k_fused_s(
    x_ptr, y_ptr, s_ptr, eps, N, D, S, sd, st, sh, BLK: tl.constexpr
):
    off = tl.arange(0, BLK)
    m = off < N
    xo = _elem_off(off, D, S, sd, st, sh)
    v = tl.load(x_ptr + xo, mask=m, other=0.0).to(tl.float32)
    scale = _scale_from_bits(tl.max(_absmax_bits(v)), eps)
    tl.store(s_ptr, scale)
    tl.store(y_ptr + off, _f8(v * (1.0 / scale)), mask=m)


@triton.jit
def k_onepass_s(
    x_ptr,
    y_ptr,
    s_ptr,
    eps,
    N,
    CHUNK,
    D,
    S,
    sd,
    st,
    sh,
    BLK: tl.constexpr,
    ABLK: tl.constexpr,
):
    pid = tl.program_id(0)
    acc = tl.zeros((ABLK,), dtype=tl.int32)
    aoffs = tl.arange(0, ABLK)
    for base in tl.range(0, N, ABLK):
        off = base + aoffs
        m = off < N
        xo = _elem_off(off, D, S, sd, st, sh)
        v = tl.load(x_ptr + xo, mask=m, other=0.0).to(tl.float32)
        acc = tl.maximum(acc, _absmax_bits(v))
    scale = _scale_from_bits(tl.max(acc), eps)
    tl.store(s_ptr, scale)
    inv = 1.0 / scale
    offs = tl.arange(0, BLK)
    my_lo = pid * CHUNK
    my_hi = tl.minimum(my_lo + CHUNK, N)
    for base in tl.range(my_lo, my_hi, BLK):
        off = base + offs
        m = off < N
        xo = _elem_off(off, D, S, sd, st, sh)
        v = tl.load(x_ptr + xo, mask=m, other=0.0).to(tl.float32)
        tl.store(y_ptr + off, _f8(v * inv), mask=m)


@triton.jit
def k_absmax_s(
    x_ptr,
    part_ptr,
    N,
    D,
    S,
    sd,
    st,
    sh,
    BLK: tl.constexpr,
    ITERS: tl.constexpr,
):
    pid = tl.program_id(0)
    base = pid * (BLK * ITERS)
    acc = tl.zeros((BLK,), dtype=tl.int32)
    for i in tl.range(0, ITERS):
        off = base + i * BLK + tl.arange(0, BLK)
        m = off < N
        xo = _elem_off(off, D, S, sd, st, sh)
        v = tl.load(x_ptr + xo, mask=m, other=0.0).to(tl.float32)
        acc = tl.maximum(acc, _absmax_bits(v))
    bits = tl.max(acc)
    tl.store(part_ptr + pid, bits.to(tl.float32, bitcast=True))


@triton.jit
def k_quant_s(
    x_ptr,
    y_ptr,
    part_ptr,
    eps,
    N,
    NP1,
    D,
    S,
    sd,
    st,
    sh,
    NPB: tl.constexpr,
    BLK: tl.constexpr,
    ITERS: tl.constexpr,
):
    pid = tl.program_id(0)
    po = tl.arange(0, NPB)
    p = tl.load(part_ptr + po, mask=po < NP1, other=0.0)
    mm = tl.max(p.to(tl.int32, bitcast=True))
    scale = _scale_from_bits(mm, eps)
    tl.store(part_ptr + NP1, scale)
    inv = 1.0 / scale
    base = pid * (BLK * ITERS)
    for i in tl.range(0, ITERS):
        off = base + i * BLK + tl.arange(0, BLK)
        m = off < N
        xo = _elem_off(off, D, S, sd, st, sh)
        v = tl.load(x_ptr + xo, mask=m, other=0.0).to(tl.float32)
        tl.store(y_ptr + off, _f8_mw(v * inv), mask=m)


def _plan(n):
    chunk = max(2048, min(262144, triton.next_power_of_2(max(1, n // 512))))
    if n >= 131072 and chunk < 4096:
        chunk = 4096
    if chunk >= 65536:
        return 8192, chunk // 8192
    if chunk >= 4096:
        return 4096, chunk // 4096
    return chunk, 1


def per_tensor_quant_mla_fp8(x, eps=1e-12):
    H, S, D = x.shape[0], x.shape[1], x.shape[2]
    n = H * S * D
    y = torch.empty((H, S, D), dtype=_FP8_DTYPE, device=x.device)
    if not x.is_contiguous():
        sd, st, sh = x.stride(2), x.stride(1), x.stride(0)
        if n <= 2048:
            s = torch.empty(1, dtype=torch.float32, device=x.device)
            blk = max(16, triton.next_power_of_2(n))
            k_fused_s[(1,)](
                x, y, s, eps, n, D, S, sd, st, sh, BLK=blk, num_warps=4
            )
            return y, s
        if n <= 32768:
            s = torch.empty(1, dtype=torch.float32, device=x.device)
            g = 16 if n <= 8192 else 64
            chunk = (n + g - 1) // g
            blk = min(8192, max(16, triton.next_power_of_2(chunk)))
            k_onepass_s[(g,)](
                x,
                y,
                s,
                eps,
                n,
                chunk,
                D,
                S,
                sd,
                st,
                sh,
                BLK=blk,
                ABLK=8192,
                num_warps=8,
            )
            return y, s
        blk, iters = _plan(n)
        chunk = blk * iters
        npart = (n + chunk - 1) // chunk
        t = torch.empty(npart + 1, dtype=torch.float32, device=x.device)
        k_absmax_s[(npart,)](
            x, t, n, D, S, sd, st, sh, BLK=blk, ITERS=iters, num_warps=4
        )
        k_quant_s[(npart,)](
            x,
            y,
            t,
            eps,
            n,
            npart,
            D,
            S,
            sd,
            st,
            sh,
            NPB=triton.next_power_of_2(npart),
            BLK=blk,
            ITERS=iters,
            num_warps=8 if blk >= 8192 else 4,
        )
        return y, t[npart:]
    if n <= 2048:
        s = torch.empty(1, dtype=torch.float32, device=x.device)
        blk = max(16, triton.next_power_of_2(n))
        k_fused[(1,)](x, y, s, eps, n, BLK=blk, num_warps=4)
        return y, s
    if n <= 32768:
        s = torch.empty(1, dtype=torch.float32, device=x.device)
        g = 16 if n <= 8192 else 64
        chunk = (n + g - 1) // g
        blk = min(8192, max(16, triton.next_power_of_2(chunk)))
        k_onepass[(g,)](
            x, y, s, eps, n, chunk, BLK=blk, ABLK=8192, num_warps=8
        )
        return y, s
    blk, iters = _plan(n)
    chunk = blk * iters
    npart = (n + chunk - 1) // chunk
    t = torch.empty(npart + 1, dtype=torch.float32, device=x.device)
    k_absmax[(npart,)](x, t, n, BLK=blk, ITERS=iters, num_warps=4)
    k_quant[(npart,)](
        x,
        y,
        t,
        eps,
        n,
        npart,
        NPB=triton.next_power_of_2(npart),
        BLK=blk,
        ITERS=iters,
        num_warps=8 if blk >= 8192 else 4,
    )
    return y, t[npart:]


__all__ = ["per_tensor_quant_mla_fp8"]
