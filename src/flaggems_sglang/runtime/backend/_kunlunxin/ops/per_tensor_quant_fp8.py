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


def _dense_permutation(shape, strides, n):
    if n == 0:
        return False
    axes = sorted(((s, d) for d, s in zip(shape, strides) if d > 1))
    extent = 1
    for stride, dim in axes:
        if stride != extent:
            return False
        extent *= dim
    return extent == n


@triton.jit
def _maximum(a, b):
    return tl.maximum(a, b, propagate_nan=tl.PropagateNan.ALL)


@triton.jit
def _read(
    X,
    i,
    N: tl.constexpr,
    SH: tl.constexpr,
    ST: tl.constexpr,
    CONTIG: tl.constexpr,
    NEG: tl.constexpr,
):
    if CONTIG:
        off = i
    else:
        rem = i
        off = tl.full(i.shape, 0, tl.int64)
        for d in tl.static_range(len(SH) - 1, -1, -1):
            off += rem % SH[d] * ST[d]
            rem = rem // SH[d]
    x = tl.load(X + off, i < N, other=0).to(tl.float32)
    if NEG:
        x = -x
    return x


@triton.jit
def _encode(value):
    E4: tl.constexpr = True
    CANONICAL_NAN: tl.constexpr = False
    MB: tl.constexpr = 3 if E4 else 2
    BIAS: tl.constexpr = 7 if E4 else 15
    SHIFT: tl.constexpr = 23 - MB
    bits = value.to(tl.uint32, bitcast=True)
    sign = bits >> 24 & 128
    raw_absolute = bits & 2147483647
    absolute = tl.minimum(raw_absolute, 1138753536 if E4 else 1197473792)
    exponent = tl.maximum(
        (absolute & 2139095040).to(tl.int32), 128 - BIAS << 23
    )
    bias_bits = exponent + (SHIFT << 23)
    rounded = absolute.to(tl.float32, bitcast=True) + bias_bits.to(
        tl.float32, bitcast=True
    )
    encoded = (
        rounded.to(tl.uint32, bitcast=True)
        + (exponent >> SHIFT)
        - (128 - BIAS << MB)
    )
    if CANONICAL_NAN:
        return tl.where(raw_absolute > 2139095040, 127, sign | encoded).to(
            tl.uint8
        )
    encoded = tl.where(raw_absolute > 2139095040, 127, encoded)
    return (sign | encoded).to(tl.uint8)


@triton.jit
def _partial(
    X,
    P,
    N: tl.constexpr,
    SH: tl.constexpr,
    ST: tl.constexpr,
    CONTIG: tl.constexpr,
    B: tl.constexpr,
    GRID: tl.constexpr,
    PAD: tl.constexpr,
):
    pid = tl.program_id(0)
    v = tl.zeros((B,), tl.float32)
    for k in range(pid, tl.cdiv(N, B), GRID):
        x = _read(
            X, k.to(tl.int64) * B + tl.arange(0, B), N, SH, ST, CONTIG, False
        )
        v = _maximum(v, tl.abs(x))
    m = tl.reduce(v, 0, _maximum)
    tl.store(P + pid * PAD, m)


@triton.jit
def _finish(P, S, G: tl.constexpr, PAD: tl.constexpr, B: tl.constexpr):
    i = tl.arange(0, B)
    v = tl.load(P + i * PAD, i < G, other=0)
    tl.store(S, tl.reduce(v, 0, _maximum) / 448.0)


@triton.jit
def _quant(
    X,
    O,
    S,
    N: tl.constexpr,
    SH: tl.constexpr,
    ST: tl.constexpr,
    CONTIG: tl.constexpr,
    NEG: tl.constexpr,
    SNEG: tl.constexpr,
    B: tl.constexpr,
    DYNAMIC_2048: tl.constexpr,
):
    i = tl.program_id(0).to(tl.int64) * B + tl.arange(0, B)
    x = _read(X, i, N, SH, ST, CONTIG, NEG)
    if DYNAMIC_2048:
        scale = tl.reduce(tl.abs(x), 0, _maximum) / 448.0
        tl.store(S, scale)
    else:
        scale = tl.load(S).to(tl.float32)
        if SNEG:
            scale = -scale
    q = _encode(x / scale)
    tl.store(O + i, q, i < N)


@triton.jit
def _quant_fullcore_finish(
    X,
    P,
    O,
    S,
    N: tl.constexpr,
    SH: tl.constexpr,
    ST: tl.constexpr,
    CONTIG: tl.constexpr,
    NEG: tl.constexpr,
    G: tl.constexpr,
    PAD: tl.constexpr,
    B: tl.constexpr,
):
    ri = tl.arange(0, 64)
    slot = (ri % G).to(tl.int64)
    partial = tl.load(P + slot * PAD)
    scale = tl.reduce(partial, 0, _maximum) / 448.0
    i = tl.program_id(0).to(tl.int64) * B + tl.arange(0, B)
    x = _read(X, i, N, SH, ST, CONTIG, NEG)
    tl.store(O + i, _encode(x / scale), i < N)
    if tl.program_id(0) == 0:
        tl.store(S, scale)


def per_tensor_quant_fp8(input, output_q, output_s, is_static):
    n = input.numel()
    shape = tuple(input.shape)
    strides = tuple(input.stride())
    physical = not input.is_contiguous() and _dense_permutation(
        shape, strides, n
    )
    contiguous = input.is_contiguous() or physical
    qbytes = (
        torch.empty_strided(
            shape, strides, dtype=torch.uint8, device=input.device
        )
        if physical
        else torch.empty(shape, dtype=torch.uint8, device=input.device)
    )
    if is_static:
        scale = output_s
    else:
        scale = torch.empty(
            output_s.shape, dtype=torch.float32, device=input.device
        )
    if not is_static and n <= 2048:
        b = triton.next_power_of_2(max(n, 1))
        _quant[1,](
            input,
            qbytes,
            scale,
            n,
            shape,
            strides,
            contiguous,
            input.is_neg(),
            False,
            b,
            True,
            num_warps=4,
            num_stages=1,
        )
    elif not is_static and n <= 65536:
        grid = triton.cdiv(n, 2048)
        p = torch.empty((grid * 32,), dtype=torch.float32, device=input.device)
        _partial[grid,](
            input,
            p,
            n,
            shape,
            strides,
            contiguous,
            2048,
            grid,
            32,
            num_warps=4,
            num_stages=1,
        )
        _quant_fullcore_finish[triton.cdiv(n, 2048),](
            input,
            p,
            qbytes,
            scale,
            n,
            shape,
            strides,
            contiguous,
            input.is_neg(),
            grid,
            32,
            2048,
            num_warps=4,
            num_stages=1,
        )
    else:
        if not is_static:
            grid = min(triton.cdiv(n, 2048), 48)
            p = torch.empty(
                (grid * 32,), dtype=torch.float32, device=input.device
            )
            _partial[grid,](
                input,
                p,
                n,
                shape,
                strides,
                contiguous,
                2048,
                grid,
                32,
                num_warps=4,
                num_stages=1,
            )
            _finish[1,](
                p,
                scale,
                grid,
                32,
                triton.next_power_of_2(grid),
                num_warps=4,
                num_stages=1,
            )
        _quant[triton.cdiv(n, 2048),](
            input,
            qbytes,
            scale,
            n,
            shape,
            strides,
            contiguous,
            input.is_neg(),
            scale.is_neg(),
            2048,
            False,
            num_warps=4,
            num_stages=1,
        )
    return (qbytes.view(torch.float8_e4m3fn), scale)


__all__ = ["per_tensor_quant_fp8"]
