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


@triton.jit
def _offset(i, SHAPE: tl.constexpr, STRIDES: tl.constexpr):
    if len(SHAPE) == 1:
        return i * STRIDES[0]
    else:
        off = tl.full(i.shape, 0, tl.int32)
        for dim in tl.static_range(len(SHAPE) - 1, -1, -1):
            off += i % SHAPE[dim] * STRIDES[dim]
            i = i // SHAPE[dim]
        return off


@triton.jit
def _encode(v):
    raw = v.to(tl.int32, bitcast=True)
    sign = raw >> 24 & 128
    mag = raw & 2147483647
    nan = mag > 2139095040
    mag = tl.minimum(mag, 1138753536)
    normal = (mag + 524287 + (mag >> 20 & 1) >> 20) - 960
    a = mag.to(tl.float32, bitcast=True)
    rounded = a + 16384.0
    sub = rounded.to(tl.int32, bitcast=True) - 1182793728
    code = tl.where(mag < 1015021568, sub, normal)
    return (tl.where(nan, 127, code) | sign).to(tl.uint8)


@triton.jit
def _partial(
    X,
    P,
    N: tl.constexpr,
    SHAPE: tl.constexpr,
    STRIDES: tl.constexpr,
    B: tl.constexpr,
):
    pid = tl.program_id(0)
    if N > 8388608:
        pid = tl.cdiv(N, B) - 1 - pid
    i = pid * B + tl.arange(0, B)
    x = tl.load(
        X + _offset(i, SHAPE, STRIDES),
        i < N,
        other=0,
        eviction_policy="evict_last",
    ).to(tl.float32)
    a = tl.max(x.to(tl.int32, bitcast=True) & 2147483647, 0).to(
        tl.float32, bitcast=True
    )
    tl.store(P + tl.program_id(0), a)


@triton.jit
def _scale(P, S, K: tl.constexpr, B: tl.constexpr):
    i = tl.arange(0, B)
    a = tl.load(P + i, i < K, other=0)
    s = (
        tl.max(a.to(tl.int32, bitcast=True) & 2147483647, 0).to(
            tl.float32, bitcast=True
        )
        / 448.0
    )
    tl.store(S, s)


@triton.jit
def _quant(
    X,
    Q: tl.pointer_type(tl.uint8),
    S,
    N: tl.constexpr,
    SHAPE: tl.constexpr,
    STRIDES: tl.constexpr,
    B: tl.constexpr,
):
    Q = Q.to(tl.pointer_type(tl.uint8))
    i = tl.program_id(0) * B + tl.arange(0, B)
    x = tl.load(X + _offset(i, SHAPE, STRIDES), i < N, other=0).to(tl.float32)
    s = tl.load(S)
    v = _divide_fp32(x, s, X.dtype.element_ty == tl.float32)
    tl.store(Q + i, _encode(v), i < N)


@triton.jit
def _small(
    X,
    Q: tl.pointer_type(tl.uint8),
    S,
    N: tl.constexpr,
    SHAPE: tl.constexpr,
    STRIDES: tl.constexpr,
    B: tl.constexpr,
):
    Q = Q.to(tl.pointer_type(tl.uint8))
    i = tl.arange(0, B)
    x = tl.load(X + _offset(i, SHAPE, STRIDES), i < N, other=0).to(tl.float32)
    s = (
        tl.max(x.to(tl.int32, bitcast=True) & 2147483647, 0).to(
            tl.float32, bitcast=True
        )
        / 448.0
    )
    tl.store(S, s)
    tl.store(
        Q + i,
        _encode_bias(_divide_fp32(x, s, X.dtype.element_ty == tl.float32)),
        i < N,
    )


@triton.jit
def _finish(
    X,
    P,
    Q: tl.pointer_type(tl.uint8),
    S,
    N: tl.constexpr,
    SHAPE: tl.constexpr,
    STRIDES: tl.constexpr,
    K: tl.constexpr,
    R: tl.constexpr,
    B: tl.constexpr,
):
    Q = Q.to(tl.pointer_type(tl.uint8))
    r = tl.arange(0, R)
    a = tl.load(P + r, r < K, other=0)
    s = (
        tl.max(a.to(tl.int32, bitcast=True) & 2147483647, 0).to(
            tl.float32, bitcast=True
        )
        / 448.0
    )
    if tl.program_id(0) == 0:
        tl.store(S, s)
    i = tl.program_id(0) * B + tl.arange(0, B)
    x = tl.load(X + _offset(i, SHAPE, STRIDES), i < N, other=0).to(tl.float32)
    tl.store(
        Q + i,
        _encode(_divide_fp32(x, s, X.dtype.element_ty == tl.float32)),
        i < N,
    )


@triton.jit
def _redundant(
    X,
    Q: tl.pointer_type(tl.uint8),
    S,
    N: tl.constexpr,
    SHAPE: tl.constexpr,
    STRIDES: tl.constexpr,
    R: tl.constexpr,
    B: tl.constexpr,
):
    Q = Q.to(tl.pointer_type(tl.uint8))
    r = tl.arange(0, R)
    x = tl.load(
        X + _offset(r, SHAPE, STRIDES),
        r < N,
        other=0,
        eviction_policy="evict_last",
    ).to(tl.float32)
    s = (
        tl.max(x.to(tl.int32, bitcast=True) & 2147483647, 0).to(
            tl.float32, bitcast=True
        )
        / 448.0
    )
    if tl.program_id(0) == 0:
        tl.store(S, s)
    i = tl.program_id(0) * B + tl.arange(0, B)
    value = tl.load(X + _offset(i, SHAPE, STRIDES), i < N, other=0).to(
        tl.float32
    )
    tl.store(
        Q + i,
        _encode_bias(_divide_fp32(value, s, X.dtype.element_ty == tl.float32)),
        i < N,
    )


@triton.jit
def _encode_bias(v):
    raw = v.to(tl.int32, bitcast=True)
    sign = raw >> 24 & 128
    mag = raw & 2147483647
    nan = mag > 2139095040
    mag = tl.minimum(mag, 1138753536)
    normal = mag - 1006108673 + (mag >> 20 & 1) >> 20
    a = mag.to(tl.float32, bitcast=True)
    rounded = a + 16384.0
    sub = rounded.to(tl.int32, bitcast=True) - 1182793728
    code = tl.where(mag < 1015021568, sub, normal)
    return (tl.where(nan, 127, code) | sign).to(tl.uint8)


@triton.jit
def _encode_dynamic(v, s):
    sb = s.to(tl.int32, bitcast=True)
    if (sb >= 8388608) & (sb < 2139095040):
        raw = v.to(tl.int32, bitcast=True)
        sign = raw >> 24 & 128
        mag = raw & 2147483647
        normal = mag - 1006108673 + (mag >> 20 & 1) >> 20
        a = mag.to(tl.float32, bitcast=True)
        rounded = a + 16384.0
        sub = rounded.to(tl.int32, bitcast=True) - 1182793728
        code = tl.where(mag < 1015021568, sub, normal)
        result = (code | sign).to(tl.uint8)
    else:
        result = _encode(v)
    return result


@triton.jit
def _quant_dynamic(
    X,
    Q: tl.pointer_type(tl.uint8),
    S,
    N: tl.constexpr,
    SHAPE: tl.constexpr,
    STRIDES: tl.constexpr,
    B: tl.constexpr,
):
    Q = Q.to(tl.pointer_type(tl.uint8))
    i = tl.program_id(0) * B + tl.arange(0, B)
    x = tl.load(X + _offset(i, SHAPE, STRIDES), i < N, other=0).to(tl.float32)
    s = tl.load(S)
    v = _divide_fp32(x, s, X.dtype.element_ty == tl.float32)
    tl.store(Q + i, _encode_dynamic(v, s), i < N, cache_modifier=".cs")


@triton.jit
def _divide_fp32(x, s, PRECISE: tl.constexpr):
    if PRECISE:
        return tl.div_rn(x, s)
    else:
        return x / s


def per_tensor_quant_fp8(input, output_q, output_s, is_static):
    n = input.numel()
    q = torch.empty_like(
        output_q,
        dtype=torch.float8_e4m3fn,
        memory_format=torch.contiguous_format,
    )
    if input.is_contiguous():
        shape, strides = ((n,), (1,))
    else:
        shape = tuple(input.shape)
        strides = tuple(input.stride())
    if is_static:
        s = output_s
        _quant[triton.cdiv(n, 1024),](input, q, s, n, shape, strides, 1024)
    else:
        s = torch.empty_like(
            output_s,
            dtype=torch.float32,
            memory_format=torch.contiguous_format,
        )
        if 32768 < n <= 65536:
            _redundant[triton.cdiv(n, 8192),](
                input,
                q,
                s,
                n,
                shape,
                strides,
                triton.next_power_of_2(n),
                8192,
                num_warps=16,
            )
        elif 4096 < n <= 8192:
            _redundant[triton.cdiv(n, 512),](
                input,
                q,
                s,
                n,
                shape,
                strides,
                triton.next_power_of_2(n),
                512,
                num_warps=4,
            )
        elif n <= 8192:
            _small[1,](
                input,
                q,
                s,
                n,
                shape,
                strides,
                triton.next_power_of_2(n),
                num_warps=4,
            )
        elif n <= 32768:
            _redundant[triton.cdiv(n, 2048),](
                input,
                q,
                s,
                n,
                shape,
                strides,
                triton.next_power_of_2(n),
                2048,
                num_warps=8,
            )
        else:
            k = triton.cdiv(n, 8192)
            p = torch.empty((k,), dtype=torch.float32, device=input.device)
            _partial[k,](input, p, n, shape, strides, 8192)
            if k <= 512:
                _finish[triton.cdiv(n, 2048),](
                    input,
                    p,
                    q,
                    s,
                    n,
                    shape,
                    strides,
                    k,
                    triton.next_power_of_2(k),
                    2048,
                )
            else:
                _scale[1,](p, s, k, triton.next_power_of_2(k))
                _quant_dynamic[triton.cdiv(n, 4096),](
                    input, q, s, n, shape, strides, 4096
                )
    return (q, s)


__all__ = ["per_tensor_quant_fp8"]
