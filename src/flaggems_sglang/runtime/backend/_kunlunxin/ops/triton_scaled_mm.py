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
def _rne_to_int(v, scale: tl.constexpr):
    t = v * scale
    i = t.to(tl.int32)
    frac = t - i.to(tl.float32)
    bump = (frac > 0.5) | ((frac == 0.5) & ((i & 1) == 1))
    return i + bump.to(tl.int32)


@triton.jit
def _encode(x, MANT: tl.constexpr, BIAS: tl.constexpr, OVERFLOW: tl.constexpr):
    bits = x.to(tl.float32).to(tl.int32, bitcast=True)
    mag = bits & 0x7FFFFFFF
    sign = (bits >> 24) & 128
    shift = 23 - MANT
    rounded = (mag + ((1 << (shift - 1)) - 1) + ((mag >> shift) & 1)) >> shift
    normal = rounded - ((127 - BIAS) << MANT)
    subnormal = _rne_to_int(tl.abs(x.to(tl.float32)), 2.0 ** (BIAS + MANT - 1))
    code = tl.where(mag < ((128 - BIAS) << 23), subnormal, normal)
    code = tl.maximum(tl.minimum(code, OVERFLOW), 0)
    code = tl.where(mag > 0x7F800000, 127, code)
    code = code | sign
    return code.to(tl.uint8)


@triton.jit
def _decode(
    byte, MANT: tl.constexpr, BIAS: tl.constexpr, IS_E4M3: tl.constexpr
):
    b = byte.to(tl.int32)
    sign = (b >> 7) & 1
    exponent = (b >> MANT) & ((1 << (8 - MANT - 1)) - 1)
    fraction = b & ((1 << MANT) - 1)
    if IS_E4M3:
        is_nan = (exponent == 15) & (fraction == 7)
        is_inf = False
    else:
        is_nan = (exponent == 31) & (fraction != 0)
        is_inf = (exponent == 31) & (fraction == 0)
    sub = fraction.to(tl.float32) * (2.0 ** (1 - BIAS - MANT))
    normal = (((exponent - BIAS + 127) << 23) | (fraction << (23 - MANT))).to(
        tl.float32, bitcast=True
    )
    val = tl.where(exponent == 0, sub, normal)
    val = tl.where(is_nan, float("nan"), val)
    val = tl.where(is_inf, float("inf"), val)
    return tl.where(sign == 1, -val, val)


@triton.jit
def e4m3_decode(byte):
    return _decode(byte, MANT=3, BIAS=7, IS_E4M3=True)


_BLOCK = 1024
_BM = 128
_BN = 128
_BK = 64


@triton.jit
def _decode_kernel(x_ptr, o_ptr, n, FP8: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    off = pid * BLOCK + tl.arange(0, BLOCK)
    live = off < n
    if FP8:
        v = e4m3_decode(tl.load(x_ptr + off, live, other=0))
    else:
        v = tl.load(x_ptr + off, live, other=0).to(tl.float32)
    tl.store(o_ptr + off, v.to(tl.bfloat16), live)


@triton.jit
def _mm_kernel(
    a_ptr,
    b_ptr,
    sa_ptr,
    sb_ptr,
    bias_ptr,
    c_ptr,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rm = pid_m * BM + tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    mm = rm < M
    nm = rn < N
    acc = tl.zeros((BM, BN), tl.float32)
    for k0 in range(0, K, BK):
        kk = k0 + rk
        kk_live = kk < K
        a = tl.load(
            a_ptr + rm[:, None] * K + kk[None, :],
            mm[:, None] & kk_live[None, :],
            other=0.0,
        )
        b = tl.load(
            b_ptr + kk[:, None] * N + rn[None, :],
            kk_live[:, None] & nm[None, :],
            other=0.0,
        )
        acc += tl.dot(a, b, input_precision="ieee")
    v = (
        acc
        * tl.load(sa_ptr + rm, mm, other=0.0)[:, None]
        * tl.load(sb_ptr + rn, nm, other=0.0)[None, :]
    )
    if HAS_BIAS:
        v = v + tl.load(bias_ptr + rn, nm, other=0.0)[None, :]
    tl.store(
        c_ptr + rm[:, None] * N + rn[None, :],
        v.to(c_ptr.dtype.element_ty),
        mm[:, None] & nm[None, :],
    )


def _is_fp8(t):
    return t.dtype in (torch.float8_e4m3fn, torch.float8_e5m2)


def triton_scaled_mm(input, weight, scale_a, scale_b, out_dtype, bias=None):
    m, k = input.shape
    n = weight.shape[1]
    out = torch.empty((m, n), device=input.device, dtype=out_dtype)
    if m == 0 or n == 0:
        return out
    fp8 = _is_fp8(input)
    a = input if input.is_contiguous() else input.contiguous()
    b = weight if weight.is_contiguous() else weight.contiguous()
    a_f = torch.empty((m, k), device=input.device, dtype=torch.bfloat16)
    b_f = torch.empty((k, n), device=input.device, dtype=torch.bfloat16)
    av = a.view(torch.uint8) if fp8 else a
    bv = b.view(torch.uint8) if fp8 else b
    _decode_kernel[(triton.cdiv(a.numel(), _BLOCK),)](
        av, a_f, a.numel(), FP8=fp8, BLOCK=_BLOCK, num_warps=4
    )
    _decode_kernel[(triton.cdiv(b.numel(), _BLOCK),)](
        bv, b_f, b.numel(), FP8=fp8, BLOCK=_BLOCK, num_warps=4
    )
    sa = scale_a.reshape(-1).to(torch.float32).contiguous()
    sb = scale_b.reshape(-1).to(torch.float32).contiguous()
    _mm_kernel[(triton.cdiv(m, _BM), triton.cdiv(n, _BN))](
        a_f,
        b_f,
        sa,
        sb,
        bias if bias is not None else sa,
        out,
        m,
        N=n,
        K=k,
        BM=_BM,
        BN=_BN,
        BK=_BK,
        HAS_BIAS=bias is not None,
        num_warps=8,
    )
    return out


__all__ = ["triton_scaled_mm"]
