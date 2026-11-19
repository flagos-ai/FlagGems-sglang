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
def _g(z):
    a = tl.abs(z)
    r = tl.minimum(a, 3.5000002384185791)
    q = r * (
        0.48036009073257446
        + r * (-0.15427060425281525 + r * 0.016526054590940475)
    )
    return 0.5 * z + a * q


@triton.jit
def _round(v):
    bits = v.to(tl.uint32, bitcast=True)
    r = ((bits + 32767 + ((bits >> 16) & 1)) & 4294901760).to(
        tl.float32, bitcast=True
    )
    return tl.where((bits & 2139095040) == 2139095040, v, r)


@triton.jit
def _r(X, BIAS, Y, BN: tl.constexpr, BF16: tl.constexpr):
    c = tl.arange(0, BN)
    o = tl.program_id(0) * BN + c
    s = tl.load(X + o).to(tl.float32) + tl.load(BIAS + c).to(tl.float32)
    if BF16:
        z = _round(s)
    else:
        z = s.to(tl.float16).to(tl.float32)
    tl.store(Y + o, _g(z).to(X.dtype.element_ty))


@triton.jit
def _p(
    X,
    BIAS,
    Y,
    N: tl.constexpr,
    H: tl.constexpr,
    BN: tl.constexpr,
    BF16: tl.constexpr,
):
    o = tl.program_id(0) * BN + tl.arange(0, BN)
    k = o < N
    s = tl.load(X + o, mask=k, other=0.0).to(tl.float32) + tl.load(
        BIAS + (o % H), mask=k, other=0.0
    ).to(tl.float32)
    if BF16:
        z = _round(s)
    else:
        z = s.to(tl.float16).to(tl.float32)
    tl.store(Y + o, _g(z).to(X.dtype.element_ty), mask=k)


@triton.jit
def _c(
    X,
    BIAS,
    Y,
    H: tl.constexpr,
    BN: tl.constexpr,
    T: tl.constexpr,
    BF16: tl.constexpr,
):
    nb = H // BN
    lane = tl.arange(0, BN)
    for t in range(tl.program_id(0), T, tl.num_programs(0)):
        row = t // nb
        c = (t - row * nb) * BN + lane
        o = row * H + c
        s = tl.load(X + o).to(tl.float32) + tl.load(BIAS + c).to(tl.float32)
        if BF16:
            z = _round(s)
        else:
            z = s.to(tl.float16).to(tl.float32)
        tl.store(Y + o, _g(z).to(X.dtype.element_ty))


@triton.jit
def _t(
    X,
    BIAS,
    Y,
    N: tl.constexpr,
    H: tl.constexpr,
    BLOCK: tl.constexpr,
    BF16: tl.constexpr,
):
    lane = tl.arange(0, BLOCK)
    hb = tl.load(BIAS + (lane % H)).to(tl.float32)
    for start in range(
        tl.program_id(0) * BLOCK, N, tl.num_programs(0) * BLOCK
    ):
        i = start + lane
        s = tl.load(X + i).to(tl.float32) + hb
        if BF16:
            z = _round(s)
        else:
            z = s.to(tl.float16).to(tl.float32)
        tl.store(Y + i, _g(z).to(X.dtype.element_ty))


def bias_gelu_tanh(x, bias):
    y = torch.empty_like(x)
    h = x.shape[-1]
    n = x.numel()
    if n <= 65536 and h < 512 and h & (h - 1) == 0:
        _p[((n + 511) // 512,)](
            x, bias, y, n, h, 512, x.dtype == torch.bfloat16
        )
        return y
    if n <= 65536 and h <= 4096 and h & (h - 1) == 0:
        _r[(n // h,)](x, bias, y, h, x.dtype == torch.bfloat16)
        return y
    m = n // h
    if h > 16384 or h & (h - 1):
        bn = h & -h
        if bn > 16384:
            bn = 16384
        t = m * (h // bn)
        g = t if t < 12 else 12
        _c[(g,)](x, bias, y, h, bn, t, x.dtype == torch.bfloat16)
        return y
    r = m & -m
    cap = 16384 // h
    if r > cap:
        r = cap
    blk = h * r
    t = n // blk
    g = t if t < 12 else 12
    _t[(g,)](x, bias, y, n, h, blk, x.dtype == torch.bfloat16)
    return y


__all__ = ["bias_gelu_tanh"]
