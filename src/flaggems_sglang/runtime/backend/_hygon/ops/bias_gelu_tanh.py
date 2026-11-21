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
    q = 0.5 * tl.exp(-a * (0.753152 + 0.396 * a))
    return tl.where(z >= 0.0, z * (1.0 - q), z * q)


@triton.jit
def _round(s, DT: tl.constexpr):
    if DT == tl.bfloat16:
        bits = s.to(tl.uint32, bitcast=True)
        rounded = (bits + 32767 + ((bits >> 16) & 1)) & 0xFFFF0000
        value = rounded.to(tl.float32, bitcast=True)
        return tl.where((bits & 2139095040) == 2139095040, s, value)
    return s.to(DT).to(tl.float32)


@triton.jit
def _r(X, BIAS, Y, BN: tl.constexpr):
    c = tl.arange(0, BN)
    o = tl.program_id(0) * BN + c
    b = tl.load(BIAS + c).to(tl.float32)
    x = tl.load(X + o).to(tl.float32)
    z = (x + b).to(X.dtype.element_ty).to(tl.float32)
    tl.store(Y + o, _out(_g(z), X.dtype.element_ty))


@triton.jit
def _out(v, DT: tl.constexpr):
    return v.to(DT)


@triton.jit
def _g_cubic(z):
    a = tl.abs(z)
    r = tl.minimum(a, 3.5000002384185791)
    q = r * (
        0.48036009073257446
        + r * (-0.15427060425281525 + r * 0.016526054590940475)
    )
    return 0.5 * z + a * q


@triton.jit
def _t(X, BIAS, Y, H: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr):
    nb = H // BN
    pid = tl.program_id(0)
    mb = pid // nb
    cols = (pid - mb * nb) * BN + tl.arange(0, BN)
    rows = mb * BM + tl.arange(0, BM)
    b = tl.load(BIAS + cols, eviction_policy="evict_last").to(tl.float32)
    off = rows[:, None] * H + cols[None, :]
    x = tl.load(X + off, eviction_policy="evict_first").to(tl.float32)
    z = _round(x + b[None, :], X.dtype.element_ty)
    tl.store(
        Y + off,
        _out(_g_cubic(z), X.dtype.element_ty),
        eviction_policy="evict_first",
    )


def bias_gelu_tanh(x, bias):
    y = torch.empty_like(x)
    h = x.shape[-1]
    n = x.numel()
    if n <= 65536 and h <= 4096 and h & (h - 1) == 0:
        _r[(n // h,)](x, bias, y, h)
        return y
    m = n // h
    bn = h & -h
    if bn > 2048:
        bn = 2048
    bm = m & -m
    cap = 2048 // bn
    if cap < 1:
        cap = 1
    if bm > cap:
        bm = cap
    _t[(m // bm * (h // bn),)](x, bias, y, h, bm, bn)
    return y


__all__ = ["bias_gelu_tanh"]
