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
    u = z * (1.5957691216057308 + 0.07135481627260025 * z * z)
    return z / (1.0 + tl.exp(-u))


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
def _t(X, BIAS, Y, H: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr):
    nb = H // BN
    pid = tl.program_id(0)
    mb = pid // nb
    cols = (pid - mb * nb) * BN + tl.arange(0, BN)
    rows = mb * BM + tl.arange(0, BM)
    b = tl.load(BIAS + cols).to(tl.float32)
    off = rows[:, None] * H + cols[None, :]
    x = tl.load(X + off, cache_modifier=".cv").to(tl.float32)
    z = (x + b[None, :]).to(X.dtype.element_ty).to(tl.float32)
    tl.store(Y + off, _out(_g(z), X.dtype.element_ty))


def bias_gelu_tanh(x, bias):
    y = torch.empty_like(x)
    h = x.shape[-1]
    n = x.numel()
    if n <= 65536 and h <= 4096 and h & (h - 1) == 0:
        _r[(n // h,)](x, bias, y, h)
        return y
    m = n // h
    bn = h & -h
    if bn > 1024:
        bn = 1024
    bm = m & -m
    cap = 1024 // bn
    if cap < 1:
        cap = 1
    if bm > cap:
        bm = cap
    _t[(m // bm * (h // bn),)](x, bias, y, h, bm, bn, num_warps=8)
    return y


__all__ = ["bias_gelu_tanh"]
