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
def _r(X, BIAS, Y, BN: tl.constexpr):
    c = tl.arange(0, BN)
    o = tl.program_id(0) * BN + c
    b = tl.load(BIAS + c, eviction_policy="evict_last").to(tl.float32)
    x = tl.load(X + o, eviction_policy="evict_first").to(tl.float32)
    z = (x + b).to(X.dtype.element_ty).to(tl.float32)
    tl.store(
        Y + o, _out(_g(z), X.dtype.element_ty), eviction_policy="evict_first"
    )


@triton.jit
def _out(v, DT: tl.constexpr):
    return v.to(DT)


@triton.jit
def _t(X, BIAS, Y, H: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    s = tl.load(X + i, eviction_policy="evict_first").to(tl.float32) + tl.load(
        BIAS + (i % H), eviction_policy="evict_last"
    ).to(tl.float32)
    z = s.to(X.dtype.element_ty).to(tl.float32)
    tl.store(
        Y + i, _out(_g(z), X.dtype.element_ty), eviction_policy="evict_first"
    )


def bias_gelu_tanh(x, bias):
    y = torch.empty_like(x)
    h = x.shape[-1]
    n = x.numel()
    if n <= 65536 and h <= 4096 and h & (h - 1) == 0:
        _r[(n // h,)](x, bias, y, h)
        return y
    blk = n & -n
    if blk > 2048:
        blk = 2048
    _t[(n // blk,)](x, bias, y, h, blk, num_stages=3)
    return y


__all__ = ["bias_gelu_tanh"]
