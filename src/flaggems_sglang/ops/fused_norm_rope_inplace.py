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

"""Triton fused_norm_rope_inplace (candidate v1)."""

import torch
import triton
import triton.language as tl


@triton.jit
def _fnri_w(X, W, F, POS, O, D, NP, H, HEAD0, EPS, XS: tl.constexpr,
            FS: tl.constexpr, PS: tl.constexpr, OS: tl.constexpr,
            B: tl.constexpr, PB: tl.constexpr):
    row = tl.program_id(0)
    t = row // H
    cols = tl.arange(0, B)
    m = cols < D
    base = X + row.to(tl.int64) * XS
    x = tl.load(base + cols, mask=m, other=0.0).to(tl.float32)
    ssq = tl.sum(x * x, axis=0)
    inv = tl.rsqrt(ssq / D + EPS)
    w = tl.load(W + cols, mask=m, other=0.0).to(tl.float32)
    y = x * inv * w
    ob = O + row.to(tl.int64) * OS
    tl.store(ob + cols, y, mask=cols < HEAD0)
    p = tl.arange(0, PB)
    mp = p < NP
    fr = tl.load(POS + t * PS).to(tl.int32)
    off = HEAD0 + 2 * p
    x1 = tl.load(base + off, mask=mp, other=0.0).to(tl.float32) * inv
    x1 = x1 * tl.load(W + off, mask=mp, other=0.0).to(tl.float32)
    x2 = tl.load(base + off + 1, mask=mp, other=0.0).to(tl.float32) * inv
    x2 = x2 * tl.load(W + off + 1, mask=mp, other=0.0).to(tl.float32)
    fbase = F + fr.to(tl.int64) * FS
    c = tl.load(fbase + 2 * p, mask=mp, other=0.0).to(tl.float32)
    s = tl.load(fbase + 2 * p + 1, mask=mp, other=0.0).to(tl.float32)
    o1 = x1 * c - x2 * s
    o2 = x1 * s + x2 * c
    tl.store(ob + off, o1, mask=mp)
    tl.store(ob + off + 1, o2, mask=mp)


@triton.jit
def _fnri_now(X, F, POS, O, D, NP, H, HEAD0, EPS, XS: tl.constexpr,
              FS: tl.constexpr, PS: tl.constexpr, OS: tl.constexpr,
              B: tl.constexpr, PB: tl.constexpr):
    row = tl.program_id(0)
    t = row // H
    cols = tl.arange(0, B)
    m = cols < D
    base = X + row.to(tl.int64) * XS
    x = tl.load(base + cols, mask=m, other=0.0).to(tl.float32)
    ssq = tl.sum(x * x, axis=0)
    inv = tl.rsqrt(ssq / D + EPS)
    y = x * inv
    ob = O + row.to(tl.int64) * OS
    tl.store(ob + cols, y, mask=cols < HEAD0)
    p = tl.arange(0, PB)
    mp = p < NP
    fr = tl.load(POS + t * PS).to(tl.int32)
    off = HEAD0 + 2 * p
    x1 = tl.load(base + off, mask=mp, other=0.0).to(tl.float32) * inv
    x2 = tl.load(base + off + 1, mask=mp, other=0.0).to(tl.float32) * inv
    fbase = F + fr.to(tl.int64) * FS
    c = tl.load(fbase + 2 * p, mask=mp, other=0.0).to(tl.float32)
    s = tl.load(fbase + 2 * p + 1, mask=mp, other=0.0).to(tl.float32)
    o1 = x1 * c - x2 * s
    o2 = x1 * s + x2 * c
    tl.store(ob + off, o1, mask=mp)
    tl.store(ob + off + 1, o2, mask=mp)


def fused_norm_rope_inplace(kv, weight, eps, freqs_cis, positions=None):
    orig = kv.shape
    h = orig[-2]
    d = orig[-1]
    kv2 = kv.reshape(-1, d)
    n = kv2.shape[0]
    out = torch.empty_like(kv2)
    if n == 0:
        return out.reshape(orig)
    npairs = freqs_cis.shape[-2]
    head0 = d - 2 * npairs
    b = triton.next_power_of_2(d)
    pb = triton.next_power_of_2(npairs)
    warps = 4 if b <= 1024 else 8
    if positions is None:
        positions = torch.arange(n // h, device=kv.device)
    pos = positions.contiguous()
    common = (
        kv2,
        freqs_cis,
        pos,
        out,
        d,
        npairs,
        h,
        head0,
        float(eps),
        kv2.stride(0),
        freqs_cis.stride(0),
        pos.stride(0),
        out.stride(0),
        b,
        pb,
    )
    if weight is None:
        _fnri_now[(n,)](*common, num_warps=warps)
    else:
        _fnri_w[(n,)](kv2, weight, *common[1:], num_warps=warps)
    return out.reshape(orig)


__all__ = ["fused_norm_rope_inplace"]