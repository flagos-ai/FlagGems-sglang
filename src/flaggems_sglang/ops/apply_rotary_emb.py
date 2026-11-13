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

"""Triton apply_rotary_emb (candidate v6, branch-free)."""

import torch
import triton
import triton.language as tl


@triton.jit
def _are_nopos(X, F, O, P, H: tl.constexpr, INV, XS: tl.constexpr,
               FS: tl.constexpr, OS: tl.constexpr, PB: tl.constexpr):
    row = tl.program_id(0)
    p = tl.arange(0, PB)
    m = p < P
    off = 2 * p
    t = row // H
    x1 = tl.load(X + row.to(tl.int64) * XS + off, mask=m,
                 other=0.0).to(tl.float32)
    x2 = tl.load(X + row.to(tl.int64) * XS + off + 1, mask=m,
                 other=0.0).to(tl.float32)
    c = tl.load(F + t * FS + off, mask=m, other=0.0).to(tl.float32)
    s = tl.load(F + t * FS + off + 1, mask=m, other=0.0).to(tl.float32) * INV
    o1 = x1 * c - x2 * s
    o2 = x1 * s + x2 * c
    tl.store(O + row.to(tl.int64) * OS + off, o1, mask=m)
    tl.store(O + row.to(tl.int64) * OS + off + 1, o2, mask=m)


@triton.jit
def _are_pos(X, F, POS, O, P, H: tl.constexpr, INV, XS: tl.constexpr,
             FS: tl.constexpr, PS: tl.constexpr, OS: tl.constexpr,
             PB: tl.constexpr):
    row = tl.program_id(0)
    p = tl.arange(0, PB)
    m = p < P
    off = 2 * p
    t = row // H
    fr = tl.load(POS + t * PS).to(tl.int32)
    x1 = tl.load(X + row.to(tl.int64) * XS + off, mask=m,
                 other=0.0).to(tl.float32)
    x2 = tl.load(X + row.to(tl.int64) * XS + off + 1, mask=m,
                 other=0.0).to(tl.float32)
    c = tl.load(F + fr * FS + off, mask=m, other=0.0).to(tl.float32)
    s = tl.load(F + fr * FS + off + 1, mask=m, other=0.0).to(tl.float32) * INV
    o1 = x1 * c - x2 * s
    o2 = x1 * s + x2 * c
    tl.store(O + row.to(tl.int64) * OS + off, o1, mask=m)
    tl.store(O + row.to(tl.int64) * OS + off + 1, o2, mask=m)


def apply_rotary_emb(x, freqs_cis, positions=None, inverse=False):
    orig = x.shape
    h = orig[-2]
    d = orig[-1]
    p = d // 2
    x2 = x.reshape(-1, d)
    n = x2.shape[0]
    out = torch.empty_like(x2)
    if n == 0:
        return out.reshape(orig)
    pb = triton.next_power_of_2(p)
    warps = 4 if pb <= 1024 else 8
    inv = -1.0 if inverse else 1.0
    if positions is None:
        _are_nopos[(n,)](
            x2,
            freqs_cis,
            out,
            p,
            h,
            inv,
            x2.stride(0),
            freqs_cis.stride(0),
            out.stride(0),
            pb,
            num_warps=warps,
        )
    else:
        pos = positions.contiguous()
        _are_pos[(n,)](
            x2,
            freqs_cis,
            pos,
            out,
            p,
            h,
            inv,
            x2.stride(0),
            freqs_cis.stride(0),
            pos.stride(0),
            out.stride(0),
            pb,
            num_warps=warps,
        )
    return out.reshape(orig)


__all__ = ["apply_rotary_emb"]