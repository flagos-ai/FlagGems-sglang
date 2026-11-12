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

"""Triton fused_qk_complex_rope (candidate v1)."""

import torch
import triton
import triton.language as tl


@triton.jit
def _qk_rope(X, F, O, P, H: tl.constexpr, XS: tl.constexpr,
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
    s = tl.load(F + t * FS + off + 1, mask=m, other=0.0).to(tl.float32)
    o1 = x1 * c - x2 * s
    o2 = x1 * s + x2 * c
    tl.store(O + row.to(tl.int64) * OS + off, o1, mask=m)
    tl.store(O + row.to(tl.int64) * OS + off + 1, o2, mask=m)


def _rope_rows(x, freqs_cis):
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
    _qk_rope[(n,)](
        x2,
        freqs_cis,
        out,
        p,
        h,
        x2.stride(0),
        freqs_cis.stride(0),
        out.stride(0),
        pb,
        num_warps=4 if pb <= 1024 else 8,
    )
    return out.reshape(orig)


def fused_qk_complex_rope(q, k, freqs_cis):
    return _rope_rows(q, freqs_cis), _rope_rows(k, freqs_cis)


__all__ = ["fused_qk_complex_rope"]