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

BLK = 512


@triton.jit
def _rot(
    X,
    C,
    S,
    O,
    N: tl.constexpr,
    T: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    BLOCK: tl.constexpr,
):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = i < N
    q = i // D
    d = i % D
    h = q % H
    row = q // H
    b = row // T
    t = row % T
    xo = q * (2 * D) + d
    co = ((b * H + h) * T + t) * D + d
    a = tl.load(X + xo, m, eviction_policy="evict_last").to(tl.float32)
    z = tl.load(X + xo + D, m, eviction_policy="evict_last").to(tl.float32)
    c = tl.load(C + co, m, eviction_policy="evict_last").to(tl.float32)
    s = tl.load(S + co, m, eviction_policy="evict_last").to(tl.float32)
    y1 = (a * c).to(tl.bfloat16).to(tl.float32) - z * s
    y2 = (z * c).to(tl.bfloat16).to(tl.float32) + a * s
    tl.store(O + xo, y1, m, eviction_policy="evict_first")
    tl.store(O + xo + D, y2, m, eviction_policy="evict_first")


def ltx2_split_rotary(x, cos, sin):
    b, h, t, d = cos.shape
    out = torch.empty_like(x)
    n = b * t * h * d
    if n:
        _rot[((n + BLK - 1) // BLK,)](
            x, cos, sin, out, n, t, h, d, BLOCK=BLK, num_warps=2
        )
    return out


__all__ = ["ltx2_split_rotary"]
