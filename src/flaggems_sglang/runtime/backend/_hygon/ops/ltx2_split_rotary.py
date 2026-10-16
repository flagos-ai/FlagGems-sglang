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
def _rot(
    X,
    C,
    S,
    O,
    T: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    BT: tl.constexpr,
):
    b = tl.program_id(0)
    h = tl.program_id(2)
    t = tl.program_id(1) * BT + tl.arange(0, BT)[:, None]
    d = tl.arange(0, D)[None, :]
    m = t < T
    xo = ((b * T + t) * H + h) * (2 * D) + d
    co = (b * H + h) * (T * D) + t * D + d
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
    if t:
        _rot[(b, (t + 31) // 32, h)](x, cos, sin, out, t, h, d, BT=32)
    return out


__all__ = ["ltx2_split_rotary"]
