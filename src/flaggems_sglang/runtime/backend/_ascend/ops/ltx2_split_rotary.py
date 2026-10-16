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
    NT: tl.constexpr,
    TB: tl.constexpr,
    T: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    BT: tl.constexpr,
    NP: tl.constexpr,
):
    pid = tl.program_id(0)
    d = tl.arange(0, D)[None, None, :]
    hh = tl.arange(0, H)[None, :, None]
    a0 = tl.arange(0, BT)[:, None, None]
    for task in tl.range(pid, NT, NP):
        tb = task % TB
        b = task // TB
        t = tb * BT + a0
        tc = tl.minimum(t, T - 1)
        xo = ((b * T + tc) * H + hh) * (2 * D) + d
        co = (b * H + hh) * (T * D) + tc * D + d
        a = tl.load(X + xo).to(tl.float32)
        z = tl.load(X + xo + D).to(tl.float32)
        c = tl.load(C + co).to(tl.float32)
        s = tl.load(S + co).to(tl.float32)
        y1 = (a * c).to(tl.bfloat16).to(tl.float32) - z * s
        y2 = (z * c).to(tl.bfloat16).to(tl.float32) + a * s
        m = t < T
        tl.store(O + xo, y1, m)
        tl.store(O + xo + D, y2, m)


def ltx2_split_rotary(x, cos, sin):
    b, h, t, d = cos.shape
    out = torch.empty_like(x)
    if t:
        bt = 8192 // (h * d)
        if bt < 1:
            bt = 1
        if bt > t:
            bt = 1 << (t - 1).bit_length()
        tb = (t + bt - 1) // bt
        nt = b * tb
        np = 40 if nt > 40 else nt
        _rot[(np,)](x, cos, sin, out, nt, tb, t, h, d, BT=bt, NP=np)
    return out


__all__ = ["ltx2_split_rotary"]
