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
    u = z * (2.302208198144325 + 0.1029432395800235 * z * z)
    return z / (1.0 + tl.exp2(-u))


@triton.jit
def _t(X, BIAS, Y, H: tl.constexpr, BN: tl.constexpr, T: tl.constexpr):
    nb = H // BN
    lane = tl.arange(0, BN)
    for t in range(tl.program_id(0), T, tl.num_programs(0)):
        row = t // nb
        c = (t - row * nb) * BN + lane
        o = row * H + c
        s = tl.load(X + o).to(tl.float32) + tl.load(BIAS + c).to(tl.float32)
        z = s.to(X.dtype.element_ty).to(tl.float32)
        tl.store(Y + o, _g(z).to(X.dtype.element_ty))


def bias_gelu_tanh(x, bias):
    y = torch.empty_like(x)
    h = x.shape[-1]
    n = x.numel()
    m = n // h
    bn = h & -h
    if bn > 4096:
        bn = 4096
    t = m * (h // bn)
    g = t if t < 32 else 32
    _t[(g,)](x, bias, y, h, bn, t)
    return y


__all__ = ["bias_gelu_tanh"]
