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
def _rotary(
    X,
    C,
    S,
    O,
    N: tl.constexpr,
    T: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    X0: tl.constexpr,
    X1: tl.constexpr,
    X2: tl.constexpr,
    C0: tl.constexpr,
    C1: tl.constexpr,
    C2: tl.constexpr,
    C3: tl.constexpr,
    S0: tl.constexpr,
    S1: tl.constexpr,
    S2: tl.constexpr,
    S3: tl.constexpr,
    BLOCK: tl.constexpr,
):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    d = i % D
    h = i // D % H
    t = i // (D * H) % T
    b = i // (D * H * T)
    xi = b * X0 + t * X1 + (h * 2 * D + d) * X2
    ci = b * C0 + h * C1 + t * C2 + d * C3
    si = b * S0 + h * S1 + t * S2 + d * S3
    a = tl.load(X + xi, i < N, other=0, eviction_policy="evict_last").to(
        tl.float32
    )
    z = tl.load(
        X + xi + D * X2, i < N, other=0, eviction_policy="evict_last"
    ).to(tl.float32)
    c = tl.load(C + ci, i < N, other=0, eviction_policy="evict_last").to(
        tl.float32
    )
    s = tl.load(S + si, i < N, other=0, eviction_policy="evict_last").to(
        tl.float32
    )
    ac = (a * c).to(tl.bfloat16).to(tl.float32)
    zc = (z * c).to(tl.bfloat16).to(tl.float32)
    y1 = ac - z * s
    y2 = zc + a * s
    oi = (i // D) * (2 * D) + d
    tl.store(O + oi, y1, i < N, eviction_policy="evict_first")
    tl.store(O + oi + D, y2, i < N, eviction_policy="evict_first")


def ltx2_split_rotary(x, cos, sin):
    b, t, inner = x.shape
    h = cos.shape[1]
    d = cos.shape[3]
    out = torch.empty(x.shape, dtype=x.dtype, device=x.device)
    n = b * t * h * d
    if n:
        _rotary[(triton.cdiv(n, 512),)](
            x,
            cos,
            sin,
            out,
            n,
            t,
            h,
            d,
            *x.stride(),
            *cos.stride(),
            *sin.stride(),
            BLOCK=512,
            enable_fp_fusion=False,
        )
    return out


__all__ = ["ltx2_split_rotary"]
