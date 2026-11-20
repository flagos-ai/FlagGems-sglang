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

"""Triton wan_rmsnorm_silu.

x is (B, C, L...): RMS-normalize over the channel axis per spatial
position, scale by ``rms_scale`` (or ``sqrt(C)`` when it is None), apply
per-channel ``gamma`` and optional ``bias``, then SiLU. The output is
the promoted dtype of x and gamma; all math runs in fp32.

One CTA per (batch, position-tile). The channel axis is walked in
``CP``-row column strides that are fully coalesced along L; the
sum-of-squares accumulates in registers during pass 1, and pass 2
re-streams the same tile (L1-resident) to normalize, gate, and write:
1 DRAM read + 1 write.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _wan_rmsnorm_silu_kernel(
    X,
    G,
    BI,
    O,
    L,
    C,
    SCALE,
    EPS,
    HAS_BIAS: tl.constexpr,
    HAS_SCALE: tl.constexpr,
    CP: tl.constexpr,
    BL: tl.constexpr,
):
    b = tl.program_id(0)
    lt = tl.program_id(1)
    l = lt * BL + tl.arange(0, BL)
    ml = l < L

    base = X + b.to(tl.int64) * C * L + l[None, :]
    ss = tl.zeros((BL,), tl.float32)
    for c0 in range(0, C, CP):
        c = c0 + tl.arange(0, CP)
        m = (c[:, None] < C) & ml[None, :]
        x = tl.load(base + c[:, None] * L, mask=m, other=0.0).to(tl.float32)
        ss += tl.sum(x * x, 0)

    scale = SCALE if HAS_SCALE else tl.sqrt(C.to(tl.float32))
    inv = scale * tl.rsqrt(ss / C.to(tl.float32) + EPS)

    ob = O + b.to(tl.int64) * C * L + l[None, :]
    for c0 in range(0, C, CP):
        c = c0 + tl.arange(0, CP)
        mc = c < C
        m = mc[:, None] & ml[None, :]
        x = tl.load(base + c[:, None] * L, mask=m, other=0.0).to(tl.float32)
        g = tl.load(G + c, mask=mc, other=0.0).to(tl.float32)
        y = x * inv[None, :] * g[:, None]
        if HAS_BIAS:
            bv = tl.load(BI + c, mask=mc, other=0.0).to(tl.float32)
            y = y + bv[:, None]
        y = y * tl.sigmoid(y)
        tl.store(ob + c[:, None] * L, y, mask=m)


def wan_rmsnorm_silu(x, gamma, bias, rms_scale, eps):
    b, c = x.shape[0], x.shape[1]
    l = x.numel() // (b * c)
    dtype = torch.result_type(x, gamma)
    out = torch.empty(x.shape, dtype=dtype, device=x.device)
    if b == 0 or c == 0 or l == 0:
        return out

    has_bias = bias is not None
    has_scale = rms_scale is not None
    scale = float(rms_scale) if has_scale else 0.0

    xf = x.contiguous()
    cp = min(1024, max(16, triton.next_power_of_2(c)))
    bl = max(16, 16384 // cp)
    grid = (b, triton.cdiv(l, bl))
    _wan_rmsnorm_silu_kernel[grid](
        xf,
        gamma,
        bias if has_bias else gamma,
        out,
        l,
        c,
        scale,
        float(eps),
        has_bias,
        has_scale,
        cp,
        bl,
        num_warps=4,
    )
    return out


__all__ = ["wan_rmsnorm_silu"]
