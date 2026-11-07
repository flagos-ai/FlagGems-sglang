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
import triton as tr
import triton.language as tl


@tr.jit
def _pad(
    X,
    C,
    Y,
    N: tl.constexpr,
    CH: tl.constexpr,
    T: tl.constexpr,
    TC: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    OT: tl.constexpr,
    OH: tl.constexpr,
    OW: tl.constexpr,
    WL: tl.constexpr,
    HT: tl.constexpr,
    DL: tl.constexpr,
    XS: tl.constexpr,
    CS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    w = i % OW - WL
    h = i // OW % OH - HT
    t = i // (OW * OH) % OT - DL
    c = i // (OW * OH * OT) % CH
    b = i // (OW * OH * OT * CH)
    valid = (i < N) & (w >= 0) & (w < W) & (h >= 0) & (h < H)
    a = tl.load(
        X + b * XS[0] + c * XS[1] + (t - TC) * XS[2] + h * XS[3] + w * XS[4],
        valid & (t >= TC) & (t < TC + T),
        other=0,
    ).to(Y.dtype.element_ty)
    if TC > 0:
        z = tl.load(
            C + b * CS[0] + c * CS[1] + t * CS[2] + h * CS[3] + w * CS[4],
            valid & (t >= 0) & (t < TC),
            other=0,
        ).to(Y.dtype.element_ty)
        a = tl.where(t < TC, z, a)
    tl.store(Y + i, a, i < N)


@tr.jit
def _padc(
    X,
    C,
    Y,
    N: tl.constexpr,
    T: tl.constexpr,
    TC: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    OT: tl.constexpr,
    OH: tl.constexpr,
    OW: tl.constexpr,
    WL: tl.constexpr,
    HT: tl.constexpr,
    DL: tl.constexpr,
    BLOCK: tl.constexpr,
):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    w = i % OW - WL
    h = i // OW % OH - HT
    t = i // (OW * OH) % OT - DL
    bc = i // (OW * OH * OT)
    valid = (i < N) & (w >= 0) & (w < W) & (h >= 0) & (h < H)
    off = h * W + w
    a = tl.load(
        X + (bc * T + t - TC) * H * W + off,
        valid & (t >= TC) & (t < TC + T),
        other=0,
    ).to(Y.dtype.element_ty)
    if TC > 0:
        z = tl.load(
            C + (bc * TC + t) * H * W + off,
            valid & (t >= 0) & (t < TC),
            other=0,
        ).to(Y.dtype.element_ty)
        a = tl.where(t < TC, z, a)
    tl.store(Y + i, a, i < N)


def causal_conv3d_cat_pad(x, cache_x, padding):
    b, c, t, h, w = x.shape
    tc = cache_x.shape[2]
    wl, wr, ht, hb, front, back = padding
    ot, oh, ow = (t + front + back, h + ht + hb, w + wl + wr)
    out = torch.empty(
        (b, c, ot, oh, ow),
        dtype=torch.promote_types(x.dtype, cache_x.dtype),
        device=x.device,
    )
    n = out.numel()
    if not n:
        return out
    if x.is_contiguous() and cache_x.is_contiguous():
        _padc[tr.cdiv(n, 2048),](
            x,
            cache_x,
            out,
            n,
            t,
            tc,
            h,
            w,
            ot,
            oh,
            ow,
            wl,
            ht,
            front - tc,
            2048,
        )
        return out
    _pad[tr.cdiv(n, 512),](
        x,
        cache_x,
        out,
        n,
        c,
        t,
        tc,
        h,
        w,
        ot,
        oh,
        ow,
        wl,
        ht,
        front - tc,
        x.stride(),
        cache_x.stride(),
        512,
    )
    return out


__all__ = ["causal_conv3d_cat_pad"]
