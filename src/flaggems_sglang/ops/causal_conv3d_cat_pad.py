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
def _bounded_div(i, D: tl.constexpr, LIMIT: tl.constexpr):
    if D > 1 and D & D - 1 != 0 and (LIMIT < 2147483648):
        M: tl.constexpr = (4294967296 + D - 1) // D
        Q: tl.constexpr = (LIMIT + 1) // D
        EXACT: tl.constexpr = (
            Q == 0 or (Q * D - 1) * M < Q * 4294967296
        ) and LIMIT * M < (LIMIT // D + 1) * 4294967296
        if EXACT:
            return tl.umulhi(i.to(tl.uint32), tl.full((), M, tl.uint32)).to(
                tl.int32
            )
        else:
            return i // D
    else:
        return i // D


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
def _gptr(
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
    BLOCK: tl.constexpr,
):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    w = i % OW - WL
    h = i // OW % OH - HT
    t = i // (OW * OH) % OT - DL
    bc = i // (OW * OH * OT)
    valid = (
        (i < N)
        & (w >= 0)
        & (w < W)
        & (h >= 0)
        & (h < H)
        & (t >= 0)
        & (t < TC + T)
    )
    off = h * W + w
    cache = t < TC
    src = tl.where(
        cache,
        C + (bc * TC + t) * H * W + off,
        X + (bc * T + t - TC) * H * W + off,
    )
    tl.store(Y + i, tl.load(src, valid, other=0), i < N)


@tr.jit
def _pairs(
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
    STREAM: tl.constexpr,
    BLOCK: tl.constexpr,
):
    xp = X.to(tl.pointer_type(tl.uint32))
    cp = C.to(tl.pointer_type(tl.uint32))
    yp = Y.to(tl.pointer_type(tl.uint32))
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    RW: tl.constexpr = OW // 2
    LIMIT: tl.constexpr = tr.cdiv(N, BLOCK) * BLOCK - 1
    row = _bounded_div(i, RW, LIMIT)
    plane = _bounded_div(i, RW * OH, LIMIT)
    bc = _bounded_div(i, RW * OH * OT, LIMIT)
    w = (i - row * RW) * 2 - WL
    h = row - plane * OH - HT
    t = plane - bc * OT - DL
    valid = (i < N) & (h >= 0) & (h < H) & (t >= 0) & (t < TC + T)
    src = tl.where(
        t < TC,
        cp + ((bc * TC + t) * H + h) * (W // 2),
        xp + ((bc * T + t - TC) * H + h) * (W // 2),
    )
    ml = valid & (w >= 0) & (w < W)
    if WL % 2 == 0:
        if STREAM:
            value = tl.load(src + w // 2, ml, other=0, cache_modifier=".cg")
        else:
            value = tl.load(src + w // 2, ml, other=0)
    else:
        mh = valid & (w + 1 >= 0) & (w + 1 < W)
        if STREAM:
            low = tl.load(src + w // 2, ml, other=0, cache_modifier=".cg")
            high = tl.load(
                src + (w + 1) // 2, mh, other=0, cache_modifier=".cg"
            )
        else:
            low = tl.load(src + w // 2, ml, other=0)
            high = tl.load(src + (w + 1) // 2, mh, other=0)
        value = low >> 16 | high << 16
    if STREAM:
        tl.store(yp + i, value, i < N, cache_modifier=".cs")
    else:
        tl.store(yp + i, value, i < N)


def causal_conv3d_cat_pad(x, cache_x, padding):
    b, c, t, h, w = x.shape
    tc = cache_x.shape[2]
    wl, wr, ht, hb, front, back = padding
    ot = t + front + back
    oh = h + ht + hb
    ow = w + wl + wr
    xd = x.dtype
    cd = cache_x.dtype
    same = xd == cd
    out = torch.empty(
        (b, c, ot, oh, ow),
        dtype=xd if same else torch.promote_types(xd, cd),
        device=x.device,
    )
    n = b * c * ot * oh * ow
    if not n:
        return out
    dl = front - tc
    if same and x.is_contiguous() and cache_x.is_contiguous():
        if (xd == torch.bfloat16 or xd == torch.float16) and (
            not (
                w & 1
                or ow & 1
                or x.storage_offset() & 1
                or cache_x.storage_offset() & 1
            )
        ):
            m = n >> 1
            _pairs.run(
                x,
                cache_x,
                out,
                m,
                t,
                tc,
                h,
                w,
                ot,
                oh,
                ow,
                wl,
                ht,
                dl,
                torch.version.hip is not None,
                512,
                grid=(m + 511 >> 9,),
                warmup=False,
            )
            return out
        _gptr.run(
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
            dl,
            1024,
            grid=(n + 1023 >> 10,),
            warmup=False,
        )
        return out
    _pad.run(
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
        dl,
        x.stride(),
        cache_x.stride(),
        512,
        grid=(n + 511 >> 9,),
        warmup=False,
    )
    return out


__all__ = ["causal_conv3d_cat_pad"]
