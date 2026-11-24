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
def _rows(X, CX, Y, Z, G: tl.constexpr, BLOCK: tl.constexpr):
    TC: tl.constexpr = G[0]
    T: tl.constexpr = G[1]
    H: tl.constexpr = G[2]
    OT: tl.constexpr = G[3]
    OH: tl.constexpr = G[4]
    FRONT: tl.constexpr = G[5]
    HT: tl.constexpr = G[6]
    RW: tl.constexpr = G[7]
    SW: tl.constexpr = G[8]
    WLC: tl.constexpr = G[9]
    K: tl.constexpr = G[11]
    KH: tl.constexpr = G[12]
    KWC: tl.constexpr = G[13]
    H0: tl.constexpr = G[14]
    W0C: tl.constexpr = G[15]
    WORK: tl.constexpr = G[16]
    XP = X
    CP = CX
    YP = Y
    ZP = Z
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row = i // RW
    r = i - row * RW
    h = row % OH
    f = row // OH
    t = f % OT
    b = f // OT
    sh = h - HT
    st = t - FRONT
    j = r - WLC
    keep = i < WORK
    m = (
        keep
        & (j >= 0)
        & (j < SW)
        & (sh >= 0)
        & (sh < H)
        & (st >= 0)
        & (st < TC + T)
    )
    src = tl.where(
        st < TC,
        CP + ((b * TC + st) * H + sh) * SW,
        XP + ((b * T + st - TC) * H + sh) * SW,
    )
    v = tl.load(src + j, m, other=0).to(Y.dtype.element_ty)
    tl.store(YP + i, v, keep)
    if K > 0:
        ci = r - W0C
        tl.store(
            ZP + ((b * K + t - OT + K) * KH + h - H0) * KWC + ci,
            v,
            keep
            & (ci >= 0)
            & (ci < KWC)
            & (t >= OT - K)
            & (h >= H0)
            & (h < H0 + KH),
        )


@tr.jit
def _general(
    X,
    CX,
    Y,
    Z,
    G: tl.constexpr,
    XS: tl.constexpr,
    CS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    TC: tl.constexpr = G[0]
    T: tl.constexpr = G[1]
    H: tl.constexpr = G[2]
    W: tl.constexpr = G[3]
    C: tl.constexpr = G[4]
    OT: tl.constexpr = G[5]
    OH: tl.constexpr = G[6]
    OW: tl.constexpr = G[7]
    FRONT: tl.constexpr = G[8]
    HT: tl.constexpr = G[9]
    WL: tl.constexpr = G[10]
    N: tl.constexpr = G[11]
    K: tl.constexpr = G[12]
    KH: tl.constexpr = G[13]
    KW: tl.constexpr = G[14]
    H0: tl.constexpr = G[15]
    W0: tl.constexpr = G[16]
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    c = i % C
    w = i // C % OW
    h = i // (C * OW) % OH
    f = i // (C * OW * OH)
    t = f % OT
    b = f // OT
    sh = h - HT
    sw = w - WL
    st = t - FRONT
    keep = i < N
    m = (
        keep
        & (sw >= 0)
        & (sw < W)
        & (sh >= 0)
        & (sh < H)
        & (st >= 0)
        & (st < TC + T)
    )
    v = tl.load(
        X
        + b * XS[0]
        + c * XS[1]
        + (st - TC) * XS[2]
        + sh * XS[3]
        + sw * XS[4],
        m & (st >= TC) & (st < TC + T),
        other=0,
    ).to(Y.dtype.element_ty)
    if TC > 0:
        u = tl.load(
            CX + b * CS[0] + c * CS[1] + st * CS[2] + sh * CS[3] + sw * CS[4],
            m & (st >= 0) & (st < TC),
            other=0,
        ).to(Y.dtype.element_ty)
        v = tl.where(st < TC, u, v)
    tl.store(Y + i, v, keep)
    if K > 0:
        ch = h - H0
        cw = w - W0
        tl.store(
            Z + ((((b * K + t - OT + K) * KH + ch) * KW + cw) * C + c),
            v,
            keep
            & (ch >= 0)
            & (ch < KH)
            & (cw >= 0)
            & (cw < KW)
            & (t >= OT - K),
        )


def cat_pad_channels_last_3d(x, cache_x, padding, keep_cache_t):
    b, c, t, h, w = x.shape
    wl, wr, ht, hb, tf, tb = padding
    if wl != wr or ht != hb or tb != 0:
        return None
    wc = w * c
    hwc = h * wc
    fast = x.stride() == (t * hwc, 1, hwc, wc, c)
    dtype = x.dtype
    if cache_x is None:
        tc = 0
        cx = x
    else:
        tc = cache_x.shape[2]
        cx = cache_x
        if cache_x.dtype != dtype or cache_x.stride() != (
            tc * hwc,
            1,
            hwc,
            wc,
            c,
        ):
            fast = False
            dtype = torch.promote_types(dtype, cache_x.dtype)
    front = tf - tc
    if front < 0:
        front = 0
    ot = tc + t + front + tb
    oh = h + ht + hb
    ow = w + wl + wr
    if fast:
        out = x.new_empty((b, ot, oh, ow, c)).permute(0, 4, 1, 2, 3)
    else:
        out = torch.empty(
            (b, ot, oh, ow, c), dtype=dtype, device=x.device
        ).permute(0, 4, 1, 2, 3)
    if keep_cache_t > 0:
        k = keep_cache_t if keep_cache_t < ot else ot
        h0, h1, _ = slice(ht, ht + h).indices(oh)
        w0, w1, _ = slice(wl, wl + w).indices(ow)
        kh = h1 - h0 if h1 > h0 else 0
        kw = w1 - w0 if w1 > w0 else 0
        if fast:
            z = x.new_empty((b, k, kh, kw, c)).permute(0, 4, 1, 2, 3)
        else:
            z = torch.empty(
                (b, k, kh, kw, c), dtype=dtype, device=x.device
            ).permute(0, 4, 1, 2, 3)
    else:
        k = 0
        kh = 0
        kw = 0
        h0 = 0
        w0 = 0
        z = out
    nrow = b * ot * oh
    if nrow < 1 or ow < 1 or c < 1:
        return (out, z) if keep_cache_t > 0 else out
    if fast:
        rw = ow * c
        work = nrow * rw
        block = 1 << (work - 1).bit_length()
        if block > 2048:
            block = 2048
        nch = 0
        _rows[(-(-work // block),)](
            x,
            cx,
            out,
            z,
            (
                tc,
                t,
                h,
                ot,
                oh,
                front,
                ht,
                rw,
                wc,
                wl * c,
                nch,
                k,
                kh,
                kw * c,
                h0,
                w0 * c,
                work,
            ),
            block,
        )
    else:
        n = nrow * ow * c
        _general[(-(-n // 1024),)](
            x,
            cx,
            out,
            z,
            (tc, t, h, w, c, ot, oh, ow, front, ht, wl, n, k, kh, kw, h0, w0),
            x.stride(),
            cx.stride(),
            1024,
        )
    return (out, z) if keep_cache_t > 0 else out


__all__ = ["cat_pad_channels_last_3d"]
