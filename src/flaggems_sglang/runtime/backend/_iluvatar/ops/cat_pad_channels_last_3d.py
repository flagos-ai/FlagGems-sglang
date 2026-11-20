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
def _source_rows(X, CX, Y, Z, G: tl.constexpr, BLOCK: tl.constexpr):
    B: tl.constexpr = G[0]
    C: tl.constexpr = G[1]
    T: tl.constexpr = G[2]
    H: tl.constexpr = G[3]
    W: tl.constexpr = G[4]
    TC: tl.constexpr = G[5]
    WL: tl.constexpr = G[6][0]
    WR: tl.constexpr = G[6][1]
    HT: tl.constexpr = G[6][2]
    HB: tl.constexpr = G[6][3]
    TF: tl.constexpr = G[6][4]
    TB: tl.constexpr = G[6][5]
    K: tl.constexpr = G[7]
    FRONT: tl.constexpr = TF - TC if TF > TC else 0
    OT: tl.constexpr = TC + T + FRONT + TB
    OH: tl.constexpr = H + HT + HB
    SW: tl.constexpr = W * C
    RW: tl.constexpr = (W + WL + WR) * C
    WLC: tl.constexpr = WL * C
    NCH: tl.constexpr = tr.cdiv(RW, BLOCK)
    COPY: tl.constexpr = B * (TC + T) * H * NCH
    ZROWS: tl.constexpr = (FRONT + TB) * OH + (TC + T) * (HT + HB)
    pid = tl.program_id(0)
    a = tl.arange(0, BLOCK)
    if pid < COPY:
        r = pid // NCH
        i = pid % NCH * BLOCK + a
        h = r % H
        f = r // H
        t = f % (TC + T)
        b = f // (TC + T)
        j = tl.where(i < SW, i + WLC, tl.where(i < SW + WLC, i - SW, i))
        if t < TC:
            v = tl.load(CX + ((b * TC + t) * H + h) * SW + i, i < SW, other=0)
        else:
            v = tl.load(
                X + ((b * T + t - TC) * H + h) * SW + i, i < SW, other=0
            )
        tl.store(Y + ((b * OT + t + FRONT) * OH + h + HT) * RW + j, v, i < RW)
        if K > 0:
            tl.store(
                Z + ((b * K + t + FRONT - OT + K) * H + h) * SW + i,
                v,
                (i < SW) & (t + FRONT >= OT - K),
            )
    else:
        r = (pid - COPY) // NCH
        i = (pid - COPY) % NCH * BLOCK + a
        b = r // ZROWS
        q = r % ZROWS
        if q < FRONT * OH:
            t = q // OH
            h = q % OH
        elif q < FRONT * OH + (TC + T) * (HT + HB):
            if HT + HB > 0:
                s = q - FRONT * OH
                t = s // (HT + HB) + FRONT
                h0 = s % (HT + HB)
                h = tl.where(h0 < HT, h0, h0 + H)
            else:
                t = 0
                h = 0
        else:
            s = q - FRONT * OH - (TC + T) * (HT + HB)
            t = s // OH + FRONT + TC + T
            h = s % OH
        tl.store(Y + ((b * OT + t) * OH + h) * RW + i, 0, i < RW)
        if K > 0:
            j = i - WLC
            tl.store(
                Z + ((b * K + t - OT + K) * H + h - HT) * SW + j,
                0,
                (j >= 0) & (j < SW) & (t >= OT - K) & (h >= HT) & (h < HT + H),
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
    front = tf - tc if tf > tc else 0
    ot = tc + t + front + tb
    oh = h + ht + hb
    ow = w + wl + wr
    out = torch.empty_strided(
        (b, c, ot, oh, ow),
        (ot * oh * ow * c, 1, oh * ow * c, ow * c, c),
        dtype=dtype,
        device=x.device,
    )
    if keep_cache_t > 0:
        k = min(keep_cache_t, ot)
        h0, h1, _ = slice(ht, ht + h).indices(oh)
        w0, w1, _ = slice(wl, wl + w).indices(ow)
        kh = max(h1 - h0, 0)
        kw = max(w1 - w0, 0)
        z = torch.empty_strided(
            (b, c, k, kh, kw),
            (k * kh * kw * c, 1, kh * kw * c, kw * c, c),
            dtype=dtype,
            device=x.device,
        )
    else:
        k = kh = kw = h0 = w0 = 0
        z = out
    n = b * ot * oh * ow * c
    if n > 0:
        if fast and min(wl, wr, ht, hb, tb) >= 0 and h > 0 and t + tc > 0:
            block = min(1 << (ow * c - 1).bit_length(), 512)
            nch = -(-(ow * c) // block)
            _source_rows.run(
                x,
                cx,
                out,
                z,
                (b, c, t, h, w, tc, padding, k),
                block,
                grid=(b * ot * oh * nch,),
                warmup=False,
            )
        else:
            _general[(-(-n // 1024),)](
                x,
                cx,
                out,
                z,
                (
                    tc,
                    t,
                    h,
                    w,
                    c,
                    ot,
                    oh,
                    ow,
                    front,
                    ht,
                    wl,
                    n,
                    k,
                    kh,
                    kw,
                    h0,
                    w0,
                ),
                x.stride(),
                cx.stride(),
                1024,
            )
    return (out, z) if keep_cache_t > 0 else out


__all__ = ["cat_pad_channels_last_3d"]
