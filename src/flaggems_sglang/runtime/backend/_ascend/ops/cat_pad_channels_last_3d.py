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
def _rows(
    X, CX, Y, Z, G: tl.constexpr, BLOCK: tl.constexpr, BLOCKZ: tl.constexpr
):
    B: tl.constexpr = G[0]
    TC: tl.constexpr = G[1]
    T: tl.constexpr = G[2]
    H: tl.constexpr = G[3]
    OT: tl.constexpr = G[4]
    OH: tl.constexpr = G[5]
    FRONT: tl.constexpr = G[6]
    TB: tl.constexpr = G[7]
    HT: tl.constexpr = G[8]
    RW: tl.constexpr = G[9]
    SW: tl.constexpr = G[10]
    WLC: tl.constexpr = G[11]
    NCH: tl.constexpr = G[12]
    K: tl.constexpr = G[13]
    KH: tl.constexpr = G[14]
    KWC: tl.constexpr = G[15]
    H0: tl.constexpr = G[16]
    J0: tl.constexpr = G[17]
    NCHZ: tl.constexpr = G[18]
    ZN0: tl.constexpr = G[19]
    ZF0: tl.constexpr = G[20]
    ZN1: tl.constexpr = G[21]
    ZF1: tl.constexpr = G[22]
    ZN2: tl.constexpr = G[23]
    ZF2: tl.constexpr = G[24]
    ZN3: tl.constexpr = G[25]
    ZF3: tl.constexpr = G[26]
    P0 = tl.program_id(0)
    NP = tl.num_programs(0)
    if FRONT > 0:
        for p in range(P0, B * FRONT * OH * NCH, NP):
            row = p // NCH
            i = p % NCH * BLOCK + tl.arange(0, BLOCK)
            h = row % OH
            fr = row // OH
            tt = fr % FRONT
            bb = fr // FRONT
            v = tl.zeros([BLOCK], dtype=Y.dtype.element_ty)

            tl.store(Y + ((bb * OT + tt + (0)) * OH + h) * RW + i, v, i < RW)
    if ZN0 > 0:
        for q in range(P0, B * ZN0 * KH * NCHZ, NP):
            zrow = q // NCHZ
            ii = q % NCHZ * BLOCKZ + tl.arange(0, BLOCKZ)
            hh = zrow % KH
            zfr = zrow // KH
            zt = zfr % ZN0
            bz = zfr // ZN0
            sz = H0 + hh - HT
            jj = J0 + ii
            u = tl.zeros([BLOCKZ], dtype=Y.dtype.element_ty)

            tl.store(
                Z + ((bz * K + zt + (ZF0) - OT + K) * KH + hh) * KWC + ii,
                u,
                ii < KWC,
            )
    if TC > 0:
        for p in range(P0, B * TC * OH * NCH, NP):
            row = p // NCH
            i = p % NCH * BLOCK + tl.arange(0, BLOCK)
            h = row % OH
            fr = row // OH
            tt = fr % TC
            bb = fr // TC
            sh = h - HT
            j = i - WLC
            v = tl.load(
                CX + ((bb * TC + tt) * H + sh) * SW + j,
                (i < RW) & (j >= 0) & (j < SW) & (sh >= 0) & (sh < H),
                other=0,
            )

            tl.store(
                Y + ((bb * OT + tt + (FRONT)) * OH + h) * RW + i, v, i < RW
            )
    if ZN1 > 0:
        for q in range(P0, B * ZN1 * KH * NCHZ, NP):
            zrow = q // NCHZ
            ii = q % NCHZ * BLOCKZ + tl.arange(0, BLOCKZ)
            hh = zrow % KH
            zfr = zrow // KH
            zt = zfr % ZN1
            bz = zfr // ZN1
            sz = H0 + hh - HT
            jj = J0 + ii
            u = tl.load(
                CX + ((bz * TC + zt + (ZF1) - (FRONT)) * H + sz) * SW + jj,
                (ii < KWC) & (jj >= 0) & (jj < SW) & (sz >= 0) & (sz < H),
                other=0,
            )

            tl.store(
                Z + ((bz * K + zt + (ZF1) - OT + K) * KH + hh) * KWC + ii,
                u,
                ii < KWC,
            )
    if T > 0:
        for p in range(P0, B * T * OH * NCH, NP):
            row = p // NCH
            i = p % NCH * BLOCK + tl.arange(0, BLOCK)
            h = row % OH
            fr = row // OH
            tt = fr % T
            bb = fr // T
            sh = h - HT
            j = i - WLC
            v = tl.load(
                X + ((bb * T + tt) * H + sh) * SW + j,
                (i < RW) & (j >= 0) & (j < SW) & (sh >= 0) & (sh < H),
                other=0,
            )

            tl.store(
                Y + ((bb * OT + tt + (FRONT + TC)) * OH + h) * RW + i,
                v,
                i < RW,
            )
    if ZN2 > 0:
        for q in range(P0, B * ZN2 * KH * NCHZ, NP):
            zrow = q // NCHZ
            ii = q % NCHZ * BLOCKZ + tl.arange(0, BLOCKZ)
            hh = zrow % KH
            zfr = zrow // KH
            zt = zfr % ZN2
            bz = zfr // ZN2
            sz = H0 + hh - HT
            jj = J0 + ii
            u = tl.load(
                X + ((bz * T + zt + (ZF2) - (FRONT + TC)) * H + sz) * SW + jj,
                (ii < KWC) & (jj >= 0) & (jj < SW) & (sz >= 0) & (sz < H),
                other=0,
            )

            tl.store(
                Z + ((bz * K + zt + (ZF2) - OT + K) * KH + hh) * KWC + ii,
                u,
                ii < KWC,
            )
    if TB > 0:
        for p in range(P0, B * TB * OH * NCH, NP):
            row = p // NCH
            i = p % NCH * BLOCK + tl.arange(0, BLOCK)
            h = row % OH
            fr = row // OH
            tt = fr % TB
            bb = fr // TB
            v = tl.zeros([BLOCK], dtype=Y.dtype.element_ty)

            tl.store(
                Y + ((bb * OT + tt + (FRONT + TC + T)) * OH + h) * RW + i,
                v,
                i < RW,
            )
    if ZN3 > 0:
        for q in range(P0, B * ZN3 * KH * NCHZ, NP):
            zrow = q // NCHZ
            ii = q % NCHZ * BLOCKZ + tl.arange(0, BLOCKZ)
            hh = zrow % KH
            zfr = zrow // KH
            zt = zfr % ZN3
            bz = zfr // ZN3
            sz = H0 + hh - HT
            jj = J0 + ii
            u = tl.zeros([BLOCKZ], dtype=Y.dtype.element_ty)

            tl.store(
                Z + ((bz * K + zt + (ZF3) - OT + K) * KH + hh) * KWC + ii,
                u,
                ii < KWC,
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
    inner = keep & (sw >= 0) & (sw < W) & (sh >= 0) & (sh < H)
    v = tl.load(
        X
        + b * XS[0]
        + c * XS[1]
        + (st - TC) * XS[2]
        + sh * XS[3]
        + sw * XS[4],
        inner & (st >= TC) & (st < TC + T),
        other=0,
    ).to(Y.dtype.element_ty)
    if TC > 0:
        u = tl.load(
            CX + b * CS[0] + c * CS[1] + st * CS[2] + sh * CS[3] + sw * CS[4],
            inner & (st >= 0) & (st < TC),
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
    b, c, t, h, w = x.size()
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
        out = torch.empty_strided(
            (b, c, ot, oh, ow),
            (ot * oh * ow * c, 1, oh * ow * c, ow * c, c),
            dtype=dtype,
            device=x.device,
        )
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
            z = torch.empty_strided(
                (b, c, k, kh, kw),
                (k * kh * kw * c, 1, kh * kw * c, kw * c, c),
                dtype=dtype,
                device=x.device,
            )
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
    if fast and tb >= 0:
        rw = ow * c
        block = 1 << (rw - 1).bit_length()
        if block > 2048:
            block = 2048
        nch = -(-rw // block)
        kwc = kw * c
        if kwc > 0:
            blockz = 1 << kwc.bit_length()
            if blockz > 2048:
                blockz = 2048
            nchz = -(-kwc // blockz)
        else:
            blockz = 1
            nchz = 0
        zspan = []
        base = 0
        for nf in (front, tc, t, tb):
            lo = base if base > ot - k else ot - k
            hi = base + nf if base + nf < ot else ot
            zspan.append(hi - lo if hi > lo and k > 0 else 0)
            zspan.append(lo)
            base += nf
        work = nrow * nch
        if k > 0:
            wz = b * k * kh * nchz
            if wz > work:
                work = wz
        _rows.run(
            x,
            cx,
            out,
            z,
            (
                b,
                tc,
                t,
                h,
                ot,
                oh,
                front,
                tb,
                ht,
                rw,
                wc,
                wl * c,
                nch,
                k,
                kh,
                kwc,
                h0,
                w0 * c - wl * c,
                nchz,
            )
            + tuple(zspan),
            block,
            blockz,
            grid=(min(work, 40),),
            warmup=False,
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
