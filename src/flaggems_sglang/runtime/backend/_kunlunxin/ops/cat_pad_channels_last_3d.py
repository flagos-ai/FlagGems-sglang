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
def _uniform_tile(
    X,
    CX,
    Y,
    Z,
    G: tl.constexpr,
    S: tl.constexpr,
    PB: tl.constexpr,
    EB: tl.constexpr,
    PACK: tl.constexpr,
    R: tl.constexpr,
):
    if PACK:
        X = X.to(tl.pointer_type(tl.uint32))
        CX = CX.to(tl.pointer_type(tl.uint32))
        Y = Y.to(tl.pointer_type(tl.uint32))
        Z = Z.to(tl.pointer_type(tl.uint32))
    TC: tl.constexpr = G[0]
    T: tl.constexpr = G[1]
    H: tl.constexpr = G[2]
    OT: tl.constexpr = G[3]
    OH: tl.constexpr = G[4]
    RW: tl.constexpr = G[5]
    FRONT: tl.constexpr = G[6]
    HT: tl.constexpr = G[7]
    LEFT: tl.constexpr = G[8]
    RIGHT: tl.constexpr = G[9]
    K: tl.constexpr = G[10]
    D: tl.constexpr = G[11]
    M: tl.constexpr = G[12]
    E: tl.constexpr = G[13]
    Q: tl.constexpr = G[14]
    NZ: tl.constexpr = G[15]
    ZF: tl.constexpr = G[16]
    ZT: tl.constexpr = G[17]
    FRAME: tl.constexpr = OH * RW
    DF: tl.constexpr = TC + T
    TB: tl.constexpr = OT - FRONT - DF
    BS: tl.constexpr = R * S
    NC: tl.constexpr = tr.cdiv(FRAME, BS)
    NE: tl.constexpr = tr.cdiv(2 * EB, BS)
    pid = tl.program_id(0)
    b = tl.program_id(1)
    i = tl.arange(0, BS)
    if pid < D:
        h = pid % (H // R) * R
        f = pid // (H // R)
        st = f
        tt = st + FRONT
        dst = (
            Y
            + ((b * OT + tt) * OH + HT + h) * RW
            + LEFT
            + (i // S) * RW
            + i % S
        )
        src_c = CX + ((b * TC + st) * H + h) * S + i
        src_x = X + ((b * T + st - TC) * H + h) * S + i
        value = tl.load(tl.where(st < TC, src_c, src_x))
        tl.store(dst, value)
        if K > 0:
            if tt >= OT - K:
                tl.store(Z + ((b * K + tt - OT + K) * H + h) * S + i, value)
        if LEFT + RIGHT > 0 and H > 1:
            for r in tl.static_range(R):
                if h + r < H - 1:
                    base = ((b * OT + tt) * OH + HT + h + r) * RW + LEFT + S
                    tl.store(Y + base + i, 0, i < LEFT + RIGHT)
    elif pid < E:
        p = pid - M
        chunk = p % NE
        edge = p // NE
        head = HT * RW + LEFT
        tail = (OH - HT - H) * RW + RIGHT
        length = tl.where(
            edge == 0, head, tl.where(edge == DF, tail, head + tail)
        )
        start = (b * OT + FRONT + edge) * FRAME - tl.where(edge == 0, 0, tail)
        j = chunk * BS + i
        tl.store(Y + start + j, 0, j < length)
    elif pid < Q:
        if FRONT + TB > 0:
            p = pid - E
            j = p % NC * BS + i
            f = p // NC
            zt = f
            tt = tl.where(zt < FRONT, zt, zt + DF)
            tl.store(Y + (b * OT + tt) * FRAME + j, 0, j < FRAME)
    else:
        if NZ > 0:
            zi = (pid - Q) * BS + i
            inner = zi % (H * S)
            zframe = zi // (H * S)
            zz = zframe
            zbatch = b
            kk = tl.where(zz < ZF, zz, zz + K - ZF - ZT)
            tl.store(Z + (zbatch * K + kk) * H * S + inner, 0, zi < NZ)


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
    v = tl.zeros([BLOCK], dtype=Y.dtype.element_ty)
    if H > 0 and W > 0:
        bc = tl.minimum(b, N // (C * OW * OH * OT) - 1)
        hc = tl.minimum(tl.maximum(sh, 0), H - 1)
        wc = tl.minimum(tl.maximum(sw, 0), W - 1)
        if T > 0:
            tx = tl.minimum(tl.maximum(st - TC, 0), T - 1)
            vx = tl.load(
                X
                + bc * XS[0]
                + c * XS[1]
                + tx * XS[2]
                + hc * XS[3]
                + wc * XS[4]
            ).to(Y.dtype.element_ty)
            v = tl.where(m & (st >= TC), vx, v)
        if TC > 0:
            tc = tl.minimum(tl.maximum(st, 0), TC - 1)
            vc = tl.load(
                CX
                + bc * CS[0]
                + c * CS[1]
                + tc * CS[2]
                + hc * CS[3]
                + wc * CS[4]
            ).to(Y.dtype.element_ty)
            v = tl.where(m & (st < TC), vc, v)
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
    fast = (
        x.stride() == (t * hwc, 1, hwc, wc, c)
        and wl >= 0
        and wr >= 0
        and ht >= 0
        and hb >= 0
        and tb >= 0
    )
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
    if dtype == x.dtype:
        out = x.new_empty((b, ot, oh, ow, c)).permute(0, 4, 1, 2, 3)
    else:
        out = torch.empty(
            (b, ot, oh, ow, c), dtype=dtype, device=x.device
        ).permute(0, 4, 1, 2, 3)
    k = 0
    kh = 0
    kw = 0
    h0 = 0
    w0 = 0
    z = out
    if keep_cache_t > 0:
        k = keep_cache_t if keep_cache_t < ot else ot
        h0, h1, _ = slice(ht, ht + h).indices(oh)
        w0, w1, _ = slice(wl, wl + w).indices(ow)
        kh = h1 - h0 if h1 > h0 else 0
        kw = w1 - w0 if w1 > w0 else 0
        if dtype == x.dtype:
            z = x.new_empty((b, k, kh, kw, c)).permute(0, 4, 1, 2, 3)
        else:
            z = torch.empty(
                (b, k, kh, kw, c), dtype=dtype, device=x.device
            ).permute(0, 4, 1, 2, 3)
    rw = ow * c
    n = b * ot * oh * rw
    if n < 1:
        return (out, z) if keep_cache_t > 0 else out
    if (
        fast
        and x.dtype in (torch.float16, torch.bfloat16)
        and wl + wr <= w
        and h > 0
        and tc + t > 0
        and wc > 0
        and wc <= 8192
        and wc & (wc - 1) == 0
        and max(ht * rw + wl * c, hb * rw + wr * c) <= 16384
    ):
        pack = (
            c % 2 == 0
            and x.dtype in (torch.float16, torch.bfloat16)
            and x.storage_offset() % 2 == 0
            and cx.storage_offset() % 2 == 0
        )
        shift = 1 if pack else 0
        prw = rw >> shift
        sw = wc >> shift
        left = (wl * c) >> shift
        right = (wr * c) >> shift
        pb = 1 << (max(left + right, 1) - 1).bit_length()
        eb = 1 << (max(ht * prw + left, hb * prw + right, 1) - 1).bit_length()
        frames = tc + t
        rows = 1
        if pack:
            if h % 16 == 0 and sw * 16 <= 16384:
                rows = 16
            elif h % 8 == 0 and sw * 8 <= 16384:
                rows = 8
            elif h % 4 == 0 and sw * 4 <= 16384:
                rows = 4
            elif h % 2 == 0 and sw * 2 <= 16384:
                rows = 2
        bs = rows * sw
        d = frames * (h // rows)
        m = d
        e = m + (tc + t + 1) * -(-(2 * eb) // bs)
        q = e + (front + tb) * -(-(oh * prw) // bs)
        zf = min(k, max(front - ot + k, 0))
        zt = min(k - zf, tb)
        nz = (zf + zt) * h * sw
        _uniform_tile[(q + -(-nz // bs), b)](
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
                prw,
                front,
                ht,
                left,
                right,
                k,
                d,
                m,
                e,
                q,
                nz,
                zf,
                zt,
            ),
            sw,
            pb,
            eb,
            pack,
            rows,
            num_warps=8,
        )
    else:
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
