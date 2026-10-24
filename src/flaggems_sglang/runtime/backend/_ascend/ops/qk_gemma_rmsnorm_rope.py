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
def _base_rows(
    X,
    W,
    C,
    P,
    Y,
    T: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    RD: tl.constexpr,
    NEOX: tl.constexpr,
    EPS: tl.constexpr,
    XS0: tl.constexpr,
    XS1: tl.constexpr,
    WS: tl.constexpr,
    CS0: tl.constexpr,
    CS1: tl.constexpr,
    PS: tl.constexpr,
    R: tl.constexpr,
    BD: tl.constexpr,
    BLOCK,
    BP: tl.constexpr,
    BT: tl.constexpr,
):
    if NEOX:
        row = BLOCK * R + tl.arange(0, R)
        token = row // H
        head = row % H
        base = token * XS0 + head * D * XS1
        out = row * D
        half: tl.constexpr = RD // 2
        pair = tl.arange(0, BP)
        i1 = pair if NEOX else 2 * pair
        i2 = half + pair if NEOX else 2 * pair + 1
        pmask = (token[:, None] < T) & (pair[None, :] < half)
        x1 = tl.load(X + base[:, None] + i1[None, :] * XS1, pmask, 0).to(
            tl.float32
        )
        x2 = tl.load(X + base[:, None] + i2[None, :] * XS1, pmask, 0).to(
            tl.float32
        )
        squares = tl.sum(x1 * x1 + x2 * x2, 1)
        if RD < D:
            tail = RD + tl.arange(0, BT)
            tmask = (token[:, None] < T) & (tail[None, :] < D)
            xt = tl.load(X + base[:, None] + tail[None, :] * XS1, tmask, 0).to(
                tl.float32
            )
            squares += tl.sum(xt * xt, 1)
        inv = tl.rsqrt(squares / D + EPS)
        if RD > 0:
            w1 = tl.load(W + i1 * WS, pair < half, 0).to(tl.float32)
            w2 = tl.load(W + i2 * WS, pair < half, 0).to(tl.float32)
            n1 = (
                (x1 * inv[:, None] * (w1[None, :] + 1.0))
                .to(Y.dtype.element_ty)
                .to(tl.float32)
            )
            n2 = (
                (x2 * inv[:, None] * (w2[None, :] + 1.0))
                .to(Y.dtype.element_ty)
                .to(tl.float32)
            )
            pos = tl.load(P + token * PS, token < T, 0)
            cos = tl.load(
                C + pos[:, None] * CS0 + pair[None, :] * CS1, pmask, 0
            ).to(tl.float32)
            sin = tl.load(
                C + pos[:, None] * CS0 + (half + pair[None, :]) * CS1, pmask, 0
            ).to(tl.float32)
            if T * H % R == 0 and BP == half:
                tl.store(Y + out[:, None] + i1[None, :], n1 * cos - n2 * sin)
            else:
                tl.store(
                    Y + out[:, None] + i1[None, :], n1 * cos - n2 * sin, pmask
                )
            if T * H % R == 0 and BP == half:
                tl.store(Y + out[:, None] + i2[None, :], n2 * cos + n1 * sin)
            else:
                tl.store(
                    Y + out[:, None] + i2[None, :], n2 * cos + n1 * sin, pmask
                )
        if RD < D:
            wt = tl.load(W + tail * WS, tail < D, 0).to(tl.float32)
            nt = (xt * inv[:, None] * (wt[None, :] + 1.0)).to(
                Y.dtype.element_ty
            )
            if T * H % R == 0 and BT == D - RD:
                tl.store(Y + out[:, None] + tail[None, :], nt)
            else:
                tl.store(Y + out[:, None] + tail[None, :], nt, tmask)
    else:
        row = BLOCK * R + tl.arange(0, R)
        token = tl.minimum(row // H, T - 1)
        head = row % H
        d = tl.arange(0, BD)
        dc = tl.minimum(d, D - 1)
        base = token * XS0 + head * D * XS1
        x = tl.load(X + base[:, None] + dc[None, :] * XS1).to(tl.float32)
        x = tl.where(d[None, :] < D, x, 0.0)
        inv = tl.rsqrt(tl.sum(x * x, 1) / D + EPS)
        w = tl.load(W + dc * WS).to(tl.float32)
        normal = (
            (x * inv[:, None] * (1.0 + w[None, :]))
            .to(Y.dtype.element_ty)
            .to(tl.float32)
        )
        if RD > 0:
            half: tl.constexpr = RD // 2
            if NEOX:
                partner = tl.where(d < half, d + half, d - half)
                ci = d % half
                first = d < half
            else:
                partner = d ^ 1
                ci = d // 2
                first = d % 2 == 0
            partner = tl.minimum(tl.maximum(partner, 0), D - 1)
            ci = tl.minimum(ci, half - 1)
            np = tl.gather(
                normal, tl.broadcast_to(partner[None, :], (R, BD)), 1
            )
            if H >= R and H % R == 0:
                shared_token = tl.minimum(BLOCK * R // H, T - 1)
                pos = tl.load(P + shared_token * PS)
                cos = tl.load(C + pos * CS0 + ci * CS1).to(tl.float32)[None, :]
                sin = tl.load(C + pos * CS0 + (ci + half) * CS1).to(
                    tl.float32
                )[None, :]
            else:
                pos = tl.load(P + token * PS)
                cos = tl.load(C + pos[:, None] * CS0 + ci[None, :] * CS1).to(
                    tl.float32
                )
                sin = tl.load(
                    C + pos[:, None] * CS0 + (ci[None, :] + half) * CS1
                ).to(tl.float32)
            rotated = normal * cos + tl.where(first[None, :], -np, np) * sin
            normal = tl.where(d[None, :] < RD, rotated, normal)
        if T * H % R == 0 and BD == D:
            tl.store(Y + row[:, None] * D + d[None, :], normal)
        else:
            tl.store(
                Y + row[:, None] * D + d[None, :],
                normal,
                (row[:, None] < T * H) & (d[None, :] < D),
            )


@triton.jit
def _base_fused(Q, K, QW, KW, C, P, QO, KO, META: tl.constexpr):
    T: tl.constexpr = META[0]
    NQ: tl.constexpr = META[1]
    NK: tl.constexpr = META[2]
    D: tl.constexpr = META[3]
    RD: tl.constexpr = META[4]
    NEOX: tl.constexpr = META[5]
    EPS: tl.constexpr = META[6]
    QS0: tl.constexpr = META[7]
    QS1: tl.constexpr = META[8]
    KS0: tl.constexpr = META[9]
    KS1: tl.constexpr = META[10]
    QWS: tl.constexpr = META[11]
    KWS: tl.constexpr = META[12]
    CS0: tl.constexpr = META[13]
    CS1: tl.constexpr = META[14]
    PS: tl.constexpr = META[15]
    R: tl.constexpr = META[16]
    BD: tl.constexpr = META[17]
    QB: tl.constexpr = META[18]
    KB: tl.constexpr = META[19]
    NP: tl.constexpr = META[20]
    for block in tl.range(tl.program_id(0), QB, NP, num_stages=1):
        _base_rows(
            Q,
            QW,
            C,
            P,
            QO,
            T,
            NQ,
            D,
            RD,
            NEOX,
            EPS,
            QS0,
            QS1,
            QWS,
            CS0,
            CS1,
            PS,
            R,
            BD,
            block,
            META[21],
            META[22],
        )
    for block in tl.range(tl.program_id(0), KB, NP, num_stages=1):
        _base_rows(
            K,
            KW,
            C,
            P,
            KO,
            T,
            NK,
            D,
            RD,
            NEOX,
            EPS,
            KS0,
            KS1,
            KWS,
            CS0,
            CS1,
            PS,
            R,
            BD,
            block,
            META[21],
            META[22],
        )


@triton.jit
def _rows(
    X,
    W,
    C,
    P,
    Y,
    T: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    RD: tl.constexpr,
    NEOX: tl.constexpr,
    EPS: tl.constexpr,
    XS0: tl.constexpr,
    XS1: tl.constexpr,
    WS: tl.constexpr,
    CS0: tl.constexpr,
    CS1: tl.constexpr,
    PS: tl.constexpr,
    R: tl.constexpr,
    BD: tl.constexpr,
    BLOCK,
    BH: tl.constexpr,
    WV,
    TOKEN: tl.constexpr,
    BP: tl.constexpr,
    BT: tl.constexpr,
):
    HG: tl.constexpr = (H + BH - 1) // BH
    tok = BLOCK // HG * R + tl.arange(0, R)
    heads = BLOCK % HG * BH + tl.arange(0, BH)
    tc = tok if T % R == 0 else tl.minimum(tok, T - 1)
    hc = heads if H % BH == 0 else tl.minimum(heads, H - 1)
    half: tl.constexpr = RD // 2
    pair = tl.arange(0, BP)
    i1 = pair if NEOX else 2 * pair
    i2 = half + pair if NEOX else 2 * pair + 1
    base = tc[:, None] * XS0 + hc[None, :] * D * XS1
    x1 = (
        tl.load(X + base[:, :, None] + i1[None, None, :] * XS1)
        if BP == half
        else tl.load(
            X + base[:, :, None] + i1[None, None, :] * XS1,
            pair[None, None, :] < half,
            0,
        )
    ).to(tl.float32)
    x2 = (
        tl.load(X + base[:, :, None] + i2[None, None, :] * XS1)
        if BP == half
        else tl.load(
            X + base[:, :, None] + i2[None, None, :] * XS1,
            pair[None, None, :] < half,
            0,
        )
    ).to(tl.float32)
    squares = tl.sum(x1 * x1 + x2 * x2, 2)
    if RD < D:
        tail = RD + tl.arange(0, BT)
        xt = (
            tl.load(X + base[:, :, None] + tail[None, None, :] * XS1)
            if BT == D - RD
            else tl.load(
                X + base[:, :, None] + tail[None, None, :] * XS1,
                tail[None, None, :] < D,
                0,
            )
        ).to(tl.float32)
        squares += tl.sum(xt * xt, 2)
    inv = tl.rsqrt(squares / D + EPS)
    offset = (tok[:, None] * H + heads[None, :]) * D
    mask = (tok[:, None, None] < T) & (heads[None, :, None] < H)
    if RD > 0:
        w1 = WV[0]
        w2 = WV[1]
        n1 = (
            (x1 * inv[:, :, None] * w1[None, None, :])
            .to(Y.dtype.element_ty)
            .to(tl.float32)
        )
        n2 = (
            (x2 * inv[:, :, None] * w2[None, None, :])
            .to(Y.dtype.element_ty)
            .to(tl.float32)
        )
        pos = tl.load(P + tc * PS)
        cos = (
            tl.load(C + pos[:, None] * CS0 + pair[None, :] * CS1)
            if BP == half
            else tl.load(
                C + pos[:, None] * CS0 + pair[None, :] * CS1,
                pair[None, :] < half,
                0,
            )
        ).to(tl.float32)
        sin = (
            tl.load(C + pos[:, None] * CS0 + (pair[None, :] + half) * CS1)
            if BP == half
            else tl.load(
                C + pos[:, None] * CS0 + (pair[None, :] + half) * CS1,
                pair[None, :] < half,
                0,
            )
        ).to(tl.float32)
        y1 = n1 * cos[:, None, :] - n2 * sin[:, None, :]
        y2 = n2 * cos[:, None, :] + n1 * sin[:, None, :]
        if T % R == 0 and H % BH == 0 and (BP == half):
            tl.store(Y + offset[:, :, None] + i1[None, None, :], y1)
        else:
            tl.store(
                Y + offset[:, :, None] + i1[None, None, :],
                y1,
                mask & (pair[None, None, :] < half),
            )
        if T % R == 0 and H % BH == 0 and (BP == half):
            tl.store(Y + offset[:, :, None] + i2[None, None, :], y2)
        else:
            tl.store(
                Y + offset[:, :, None] + i2[None, None, :],
                y2,
                mask & (pair[None, None, :] < half),
            )
    if RD < D:
        wt = WV[2]
        nt = (xt * inv[:, :, None] * wt[None, None, :]).to(Y.dtype.element_ty)
        if T % R == 0 and H % BH == 0 and (BT == D - RD):
            tl.store(Y + offset[:, :, None] + tail[None, None, :], nt)
        else:
            tl.store(
                Y + offset[:, :, None] + tail[None, None, :],
                nt,
                mask & (tail[None, None, :] < D),
            )


@triton.jit
def _fused(Q, K, QW, KW, C, P, QO, KO, META: tl.constexpr):
    T: tl.constexpr = META[0]
    NQ: tl.constexpr = META[1]
    NK: tl.constexpr = META[2]
    D: tl.constexpr = META[3]
    RD: tl.constexpr = META[4]
    NEOX: tl.constexpr = META[5]
    EPS: tl.constexpr = META[6]
    QS0: tl.constexpr = META[7]
    QS1: tl.constexpr = META[8]
    KS0: tl.constexpr = META[9]
    KS1: tl.constexpr = META[10]
    QWS: tl.constexpr = META[11]
    KWS: tl.constexpr = META[12]
    CS0: tl.constexpr = META[13]
    CS1: tl.constexpr = META[14]
    PS: tl.constexpr = META[15]
    R: tl.constexpr = META[16]
    BD: tl.constexpr = META[17]
    QB: tl.constexpr = META[18]
    KB: tl.constexpr = META[19]
    NP: tl.constexpr = META[20]
    pair = tl.arange(0, META[24])
    half: tl.constexpr = RD // 2
    i1 = pair if NEOX else 2 * pair
    i2 = half + pair if NEOX else 2 * pair + 1
    v1 = 1.0 + tl.load(QW + i1 * QWS, pair < half, 0).to(tl.float32)
    v2 = 1.0 + tl.load(QW + i2 * QWS, pair < half, 0).to(tl.float32)
    if RD < D:
        tail = RD + tl.arange(0, META[25])
        vt = 1.0 + tl.load(QW + tail * QWS, tail < D, 0).to(tl.float32)
    else:
        vt = 0.0
    qw = (v1, v2, vt)
    for block in tl.range(tl.program_id(0), QB, NP, num_stages=1):
        _rows(
            Q,
            QW,
            C,
            P,
            QO,
            T,
            NQ,
            D,
            RD,
            NEOX,
            EPS,
            QS0,
            QS1,
            QWS,
            CS0,
            CS1,
            PS,
            R,
            BD,
            block,
            META[21],
            qw,
            META[23],
            META[24],
            META[25],
        )
    pair = tl.arange(0, META[24])
    i1 = pair if NEOX else 2 * pair
    i2 = half + pair if NEOX else 2 * pair + 1
    v1 = 1.0 + tl.load(KW + i1 * KWS, pair < half, 0).to(tl.float32)
    v2 = 1.0 + tl.load(KW + i2 * KWS, pair < half, 0).to(tl.float32)
    if RD < D:
        tail = RD + tl.arange(0, META[25])
        vt = 1.0 + tl.load(KW + tail * KWS, tail < D, 0).to(tl.float32)
    else:
        vt = 0.0
    kw = (v1, v2, vt)
    for block in tl.range(tl.program_id(0), KB, NP, num_stages=1):
        _rows(
            K,
            KW,
            C,
            P,
            KO,
            T,
            NK,
            D,
            RD,
            NEOX,
            EPS,
            KS0,
            KS1,
            KWS,
            CS0,
            CS1,
            PS,
            R,
            BD,
            block,
            META[22],
            kw,
            META[23],
            META[24],
            META[25],
        )


def qk_gemma_rmsnorm_rope(
    q,
    k,
    q_weight,
    k_weight,
    positions,
    cos_sin_cache,
    eps,
    head_dim,
    rotary_dim,
    is_neox_style,
):
    t, cols = q.shape
    nq = cols // head_dim
    nk = k.shape[1] // head_dim
    total = t * nq
    token_tile = total >= 8192
    rows = 2 if token_tile else 4
    if token_tile:
        head_tile = 32 if total >= 65536 else 16
        hq = min(head_tile, 1 << (nq - 1).bit_length())
        hk = min(head_tile, 1 << (nk - 1).bit_length())
        blocks = (t + rows - 1) // rows
        qb = blocks * ((nq + hq - 1) // hq)
        kb = blocks * ((nk + hk - 1) // hk)
    else:
        qb = (total + rows - 1) // rows
        kb = (t * nk + rows - 1) // rows
    np = min(40, max(qb, kb))
    qo = torch.empty_like(q, memory_format=torch.contiguous_format)
    ko = torch.empty_like(k, memory_format=torch.contiguous_format)
    meta = (
        t,
        nq,
        nk,
        head_dim,
        rotary_dim,
        is_neox_style,
        eps,
        *q.stride(),
        *k.stride(),
        q_weight.stride(0),
        k_weight.stride(0),
        *cos_sin_cache.stride(),
        positions.stride(0),
        rows,
        1 << (head_dim - 1).bit_length(),
        qb,
        kb,
        np,
    )
    if token_tile:
        meta += (
            hq,
            hk,
            True,
            1 << (max(1, rotary_dim // 2) - 1).bit_length(),
            1 << (max(1, head_dim - rotary_dim) - 1).bit_length(),
        )
    else:
        meta += (
            1 << (max(1, rotary_dim // 2) - 1).bit_length(),
            1 << (max(1, head_dim - rotary_dim) - 1).bit_length(),
        )
    kernel = _fused if token_tile else _base_fused
    kernel.run(
        q,
        k,
        q_weight,
        k_weight,
        cos_sin_cache,
        positions,
        qo,
        ko,
        meta,
        num_warps=4,
        num_stages=1,
        enable_fp_fusion=False,
        grid=(np,),
        warmup=False,
    )
    return (qo, ko)


__all__ = ["qk_gemma_rmsnorm_rope"]
