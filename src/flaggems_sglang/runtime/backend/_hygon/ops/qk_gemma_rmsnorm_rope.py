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
    BP: tl.constexpr,
    BT: tl.constexpr,
    BLOCK,
):
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
        tl.store(Y + out[:, None] + i1[None, :], n1 * cos - n2 * sin, pmask)
        tl.store(Y + out[:, None] + i2[None, :], n2 * cos + n1 * sin, pmask)
    if RD < D:
        wt = tl.load(W + tail * WS, tail < D, 0).to(tl.float32)
        nt = (xt * inv[:, None] * (wt[None, :] + 1.0)).to(Y.dtype.element_ty)
        tl.store(Y + out[:, None] + tail[None, :], nt, tmask)


@triton.jit
def _q_heads(
    X,
    W,
    C,
    P,
    Y,
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
    BP: tl.constexpr,
    BT: tl.constexpr,
    GROUPS: tl.constexpr,
    BLOCK,
):
    token = BLOCK // GROUPS
    head = BLOCK % GROUPS * R + tl.arange(0, R)
    hc = tl.minimum(head, H - 1)
    base = token * XS0 + hc * D * XS1
    out = (token * H + head) * D
    half: tl.constexpr = RD // 2
    pair = tl.arange(0, BP)
    i1 = pair if NEOX else 2 * pair
    i2 = half + pair if NEOX else 2 * pair + 1
    pmask = (head[:, None] < H) & (pair[None, :] < half)
    x1 = tl.load(X + base[:, None] + i1[None, :] * XS1, pmask, 0).to(
        tl.float32
    )
    x2 = tl.load(X + base[:, None] + i2[None, :] * XS1, pmask, 0).to(
        tl.float32
    )
    squares = tl.sum(x1 * x1 + x2 * x2, 1)
    if RD < D:
        tail = RD + tl.arange(0, BT)
        tmask = (head[:, None] < H) & (tail[None, :] < D)
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
        pos = tl.load(P + token * PS)
        cos = tl.load(C + pos * CS0 + pair * CS1, pair < half, 0).to(
            tl.float32
        )
        sin = tl.load(C + pos * CS0 + (half + pair) * CS1, pair < half, 0).to(
            tl.float32
        )
        tl.store(
            Y + out[:, None] + i1[None, :],
            n1 * cos[None, :] - n2 * sin[None, :],
            pmask,
        )
        tl.store(
            Y + out[:, None] + i2[None, :],
            n2 * cos[None, :] + n1 * sin[None, :],
            pmask,
        )
    if RD < D:
        wt = tl.load(W + tail * WS, tail < D, 0).to(tl.float32)
        nt = (xt * inv[:, None] * (wt[None, :] + 1.0)).to(Y.dtype.element_ty)
        tl.store(Y + out[:, None] + tail[None, :], nt, tmask)


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
    RQ: tl.constexpr = META[16]
    RK: tl.constexpr = META[17]
    BP: tl.constexpr = META[18]
    BT: tl.constexpr = META[19]
    QB: tl.constexpr = META[20]
    Q_TOKEN: tl.constexpr = META[22]
    K_TOKEN: tl.constexpr = META[23]
    block = tl.program_id(0)
    if block < QB:
        if Q_TOKEN:
            _q_heads(
                Q,
                QW,
                C,
                P,
                QO,
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
                RQ,
                BP,
                BT,
                triton.cdiv(NQ, RQ),
                block,
            )
        else:
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
                RQ,
                BP,
                BT,
                block,
            )
    elif K_TOKEN:
        _q_heads(
            K,
            KW,
            C,
            P,
            KO,
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
            RK,
            BP,
            BT,
            triton.cdiv(NK, RK),
            block - QB,
        )
    else:
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
            RK,
            BP,
            BT,
            block - QB,
        )


def _call(
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
    rows=8,
    warps=4,
    share_k=False,
):
    t = q.shape[0]
    nq = q.shape[1] // head_dim
    nk = k.shape[1] // head_dim
    qo = torch.empty_like(q, memory_format=torch.contiguous_format)
    ko = torch.empty_like(k, memory_format=torch.contiguous_format)
    rq = min(rows, triton.next_power_of_2(nq))
    rk = min(rows, triton.next_power_of_2(nk))
    q_token = nq >= rows and nq % rows == 0
    k_token = share_k and nk % rk == 0
    qb = t * triton.cdiv(nq, rq) if q_token else triton.cdiv(t * nq, rq)
    kb = t * triton.cdiv(nk, rk) if k_token else triton.cdiv(t * nk, rk)
    _fused.run(
        q,
        k,
        q_weight,
        k_weight,
        cos_sin_cache,
        positions,
        qo,
        ko,
        (
            t,
            nq,
            nk,
            head_dim,
            rotary_dim,
            is_neox_style,
            eps,
            q.stride(0),
            q.stride(1),
            k.stride(0),
            k.stride(1),
            q_weight.stride(0),
            k_weight.stride(0),
            cos_sin_cache.stride(0),
            cos_sin_cache.stride(1),
            positions.stride(0),
            rq,
            rk,
            triton.next_power_of_2(max(1, rotary_dim // 2)),
            triton.next_power_of_2(max(1, head_dim - rotary_dim)),
            qb,
            kb,
            q_token,
            k_token,
        ),
        num_warps=warps,
        num_stages=1,
        enable_fp_fusion=False,
        grid=(qb + kb,),
        warmup=False,
    )
    return (qo, ko)


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
    total = q.shape[0] * (q.shape[1] // head_dim)
    rows = 4 if total <= 2048 and not is_neox_style else 8
    if total >= 8192:
        rows = 32 if rotary_dim < head_dim or total >= 65536 else 16
    return _call(
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
        rows,
        4,
        True,
    )


__all__ = ["qk_gemma_rmsnorm_rope"]
