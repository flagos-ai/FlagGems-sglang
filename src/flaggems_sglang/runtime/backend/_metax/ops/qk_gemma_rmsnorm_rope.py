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
    if NEOX:
        d1 = pair
        d2 = half + pair
    else:
        d1 = 2 * pair
        d2 = 2 * pair + 1
    pmask = (token[:, None] < T) & (pair[None, :] < half)
    x1 = tl.load(X + base[:, None] + d1[None, :] * XS1, pmask, 0).to(
        tl.float32
    )
    x2 = tl.load(X + base[:, None] + d2[None, :] * XS1, pmask, 0).to(
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
        w1 = tl.load(W + d1 * WS, pair < half, 0).to(tl.float32)
        w2 = tl.load(W + d2 * WS, pair < half, 0).to(tl.float32)
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
        tl.store(Y + out[:, None] + d1[None, :], n1 * cos - n2 * sin, pmask)
        tl.store(Y + out[:, None] + d2[None, :], n2 * cos + n1 * sin, pmask)
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
    R: tl.constexpr = META[16]
    BP: tl.constexpr = META[17]
    BT: tl.constexpr = META[18]
    QB: tl.constexpr = META[19]
    KB: tl.constexpr = META[20]
    NP: tl.constexpr = META[21]
    PERSISTENT: tl.constexpr = META[22]
    if PERSISTENT:
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
                BP,
                BT,
                block,
            )
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
                BP,
                BT,
                block,
            )
    else:
        block = tl.program_id(0)
        if block < QB:
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
                BP,
                BT,
                block,
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
                R,
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
    rows,
    warps,
    programs,
):
    t = q.shape[0]
    nq = q.shape[1] // head_dim
    nk = k.shape[1] // head_dim
    qo = torch.empty(q.shape, dtype=q.dtype, device=q.device)
    ko = torch.empty(k.shape, dtype=k.dtype, device=k.device)
    qb = triton.cdiv(t * nq, rows)
    kb = triton.cdiv(t * nk, rows)
    np = min(programs, max(qb, kb)) if programs else qb + kb
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
            rows,
            triton.next_power_of_2(max(1, rotary_dim // 2)),
            triton.next_power_of_2(max(1, head_dim - rotary_dim)),
            qb,
            kb,
            np,
            bool(programs),
        ),
        num_warps=warps,
        num_stages=1,
        enable_fp_fusion=False,
        grid=(np,),
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
    total_heads = q.shape[0] * (q.shape[1] + k.shape[1]) // head_dim
    rows = 32 if total_heads >= 16384 else (1 if total_heads < 512 else 8)
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
        0,
    )


__all__ = ["qk_gemma_rmsnorm_rope"]
