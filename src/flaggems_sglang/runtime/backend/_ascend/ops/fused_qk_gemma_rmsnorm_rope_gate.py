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
    G,
    T: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    RD: tl.constexpr,
    EPS: tl.constexpr,
    XS0: tl.constexpr,
    XS1: tl.constexpr,
    WS: tl.constexpr,
    CS0: tl.constexpr,
    CS1: tl.constexpr,
    PS: tl.constexpr,
    HAS_GATE: tl.constexpr,
    R: tl.constexpr,
    BD: tl.constexpr,
    BP: tl.constexpr,
    BLOCK,
):
    row = BLOCK * R + tl.arange(0, R)
    token = row // H
    head = row % H
    d = tl.arange(0, BD)
    base = token * XS0 + head * D * (1 + HAS_GATE) * XS1
    mask = (token[:, None] < T) & (d[None, :] < D)
    x = tl.load(X + base[:, None] + d[None, :] * XS1, mask, 0).to(tl.float32)
    inv = tl.rsqrt(tl.sum(x * x, 1) / D + EPS)
    out = row * D
    if RD > 0:
        half: tl.constexpr = RD // 2
        pair = tl.arange(0, BP)
        pair_mask = (token[:, None] < T) & (pair[None, :] < half)
        pos = tl.load(P + token * PS, token < T, 0)
        cos = tl.load(
            C + pos[:, None] * CS0 + pair[None, :] * CS1, pair_mask, 0
        ).to(tl.float32)
        sin = tl.load(
            C + pos[:, None] * CS0 + (half + pair[None, :]) * CS1, pair_mask, 0
        ).to(tl.float32)
        x1 = tl.load(X + base[:, None] + pair[None, :] * XS1, pair_mask, 0).to(
            tl.float32
        )
        x2 = tl.load(
            X + base[:, None] + (half + pair[None, :]) * XS1, pair_mask, 0
        ).to(tl.float32)
        w1 = tl.load(W + pair * WS, pair < half, 0).to(tl.float32)
        w2 = tl.load(W + (half + pair) * WS, pair < half, 0).to(tl.float32)
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
        r1 = n1 * cos - n2 * sin
        r2 = n2 * cos + n1 * sin
        tl.store(Y + out[:, None] + pair[None, :], r1, pair_mask)
        tl.store(Y + out[:, None] + half + pair[None, :], r2, pair_mask)
    if RD < D:
        w = tl.load(W + d * WS, d < D, 0).to(tl.float32)
        normal = (x * inv[:, None] * (w[None, :] + 1.0)).to(Y.dtype.element_ty)
        tl.store(
            Y + out[:, None] + d[None, :], normal, mask & (d[None, :] >= RD)
        )
    if HAS_GATE:
        gate = tl.load(X + base[:, None] + (D + d[None, :]) * XS1, mask, 0)
        tl.store(G + out[:, None] + d[None, :], gate, mask)


@triton.jit
def _fused(
    Q,
    K,
    QW,
    KW,
    C,
    P,
    QO,
    KO,
    GO,
    T: tl.constexpr,
    NQ: tl.constexpr,
    NK: tl.constexpr,
    D: tl.constexpr,
    RD: tl.constexpr,
    EPS: tl.constexpr,
    QS0: tl.constexpr,
    QS1: tl.constexpr,
    KS0: tl.constexpr,
    KS1: tl.constexpr,
    QWS: tl.constexpr,
    KWS: tl.constexpr,
    CS0: tl.constexpr,
    CS1: tl.constexpr,
    PS: tl.constexpr,
    HAS_GATE: tl.constexpr,
    R: tl.constexpr,
    BD: tl.constexpr,
    BP: tl.constexpr,
    Q_BLOCKS: tl.constexpr,
    K_BLOCKS: tl.constexpr,
    NP: tl.constexpr,
):
    for block in tl.range(
        tl.program_id(0), Q_BLOCKS + K_BLOCKS, NP, num_stages=1
    ):
        if block < Q_BLOCKS:
            _rows(
                Q,
                QW,
                C,
                P,
                QO,
                GO,
                T,
                NQ,
                D,
                RD,
                EPS,
                QS0,
                QS1,
                QWS,
                CS0,
                CS1,
                PS,
                HAS_GATE,
                R,
                BD,
                BP,
                block,
            )
        else:
            _rows(
                K,
                KW,
                C,
                P,
                KO,
                GO,
                T,
                NK,
                D,
                RD,
                EPS,
                KS0,
                KS1,
                KWS,
                CS0,
                CS1,
                PS,
                False,
                R,
                BD,
                BP,
                block - Q_BLOCKS,
            )


@triton.jit
def _paired_head(
    X,
    W,
    C,
    P,
    O,
    G,
    h: tl.constexpr,
    D: tl.constexpr,
    E: tl.constexpr,
    x0: tl.constexpr,
    x1: tl.constexpr,
    ws: tl.constexpr,
    CS0: tl.constexpr,
    CS1: tl.constexpr,
    PS: tl.constexpr,
    GATE: tl.constexpr,
    n: tl.constexpr,
    GRID: tl.constexpr,
):
    stride: tl.constexpr = 1 + GATE
    d = tl.arange(0, D // 2).to(tl.int64)
    wa = tl.load(W + d * ws).to(tl.float32)
    wb = tl.load(W + (d + D // 2) * ws).to(tl.float32)
    for row in range(tl.program_id(0).to(tl.int64), n, GRID):
        token = row // h
        head = row % h
        base = token * x0 + head * D * stride * x1
        a = tl.load(X + base + d * x1).to(tl.float32)
        b = tl.load(X + base + (d + D // 2) * x1).to(tl.float32)
        inv = tl.rsqrt(tl.sum(a * a + b * b, 0) / D + E)
        ya = (a * inv * (wa + 1.0)).to(O.dtype.element_ty).to(tl.float32)
        yb = (b * inv * (wb + 1.0)).to(O.dtype.element_ty).to(tl.float32)
        pos = tl.load(P + token * PS).to(tl.int64)
        c = tl.load(C + pos * CS0 + d * CS1).to(tl.float32)
        s = tl.load(C + pos * CS0 + (d + D // 2) * CS1).to(tl.float32)
        tl.store(O + row * D + d, ya * c - yb * s)
        tl.store(O + row * D + d + D // 2, yb * c + ya * s)
        if GATE:
            ga = tl.load(X + base + (D + d) * x1)
            gb = tl.load(X + base + (D + D // 2 + d) * x1)
            tl.store(G + row * D + d, ga)
            tl.store(G + row * D + d + D // 2, gb)


@triton.jit
def _paired_axis(
    Q,
    K,
    QW,
    KW,
    C,
    P,
    QO,
    KO,
    G,
    HQ: tl.constexpr,
    HK: tl.constexpr,
    D: tl.constexpr,
    E: tl.constexpr,
    Q0: tl.constexpr,
    Q1: tl.constexpr,
    K0: tl.constexpr,
    K1: tl.constexpr,
    QWS: tl.constexpr,
    KWS: tl.constexpr,
    CS0: tl.constexpr,
    CS1: tl.constexpr,
    PS: tl.constexpr,
    GATE: tl.constexpr,
    NQ: tl.constexpr,
    NK: tl.constexpr,
    GRID: tl.constexpr,
):
    if tl.program_id(1) == 0:
        _paired_head(
            Q,
            QW,
            C,
            P,
            QO,
            G,
            HQ,
            D,
            E,
            Q0,
            Q1,
            QWS,
            CS0,
            CS1,
            PS,
            GATE,
            NQ,
            GRID,
        )
    else:
        _paired_head(
            K,
            KW,
            C,
            P,
            KO,
            KO,
            HK,
            D,
            E,
            K0,
            K1,
            KWS,
            CS0,
            CS1,
            PS,
            False,
            NK,
            GRID,
        )


@triton.jit
def _batched_head(
    X,
    W,
    C,
    P,
    O,
    G,
    H: tl.constexpr,
    D: tl.constexpr,
    E: tl.constexpr,
    X0: tl.constexpr,
    X1: tl.constexpr,
    WS: tl.constexpr,
    CS0: tl.constexpr,
    CS1: tl.constexpr,
    PS: tl.constexpr,
    GATE: tl.constexpr,
    N: tl.constexpr,
    GRID: tl.constexpr,
    R: tl.constexpr,
):
    d = tl.arange(0, D // 2).to(tl.int64)
    offsets = tl.arange(0, R).to(tl.int64)
    wa = tl.load(W + d * WS).to(tl.float32)
    wb = tl.load(W + (d + D // 2) * WS).to(tl.float32)
    for block in tl.range(
        tl.program_id(0).to(tl.int64), tl.cdiv(N, R), GRID, num_stages=1
    ):
        if N % R == 0:
            row = block * R + offsets
            token = row // H
            head = row % H
            base = token * X0 + head * D * (1 + GATE) * X1
            mask = row[:, None] < N
            a = tl.load(X + base[:, None] + d[None, :] * X1, mask, 0).to(
                tl.float32
            )
            b = tl.load(
                X + base[:, None] + (d[None, :] + D // 2) * X1, mask, 0
            ).to(tl.float32)
            inv = tl.rsqrt(tl.sum(a * a + b * b, 1) / D + E)
            ya = (
                (a * inv[:, None] * (wa[None, :] + 1.0))
                .to(O.dtype.element_ty)
                .to(tl.float32)
            )
            yb = (
                (b * inv[:, None] * (wb[None, :] + 1.0))
                .to(O.dtype.element_ty)
                .to(tl.float32)
            )
            if R % H == 0 and N % R == 0:
                if R == H:
                    pos = tl.load(P + block * PS).to(tl.int64)
                    token_c = tl.load(C + pos * CS0 + d * CS1).to(tl.float32)
                    token_s = tl.load(C + pos * CS0 + (d + D // 2) * CS1).to(
                        tl.float32
                    )
                    c = tl.broadcast_to(token_c[None, :], (R, D // 2))
                    s = tl.broadcast_to(token_s[None, :], (R, D // 2))
                else:
                    cache_token = block * (R // H) + tl.arange(0, R // H).to(
                        tl.int64
                    )
                    pos = tl.load(P + cache_token * PS).to(tl.int64)
                    token_c = tl.load(
                        C + pos[:, None] * CS0 + d[None, :] * CS1
                    ).to(tl.float32)
                    token_s = tl.load(
                        C + pos[:, None] * CS0 + (d[None, :] + D // 2) * CS1
                    ).to(tl.float32)
                    c = tl.broadcast_to(
                        token_c[:, None, :], (R // H, H, D // 2)
                    ).reshape(R, D // 2)
                    s = tl.broadcast_to(
                        token_s[:, None, :], (R // H, H, D // 2)
                    ).reshape(R, D // 2)
            else:
                pos = tl.load(P + token * PS, row < N, 0).to(tl.int64)
                c = tl.load(
                    C + pos[:, None] * CS0 + d[None, :] * CS1, mask, 0
                ).to(tl.float32)
                s = tl.load(
                    C + pos[:, None] * CS0 + (d[None, :] + D // 2) * CS1,
                    mask,
                    0,
                ).to(tl.float32)
            tl.store(O + row[:, None] * D + d[None, :], ya * c - yb * s, mask)
            tl.store(
                O + row[:, None] * D + d[None, :] + D // 2,
                yb * c + ya * s,
                mask,
            )
            if GATE:
                gate_d = tl.arange(0, D).to(tl.int64)
                gate_value = tl.load(
                    X + base[:, None] + (D + gate_d[None, :]) * X1, mask, 0
                )
                tl.store(
                    G + row[:, None] * D + gate_d[None, :], gate_value, mask
                )
        else:
            row = block * R + offsets
            token = row // H
            head = row % H
            base = token * X0 + head * D * (1 + GATE) * X1
            mask = row[:, None] < N
            a = tl.load(X + base[:, None] + d[None, :] * X1, mask, 0).to(
                tl.float32
            )
            b = tl.load(
                X + base[:, None] + (d[None, :] + D // 2) * X1, mask, 0
            ).to(tl.float32)
            inv = tl.rsqrt(tl.sum(a * a + b * b, 1) / D + E)
            ya = (
                (a * inv[:, None] * (wa[None, :] + 1.0))
                .to(O.dtype.element_ty)
                .to(tl.float32)
            )
            yb = (
                (b * inv[:, None] * (wb[None, :] + 1.0))
                .to(O.dtype.element_ty)
                .to(tl.float32)
            )
            if R % H == 0 and N % R == 0:
                if R == H:
                    pos = tl.load(P + block * PS).to(tl.int64)
                    token_c = tl.load(C + pos * CS0 + d * CS1).to(tl.float32)
                    token_s = tl.load(C + pos * CS0 + (d + D // 2) * CS1).to(
                        tl.float32
                    )
                    c = tl.broadcast_to(token_c[None, :], (R, D // 2))
                    s = tl.broadcast_to(token_s[None, :], (R, D // 2))
                else:
                    cache_token = block * (R // H) + tl.arange(0, R // H).to(
                        tl.int64
                    )
                    pos = tl.load(P + cache_token * PS).to(tl.int64)
                    token_c = tl.load(
                        C + pos[:, None] * CS0 + d[None, :] * CS1
                    ).to(tl.float32)
                    token_s = tl.load(
                        C + pos[:, None] * CS0 + (d[None, :] + D // 2) * CS1
                    ).to(tl.float32)
                    c = tl.broadcast_to(
                        token_c[:, None, :], (R // H, H, D // 2)
                    ).reshape(R, D // 2)
                    s = tl.broadcast_to(
                        token_s[:, None, :], (R // H, H, D // 2)
                    ).reshape(R, D // 2)
            else:
                pos = tl.load(P + token * PS, row < N, 0).to(tl.int64)
                c = tl.load(
                    C + pos[:, None] * CS0 + d[None, :] * CS1, mask, 0
                ).to(tl.float32)
                s = tl.load(
                    C + pos[:, None] * CS0 + (d[None, :] + D // 2) * CS1,
                    mask,
                    0,
                ).to(tl.float32)
            tl.store(O + row[:, None] * D + d[None, :], ya * c - yb * s, mask)
            tl.store(
                O + row[:, None] * D + d[None, :] + D // 2,
                yb * c + ya * s,
                mask,
            )
            if GATE:
                ga = tl.load(
                    X + base[:, None] + (D + d[None, :]) * X1, mask, 0
                )
                gb = tl.load(
                    X + base[:, None] + (D + D // 2 + d[None, :]) * X1, mask, 0
                )
                tl.store(G + row[:, None] * D + d[None, :], ga, mask)
                tl.store(G + row[:, None] * D + d[None, :] + D // 2, gb, mask)


@triton.jit
def _batched_axis(
    Q,
    K,
    QW,
    KW,
    C,
    P,
    QO,
    KO,
    G,
    HQ: tl.constexpr,
    HK: tl.constexpr,
    D: tl.constexpr,
    E: tl.constexpr,
    Q0: tl.constexpr,
    Q1: tl.constexpr,
    K0: tl.constexpr,
    K1: tl.constexpr,
    QWS: tl.constexpr,
    KWS: tl.constexpr,
    CS0: tl.constexpr,
    CS1: tl.constexpr,
    PS: tl.constexpr,
    GATE: tl.constexpr,
    NQ: tl.constexpr,
    NK: tl.constexpr,
    GRID: tl.constexpr,
    R: tl.constexpr,
):
    if tl.program_id(1) == 0:
        _batched_head(
            Q,
            QW,
            C,
            P,
            QO,
            G,
            HQ,
            D,
            E,
            Q0,
            Q1,
            QWS,
            CS0,
            CS1,
            PS,
            GATE,
            NQ,
            GRID,
            R,
        )
    else:
        _batched_head(
            K,
            KW,
            C,
            P,
            KO,
            KO,
            HK,
            D,
            E,
            K0,
            K1,
            KWS,
            CS0,
            CS1,
            PS,
            False,
            NK,
            GRID,
            R,
        )


@triton.jit
def _partial_head(
    X,
    W,
    C,
    P,
    O,
    G,
    H: tl.constexpr,
    D: tl.constexpr,
    E: tl.constexpr,
    X0: tl.constexpr,
    X1: tl.constexpr,
    WS: tl.constexpr,
    CS0: tl.constexpr,
    CS1: tl.constexpr,
    PS: tl.constexpr,
    GATE: tl.constexpr,
    N: tl.constexpr,
    GRID: tl.constexpr,
    R: tl.constexpr,
):
    d = tl.arange(0, D // 4).to(tl.int64)
    offsets = tl.arange(0, R).to(tl.int64)
    wa = tl.load(W + d * WS).to(tl.float32)
    wb = tl.load(W + (d + D // 4) * WS).to(tl.float32)
    wc = tl.load(W + (d + D // 2) * WS).to(tl.float32)
    wd = tl.load(W + (d + 3 * D // 4) * WS).to(tl.float32)
    for block in tl.range(
        tl.program_id(0).to(tl.int64), tl.cdiv(N, R), GRID, num_stages=1
    ):
        if N % R == 0:
            row = block * R + offsets
            token = row // H
            head = row % H
            base = token * X0 + head * D * (1 + GATE) * X1
            mask = row[:, None] < N
            a = tl.load(X + base[:, None] + d[None, :] * X1).to(tl.float32)
            b = tl.load(X + base[:, None] + (d[None, :] + D // 4) * X1).to(
                tl.float32
            )
            c = tl.load(X + base[:, None] + (d[None, :] + D // 2) * X1).to(
                tl.float32
            )
            e = tl.load(X + base[:, None] + (d[None, :] + 3 * D // 4) * X1).to(
                tl.float32
            )
            inv = tl.rsqrt(tl.sum(a * a + b * b + (c * c + e * e), 1) / D + E)
            ya = (
                (a * inv[:, None] * (wa[None, :] + 1.0))
                .to(O.dtype.element_ty)
                .to(tl.float32)
            )
            yb = (
                (b * inv[:, None] * (wb[None, :] + 1.0))
                .to(O.dtype.element_ty)
                .to(tl.float32)
            )
            yc = (c * inv[:, None] * (wc[None, :] + 1.0)).to(
                O.dtype.element_ty
            )
            yd = (e * inv[:, None] * (wd[None, :] + 1.0)).to(
                O.dtype.element_ty
            )
            if R % H == 0 and N % R == 0:
                if R == H:
                    pos = tl.load(P + block * PS).to(tl.int64)
                    token_c = tl.load(C + pos * CS0 + d * CS1).to(tl.float32)
                    token_s = tl.load(C + pos * CS0 + (d + D // 4) * CS1).to(
                        tl.float32
                    )
                    cosine = tl.broadcast_to(token_c[None, :], (R, D // 4))
                    sine = tl.broadcast_to(token_s[None, :], (R, D // 4))
                else:
                    cache_token = block * (R // H) + tl.arange(0, R // H).to(
                        tl.int64
                    )
                    pos = tl.load(P + cache_token * PS).to(tl.int64)
                    token_c = tl.load(
                        C + pos[:, None] * CS0 + d[None, :] * CS1
                    ).to(tl.float32)
                    token_s = tl.load(
                        C + pos[:, None] * CS0 + (d[None, :] + D // 4) * CS1
                    ).to(tl.float32)
                    cosine = tl.broadcast_to(
                        token_c[:, None, :], (R // H, H, D // 4)
                    ).reshape(R, D // 4)
                    sine = tl.broadcast_to(
                        token_s[:, None, :], (R // H, H, D // 4)
                    ).reshape(R, D // 4)
            else:
                pos = tl.load(P + token * PS).to(tl.int64)
                cosine = tl.load(C + pos[:, None] * CS0 + d[None, :] * CS1).to(
                    tl.float32
                )
                sine = tl.load(
                    C + pos[:, None] * CS0 + (d[None, :] + D // 4) * CS1
                ).to(tl.float32)
            out = row[:, None] * D + d[None, :]
            tl.store(O + out, ya * cosine - yb * sine)
            tl.store(O + out + D // 4, yb * cosine + ya * sine)
            tl.store(O + out + D // 2, yc)
            tl.store(O + out + 3 * D // 4, yd)
            if GATE:
                gate_d = tl.arange(0, D).to(tl.int64)
                gate_value = tl.load(
                    X + base[:, None] + (D + gate_d[None, :]) * X1
                )
                tl.store(G + row[:, None] * D + gate_d[None, :], gate_value)
        else:
            row = block * R + offsets
            token = row // H
            head = row % H
            base = token * X0 + head * D * (1 + GATE) * X1
            mask = row[:, None] < N
            a = tl.load(X + base[:, None] + d[None, :] * X1, mask, 0).to(
                tl.float32
            )
            b = tl.load(
                X + base[:, None] + (d[None, :] + D // 4) * X1, mask, 0
            ).to(tl.float32)
            c = tl.load(
                X + base[:, None] + (d[None, :] + D // 2) * X1, mask, 0
            ).to(tl.float32)
            e = tl.load(
                X + base[:, None] + (d[None, :] + 3 * D // 4) * X1, mask, 0
            ).to(tl.float32)
            inv = tl.rsqrt(tl.sum(a * a + b * b + (c * c + e * e), 1) / D + E)
            ya = (
                (a * inv[:, None] * (wa[None, :] + 1.0))
                .to(O.dtype.element_ty)
                .to(tl.float32)
            )
            yb = (
                (b * inv[:, None] * (wb[None, :] + 1.0))
                .to(O.dtype.element_ty)
                .to(tl.float32)
            )
            yc = (c * inv[:, None] * (wc[None, :] + 1.0)).to(
                O.dtype.element_ty
            )
            yd = (e * inv[:, None] * (wd[None, :] + 1.0)).to(
                O.dtype.element_ty
            )
            if R % H == 0 and N % R == 0:
                if R == H:
                    pos = tl.load(P + block * PS).to(tl.int64)
                    token_c = tl.load(C + pos * CS0 + d * CS1).to(tl.float32)
                    token_s = tl.load(C + pos * CS0 + (d + D // 4) * CS1).to(
                        tl.float32
                    )
                    cosine = tl.broadcast_to(token_c[None, :], (R, D // 4))
                    sine = tl.broadcast_to(token_s[None, :], (R, D // 4))
                else:
                    cache_token = block * (R // H) + tl.arange(0, R // H).to(
                        tl.int64
                    )
                    pos = tl.load(P + cache_token * PS).to(tl.int64)
                    token_c = tl.load(
                        C + pos[:, None] * CS0 + d[None, :] * CS1
                    ).to(tl.float32)
                    token_s = tl.load(
                        C + pos[:, None] * CS0 + (d[None, :] + D // 4) * CS1
                    ).to(tl.float32)
                    cosine = tl.broadcast_to(
                        token_c[:, None, :], (R // H, H, D // 4)
                    ).reshape(R, D // 4)
                    sine = tl.broadcast_to(
                        token_s[:, None, :], (R // H, H, D // 4)
                    ).reshape(R, D // 4)
            else:
                pos = tl.load(P + token * PS, row < N, 0).to(tl.int64)
                cosine = tl.load(
                    C + pos[:, None] * CS0 + d[None, :] * CS1, mask, 0
                ).to(tl.float32)
                sine = tl.load(
                    C + pos[:, None] * CS0 + (d[None, :] + D // 4) * CS1,
                    mask,
                    0,
                ).to(tl.float32)
            out = row[:, None] * D + d[None, :]
            tl.store(O + out, ya * cosine - yb * sine, mask)
            tl.store(O + out + D // 4, yb * cosine + ya * sine, mask)
            tl.store(O + out + D // 2, yc, mask)
            tl.store(O + out + 3 * D // 4, yd, mask)
            if GATE:
                ga = tl.load(
                    X + base[:, None] + (D + d[None, :]) * X1, mask, 0
                )
                gb = tl.load(
                    X + base[:, None] + (D + D // 4 + d[None, :]) * X1, mask, 0
                )
                gc = tl.load(
                    X + base[:, None] + (D + D // 2 + d[None, :]) * X1, mask, 0
                )
                gd = tl.load(
                    X + base[:, None] + (D + 3 * D // 4 + d[None, :]) * X1,
                    mask,
                    0,
                )
                tl.store(G + out, ga, mask)
                tl.store(G + out + D // 4, gb, mask)
                tl.store(G + out + D // 2, gc, mask)
                tl.store(G + out + 3 * D // 4, gd, mask)


@triton.jit
def _partial_axis(
    Q,
    K,
    QW,
    KW,
    C,
    P,
    QO,
    KO,
    G,
    HQ: tl.constexpr,
    HK: tl.constexpr,
    D: tl.constexpr,
    E: tl.constexpr,
    Q0: tl.constexpr,
    Q1: tl.constexpr,
    K0: tl.constexpr,
    K1: tl.constexpr,
    QWS: tl.constexpr,
    KWS: tl.constexpr,
    CS0: tl.constexpr,
    CS1: tl.constexpr,
    PS: tl.constexpr,
    GATE: tl.constexpr,
    NQ: tl.constexpr,
    NK: tl.constexpr,
    GRID: tl.constexpr,
    R: tl.constexpr,
):
    if tl.program_id(1) == 0:
        _partial_head(
            Q,
            QW,
            C,
            P,
            QO,
            G,
            HQ,
            D,
            E,
            Q0,
            Q1,
            QWS,
            CS0,
            CS1,
            PS,
            GATE,
            NQ,
            GRID,
            R,
        )
    else:
        _partial_head(
            K,
            KW,
            C,
            P,
            KO,
            KO,
            HK,
            D,
            E,
            K0,
            K1,
            KWS,
            CS0,
            CS1,
            PS,
            False,
            NK,
            GRID,
            R,
        )


@triton.jit
def _weighted_head(
    X,
    W,
    C,
    P,
    O,
    G,
    H: tl.constexpr,
    D: tl.constexpr,
    E: tl.constexpr,
    X0: tl.constexpr,
    X1: tl.constexpr,
    WS: tl.constexpr,
    CS0: tl.constexpr,
    CS1: tl.constexpr,
    PS: tl.constexpr,
    GATE: tl.constexpr,
    N: tl.constexpr,
    GRID: tl.constexpr,
    R: tl.constexpr,
    PID,
):
    d = tl.arange(0, D // 2).to(tl.int64)
    offsets = tl.arange(0, R).to(tl.int64)
    wa = tl.load(W + d * WS).to(tl.float32)
    wb = tl.load(W + (d + D // 2) * WS).to(tl.float32)
    for block in tl.range(PID.to(tl.int64), tl.cdiv(N, R), GRID, num_stages=1):
        if N % R == 0:
            row = block * R + offsets
            token = row // H
            head = row % H
            base = token * X0 + head * D * (1 + GATE) * X1
            mask = row[:, None] < N
            a = tl.load(X + base[:, None] + d[None, :] * X1).to(tl.float32)
            b = tl.load(X + base[:, None] + (d[None, :] + D // 2) * X1).to(
                tl.float32
            )
            inv = tl.rsqrt(tl.sum(a * a + b * b, 1) / D + E)
            ya = (
                (a * inv[:, None] * (wa[None, :] + 1.0))
                .to(O.dtype.element_ty)
                .to(tl.float32)
            )
            yb = (
                (b * inv[:, None] * (wb[None, :] + 1.0))
                .to(O.dtype.element_ty)
                .to(tl.float32)
            )
            if R % H == 0 and N % R == 0:
                cache_token = block * (R // H) + tl.arange(0, R // H).to(
                    tl.int64
                )
                pos = tl.load(P + cache_token * PS).to(tl.int64)
                token_c = tl.load(
                    C + pos[:, None] * CS0 + d[None, :] * CS1
                ).to(tl.float32)
                token_s = tl.load(
                    C + pos[:, None] * CS0 + (d[None, :] + D // 2) * CS1
                ).to(tl.float32)
                c = tl.broadcast_to(
                    token_c[:, None, :], (R // H, H, D // 2)
                ).reshape(R, D // 2)
                s = tl.broadcast_to(
                    token_s[:, None, :], (R // H, H, D // 2)
                ).reshape(R, D // 2)
            else:
                pos = tl.load(P + token * PS).to(tl.int64)
                c = tl.load(C + pos[:, None] * CS0 + d[None, :] * CS1).to(
                    tl.float32
                )
                s = tl.load(
                    C + pos[:, None] * CS0 + (d[None, :] + D // 2) * CS1
                ).to(tl.float32)
            tl.store(O + row[:, None] * D + d[None, :], ya * c - yb * s)
            tl.store(
                O + row[:, None] * D + d[None, :] + D // 2, yb * c + ya * s
            )
            if GATE:
                gd = tl.arange(0, D).to(tl.int64)
                g = tl.load(X + base[:, None] + (D + gd[None, :]) * X1)
                tl.store(G + row[:, None] * D + gd[None, :], g)
        else:
            row = block * R + offsets
            token = row // H
            head = row % H
            base = token * X0 + head * D * (1 + GATE) * X1
            mask = row[:, None] < N
            a = tl.load(X + base[:, None] + d[None, :] * X1, mask, 0).to(
                tl.float32
            )
            b = tl.load(
                X + base[:, None] + (d[None, :] + D // 2) * X1, mask, 0
            ).to(tl.float32)
            inv = tl.rsqrt(tl.sum(a * a + b * b, 1) / D + E)
            ya = (
                (a * inv[:, None] * (wa[None, :] + 1.0))
                .to(O.dtype.element_ty)
                .to(tl.float32)
            )
            yb = (
                (b * inv[:, None] * (wb[None, :] + 1.0))
                .to(O.dtype.element_ty)
                .to(tl.float32)
            )
            pos = tl.load(P + token * PS, row < N, 0).to(tl.int64)
            c = tl.load(C + pos[:, None] * CS0 + d[None, :] * CS1, mask, 0).to(
                tl.float32
            )
            s = tl.load(
                C + pos[:, None] * CS0 + (d[None, :] + D // 2) * CS1, mask, 0
            ).to(tl.float32)
            tl.store(O + row[:, None] * D + d[None, :], ya * c - yb * s, mask)
            tl.store(
                O + row[:, None] * D + d[None, :] + D // 2,
                yb * c + ya * s,
                mask,
            )
            if GATE:
                ga = tl.load(
                    X + base[:, None] + (D + d[None, :]) * X1, mask, 0
                )
                gb = tl.load(
                    X + base[:, None] + (D + D // 2 + d[None, :]) * X1, mask, 0
                )
                tl.store(G + row[:, None] * D + d[None, :], ga, mask)
                tl.store(G + row[:, None] * D + d[None, :] + D // 2, gb, mask)


@triton.jit
def _weighted_axis(
    Q,
    K,
    QW,
    KW,
    C,
    P,
    QO,
    KO,
    G,
    HQ: tl.constexpr,
    HK: tl.constexpr,
    D: tl.constexpr,
    E: tl.constexpr,
    Q0: tl.constexpr,
    Q1: tl.constexpr,
    K0: tl.constexpr,
    K1: tl.constexpr,
    QWS: tl.constexpr,
    KWS: tl.constexpr,
    CS0: tl.constexpr,
    CS1: tl.constexpr,
    PS: tl.constexpr,
    GATE: tl.constexpr,
    NQ: tl.constexpr,
    NK: tl.constexpr,
    QGRID: tl.constexpr,
    KGRID: tl.constexpr,
    R: tl.constexpr,
):
    if tl.program_id(0) < QGRID:
        _weighted_head(
            Q,
            QW,
            C,
            P,
            QO,
            G,
            HQ,
            D,
            E,
            Q0,
            Q1,
            QWS,
            CS0,
            CS1,
            PS,
            GATE,
            NQ,
            QGRID,
            R,
            tl.program_id(0),
        )
    else:
        _weighted_head(
            K,
            KW,
            C,
            P,
            KO,
            KO,
            HK,
            D,
            E,
            K0,
            K1,
            KWS,
            CS0,
            CS1,
            PS,
            False,
            NK,
            KGRID,
            R,
            tl.program_id(0) - QGRID,
        )


def fused_qk_gemma_rmsnorm_rope_gate(
    q_gate,
    k,
    q_weight,
    k_weight,
    cos_sin_cache,
    positions,
    eps,
    num_q_heads,
    num_kv_heads,
    head_dim,
    rotary_dim,
    has_gate,
):
    t = q_gate.shape[0]
    q_out = torch.empty(
        (t, num_q_heads * head_dim), dtype=q_gate.dtype, device=q_gate.device
    )
    k_out = torch.empty(
        (t, num_kv_heads * head_dim), dtype=q_gate.dtype, device=q_gate.device
    )
    gate_out = (
        torch.empty(
            (t, num_q_heads, head_dim),
            dtype=q_gate.dtype,
            device=q_gate.device,
        )
        if has_gate
        else None
    )
    if t > 0:
        nq = t * num_q_heads
        nk = t * num_kv_heads
        if (
            head_dim == 128
            and rotary_dim == 128
            and (nq >= 65536)
            and (q_gate.dtype == k.dtype)
            and (q_weight.dtype == k_weight.dtype)
        ):
            rows = 64
            q_blocks = triton.cdiv(nq, rows)
            k_blocks = triton.cdiv(nk, rows)
            programs = min(80, q_blocks + k_blocks)
            qgrid = min(q_blocks, max(1, programs * nq // (nq + nk)))
            kgrid = min(k_blocks, max(1, programs - qgrid))
            _weighted_axis[qgrid + kgrid,](
                q_gate,
                k,
                q_weight,
                k_weight,
                cos_sin_cache,
                positions,
                q_out,
                k_out,
                gate_out if has_gate else q_out,
                num_q_heads,
                num_kv_heads,
                head_dim,
                eps,
                q_gate.stride(0),
                q_gate.stride(1),
                k.stride(0),
                k.stride(1),
                q_weight.stride(0),
                k_weight.stride(0),
                cos_sin_cache.stride(0),
                cos_sin_cache.stride(1),
                positions.stride(0),
                has_gate,
                nq,
                nk,
                qgrid,
                kgrid,
                rows,
                num_warps=4,
                num_stages=1,
                enable_fp_fusion=False,
            )
        elif (
            head_dim == 128
            and rotary_dim == 64
            and (nq >= 16384)
            and (q_gate.dtype == k.dtype)
            and (q_weight.dtype == k_weight.dtype)
        ):
            rows = 32
            grid = min(triton.cdiv(max(nq, nk), rows), 40)
            _partial_axis[grid, 2](
                q_gate,
                k,
                q_weight,
                k_weight,
                cos_sin_cache,
                positions,
                q_out,
                k_out,
                gate_out if has_gate else q_out,
                num_q_heads,
                num_kv_heads,
                head_dim,
                eps,
                q_gate.stride(0),
                q_gate.stride(1),
                k.stride(0),
                k.stride(1),
                q_weight.stride(0),
                k_weight.stride(0),
                cos_sin_cache.stride(0),
                cos_sin_cache.stride(1),
                positions.stride(0),
                has_gate,
                nq,
                nk,
                grid,
                rows,
                num_warps=4,
                num_stages=1,
                enable_fp_fusion=False,
            )
        elif (
            head_dim == 128
            and rotary_dim == 128
            and (nq >= 16384)
            and (q_gate.dtype == k.dtype)
            and (q_weight.dtype == k_weight.dtype)
        ):
            rows = 32
            grid = min(triton.cdiv(max(nq, nk), rows), 40)
            _batched_axis[grid, 2](
                q_gate,
                k,
                q_weight,
                k_weight,
                cos_sin_cache,
                positions,
                q_out,
                k_out,
                gate_out if has_gate else q_out,
                num_q_heads,
                num_kv_heads,
                head_dim,
                eps,
                q_gate.stride(0),
                q_gate.stride(1),
                k.stride(0),
                k.stride(1),
                q_weight.stride(0),
                k_weight.stride(0),
                cos_sin_cache.stride(0),
                cos_sin_cache.stride(1),
                positions.stride(0),
                has_gate,
                nq,
                nk,
                grid,
                rows,
                num_warps=4,
                num_stages=1,
                enable_fp_fusion=False,
            )
        elif (
            head_dim == 128
            and rotary_dim == 128
            and (nq <= 2048)
            and (q_gate.dtype == k.dtype)
            and (q_weight.dtype == k_weight.dtype)
        ):
            grid = min(max(nq, nk), 40)
            _paired_axis[grid, 2](
                q_gate,
                k,
                q_weight,
                k_weight,
                cos_sin_cache,
                positions,
                q_out,
                k_out,
                gate_out if has_gate else q_out,
                num_q_heads,
                num_kv_heads,
                head_dim,
                eps,
                q_gate.stride(0),
                q_gate.stride(1),
                k.stride(0),
                k.stride(1),
                q_weight.stride(0),
                k_weight.stride(0),
                cos_sin_cache.stride(0),
                cos_sin_cache.stride(1),
                positions.stride(0),
                has_gate,
                nq,
                nk,
                grid,
                num_warps=4,
                enable_fp_fusion=False,
            )
        else:
            rows = 8 if head_dim <= 128 else 4
            q_blocks = triton.cdiv(nq, rows)
            k_blocks = triton.cdiv(nk, rows)
            programs = min(40, q_blocks + k_blocks)
            _fused[programs,](
                q_gate,
                k,
                q_weight,
                k_weight,
                cos_sin_cache,
                positions,
                q_out,
                k_out,
                gate_out if has_gate else q_out,
                t,
                num_q_heads,
                num_kv_heads,
                head_dim,
                rotary_dim,
                eps,
                q_gate.stride(0),
                q_gate.stride(1),
                k.stride(0),
                k.stride(1),
                q_weight.stride(0),
                k_weight.stride(0),
                cos_sin_cache.stride(0),
                cos_sin_cache.stride(1),
                positions.stride(0),
                has_gate,
                rows,
                triton.next_power_of_2(head_dim),
                triton.next_power_of_2(max(1, rotary_dim // 2)),
                q_blocks,
                k_blocks,
                programs,
                num_warps=4,
                num_stages=1,
                enable_fp_fusion=False,
            )
    return (q_out, k_out, gate_out)


__all__ = ["fused_qk_gemma_rmsnorm_rope_gate"]
