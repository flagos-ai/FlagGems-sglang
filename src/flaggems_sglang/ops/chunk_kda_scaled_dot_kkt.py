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
def _pair_sum(ah, al, bh, bl):
    s = ah + bh
    bv = s - ah
    e = ah - (s - bv) + (bh - bv)
    e = e + al + bl
    high = s + e
    low = e - (high - s)
    return (high, low)


@triton.jit
def _weighted_sum(x, y, decay):
    product = x * y
    product_low = tl.fma(x, y, -product)
    high = product * decay
    low = tl.fma(product, decay, -high) + product_low * decay
    high, low = tl.reduce((high, low), 1, _pair_sum)
    return high + low


@triton.jit
def _bounded_rows(
    X,
    K,
    G,
    BETA,
    b,
    h,
    chunk,
    i,
    j,
    ti,
    tj,
    c,
    xs0,
    xs1,
    xs2,
    xs3,
    SK: tl.constexpr,
    SG: tl.constexpr,
    SB: tl.constexpr,
    T: tl.constexpr,
    D: tl.constexpr,
    BT: tl.constexpr,
    GS: tl.constexpr,
    SCALE: tl.constexpr,
):
    if BT == 64:
        mi = (ti[:, None] < T) & (i[:, None] < BT) & (c[None, :] < D)
        mj = (tj[:, None] < T) & (j[:, None] < BT) & (c[None, :] < D)
        x = tl.load(
            X + b * xs0 + ti[:, None] * xs1 + h * xs2 + c[None, :] * xs3, mi, 0
        ).to(tl.float32)
        k = tl.load(
            K
            + b * SK[0]
            + tj[:, None] * SK[1]
            + h * SK[2]
            + c[None, :] * SK[3],
            mj,
            0,
        ).to(tl.float32)
        energy = tl.maximum(
            tl.max(tl.sum(x * x, 1), 0), tl.max(tl.sum(k * k, 1), 0)
        )
        magnitude = tl.maximum(
            tl.max(tl.max(tl.abs(x), 1), 0), tl.max(tl.max(tl.abs(k), 1), 0)
        )
        gi = (
            tl.load(
                G
                + b * SG[0]
                + ti[:, None] * SG[1]
                + h * SG[2]
                + c[None, :] * SG[3],
                mi,
                0,
            ).to(tl.float32)
            * GS
        )
        gj = (
            tl.load(
                G
                + b * SG[0]
                + tj[:, None] * SG[1]
                + h * SG[2]
                + c[None, :] * SG[3],
                mj,
                0,
            ).to(tl.float32)
            * GS
        )
        pi = chunk * BT + tl.maximum(i - 1, 0)
        pj = chunk * BT + tl.maximum(j - 1, 0)
        prev_i = (
            tl.load(
                G
                + b * SG[0]
                + pi[:, None] * SG[1]
                + h * SG[2]
                + c[None, :] * SG[3],
                mi,
                0,
            ).to(tl.float32)
            * GS
        )
        prev_j = (
            tl.load(
                G
                + b * SG[0]
                + pj[:, None] * SG[1]
                + h * SG[2]
                + c[None, :] * SG[3],
                mj,
                0,
            ).to(tl.float32)
            * GS
        )
        growth_i = tl.min(
            tl.min(
                (tl.where(mi, gi - prev_i, 0.0) <= 0.0009765625).to(tl.int32),
                1,
            ),
            0,
        )
        growth_j = tl.min(
            tl.min(
                (tl.where(mj, gj - prev_j, 0.0) <= 0.0009765625).to(tl.int32),
                1,
            ),
            0,
        )
        beta = tl.load(
            BETA + b * SB[0] + ti * SB[1] + h * SB[2], (ti < T) & (i < BT), 0
        ).to(tl.float32)
        return (
            (magnitude < 6.0)
            & (energy < 256.0)
            & (growth_i > 0)
            & (growth_j > 0)
            & (tl.max(tl.abs(beta), 0) <= 1.0)
            & (tl.abs(SCALE) <= 1.0)
        )
    else:
        return tl.full((), False, tl.int1)


@triton.jit
def single_compute(Q, K, G, BETA, A, AQ, META: tl.constexpr):
    BK: tl.constexpr = 64
    AXIS: tl.constexpr = 1
    STEPS: tl.constexpr = 1
    T: tl.constexpr = META[0]
    H: tl.constexpr = META[1]
    D: tl.constexpr = META[2]
    BT: tl.constexpr = META[3]
    if len(META) == 6:
        SQ: tl.constexpr = (T * H * D, H * D, D, 1)
        SK: tl.constexpr = SQ
        SG: tl.constexpr = SQ
        SB: tl.constexpr = (T * H, H, 1)
        SCALE: tl.constexpr = META[4]
        GS: tl.constexpr = META[5]
    else:
        SQ: tl.constexpr = META[4]
        SK: tl.constexpr = META[5]
        SG: tl.constexpr = META[6]
        SB: tl.constexpr = META[7]
        SCALE: tl.constexpr = META[8]
        GS: tl.constexpr = META[9]
    BM: tl.constexpr = 32
    BN: tl.constexpr = 32
    BD: tl.constexpr = (
        32 if D <= 32 else 64 if D <= 64 else 128 if D <= 128 else 256
    )
    p = tl.program_id(0)
    isq = tl.program_id(1) == 1
    X = tl.where(isq, Q, K)
    Y = tl.where(isq, AQ, A)
    xs0 = tl.where(isq, SQ[0], SK[0])
    xs1 = tl.where(isq, SQ[1], SK[1])
    xs2 = tl.where(isq, SQ[2], SK[2])
    xs3 = tl.where(isq, SQ[3], SK[3])
    nc: tl.constexpr = triton.cdiv(BT, BN)
    nr: tl.constexpr = triton.cdiv(BT, BM)
    nt: tl.constexpr = triton.cdiv(T, BT)
    if BT == 64:
        local = p % 3
        jc = tl.where(local == 2, 32, 0)
        ir = tl.where(local == 0, 0, 32)
        chunk = p // 3 % nt
        h = p // (3 * nt) % H
        b = p // (3 * nt * H)
    else:
        jc = p % nc * BN
        ir = p // nc % nr * BM
        chunk = p // (nc * nr) % nt
        h = p // (nc * nr * nt) % H
        b = p // (nc * nr * nt * H)
    i = ir + tl.arange(0, BM)
    j = jc + tl.arange(0, BN)
    ti = chunk * BT + i
    tj = chunk * BT + j
    c = tl.arange(0, BD)
    offset = ((b * T + ti[:, None]) * H + h) * BT + j[None, :]
    mask = (ti[:, None] < T) & (i[:, None] < BT) & (j[None, :] < BT)
    if jc >= ir + BM:
        z = tl.full((BM, BN), 0, tl.float32)
        tl.store(Y + offset, z, mask)
    elif D >= 128 and T >= 128:
        f1 = tl.minimum(chunk * BT + BT // 2, T - 1)
        f0 = (
            tl.load(G + b * SG[0] + f1 * SG[1] + h * SG[2]).to(tl.float32) * GS
        )
        GUARD_AXIS: tl.constexpr = AXIS
        PIPE: tl.constexpr = (
            STEPS
            if T % BT == 0
            and BT % 32 == 0
            and (D % BK == 0)
            and (len(META) == 6)
            else 1
        )
        if GUARD_AXIS == 0:
            f5 = tl.full((BM, BK), 0, tl.float32)
        elif GUARD_AXIS == 1:
            f5 = tl.full((BM,), 0, tl.float32)
        else:
            f5 = tl.full((BK,), 0, tl.float32)
        f18 = tl.full((BM, BN), 0, tl.float32)
        for f3 in tl.range(
            0, triton.cdiv(D, BK), num_stages=PIPE, loop_unroll_factor=1
        ):
            f6 = f3 * BK + tl.arange(0, BK)
            f12 = (ti[:, None] < T) & (i[:, None] < BT) & (f6[None, :] < D)
            f13 = (tj[:, None] < T) & (j[:, None] < BT) & (f6[None, :] < D)
            f21 = tl.load(
                X + b * xs0 + ti[:, None] * xs1 + h * xs2 + f6[None, :] * xs3,
                f12,
                0,
            ).to(tl.float32)
            f8 = (
                tl.load(
                    G
                    + b * SG[0]
                    + ti[:, None] * SG[1]
                    + h * SG[2]
                    + f6[None, :] * SG[3],
                    f12,
                    0,
                ).to(tl.float32)
                * GS
            )
            f10 = tl.load(
                K
                + b * SK[0]
                + tj[:, None] * SK[1]
                + h * SK[2]
                + f6[None, :] * SK[3],
                f13,
                0,
            ).to(tl.float32)
            f9 = (
                tl.load(
                    G
                    + b * SG[0]
                    + tj[:, None] * SG[1]
                    + h * SG[2]
                    + f6[None, :] * SG[3],
                    f13,
                    0,
                ).to(tl.float32)
                * GS
            )
            f4 = tl.maximum(
                tl.maximum(tl.abs(f8 - f0), tl.abs(f9 - f0)) * 0.125,
                tl.maximum(tl.abs(f21), tl.abs(f10)) * 0.015625,
            )
            if GUARD_AXIS == 0:
                f5 = tl.maximum(f5, f4)
            elif GUARD_AXIS == 1:
                f5 = tl.maximum(f5, tl.max(f4, 1))
            else:
                f5 = tl.maximum(f5, tl.max(f4, 0))
            f11 = f21 * tl.exp2(f8 - f0)
            f16 = tl.trans(f10 * tl.exp2(f0 - f9))
            f19 = f11.to(tl.float16)
            f15 = f16.to(tl.float16)
            f20 = (f11 - f19.to(tl.float32)).to(tl.float16)
            f17 = (f16 - f15.to(tl.float32)).to(tl.float16)
            if 1 == 0:
                f18 = tl.dot(f20, f15, f18)
                f18 = tl.dot(f19, f17, f18)
                f18 = tl.dot(f19, f15, f18)
            else:
                f14 = tl.dot(f20, f15)
                f14 = tl.dot(f19, f17, f14)
                f14 = tl.dot(f19, f15, f14)
                f18 = f18 + f14
        if GUARD_AXIS == 0:
            final_guard = tl.max(tl.max(f5, 1), 0)
        else:
            final_guard = tl.max(f5, 0)
        if (final_guard < 1.0) & _bounded_rows(
            X,
            K,
            G,
            BETA,
            b,
            h,
            chunk,
            i,
            j,
            ti,
            tj,
            c,
            xs0,
            xs1,
            xs2,
            xs3,
            SK,
            SG,
            SB,
            T,
            D,
            BT,
            GS,
            SCALE,
        ):
            f2 = tl.load(
                BETA + b * SB[0] + ti * SB[1] + h * SB[2],
                (ti < T) & (i < BT),
                0,
            ).to(tl.float32)
            f7 = tl.where(isq, SCALE, f2)
            f18 = tl.where(
                (i[:, None] + isq.to(tl.int32) > j[None, :])
                & (tj[None, :] < T),
                f18 * f7[:, None],
                0.0,
            )
            tl.store(Y + offset, f18, mask)
        else:
            mj = (tj[:, None] < T) & (j[:, None] < BT) & (c[None, :] < D)
            kj = tl.load(
                K
                + b * SK[0]
                + tj[:, None] * SK[1]
                + h * SK[2]
                + c[None, :] * SK[3],
                mj,
                0,
            ).to(tl.float32)
            gj = (
                tl.load(
                    G
                    + b * SG[0]
                    + tj[:, None] * SG[1]
                    + h * SG[2]
                    + c[None, :] * SG[3],
                    mj,
                    0,
                ).to(tl.float32)
                * GS
            )
            for r in range(BM):
                ri = ir + r
                rt = chunk * BT + ri
                xr = tl.load(
                    X + b * xs0 + rt * xs1 + h * xs2 + c * xs3,
                    (rt < T) & (ri < BT) & (c < D),
                    0,
                ).to(tl.float32)
                gr = (
                    tl.load(
                        G + b * SG[0] + rt * SG[1] + h * SG[2] + c * SG[3],
                        (rt < T) & (ri < BT) & (c < D),
                        0,
                    ).to(tl.float32)
                    * GS
                )
                decay = tl.exp2(gr[None, :] - gj)
                v = _weighted_sum(xr[None, :], kj, decay)
                beta = tl.load(
                    BETA + b * SB[0] + rt * SB[1] + h * SB[2],
                    (rt < T) & (ri < BT),
                    0,
                ).to(tl.float32)
                factor = tl.where(isq, SCALE, beta)
                v = tl.where(
                    (ri + isq.to(tl.int32) > j) & (tj < T), v * factor, 0.0
                )
                out = ((b * T + rt) * H + h) * BT + j
                valid = (rt < T) & (ri < BT) & (j < BT)
                tl.store(Y + out, v, valid)
    else:
        mi = (ti[:, None] < T) & (i[:, None] < BT) & (c[None, :] < D)
        mj = (tj[:, None] < T) & (j[:, None] < BT) & (c[None, :] < D)
        xi = tl.load(
            X + b * xs0 + ti[:, None] * xs1 + h * xs2 + c[None, :] * xs3, mi, 0
        ).to(tl.float32)
        gi = (
            tl.load(
                G
                + b * SG[0]
                + ti[:, None] * SG[1]
                + h * SG[2]
                + c[None, :] * SG[3],
                mi,
                0,
            ).to(tl.float32)
            * GS
        )
        kj = tl.load(
            K
            + b * SK[0]
            + tj[:, None] * SK[1]
            + h * SK[2]
            + c[None, :] * SK[3],
            mj,
            0,
        ).to(tl.float32)
        gj = (
            tl.load(
                G
                + b * SG[0]
                + tj[:, None] * SG[1]
                + h * SG[2]
                + c[None, :] * SG[3],
                mj,
                0,
            ).to(tl.float32)
            * GS
        )
        at = tl.minimum(chunk * BT + BT // 2, T - 1)
        anchor = (
            tl.load(G + b * SG[0] + at * SG[1] + h * SG[2]).to(tl.float32) * GS
        )
        if BM == BN:
            point_bound = tl.maximum(
                tl.maximum(tl.abs(gi - anchor), tl.abs(gj - anchor)) * 0.125,
                tl.maximum(tl.abs(xi), tl.abs(kj)) * 0.015625,
            )
            bound = tl.max(tl.max(point_bound, 1), 0)
        else:
            left_bound = tl.maximum(
                tl.abs(gi - anchor) * 0.125, tl.abs(xi) * 0.015625
            )
            right_bound = tl.maximum(
                tl.abs(gj - anchor) * 0.125, tl.abs(kj) * 0.015625
            )
            bound = tl.maximum(
                tl.max(tl.max(left_bound, 1), 0),
                tl.max(tl.max(right_bound, 1), 0),
            )
        if (bound < 1.0) & _bounded_rows(
            X,
            K,
            G,
            BETA,
            b,
            h,
            chunk,
            i,
            j,
            ti,
            tj,
            c,
            xs0,
            xs1,
            xs2,
            xs3,
            SK,
            SG,
            SB,
            T,
            D,
            BT,
            GS,
            SCALE,
        ):
            left = tl.exp2(gi - anchor)
            right = tl.exp2(anchor - gj)
            keys = tl.trans(kj * right)
            xl = xi * left
            xh = xl.to(tl.float16)
            rh = keys.to(tl.float16)
            xr = (xl - xh.to(tl.float32)).to(tl.float16)
            rr = (keys - rh.to(tl.float32)).to(tl.float16)
            v = tl.dot(xr, rh)
            v = tl.dot(xh, rr, v)
            v = tl.dot(xh, rh, v)
            beta = tl.load(
                BETA + b * SB[0] + ti * SB[1] + h * SB[2],
                (ti < T) & (i < BT),
                0,
            ).to(tl.float32)
            factor = tl.where(isq, SCALE, beta)
            v = tl.where(
                (i[:, None] + isq.to(tl.int32) > j[None, :])
                & (tj[None, :] < T),
                v * factor[:, None],
                0.0,
            )
            tl.store(Y + offset, v, mask)
        else:
            for r in range(BM):
                ri = ir + r
                rt = chunk * BT + ri
                xr = tl.load(
                    X + b * xs0 + rt * xs1 + h * xs2 + c * xs3,
                    (rt < T) & (ri < BT) & (c < D),
                    0,
                ).to(tl.float32)
                gr = (
                    tl.load(
                        G + b * SG[0] + rt * SG[1] + h * SG[2] + c * SG[3],
                        (rt < T) & (ri < BT) & (c < D),
                        0,
                    ).to(tl.float32)
                    * GS
                )
                decay = tl.exp2(gr[None, :] - gj)
                v = _weighted_sum(xr[None, :], kj, decay)
                beta = tl.load(
                    BETA + b * SB[0] + rt * SB[1] + h * SB[2],
                    (rt < T) & (ri < BT),
                    0,
                ).to(tl.float32)
                factor = tl.where(isq, SCALE, beta)
                v = tl.where(
                    (ri + isq.to(tl.int32) > j) & (tj < T), v * factor, 0.0
                )
                out = ((b * T + rt) * H + h) * BT + j
                valid = (rt < T) & (ri < BT) & (j < BT)
                tl.store(Y + out, v, valid)
    if BT == 64:
        if p % 3 == 1:
            zr = chunk * BT + tl.arange(0, BM)
            zc = 32 + tl.arange(0, BN)
            zout = ((b * T + zr[:, None]) * H + h) * BT + zc[None, :]
            zeros = tl.full((BM, BN), 0.0, tl.float32)
            tl.store(Y + zout, zeros, zr[:, None] < T)


@triton.jit
def metax_small_role_inline(
    Q,
    K,
    G,
    BETA,
    A,
    AQ,
    META: tl.constexpr,
    BK: tl.constexpr,
    AXIS: tl.constexpr,
    STEPS: tl.constexpr,
    p,
    ISQ: tl.constexpr,
):
    SCHEDULE: tl.constexpr = 3
    T: tl.constexpr = META[0]
    H: tl.constexpr = META[1]
    D: tl.constexpr = META[2]
    BT: tl.constexpr = META[3]
    if len(META) == 6:
        SQ: tl.constexpr = (T * H * D, H * D, D, 1)
        SK: tl.constexpr = SQ
        SG: tl.constexpr = SQ
        SB: tl.constexpr = (T * H, H, 1)
        SCALE: tl.constexpr = META[4]
        GS: tl.constexpr = META[5]
    else:
        SQ: tl.constexpr = META[4]
        SK: tl.constexpr = META[5]
        SG: tl.constexpr = META[6]
        SB: tl.constexpr = META[7]
        SCALE: tl.constexpr = META[8]
        GS: tl.constexpr = META[9]
    BM: tl.constexpr = 32
    BN: tl.constexpr = 32
    BD: tl.constexpr = (
        32 if D <= 32 else 64 if D <= 64 else 128 if D <= 128 else 256
    )
    isq = tl.full((), ISQ, tl.int1)
    if ISQ:
        X = Q
        Y = AQ
        xs0: tl.constexpr = SQ[0]
        xs1: tl.constexpr = SQ[1]
        xs2: tl.constexpr = SQ[2]
        xs3: tl.constexpr = SQ[3]
    else:
        X = K
        Y = A
        xs0: tl.constexpr = SK[0]
        xs1: tl.constexpr = SK[1]
        xs2: tl.constexpr = SK[2]
        xs3: tl.constexpr = SK[3]
    nc: tl.constexpr = triton.cdiv(BT, BN)
    nr: tl.constexpr = triton.cdiv(BT, BM)
    nt: tl.constexpr = triton.cdiv(T, BT)
    if BT == 64:
        local = p % 3
        jc = tl.where(local == 2, 32, 0)
        ir = tl.where(local == 0, 0, 32)
        chunk = p // 3 % nt
        h = p // (3 * nt) % H
        b = p // (3 * nt * H)
    else:
        jc = p % nc * BN
        ir = p // nc % nr * BM
        chunk = p // (nc * nr) % nt
        h = p // (nc * nr * nt) % H
        b = p // (nc * nr * nt * H)
    i = ir + tl.arange(0, BM)
    j = jc + tl.arange(0, BN)
    ti = chunk * BT + i
    tj = chunk * BT + j
    c = tl.arange(0, BD)
    offset = ((b * T + ti[:, None]) * H + h) * BT + j[None, :]
    mask = (ti[:, None] < T) & (i[:, None] < BT) & (j[None, :] < BT)
    if SCHEDULE >= 3:
        if BT == 64:
            if p % 3 == 1:
                zr = chunk * BT + tl.arange(0, BM)
                zc = 32 + tl.arange(0, BN)
                zout = ((b * T + zr[:, None]) * H + h) * BT + zc[None, :]
                zeros = tl.full((BM, BN), 0.0, tl.float32)
                tl.store(Y + zout, zeros, zr[:, None] < T)
    if jc >= ir + BM:
        z = tl.full((BM, BN), 0, tl.float32)
        tl.store(Y + offset, z, mask)
    elif True:
        f1 = tl.minimum(chunk * BT + BT // 2, T - 1)
        f0 = (
            tl.load(G + b * SG[0] + f1 * SG[1] + h * SG[2]).to(tl.float32) * GS
        )
        GUARD_AXIS: tl.constexpr = AXIS
        PIPE: tl.constexpr = (
            STEPS
            if T % BT == 0
            and BT % 32 == 0
            and (D % BK == 0)
            and (len(META) == 6)
            else 1
        )
        if GUARD_AXIS == 0:
            f5 = tl.full((BM, BK), 0, tl.float32)
        elif GUARD_AXIS == 1:
            f5 = tl.full((BM,), 0, tl.float32)
        else:
            f5 = tl.full((BK,), 0, tl.float32)
        f18 = tl.full((BM, BN), 0, tl.float32)
        if SCHEDULE % 3 == 0:
            if ISQ:
                f7 = tl.full((BM,), SCALE, tl.float32)
            else:
                f2 = tl.load(
                    BETA + b * SB[0] + ti * SB[1] + h * SB[2],
                    (ti < T) & (i < BT),
                    0,
                ).to(tl.float32)
                f7 = f2
        for f3 in tl.range(
            0, triton.cdiv(D, BK), num_stages=PIPE, loop_unroll_factor=1
        ):
            f6 = f3 * BK + tl.arange(0, BK)
            f12 = (ti[:, None] < T) & (i[:, None] < BT) & (f6[None, :] < D)
            f13 = (tj[:, None] < T) & (j[:, None] < BT) & (f6[None, :] < D)
            f21 = tl.load(
                X + b * xs0 + ti[:, None] * xs1 + h * xs2 + f6[None, :] * xs3,
                f12,
                0,
            ).to(tl.float32)
            f8 = (
                tl.load(
                    G
                    + b * SG[0]
                    + ti[:, None] * SG[1]
                    + h * SG[2]
                    + f6[None, :] * SG[3],
                    f12,
                    0,
                ).to(tl.float32)
                * GS
            )
            f10 = tl.load(
                K
                + b * SK[0]
                + tj[:, None] * SK[1]
                + h * SK[2]
                + f6[None, :] * SK[3],
                f13,
                0,
            ).to(tl.float32)
            f9 = (
                tl.load(
                    G
                    + b * SG[0]
                    + tj[:, None] * SG[1]
                    + h * SG[2]
                    + f6[None, :] * SG[3],
                    f13,
                    0,
                ).to(tl.float32)
                * GS
            )
            f4 = tl.maximum(
                tl.maximum(tl.abs(f8 - f0), tl.abs(f9 - f0)) * 0.125,
                tl.maximum(tl.abs(f21), tl.abs(f10)) * 0.015625,
            )
            if GUARD_AXIS == 0:
                f5 = tl.maximum(f5, f4)
            elif GUARD_AXIS == 1:
                f5 = tl.maximum(f5, tl.max(f4, 1))
            else:
                f5 = tl.maximum(f5, tl.max(f4, 0))
            f11 = f21 * tl.exp2(f8 - f0)
            f16 = tl.trans(f10 * tl.exp2(f0 - f9))
            f19 = f11.to(tl.float16)
            f15 = f16.to(tl.float16)
            f20 = (f11 - f19.to(tl.float32)).to(tl.float16)
            f17 = (f16 - f15.to(tl.float32)).to(tl.float16)
            if 1 == 0:
                f18 = tl.dot(f20, f15, f18)
                f18 = tl.dot(f19, f17, f18)
                f18 = tl.dot(f19, f15, f18)
            else:
                f14 = tl.dot(f20, f15)
                f14 = tl.dot(f19, f17, f14)
                f14 = tl.dot(f19, f15, f14)
                f18 = f18 + f14
        if SCHEDULE % 3 == 1:
            if ISQ:
                f7 = tl.full((BM,), SCALE, tl.float32)
            else:
                f2 = tl.load(
                    BETA + b * SB[0] + ti * SB[1] + h * SB[2],
                    (ti < T) & (i < BT),
                    0,
                ).to(tl.float32)
                f7 = f2
        if GUARD_AXIS == 0:
            final_guard = tl.max(tl.max(f5, 1), 0)
        else:
            final_guard = tl.max(f5, 0)
        if (final_guard < 1.0) & _bounded_rows(
            X,
            K,
            G,
            BETA,
            b,
            h,
            chunk,
            i,
            j,
            ti,
            tj,
            c,
            xs0,
            xs1,
            xs2,
            xs3,
            SK,
            SG,
            SB,
            T,
            D,
            BT,
            GS,
            SCALE,
        ):
            if SCHEDULE % 3 == 2:
                if ISQ:
                    f7 = tl.full((BM,), SCALE, tl.float32)
                else:
                    f2 = tl.load(
                        BETA + b * SB[0] + ti * SB[1] + h * SB[2],
                        (ti < T) & (i < BT),
                        0,
                    ).to(tl.float32)
                    f7 = f2
            f18 = tl.where(
                (i[:, None] + isq.to(tl.int32) > j[None, :])
                & (tj[None, :] < T),
                f18 * f7[:, None],
                0.0,
            )
            tl.store(Y + offset, f18, mask)
        else:
            mj = (tj[:, None] < T) & (j[:, None] < BT) & (c[None, :] < D)
            kj = tl.load(
                K
                + b * SK[0]
                + tj[:, None] * SK[1]
                + h * SK[2]
                + c[None, :] * SK[3],
                mj,
                0,
            ).to(tl.float32)
            gj = (
                tl.load(
                    G
                    + b * SG[0]
                    + tj[:, None] * SG[1]
                    + h * SG[2]
                    + c[None, :] * SG[3],
                    mj,
                    0,
                ).to(tl.float32)
                * GS
            )
            for r in range(BM):
                ri = ir + r
                rt = chunk * BT + ri
                xr = tl.load(
                    X + b * xs0 + rt * xs1 + h * xs2 + c * xs3,
                    (rt < T) & (ri < BT) & (c < D),
                    0,
                ).to(tl.float32)
                gr = (
                    tl.load(
                        G + b * SG[0] + rt * SG[1] + h * SG[2] + c * SG[3],
                        (rt < T) & (ri < BT) & (c < D),
                        0,
                    ).to(tl.float32)
                    * GS
                )
                decay = tl.exp2(gr[None, :] - gj)
                v = _weighted_sum(xr[None, :], kj, decay)
                if ISQ:
                    factor = SCALE
                else:
                    beta = tl.load(
                        BETA + b * SB[0] + rt * SB[1] + h * SB[2],
                        (rt < T) & (ri < BT),
                        0,
                    ).to(tl.float32)
                    factor = beta
                v = tl.where(
                    (ri + isq.to(tl.int32) > j) & (tj < T), v * factor, 0.0
                )
                out = ((b * T + rt) * H + h) * BT + j
                valid = (rt < T) & (ri < BT) & (j < BT)
                tl.store(Y + out, v, valid)
    if SCHEDULE < 3:
        if BT == 64:
            if p % 3 == 1:
                zr = chunk * BT + tl.arange(0, BM)
                zc = 32 + tl.arange(0, BN)
                zout = ((b * T + zr[:, None]) * H + h) * BT + zc[None, :]
                zeros = tl.full((BM, BN), 0.0, tl.float32)
                tl.store(Y + zout, zeros, zr[:, None] < T)


@triton.jit
def exact_role_compute(
    Q, K, G, BETA, A, AQ, META: tl.constexpr, TOTAL: tl.constexpr
):
    raw = tl.program_id(0)
    if META[2] <= 64:
        p = raw // 2
        role = raw % 2 == 1
    else:
        p = raw % TOTAL
        role = raw // TOTAL == 1
    if role:
        metax_small_role_inline(Q, K, G, BETA, A, AQ, META, 64, 1, 1, p, True)
    else:
        metax_small_role_inline(Q, K, G, BETA, A, AQ, META, 64, 1, 1, p, False)


def singlechunk_kda_scaled_dot_kkt(
    q, k, gk, beta, scale, gk_scale, chunk_size
):
    b, t, h, d = k.shape
    bt = int(chunk_size)
    a = torch.empty((b, t, h, bt), device=k.device, dtype=torch.float32)
    aq = torch.empty_like(a)
    bm, bn = (32, 32)
    tiles = 3 if bt == 64 else triton.cdiv(bt, bm) * triton.cdiv(bt, bn)
    total = b * h * triton.cdiv(t, bt) * tiles
    if (
        q.is_contiguous()
        and k.is_contiguous()
        and gk.is_contiguous()
        and beta.is_contiguous()
    ):
        meta = (t, h, d, bt, scale, gk_scale)
    else:
        meta = (
            t,
            h,
            d,
            bt,
            q.stride(),
            k.stride(),
            gk.stride(),
            beta.stride(),
            scale,
            gk_scale,
        )
    if total:
        single_compute.run(
            q,
            k,
            gk,
            beta,
            a,
            aq,
            meta,
            num_warps=4,
            num_stages=1,
            enable_fp_fusion=False,
            grid=(total, 2),
            warmup=False,
        )
    return (a, aq)


@triton.jit
def tile_compute(
    Q,
    K,
    G,
    BETA,
    A,
    AQ,
    META: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    PART_SIZE: tl.constexpr,
    ACC_ORDER: tl.constexpr,
    ENCODING: tl.constexpr,
):
    T: tl.constexpr = META[0]
    H: tl.constexpr = META[1]
    D: tl.constexpr = META[2]
    BT: tl.constexpr = META[3]
    if len(META) == 6:
        SQ: tl.constexpr = (T * H * D, H * D, D, 1)
        SK: tl.constexpr = SQ
        SG: tl.constexpr = SQ
        SB: tl.constexpr = (T * H, H, 1)
        SCALE: tl.constexpr = META[4]
        GS: tl.constexpr = META[5]
    else:
        SQ: tl.constexpr = META[4]
        SK: tl.constexpr = META[5]
        SG: tl.constexpr = META[6]
        SB: tl.constexpr = META[7]
        SCALE: tl.constexpr = META[8]
        GS: tl.constexpr = META[9]
    BM: tl.constexpr = BLOCK_SIZE
    BN: tl.constexpr = BLOCK_SIZE
    BD: tl.constexpr = (
        32 if D <= 32 else 64 if D <= 64 else 128 if D <= 128 else 256
    )
    p = tl.program_id(0)
    nc: tl.constexpr = triton.cdiv(BT, BN)
    nr: tl.constexpr = triton.cdiv(BT, BM)
    nt: tl.constexpr = triton.cdiv(T, BT)
    GC: tl.constexpr = 1 if nt % 1 == 0 else 1
    if BT == 64 and BM == 32 and (BN == 32):
        local = p // H % 3
        jc = tl.where(local == 2, 32, 0)
        ir = tl.where(local == 0, 0, 32)
        chunk = p // (3 * H) % nt
        h = p % H
        b = p // (3 * nt * H)
    else:
        jc = p % nc * BN
        ir = p // nc % nr * BM
        chunk = (p // (nc * nr) // (H * GC) * GC + p // (nc * nr) % GC) % nt
        h = p // (nc * nr) // GC % H
        b = p // (nc * nr * nt * H)
    i = ir + tl.arange(0, BM)
    j = jc + tl.arange(0, BN)
    ti = chunk * BT + i
    tj = chunk * BT + j
    c = tl.arange(0, BD)
    offset = ((b * T + ti[:, None]) * H + h) * BT + j[None, :]
    mask = (ti[:, None] < T) & (i[:, None] < BT) & (j[None, :] < BT)
    if jc >= ir + BM:
        z = tl.full((BM, BN), 0, tl.float32)
        tl.store(A + offset, z, mask)
        tl.store(AQ + offset, z, mask)
    else:
        at = tl.minimum(chunk * BT + BT // 2, T - 1)
        anchor = (
            tl.load(G + b * SG[0] + at * SG[1] + h * SG[2]).to(tl.float32) * GS
        )
        BK: tl.constexpr = PART_SIZE
        ACCMODE: tl.constexpr = ACC_ORDER
        av = tl.full((BM, BN), 0.0, tl.float32)
        qv = tl.full((BM, BN), 0.0, tl.float32)
        if ACCMODE == 2:
            ac = tl.full((BM, BN), 0.0, tl.float32)
            qc = tl.full((BM, BN), 0.0, tl.float32)
        bounds = tl.full((BM,), 0, tl.uint32)
        q_energy = tl.full((BM,), 0.0, tl.float32)
        k_energy = tl.full((BM,), 0.0, tl.float32)
        bounded_growth = tl.full((), True, tl.int1)
        for part in tl.range(
            0, triton.cdiv(D, BK), loop_unroll_factor=1, num_stages=1
        ):
            ch = part * BK + tl.arange(0, BK)
            ml = (ti[:, None] < T) & (i[:, None] < BT) & (ch[None, :] < D)
            mr = (tj[:, None] < T) & (j[:, None] < BT) & (ch[None, :] < D)
            vk = tl.load(
                K
                + b * SK[0]
                + ti[:, None] * SK[1]
                + h * SK[2]
                + ch[None, :] * SK[3],
                ml,
                0,
            ).to(tl.float32)
            vg = (
                tl.load(
                    G
                    + b * SG[0]
                    + ti[:, None] * SG[1]
                    + h * SG[2]
                    + ch[None, :] * SG[3],
                    ml,
                    0,
                ).to(tl.float32)
                * GS
            )
            if jc == ir:
                rk = vk
                rg = vg
            else:
                rk = tl.load(
                    K
                    + b * SK[0]
                    + tj[:, None] * SK[1]
                    + h * SK[2]
                    + ch[None, :] * SK[3],
                    mr,
                    0,
                ).to(tl.float32)
                rg = (
                    tl.load(
                        G
                        + b * SG[0]
                        + tj[:, None] * SG[1]
                        + h * SG[2]
                        + ch[None, :] * SG[3],
                        mr,
                        0,
                    ).to(tl.float32)
                    * GS
                )
            vq = tl.load(
                Q
                + b * SQ[0]
                + ti[:, None] * SQ[1]
                + h * SQ[2]
                + ch[None, :] * SQ[3],
                ml,
                0,
            ).to(tl.float32)
            if BT == 64 and BM == 64:
                q_energy += tl.sum(vq * vq, 1)
                k_energy += tl.sum(vk * vk, 1)
                previous_time = chunk * BT + tl.maximum(i - 1, 0)
                previous_g = (
                    tl.load(
                        G
                        + b * SG[0]
                        + previous_time[:, None] * SG[1]
                        + h * SG[2]
                        + ch[None, :] * SG[3],
                        ml,
                        0,
                    ).to(tl.float32)
                    * GS
                )
                growth = tl.where(ml, vg - previous_g, 0.0)
                bounded_growth = bounded_growth & (
                    tl.min(tl.min((growth <= 0.0009765625).to(tl.int32), 1), 0)
                    > 0
                )
            if ENCODING == 0:
                point_bound = tl.maximum(
                    tl.maximum(tl.abs(vg - anchor), tl.abs(rg - anchor))
                    * 0.125,
                    tl.maximum(tl.maximum(tl.abs(vq), tl.abs(vk)), tl.abs(rk))
                    * 0.125,
                )
                raw = tl.where(
                    point_bound == point_bound, point_bound, float("inf")
                ).to(tl.uint32, bitcast=True)
                bounds = tl.maximum(bounds, tl.max(raw, 1))
            else:
                valid = (
                    (tl.abs(vg - anchor) < 8.0)
                    & (tl.abs(rg - anchor) < 8.0)
                    & (tl.abs(vq) < 6.0)
                    & (tl.abs(vk) < 6.0)
                    & (tl.abs(rk) < 6.0)
                )
                bad = (~valid).to(tl.uint32)
                if ENCODING == 1:
                    bounds = tl.maximum(bounds, tl.max(bad, 1))
                else:
                    bounds = bounds + tl.sum(bad, 1)
            left = tl.exp2(vg - anchor)
            right = tl.exp2(anchor - rg)
            keys = tl.trans(rk * right)
            lc = vk * left
            lh = lc.to(tl.float16)
            lr = (lc - lh.to(tl.float32)).to(tl.float16)
            rh = keys.to(tl.float16)
            rr = (keys - rh.to(tl.float32)).to(tl.float16)
            if ACCMODE == 0:
                av = tl.dot(lr, rh, av)
                av = tl.dot(lh, rr, av)
                av = tl.dot(lh, rh, av)
            elif ACCMODE == 1:
                piece = tl.dot(lr, rh)
                piece = tl.dot(lh, rr, piece)
                piece = tl.dot(lh, rh, piece)
                av = av + piece
            else:
                ac = tl.dot(lr, rh, ac)
                ac = tl.dot(lh, rr, ac)
                av = tl.dot(lh, rh, av)
            lc = vq * left
            lh = lc.to(tl.float16)
            lr = (lc - lh.to(tl.float32)).to(tl.float16)
            if ACCMODE == 0:
                qv = tl.dot(lr, rh, qv)
                qv = tl.dot(lh, rr, qv)
                qv = tl.dot(lh, rh, qv)
            elif ACCMODE == 1:
                piece = tl.dot(lr, rh)
                piece = tl.dot(lh, rr, piece)
                piece = tl.dot(lh, rh, piece)
                qv = qv + piece
            else:
                qc = tl.dot(lr, rh, qc)
                qc = tl.dot(lh, rr, qc)
                qv = tl.dot(lh, rh, qv)
        bound = tl.max(bounds, 0)
        if BT == 64 and BM == 64:
            guard_beta = tl.load(
                BETA + b * SB[0] + ti * SB[1] + h * SB[2],
                (ti < T) & (i < BT),
                0,
            ).to(tl.float32)
            bounded = (
                bounded_growth
                & (tl.max(q_energy, 0) < 256.0)
                & (tl.max(k_energy, 0) < 256.0)
                & (tl.max(tl.abs(guard_beta), 0) <= 1.0)
                & (tl.abs(SCALE) <= 1.0)
            )
        else:
            bounded = tl.full((), False, tl.int1)
        if (bound < (1065353216 if ENCODING == 0 else 1)) & bounded:
            if ACCMODE == 2:
                av = av + ac
                qv = qv + qc
            beta = tl.load(
                BETA + b * SB[0] + ti * SB[1] + h * SB[2],
                (ti < T) & (i < BT),
                0,
            ).to(tl.float32)
            av = tl.where(
                (i[:, None] > j[None, :]) & (tj[None, :] < T),
                av * beta[:, None],
                0.0,
            )
            qv = tl.where(
                (i[:, None] >= j[None, :]) & (tj[None, :] < T), qv * SCALE, 0.0
            )
            tl.store(A + offset, av, mask)
            tl.store(AQ + offset, qv, mask)
        else:
            mj = (tj[:, None] < T) & (j[:, None] < BT) & (c[None, :] < D)
            kj = tl.load(
                K
                + b * SK[0]
                + tj[:, None] * SK[1]
                + h * SK[2]
                + c[None, :] * SK[3],
                mj,
                0,
            ).to(tl.float32)
            gj = (
                tl.load(
                    G
                    + b * SG[0]
                    + tj[:, None] * SG[1]
                    + h * SG[2]
                    + c[None, :] * SG[3],
                    mj,
                    0,
                ).to(tl.float32)
                * GS
            )
            for r in range(BM):
                ri = ir + r
                rt = chunk * BT + ri
                qr = tl.load(
                    Q + b * SQ[0] + rt * SQ[1] + h * SQ[2] + c * SQ[3],
                    (rt < T) & (ri < BT) & (c < D),
                    0,
                ).to(tl.float32)
                kr = tl.load(
                    K + b * SK[0] + rt * SK[1] + h * SK[2] + c * SK[3],
                    (rt < T) & (ri < BT) & (c < D),
                    0,
                ).to(tl.float32)
                gr = (
                    tl.load(
                        G + b * SG[0] + rt * SG[1] + h * SG[2] + c * SG[3],
                        (rt < T) & (ri < BT) & (c < D),
                        0,
                    ).to(tl.float32)
                    * GS
                )
                decay = tl.exp2(gr[None, :] - gj)
                unsafe_av = _weighted_sum(kr[None, :], kj, decay)
                unsafe_qv = _weighted_sum(qr[None, :], kj, decay)
                beta = tl.load(
                    BETA + b * SB[0] + rt * SB[1] + h * SB[2],
                    (rt < T) & (ri < BT),
                    0,
                ).to(tl.float32)
                unsafe_av = tl.where(
                    (ri > j) & (tj < T), unsafe_av * beta, 0.0
                )
                unsafe_qv = tl.where(
                    (ri >= j) & (tj < T), unsafe_qv * SCALE, 0.0
                )
                out = ((b * T + rt) * H + h) * BT + j
                valid = (rt < T) & (ri < BT) & (j < BT)
                tl.store(A + out, unsafe_av, valid)
                tl.store(AQ + out, unsafe_qv, valid)
    if BT == 64 and BM == 32 and (BN == 32):
        if local == 1:
            zr = chunk * BT + tl.arange(0, BM)
            zc = 32 + tl.arange(0, BN)
            zout = ((b * T + zr[:, None]) * H + h) * BT + zc[None, :]
            zeros = tl.full((BM, BN), 0.0, tl.float32)
            tl.store(A + zout, zeros, zr[:, None] < T)
            tl.store(AQ + zout, zeros, zr[:, None] < T)


def tile_entry(
    q, k, gk, beta, scale, gk_scale, chunk_size, bm, bk, mode, encoding, warps
):
    b, t, h, d = k.shape
    bt = int(chunk_size)
    a = torch.empty((b, t, h, bt), device=k.device, dtype=torch.float32)
    aq = torch.empty_like(a)
    bm, bn = (bm, bm)
    tiles = (
        3
        if bt == 64 and bm == 32 and (bn == 32)
        else triton.cdiv(bt, bm) * triton.cdiv(bt, bn)
    )
    total = b * h * triton.cdiv(t, bt) * tiles
    if (
        q.is_contiguous()
        and k.is_contiguous()
        and gk.is_contiguous()
        and beta.is_contiguous()
    ):
        meta = (t, h, d, bt, scale, gk_scale)
    else:
        meta = (
            t,
            h,
            d,
            bt,
            q.stride(),
            k.stride(),
            gk.stride(),
            beta.stride(),
            scale,
            gk_scale,
        )
    if total:
        tile_compute.run(
            q,
            k,
            gk,
            beta,
            a,
            aq,
            meta,
            bm,
            bk,
            mode,
            encoding,
            num_warps=warps,
            enable_fp_fusion=False,
            grid=(total,),
            warmup=False,
        )
    return (a, aq)


def chunk_kda_scaled_dot_kkt(q, k, gk, beta, scale, gk_scale, chunk_size):
    if k.shape[0] * k.shape[1] * k.shape[2] < 4096:
        return singlechunk_kda_scaled_dot_kkt(
            q, k, gk, beta, scale, gk_scale, chunk_size
        )
    return tile_entry(
        q, k, gk, beta, scale, gk_scale, chunk_size, 64, 64, 0, 1, 4
    )


__all__ = ["chunk_kda_scaled_dot_kkt"]
