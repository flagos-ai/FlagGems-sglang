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
def _product_error(x, y, product):
    xh = (x.to(tl.int32, bitcast=True) & -4096).to(tl.float32, bitcast=True)
    yh = (y.to(tl.int32, bitcast=True) & -4096).to(tl.float32, bitcast=True)
    xl = x - xh
    yl = y - yh
    return xh * yh - product + xh * yl + xl * yh + xl * yl


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
def _reduce_stage(high, low):
    half: tl.constexpr = high.shape[2] // 2
    index = tl.broadcast_to(
        (2 * tl.arange(0, half))[None, None, :],
        (high.shape[0], high.shape[1], half),
    )
    ah = tl.gather(high, index, 2)
    bh = tl.gather(high, index + 1, 2)
    al = tl.gather(low, index, 2)
    bl = tl.gather(low, index + 1, 2)
    return _pair_sum(ah, al, bh, bl)


@triton.jit
def _weighted_sum(
    x, y, decay, axis: tl.constexpr, ZERO_LOW: tl.constexpr = False
):
    product = x * y
    if ZERO_LOW:
        product_low = product - product
    else:
        product_low = _product_error(x, y, product)
    high = product * decay
    low = _product_error(product, decay, high) + product_low * decay
    tl.static_assert(axis == 2)
    if high.shape[2] > 1:
        high, low = _reduce_stage(high, low)
    if high.shape[2] > 1:
        high, low = _reduce_stage(high, low)
    if high.shape[2] > 1:
        high, low = _reduce_stage(high, low)
    if high.shape[2] > 1:
        high, low = _reduce_stage(high, low)
    if high.shape[2] > 1:
        high, low = _reduce_stage(high, low)
    if high.shape[2] > 1:
        high, low = _reduce_stage(high, low)
    if high.shape[2] > 1:
        high, low = _reduce_stage(high, low)
    if high.shape[2] > 1:
        high, low = _reduce_stage(high, low)
    return tl.sum(high + low, 2)


@triton.jit
def _compensated_group(
    Q,
    K,
    G,
    BETA,
    A,
    AQ,
    b,
    h,
    chunk,
    row_start,
    row_end,
    T: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    SCALE: tl.constexpr,
    GS: tl.constexpr,
    SQ: tl.constexpr,
    SK: tl.constexpr,
    SG: tl.constexpr,
    SB: tl.constexpr,
):
    c = tl.arange(0, D)[None, None, :]
    for local_row in range(row_end - row_start):
        r = row_start // 16 + local_row * 4
        rt = chunk * 64 + r
        qi = tl.load(Q + b * SQ[0] + rt * SQ[1] + h * SQ[2] + c * SQ[3]).to(
            tl.float32
        )
        ki = tl.load(K + b * SK[0] + rt * SK[1] + h * SK[2] + c * SK[3]).to(
            tl.float32
        )
        gi = (
            tl.load(G + b * SG[0] + rt * SG[1] + h * SG[2] + c * SG[3]).to(
                tl.float32
            )
            * GS
        )
        beta_i = tl.load(BETA + b * SB[0] + rt * SB[1] + h * SB[2]).to(
            tl.float32
        )
        zcol = tl.arange(0, 64)
        zoffset = ((b * T + rt) * H + h) * 64 + zcol
        tl.store(A + zoffset, tl.full((64,), 0.0, tl.float32))
        tl.store(AQ + zoffset, tl.full((64,), 0.0, tl.float32))
        for col in range((r + 4) // 4):
            j = col * 4 + tl.arange(0, 4)
            jt = chunk * 64 + j[None, :, None]
            kj = tl.load(
                K + b * SK[0] + jt * SK[1] + h * SK[2] + c * SK[3]
            ).to(tl.float32)
            gj = (
                tl.load(G + b * SG[0] + jt * SG[1] + h * SG[2] + c * SG[3]).to(
                    tl.float32
                )
                * GS
            )
            decay = tl.exp2(gi - gj)
            av = (
                tl.sum(
                    _weighted_sum(
                        ki,
                        kj,
                        decay,
                        2,
                        ZERO_LOW=K.dtype.element_ty == tl.bfloat16,
                    ),
                    0,
                )
                * beta_i
            )
            qv = (
                tl.sum(
                    _weighted_sum(
                        qi,
                        kj,
                        decay,
                        2,
                        ZERO_LOW=Q.dtype.element_ty == tl.bfloat16
                        and K.dtype.element_ty == tl.bfloat16,
                    ),
                    0,
                )
                * SCALE
            )
            qv = tl.where(j <= r, qv, 0.0)
            av = tl.where(j < r, av, 0.0)
            offset = ((b * T + rt) * H + h) * 64 + j
            tl.store(A + offset, av)
            tl.store(AQ + offset, qv)


@triton.jit(do_not_specialize=["Q", "K", "G", "BETA", "A", "AQ"])
def _compute(Q, K, G, BETA, A, AQ, META: tl.constexpr):
    T: tl.constexpr = META[0]
    H: tl.constexpr = META[1]
    D: tl.constexpr = META[2]
    BT: tl.constexpr = META[3]
    SQ: tl.constexpr = META[4]
    SK: tl.constexpr = META[5]
    SG: tl.constexpr = META[6]
    SB: tl.constexpr = META[7]
    SCALE: tl.constexpr = META[8]
    GS: tl.constexpr = META[9]
    TOTAL: tl.constexpr = META[10]
    BM: tl.constexpr = META[11]
    BD: tl.constexpr = META[12]
    BN: tl.constexpr = 1
    NC: tl.constexpr = triton.cdiv(BT, BN)
    NR: tl.constexpr = triton.cdiv(BT, BM)
    NT: tl.constexpr = triton.cdiv(T, BT)
    for p in range(tl.program_id(0), TOTAL, tl.num_programs(0)):
        if False:
            local = p % 36
            row = ((tl.sqrt((8 * local + 1).to(tl.float32)) - 1) * 0.5).to(
                tl.int32
            )
            col = local - row * (row + 1) // 2
            ir = row * BM
            jc = col * BN
            group = p // 36
        else:
            jc = p % NC * BN
            ir = p // NC % NR * BM
            group = p // (NC * NR)
        chunk = group % NT
        h = group // NT % H
        b = group // (NT * H)
        i = ir + tl.arange(0, BM)
        j = jc + tl.arange(0, BN)
        ti = chunk * BT + i
        tj = chunk * BT + j
        c = tl.arange(0, BD)
        out = ((b * T + ti[:, None]) * H + h) * BT + j[None, :]
        mask = (ti[:, None] < T) & (i[:, None] < BT) & (j[None, :] < BT)
        if jc >= ir + BM:
            zeros = tl.full((BM, BN), 0.0, tl.float32)
            tl.store(A + out, zeros, mask)
            tl.store(AQ + out, zeros, mask)
        else:
            mi = (ti[:, None] < T) & (i[:, None] < BT) & (c[None, :] < D)
            mj = (tj[:, None] < T) & (j[:, None] < BT) & (c[None, :] < D)
            qi = tl.load(
                Q
                + b * SQ[0]
                + ti[:, None] * SQ[1]
                + h * SQ[2]
                + c[None, :] * SQ[3],
                mi,
                0,
            ).to(tl.float32)
            ki = tl.load(
                K
                + b * SK[0]
                + ti[:, None] * SK[1]
                + h * SK[2]
                + c[None, :] * SK[3],
                mi,
                0,
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
            qi = tl.where(mi, qi, 0.0)
            ki = tl.where(mi, ki, 0.0)
            gi = tl.where(mi, gi, 0.0)
            kj = tl.where(mj, kj, 0.0)
            gj = tl.where(mj, gj, 0.0)
            decay = tl.exp2(gi[:, None, :] - gj[None, :, :])
            av = _weighted_sum(ki[:, None, :], kj[None, :, :], decay, 2)
            qv = _weighted_sum(qi[:, None, :], kj[None, :, :], decay, 2)
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
            tl.store(A + out, av, mask)
            tl.store(AQ + out, qv, mask)
        if False:
            upper = ((b * T + tj[:, None]) * H + h) * BT + i[None, :]
            upper_mask = (
                (tj[:, None] < T) & (j[:, None] < BT) & (i[None, :] < BT)
            )
            zero = tl.full((BN, BM), 0.0, tl.float32)
            tl.store(A + upper, zero, upper_mask)
            tl.store(AQ + upper, zero, upper_mask)


def _direct(q, k, gk, beta, scale, gk_scale, chunk_size):
    b, t, h, d = k.shape
    bc = 1
    bt = int(chunk_size)
    a = k.new_empty((b, t, h, bt), dtype=torch.float32)
    aq = torch.empty_like(a)
    total = (
        b * h * ((t + bt - 1) // bt) * 36
        if bt == 64 and bc == 8
        else b * h * ((t + bt - 1) // bt) * ((bt + bc - 1) // bc) * bt
    )
    if total:
        _compute.run(
            q,
            k,
            gk,
            beta,
            a,
            aq,
            (
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
                total,
                bc,
                max(32, triton.next_power_of_2(d)),
            ),
            num_warps=4,
            enable_fp_fusion=False,
            multibuffer=False,
            grid=(min(total, 40),),
            warmup=False,
        )
    return (a, aq)


@triton.jit(do_not_specialize=["Q", "K", "G", "LQ"])
def _prepare(Q, K, G, LQ, META: tl.constexpr):
    LK = LQ
    RK = LQ
    FLAGS = LQ
    T: tl.constexpr = META[0]
    H: tl.constexpr = META[1]
    D: tl.constexpr = META[2]
    GS: tl.constexpr = META[3]
    TOTAL: tl.constexpr = META[4]
    BR: tl.constexpr = META[5]
    WORK_N: tl.constexpr = META[6]
    LK = LK + WORK_N
    RK = RK + 2 * WORK_N
    FLAGS = FLAGS + 3 * WORK_N
    for order in range(tl.program_id(0), TOTAL, tl.num_programs(0)):
        nt0: tl.constexpr = T // 64
        parts: tl.constexpr = 64 // BR
        logical_group = (
            order // (nt0 * H * parts) * H + order % H
        ) * nt0 + order // (H * parts) % nt0
        tile = logical_group * parts + order // H % parts
        group = tile // (64 // BR)
        part = tile % (64 // BR)
        nt: tl.constexpr = T // 64
        chunk = group % nt
        h = group // nt % H
        b = group // (nt * H)
        r = part * BR + tl.arange(0, BR)
        c = tl.arange(0, D)
        addr = ((b * T + chunk * 64 + r[:, None]) * H + h) * D + c[None, :]
        q = tl.load(Q + addr).to(tl.float32)
        k = tl.load(K + addr).to(tl.float32)
        g = tl.load(G + addr).to(tl.float32) * GS
        magnitude = tl.maximum(
            tl.max(tl.max(tl.abs(q), 1), 0), tl.max(tl.max(tl.abs(k), 1), 0)
        )
        anchor = tl.where(
            tl.load(G + ((b * T + chunk * 64 + 32) * H + h) * D).to(tl.float32)
            * GS
            < 0.0,
            -4.0,
            4.0,
        )
        centered = g - anchor
        radius = tl.max(tl.max(tl.abs(centered), 1), 0)
        previous_r = tl.maximum(r - 1, 0)
        previous_addr = (
            (b * T + chunk * 64 + previous_r[:, None]) * H + h
        ) * D + c[None, :]
        previous_g = tl.load(G + previous_addr).to(tl.float32) * GS
        bounded_growth = tl.min(
            tl.min((g - previous_g <= 0.0009765625).to(tl.int32), 1), 0
        )
        energy = tl.maximum(
            tl.max(tl.sum(q * q, 1), 0), tl.max(tl.sum(k * k, 1), 0)
        )
        safe = (
            (radius < 8.0)
            & (magnitude < 6.0)
            & bounded_growth
            & (energy < 256.0)
        )
        delta = tl.where(safe, centered, 0.0)
        q = tl.where(safe, q, 0.0)
        k = tl.where(safe, k, 0.0)
        left = tl.exp2(delta)
        right = tl.exp2(-delta)
        out = (group * 64 + r[:, None]) * D + c[None, :]
        tl.store(LQ + out, q * left)
        tl.store(LK + out, k * left)
        tl.store(RK + out, k * right)
        tl.store(FLAGS + tile, safe.to(tl.int32))


@triton.jit(
    do_not_specialize=["Q", "K", "G", "LQ", "A", "AQ", "BETA", "GROUP_OFFSET"]
)
def _finish(Q, K, G, LQ, A, AQ, BETA, META: tl.constexpr, GROUP_OFFSET):
    LK = LQ
    RK = LQ
    FLAGS = LQ
    T: tl.constexpr = META[0]
    H: tl.constexpr = META[1]
    D: tl.constexpr = META[2]
    SB: tl.constexpr = META[4]
    SCALE: tl.constexpr = META[5]
    WORK_N: tl.constexpr = META[7]
    LK = LK + WORK_N
    RK = RK + 2 * WORK_N
    FLAGS = FLAGS + 3 * WORK_N
    group = tl.program_id(0) + GROUP_OFFSET
    valid = tl.load(FLAGS + group * 4 + tl.arange(0, 4))
    safe = tl.min(valid, 0)
    nt: tl.constexpr = T // 64
    chunk = group % nt
    h = group // nt % H
    b = group // (nt * H)
    guard_rows = tl.arange(0, 64)
    guard_beta = tl.load(
        BETA + b * SB[0] + (chunk * 64 + guard_rows) * SB[1] + h * SB[2]
    ).to(tl.float32)
    safe = (
        (safe > 0)
        & (tl.min((tl.abs(guard_beta) <= 1.0).to(tl.int32), 0) > 0)
        & (tl.abs(SCALE) <= 1.0)
    )
    if safe:
        si = tl.arange(0, 64)
        sj = tl.arange(0, 64)
        sc = tl.arange(0, D)
        sk = tl.load(LK + (group * 64 + si[:, None]) * D + sc[None, :])
        sr = tl.load(RK + (group * 64 + sj[None, :]) * D + sc[:, None])
        sav = tl.dot(sk, sr, input_precision="ieee")
        sbeta = tl.load(
            BETA + b * SB[0] + (chunk * 64 + si) * SB[1] + h * SB[2]
        ).to(tl.float32)
        sav = tl.where(sj[None, :] < si[:, None], sav * sbeta[:, None], 0.0)
        sout = ((b * T + chunk * 64 + si[:, None]) * H + h) * 64 + sj[None, :]
        tl.store(A + sout, sav)
        sq = tl.load(LQ + (group * 64 + si[:, None]) * D + sc[None, :])
        sqv = tl.dot(sq, sr, input_precision="ieee")
        sqv = tl.where(sj[None, :] <= si[:, None], sqv * SCALE, 0.0)
        tl.store(AQ + sout, sqv)


@triton.jit(do_not_specialize=["Q", "K", "G", "LQ", "A", "AQ", "BETA"])
def _repair(Q, K, G, LQ, A, AQ, BETA, META: tl.constexpr):
    T: tl.constexpr = META[0]
    H: tl.constexpr = META[1]
    D: tl.constexpr = META[2]
    GS: tl.constexpr = META[3]
    SB: tl.constexpr = META[4]
    SCALE: tl.constexpr = META[5]
    GROUPS: tl.constexpr = META[6]
    WORK_N: tl.constexpr = META[7]
    FLAGS = LQ + 3 * WORK_N
    for order in range(tl.program_id(0), GROUPS * 4, tl.num_programs(0)):
        group = order // 4
        row_start = order % 4 * 16
        flags = tl.load(FLAGS + group * 4 + tl.arange(0, 4))
        nt: tl.constexpr = T // 64
        chunk = group % nt
        h = group // nt % H
        b = group // (nt * H)
        rows = tl.arange(0, 64)
        beta_rows = tl.load(
            BETA + b * SB[0] + (chunk * 64 + rows) * SB[1] + h * SB[2]
        ).to(tl.float32)
        safe = (
            (tl.min(flags, 0) > 0)
            & (tl.min((tl.abs(beta_rows) <= 1.0).to(tl.int32), 0) > 0)
            & (tl.abs(SCALE) <= 1.0)
        )
        if not safe:
            stride: tl.constexpr = (T * H * D, H * D, D, 1)
            _compensated_group(
                Q,
                K,
                G,
                BETA,
                A,
                AQ,
                b,
                h,
                chunk,
                row_start,
                row_start + 16,
                T,
                H,
                D,
                SCALE,
                GS,
                stride,
                stride,
                stride,
                SB,
            )


def _mixed(q, k, gk, beta, scale, gk_scale, chunk_size):
    b, t, h, d = k.shape
    if chunk_size != 64 or t % 64 or d not in (64, 128):
        return _direct(q, k, gk, beta, scale, gk_scale, chunk_size)
    if (
        not q.is_contiguous()
        or not k.is_contiguous()
        or (not gk.is_contiguous())
    ):
        q, k, gk = _pack_inputs(q, k, gk)
    a = k.new_empty((b, t, h, 64), dtype=torch.float32)
    aq = torch.empty_like(a)
    groups = b * h * (t // 64)
    if groups:
        work_n = groups * 64 * d
        lq = k.new_empty((3 * work_n + groups * 4,), dtype=torch.float32)
        _prepare.run(
            q,
            k,
            gk,
            lq,
            (t, h, d, gk_scale, groups * 4, 16, work_n),
            num_warps=4,
            enable_fp_fusion=False,
            grid=(min(groups * 4, 40),),
            warmup=False,
        )
        for group_offset in range(0, groups, 65535):
            _finish.run(
                q,
                k,
                gk,
                lq,
                a,
                aq,
                beta,
                (t, h, d, gk_scale, beta.stride(), scale, groups, work_n),
                group_offset,
                num_warps=4,
                enable_fp_fusion=False,
                grid=(min(groups - group_offset, 65535),),
                warmup=False,
                multibuffer=False,
            )
        _repair.run(
            q,
            k,
            gk,
            lq,
            a,
            aq,
            beta,
            (t, h, d, gk_scale, beta.stride(), scale, groups, work_n),
            num_warps=4,
            enable_fp_fusion=False,
            grid=(min(groups * 4, 40),),
            warmup=False,
            multibuffer=False,
        )
    return (a, aq)


def _fused(q, k, gk, beta, scale, gk_scale, chunk_size):
    return _mixed(q, k, gk, beta, scale, gk_scale, chunk_size)


def chunk_kda_scaled_dot_kkt(q, k, gk, beta, scale, gk_scale, chunk_size):
    return _mixed(q, k, gk, beta, scale, gk_scale, chunk_size)


@triton.jit(do_not_specialize=["Q", "K", "G", "CQ", "CK", "CG"])
def _pack(Q, K, G, CQ, CK, CG, META: tl.constexpr):
    T: tl.constexpr = META[0]
    H: tl.constexpr = META[1]
    D: tl.constexpr = META[2]
    N: tl.constexpr = META[3]
    SQ: tl.constexpr = META[4]
    SK: tl.constexpr = META[5]
    SG: tl.constexpr = META[6]
    BN: tl.constexpr = 256
    for block in range(tl.program_id(0), tl.cdiv(N, BN), tl.num_programs(0)):
        index = (block * BN + tl.arange(0, BN)).to(tl.int64)
        c = index % D
        h = index // D % H
        t = index // (D * H) % T
        b = index // (D * H * T)
        q = tl.load(
            Q + b * SQ[0] + t * SQ[1] + h * SQ[2] + c * SQ[3], index < N, 0
        )
        tl.store(CQ + index, q, index < N)
        k = tl.load(
            K + b * SK[0] + t * SK[1] + h * SK[2] + c * SK[3], index < N, 0
        )
        tl.store(CK + index, k, index < N)
        g = tl.load(
            G + b * SG[0] + t * SG[1] + h * SG[2] + c * SG[3], index < N, 0
        )
        tl.store(CG + index, g, index < N)


def _pack_inputs(q, k, gk):
    b, t, h, d = k.shape
    cq = q.new_empty(q.shape)
    ck = k.new_empty(k.shape)
    cg = gk.new_empty(gk.shape)
    n = b * t * h * d
    if n:
        meta = (t, h, d, n, q.stride(), k.stride(), gk.stride())
        if (
            b * t * h >= (2048 if d >= 128 else 3072)
            and q.dtype in (torch.float32, torch.float16, torch.bfloat16)
            and (k.dtype in (torch.float32, torch.float16, torch.bfloat16))
            and (gk.dtype in (torch.float32, torch.float16, torch.bfloat16))
            and (0 < q.stride(3) <= 4)
            and (0 < k.stride(3) <= 4)
            and (0 < gk.stride(3) <= 4)
        ):
            qbits = torch.int32 if q.dtype == torch.float32 else torch.int16
            kbits = torch.int32 if k.dtype == torch.float32 else torch.int16
            gbits = torch.int32 if gk.dtype == torch.float32 else torch.int16
            _pack_dense.run(
                q.view(qbits),
                k.view(kbits),
                gk.view(gbits),
                cq.view(qbits),
                ck.view(kbits),
                cg.view(gbits),
                meta,
                grid=(min(triton.cdiv(b * t * h, 4), 40),),
                num_warps=4,
                enable_fp_fusion=False,
                multibuffer=False,
                warmup=False,
            )
        else:
            _pack.run(
                q,
                k,
                gk,
                cq,
                ck,
                cg,
                meta,
                grid=(min(triton.cdiv(n, 256), 40),),
                num_warps=4,
                enable_fp_fusion=False,
                multibuffer=False,
                warmup=False,
            )
    return (cq, ck, cg)


@triton.jit
def _dense_rows(
    X,
    Y,
    row,
    b,
    t,
    h,
    T: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    ROWS: tl.constexpr,
    STRIDE: tl.constexpr,
    BR: tl.constexpr,
):
    WIDTH: tl.constexpr = triton.next_power_of_2((D - 1) * STRIDE[3] + 1)
    CD: tl.constexpr = triton.next_power_of_2(D)
    physical = tl.arange(0, WIDTH)
    base = b * STRIDE[0] + t * STRIDE[1] + h * STRIDE[2]
    values = tl.load(
        X + base[:, None] + physical[None, :],
        (row[:, None] < ROWS) & (physical[None, :] <= (D - 1) * STRIDE[3]),
        0,
    )
    c = tl.arange(0, CD)
    indices = tl.broadcast_to(
        (tl.minimum(c, D - 1) * STRIDE[3])[None, :], (BR, CD)
    )
    if X.dtype.element_ty == tl.int32:
        low = (values & 65535).to(tl.float32)
        high = (values >> 16).to(tl.float32)
        selected_low = tl.gather(low, indices, 1).to(tl.int32)
        selected_high = tl.gather(high, indices, 1).to(tl.int32)
        selected = selected_low | selected_high << 16
    else:
        selected = tl.gather(values.to(tl.float32), indices, 1).to(tl.int16)
    tl.store(
        Y + row[:, None] * D + c[None, :],
        selected,
        (row[:, None] < ROWS) & (c[None, :] < D),
    )


@triton.jit(do_not_specialize=["Q", "K", "G", "CQ", "CK", "CG"])
def _pack_dense(Q, K, G, CQ, CK, CG, META: tl.constexpr):
    T: tl.constexpr = META[0]
    H: tl.constexpr = META[1]
    D: tl.constexpr = META[2]
    N: tl.constexpr = META[3]
    SQ: tl.constexpr = META[4]
    SK: tl.constexpr = META[5]
    SG: tl.constexpr = META[6]
    ROWS: tl.constexpr = N // D
    BR: tl.constexpr = 4
    for block in range(
        tl.program_id(0), tl.cdiv(ROWS, BR), tl.num_programs(0)
    ):
        row = (block * BR + tl.arange(0, BR)).to(tl.int64)
        h = row % H
        t = row // H % T
        b = row // (H * T)
        _dense_rows(Q, CQ, row, b, t, h, T, H, D, ROWS, SQ, BR)
        _dense_rows(K, CK, row, b, t, h, T, H, D, ROWS, SK, BR)
        _dense_rows(G, CG, row, b, t, h, T, H, D, ROWS, SG, BR)


__all__ = ["chunk_kda_scaled_dot_kkt"]
