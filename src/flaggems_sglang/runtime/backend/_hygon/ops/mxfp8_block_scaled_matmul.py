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
def _decode(u, s):
    u = u.to(tl.int32) & 255
    s = s.to(tl.int32) & 255
    e = u >> 3 & 15
    m = u & 7
    normal = (e + 120 << 23 | m << 20).to(tl.float32, bitcast=True)
    v = tl.where(e == 0, m.to(tl.float32) * 0.001953125, normal)
    v = tl.where(u & 128 != 0, -v, v)
    v = tl.where(u & 127 == 127, float("nan"), v)
    scale = tl.where(s == 0, 4194304, s << 23).to(tl.float32, bitcast=True)
    return v * scale


@triton.jit
def _decode_halfbase(u, s):
    u = u.to(tl.int32) & 255
    s = s.to(tl.int32) & 255
    half_bits = (u & 128) << 8 | (u & 127) << 7
    value = half_bits.to(tl.uint16).to(tl.float16, bitcast=True).to(tl.float32)
    scale = (s + 8 << 23).to(tl.float32, bitcast=True)
    return tl.where(u & 127 == 127, float("nan"), value * scale)


@triton.jit
def hg_fast_merged(A, B, FA, Output, RA, RAS, RB, RBS, META: tl.constexpr):
    M: tl.constexpr = META[0]
    N: tl.constexpr = META[1]
    K: tl.constexpr = META[2]
    BM: tl.constexpr = META[3]
    BN: tl.constexpr = META[4]
    BK: tl.constexpr = META[5]
    A0: tl.constexpr = META[6]
    A1: tl.constexpr = META[7]
    AS0: tl.constexpr = META[8]
    AS1: tl.constexpr = META[9]
    B0: tl.constexpr = META[10]
    B1: tl.constexpr = META[11]
    BS0: tl.constexpr = META[12]
    BS1: tl.constexpr = META[13]
    RA = RA.to(tl.pointer_type(tl.int8))
    RB = RB.to(tl.pointer_type(tl.int8))
    ADDRESS64: tl.constexpr = (
        (M + 4 * BM) * N + (N + 4 * BN) >= 2147483648
        or (M + 4 * BM) * K + (K + 4 * BK) >= 2147483648
        or (N + 4 * BN) * K + (K + 4 * BK) >= 2147483648
        or ((M + 4 * BM) * A0 + (K + 4 * BK) * A1 >= 2147483648)
        or ((M + 4 * BM) * AS0 + ((K + 4 * BK) // 32 + 1) * AS1 >= 2147483648)
        or ((N + 4 * BN) * B0 + (K + 4 * BK) * B1 >= 2147483648)
        or ((N + 4 * BN) * BS0 + ((K + 4 * BK) // 32 + 1) * BS1 >= 2147483648)
    )
    FB = FA + (tl.cast(M, tl.int64) if ADDRESS64 else M)
    pid = (
        tl.cast(tl.program_id(0), tl.int64) if ADDRESS64 else tl.program_id(0)
    )
    if M >= 512:
        num_n710: tl.constexpr = (N + BN - 1) // BN
        first_m710 = pid // (8 * num_n710) * 8
        group_m710 = tl.minimum(tl.cdiv(M, BM) - first_m710, 8)
        within710 = pid % (8 * num_n710)
        pid = (
            first_m710 + within710 % group_m710
        ) * num_n710 + within710 // group_m710
    rm = pid // tl.cdiv(N, BN) * BM + (
        tl.cast(tl.arange(0, BM), tl.int64) if ADDRESS64 else tl.arange(0, BM)
    )
    rn = pid % tl.cdiv(N, BN) * BN + (
        tl.cast(tl.arange(0, BN), tl.int64) if ADDRESS64 else tl.arange(0, BN)
    )
    rk = tl.cast(tl.arange(0, BK), tl.int64) if ADDRESS64 else tl.arange(0, BK)
    acc = tl.zeros((BM, BN), tl.float32)
    for k in range(tl.cdiv(K, BK)):
        kk = (tl.cast(k, tl.int64) if ADDRESS64 else k) * BK + rk
        av = tl.load(
            A + rm[:, None] * K + kk[None, :],
            (rm[:, None] < M) & (kk[None, :] < K),
            0,
        )
        bv = tl.load(
            B + rn[None, :] * K + kk[:, None],
            (rn[None, :] < N) & (kk[:, None] < K),
            0,
        )
        acc = tl.dot(av, bv, acc)
    tl.store(
        Output + rm[:, None] * N + rn[None, :],
        acc,
        (rm[:, None] < M) & (rn[None, :] < N),
    )
    tile_a = tl.min(tl.load(FA + rm, rm < M, 1), 0)
    tile_b = tl.min(tl.load(FB + rn, rn < N, 1), 0)
    if tile_a & tile_b == 0:
        for sub in range(BM // 16 * (BN // 32)):
            sr = (
                pid // tl.cdiv(N, BN) * BM
                + sub // (BN // 32) * 16
                + (
                    tl.cast(tl.arange(0, 16), tl.int64)
                    if ADDRESS64
                    else tl.arange(0, 16)
                )
            )
            sc = (
                pid % tl.cdiv(N, BN) * BN
                + sub % (BN // 32) * 32
                + (
                    tl.cast(tl.arange(0, 32), tl.int64)
                    if ADDRESS64
                    else tl.arange(0, 32)
                )
            )
            sub_a = tl.min(tl.load(FA + sr, sr < M, 1), 0)
            sub_b = tl.min(tl.load(FB + sc, sc < N, 1), 0)
            if sub_a & sub_b == 0:
                sk = (
                    tl.cast(tl.arange(0, 32), tl.int64)
                    if ADDRESS64
                    else tl.arange(0, 32)
                )
                repaired = tl.zeros((16, 32), tl.float32)
                for chunk in range(tl.cdiv(K, 32)):
                    kk = (
                        tl.cast(chunk, tl.int64) if ADDRESS64 else chunk
                    ) * 32 + sk
                    sa = (
                        tl.load(
                            RAS + sr[:, None] * AS0 + kk[None, :] // 32 * AS1,
                            (sr[:, None] < M) & (kk[None, :] < K),
                            127,
                        ).to(tl.int32)
                        & 255
                    )
                    sb = (
                        tl.load(
                            RBS + sc[None, :] * BS0 + kk[:, None] // 32 * BS1,
                            (sc[None, :] < N) & (kk[:, None] < K),
                            127,
                        ).to(tl.int32)
                        & 255
                    )
                    ua = tl.load(
                        RA + sr[:, None] * A0 + kk[None, :] * A1,
                        (sr[:, None] < M) & (kk[None, :] < K),
                        0,
                    )
                    ub = tl.load(
                        RB + sc[None, :] * B0 + kk[:, None] * B1,
                        (sc[None, :] < N) & (kk[:, None] < K),
                        0,
                    )
                    av = _decode(ua, sa)
                    bv = _decode(ub, sb)
                    repaired += tl.sum(av[:, :, None] * bv[None, :, :], 1)
                tl.store(
                    Output + sr[:, None] * N + sc[None, :],
                    repaired,
                    (sr[:, None] < M) & (sc[None, :] < N),
                )


@triton.jit
def hg_fast_fullmask_merged(
    A, B, FA, Output, RA, RAS, RB, RBS, META: tl.constexpr
):
    M: tl.constexpr = META[0]
    N: tl.constexpr = META[1]
    K: tl.constexpr = META[2]
    BM: tl.constexpr = META[3]
    BN: tl.constexpr = META[4]
    BK: tl.constexpr = META[5]
    A0: tl.constexpr = META[6]
    A1: tl.constexpr = META[7]
    AS0: tl.constexpr = META[8]
    AS1: tl.constexpr = META[9]
    B0: tl.constexpr = META[10]
    B1: tl.constexpr = META[11]
    BS0: tl.constexpr = META[12]
    BS1: tl.constexpr = META[13]
    RA = RA.to(tl.pointer_type(tl.int8))
    RB = RB.to(tl.pointer_type(tl.int8))
    ADDRESS64: tl.constexpr = (
        (M + 4 * BM) * N + (N + 4 * BN) >= 2147483648
        or (M + 4 * BM) * K + (K + 4 * BK) >= 2147483648
        or (N + 4 * BN) * K + (K + 4 * BK) >= 2147483648
        or ((M + 4 * BM) * A0 + (K + 4 * BK) * A1 >= 2147483648)
        or ((M + 4 * BM) * AS0 + ((K + 4 * BK) // 32 + 1) * AS1 >= 2147483648)
        or ((N + 4 * BN) * B0 + (K + 4 * BK) * B1 >= 2147483648)
        or ((N + 4 * BN) * BS0 + ((K + 4 * BK) // 32 + 1) * BS1 >= 2147483648)
    )
    FB = FA + (tl.cast(M, tl.int64) if ADDRESS64 else M)
    pid = (
        tl.cast(tl.program_id(0), tl.int64) if ADDRESS64 else tl.program_id(0)
    )
    if M >= 512:
        num_n710: tl.constexpr = (N + BN - 1) // BN
        first_m710 = pid // (8 * num_n710) * 8
        group_m710 = tl.minimum(tl.cdiv(M, BM) - first_m710, 8)
        within710 = pid % (8 * num_n710)
        pid = (
            first_m710 + within710 % group_m710
        ) * num_n710 + within710 // group_m710
    rm = pid // tl.cdiv(N, BN) * BM + (
        tl.cast(tl.arange(0, BM), tl.int64) if ADDRESS64 else tl.arange(0, BM)
    )
    rn = pid % tl.cdiv(N, BN) * BN + (
        tl.cast(tl.arange(0, BN), tl.int64) if ADDRESS64 else tl.arange(0, BN)
    )
    rk = tl.cast(tl.arange(0, BK), tl.int64) if ADDRESS64 else tl.arange(0, BK)
    acc = tl.zeros((BN, BM), tl.float32)
    for k in range(tl.cdiv(K, BK)):
        kk = (tl.cast(k, tl.int64) if ADDRESS64 else k) * BK + rk
        if M % BM == 0 and K % BK == 0:
            av = tl.load(A + rm[None, :] * K + kk[:, None])
        else:
            av = tl.load(
                A + rm[None, :] * K + kk[:, None],
                (rm[None, :] < M) & (kk[:, None] < K),
                0,
            )
        if N % BN == 0 and K % BK == 0:
            bv = tl.load(B + rn[:, None] * K + kk[None, :])
        else:
            bv = tl.load(
                B + rn[:, None] * K + kk[None, :],
                (rn[:, None] < N) & (kk[None, :] < K),
                0,
            )
        acc = tl.dot(bv, av, acc)
    if M % BM == 0 and N % BN == 0:
        tl.store(Output + rm[:, None] * N + rn[None, :], tl.trans(acc))
    else:
        tl.store(
            Output + rm[:, None] * N + rn[None, :],
            tl.trans(acc),
            (rm[:, None] < M) & (rn[None, :] < N),
        )
    tile_a = tl.min(tl.load(FA + rm, rm < M, 1), 0)
    tile_b = tl.min(tl.load(FB + rn, rn < N, 1), 0)
    if tile_a & tile_b == 0:
        for sub in range(BM // 16 * (BN // 32)):
            sr = (
                pid // tl.cdiv(N, BN) * BM
                + sub // (BN // 32) * 16
                + (
                    tl.cast(tl.arange(0, 16), tl.int64)
                    if ADDRESS64
                    else tl.arange(0, 16)
                )
            )
            sc = (
                pid % tl.cdiv(N, BN) * BN
                + sub % (BN // 32) * 32
                + (
                    tl.cast(tl.arange(0, 32), tl.int64)
                    if ADDRESS64
                    else tl.arange(0, 32)
                )
            )
            sub_a = tl.min(tl.load(FA + sr, sr < M, 1), 0)
            sub_b = tl.min(tl.load(FB + sc, sc < N, 1), 0)
            if sub_a & sub_b == 0:
                sk = (
                    tl.cast(tl.arange(0, 32), tl.int64)
                    if ADDRESS64
                    else tl.arange(0, 32)
                )
                repaired = tl.zeros((16, 32), tl.float32)
                for chunk in range(tl.cdiv(K, 32)):
                    kk = (
                        tl.cast(chunk, tl.int64) if ADDRESS64 else chunk
                    ) * 32 + sk
                    sa = (
                        tl.load(
                            RAS + sr[:, None] * AS0 + kk[None, :] // 32 * AS1,
                            (sr[:, None] < M) & (kk[None, :] < K),
                            127,
                        ).to(tl.int32)
                        & 255
                    )
                    sb = (
                        tl.load(
                            RBS + sc[None, :] * BS0 + kk[:, None] // 32 * BS1,
                            (sc[None, :] < N) & (kk[:, None] < K),
                            127,
                        ).to(tl.int32)
                        & 255
                    )
                    ua = tl.load(
                        RA + sr[:, None] * A0 + kk[None, :] * A1,
                        (sr[:, None] < M) & (kk[None, :] < K),
                        0,
                    )
                    ub = tl.load(
                        RB + sc[None, :] * B0 + kk[:, None] * B1,
                        (sc[None, :] < N) & (kk[:, None] < K),
                        0,
                    )
                    av = _decode(ua, sa)
                    bv = _decode(ub, sb)
                    repaired += tl.sum(av[:, :, None] * bv[None, :, :], 1)
                tl.store(
                    Output + sr[:, None] * N + sc[None, :],
                    repaired,
                    (sr[:, None] < M) & (sc[None, :] < N),
                )


@triton.jit
def hg_dequant_both_selected_math(
    A, AS, B, BS, AD, BD, FA, META: tl.constexpr
):
    M: tl.constexpr = META[0]
    K: tl.constexpr = META[1]
    A0: tl.constexpr = META[2]
    A1: tl.constexpr = META[3]
    AS0: tl.constexpr = META[4]
    AS1: tl.constexpr = META[5]
    B0: tl.constexpr = META[6]
    B1: tl.constexpr = META[7]
    BS0: tl.constexpr = META[8]
    BS1: tl.constexpr = META[9]
    BLOCK: tl.constexpr = META[10]
    HALFSHIFT: tl.constexpr = META[11]
    N: tl.constexpr = META[12]
    A = A.to(tl.pointer_type(tl.int8))
    B = B.to(tl.pointer_type(tl.int8))
    ADDRESS64: tl.constexpr = (
        (M + 1024) * N + (N + 1024) >= 2147483648
        or (M + 1024) * K + (2 * K + 4096) >= 2147483648
        or (N + 1024) * K + (2 * K + 4096) >= 2147483648
        or ((M + 1024) * A0 + (2 * K + 4096) * A1 >= 2147483648)
        or ((M + 1024) * AS0 + ((2 * K + 4096) // 32 + 1) * AS1 >= 2147483648)
        or ((N + 1024) * B0 + (2 * K + 4096) * B1 >= 2147483648)
        or ((N + 1024) * BS0 + ((2 * K + 4096) // 32 + 1) * BS1 >= 2147483648)
    )
    row = (
        tl.cast(tl.program_id(0), tl.int64) if ADDRESS64 else tl.program_id(0)
    )
    is_a = row < M
    cols = (
        tl.cast(tl.arange(0, BLOCK), tl.int64)
        if ADDRESS64
        else tl.arange(0, BLOCK)
    )
    src = tl.where(is_a, A + row * A0, B + (row - M) * B0)
    scale = tl.where(is_a, AS + row * AS0, BS + (row - M) * BS0)
    dst = tl.where(is_a, AD + row * K, BD + (row - M) * K)
    flag = FA + row
    stride = tl.where(is_a, A1, B1)
    scale_stride = tl.where(is_a, AS1, BS1)
    u = tl.load(src + cols * stride, cols < K, 0)
    s = (
        tl.load(scale + cols // 32 * scale_stride, cols < K, 127).to(tl.int32)
        & 255
    )
    if HALFSHIFT:
        value = _decode_halfbase(u, s).to(
            tl.bfloat16, fp_downcast_rounding="rtz"
        )
    else:
        value = _decode(u, s)
    tl.store(dst + cols, value, cols < K)
    safe = tl.min(((s >= 10) & (s <= 246)).to(tl.int32), 0)
    tl.store(flag, safe)


def mxfp8_block_scaled_matmul(a, a_scale, b, b_scale, output_dtype):
    m, k = a.shape
    n = b.shape[0]
    out = a.new_empty((m, n), dtype=output_dtype)
    if m == 0 or n == 0:
        return out
    if m < 512 and n <= 512 and (k <= 1024):
        _hg_lean_fused783.run(
            out,
            a,
            a_scale,
            b,
            b_scale,
            (
                m,
                n,
                k,
                64,
                64,
                64,
                a.stride(0),
                a.stride(1),
                a_scale.stride(0),
                a_scale.stride(1),
                b.stride(0),
                b.stride(1),
                b_scale.stride(0),
                b_scale.stride(1),
                max(
                    1, 0 if k == 0 else 1 << ((k + 31) // 32 - 1).bit_length()
                ),
            ),
            num_warps=4,
            num_stages=3,
            grid=((m + 63) // 64 * ((n + 63) // 64),),
            warmup=False,
        )
        return out
    ad = a.new_empty((m, k), dtype=torch.bfloat16)
    bd = a.new_empty((n, k), dtype=torch.bfloat16)
    fa = a.new_empty((m + n,), dtype=torch.int32)
    lean = m < 512 and n <= 512 and (k <= 1024)
    prepare = (
        _dns652_hg_dequant_both_selected_math
        if lean
        else hg_dequant_both_selected_math
    )
    fast = (
        _dns652_hg_fast_merged
        if lean
        else hg_fast_fullmask_merged if m >= 2048 else hg_fast_merged
    )
    a_bytes = a
    b_bytes = b
    a_stride0 = a.stride(0)
    a_stride1 = a.stride(1)
    a_scale_stride0 = a_scale.stride(0)
    a_scale_stride1 = a_scale.stride(1)
    b_stride0 = b.stride(0)
    b_stride1 = b.stride(1)
    b_scale_stride0 = b_scale.stride(0)
    b_scale_stride1 = b_scale.stride(1)
    prepare.run(
        a_bytes,
        a_scale,
        b_bytes,
        b_scale,
        ad,
        bd,
        fa,
        (
            m,
            k,
            a_stride0,
            a_stride1,
            a_scale_stride0,
            a_scale_stride1,
            b_stride0,
            b_stride1,
            b_scale_stride0,
            b_scale_stride1,
            max(32, 0 if k == 0 else 1 << (k - 1).bit_length()),
            m >= 512,
            n,
        ),
        num_warps=4,
        grid=(m + n,),
        warmup=False,
    )
    bm, bn, bk = (64, 64, 64)
    if m >= 2048:
        bm, bn, bk = (128, 128, 64)
        fast.run(
            ad,
            bd,
            fa,
            out,
            a_bytes,
            a_scale,
            b_bytes,
            b_scale,
            (
                m,
                n,
                k,
                bm,
                bn,
                bk,
                a_stride0,
                a_stride1,
                a_scale_stride0,
                a_scale_stride1,
                b_stride0,
                b_stride1,
                b_scale_stride0,
                b_scale_stride1,
            ),
            num_warps=8 if m >= 2048 else 4,
            num_stages=2 if m >= 2048 else 3,
            grid=((m + bm - 1) // bm * ((n + bn - 1) // bn),),
            warmup=False,
            sched_latency="mmac5-ds6",
        )
        return out
    elif m >= 512:
        bm, bn, bk = (128, 128, 32)
    fast.run(
        ad,
        bd,
        fa,
        out,
        a_bytes,
        a_scale,
        b_bytes,
        b_scale,
        (
            m,
            n,
            k,
            bm,
            bn,
            bk,
            a_stride0,
            a_stride1,
            a_scale_stride0,
            a_scale_stride1,
            b_stride0,
            b_stride1,
            b_scale_stride0,
            b_scale_stride1,
        ),
        num_warps=8 if m >= 2048 else 4,
        num_stages=2 if m >= 2048 else 3,
        grid=((m + bm - 1) // bm * ((n + bn - 1) // bn),),
        warmup=False,
    )
    return out


@triton.jit(do_not_specialize=["A", "AS", "B", "BS", "AD", "BD", "FA"])
def _dns652_hg_dequant_both_selected_math(
    A, AS, B, BS, AD, BD, FA, META: tl.constexpr
):
    M: tl.constexpr = META[0]
    K: tl.constexpr = META[1]
    A0: tl.constexpr = META[2]
    A1: tl.constexpr = META[3]
    AS0: tl.constexpr = META[4]
    AS1: tl.constexpr = META[5]
    B0: tl.constexpr = META[6]
    B1: tl.constexpr = META[7]
    BS0: tl.constexpr = META[8]
    BS1: tl.constexpr = META[9]
    BLOCK: tl.constexpr = META[10]
    HALFSHIFT: tl.constexpr = META[11]
    N: tl.constexpr = META[12]
    A = A.to(tl.pointer_type(tl.int8))
    B = B.to(tl.pointer_type(tl.int8))
    ADDRESS64: tl.constexpr = (
        (M + 1024) * N + (N + 1024) >= 2147483648
        or (M + 1024) * K + (2 * K + 4096) >= 2147483648
        or (N + 1024) * K + (2 * K + 4096) >= 2147483648
        or ((M + 1024) * A0 + (2 * K + 4096) * A1 >= 2147483648)
        or ((M + 1024) * AS0 + ((2 * K + 4096) // 32 + 1) * AS1 >= 2147483648)
        or ((N + 1024) * B0 + (2 * K + 4096) * B1 >= 2147483648)
        or ((N + 1024) * BS0 + ((2 * K + 4096) // 32 + 1) * BS1 >= 2147483648)
    )
    row = (
        tl.cast(tl.program_id(0), tl.int64) if ADDRESS64 else tl.program_id(0)
    )
    is_a = row < M
    cols = (
        tl.cast(tl.arange(0, BLOCK), tl.int64)
        if ADDRESS64
        else tl.arange(0, BLOCK)
    )
    src = tl.where(is_a, A + row * A0, B + (row - M) * B0)
    scale = tl.where(is_a, AS + row * AS0, BS + (row - M) * BS0)
    dst = tl.where(is_a, AD + row * K, BD + (row - M) * K)
    flag = FA + row
    stride = tl.where(is_a, A1, B1)
    scale_stride = tl.where(is_a, AS1, BS1)
    u = tl.load(src + cols * stride, cols < K, 0)
    s = (
        tl.load(scale + cols // 32 * scale_stride, cols < K, 127).to(tl.int32)
        & 255
    )
    if HALFSHIFT:
        value = _decode_halfbase(u, s)
    else:
        value = _decode(u, s)
    tl.store(dst + cols, value, cols < K)
    safe = tl.min(((s >= 10) & (s <= 246)).to(tl.int32), 0)
    tl.store(flag, safe)


@triton.jit(
    do_not_specialize=["A", "B", "FA", "Output", "RA", "RAS", "RB", "RBS"]
)
def _dns652_hg_fast_merged(
    A, B, FA, Output, RA, RAS, RB, RBS, META: tl.constexpr
):
    M: tl.constexpr = META[0]
    N: tl.constexpr = META[1]
    K: tl.constexpr = META[2]
    BM: tl.constexpr = META[3]
    BN: tl.constexpr = META[4]
    BK: tl.constexpr = META[5]
    A0: tl.constexpr = META[6]
    A1: tl.constexpr = META[7]
    AS0: tl.constexpr = META[8]
    AS1: tl.constexpr = META[9]
    B0: tl.constexpr = META[10]
    B1: tl.constexpr = META[11]
    BS0: tl.constexpr = META[12]
    BS1: tl.constexpr = META[13]
    RA = RA.to(tl.pointer_type(tl.int8))
    RB = RB.to(tl.pointer_type(tl.int8))
    ADDRESS64: tl.constexpr = (
        (M + 4 * BM) * N + (N + 4 * BN) >= 2147483648
        or (M + 4 * BM) * K + (K + 4 * BK) >= 2147483648
        or (N + 4 * BN) * K + (K + 4 * BK) >= 2147483648
        or ((M + 4 * BM) * A0 + (K + 4 * BK) * A1 >= 2147483648)
        or ((M + 4 * BM) * AS0 + ((K + 4 * BK) // 32 + 1) * AS1 >= 2147483648)
        or ((N + 4 * BN) * B0 + (K + 4 * BK) * B1 >= 2147483648)
        or ((N + 4 * BN) * BS0 + ((K + 4 * BK) // 32 + 1) * BS1 >= 2147483648)
    )
    FB = FA + (tl.cast(M, tl.int64) if ADDRESS64 else M)
    pid = (
        tl.cast(tl.program_id(0), tl.int64) if ADDRESS64 else tl.program_id(0)
    )
    rm = pid // tl.cdiv(N, BN) * BM + (
        tl.cast(tl.arange(0, BM), tl.int64) if ADDRESS64 else tl.arange(0, BM)
    )
    rn = pid % tl.cdiv(N, BN) * BN + (
        tl.cast(tl.arange(0, BN), tl.int64) if ADDRESS64 else tl.arange(0, BN)
    )
    rk = tl.cast(tl.arange(0, BK), tl.int64) if ADDRESS64 else tl.arange(0, BK)
    acc = tl.zeros((BM, BN), tl.float32)
    for k in range(tl.cdiv(K, BK)):
        kk = (tl.cast(k, tl.int64) if ADDRESS64 else k) * BK + rk
        av = tl.load(
            A + rm[:, None] * K + kk[None, :],
            (rm[:, None] < M) & (kk[None, :] < K),
            0,
        )
        bv = tl.load(
            B + rn[None, :] * K + kk[:, None],
            (rn[None, :] < N) & (kk[:, None] < K),
            0,
        )
        acc = tl.dot(av, bv, acc)
    tl.store(
        Output + rm[:, None] * N + rn[None, :],
        acc,
        (rm[:, None] < M) & (rn[None, :] < N),
    )
    tile_a = tl.min(tl.load(FA + rm, rm < M, 1), 0)
    tile_b = tl.min(tl.load(FB + rn, rn < N, 1), 0)
    if tile_a & tile_b == 0:
        for sub in range(BM // 16 * (BN // 32)):
            sr = (
                pid // tl.cdiv(N, BN) * BM
                + sub // (BN // 32) * 16
                + (
                    tl.cast(tl.arange(0, 16), tl.int64)
                    if ADDRESS64
                    else tl.arange(0, 16)
                )
            )
            sc = (
                pid % tl.cdiv(N, BN) * BN
                + sub % (BN // 32) * 32
                + (
                    tl.cast(tl.arange(0, 32), tl.int64)
                    if ADDRESS64
                    else tl.arange(0, 32)
                )
            )
            sub_a = tl.min(tl.load(FA + sr, sr < M, 1), 0)
            sub_b = tl.min(tl.load(FB + sc, sc < N, 1), 0)
            if sub_a & sub_b == 0:
                sk = (
                    tl.cast(tl.arange(0, 32), tl.int64)
                    if ADDRESS64
                    else tl.arange(0, 32)
                )
                repaired = tl.zeros((16, 32), tl.float32)
                for chunk in range(tl.cdiv(K, 32)):
                    kk = (
                        tl.cast(chunk, tl.int64) if ADDRESS64 else chunk
                    ) * 32 + sk
                    sa = (
                        tl.load(
                            RAS + sr[:, None] * AS0 + kk[None, :] // 32 * AS1,
                            (sr[:, None] < M) & (kk[None, :] < K),
                            127,
                        ).to(tl.int32)
                        & 255
                    )
                    sb = (
                        tl.load(
                            RBS + sc[None, :] * BS0 + kk[:, None] // 32 * BS1,
                            (sc[None, :] < N) & (kk[:, None] < K),
                            127,
                        ).to(tl.int32)
                        & 255
                    )
                    ua = tl.load(
                        RA + sr[:, None] * A0 + kk[None, :] * A1,
                        (sr[:, None] < M) & (kk[None, :] < K),
                        0,
                    )
                    ub = tl.load(
                        RB + sc[None, :] * B0 + kk[:, None] * B1,
                        (sc[None, :] < N) & (kk[:, None] < K),
                        0,
                    )
                    av = _decode(ua, sa)
                    bv = _decode(ub, sb)
                    repaired += tl.sum(av[:, :, None] * bv[None, :, :], 1)
                tl.store(
                    Output + sr[:, None] * N + sc[None, :],
                    repaired,
                    (sr[:, None] < M) & (sc[None, :] < N),
                )


@triton.jit(do_not_specialize=["Output", "RA", "RAS", "RB", "RBS"])
def _hg_lean_fused783(Output, RA, RAS, RB, RBS, META: tl.constexpr):
    M: tl.constexpr = META[0]
    N: tl.constexpr = META[1]
    K: tl.constexpr = META[2]
    BM: tl.constexpr = META[3]
    BN: tl.constexpr = META[4]
    BK: tl.constexpr = META[5]
    A0: tl.constexpr = META[6]
    A1: tl.constexpr = META[7]
    AS0: tl.constexpr = META[8]
    AS1: tl.constexpr = META[9]
    B0: tl.constexpr = META[10]
    B1: tl.constexpr = META[11]
    BS0: tl.constexpr = META[12]
    BS1: tl.constexpr = META[13]
    BG: tl.constexpr = META[14]
    RA = RA.to(tl.pointer_type(tl.int8))
    RB = RB.to(tl.pointer_type(tl.int8))
    ADDRESS64: tl.constexpr = (
        (M + 4 * BM) * N + (N + 4 * BN) >= 2147483648
        or (M + 4 * BM) * K + (K + 4 * BK) >= 2147483648
        or (N + 4 * BN) * K + (K + 4 * BK) >= 2147483648
        or ((M + 4 * BM) * A0 + (K + 4 * BK) * A1 >= 2147483648)
        or ((M + 4 * BM) * AS0 + ((K + 4 * BK) // 32 + 1) * AS1 >= 2147483648)
        or ((N + 4 * BN) * B0 + (K + 4 * BK) * B1 >= 2147483648)
        or ((N + 4 * BN) * BS0 + ((K + 4 * BK) // 32 + 1) * BS1 >= 2147483648)
    )
    fg = tl.cast(tl.arange(0, BG), tl.int64) if ADDRESS64 else tl.arange(0, BG)
    pid = (
        tl.cast(tl.program_id(0), tl.int64) if ADDRESS64 else tl.program_id(0)
    )
    rm = pid // tl.cdiv(N, BN) * BM + (
        tl.cast(tl.arange(0, BM), tl.int64) if ADDRESS64 else tl.arange(0, BM)
    )
    rn = pid % tl.cdiv(N, BN) * BN + (
        tl.cast(tl.arange(0, BN), tl.int64) if ADDRESS64 else tl.arange(0, BN)
    )
    rk = tl.cast(tl.arange(0, BK), tl.int64) if ADDRESS64 else tl.arange(0, BK)
    acc = tl.zeros((BM, BN), tl.float32)
    for k in range(tl.cdiv(K, BK)):
        kk = (tl.cast(k, tl.int64) if ADDRESS64 else k) * BK + rk
        group_k786 = (tl.cast(k, tl.int64) if ADDRESS64 else k) * (
            BK // 32
        ) + (
            tl.cast(tl.arange(0, BK // 32), tl.int64)
            if ADDRESS64
            else tl.arange(0, BK // 32)
        )
        scale_a_compact786 = tl.load(
            RAS + rm[:, None] * AS0 + group_k786[None, :] * AS1,
            (rm[:, None] < M) & (group_k786[None, :] < tl.cdiv(K, 32)),
            127,
        )
        scale_b_compact786 = tl.load(
            RBS + rn[None, :] * BS0 + group_k786[:, None] * BS1,
            (rn[None, :] < N) & (group_k786[:, None] < tl.cdiv(K, 32)),
            127,
        )
        sa783 = tl.reshape(
            tl.broadcast_to(
                scale_a_compact786[:, :, None], (BM, BK // 32, 32)
            ),
            (BM, BK),
        )
        sb783 = tl.reshape(
            tl.broadcast_to(
                scale_b_compact786[:, None, :], (BK // 32, 32, BN)
            ),
            (BK, BN),
        )
        sa783 = tl.where(kk[None, :] < K, sa783, 127)
        sb783 = tl.where(kk[:, None] < K, sb783, 127)
        ua783 = tl.load(
            RA + rm[:, None] * A0 + kk[None, :] * A1,
            (rm[:, None] < M) & (kk[None, :] < K),
            0,
        )
        ub783 = tl.load(
            RB + rn[None, :] * B0 + kk[:, None] * B1,
            (rn[None, :] < N) & (kk[:, None] < K),
            0,
        )
        av = _decode_halfbase(ua783, sa783).to(
            tl.bfloat16, fp_downcast_rounding="rtz"
        )
        bv = _decode_halfbase(ub783, sb783).to(
            tl.bfloat16, fp_downcast_rounding="rtz"
        )
        acc = tl.dot(av, bv, acc)
    tl.store(
        Output + rm[:, None] * N + rn[None, :],
        acc,
        (rm[:, None] < M) & (rn[None, :] < N),
    )
    scale_a_parent784 = (
        tl.load(
            RAS + rm[:, None] * AS0 + fg[None, :] * AS1,
            (rm[:, None] < M) & (fg[None, :] < tl.cdiv(K, 32)),
            127,
        ).to(tl.int32)
        & 255
    )
    scale_b_parent784 = (
        tl.load(
            RBS + rn[:, None] * BS0 + fg[None, :] * BS1,
            (rn[:, None] < N) & (fg[None, :] < tl.cdiv(K, 32)),
            127,
        ).to(tl.int32)
        & 255
    )
    tile_a = tl.min(
        tl.min(
            ((scale_a_parent784 >= 10) & (scale_a_parent784 <= 246)).to(
                tl.int32
            ),
            1,
        ),
        0,
    )
    tile_b = tl.min(
        tl.min(
            ((scale_b_parent784 >= 10) & (scale_b_parent784 <= 246)).to(
                tl.int32
            ),
            1,
        ),
        0,
    )
    if tile_a & tile_b == 0:
        for sub in range(BM // 16 * (BN // 32)):
            sr = (
                pid // tl.cdiv(N, BN) * BM
                + sub // (BN // 32) * 16
                + (
                    tl.cast(tl.arange(0, 16), tl.int64)
                    if ADDRESS64
                    else tl.arange(0, 16)
                )
            )
            sc = (
                pid % tl.cdiv(N, BN) * BN
                + sub % (BN // 32) * 32
                + (
                    tl.cast(tl.arange(0, 32), tl.int64)
                    if ADDRESS64
                    else tl.arange(0, 32)
                )
            )
            scale_a_sub784 = (
                tl.load(
                    RAS + sr[:, None] * AS0 + fg[None, :] * AS1,
                    (sr[:, None] < M) & (fg[None, :] < tl.cdiv(K, 32)),
                    127,
                ).to(tl.int32)
                & 255
            )
            scale_b_sub784 = (
                tl.load(
                    RBS + sc[:, None] * BS0 + fg[None, :] * BS1,
                    (sc[:, None] < N) & (fg[None, :] < tl.cdiv(K, 32)),
                    127,
                ).to(tl.int32)
                & 255
            )
            sub_a = tl.min(
                tl.min(
                    ((scale_a_sub784 >= 10) & (scale_a_sub784 <= 246)).to(
                        tl.int32
                    ),
                    1,
                ),
                0,
            )
            sub_b = tl.min(
                tl.min(
                    ((scale_b_sub784 >= 10) & (scale_b_sub784 <= 246)).to(
                        tl.int32
                    ),
                    1,
                ),
                0,
            )
            if sub_a & sub_b == 0:
                sk = (
                    tl.cast(tl.arange(0, 32), tl.int64)
                    if ADDRESS64
                    else tl.arange(0, 32)
                )
                repaired = tl.zeros((16, 32), tl.float32)
                for chunk in range(tl.cdiv(K, 32)):
                    kk = (
                        tl.cast(chunk, tl.int64) if ADDRESS64 else chunk
                    ) * 32 + sk
                    sa = (
                        tl.load(
                            RAS + sr[:, None] * AS0 + kk[None, :] // 32 * AS1,
                            (sr[:, None] < M) & (kk[None, :] < K),
                            127,
                        ).to(tl.int32)
                        & 255
                    )
                    sb = (
                        tl.load(
                            RBS + sc[None, :] * BS0 + kk[:, None] // 32 * BS1,
                            (sc[None, :] < N) & (kk[:, None] < K),
                            127,
                        ).to(tl.int32)
                        & 255
                    )
                    ua = tl.load(
                        RA + sr[:, None] * A0 + kk[None, :] * A1,
                        (sr[:, None] < M) & (kk[None, :] < K),
                        0,
                    )
                    ub = tl.load(
                        RB + sc[None, :] * B0 + kk[:, None] * B1,
                        (sc[None, :] < N) & (kk[:, None] < K),
                        0,
                    )
                    av = _decode(ua, sa)
                    bv = _decode(ub, sb)
                    repaired += tl.sum(av[:, :, None] * bv[None, :, :], 1)
                tl.store(
                    Output + sr[:, None] * N + sc[None, :],
                    repaired,
                    (sr[:, None] < M) & (sc[None, :] < N),
                )


__all__ = ["mxfp8_block_scaled_matmul"]
