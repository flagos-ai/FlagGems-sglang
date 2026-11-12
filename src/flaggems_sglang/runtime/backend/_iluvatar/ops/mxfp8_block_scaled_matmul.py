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

import struct

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


@triton.jit(do_not_specialize=["A", "B", "AS", "BS", "FA", "FB"])
def _slow(
    A: tl.float64,
    AS: tl.float64,
    B: tl.float64,
    BS: tl.float64,
    FA: tl.float64,
    FB: tl.float64,
    Output,
    META: tl.constexpr,
    ALIGN: tl.constexpr,
):
    AS = AS.to(tl.uint64, bitcast=True).to(tl.pointer_type(tl.uint8))
    if ALIGN[2]:
        AS = tl.multiple_of(AS, 16)
    BS = BS.to(tl.uint64, bitcast=True).to(tl.pointer_type(tl.uint8))
    if ALIGN[3]:
        BS = tl.multiple_of(BS, 16)
    FA = FA.to(tl.uint64, bitcast=True).to(tl.pointer_type(tl.int32))
    if ALIGN[4]:
        FA = tl.multiple_of(FA, 16)
    FB = FB.to(tl.uint64, bitcast=True).to(tl.pointer_type(tl.int32))
    if ALIGN[5]:
        FB = tl.multiple_of(FB, 16)
    A = A.to(tl.uint64, bitcast=True).to(tl.pointer_type(tl.int8))
    if ALIGN[0]:
        A = tl.multiple_of(A, 16)
    B = B.to(tl.uint64, bitcast=True).to(tl.pointer_type(tl.int8))
    if ALIGN[1]:
        B = tl.multiple_of(B, 16)
    M: tl.constexpr = META[0]
    N: tl.constexpr = META[1]
    K: tl.constexpr = META[2]
    A0: tl.constexpr = META[3]
    A1: tl.constexpr = META[4]
    AS0: tl.constexpr = META[5]
    AS1: tl.constexpr = META[6]
    B0: tl.constexpr = META[7]
    B1: tl.constexpr = META[8]
    BS0: tl.constexpr = META[9]
    BS1: tl.constexpr = META[10]
    BM: tl.constexpr = META[11]
    BN: tl.constexpr = META[12]
    BK: tl.constexpr = META[13]
    ADDRESS64: tl.constexpr = (
        (M + 4 * BM) * N + (N + 4 * BN) >= 2147483648
        or (M + 4 * BM) * K + (K + 4 * BK) >= 2147483648
        or (N + 4 * BN) * K + (K + 4 * BK) >= 2147483648
        or ((M + 4 * BM) * A0 + (K + 4 * BK) * A1 >= 2147483648)
        or ((M + 4 * BM) * AS0 + ((K + 4 * BK) // 32 + 1) * AS1 >= 2147483648)
        or ((N + 4 * BN) * B0 + (K + 4 * BK) * B1 >= 2147483648)
        or ((N + 4 * BN) * BS0 + ((K + 4 * BK) // 32 + 1) * BS1 >= 2147483648)
    )
    pid = (
        tl.cast(tl.program_id(0), tl.int64) if ADDRESS64 else tl.program_id(0)
    )
    rm = pid // tl.cdiv(N, BN) * BM + (
        tl.cast(tl.arange(0, BM), tl.int64) if ADDRESS64 else tl.arange(0, BM)
    )
    rn = pid % tl.cdiv(N, BN) * BN + (
        tl.cast(tl.arange(0, BN), tl.int64) if ADDRESS64 else tl.arange(0, BN)
    )
    fa = tl.min(tl.load(FA + rm, rm < M, 1), 0)
    fb = tl.min(tl.load(FB + rn, rn < N, 1), 0)
    if fa & fb == 0:
        rk = (
            tl.cast(tl.arange(0, BK), tl.int64)
            if ADDRESS64
            else tl.arange(0, BK)
        )
        acc = tl.zeros((BM, BN), tl.float32)
        for k in range(tl.cdiv(K, BK)):
            kk = (tl.cast(k, tl.int64) if ADDRESS64 else k) * BK + rk
            sa = (
                tl.load(
                    AS + rm[:, None] * AS0 + kk[None, :] // 32 * AS1,
                    (rm[:, None] < M) & (kk[None, :] < K),
                    127,
                ).to(tl.int32)
                & 255
            )
            sb = (
                tl.load(
                    BS + rn[None, :] * BS0 + kk[:, None] // 32 * BS1,
                    (rn[None, :] < N) & (kk[:, None] < K),
                    127,
                ).to(tl.int32)
                & 255
            )
            ua = tl.load(
                A + rm[:, None] * A0 + kk[None, :] * A1,
                (rm[:, None] < M) & (kk[None, :] < K),
                0,
            )
            ub = tl.load(
                B + rn[None, :] * B0 + kk[:, None] * B1,
                (rn[None, :] < N) & (kk[:, None] < K),
                0,
            )
            av = _decode(ua, sa)
            bv = _decode(ub, sb)
            av = tl.gather(
                av, tl.broadcast_to(tl.arange(0, BK)[None, :], (BM, BK)), 1
            )
            bv = tl.gather(
                bv, tl.broadcast_to(tl.arange(0, BK)[:, None], (BK, BN)), 0
            )
            acc += tl.sum(av[:, :, None] * bv[None, :, :], 1)
        tl.store(
            Output + rm[:, None] * N + rn[None, :],
            acc,
            (rm[:, None] < M) & (rn[None, :] < N),
        )


@triton.jit
def _fast(A, B, Output, META: tl.constexpr):
    M: tl.constexpr = META[0]
    N: tl.constexpr = META[1]
    K: tl.constexpr = META[2]
    BM: tl.constexpr = META[3]
    BN: tl.constexpr = META[4]
    BK: tl.constexpr = META[5]
    ADDRESS64: tl.constexpr = (
        (M + 4 * BM) * N + (N + 4 * BN) >= 2147483648
        or (M + 4 * BM) * K + (K + 4 * BK) >= 2147483648
        or (N + 4 * BN) * K + (K + 4 * BK) >= 2147483648
    )
    pid = (
        tl.cast(tl.program_id(0), tl.int64) if ADDRESS64 else tl.program_id(0)
    )
    if BM == 128 and BN == 256:
        gm809 = tl.cdiv(M, BM)
        gn809 = tl.cdiv(N, BN)
        group809 = pid // (8 * gn809)
        first809 = group809 * 8
        size809 = tl.minimum(gm809 - first809, 8)
        local809 = pid % (8 * gn809)
        pm809 = first809 + local809 % size809
        pn809 = local809 // size809
    else:
        pm809 = pid // tl.cdiv(N, BN)
        pn809 = pid % tl.cdiv(N, BN)
    rm = pm809 * BM + (
        tl.cast(tl.arange(0, BM), tl.int64) if ADDRESS64 else tl.arange(0, BM)
    )
    rn = pn809 * BN + (
        tl.cast(tl.arange(0, BN), tl.int64) if ADDRESS64 else tl.arange(0, BN)
    )
    am = rm % M
    bn = rn % N
    rk = tl.cast(tl.arange(0, BK), tl.int64) if ADDRESS64 else tl.arange(0, BK)
    acc = tl.zeros((BM, BN), tl.float32)
    for k in range(K // BK):
        kk = (tl.cast(k, tl.int64) if ADDRESS64 else k) * BK + rk
        av = tl.load(A + am[:, None] * K + kk[None, :])
        bv = tl.load(B + bn[None, :] * K + kk[:, None])
        acc = tl.dot(av, bv, acc)
    if K % BK != 0:
        kk = K // BK * BK + rk
        av = tl.load(A + am[:, None] * K + kk[None, :], kk[None, :] < K, 0)
        bv = tl.load(B + bn[None, :] * K + kk[:, None], kk[:, None] < K, 0)
        acc = tl.dot(av, bv, acc)
    tl.store(
        Output + rm[:, None] * N + rn[None, :],
        acc,
        (rm[:, None] < M) & (rn[None, :] < N),
    )


def _prepared960(a, a_scale, b, b_scale, output_dtype):
    m, k = a.shape
    n = b.shape[0]
    out = a.new_empty((m, n), dtype=output_dtype)
    if m == 0 or n == 0:
        return out
    half_storage = m < 512
    storage_dtype = torch.float16 if half_storage else torch.bfloat16
    ad = a.new_empty((m, k), dtype=storage_dtype)
    bd = a.new_empty((n, k), dtype=storage_dtype)
    fa = a.new_empty((m + n,), dtype=torch.int32)
    prepare = (
        ti_dequant_half_storage if half_storage else _dequant_both_reshape
    )
    a_bytes819, b_bytes819, as981, bs981, fa981 = (
        a.data_ptr(),
        b.data_ptr(),
        a_scale.data_ptr(),
        b_scale.data_ptr(),
        fa.data_ptr(),
    )
    fb981 = fa981 + 4 * m
    align974 = (
        a_bytes819 % 16 == 0,
        b_bytes819 % 16 == 0,
        as981 % 16 == 0,
        bs981 % 16 == 0,
        fa981 % 16 == 0,
        fb981 % 16 == 0,
    )
    a_bytes819, b_bytes819, as981, bs981, fa981, fb981 = struct.unpack(
        "=6d",
        struct.pack("=6Q", a_bytes819, b_bytes819, as981, bs981, fa981, fb981),
    )
    a_s0_823 = a.stride(0)
    a_s1_823 = a.stride(1)
    a_scale_s0_823 = a_scale.stride(0)
    a_scale_s1_823 = a_scale.stride(1)
    b_s0_823 = b.stride(0)
    b_s1_823 = b.stride(1)
    b_scale_s0_823 = b_scale.stride(0)
    b_scale_s1_823 = b_scale.stride(1)
    prepare.run(
        a_bytes819,
        as981,
        b_bytes819,
        bs981,
        ad,
        bd,
        fa981,
        fb981,
        (
            m,
            k,
            a_s0_823,
            a_s1_823,
            a_scale_s0_823,
            a_scale_s1_823,
            b_s0_823,
            b_s1_823,
            b_scale_s0_823,
            b_scale_s1_823,
            max(32, 0 if k == 0 else 1 << (k - 1).bit_length()),
            n,
        ),
        align974,
        num_warps=4,
        grid=(m + n,),
        warmup=False,
    )
    bm, bn, bk = (128, 128, 64) if k >= 2048 else (64, 64, 64)
    if m >= 2048 and k >= 2048:
        bm, bn, bk = (128, 256, 32)
    fast_grid996 = (m + bm - 1) // bm * ((n + bn - 1) // bn)
    coarse_scale996 = 4 if m >= 2048 else 1
    repair_grid996 = (
        (m + 16 * coarse_scale996 - 1)
        // (16 * coarse_scale996)
        * ((n + 32 * coarse_scale996 - 1) // (32 * coarse_scale996))
    )
    if coarse_scale996 == 1 or fast_grid996 + repair_grid996 > 2147483647:
        (ti_fast_store if 512 <= m < 2048 and k >= 2048 else _fast).run(
            ad,
            bd,
            out,
            (m, n, k, bm, bn, bk),
            num_warps=8 if bm == 128 else 4,
            num_stages=3,
            grid=((m + bm - 1) // bm * ((n + bn - 1) // bn),),
            warmup=False,
        )
        bm, bn, bk = (16, 32, 32)
        slow = _slow_coarse if m >= 2048 else _slow
        coarse_scale = 4 if m >= 2048 else 1
        slow.run(
            a_bytes819,
            as981,
            b_bytes819,
            bs981,
            fa981,
            fb981,
            out,
            (
                m,
                n,
                k,
                a_s0_823,
                a_s1_823,
                a_scale_s0_823,
                a_scale_s1_823,
                b_s0_823,
                b_s1_823,
                b_scale_s0_823,
                b_scale_s1_823,
                bm,
                bn,
                bk,
            ),
            align974,
            num_warps=4,
            num_stages=2,
            grid=(
                (m + bm * coarse_scale - 1)
                // (bm * coarse_scale)
                * ((n + bn * coarse_scale - 1) // (bn * coarse_scale)),
            ),
            warmup=False,
        )
    else:
        slow_meta996 = (
            m,
            n,
            k,
            a_s0_823,
            a_s1_823,
            a_scale_s0_823,
            a_scale_s1_823,
            b_s0_823,
            b_s1_823,
            b_scale_s0_823,
            b_scale_s1_823,
            16,
            32,
            32,
        )
        (
            _fast_store_union996
            if 512 <= m < 2048 and k >= 2048
            else _fast_union996
        ).run(
            ad,
            bd,
            out,
            a_bytes819,
            as981,
            b_bytes819,
            bs981,
            fa981,
            fb981,
            (m, n, k, bm, bn, bk),
            slow_meta996,
            align974,
            num_warps=8 if bm == 128 else 4,
            num_stages=3,
            grid=(fast_grid996 + repair_grid996,),
            warmup=False,
        )
    return out


@triton.jit(do_not_specialize=["A", "B", "AS", "BS", "FA", "FB"])
def _slow_coarse(
    A: tl.float64,
    AS: tl.float64,
    B: tl.float64,
    BS: tl.float64,
    FA: tl.float64,
    FB: tl.float64,
    Output,
    META: tl.constexpr,
    ALIGN: tl.constexpr,
):
    AS = AS.to(tl.uint64, bitcast=True).to(tl.pointer_type(tl.uint8))
    if ALIGN[2]:
        AS = tl.multiple_of(AS, 16)
    BS = BS.to(tl.uint64, bitcast=True).to(tl.pointer_type(tl.uint8))
    if ALIGN[3]:
        BS = tl.multiple_of(BS, 16)
    FA = FA.to(tl.uint64, bitcast=True).to(tl.pointer_type(tl.int32))
    if ALIGN[4]:
        FA = tl.multiple_of(FA, 16)
    FB = FB.to(tl.uint64, bitcast=True).to(tl.pointer_type(tl.int32))
    if ALIGN[5]:
        FB = tl.multiple_of(FB, 16)
    A = A.to(tl.uint64, bitcast=True).to(tl.pointer_type(tl.int8))
    if ALIGN[0]:
        A = tl.multiple_of(A, 16)
    B = B.to(tl.uint64, bitcast=True).to(tl.pointer_type(tl.int8))
    if ALIGN[1]:
        B = tl.multiple_of(B, 16)
    M: tl.constexpr = META[0]
    N: tl.constexpr = META[1]
    K: tl.constexpr = META[2]
    A0: tl.constexpr = META[3]
    A1: tl.constexpr = META[4]
    AS0: tl.constexpr = META[5]
    AS1: tl.constexpr = META[6]
    B0: tl.constexpr = META[7]
    B1: tl.constexpr = META[8]
    BS0: tl.constexpr = META[9]
    BS1: tl.constexpr = META[10]
    BM: tl.constexpr = META[11]
    BN: tl.constexpr = META[12]
    BK: tl.constexpr = META[13]
    ADDRESS64: tl.constexpr = (
        (M + 4 * BM) * N + (N + 4 * BN) >= 2147483648
        or (M + 4 * BM) * K + (K + 4 * BK) >= 2147483648
        or (N + 4 * BN) * K + (K + 4 * BK) >= 2147483648
        or ((M + 4 * BM) * A0 + (K + 4 * BK) * A1 >= 2147483648)
        or ((M + 4 * BM) * AS0 + ((K + 4 * BK) // 32 + 1) * AS1 >= 2147483648)
        or ((N + 4 * BN) * B0 + (K + 4 * BK) * B1 >= 2147483648)
        or ((N + 4 * BN) * BS0 + ((K + 4 * BK) // 32 + 1) * BS1 >= 2147483648)
    )
    pid = (
        tl.cast(tl.program_id(0), tl.int64) if ADDRESS64 else tl.program_id(0)
    )
    gm = pid // tl.cdiv(N, BN * 4)
    gn = pid % tl.cdiv(N, BN * 4)
    cr = gm * BM * 4 + (
        tl.cast(tl.arange(0, BM * 4), tl.int64)
        if ADDRESS64
        else tl.arange(0, BM * 4)
    )
    cc = gn * BN * 4 + (
        tl.cast(tl.arange(0, BN * 4), tl.int64)
        if ADDRESS64
        else tl.arange(0, BN * 4)
    )
    cfa = tl.min(tl.load(FA + cr, cr < M, 1), 0)
    cfb = tl.min(tl.load(FB + cc, cc < N, 1), 0)
    if cfa & cfb == 0:
        for tile818 in range(16):
            mi = tile818 // 4
            ni = tile818 % 4
            rm = (gm * 4 + mi) * BM + (
                tl.cast(tl.arange(0, BM), tl.int64)
                if ADDRESS64
                else tl.arange(0, BM)
            )
            rn = (gn * 4 + ni) * BN + (
                tl.cast(tl.arange(0, BN), tl.int64)
                if ADDRESS64
                else tl.arange(0, BN)
            )
            fa = tl.min(tl.load(FA + rm, rm < M, 1), 0)
            fb = tl.min(tl.load(FB + rn, rn < N, 1), 0)
            if fa & fb == 0:
                rk = (
                    tl.cast(tl.arange(0, BK), tl.int64)
                    if ADDRESS64
                    else tl.arange(0, BK)
                )
                acc = tl.zeros((BM, BN), tl.float32)
                for k in range(tl.cdiv(K, BK)):
                    kk = (tl.cast(k, tl.int64) if ADDRESS64 else k) * BK + rk
                    sa = (
                        tl.load(
                            AS + rm[:, None] * AS0 + kk[None, :] // 32 * AS1,
                            (rm[:, None] < M) & (kk[None, :] < K),
                            127,
                        ).to(tl.int32)
                        & 255
                    )
                    sb = (
                        tl.load(
                            BS + rn[None, :] * BS0 + kk[:, None] // 32 * BS1,
                            (rn[None, :] < N) & (kk[:, None] < K),
                            127,
                        ).to(tl.int32)
                        & 255
                    )
                    ua = tl.load(
                        A + rm[:, None] * A0 + kk[None, :] * A1,
                        (rm[:, None] < M) & (kk[None, :] < K),
                        0,
                    )
                    ub = tl.load(
                        B + rn[None, :] * B0 + kk[:, None] * B1,
                        (rn[None, :] < N) & (kk[:, None] < K),
                        0,
                    )
                    av = _decode(ua, sa)
                    bv = _decode(ub, sb)
                    av = tl.gather(
                        av,
                        tl.broadcast_to(tl.arange(0, BK)[None, :], (BM, BK)),
                        1,
                    )
                    bv = tl.gather(
                        bv,
                        tl.broadcast_to(tl.arange(0, BK)[:, None], (BK, BN)),
                        0,
                    )
                    acc += tl.sum(av[:, :, None] * bv[None, :, :], 1)
                tl.store(
                    Output + rm[:, None] * N + rn[None, :],
                    acc,
                    (rm[:, None] < M) & (rn[None, :] < N),
                )


@triton.jit(do_not_specialize=["A", "B", "AS", "BS", "FA", "FB"])
def _dequant_both_reshape(
    A: tl.float64,
    AS: tl.float64,
    B: tl.float64,
    BS: tl.float64,
    AD,
    BD,
    FA: tl.float64,
    FB: tl.float64,
    META: tl.constexpr,
    ALIGN: tl.constexpr,
):
    AS = AS.to(tl.uint64, bitcast=True).to(tl.pointer_type(tl.uint8))
    if ALIGN[2]:
        AS = tl.multiple_of(AS, 16)
    BS = BS.to(tl.uint64, bitcast=True).to(tl.pointer_type(tl.uint8))
    if ALIGN[3]:
        BS = tl.multiple_of(BS, 16)
    FA = FA.to(tl.uint64, bitcast=True).to(tl.pointer_type(tl.int32))
    if ALIGN[4]:
        FA = tl.multiple_of(FA, 16)
    FB = FB.to(tl.uint64, bitcast=True).to(tl.pointer_type(tl.int32))
    if ALIGN[5]:
        FB = tl.multiple_of(FB, 16)
    A = A.to(tl.uint64, bitcast=True).to(tl.pointer_type(tl.int8))
    if ALIGN[0]:
        A = tl.multiple_of(A, 16)
    B = B.to(tl.uint64, bitcast=True).to(tl.pointer_type(tl.int8))
    if ALIGN[1]:
        B = tl.multiple_of(B, 16)
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
    N: tl.constexpr = META[11]
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
    flag = tl.where(is_a, FA + row, FB + row - M)
    stride = tl.where(is_a, A1, B1)
    scale_stride = tl.where(is_a, AS1, BS1)
    u = tl.load(src + cols * stride, cols < K, 0)
    groups = (
        tl.cast(tl.arange(0, BLOCK // 32), tl.int64)
        if ADDRESS64
        else tl.arange(0, BLOCK // 32)
    )
    sg = (
        tl.load(
            scale + groups * scale_stride, groups < (K + 31) // 32, 127
        ).to(tl.int32)
        & 255
    )
    s = tl.reshape(tl.broadcast_to(sg[:, None], (BLOCK // 32, 32)), (BLOCK,))
    value = _decode(u, s)
    tl.store(dst + cols, value, cols < K)
    safe = tl.min(((sg >= 10) & (sg <= 246)).to(tl.int32), 0)
    tl.store(flag, safe)


@triton.jit
def ti_fast_store(A, B, Output, META: tl.constexpr):
    M: tl.constexpr = META[0]
    N: tl.constexpr = META[1]
    K: tl.constexpr = META[2]
    BM: tl.constexpr = META[3]
    BN: tl.constexpr = META[4]
    BK: tl.constexpr = META[5]
    ADDRESS64: tl.constexpr = (
        (M + 4 * BM) * N + (N + 4 * BN) >= 2147483648
        or (M + 4 * BM) * K + (K + 4 * BK) >= 2147483648
        or (N + 4 * BN) * K + (K + 4 * BK) >= 2147483648
    )
    pid = (
        tl.cast(tl.program_id(0), tl.int64) if ADDRESS64 else tl.program_id(0)
    )
    rm = pid // tl.cdiv(N, BN) * BM + (
        tl.cast(tl.arange(0, BM), tl.int64) if ADDRESS64 else tl.arange(0, BM)
    )
    rn = pid % tl.cdiv(N, BN) * BN + (
        tl.cast(tl.arange(0, BN), tl.int64) if ADDRESS64 else tl.arange(0, BN)
    )
    am = rm % M
    bn = rn % N
    rk = tl.cast(tl.arange(0, BK), tl.int64) if ADDRESS64 else tl.arange(0, BK)
    acc = tl.zeros((BM, BN), tl.float32)
    for k in range(K // BK):
        kk = (tl.cast(k, tl.int64) if ADDRESS64 else k) * BK + rk
        av = tl.load(A + am[:, None] * K + kk[None, :])
        bv = tl.load(B + bn[None, :] * K + kk[:, None])
        acc = tl.dot(av, bv, acc)
    if K % BK != 0:
        kk = K // BK * BK + rk
        av = tl.load(A + am[:, None] * K + kk[None, :], kk[None, :] < K, 0)
        bv = tl.load(B + bn[None, :] * K + kk[:, None], kk[:, None] < K, 0)
        acc = tl.dot(av, bv, acc)
    if M % BM == 0 and N % BN == 0:
        tl.store(Output + rm[:, None] * N + rn[None, :], acc)
    else:
        tl.store(
            Output + rm[:, None] * N + rn[None, :],
            acc,
            (rm[:, None] < M) & (rn[None, :] < N),
        )


@triton.jit(do_not_specialize=["A", "B", "AS", "BS", "FA", "FB"])
def ti_dequant_half_storage(
    A: tl.float64,
    AS: tl.float64,
    B: tl.float64,
    BS: tl.float64,
    AD,
    BD,
    FA: tl.float64,
    FB: tl.float64,
    META: tl.constexpr,
    ALIGN: tl.constexpr,
):
    AS = AS.to(tl.uint64, bitcast=True).to(tl.pointer_type(tl.uint8))
    if ALIGN[2]:
        AS = tl.multiple_of(AS, 16)
    BS = BS.to(tl.uint64, bitcast=True).to(tl.pointer_type(tl.uint8))
    if ALIGN[3]:
        BS = tl.multiple_of(BS, 16)
    FA = FA.to(tl.uint64, bitcast=True).to(tl.pointer_type(tl.int32))
    if ALIGN[4]:
        FA = tl.multiple_of(FA, 16)
    FB = FB.to(tl.uint64, bitcast=True).to(tl.pointer_type(tl.int32))
    if ALIGN[5]:
        FB = tl.multiple_of(FB, 16)
    A = A.to(tl.uint64, bitcast=True).to(tl.pointer_type(tl.int8))
    if ALIGN[0]:
        A = tl.multiple_of(A, 16)
    B = B.to(tl.uint64, bitcast=True).to(tl.pointer_type(tl.int8))
    if ALIGN[1]:
        B = tl.multiple_of(B, 16)
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
    N: tl.constexpr = META[11]
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
    flag = tl.where(is_a, FA + row, FB + row - M)
    stride = tl.where(is_a, A1, B1)
    scale_stride = tl.where(is_a, AS1, BS1)
    u = tl.load(src + cols * stride, cols < K, 0)
    groups = (
        tl.cast(tl.arange(0, BLOCK // 32), tl.int64)
        if ADDRESS64
        else tl.arange(0, BLOCK // 32)
    )
    sg = (
        tl.load(
            scale + groups * scale_stride, groups < (K + 31) // 32, 127
        ).to(tl.int32)
        & 255
    )
    s = tl.reshape(tl.broadcast_to(sg[:, None], (BLOCK // 32, 32)), (BLOCK,))
    value = _decode(u, s)
    tl.store(dst + cols, value, cols < K)
    safe = tl.min(((sg >= 112) & (sg <= 134)).to(tl.int32), 0)
    tl.store(flag, safe)


@triton.jit
def _identity_or955(x, y):
    return x | y


@triton.jit
def _decode_manual945(u, s):
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
def _decode_bf16bits946(u, s):
    u = u.to(tl.int32) & 255
    s = s.to(tl.int32) & 255
    e = u >> 3 & 15
    m = u & 7
    normal = e + s - 7 << 7 | m << 4
    subnormal = tl.where(
        m >= 4,
        s - 7 << 7 | m - 4 << 5,
        tl.where(m >= 2, s - 8 << 7 | m - 2 << 6, s - 9 << 7),
    )
    magnitude = tl.where(e != 0, normal, tl.where(m != 0, subnormal, 0))
    signed = magnitude | (u & 128) << 8
    result = tl.where(u & 127 == 127, 32704, signed)
    return result.to(tl.uint16).to(tl.bfloat16, bitcast=True)


@triton.jit(do_not_specialize=["A", "B", "AS", "BS"])
def _fused_kpairidentity955(
    A: tl.float64,
    AS: tl.float64,
    B: tl.float64,
    BS: tl.float64,
    Output,
    META: tl.constexpr,
    ALIGN: tl.constexpr,
):
    AS = AS.to(tl.uint64, bitcast=True).to(tl.pointer_type(tl.uint8))
    if ALIGN[2]:
        AS = tl.multiple_of(AS, 16)
    BS = BS.to(tl.uint64, bitcast=True).to(tl.pointer_type(tl.uint8))
    if ALIGN[3]:
        BS = tl.multiple_of(BS, 16)
    A = A.to(tl.uint64, bitcast=True).to(tl.pointer_type(tl.int8))
    if ALIGN[0]:
        A = tl.multiple_of(A, 16)
    B = B.to(tl.uint64, bitcast=True).to(tl.pointer_type(tl.int8))
    if ALIGN[1]:
        B = tl.multiple_of(B, 16)
    M: tl.constexpr = META[0]
    N: tl.constexpr = META[1]
    K: tl.constexpr = META[2]
    A0: tl.constexpr = META[3]
    A1: tl.constexpr = META[4]
    AS0: tl.constexpr = META[5]
    AS1: tl.constexpr = META[6]
    B0: tl.constexpr = META[7]
    B1: tl.constexpr = META[8]
    BS0: tl.constexpr = META[9]
    BS1: tl.constexpr = META[10]
    BM: tl.constexpr = META[11]
    BN: tl.constexpr = META[12]
    BK: tl.constexpr = META[13]
    ADDRESS64: tl.constexpr = (
        (M + 4 * BM) * N + (N + 4 * BN) >= 2147483648
        or (M + 4 * BM) * K + (K + 4 * BK) >= 2147483648
        or (N + 4 * BN) * K + (K + 4 * BK) >= 2147483648
        or ((M + 4 * BM) * A0 + (K + 4 * BK) * A1 >= 2147483648)
        or ((M + 4 * BM) * AS0 + ((K + 4 * BK) // 32 + 1) * AS1 >= 2147483648)
        or ((N + 4 * BN) * B0 + (K + 4 * BK) * B1 >= 2147483648)
        or ((N + 4 * BN) * BS0 + ((K + 4 * BK) // 32 + 1) * BS1 >= 2147483648)
    )
    pid = (
        tl.cast(tl.program_id(0), tl.int64) if ADDRESS64 else tl.program_id(0)
    )
    rm = pid // tl.cdiv(N, BN) * BM + (
        tl.cast(tl.arange(0, BM), tl.int64) if ADDRESS64 else tl.arange(0, BM)
    )
    rn = pid % tl.cdiv(N, BN) * BN + (
        tl.cast(tl.arange(0, BN), tl.int64) if ADDRESS64 else tl.arange(0, BN)
    )
    BG: tl.constexpr = META[14]
    groups = (
        tl.cast(tl.arange(0, BG), tl.int64) if ADDRESS64 else tl.arange(0, BG)
    )
    ga = (
        tl.load(
            AS + rm[:, None] * AS0 + groups[None, :] * AS1,
            (rm[:, None] < M) & (groups[None, :] < tl.cdiv(K, 32)),
            127,
        ).to(tl.int32)
        & 255
    )
    gb = (
        tl.load(
            BS + rn[:, None] * BS0 + groups[None, :] * BS1,
            (rn[:, None] < N) & (groups[None, :] < tl.cdiv(K, 32)),
            127,
        ).to(tl.int32)
        & 255
    )
    safe_a = tl.min(tl.min(((ga >= 10) & (ga <= 246)).to(tl.int32), 0), 0)
    safe_b = tl.min(tl.min(((gb >= 10) & (gb <= 246)).to(tl.int32), 0), 0)
    rk = tl.cast(tl.arange(0, BK), tl.int64) if ADDRESS64 else tl.arange(0, BK)
    acc = tl.zeros((BM, BN), tl.float32)
    if safe_a & safe_b:
        for k in range(tl.cdiv(K, BK)):
            kk = (tl.cast(k, tl.int64) if ADDRESS64 else k) * BK + rk
            scale_groups = (tl.cast(k, tl.int64) if ADDRESS64 else k) * 2 + (
                tl.cast(tl.arange(0, 2), tl.int64)
                if ADDRESS64
                else tl.arange(0, 2)
            )
            sa_groups = (
                tl.load(
                    AS + rm[:, None] * AS0 + scale_groups[None, :] * AS1,
                    (rm[:, None] < M)
                    & (scale_groups[None, :] < tl.cdiv(K, 32)),
                    127,
                ).to(tl.int32)
                & 255
            )
            sb_groups = (
                tl.load(
                    BS + rn[None, :] * BS0 + scale_groups[:, None] * BS1,
                    (rn[None, :] < N)
                    & (scale_groups[:, None] < tl.cdiv(K, 32)),
                    127,
                ).to(tl.int32)
                & 255
            )
            sa = tl.reshape(
                tl.broadcast_to(sa_groups[:, :, None], (BM, 2, 32)), (BM, 64)
            )
            sb = tl.reshape(
                tl.broadcast_to(sb_groups[:, None, :], (2, 32, BN)), (64, BN)
            )
            ua = tl.load(
                A + rm[:, None] * A0 + kk[None, :] * A1,
                (rm[:, None] < M) & (kk[None, :] < K),
                0,
            )
            ub = tl.load(
                B + rn[None, :] * B0 + kk[:, None] * B1,
                (rn[None, :] < N) & (kk[:, None] < K),
                0,
            )
            av = _decode_bf16bits946(ua, sa)
            bv = _decode_bf16bits946(ub, sb)
            av_bits = av.to(tl.uint16, bitcast=True)
            bv_bits = bv.to(tl.uint16, bitcast=True)
            av = tl.reduce(tl.join(av_bits, av_bits), 2, _identity_or955).to(
                tl.bfloat16, bitcast=True
            )
            bv = tl.reduce(tl.join(bv_bits, bv_bits), 2, _identity_or955).to(
                tl.bfloat16, bitcast=True
            )
            acc = tl.dot(av, bv, acc)
    else:
        rk_unsafe = (
            tl.cast(tl.arange(0, 32), tl.int64)
            if ADDRESS64
            else tl.arange(0, 32)
        )
        for k in range(tl.cdiv(K, 32)):
            kk = (tl.cast(k, tl.int64) if ADDRESS64 else k) * 32 + rk_unsafe
            sa = (
                tl.load(
                    AS + rm[:, None] * AS0 + kk[None, :] // 32 * AS1,
                    (rm[:, None] < M) & (kk[None, :] < K),
                    127,
                ).to(tl.int32)
                & 255
            )
            sb = (
                tl.load(
                    BS + rn[None, :] * BS0 + kk[:, None] // 32 * BS1,
                    (rn[None, :] < N) & (kk[:, None] < K),
                    127,
                ).to(tl.int32)
                & 255
            )
            ua = tl.load(
                A + rm[:, None] * A0 + kk[None, :] * A1,
                (rm[:, None] < M) & (kk[None, :] < K),
                0,
            )
            ub = tl.load(
                B + rn[None, :] * B0 + kk[:, None] * B1,
                (rn[None, :] < N) & (kk[:, None] < K),
                0,
            )
            av = _decode_manual945(ua, sa)
            bv = _decode_manual945(ub, sb)
            av_bits = av.to(tl.uint32, bitcast=True)
            bv_bits = bv.to(tl.uint32, bitcast=True)
            av = tl.reduce(tl.join(av_bits, av_bits), 2, _identity_or955).to(
                tl.float32, bitcast=True
            )
            bv = tl.reduce(tl.join(bv_bits, bv_bits), 2, _identity_or955).to(
                tl.float32, bitcast=True
            )
            acc += tl.sum(av[:, :, None] * bv[None, :, :], 1)
    tl.store(
        Output + rm[:, None] * N + rn[None, :],
        acc,
        (rm[:, None] < M) & (rn[None, :] < N),
    )


def mxfp8_block_scaled_matmul(a, a_scale, b, b_scale, output_dtype):
    m, k = a.shape
    n = b.shape[0]
    tiles = (m + 15) // 16 * ((n + 31) // 32)
    if (
        k <= 0
        or k > 1024
        or tiles > 256
        or ((tiles + 15) // 16 * ((k + 63) // 64) > 64)
    ):
        return _prepared960(a, a_scale, b, b_scale, output_dtype)
    out = a.new_empty((m, n), dtype=output_dtype)
    if m == 0 or n == 0:
        return out
    groups = (k + 31) // 32
    meta = (
        m,
        n,
        k,
        a.stride(0),
        a.stride(1),
        a_scale.stride(0),
        a_scale.stride(1),
        b.stride(0),
        b.stride(1),
        b_scale.stride(0),
        b_scale.stride(1),
        16,
        32,
        64,
        1 << (groups - 1).bit_length(),
    )
    pa970, pb970, as981, bs981 = (
        a.data_ptr(),
        b.data_ptr(),
        a_scale.data_ptr(),
        b_scale.data_ptr(),
    )
    align970 = (
        pa970 % 16 == 0,
        pb970 % 16 == 0,
        as981 % 16 == 0,
        bs981 % 16 == 0,
    )
    pa970, pb970, as981, bs981 = struct.unpack(
        "=4d", struct.pack("=4Q", pa970, pb970, as981, bs981)
    )
    _fused_kpairidentity955.run(
        pa970,
        as981,
        pb970,
        bs981,
        out,
        meta,
        align970,
        grid=((m + 15) // 16 * ((n + 31) // 32),),
        num_warps=4,
        num_stages=2,
        warmup=False,
    )
    return out


@triton.jit(do_not_specialize=["A", "B", "AS", "BS", "FA", "FB"])
def _repair_fine_union996(
    A: tl.float64,
    AS: tl.float64,
    B: tl.float64,
    BS: tl.float64,
    FA: tl.float64,
    FB: tl.float64,
    Output,
    META: tl.constexpr,
    ALIGN: tl.constexpr,
    PID996,
    FIRST996: tl.constexpr,
):
    AS = AS.to(tl.uint64, bitcast=True).to(tl.pointer_type(tl.uint8))
    if ALIGN[2]:
        AS = tl.multiple_of(AS, 16)
    BS = BS.to(tl.uint64, bitcast=True).to(tl.pointer_type(tl.uint8))
    if ALIGN[3]:
        BS = tl.multiple_of(BS, 16)
    FA = FA.to(tl.uint64, bitcast=True).to(tl.pointer_type(tl.int32))
    if ALIGN[4]:
        FA = tl.multiple_of(FA, 16)
    FB = FB.to(tl.uint64, bitcast=True).to(tl.pointer_type(tl.int32))
    if ALIGN[5]:
        FB = tl.multiple_of(FB, 16)
    A = A.to(tl.uint64, bitcast=True).to(tl.pointer_type(tl.int8))
    if ALIGN[0]:
        A = tl.multiple_of(A, 16)
    B = B.to(tl.uint64, bitcast=True).to(tl.pointer_type(tl.int8))
    if ALIGN[1]:
        B = tl.multiple_of(B, 16)
    M: tl.constexpr = META[0]
    N: tl.constexpr = META[1]
    K: tl.constexpr = META[2]
    A0: tl.constexpr = META[3]
    A1: tl.constexpr = META[4]
    AS0: tl.constexpr = META[5]
    AS1: tl.constexpr = META[6]
    B0: tl.constexpr = META[7]
    B1: tl.constexpr = META[8]
    BS0: tl.constexpr = META[9]
    BS1: tl.constexpr = META[10]
    BM: tl.constexpr = META[11]
    BN: tl.constexpr = META[12]
    BK: tl.constexpr = META[13]
    ADDRESS64: tl.constexpr = (
        (M + 4 * BM) * N + (N + 4 * BN) >= 2147483648
        or (M + 4 * BM) * K + (K + 4 * BK) >= 2147483648
        or (N + 4 * BN) * K + (K + 4 * BK) >= 2147483648
        or ((M + 4 * BM) * A0 + (K + 4 * BK) * A1 >= 2147483648)
        or ((M + 4 * BM) * AS0 + ((K + 4 * BK) // 32 + 1) * AS1 >= 2147483648)
        or ((N + 4 * BN) * B0 + (K + 4 * BK) * B1 >= 2147483648)
        or ((N + 4 * BN) * BS0 + ((K + 4 * BK) // 32 + 1) * BS1 >= 2147483648)
    )
    pid = (tl.cast(PID996, tl.int64) if ADDRESS64 else PID996) - FIRST996
    rm = pid // tl.cdiv(N, BN) * BM + (
        tl.cast(tl.arange(0, BM), tl.int64) if ADDRESS64 else tl.arange(0, BM)
    )
    rn = pid % tl.cdiv(N, BN) * BN + (
        tl.cast(tl.arange(0, BN), tl.int64) if ADDRESS64 else tl.arange(0, BN)
    )
    fa = tl.min(tl.load(FA + rm, rm < M, 1), 0)
    fb = tl.min(tl.load(FB + rn, rn < N, 1), 0)
    if fa & fb == 0:
        rk = (
            tl.cast(tl.arange(0, BK), tl.int64)
            if ADDRESS64
            else tl.arange(0, BK)
        )
        acc = tl.zeros((BM, BN), tl.float32)
        for k in range(tl.cdiv(K, BK)):
            kk = (tl.cast(k, tl.int64) if ADDRESS64 else k) * BK + rk
            sa = (
                tl.load(
                    AS + rm[:, None] * AS0 + kk[None, :] // 32 * AS1,
                    (rm[:, None] < M) & (kk[None, :] < K),
                    127,
                ).to(tl.int32)
                & 255
            )
            sb = (
                tl.load(
                    BS + rn[None, :] * BS0 + kk[:, None] // 32 * BS1,
                    (rn[None, :] < N) & (kk[:, None] < K),
                    127,
                ).to(tl.int32)
                & 255
            )
            ua = tl.load(
                A + rm[:, None] * A0 + kk[None, :] * A1,
                (rm[:, None] < M) & (kk[None, :] < K),
                0,
            )
            ub = tl.load(
                B + rn[None, :] * B0 + kk[:, None] * B1,
                (rn[None, :] < N) & (kk[:, None] < K),
                0,
            )
            av = _decode(ua, sa)
            bv = _decode(ub, sb)
            av = tl.gather(
                av, tl.broadcast_to(tl.arange(0, BK)[None, :], (BM, BK)), 1
            )
            bv = tl.gather(
                bv, tl.broadcast_to(tl.arange(0, BK)[:, None], (BK, BN)), 0
            )
            acc += tl.sum(av[:, :, None] * bv[None, :, :], 1)
        tl.store(
            Output + rm[:, None] * N + rn[None, :],
            acc,
            (rm[:, None] < M) & (rn[None, :] < N),
        )


@triton.jit(do_not_specialize=["A", "B", "AS", "BS", "FA", "FB"])
def _repair_coarse_union996(
    A: tl.float64,
    AS: tl.float64,
    B: tl.float64,
    BS: tl.float64,
    FA: tl.float64,
    FB: tl.float64,
    Output,
    META: tl.constexpr,
    ALIGN: tl.constexpr,
    PID996,
    FIRST996: tl.constexpr,
):
    AS = AS.to(tl.uint64, bitcast=True).to(tl.pointer_type(tl.uint8))
    if ALIGN[2]:
        AS = tl.multiple_of(AS, 16)
    BS = BS.to(tl.uint64, bitcast=True).to(tl.pointer_type(tl.uint8))
    if ALIGN[3]:
        BS = tl.multiple_of(BS, 16)
    FA = FA.to(tl.uint64, bitcast=True).to(tl.pointer_type(tl.int32))
    if ALIGN[4]:
        FA = tl.multiple_of(FA, 16)
    FB = FB.to(tl.uint64, bitcast=True).to(tl.pointer_type(tl.int32))
    if ALIGN[5]:
        FB = tl.multiple_of(FB, 16)
    A = A.to(tl.uint64, bitcast=True).to(tl.pointer_type(tl.int8))
    if ALIGN[0]:
        A = tl.multiple_of(A, 16)
    B = B.to(tl.uint64, bitcast=True).to(tl.pointer_type(tl.int8))
    if ALIGN[1]:
        B = tl.multiple_of(B, 16)
    M: tl.constexpr = META[0]
    N: tl.constexpr = META[1]
    K: tl.constexpr = META[2]
    A0: tl.constexpr = META[3]
    A1: tl.constexpr = META[4]
    AS0: tl.constexpr = META[5]
    AS1: tl.constexpr = META[6]
    B0: tl.constexpr = META[7]
    B1: tl.constexpr = META[8]
    BS0: tl.constexpr = META[9]
    BS1: tl.constexpr = META[10]
    BM: tl.constexpr = META[11]
    BN: tl.constexpr = META[12]
    BK: tl.constexpr = META[13]
    ADDRESS64: tl.constexpr = (
        (M + 4 * BM) * N + (N + 4 * BN) >= 2147483648
        or (M + 4 * BM) * K + (K + 4 * BK) >= 2147483648
        or (N + 4 * BN) * K + (K + 4 * BK) >= 2147483648
        or ((M + 4 * BM) * A0 + (K + 4 * BK) * A1 >= 2147483648)
        or ((M + 4 * BM) * AS0 + ((K + 4 * BK) // 32 + 1) * AS1 >= 2147483648)
        or ((N + 4 * BN) * B0 + (K + 4 * BK) * B1 >= 2147483648)
        or ((N + 4 * BN) * BS0 + ((K + 4 * BK) // 32 + 1) * BS1 >= 2147483648)
    )
    pid = (tl.cast(PID996, tl.int64) if ADDRESS64 else PID996) - FIRST996
    gm = pid // tl.cdiv(N, BN * 4)
    gn = pid % tl.cdiv(N, BN * 4)
    cr = gm * BM * 4 + (
        tl.cast(tl.arange(0, BM * 4), tl.int64)
        if ADDRESS64
        else tl.arange(0, BM * 4)
    )
    cc = gn * BN * 4 + (
        tl.cast(tl.arange(0, BN * 4), tl.int64)
        if ADDRESS64
        else tl.arange(0, BN * 4)
    )
    cfa = tl.min(tl.load(FA + cr, cr < M, 1), 0)
    cfb = tl.min(tl.load(FB + cc, cc < N, 1), 0)
    if cfa & cfb == 0:
        for tile818 in range(16):
            mi = tile818 // 4
            ni = tile818 % 4
            rm = (gm * 4 + mi) * BM + (
                tl.cast(tl.arange(0, BM), tl.int64)
                if ADDRESS64
                else tl.arange(0, BM)
            )
            rn = (gn * 4 + ni) * BN + (
                tl.cast(tl.arange(0, BN), tl.int64)
                if ADDRESS64
                else tl.arange(0, BN)
            )
            fa = tl.min(tl.load(FA + rm, rm < M, 1), 0)
            fb = tl.min(tl.load(FB + rn, rn < N, 1), 0)
            if fa & fb == 0:
                rk = (
                    tl.cast(tl.arange(0, BK), tl.int64)
                    if ADDRESS64
                    else tl.arange(0, BK)
                )
                acc = tl.zeros((BM, BN), tl.float32)
                for k in range(tl.cdiv(K, BK)):
                    kk = (tl.cast(k, tl.int64) if ADDRESS64 else k) * BK + rk
                    sa = (
                        tl.load(
                            AS + rm[:, None] * AS0 + kk[None, :] // 32 * AS1,
                            (rm[:, None] < M) & (kk[None, :] < K),
                            127,
                        ).to(tl.int32)
                        & 255
                    )
                    sb = (
                        tl.load(
                            BS + rn[None, :] * BS0 + kk[:, None] // 32 * BS1,
                            (rn[None, :] < N) & (kk[:, None] < K),
                            127,
                        ).to(tl.int32)
                        & 255
                    )
                    ua = tl.load(
                        A + rm[:, None] * A0 + kk[None, :] * A1,
                        (rm[:, None] < M) & (kk[None, :] < K),
                        0,
                    )
                    ub = tl.load(
                        B + rn[None, :] * B0 + kk[:, None] * B1,
                        (rn[None, :] < N) & (kk[:, None] < K),
                        0,
                    )
                    av = _decode(ua, sa)
                    bv = _decode(ub, sb)
                    av = tl.gather(
                        av,
                        tl.broadcast_to(tl.arange(0, BK)[None, :], (BM, BK)),
                        1,
                    )
                    bv = tl.gather(
                        bv,
                        tl.broadcast_to(tl.arange(0, BK)[:, None], (BK, BN)),
                        0,
                    )
                    acc += tl.sum(av[:, :, None] * bv[None, :, :], 1)
                tl.store(
                    Output + rm[:, None] * N + rn[None, :],
                    acc,
                    (rm[:, None] < M) & (rn[None, :] < N),
                )


@triton.jit(
    do_not_specialize=["RA993", "RAS993", "RB993", "RBS993", "FA993", "FB993"]
)
def _fast_union996(
    A,
    B,
    Output,
    RA993: tl.float64,
    RAS993: tl.float64,
    RB993: tl.float64,
    RBS993: tl.float64,
    FA993: tl.float64,
    FB993: tl.float64,
    META: tl.constexpr,
    SMETA993: tl.constexpr,
    ALIGN993: tl.constexpr,
):
    FIRST996: tl.constexpr = (
        (META[0] + META[3] - 1)
        // META[3]
        * ((META[1] + META[4] - 1) // META[4])
    )
    if tl.program_id(0) < FIRST996:
        M: tl.constexpr = META[0]
        N: tl.constexpr = META[1]
        K: tl.constexpr = META[2]
        BM: tl.constexpr = META[3]
        BN: tl.constexpr = META[4]
        BK: tl.constexpr = META[5]
        ADDRESS64: tl.constexpr = (
            (M + 4 * BM) * N + (N + 4 * BN) >= 2147483648
            or (M + 4 * BM) * K + (K + 4 * BK) >= 2147483648
            or (N + 4 * BN) * K + (K + 4 * BK) >= 2147483648
        )
        pid = (
            tl.cast(tl.program_id(0), tl.int64)
            if ADDRESS64
            else tl.program_id(0)
        )
        if BM == 128 and BN == 256:
            gm809 = tl.cdiv(M, BM)
            gn809 = tl.cdiv(N, BN)
            group809 = pid // (8 * gn809)
            first809 = group809 * 8
            size809 = tl.minimum(gm809 - first809, 8)
            local809 = pid % (8 * gn809)
            pm809 = first809 + local809 % size809
            pn809 = local809 // size809
        else:
            pm809 = pid // tl.cdiv(N, BN)
            pn809 = pid % tl.cdiv(N, BN)
        rm = pm809 * BM + (
            tl.cast(tl.arange(0, BM), tl.int64)
            if ADDRESS64
            else tl.arange(0, BM)
        )
        rn = pn809 * BN + (
            tl.cast(tl.arange(0, BN), tl.int64)
            if ADDRESS64
            else tl.arange(0, BN)
        )
        am = rm % M
        bn = rn % N
        rk = (
            tl.cast(tl.arange(0, BK), tl.int64)
            if ADDRESS64
            else tl.arange(0, BK)
        )
        acc = tl.zeros((BM, BN), tl.float32)
        for k in range(K // BK):
            kk = (tl.cast(k, tl.int64) if ADDRESS64 else k) * BK + rk
            av = tl.load(A + am[:, None] * K + kk[None, :])
            bv = tl.load(B + bn[None, :] * K + kk[:, None])
            acc = tl.dot(av, bv, acc)
        if K % BK != 0:
            kk = K // BK * BK + rk
            av = tl.load(A + am[:, None] * K + kk[None, :], kk[None, :] < K, 0)
            bv = tl.load(B + bn[None, :] * K + kk[:, None], kk[:, None] < K, 0)
            acc = tl.dot(av, bv, acc)
        flags_a993 = FA993.to(tl.uint64, bitcast=True).to(
            tl.pointer_type(tl.int32)
        )
        flags_b993 = FB993.to(tl.uint64, bitcast=True).to(
            tl.pointer_type(tl.int32)
        )
        if ALIGN993[4]:
            flags_a993 = tl.multiple_of(flags_a993, 16)
        if ALIGN993[5]:
            flags_b993 = tl.multiple_of(flags_b993, 16)
        fa993 = tl.load(flags_a993 + rm, rm < M, 1)
        fb993 = tl.load(flags_b993 + rn, rn < N, 1)
        groups_a993 = tl.min(tl.reshape(fa993, (BM // 16, 16)), 1)
        groups_b993 = tl.min(tl.reshape(fb993, (BN // 32, 32)), 1)
        safe_a993 = tl.reshape(
            tl.broadcast_to(groups_a993[:, None], (BM // 16, 16)), (BM,)
        )
        safe_b993 = tl.reshape(
            tl.broadcast_to(groups_b993[:, None], (BN // 32, 32)), (BN,)
        )
        safe993 = safe_a993[:, None] & safe_b993[None, :] != 0
        safe993 = tl.max_constancy(safe993, [16, 32])
        tl.store(
            Output + rm[:, None] * N + rn[None, :],
            acc,
            (rm[:, None] < M) & (rn[None, :] < N) & safe993,
        )
    elif SMETA993[0] >= 2048:
        _repair_coarse_union996(
            RA993,
            RAS993,
            RB993,
            RBS993,
            FA993,
            FB993,
            Output,
            SMETA993,
            ALIGN993,
            tl.program_id(0),
            FIRST996,
        )
    else:
        _repair_fine_union996(
            RA993,
            RAS993,
            RB993,
            RBS993,
            FA993,
            FB993,
            Output,
            SMETA993,
            ALIGN993,
            tl.program_id(0),
            FIRST996,
        )


@triton.jit(
    do_not_specialize=["RA993", "RAS993", "RB993", "RBS993", "FA993", "FB993"]
)
def _fast_store_union996(
    A,
    B,
    Output,
    RA993: tl.float64,
    RAS993: tl.float64,
    RB993: tl.float64,
    RBS993: tl.float64,
    FA993: tl.float64,
    FB993: tl.float64,
    META: tl.constexpr,
    SMETA993: tl.constexpr,
    ALIGN993: tl.constexpr,
):
    FIRST996: tl.constexpr = (
        (META[0] + META[3] - 1)
        // META[3]
        * ((META[1] + META[4] - 1) // META[4])
    )
    if tl.program_id(0) < FIRST996:
        M: tl.constexpr = META[0]
        N: tl.constexpr = META[1]
        K: tl.constexpr = META[2]
        BM: tl.constexpr = META[3]
        BN: tl.constexpr = META[4]
        BK: tl.constexpr = META[5]
        ADDRESS64: tl.constexpr = (
            (M + 4 * BM) * N + (N + 4 * BN) >= 2147483648
            or (M + 4 * BM) * K + (K + 4 * BK) >= 2147483648
            or (N + 4 * BN) * K + (K + 4 * BK) >= 2147483648
        )
        pid = (
            tl.cast(tl.program_id(0), tl.int64)
            if ADDRESS64
            else tl.program_id(0)
        )
        rm = pid // tl.cdiv(N, BN) * BM + (
            tl.cast(tl.arange(0, BM), tl.int64)
            if ADDRESS64
            else tl.arange(0, BM)
        )
        rn = pid % tl.cdiv(N, BN) * BN + (
            tl.cast(tl.arange(0, BN), tl.int64)
            if ADDRESS64
            else tl.arange(0, BN)
        )
        am = rm % M
        bn = rn % N
        rk = (
            tl.cast(tl.arange(0, BK), tl.int64)
            if ADDRESS64
            else tl.arange(0, BK)
        )
        acc = tl.zeros((BM, BN), tl.float32)
        for k in range(K // BK):
            kk = (tl.cast(k, tl.int64) if ADDRESS64 else k) * BK + rk
            av = tl.load(A + am[:, None] * K + kk[None, :])
            bv = tl.load(B + bn[None, :] * K + kk[:, None])
            acc = tl.dot(av, bv, acc)
        if K % BK != 0:
            kk = K // BK * BK + rk
            av = tl.load(A + am[:, None] * K + kk[None, :], kk[None, :] < K, 0)
            bv = tl.load(B + bn[None, :] * K + kk[:, None], kk[:, None] < K, 0)
            acc = tl.dot(av, bv, acc)
        flags_a993 = FA993.to(tl.uint64, bitcast=True).to(
            tl.pointer_type(tl.int32)
        )
        flags_b993 = FB993.to(tl.uint64, bitcast=True).to(
            tl.pointer_type(tl.int32)
        )
        if ALIGN993[4]:
            flags_a993 = tl.multiple_of(flags_a993, 16)
        if ALIGN993[5]:
            flags_b993 = tl.multiple_of(flags_b993, 16)
        fa993 = tl.load(flags_a993 + rm, rm < M, 1)
        fb993 = tl.load(flags_b993 + rn, rn < N, 1)
        groups_a993 = tl.min(tl.reshape(fa993, (BM // 16, 16)), 1)
        groups_b993 = tl.min(tl.reshape(fb993, (BN // 32, 32)), 1)
        safe_a993 = tl.reshape(
            tl.broadcast_to(groups_a993[:, None], (BM // 16, 16)), (BM,)
        )
        safe_b993 = tl.reshape(
            tl.broadcast_to(groups_b993[:, None], (BN // 32, 32)), (BN,)
        )
        safe993 = safe_a993[:, None] & safe_b993[None, :] != 0
        safe993 = tl.max_constancy(safe993, [16, 32])
        tl.store(
            Output + rm[:, None] * N + rn[None, :],
            acc,
            (rm[:, None] < M) & (rn[None, :] < N) & safe993,
        )
    elif SMETA993[0] >= 2048:
        _repair_coarse_union996(
            RA993,
            RAS993,
            RB993,
            RBS993,
            FA993,
            FB993,
            Output,
            SMETA993,
            ALIGN993,
            tl.program_id(0),
            FIRST996,
        )
    else:
        _repair_fine_union996(
            RA993,
            RAS993,
            RB993,
            RBS993,
            FA993,
            FB993,
            Output,
            SMETA993,
            ALIGN993,
            tl.program_id(0),
            FIRST996,
        )


__all__ = ["mxfp8_block_scaled_matmul"]
