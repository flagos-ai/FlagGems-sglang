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
def _slow(A, AS, B, BS, FA, FB, Output, META: tl.constexpr):
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
            acc = tl.dot(av, bv, acc, input_precision="ieee")
        tl.store(
            Output + rm[:, None] * N + rn[None, :],
            acc,
            (rm[:, None] < M) & (rn[None, :] < N),
        )


@triton.jit
def _fast(A, B, FA, FB, Output, META: tl.constexpr):
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


def _base_entry(a, a_scale, b, b_scale, output_dtype):
    m, k = a.shape
    n = b.shape[0]
    out = torch.empty((m, n), device=a.device, dtype=output_dtype)
    if m == 0 or n == 0:
        return out
    ad = torch.empty((m, k), device=a.device, dtype=torch.bfloat16)
    bd = torch.empty((n, k), device=a.device, dtype=torch.bfloat16)
    fa = torch.empty((m,), device=a.device, dtype=torch.int32)
    fb = torch.empty((n,), device=a.device, dtype=torch.int32)
    _thin921_dequant_both.run(
        a,
        a_scale,
        b,
        b_scale,
        ad,
        bd,
        fa,
        fb,
        (
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
            max(32, 0 if k == 0 else 1 << (k - 1).bit_length()),
            False,
            a.stride(1) == 1
            and a.stride(0) % 2 == 0
            and (a.data_ptr() % 2 == 0),
            b.stride(1) == 1
            and b.stride(0) % 2 == 0
            and (b.data_ptr() % 2 == 0),
            ad.stride(0),
            bd.stride(0),
            a.stride(1) == 1
            and a.stride(0) % 4 == 0
            and (a.data_ptr() % 4 == 0),
            b.stride(1) == 1
            and b.stride(0) % 4 == 0
            and (b.data_ptr() % 4 == 0),
        ),
        num_warps=1 if k > 8192 else 4,
        grid=(m + n,),
        warmup=False,
    )
    bm, bn, bk = (64, 128, 64)
    _fast.run(
        ad,
        bd,
        fa,
        fb,
        out,
        (m, n, k, bm, bn, bk),
        num_warps=4,
        num_stages=3,
        grid=((m + bm - 1) // bm * ((n + bn - 1) // bn),),
        warmup=False,
    )
    bm, bn, bk = (16, 32, 32)
    _thin921_slow.run(
        a,
        a_scale,
        b,
        b_scale,
        fa,
        fb,
        out,
        (
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
            bm,
            bn,
            bk,
        ),
        num_warps=4,
        num_stages=2,
        grid=((m + bm - 1) // bm * ((n + bn - 1) // bn),),
        warmup=False,
    )
    return out


@triton.jit
def _group_fast(A, B, FA, FB, Output, META: tl.constexpr):
    M: tl.constexpr = META[0]
    N: tl.constexpr = META[1]
    K: tl.constexpr = META[2]
    BM: tl.constexpr = META[3]
    BN: tl.constexpr = META[4]
    BK: tl.constexpr = META[5]
    GROUP: tl.constexpr = META[6]
    LDB: tl.constexpr = META[7]
    LDA: tl.constexpr = META[8]
    ADDRESS64: tl.constexpr = (
        (M + 4 * BM) * N + (N + 4 * BN) >= 2147483648
        or (M + 4 * BM) * K + (K + 4 * BK) >= 2147483648
        or (N + 4 * BN) * K + (K + 4 * BK) >= 2147483648
        or ((M + 4 * BM) * LDA + (K + 4 * BK) >= 2147483648)
        or ((N + 4 * BN) * LDB + (K + 4 * BK) >= 2147483648)
    )
    pid = (
        tl.cast(tl.program_id(0), tl.int64) if ADDRESS64 else tl.program_id(0)
    )
    gm = tl.cdiv(M, BM)
    gn = tl.cdiv(N, BN)
    group = pid // (GROUP * gn)
    first_m = group * GROUP
    group_size = tl.minimum(gm - first_m, GROUP)
    pm = first_m + pid % (GROUP * gn) % group_size
    pn = pid % (GROUP * gn) // group_size
    rm = pm * BM + (
        tl.cast(tl.arange(0, BM), tl.int64) if ADDRESS64 else tl.arange(0, BM)
    )
    rn = pn * BN + (
        tl.cast(tl.arange(0, BN), tl.int64) if ADDRESS64 else tl.arange(0, BN)
    )
    rk = tl.cast(tl.arange(0, BK), tl.int64) if ADDRESS64 else tl.arange(0, BK)
    acc = tl.zeros((BM, BN), tl.float32)
    for k in range(tl.cdiv(K, BK)):
        kk = (tl.cast(k, tl.int64) if ADDRESS64 else k) * BK + rk
        if M >= 2048:
            if M % BM == 0 and K % BK == 0:
                av = tl.load(A + rm[:, None] * LDA + kk[None, :])
            elif M % BM == 0:
                av = tl.load(
                    A + rm[:, None] * LDA + kk[None, :], kk[None, :] < K, 0
                )
            elif K % BK == 0:
                av = tl.load(
                    A + rm[:, None] * LDA + kk[None, :], rm[:, None] < M, 0
                )
            else:
                av = tl.load(
                    A + rm[:, None] * LDA + kk[None, :],
                    (rm[:, None] < M) & (kk[None, :] < K),
                    0,
                )
        else:
            av = tl.load(
                A + rm[:, None] * LDA + kk[None, :],
                (rm[:, None] < M) & (kk[None, :] < K),
                0,
            )
        if M >= 2048:
            if N % BN == 0 and K % BK == 0:
                bv = tl.load(B + rn[None, :] * LDB + kk[:, None])
            elif N % BN == 0:
                bv = tl.load(
                    B + rn[None, :] * LDB + kk[:, None], kk[:, None] < K, 0
                )
            elif K % BK == 0:
                bv = tl.load(
                    B + rn[None, :] * LDB + kk[:, None], rn[None, :] < N, 0
                )
            else:
                bv = tl.load(
                    B + rn[None, :] * LDB + kk[:, None],
                    (rn[None, :] < N) & (kk[:, None] < K),
                    0,
                )
        else:
            bv = tl.load(
                B + rn[None, :] * LDB + kk[:, None],
                (rn[None, :] < N) & (kk[:, None] < K),
                0,
            )
        acc = tl.dot(av, bv, acc)
    if M >= 2048:
        if M % BM == 0 and N % BN == 0:
            tl.store(Output + rm[:, None] * N + rn[None, :], acc)
        elif M % BM == 0:
            tl.store(
                Output + rm[:, None] * N + rn[None, :], acc, rn[None, :] < N
            )
        elif N % BN == 0:
            tl.store(
                Output + rm[:, None] * N + rn[None, :], acc, rm[:, None] < M
            )
        else:
            tl.store(
                Output + rm[:, None] * N + rn[None, :],
                acc,
                (rm[:, None] < M) & (rn[None, :] < N),
            )
    else:
        tl.store(
            Output + rm[:, None] * N + rn[None, :],
            acc,
            (rm[:, None] < M) & (rn[None, :] < N),
        )


def _group_entry(a, a_scale, b, b_scale, output_dtype):
    m, k = a.shape
    variant = "coarse" if m >= 2048 else "base"
    n = b.shape[0]
    out = torch.empty((m, n), device=a.device, dtype=output_dtype)
    if m == 0 or n == 0:
        return out
    ad = torch.empty(
        (m, k + (32 if m >= 2048 and n >= 2048 and (k <= 4096) else 0)),
        device=a.device,
        dtype=torch.bfloat16,
    )
    bd = torch.empty(
        (n, k + (32 if m >= 2048 else 0)),
        device=a.device,
        dtype=torch.bfloat16,
    )
    fa = torch.empty((m,), device=a.device, dtype=torch.int32)
    fb = torch.empty((n,), device=a.device, dtype=torch.int32)
    _thin921_dequant_both.run(
        a,
        a_scale,
        b,
        b_scale,
        ad,
        bd,
        fa,
        fb,
        (
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
            max(32, 0 if k == 0 else 1 << (k - 1).bit_length()),
            k <= 8192,
            a.stride(1) == 1
            and a.stride(0) % 2 == 0
            and (a.data_ptr() % 2 == 0),
            b.stride(1) == 1
            and b.stride(0) % 2 == 0
            and (b.data_ptr() % 2 == 0),
            ad.stride(0),
            bd.stride(0),
            a.stride(1) == 1
            and a.stride(0) % 4 == 0
            and (a.data_ptr() % 4 == 0),
            b.stride(1) == 1
            and b.stride(0) % 4 == 0
            and (b.data_ptr() % 4 == 0),
        ),
        num_warps=1 if k > 8192 else 4,
        grid=(m + n,),
        warmup=False,
    )
    bm, bn, bk = (128, 128, 64) if m >= 512 else (64, 64, 64)
    if m >= 2048:
        bm, bn, bk = (128, 256, 32)
    if output_dtype == torch.float32:
        bn = min(bn, 128)
    if m >= 2048 and output_dtype == torch.bfloat16:
        wide_tiles = (m + bm - 1) // bm * ((n + bn - 1) // bn)
        compact_tiles = (m + 64 - 1) // 64 * ((n + 128 - 1) // 128)
        if (compact_tiles + 104 - 1) // 104 <= 2 * (
            (wide_tiles + 104 - 1) // 104
        ):
            bm, bn, bk = (64, 128, 64)
    warps = 8 if bm == 128 else 4
    stages = 3
    if (
        m >= 2048
        and k >= 2048
        and (output_dtype == torch.bfloat16)
        and (bm == 128)
        and (bn == 256)
        and (m % 256 == 0)
        and (n % 256 == 0)
        and (k % 64 == 0)
    ):
        regular_waves = (
            (m + bm - 1) // bm * ((n + bn - 1) // bn) + 104 - 1
        ) // 104
        large_waves = (
            (m + 256 - 1) // 256 * ((n + 256 - 1) // 256) + 104 - 1
        ) // 104
        if 5 * large_waves < 3 * regular_waves:
            bm = 256
            stages = 2
    group = 8
    hybrid_cut = 0
    if (
        m >= 2048
        and n >= 2048
        and (2048 <= k <= 4096)
        and (output_dtype == torch.bfloat16)
        and (bm >= 128)
        and (bn == 256)
        and (m % 256 == 0)
        and (n % 256 == 0)
        and (k % 64 == 0)
    ):
        total = (m + 256 - 1) // 256 * ((n + 256 - 1) // 256)
        remainder = total % 104
        if total >= 104 and 0 < remainder <= 52:
            hybrid_cut = total // 104 * 104
    if hybrid_cut:
        _hybrid_main506.run(
            ad,
            bd,
            fa,
            fb,
            out,
            (
                m,
                n,
                k,
                256,
                256,
                32,
                group,
                bd.stride(0),
                ad.stride(0),
                hybrid_cut,
            ),
            num_warps=8,
            num_stages=2,
            scenario="reduceSmemUsage",
            grid=(hybrid_cut,),
            warmup=False,
        )
        _hybrid_tail506.run(
            ad,
            bd,
            fa,
            fb,
            out,
            (
                m,
                n,
                k,
                128,
                256,
                32,
                group,
                bd.stride(0),
                ad.stride(0),
                hybrid_cut,
            ),
            num_warps=8,
            num_stages=3,
            scenario="reduceSmemUsage",
            grid=(2 * (total - hybrid_cut),),
            warmup=False,
        )
    else:
        _group_fast.run(
            ad,
            bd,
            fa,
            fb,
            out,
            (m, n, k, bm, bn, bk, group, bd.stride(0), ad.stride(0)),
            num_warps=warps,
            num_stages=stages,
            scenario=(
                "reduceSmemUsage"
                if m >= 2048
                else (
                    "flashattn-fwd"
                    if m >= 512 and n % 128 == 0 and (k % 64 == 0)
                    else ""
                )
            ),
            grid=((m + bm - 1) // bm * ((n + bn - 1) // bn),),
            warmup=False,
        )
    bm, bn, bk = (32, 64, 32)
    slow = _thin921_slow_coarse if variant == "coarse" else _thin921_slow
    factor = 4 if variant == "coarse" else 1
    slow.run(
        a,
        a_scale,
        b,
        b_scale,
        fa,
        fb,
        out,
        (
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
            bm,
            bn,
            bk,
        ),
        num_warps=4,
        num_stages=2,
        grid=(
            (m + bm * factor - 1)
            // (bm * factor)
            * ((n + bn * factor - 1) // (bn * factor)),
        ),
        warmup=False,
    )
    return out


def _selected_entry(a, a_scale, b, b_scale, output_dtype):
    if a.shape[1] >= 1024:
        return _group_entry(a, a_scale, b, b_scale, output_dtype)
    return _base_entry(a, a_scale, b, b_scale, output_dtype)


@triton.jit
def _slow_coarse(A, AS, B, BS, FA, FB, Output, META: tl.constexpr):
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
    pm = pid // tl.cdiv(N, BN * 4)
    pn = pid % tl.cdiv(N, BN * 4)
    crm = pm * BM * 4 + (
        tl.cast(tl.arange(0, BM * 4), tl.int64)
        if ADDRESS64
        else tl.arange(0, BM * 4)
    )
    crn = pn * BN * 4 + (
        tl.cast(tl.arange(0, BN * 4), tl.int64)
        if ADDRESS64
        else tl.arange(0, BN * 4)
    )
    safe_a = tl.min(tl.load(FA + crm, crm < M, 1), 0)
    safe_b = tl.min(tl.load(FB + crn, crn < N, 1), 0)
    if safe_a & safe_b == 0:
        for sub in range(16):
            rm = (pm * 4 + sub // 4) * BM + (
                tl.cast(tl.arange(0, BM), tl.int64)
                if ADDRESS64
                else tl.arange(0, BM)
            )
            rn = (pn * 4 + sub % 4) * BN + (
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
                    acc = tl.dot(av, bv, acc, input_precision="ieee")
                tl.store(
                    Output + rm[:, None] * N + rn[None, :],
                    acc,
                    (rm[:, None] < M) & (rn[None, :] < N),
                )


@triton.jit
def _decode_halfbase(u, s):
    u = u.to(tl.int32) & 255
    s = s.to(tl.int32) & 255
    half_bits = (u & 128) << 8 | (u & 127) << 7
    value = half_bits.to(tl.uint16).to(tl.float16, bitcast=True).to(tl.float32)
    scale = (s + 8 << 23).to(tl.float32, bitcast=True)
    return tl.where(u & 127 == 127, float("nan"), value * scale)


@triton.jit
def _dequant_row_chunked(
    A,
    S,
    D,
    F,
    row,
    R: tl.constexpr,
    K: tl.constexpr,
    A0: tl.constexpr,
    A1: tl.constexpr,
    S0: tl.constexpr,
    S1: tl.constexpr,
    B: tl.constexpr,
    D0: tl.constexpr,
):
    ADDRESS64: tl.constexpr = (
        (R + 1) * A0 + (2 * K + B) * A1 >= 2147483648
        or (R + 1) * S0 + ((2 * K + B) // 32 + 1) * S1 >= 2147483648
        or (R + 1) * D0 + 2 * K + B >= 2147483648
    )
    row = tl.cast(row, tl.int64) if ADDRESS64 else row
    CHUNK: tl.constexpr = 1024 if K > 8192 else 256
    safe = tl.full((), 1, tl.int32)
    for chunk in range(tl.cdiv(K, CHUNK)):
        cols = (tl.cast(chunk, tl.int64) if ADDRESS64 else chunk) * CHUNK + (
            tl.cast(tl.arange(0, CHUNK), tl.int64)
            if ADDRESS64
            else tl.arange(0, CHUNK)
        )
        u = tl.load(A + row * A0 + cols * A1, cols < K, 0)
        s = (
            tl.load(S + row * S0 + cols // 32 * S1, cols < K, 127).to(tl.int32)
            & 255
        )
        value = _decode_halfbase(u, s)
        tl.store(D + row * D0 + cols, value, cols < K)
        safe = safe & tl.min(((s >= 10) & (s <= 246)).to(tl.int32), 0)
    tl.store(F + row, safe)


@triton.jit
def _dequant_row_full(
    A,
    S,
    D,
    F,
    row,
    R: tl.constexpr,
    K: tl.constexpr,
    A0: tl.constexpr,
    A1: tl.constexpr,
    S0: tl.constexpr,
    S1: tl.constexpr,
    B: tl.constexpr,
    D0: tl.constexpr,
):
    ADDRESS64: tl.constexpr = (
        (R + 1) * A0 + (2 * K + B) * A1 >= 2147483648
        or (R + 1) * S0 + ((2 * K + B) // 32 + 1) * S1 >= 2147483648
        or (R + 1) * D0 + 2 * K + B >= 2147483648
    )
    row = tl.cast(row, tl.int64) if ADDRESS64 else row
    cols = tl.cast(tl.arange(0, B), tl.int64) if ADDRESS64 else tl.arange(0, B)
    u = tl.load(A + row * A0 + cols * A1, cols < K, 0)
    s = (
        tl.load(S + row * S0 + cols // 32 * S1, cols < K, 127).to(tl.int32)
        & 255
    )
    value = _decode_halfbase(u, s)
    tl.store(D + row * D0 + cols, value, cols < K)
    safe = tl.min(((s >= 10) & (s <= 246)).to(tl.int32), 0)
    tl.store(F + row, safe)


@triton.jit
def _dequant_both(A, AS, B, BS, AD, BD, FA, FB, META: tl.constexpr):
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
    BLOCK: tl.constexpr = META[11]
    FULL: tl.constexpr = META[12]
    PACK_A: tl.constexpr = META[13]
    PACK_B: tl.constexpr = META[14]
    AD0: tl.constexpr = META[15]
    BD0: tl.constexpr = META[16]
    QUAD_A: tl.constexpr = META[17]
    QUAD_B: tl.constexpr = META[18]
    ADDRESS64: tl.constexpr = (
        (M + 1024) * N + (N + 1024) >= 2147483648
        or (M + 1024) * K + (2 * K + 4096) >= 2147483648
        or (N + 1024) * K + (2 * K + 4096) >= 2147483648
        or ((M + 1024) * A0 + (2 * K + 4096) * A1 >= 2147483648)
        or ((M + 1024) * AS0 + ((2 * K + 4096) // 32 + 1) * AS1 >= 2147483648)
        or ((N + 1024) * B0 + (2 * K + 4096) * B1 >= 2147483648)
        or ((N + 1024) * BS0 + ((2 * K + 4096) // 32 + 1) * BS1 >= 2147483648)
        or ((M + 1024) * AD0 + (2 * K + 4096) >= 2147483648)
        or ((N + 1024) * BD0 + (2 * K + 4096) >= 2147483648)
    )
    row = (
        tl.cast(tl.program_id(0), tl.int64) if ADDRESS64 else tl.program_id(0)
    )
    if row < M:
        if FULL and QUAD_A:
            _dequant_row_quad451(
                A, AS, AD, FA, row, M, K, A0, A1, AS0, AS1, BLOCK, AD0
            )
        elif FULL and PACK_A:
            _dequant_row_packed425(
                A, AS, AD, FA, row, M, K, A0, A1, AS0, AS1, BLOCK, AD0
            )
        elif FULL:
            _dequant_row_full(
                A, AS, AD, FA, row, M, K, A0, A1, AS0, AS1, BLOCK, AD0
            )
        else:
            _dequant_row_chunked(
                A, AS, AD, FA, row, M, K, A0, A1, AS0, AS1, BLOCK, AD0
            )
    elif FULL and QUAD_B:
        _dequant_row_quad451(
            B, BS, BD, FB, row - M, N, K, B0, B1, BS0, BS1, BLOCK, BD0
        )
    elif FULL and PACK_B:
        _dequant_row_packed425(
            B, BS, BD, FB, row - M, N, K, B0, B1, BS0, BS1, BLOCK, BD0
        )
    elif FULL:
        _dequant_row_full(
            B, BS, BD, FB, row - M, N, K, B0, B1, BS0, BS1, BLOCK, BD0
        )
    else:
        _dequant_row_chunked(
            B, BS, BD, FB, row - M, N, K, B0, B1, BS0, BS1, BLOCK, BD0
        )


@triton.jit
def _group_split412(A, B, FA, FB, Output, META: tl.constexpr):
    M: tl.constexpr = META[0]
    N: tl.constexpr = META[1]
    K: tl.constexpr = META[2]
    BM: tl.constexpr = META[3]
    BN: tl.constexpr = META[4]
    BK: tl.constexpr = META[5]
    GROUP: tl.constexpr = META[6]
    SPLIT: tl.constexpr = META[7]
    ADDRESS64: tl.constexpr = (
        (M + 4 * BM) * N + (N + 4 * BN) >= 2147483648
        or (M + 4 * BM) * K + (K + 4 * BK) >= 2147483648
        or (N + 4 * BN) * K + (K + 4 * BK) >= 2147483648
        or (
            (SPLIT + 1) * M * N + (M + 4 * BM) * N + (N + 4 * BN) >= 2147483648
        )
    )
    partition = (
        tl.cast(tl.program_id(1), tl.int64) if ADDRESS64 else tl.program_id(1)
    )
    PART_K: tl.constexpr = (K + SPLIT * BK - 1) // (SPLIT * BK) * BK
    pid = (
        tl.cast(tl.program_id(0), tl.int64) if ADDRESS64 else tl.program_id(0)
    )
    gm = tl.cdiv(M, BM)
    gn = tl.cdiv(N, BN)
    group = pid // (GROUP * gn)
    first_m = group * GROUP
    group_size = tl.minimum(gm - first_m, GROUP)
    pm = first_m + pid % (GROUP * gn) % group_size
    pn = pid % (GROUP * gn) // group_size
    rm = pm * BM + (
        tl.cast(tl.arange(0, BM), tl.int64) if ADDRESS64 else tl.arange(0, BM)
    )
    rn = pn * BN + (
        tl.cast(tl.arange(0, BN), tl.int64) if ADDRESS64 else tl.arange(0, BN)
    )
    rk = tl.cast(tl.arange(0, BK), tl.int64) if ADDRESS64 else tl.arange(0, BK)
    acc = tl.zeros((BM, BN), tl.float32)
    for k in range(PART_K // BK):
        kk = (
            partition * PART_K
            + (tl.cast(k, tl.int64) if ADDRESS64 else k) * BK
            + rk
        )
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
        Output + partition * M * N + rm[:, None] * N + rn[None, :],
        acc,
        (rm[:, None] < M) & (rn[None, :] < N),
    )


@triton.jit
def _slow_merge419(A, AS, B, BS, FA, FB, Output, PARTIAL, META: tl.constexpr):
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
        or (3 * M * N + (M + 4 * BM) * N + (N + 4 * BN) >= 2147483648)
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
            acc = tl.dot(av, bv, acc, input_precision="ieee")
        tl.store(
            Output + rm[:, None] * N + rn[None, :],
            acc,
            (rm[:, None] < M) & (rn[None, :] < N),
        )
    else:
        i = rm[:, None] * N + rn[None, :]
        mask = (rm[:, None] < M) & (rn[None, :] < N)
        p0 = tl.load(PARTIAL + i, mask, 0)
        p1 = tl.load(PARTIAL + M * N + i, mask, 0)
        tl.store(Output + i, p0 + p1, mask)


def _group_split_merge_entry419(a, a_scale, b, b_scale, output_dtype):
    m, k = a.shape

    n = b.shape[0]
    out = torch.empty((m, n), device=a.device, dtype=output_dtype)
    if m == 0 or n == 0:
        return out
    ad = torch.empty((m, k), device=a.device, dtype=torch.bfloat16)
    bd = torch.empty((n, k), device=a.device, dtype=torch.bfloat16)
    fa = torch.empty((m,), device=a.device, dtype=torch.int32)
    fb = torch.empty((n,), device=a.device, dtype=torch.int32)
    _thin921_dequant_both.run(
        a,
        a_scale,
        b,
        b_scale,
        ad,
        bd,
        fa,
        fb,
        (
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
            max(32, 0 if k == 0 else 1 << (k - 1).bit_length()),
            k <= 8192,
            a.stride(1) == 1
            and a.stride(0) % 2 == 0
            and (a.data_ptr() % 2 == 0),
            b.stride(1) == 1
            and b.stride(0) % 2 == 0
            and (b.data_ptr() % 2 == 0),
            ad.stride(0),
            bd.stride(0),
            a.stride(1) == 1
            and a.stride(0) % 4 == 0
            and (a.data_ptr() % 4 == 0),
            b.stride(1) == 1
            and b.stride(0) % 4 == 0
            and (b.data_ptr() % 4 == 0),
        ),
        num_warps=1 if k > 8192 else 4,
        grid=(m + n,),
        warmup=False,
    )
    bm, bn, bk = (128, 128, 64) if m >= 512 else (64, 64, 64)
    if m >= 2048:
        bm, bn, bk = (128, 256, 32)
    if output_dtype == torch.float32:
        bn = min(bn, 128)
    warps = 8 if bm == 128 else 4
    stages = 3
    group = 8
    partial = torch.empty((2, m, n), device=a.device, dtype=torch.float32)
    _group_split412.run(
        ad,
        bd,
        fa,
        fb,
        partial,
        (m, n, k, bm, bn, bk, group, 2),
        num_warps=warps,
        num_stages=stages,
        grid=((m + bm - 1) // bm * ((n + bn - 1) // bn), 2),
        warmup=False,
    )
    bm, bn, bk = (32, 64, 32)

    factor = 1
    _thin921_slow_merge419.run(
        a,
        a_scale,
        b,
        b_scale,
        fa,
        fb,
        out,
        partial,
        (
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
            bm,
            bn,
            bk,
        ),
        num_warps=4,
        num_stages=2,
        grid=(
            (m + bm * factor - 1)
            // (bm * factor)
            * ((n + bn * factor - 1) // (bn * factor)),
        ),
        warmup=False,
    )
    return out


def mxfp8_block_scaled_matmul(a, a_scale, b, b_scale, output_dtype):
    m, k = a.shape
    n = b.shape[0]
    if m > 0 and n > 0 and (m < 2048) and (k >= 4096) and (m * n <= 2097152):
        block = 128 if m >= 512 else 64
        tiles = (m + block - 1) // block * ((n + block - 1) // block)
        waves = (tiles + 104 - 1) // 104
        split_waves = (2 * tiles + 104 - 1) // 104
        if split_waves < 2 * waves:
            return _group_split_merge_entry419(
                a, a_scale, b, b_scale, output_dtype
            )
    return _selected_entry(a, a_scale, b, b_scale, output_dtype)


@triton.jit
def _dequant_row_packed425(
    A,
    S,
    D,
    F,
    row,
    R: tl.constexpr,
    K: tl.constexpr,
    A0: tl.constexpr,
    A1: tl.constexpr,
    S0: tl.constexpr,
    S1: tl.constexpr,
    B: tl.constexpr,
    D0: tl.constexpr,
):
    ADDRESS64: tl.constexpr = (
        (R + 1) * A0 + (2 * K + B) * A1 >= 2147483648
        or (R + 1) * S0 + ((2 * K + B) // 32 + 1) * S1 >= 2147483648
        or (R + 1) * D0 + 2 * K + B >= 2147483648
    )
    row = tl.cast(row, tl.int64) if ADDRESS64 else row
    pairs = (
        tl.cast(tl.arange(0, B // 2), tl.int64)
        if ADDRESS64
        else tl.arange(0, B // 2)
    )
    word = tl.load(
        A.to(tl.pointer_type(tl.uint16)) + row * (A0 // 2) + pairs,
        pairs < K // 2,
        0,
    ).to(tl.int32)
    s = (
        tl.load(S + row * S0 + pairs // 16 * S1, pairs < K // 2, 127).to(
            tl.int32
        )
        & 255
    )
    lowmag = word & 127
    highmag = word & 32512
    lowbits = (word & 128) << 8 | lowmag << 7
    highbits = word & 32768 | highmag >> 1
    lowvalue = (
        lowbits.to(tl.uint16).to(tl.float16, bitcast=True).to(tl.float32)
    )
    highvalue = (
        highbits.to(tl.uint16).to(tl.float16, bitcast=True).to(tl.float32)
    )
    scale = (s + 8 << 23).to(tl.float32, bitcast=True)
    lv = tl.where(lowmag == 127, float("nan"), lowvalue * scale)
    hv = tl.where(highmag == 32512, float("nan"), highvalue * scale)
    low = lv.to(tl.uint32, bitcast=True) >> 16
    high = hv.to(tl.uint32, bitcast=True) & 4294901760
    tl.store(
        D.to(tl.pointer_type(tl.uint32)) + row * (D0 // 2) + pairs,
        low | high,
        pairs < K // 2,
    )
    safe = tl.min(((s >= 10) & (s <= 246)).to(tl.int32), 0)
    tl.store(F + row, safe)


@triton.jit
def _pair_word451(word, s):
    lowmag = word & 127
    highmag = word & 32512
    lowbits = (word & 128) << 8 | lowmag << 7
    highbits = word & 32768 | highmag >> 1
    lowvalue = (
        lowbits.to(tl.uint16).to(tl.float16, bitcast=True).to(tl.float32)
    )
    highvalue = (
        highbits.to(tl.uint16).to(tl.float16, bitcast=True).to(tl.float32)
    )
    scale = (s + 8 << 23).to(tl.float32, bitcast=True)
    lv = tl.where(lowmag == 127, float("nan"), lowvalue * scale)
    hv = tl.where(highmag == 32512, float("nan"), highvalue * scale)
    low = lv.to(tl.uint32, bitcast=True) >> 16
    high = hv.to(tl.uint32, bitcast=True) & 4294901760
    return low | high


@triton.jit
def _dequant_row_quad451(
    A,
    S,
    D,
    F,
    row,
    R: tl.constexpr,
    K: tl.constexpr,
    A0: tl.constexpr,
    A1: tl.constexpr,
    S0: tl.constexpr,
    S1: tl.constexpr,
    B: tl.constexpr,
    D0: tl.constexpr,
):
    ADDRESS64: tl.constexpr = (
        (R + 1) * A0 + (2 * K + B) * A1 >= 2147483648
        or (R + 1) * S0 + ((2 * K + B) // 32 + 1) * S1 >= 2147483648
        or (R + 1) * D0 + 2 * K + B >= 2147483648
    )
    row = tl.cast(row, tl.int64) if ADDRESS64 else row
    quads = (
        tl.cast(tl.arange(0, B // 4), tl.int64)
        if ADDRESS64
        else tl.arange(0, B // 4)
    )
    word = tl.load(
        A.to(tl.pointer_type(tl.uint32)) + row * (A0 // 4) + quads,
        quads < K // 4,
        0,
    )
    s = (
        tl.load(S + row * S0 + quads // 8 * S1, quads < K // 4, 127).to(
            tl.int32
        )
        & 255
    )
    first = _pair_word451(word & 65535, s)
    second = _pair_word451(word >> 16, s)
    dest = D.to(tl.pointer_type(tl.uint64)) + row * (D0 // 4) + quads
    packed = first.to(tl.uint64) | second.to(tl.uint64) << 32
    tl.store(dest, packed, quads < K // 4)
    safe = tl.min(((s >= 10) & (s <= 246)).to(tl.int32), 0)
    tl.store(F + row, safe)


@triton.jit
def _tile_hybrid505(
    A,
    B,
    FA,
    FB,
    Output,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    GROUP: tl.constexpr,
    LDB: tl.constexpr,
    LDA: tl.constexpr,
    pid,
    GRID_BM: tl.constexpr,
    ROW_OFFSET,
    PIPE_STAGES: tl.constexpr,
):
    ADDRESS64: tl.constexpr = (
        (M + 4 * BM) * N + (N + 4 * BN) >= 2147483648
        or (M + 4 * BM) * K + (K + 4 * BK) >= 2147483648
        or (N + 4 * BN) * K + (K + 4 * BK) >= 2147483648
        or ((M + 4 * BM) * LDA + (K + 4 * BK) >= 2147483648)
        or ((N + 4 * BN) * LDB + (K + 4 * BK) >= 2147483648)
    )
    ROW_OFFSET = tl.cast(ROW_OFFSET, tl.int64) if ADDRESS64 else ROW_OFFSET
    pid = tl.cast(pid, tl.int64) if ADDRESS64 else pid
    gm = tl.cdiv(M, GRID_BM)
    gn = tl.cdiv(N, BN)
    group = pid // (GROUP * gn)
    first_m = group * GROUP
    group_size = tl.minimum(gm - first_m, GROUP)
    pm = first_m + pid % (GROUP * gn) % group_size
    pn = pid % (GROUP * gn) // group_size
    rm = (
        pm * GRID_BM
        + ROW_OFFSET
        + (
            tl.cast(tl.arange(0, BM), tl.int64)
            if ADDRESS64
            else tl.arange(0, BM)
        )
    )
    rn = pn * BN + (
        tl.cast(tl.arange(0, BN), tl.int64) if ADDRESS64 else tl.arange(0, BN)
    )
    rk = tl.cast(tl.arange(0, BK), tl.int64) if ADDRESS64 else tl.arange(0, BK)
    acc = tl.zeros((BM, BN), tl.float32)
    for k in tl.range(0, tl.cdiv(K, BK), num_stages=PIPE_STAGES):
        kk = (tl.cast(k, tl.int64) if ADDRESS64 else k) * BK + rk
        if M % GRID_BM == 0 and K % BK == 0:
            av = tl.load(A + rm[:, None] * LDA + kk[None, :])
        elif M % GRID_BM == 0:
            av = tl.load(
                A + rm[:, None] * LDA + kk[None, :], kk[None, :] < K, 0
            )
        elif K % BK == 0:
            av = tl.load(
                A + rm[:, None] * LDA + kk[None, :], rm[:, None] < M, 0
            )
        else:
            av = tl.load(
                A + rm[:, None] * LDA + kk[None, :],
                (rm[:, None] < M) & (kk[None, :] < K),
                0,
            )
        if N % BN == 0 and K % BK == 0:
            bv = tl.load(B + rn[None, :] * LDB + kk[:, None])
        elif N % BN == 0:
            bv = tl.load(
                B + rn[None, :] * LDB + kk[:, None], kk[:, None] < K, 0
            )
        elif K % BK == 0:
            bv = tl.load(
                B + rn[None, :] * LDB + kk[:, None], rn[None, :] < N, 0
            )
        else:
            bv = tl.load(
                B + rn[None, :] * LDB + kk[:, None],
                (rn[None, :] < N) & (kk[:, None] < K),
                0,
            )
        acc = tl.dot(av, bv, acc)
    if M % GRID_BM == 0 and N % BN == 0:
        tl.store(Output + rm[:, None] * N + rn[None, :], acc)
    elif M % GRID_BM == 0:
        tl.store(Output + rm[:, None] * N + rn[None, :], acc, rn[None, :] < N)
    elif N % BN == 0:
        tl.store(Output + rm[:, None] * N + rn[None, :], acc, rm[:, None] < M)
    else:
        tl.store(
            Output + rm[:, None] * N + rn[None, :],
            acc,
            (rm[:, None] < M) & (rn[None, :] < N),
        )


@triton.jit
def _hybrid_main506(A, B, FA, FB, Output, META: tl.constexpr):
    M: tl.constexpr = META[0]
    N: tl.constexpr = META[1]
    K: tl.constexpr = META[2]
    BM: tl.constexpr = META[3]
    BN: tl.constexpr = META[4]
    BK: tl.constexpr = META[5]
    GROUP: tl.constexpr = META[6]
    LDB: tl.constexpr = META[7]
    LDA: tl.constexpr = META[8]

    ADDRESS64: tl.constexpr = (
        (M + 4 * BM) * N + (N + 4 * BN) >= 2147483648
        or (M + 4 * BM) * K + (K + 4 * BK) >= 2147483648
        or (N + 4 * BN) * K + (K + 4 * BK) >= 2147483648
        or ((M + 4 * BM) * LDA + (K + 4 * BK) >= 2147483648)
        or ((N + 4 * BN) * LDB + (K + 4 * BK) >= 2147483648)
    )
    pid = (
        tl.cast(tl.program_id(0), tl.int64) if ADDRESS64 else tl.program_id(0)
    )
    _tile_hybrid505(
        A,
        B,
        FA,
        FB,
        Output,
        M,
        N,
        K,
        256,
        BN,
        BK,
        GROUP,
        LDB,
        LDA,
        pid,
        256,
        0,
        2,
    )


@triton.jit
def _hybrid_tail506(A, B, FA, FB, Output, META: tl.constexpr):
    M: tl.constexpr = META[0]
    N: tl.constexpr = META[1]
    K: tl.constexpr = META[2]
    BM: tl.constexpr = META[3]
    BN: tl.constexpr = META[4]
    BK: tl.constexpr = META[5]
    GROUP: tl.constexpr = META[6]
    LDB: tl.constexpr = META[7]
    LDA: tl.constexpr = META[8]
    CUT: tl.constexpr = META[9]
    ADDRESS64: tl.constexpr = (
        (M + 4 * BM) * N + (N + 4 * BN) >= 2147483648
        or (M + 4 * BM) * K + (K + 4 * BK) >= 2147483648
        or (N + 4 * BN) * K + (K + 4 * BK) >= 2147483648
        or ((M + 4 * BM) * LDA + (K + 4 * BK) >= 2147483648)
        or ((N + 4 * BN) * LDB + (K + 4 * BK) >= 2147483648)
    )
    pid = (
        tl.cast(tl.program_id(0), tl.int64) if ADDRESS64 else tl.program_id(0)
    )
    tile = CUT + pid // 2
    subrow = pid % 2 * 128
    _tile_hybrid505(
        A,
        B,
        FA,
        FB,
        Output,
        M,
        N,
        K,
        128,
        BN,
        BK,
        GROUP,
        LDB,
        LDA,
        tile,
        256,
        subrow,
        3,
    )


@triton.jit
def _thin921_slow(A, AS, B, BS, FA, FB, Output, META: tl.constexpr):
    A = A.to(tl.pointer_type(tl.int8))
    B = B.to(tl.pointer_type(tl.int8))
    _slow(A, AS, B, BS, FA, FB, Output, META)


@triton.jit
def _thin921_slow_coarse(A, AS, B, BS, FA, FB, Output, META: tl.constexpr):
    A = A.to(tl.pointer_type(tl.int8))
    B = B.to(tl.pointer_type(tl.int8))
    _slow_coarse(A, AS, B, BS, FA, FB, Output, META)


@triton.jit
def _thin921_dequant_both(A, AS, B, BS, AD, BD, FA, FB, META: tl.constexpr):
    A = A.to(tl.pointer_type(tl.int8))
    B = B.to(tl.pointer_type(tl.int8))
    _dequant_both(A, AS, B, BS, AD, BD, FA, FB, META)


@triton.jit
def _thin921_slow_merge419(
    A, AS, B, BS, FA, FB, Output, PARTIAL, META: tl.constexpr
):
    A = A.to(tl.pointer_type(tl.int8))
    B = B.to(tl.pointer_type(tl.int8))
    _slow_merge419(A, AS, B, BS, FA, FB, Output, PARTIAL, META)


__all__ = ["mxfp8_block_scaled_matmul"]
