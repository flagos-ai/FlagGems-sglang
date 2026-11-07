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
def _dequant_rows(
    A,
    S,
    D,
    F,
    R: tl.constexpr,
    K: tl.constexpr,
    A0: tl.constexpr,
    A1: tl.constexpr,
    S0: tl.constexpr,
    S1: tl.constexpr,
    B: tl.constexpr,
    F_OFFSET: tl.constexpr = 0,
):
    ADDRESS64: tl.constexpr = (
        (R + 1) * A0 + (2 * K + B + 1024) * A1 >= 2147483648
        or (R + 1) * S0 + ((2 * K + B + 1024) // 32 + 1) * S1 >= 2147483648
        or (R + 1) * K + 2 * K + B + 1024 >= 2147483648
    )
    F = F + tl.cast(F_OFFSET, tl.int64)
    row = _wide_index926(tl.program_id(0), ADDRESS64)
    safe = tl.full((), 1, tl.int32)
    for chunk in range(tl.cdiv(K, 256)):
        cols = (
            tl.cast(chunk, tl.int64) if ADDRESS64 else chunk
        ) * 256 + _wide_index926(tl.arange(0, 256), ADDRESS64)
        u = tl.load(A + row * A0 + cols * A1, cols < K, 0)
        s = (
            tl.load(S + row * S0 + cols // 32 * S1, cols < K, 127).to(tl.int32)
            & 255
        )
        value = _decode(u, s)
        tl.store(D + row * K + cols, value, cols < K)
        safe = safe & tl.min(((s >= 10) & (s <= 246)).to(tl.int32), 0)
    tl.store(F + row, safe)


def _dense_entry(a, a_scale, b, b_scale, output_dtype):
    m, k = a.shape
    n = b.shape[0]
    out = torch.empty((m, n), device=a.device, dtype=output_dtype)
    if m == 0 or n == 0:
        return out
    ad = torch.empty((m, k), device=a.device, dtype=torch.bfloat16)
    bd = torch.empty((n, k), device=a.device, dtype=torch.bfloat16)
    fa = torch.empty((m + n,), device=a.device, dtype=torch.int32)
    fb = fa
    a_bytes759 = a.view(torch.int8)
    a_scale_bytes759 = a_scale
    b_bytes759 = b.view(torch.int8)
    b_scale_bytes759 = b_scale
    a_s0_760 = a.stride(0)
    a_s1_760 = a.stride(1)
    a_scale_s0_760 = a_scale.stride(0)
    a_scale_s1_760 = a_scale.stride(1)
    b_s0_760 = b.stride(0)
    b_s1_760 = b.stride(1)
    b_scale_s0_760 = b_scale.stride(0)
    b_scale_s1_760 = b_scale.stride(1)
    decode = _dequant_rows
    if k > 8192:
        decode.run(
            a_bytes759,
            a_scale_bytes759,
            ad,
            fa,
            m,
            k,
            a_s0_760,
            a_s1_760,
            a_scale_s0_760,
            a_scale_s1_760,
            max(32, triton.next_power_of_2(k)),
            num_warps=4,
            grid=(m,),
            warmup=False,
        )
        decode.run(
            b_bytes759,
            b_scale_bytes759,
            bd,
            fb,
            n,
            k,
            b_s0_760,
            b_s1_760,
            b_scale_s0_760,
            b_scale_s1_760,
            max(32, triton.next_power_of_2(k)),
            num_warps=4,
            F_OFFSET=m,
            grid=(n,),
            warmup=False,
        )
    else:
        _dequant_both.run(
            a_bytes759,
            a_scale_bytes759,
            b_bytes759,
            b_scale_bytes759,
            ad,
            bd,
            fa,
            m,
            k,
            a_s0_760,
            a_s1_760,
            a_scale_s0_760,
            a_scale_s1_760,
            b_s0_760,
            b_s1_760,
            b_scale_s0_760,
            b_scale_s1_760,
            max(32, triton.next_power_of_2(k)),
            num_warps=4,
            N=n,
            grid=(m + n,),
            warmup=False,
        )
    bm, bn, bk = (
        (128, 256 if m >= 2048 else 128, 64) if m >= 512 else (64, 64, 64)
    )
    if m < 512 and 512 <= k <= 1024:
        small_sms = torch.cuda.get_device_properties(
            a.device
        ).multi_processor_count
        if triton.cdiv(m, 32) * triton.cdiv(n, 64) <= small_sms:
            bm, bn, bk = (32, 64, 64)
    if 512 <= m < 2048:
        bm, bn, bk = (64, 128, 32)
    medium_persistent = False
    if 512 <= m < 2048 and n >= 2048 and (k >= 2048):
        sms = torch.cuda.get_device_properties(a.device).multi_processor_count
        medium_persistent = triton.cdiv(m, 128) * triton.cdiv(n, 256) >= sms
    two_cta = m >= 4096 and n >= 6144 and (k >= 2048)
    if two_cta:
        bm, bn, bk = (128, 128, 32)
    warps = 4 if two_cta else 8 if bm == 128 else 4
    if medium_persistent:
        two_cta = n >= 6144
        bm, bn, bk, warps = (128, 128, 32, 4) if two_cta else (128, 256, 64, 8)
    stages = 3
    group = 8
    if m < 2048 and (not medium_persistent):
        if not (64 <= m <= 128 and 2048 <= n <= 4096 and (k >= 2048)):
            _fast_guarded.run(
                ad,
                bd,
                fa,
                out,
                m,
                n,
                k,
                bm,
                bn,
                bk,
                group,
                a_bytes759,
                a_scale_bytes759,
                b_bytes759,
                b_scale_bytes759,
                a_s0_760,
                a_s1_760,
                a_scale_s0_760,
                a_scale_s1_760,
                b_s0_760,
                b_s1_760,
                b_scale_s0_760,
                b_scale_s1_760,
                num_warps=warps,
                num_stages=stages,
                grid=(triton.cdiv(m, bm) * triton.cdiv(n, bn),),
                warmup=False,
            )
        else:
            bm, bn, bk = (64, 128, 64)
            warps = 4
            split = 4
            partial = torch.empty(
                (split, m, n), device=a.device, dtype=torch.float32
            )
            _guarded_small_split.run(
                ad,
                bd,
                fa,
                partial,
                m,
                n,
                k,
                bm,
                bn,
                bk,
                group,
                a_bytes759,
                a_scale_bytes759,
                b_bytes759,
                b_scale_bytes759,
                a_s0_760,
                a_s1_760,
                a_scale_s0_760,
                a_scale_s1_760,
                b_s0_760,
                b_s1_760,
                b_scale_s0_760,
                b_scale_s1_760,
                split,
                num_warps=warps,
                num_stages=stages,
                grid=(triton.cdiv(m, bm) * triton.cdiv(n, bn), split),
                warmup=False,
            )
            _reduce_small_split.run(
                partial,
                out,
                m * n,
                split,
                1024,
                num_warps=4,
                grid=(triton.cdiv(m * n, 1024),),
                warmup=False,
            )
    else:
        if not medium_persistent:
            sms = torch.cuda.get_device_properties(
                a.device
            ).multi_processor_count
        grid = min(
            sms * (2 if two_cta else 1),
            triton.cdiv(m, bm) * triton.cdiv(n, bn),
        )
        tiles = triton.cdiv(m, bm) * triton.cdiv(n, bn)
        tail = tiles % grid
        split = (
            (
                min(16, grid // tail, max(1, triton.cdiv(k, bk) // 8))
                if medium_persistent
                else min(4, grid // tail)
            )
            if tail
            else 1
        )
        if k < 2048 or split <= 1:
            _fast_persistent.run(
                ad,
                bd,
                out,
                m,
                n,
                k,
                bm,
                bn,
                bk,
                group,
                grid,
                False,
                num_warps=warps,
                num_stages=stages,
                grid=(grid,),
                warmup=False,
            )
        else:
            partials = torch.empty(
                (split * tail * bm * bn,), device=a.device, dtype=torch.float32
            )
            _fast_tailwave_static.run(
                ad,
                bd,
                partials,
                out,
                m,
                n,
                k,
                bm,
                bn,
                bk,
                group,
                grid,
                split,
                num_warps=warps,
                num_stages=stages,
                grid=(grid,),
                warmup=False,
            )
            _reduce_tailwave.run(
                partials,
                out,
                m,
                n,
                bm,
                bn,
                group,
                grid,
                split,
                1024,
                num_warps=4,
                grid=(tail * triton.cdiv(bm * bn, 1024),),
                warmup=False,
            )
    bm, bn, bk = (32, 64, 32)
    if m >= 2048 or medium_persistent:
        _slow_coarse.run(
            a_bytes759,
            a_scale_bytes759,
            b_bytes759,
            b_scale_bytes759,
            fa,
            out,
            m,
            n,
            k,
            a_s0_760,
            a_s1_760,
            a_scale_s0_760,
            a_scale_s1_760,
            b_s0_760,
            b_s1_760,
            b_scale_s0_760,
            b_scale_s1_760,
            bm,
            bn,
            bk,
            num_warps=4,
            num_stages=2,
            grid=(triton.cdiv(m, bm * 4) * triton.cdiv(n, bn * 4),),
            warmup=False,
        )
    return out


def mxfp8_block_scaled_matmul(a, a_scale, b, b_scale, output_dtype):
    if torch.version.hip is not None:
        return _hip_scalar876(a, a_scale, b, b_scale, output_dtype)
    m, k = a.shape
    n = b.shape[0]
    if k >= 1024 or (k >= 512 and m * n >= 131072) or m * n > 262144:
        return _dense_entry(a, a_scale, b, b_scale, output_dtype)
    out = torch.empty((m, n), device=a.device, dtype=output_dtype)
    if m == 0 or n == 0:
        return out
    bm, bn, bk = (16, 32, 128)
    kernel = _fused_native
    a_ptr925 = a.data_ptr()
    as_ptr925 = a_scale.data_ptr()
    b_ptr925 = b.data_ptr()
    bs_ptr925 = b_scale.data_ptr()
    out_ptr925 = out.data_ptr()
    output_t925 = (
        tl.float16
        if output_dtype == torch.float16
        else (
            tl.bfloat16
            if output_dtype == torch.bfloat16
            else (
                tl.float32
                if output_dtype == torch.float32
                else output_type876(output_dtype)
            )
        )
    )
    kernel.run(
        a_ptr925,
        as_ptr925,
        b_ptr925,
        bs_ptr925,
        out_ptr925,
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
        max(1, triton.next_power_of_2(triton.cdiv(k, 32))),
        (
            a_ptr925 % 16 == 0,
            as_ptr925 % 16 == 0,
            b_ptr925 % 16 == 0,
            bs_ptr925 % 16 == 0,
            out_ptr925 % 16 == 0,
        ),
        output_t925,
        num_warps=4,
        num_stages=2,
        grid=(triton.cdiv(m, bm) * triton.cdiv(n, bn),),
        warmup=False,
    )
    return out


@triton.jit
def _decode_native(u, s):
    u = u.to(tl.uint8)
    s = s.to(tl.int32) & 255
    v = (
        u.to(tl.float8e4b15, bitcast=True).to(tl.float16).to(tl.float32)
        * 256.0
    )
    v = tl.where(u & 127 == 127, float("nan"), v)
    scale = tl.where(s == 0, 4194304, s << 23).to(tl.float32, bitcast=True)
    return v * scale


@triton.jit(do_not_specialize=["A", "AS", "B", "BS", "Output"])
def _fused_native(
    A: tl.uint64,
    AS: tl.uint64,
    B: tl.uint64,
    BS: tl.uint64,
    Output: tl.uint64,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    A0: tl.constexpr,
    A1: tl.constexpr,
    AS0: tl.constexpr,
    AS1: tl.constexpr,
    B0: tl.constexpr,
    B1: tl.constexpr,
    BS0: tl.constexpr,
    BS1: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    BG: tl.constexpr,
    ALIGN925: tl.constexpr,
    OUTPUT925: tl.constexpr,
):
    A = cast_ptr878(A, tl.int8, ALIGN925[0])
    AS = cast_ptr878(AS, tl.uint8, ALIGN925[1])
    B = cast_ptr878(B, tl.int8, ALIGN925[2])
    BS = cast_ptr878(BS, tl.uint8, ALIGN925[3])
    Output = cast_ptr878(Output, OUTPUT925, ALIGN925[4])
    ADDRESS64: tl.constexpr = (
        (M + 4 * BM) * N + (N + 4 * BN) >= 2147483648
        or (M + 4 * BM) * K + (K + 4 * BK) >= 2147483648
        or (N + 4 * BN) * K + (K + 4 * BK) >= 2147483648
        or ((M + 4 * BM) * A0 + (K + 4 * BK) * A1 >= 2147483648)
        or ((M + 4 * BM) * AS0 + ((K + 4 * BK) // 32 + 1) * AS1 >= 2147483648)
        or ((N + 4 * BN) * B0 + (K + 4 * BK) * B1 >= 2147483648)
        or ((N + 4 * BN) * BS0 + ((K + 4 * BK) // 32 + 1) * BS1 >= 2147483648)
    )
    pid = _wide_index926(tl.program_id(0), ADDRESS64)
    rm = pid // tl.cdiv(N, BN) * BM + _wide_index926(
        tl.arange(0, BM), ADDRESS64
    )
    rn = pid % tl.cdiv(N, BN) * BN + _wide_index926(
        tl.arange(0, BN), ADDRESS64
    )
    rk = _wide_index926(tl.arange(0, BK), ADDRESS64)
    acc = tl.zeros((BM, BN), tl.float32)
    groups = _wide_index926(tl.arange(0, BG), ADDRESS64)
    asafe = (
        tl.load(
            AS + rm[:, None] * AS0 + groups[None, :] * AS1,
            (rm[:, None] < M) & (groups[None, :] < tl.cdiv(K, 32)),
            127,
        ).to(tl.int32)
        & 255
    )
    bsafe = (
        tl.load(
            BS + rn[:, None] * BS0 + groups[None, :] * BS1,
            (rn[:, None] < N) & (groups[None, :] < tl.cdiv(K, 32)),
            127,
        ).to(tl.int32)
        & 255
    )
    safe_a = tl.min(
        tl.min(((asafe >= 10) & (asafe <= 246)).to(tl.int32), 0), 0
    )
    safe_b = tl.min(
        tl.min(((bsafe >= 10) & (bsafe <= 246)).to(tl.int32), 0), 0
    )
    if safe_a & safe_b:
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
            av = _decode_native(ua, sa)
            bv = _decode_native(ub, sb)
            acc = tl.dot(av.to(tl.bfloat16), bv.to(tl.bfloat16), acc)
    else:
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
def _fast_persistent(
    A,
    B,
    Output,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    GROUP: tl.constexpr,
    GRID: tl.constexpr,
    FLATTEN: tl.constexpr,
):
    ADDRESS64: tl.constexpr = (
        (M + 4 * BM) * N + (N + 4 * BN) >= 2147483648
        or (M + 4 * BM) * K + (K + 4 * BK) >= 2147483648
        or (N + 4 * BN) * K + (K + 4 * BK) >= 2147483648
        or ((GRID + 1) * BM * BN >= 2147483648)
    )
    gm = tl.cdiv(M, BM)
    gn = tl.cdiv(N, BN)
    rk = _wide_index926(tl.arange(0, BK), ADDRESS64)
    for pid in tl.range(
        _wide_index926(tl.program_id(0), ADDRESS64),
        gm * gn,
        GRID,
        flatten=FLATTEN,
    ):
        group = pid // (GROUP * gn)
        first_m = group * GROUP
        group_size = tl.minimum(gm - first_m, GROUP)
        pm = first_m + pid % (GROUP * gn) % group_size
        pn = pid % (GROUP * gn) // group_size
        rm = pm * BM + _wide_index926(tl.arange(0, BM), ADDRESS64)
        rn = pn * BN + _wide_index926(tl.arange(0, BN), ADDRESS64)
        acc = tl.zeros((BM, BN), tl.float32)
        for ki in range(tl.cdiv(K, BK)):
            kk = (tl.cast(ki, tl.int64) if ADDRESS64 else ki) * BK + rk
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


@triton.jit
def _dequant_both(
    A,
    AS,
    B,
    BS,
    AD,
    BD,
    FA,
    M: tl.constexpr,
    K: tl.constexpr,
    A0: tl.constexpr,
    A1: tl.constexpr,
    AS0: tl.constexpr,
    AS1: tl.constexpr,
    B0: tl.constexpr,
    B1: tl.constexpr,
    BS0: tl.constexpr,
    BS1: tl.constexpr,
    BLOCK: tl.constexpr,
    N: tl.constexpr,
):
    ADDRESS64: tl.constexpr = (
        (M + 1024) * N + (N + 1024) >= 2147483648
        or (M + 1024) * K + (2 * K + 4096) >= 2147483648
        or (N + 1024) * K + (2 * K + 4096) >= 2147483648
        or ((M + 1024) * A0 + (2 * K + 4096) * A1 >= 2147483648)
        or ((M + 1024) * AS0 + ((2 * K + 4096) // 32 + 1) * AS1 >= 2147483648)
        or ((N + 1024) * B0 + (2 * K + 4096) * B1 >= 2147483648)
        or ((N + 1024) * BS0 + ((2 * K + 4096) // 32 + 1) * BS1 >= 2147483648)
    )
    row = _wide_index926(tl.program_id(0), ADDRESS64)
    is_a = row < M
    cols = _wide_index926(tl.arange(0, BLOCK), ADDRESS64)
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
    value = _decode(u, s)
    tl.store(dst + cols, value, cols < K)
    safe = tl.min(((s >= 10) & (s <= 246)).to(tl.int32), 0)
    tl.store(flag, safe)


@triton.jit
def _fast_guarded(
    A,
    B,
    FA,
    Output,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    GROUP: tl.constexpr,
    OA,
    AS,
    OB,
    BS,
    A0: tl.constexpr,
    A1: tl.constexpr,
    AS0: tl.constexpr,
    AS1: tl.constexpr,
    B0: tl.constexpr,
    B1: tl.constexpr,
    BS0: tl.constexpr,
    BS1: tl.constexpr,
):
    ADDRESS64: tl.constexpr = (
        (M + 4 * BM) * N + (N + 4 * BN) >= 2147483648
        or (M + 4 * BM) * K + (K + 4 * BK) >= 2147483648
        or (N + 4 * BN) * K + (K + 4 * BK) >= 2147483648
        or ((M + 4 * BM) * A0 + (K + 4 * BK) * A1 >= 2147483648)
        or ((M + 4 * BM) * AS0 + ((K + 4 * BK) // 32 + 1) * AS1 >= 2147483648)
        or ((N + 4 * BN) * B0 + (K + 4 * BK) * B1 >= 2147483648)
        or ((N + 4 * BN) * BS0 + ((K + 4 * BK) // 32 + 1) * BS1 >= 2147483648)
    )
    FB = FA + tl.cast(M, tl.int64)
    pid = _wide_index926(tl.program_id(0), ADDRESS64)
    gm = tl.cdiv(M, BM)
    gn = tl.cdiv(N, BN)
    group = pid // (GROUP * gn)
    first_m = group * GROUP
    group_size = tl.minimum(gm - first_m, GROUP)
    pm = first_m + pid % (GROUP * gn) % group_size
    pn = pid % (GROUP * gn) // group_size
    rm = pm * BM + _wide_index926(tl.arange(0, BM), ADDRESS64)
    rn = pn * BN + _wide_index926(tl.arange(0, BN), ADDRESS64)
    rk = _wide_index926(tl.arange(0, BK), ADDRESS64)
    acc = tl.zeros((BM, BN), tl.float32)
    fa = tl.min(tl.load(FA + rm, rm < M, 1), 0)
    fb = tl.min(tl.load(FB + rn, rn < N, 1), 0)
    if fa & fb:
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
    else:
        srk = _wide_index926(tl.arange(0, 32), ADDRESS64)
        for sk in range(tl.cdiv(K, 32)):
            skk = (tl.cast(sk, tl.int64) if ADDRESS64 else sk) * 32 + srk
            sa = (
                tl.load(
                    AS + rm[:, None] * AS0 + skk[None, :] // 32 * AS1,
                    (rm[:, None] < M) & (skk[None, :] < K),
                    127,
                ).to(tl.int32)
                & 255
            )
            sb = (
                tl.load(
                    BS + rn[None, :] * BS0 + skk[:, None] // 32 * BS1,
                    (rn[None, :] < N) & (skk[:, None] < K),
                    127,
                ).to(tl.int32)
                & 255
            )
            ua = tl.load(
                OA + rm[:, None] * A0 + skk[None, :] * A1,
                (rm[:, None] < M) & (skk[None, :] < K),
                0,
            )
            ub = tl.load(
                OB + rn[None, :] * B0 + skk[:, None] * B1,
                (rn[None, :] < N) & (skk[:, None] < K),
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
def _tailwave_dot(
    A,
    B,
    PM,
    PN,
    START,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    STEPS: tl.constexpr,
):
    ADDRESS64: tl.constexpr = (
        (M + 4 * BM) * N + (N + 4 * BN) >= 2147483648
        or (M + 4 * BM) * K + (K + 4 * BK) >= 2147483648
        or (N + 4 * BN) * K + (K + 4 * BK) >= 2147483648
    )
    PM = tl.cast(PM, tl.int64) if ADDRESS64 else PM
    PN = tl.cast(PN, tl.int64) if ADDRESS64 else PN
    START = tl.cast(START, tl.int64) if ADDRESS64 else START
    rm = PM * BM + _wide_index926(tl.arange(0, BM), ADDRESS64)
    rn = PN * BN + _wide_index926(tl.arange(0, BN), ADDRESS64)
    rk = _wide_index926(tl.arange(0, BK), ADDRESS64)
    acc = tl.zeros((BM, BN), tl.float32)
    for ki in range(STEPS):
        kk = (START + ki) * BK + rk
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
    return acc


@triton.jit
def _tailwave_coords(
    pid, GM: tl.constexpr, GN: tl.constexpr, GROUP: tl.constexpr
):
    group = pid // (GROUP * GN)
    first_m = group * GROUP
    group_size = tl.minimum(GM - first_m, GROUP)
    pm = first_m + pid % (GROUP * GN) % group_size
    pn = pid % (GROUP * GN) // group_size
    return (pm, pn)


@triton.jit
def _fast_tailwave_static(
    A,
    B,
    P,
    Output,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    GROUP: tl.constexpr,
    GRID: tl.constexpr,
    SPLIT: tl.constexpr,
):
    ADDRESS64: tl.constexpr = (
        (M + 4 * BM) * N + (N + 4 * BN) >= 2147483648
        or (M + 4 * BM) * K + (K + 4 * BK) >= 2147483648
        or (N + 4 * BN) * K + (K + 4 * BK) >= 2147483648
        or (
            (SPLIT + 1) * M * N + (M + 4 * BM) * N + (N + 4 * BN) >= 2147483648
        )
        or ((GRID + 1) * BM * BN >= 2147483648)
    )
    gm: tl.constexpr = (M + BM - 1) // BM
    gn: tl.constexpr = (N + BN - 1) // BN
    tiles: tl.constexpr = gm * gn
    full: tl.constexpr = tiles // GRID * GRID
    tail: tl.constexpr = tiles - full
    ktiles: tl.constexpr = (K + BK - 1) // BK
    segment: tl.constexpr = (ktiles + SPLIT - 1) // SPLIT
    worker = _wide_index926(tl.program_id(0), ADDRESS64)
    for pid in range(worker, full, GRID):
        pm, pn = _tailwave_coords(pid, gm, gn, GROUP)
        acc = _tailwave_dot(A, B, pm, pn, 0, M, N, K, BM, BN, BK, ktiles)
        rm = pm * BM + _wide_index926(tl.arange(0, BM), ADDRESS64)
        rn = pn * BN + _wide_index926(tl.arange(0, BN), ADDRESS64)
        tl.store(
            Output + rm[:, None] * N + rn[None, :],
            acc,
            (rm[:, None] < M) & (rn[None, :] < N),
        )
    if worker < tail * SPLIT:
        tid = worker % tail
        part = worker // tail
        pm, pn = _tailwave_coords(full + tid, gm, gn, GROUP)
        acc = _tailwave_dot(
            A, B, pm, pn, part * segment, M, N, K, BM, BN, BK, segment
        )
        offsets = (
            worker * BM * BN
            + _wide_index926(tl.arange(0, BM), ADDRESS64)[:, None] * BN
            + _wide_index926(tl.arange(0, BN), ADDRESS64)[None, :]
        )
        tl.store(P + offsets, acc)


@triton.jit
def _reduce_tailwave(
    P,
    Output,
    M: tl.constexpr,
    N: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    GROUP: tl.constexpr,
    GRID: tl.constexpr,
    SPLIT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    ADDRESS64: tl.constexpr = (
        (M + 4 * BM) * N + (N + 4 * BN) >= 2147483648
        or (SPLIT + 1) * M * N + (M + 4 * BM) * N + (N + 4 * BN) >= 2147483648
        or (GRID + 1) * BM * BN >= 2147483648
    )
    gm = tl.cdiv(M, BM)
    gn = tl.cdiv(N, BN)
    tiles = gm * gn
    full = tiles // GRID * GRID
    tail = tiles - full
    chunks = tl.cdiv(BM * BN, BLOCK)
    tid = _wide_index926(tl.program_id(0), ADDRESS64) // chunks
    offsets = _wide_index926(
        tl.program_id(0), ADDRESS64
    ) % chunks * BLOCK + _wide_index926(tl.arange(0, BLOCK), ADDRESS64)
    acc = tl.full((BLOCK,), 0, tl.float32)
    for part in range(SPLIT):
        acc += tl.load(
            P
            + ((tl.cast(part, tl.int64) if ADDRESS64 else part) * tail + tid)
            * BM
            * BN
            + offsets,
            offsets < BM * BN,
            0,
        )
    pid = full + tid
    group = pid // (GROUP * gn)
    first_m = group * GROUP
    group_size = tl.minimum(gm - first_m, GROUP)
    pm = first_m + pid % (GROUP * gn) % group_size
    pn = pid % (GROUP * gn) // group_size
    rows = pm * BM + offsets // BN
    cols = pn * BN + offsets % BN
    tl.store(
        Output + rows * N + cols,
        acc,
        (offsets < BM * BN) & (rows < M) & (cols < N),
    )


@triton.jit
def _slow_coarse(
    A,
    AS,
    B,
    BS,
    FA,
    Output,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    A0: tl.constexpr,
    A1: tl.constexpr,
    AS0: tl.constexpr,
    AS1: tl.constexpr,
    B0: tl.constexpr,
    B1: tl.constexpr,
    BS0: tl.constexpr,
    BS1: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    ADDRESS64: tl.constexpr = (
        (M + 4 * BM) * N + (N + 4 * BN) >= 2147483648
        or (M + 4 * BM) * K + (K + 4 * BK) >= 2147483648
        or (N + 4 * BN) * K + (K + 4 * BK) >= 2147483648
        or ((M + 4 * BM) * A0 + (K + 4 * BK) * A1 >= 2147483648)
        or ((M + 4 * BM) * AS0 + ((K + 4 * BK) // 32 + 1) * AS1 >= 2147483648)
        or ((N + 4 * BN) * B0 + (K + 4 * BK) * B1 >= 2147483648)
        or ((N + 4 * BN) * BS0 + ((K + 4 * BK) // 32 + 1) * BS1 >= 2147483648)
    )
    FB = FA + tl.cast(M, tl.int64)
    pid = _wide_index926(tl.program_id(0), ADDRESS64)
    pm = pid // tl.cdiv(N, BN * 4)
    pn = pid % tl.cdiv(N, BN * 4)
    crm = pm * BM * 4 + _wide_index926(tl.arange(0, BM * 4), ADDRESS64)
    crn = pn * BN * 4 + _wide_index926(tl.arange(0, BN * 4), ADDRESS64)
    safe_a = tl.min(tl.load(FA + crm, crm < M, 1), 0)
    safe_b = tl.min(tl.load(FB + crn, crn < N, 1), 0)
    if safe_a & safe_b == 0:
        for sub in range(16):
            rm = (pm * 4 + sub // 4) * BM + _wide_index926(
                tl.arange(0, BM), ADDRESS64
            )
            rn = (pn * 4 + sub % 4) * BN + _wide_index926(
                tl.arange(0, BN), ADDRESS64
            )
            fa = tl.min(tl.load(FA + rm, rm < M, 1), 0)
            fb = tl.min(tl.load(FB + rn, rn < N, 1), 0)
            if fa & fb == 0:
                rk = _wide_index926(tl.arange(0, BK), ADDRESS64)
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
def _guarded_small_split(
    A,
    B,
    FA,
    Output,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    GROUP: tl.constexpr,
    OA,
    AS,
    OB,
    BS,
    A0: tl.constexpr,
    A1: tl.constexpr,
    AS0: tl.constexpr,
    AS1: tl.constexpr,
    B0: tl.constexpr,
    B1: tl.constexpr,
    BS0: tl.constexpr,
    BS1: tl.constexpr,
    SPLIT: tl.constexpr,
):
    ADDRESS64: tl.constexpr = (
        (M + 4 * BM) * N + (N + 4 * BN) >= 2147483648
        or (M + 4 * BM) * K + (K + 4 * BK) >= 2147483648
        or (N + 4 * BN) * K + (K + 4 * BK) >= 2147483648
        or ((M + 4 * BM) * A0 + (K + 4 * BK) * A1 >= 2147483648)
        or ((M + 4 * BM) * AS0 + ((K + 4 * BK) // 32 + 1) * AS1 >= 2147483648)
        or ((N + 4 * BN) * B0 + (K + 4 * BK) * B1 >= 2147483648)
        or ((N + 4 * BN) * BS0 + ((K + 4 * BK) // 32 + 1) * BS1 >= 2147483648)
        or (
            (SPLIT + 1) * M * N + (M + 4 * BM) * N + (N + 4 * BN) >= 2147483648
        )
    )
    FB = FA + tl.cast(M, tl.int64)
    pid = _wide_index926(tl.program_id(0), ADDRESS64)
    part = _wide_index926(tl.program_id(1), ADDRESS64)
    steps: tl.constexpr = (K + BK * SPLIT - 1) // (BK * SPLIT)
    segment: tl.constexpr = (K + BK * SPLIT - 1) // (BK * SPLIT) * BK
    gm = tl.cdiv(M, BM)
    gn = tl.cdiv(N, BN)
    group = pid // (GROUP * gn)
    first_m = group * GROUP
    group_size = tl.minimum(gm - first_m, GROUP)
    pm = first_m + pid % (GROUP * gn) % group_size
    pn = pid % (GROUP * gn) // group_size
    rm = pm * BM + _wide_index926(tl.arange(0, BM), ADDRESS64)
    rn = pn * BN + _wide_index926(tl.arange(0, BN), ADDRESS64)
    rk = _wide_index926(tl.arange(0, BK), ADDRESS64)
    acc = tl.zeros((BM, BN), tl.float32)
    fa = tl.min(tl.load(FA + rm, rm < M, 1), 0)
    fb = tl.min(tl.load(FB + rn, rn < N, 1), 0)
    if fa & fb:
        for k in range(steps):
            kk = (k + part * steps) * BK + rk
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
    else:
        srk = _wide_index926(tl.arange(0, 32), ADDRESS64)
        for sk in range(segment // 32):
            skk = (
                part * segment
                + (tl.cast(sk, tl.int64) if ADDRESS64 else sk) * 32
                + srk
            )
            sa = (
                tl.load(
                    AS + rm[:, None] * AS0 + skk[None, :] // 32 * AS1,
                    (rm[:, None] < M) & (skk[None, :] < K),
                    127,
                ).to(tl.int32)
                & 255
            )
            sb = (
                tl.load(
                    BS + rn[None, :] * BS0 + skk[:, None] // 32 * BS1,
                    (rn[None, :] < N) & (skk[:, None] < K),
                    127,
                ).to(tl.int32)
                & 255
            )
            ua = tl.load(
                OA + rm[:, None] * A0 + skk[None, :] * A1,
                (rm[:, None] < M) & (skk[None, :] < K),
                0,
            )
            ub = tl.load(
                OB + rn[None, :] * B0 + skk[:, None] * B1,
                (rn[None, :] < N) & (skk[:, None] < K),
                0,
            )
            av = _decode(ua, sa)
            bv = _decode(ub, sb)
            acc = tl.dot(av, bv, acc, input_precision="ieee")
    tl.store(
        Output + part * M * N + rm[:, None] * N + rn[None, :],
        acc,
        (rm[:, None] < M) & (rn[None, :] < N),
    )


@triton.jit
def _reduce_small_split(
    P, Output, SIZE: tl.constexpr, SPLIT: tl.constexpr, BLOCK: tl.constexpr
):
    ADDRESS64: tl.constexpr = (SPLIT + 1) * SIZE + BLOCK >= 2147483648
    offsets = _wide_index926(
        tl.program_id(0), ADDRESS64
    ) * BLOCK + _wide_index926(tl.arange(0, BLOCK), ADDRESS64)
    value = tl.zeros((BLOCK,), tl.float32)
    for part in range(SPLIT):
        value += tl.load(
            P
            + (tl.cast(part, tl.int64) if ADDRESS64 else part) * SIZE
            + offsets,
            offsets < SIZE,
            0,
        )
    tl.store(Output + offsets, value, offsets < SIZE)


@triton.jit
def _decode_signbit(u, s):
    u = u.to(tl.int32) & 255
    s = s.to(tl.int32) & 255
    e = u >> 3 & 15
    m = u & 7
    normal = (e + 120 << 23 | m << 20).to(tl.float32, bitcast=True)
    magnitude = tl.where(e == 0, m.to(tl.float32) * 0.001953125, normal)
    bits = magnitude.to(tl.int32, bitcast=True) | (u & 128) << 24
    value = bits.to(tl.float32, bitcast=True)
    value = tl.where(u & 127 == 127, float("nan"), value)
    scale = tl.where(s == 0, 4194304, s << 23).to(tl.float32, bitcast=True)
    return value * scale


def output_type876(dtype):
    name = str(dtype).split(".")[-1]
    name = (
        name.replace("float8_e4m3fnuz", "fp8e4b8")
        .replace("float8_e4m3fn", "fp8e4nv")
        .replace("float8_e5m2fnuz", "fp8e5b16")
        .replace("float8_e5m2", "fp8e5")
    )
    return tl.dtype(
        name.replace("bfloat", "bf")
        .replace("float", "fp")
        .replace("bool", "int1")
    )


@triton.jit(do_not_specialize=["A", "B", "Output"])
def _fast_meta853_scalar876(
    A: tl.uint64,
    B: tl.uint64,
    Output: tl.uint64,
    META: tl.constexpr,
    ALIGN876: tl.constexpr,
    OUTPUT876: tl.constexpr,
):
    A = cast_ptr878(A, tl.bfloat16, ALIGN876[0])
    B = cast_ptr878(B, tl.bfloat16, ALIGN876[1])
    Output = cast_ptr878(
        Output,
        (
            tl.float16
            if OUTPUT876 == 0
            else (
                tl.bfloat16
                if OUTPUT876 == 1
                else tl.float32 if OUTPUT876 == 2 else OUTPUT876
            )
        ),
        ALIGN876[2],
    )
    M: tl.constexpr = META[0]
    N: tl.constexpr = META[1]
    K: tl.constexpr = META[2]
    BM: tl.constexpr = META[3]
    BN: tl.constexpr = META[4]
    BK: tl.constexpr = META[5]
    GROUP: tl.constexpr = META[6]
    LDA: tl.constexpr = META[7]
    LDB: tl.constexpr = META[8]
    ADDRESS64: tl.constexpr = (
        (M + 4 * BM) * N + (N + 4 * BN) >= 2147483648
        or (M + 4 * BM) * LDA + K + 4 * BK >= 2147483648
        or (N + 4 * BN) * LDB + K + 4 * BK >= 2147483648
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
        av = tl.load(
            A + rm[:, None] * LDA + kk[None, :],
            (rm[:, None] < M) & (kk[None, :] < K),
            0,
        )
        bv = tl.load(
            B + rn[None, :] * LDB + kk[:, None],
            (rn[None, :] < N) & (kk[:, None] < K),
            0,
        )
        acc = tl.dot(av, bv, acc)
    tl.store(
        Output + rm[:, None] * N + rn[None, :],
        acc,
        (rm[:, None] < M) & (rn[None, :] < N),
    )


@triton.jit(do_not_specialize=["A", "AS", "B", "BS", "AD", "BD", "FA"])
def signbit_dequant_both_meta853_scalar876(
    A: tl.uint64,
    AS: tl.uint64,
    B: tl.uint64,
    BS: tl.uint64,
    AD: tl.uint64,
    BD: tl.uint64,
    FA: tl.uint64,
    META: tl.constexpr,
    ALIGN876: tl.constexpr,
):
    A = cast_ptr878(A, tl.int8, ALIGN876[0])
    AS = cast_ptr878(AS, tl.int8, ALIGN876[1])
    B = cast_ptr878(B, tl.int8, ALIGN876[2])
    BS = cast_ptr878(BS, tl.int8, ALIGN876[3])
    AD = cast_ptr878(AD, tl.bfloat16, ALIGN876[4])
    BD = cast_ptr878(BD, tl.bfloat16, ALIGN876[5])
    FA = cast_ptr878(FA, tl.int32, ALIGN876[6])
    M: tl.constexpr = META[0]
    FB = FA + tl.cast(M, tl.int64)
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
    LDA: tl.constexpr = META[12]
    LDB: tl.constexpr = META[13]
    ADDRESS64: tl.constexpr = (
        (M + 1024) * N + (N + 1024) >= 2147483648
        or (M + 1024) * A0 + (2 * K + 4096) * A1 >= 2147483648
        or (M + 1024) * AS0 + ((2 * K + 4096) // 32 + 1) * AS1 >= 2147483648
        or ((N + 1024) * B0 + (2 * K + 4096) * B1 >= 2147483648)
        or ((N + 1024) * BS0 + ((2 * K + 4096) // 32 + 1) * BS1 >= 2147483648)
        or ((M + 1024) * LDA + 2 * K + 4096 >= 2147483648)
        or ((N + 1024) * LDB + 2 * K + 4096 >= 2147483648)
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
    dst = tl.where(is_a, AD + row * LDA, BD + (row - M) * LDB)
    flag = tl.where(is_a, FA + row, FB + row - M)
    stride = tl.where(is_a, A1, B1)
    scale_stride = tl.where(is_a, AS1, BS1)
    u = tl.load(src + cols * stride, cols < K, 0)
    s = (
        tl.load(scale + cols // 32 * scale_stride, cols < K, 127).to(tl.int32)
        & 255
    )
    value = _decode_signbit(u, s)
    tl.store(dst + cols, value, cols < K)
    safe = tl.min(((s >= 10) & (s <= 246)).to(tl.int32), 0)
    tl.store(flag, safe)


@triton.jit(do_not_specialize=["A", "S", "D", "F"])
def signbit_dequant_rows_meta853_scalar876(
    A: tl.uint64,
    S: tl.uint64,
    D: tl.uint64,
    F: tl.uint64,
    META: tl.constexpr,
    ALIGN876: tl.constexpr,
):
    A = cast_ptr878(A, tl.int8, ALIGN876[0])
    S = cast_ptr878(S, tl.int8, ALIGN876[1])
    D = cast_ptr878(D, tl.bfloat16, ALIGN876[2])
    F = cast_ptr878(F, tl.int32, ALIGN876[3])
    R: tl.constexpr = META[0]
    K: tl.constexpr = META[1]
    A0: tl.constexpr = META[2]
    A1: tl.constexpr = META[3]
    S0: tl.constexpr = META[4]
    S1: tl.constexpr = META[5]
    B: tl.constexpr = META[6]
    F = F + tl.cast(META[7], tl.int64)
    LD: tl.constexpr = META[8]
    ADDRESS64: tl.constexpr = (
        (R + 1) * A0 + (2 * K + B + 1024) * A1 >= 2147483648
        or (R + 1) * S0 + ((2 * K + B + 1024) // 32 + 1) * S1 >= 2147483648
        or (R + 1) * LD + 2 * K + B + 1024 >= 2147483648
    )
    row = (
        tl.cast(tl.program_id(0), tl.int64) if ADDRESS64 else tl.program_id(0)
    )
    safe = tl.full((), 1, tl.int32)
    for chunk in range(tl.cdiv(K, 256)):
        cols = (tl.cast(chunk, tl.int64) if ADDRESS64 else chunk) * 256 + (
            tl.cast(tl.arange(0, 256), tl.int64)
            if ADDRESS64
            else tl.arange(0, 256)
        )
        u = tl.load(A + row * A0 + cols * A1, cols < K, 0)
        s = (
            tl.load(S + row * S0 + cols // 32 * S1, cols < K, 127).to(tl.int32)
            & 255
        )
        value = _decode_signbit(u, s)
        tl.store(D + row * LD + cols, value, cols < K)
        safe = safe & tl.min(((s >= 10) & (s <= 246)).to(tl.int32), 0)
    tl.store(F + row, safe)


@triton.jit(do_not_specialize=["A", "AS", "B", "BS", "FA", "Output"])
def signbit_slow_meta853_scalar876(
    A: tl.uint64,
    AS: tl.uint64,
    B: tl.uint64,
    BS: tl.uint64,
    FA: tl.uint64,
    Output: tl.uint64,
    META: tl.constexpr,
    ALIGN876: tl.constexpr,
    OUTPUT876: tl.constexpr,
):
    A = cast_ptr878(A, tl.int8, ALIGN876[0])
    AS = cast_ptr878(AS, tl.int8, ALIGN876[1])
    B = cast_ptr878(B, tl.int8, ALIGN876[2])
    BS = cast_ptr878(BS, tl.int8, ALIGN876[3])
    FA = cast_ptr878(FA, tl.int32, ALIGN876[4])
    Output = cast_ptr878(
        Output,
        (
            tl.float16
            if OUTPUT876 == 0
            else (
                tl.bfloat16
                if OUTPUT876 == 1
                else tl.float32 if OUTPUT876 == 2 else OUTPUT876
            )
        ),
        ALIGN876[5],
    )
    M: tl.constexpr = META[0]
    FB = FA + tl.cast(M, tl.int64)
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
            av = _decode_signbit(ua, sa)
            bv = _decode_signbit(ub, sb)
            acc = tl.dot(av, bv, acc, input_precision="ieee")
        tl.store(
            Output + rm[:, None] * N + rn[None, :],
            acc,
            (rm[:, None] < M) & (rn[None, :] < N),
        )


@triton.jit(do_not_specialize=["A", "AS", "B", "BS", "Output"])
def hip_fused_signbit_meta853_scalar876(
    A: tl.uint64,
    AS: tl.uint64,
    B: tl.uint64,
    BS: tl.uint64,
    Output: tl.uint64,
    META: tl.constexpr,
    ALIGN876: tl.constexpr,
    OUTPUT876: tl.constexpr,
):
    A = cast_ptr878(A, tl.int8, ALIGN876[0])
    AS = cast_ptr878(AS, tl.int8, ALIGN876[1])
    B = cast_ptr878(B, tl.int8, ALIGN876[2])
    BS = cast_ptr878(BS, tl.int8, ALIGN876[3])
    Output = cast_ptr878(
        Output,
        (
            tl.float16
            if OUTPUT876 == 0
            else (
                tl.bfloat16
                if OUTPUT876 == 1
                else tl.float32 if OUTPUT876 == 2 else OUTPUT876
            )
        ),
        ALIGN876[4],
    )
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
    BG: tl.constexpr = META[14]
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
    rk = tl.cast(tl.arange(0, BK), tl.int64) if ADDRESS64 else tl.arange(0, BK)
    acc = tl.zeros((BM, BN), tl.float32)
    groups = (
        tl.cast(tl.arange(0, BG), tl.int64) if ADDRESS64 else tl.arange(0, BG)
    )
    asafe = (
        tl.load(
            AS + rm[:, None] * AS0 + groups[None, :] * AS1,
            (rm[:, None] < M) & (groups[None, :] < tl.cdiv(K, 32)),
            127,
        ).to(tl.int32)
        & 255
    )
    bsafe = (
        tl.load(
            BS + rn[:, None] * BS0 + groups[None, :] * BS1,
            (rn[:, None] < N) & (groups[None, :] < tl.cdiv(K, 32)),
            127,
        ).to(tl.int32)
        & 255
    )
    safe_a = tl.min(
        tl.min(((asafe >= 10) & (asafe <= 246)).to(tl.int32), 0), 0
    )
    safe_b = tl.min(
        tl.min(((bsafe >= 10) & (bsafe <= 246)).to(tl.int32), 0), 0
    )
    if safe_a & safe_b:
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
            av = _decode_signbit(ua, sa)
            bv = _decode_signbit(ub, sb)
            acc = tl.dot(av.to(tl.bfloat16), bv.to(tl.bfloat16), acc)
    else:
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
            av = _decode_signbit(ua, sa)
            bv = _decode_signbit(ub, sb)
            acc = tl.dot(av, bv, acc, input_precision="ieee")
    tl.store(
        Output + rm[:, None] * N + rn[None, :],
        acc,
        (rm[:, None] < M) & (rn[None, :] < N),
    )


@triton.jit(
    do_not_specialize=["A", "B", "FA", "Output", "OA", "AS", "OB", "BS"]
)
def hip_guarded845_meta853_scalar876(
    A: tl.uint64,
    B: tl.uint64,
    FA: tl.uint64,
    Output: tl.uint64,
    OA: tl.uint64,
    AS: tl.uint64,
    OB: tl.uint64,
    BS: tl.uint64,
    META: tl.constexpr,
    ALIGN876: tl.constexpr,
    OUTPUT876: tl.constexpr,
):
    A = cast_ptr878(A, tl.bfloat16, ALIGN876[0])
    B = cast_ptr878(B, tl.bfloat16, ALIGN876[1])
    FA = cast_ptr878(FA, tl.int32, ALIGN876[2])
    Output = cast_ptr878(
        Output,
        (
            tl.float16
            if OUTPUT876 == 0
            else (
                tl.bfloat16
                if OUTPUT876 == 1
                else tl.float32 if OUTPUT876 == 2 else OUTPUT876
            )
        ),
        ALIGN876[3],
    )
    OA = cast_ptr878(OA, tl.int8, ALIGN876[4])
    AS = cast_ptr878(AS, tl.int8, ALIGN876[5])
    OB = cast_ptr878(OB, tl.int8, ALIGN876[6])
    BS = cast_ptr878(BS, tl.int8, ALIGN876[7])
    M: tl.constexpr = META[0]
    FB = FA + tl.cast(M, tl.int64)
    N: tl.constexpr = META[1]
    K: tl.constexpr = META[2]
    BM: tl.constexpr = META[3]
    BN: tl.constexpr = META[4]
    BK: tl.constexpr = META[5]
    GROUP: tl.constexpr = META[6]
    A0: tl.constexpr = META[7]
    A1: tl.constexpr = META[8]
    AS0: tl.constexpr = META[9]
    AS1: tl.constexpr = META[10]
    B0: tl.constexpr = META[11]
    B1: tl.constexpr = META[12]
    BS0: tl.constexpr = META[13]
    BS1: tl.constexpr = META[14]
    LDA: tl.constexpr = META[15]
    LDB: tl.constexpr = META[16]
    ADDRESS64: tl.constexpr = (
        (M + 4 * BM) * N + (N + 4 * BN) >= 2147483648
        or (M + 4 * BM) * A0 + (K + 4 * BK) * A1 >= 2147483648
        or (M + 4 * BM) * AS0 + ((K + 4 * BK) // 32 + 1) * AS1 >= 2147483648
        or ((N + 4 * BN) * B0 + (K + 4 * BK) * B1 >= 2147483648)
        or ((N + 4 * BN) * BS0 + ((K + 4 * BK) // 32 + 1) * BS1 >= 2147483648)
        or ((M + 4 * BM) * LDA + K + 4 * BK >= 2147483648)
        or ((N + 4 * BN) * LDB + K + 4 * BK >= 2147483648)
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
    fa = tl.min(tl.load(FA + rm, rm < M, 1), 0)
    fb = tl.min(tl.load(FB + rn, rn < N, 1), 0)
    if fa & fb:
        for k in range(tl.cdiv(K, BK)):
            kk = (tl.cast(k, tl.int64) if ADDRESS64 else k) * BK + rk
            av = tl.load(
                A + rm[:, None] * LDA + kk[None, :],
                (rm[:, None] < M) & (kk[None, :] < K),
                0,
            )
            bv = tl.load(
                B + rn[None, :] * LDB + kk[:, None],
                (rn[None, :] < N) & (kk[:, None] < K),
                0,
            )
            acc = tl.dot(av, bv, acc)
    else:
        srk = (
            tl.cast(tl.arange(0, 32), tl.int64)
            if ADDRESS64
            else tl.arange(0, 32)
        )
        for sk in range(tl.cdiv(K, 32)):
            skk = (tl.cast(sk, tl.int64) if ADDRESS64 else sk) * 32 + srk
            sa = (
                tl.load(
                    AS + rm[:, None] * AS0 + skk[None, :] // 32 * AS1,
                    (rm[:, None] < M) & (skk[None, :] < K),
                    127,
                ).to(tl.int32)
                & 255
            )
            sb = (
                tl.load(
                    BS + rn[None, :] * BS0 + skk[:, None] // 32 * BS1,
                    (rn[None, :] < N) & (skk[:, None] < K),
                    127,
                ).to(tl.int32)
                & 255
            )
            ua = tl.load(
                OA + rm[:, None] * A0 + skk[None, :] * A1,
                (rm[:, None] < M) & (skk[None, :] < K),
                0,
            )
            ub = tl.load(
                OB + rn[None, :] * B0 + skk[:, None] * B1,
                (rn[None, :] < N) & (skk[:, None] < K),
                0,
            )
            av = _decode_signbit(ua, sa)
            bv = _decode_signbit(ub, sb)
            acc = tl.dot(av, bv, acc, input_precision="ieee")
    tl.store(
        Output + rm[:, None] * N + rn[None, :],
        acc,
        (rm[:, None] < M) & (rn[None, :] < N),
    )


def _hip_short_scalar876(a, a_scale, b, b_scale, output_dtype):
    m, k = a.shape
    n = b.shape[0]
    out = torch.empty((m, n), device=a.device, dtype=output_dtype)
    if not m or not n:
        return out
    a_ptr876 = a.data_ptr()
    as_ptr876 = a_scale.data_ptr()
    b_ptr876 = b.data_ptr()
    bs_ptr876 = b_scale.data_ptr()
    out_ptr876 = out.data_ptr()
    output_t876 = (
        0
        if output_dtype == torch.float16
        else (
            1
            if output_dtype == torch.bfloat16
            else (
                2
                if output_dtype == torch.float32
                else output_type876(output_dtype)
            )
        )
    )
    a_ptr876_aligned = a_ptr876 % 16 == 0
    as_ptr876_aligned = as_ptr876 % 16 == 0
    b_ptr876_aligned = b_ptr876 % 16 == 0
    bs_ptr876_aligned = bs_ptr876 % 16 == 0
    out_ptr876_aligned = out_ptr876 % 16 == 0
    bm, bn, bk = (16, 32, 128)
    hip_fused_signbit_meta853_scalar876.run(
        a_ptr876,
        as_ptr876,
        b_ptr876,
        bs_ptr876,
        out_ptr876,
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
            max(1, triton.next_power_of_2(triton.cdiv(k, 32))),
        ),
        (
            a_ptr876_aligned,
            as_ptr876_aligned,
            b_ptr876_aligned,
            bs_ptr876_aligned,
            out_ptr876_aligned,
        ),
        output_t876,
        num_warps=4,
        num_stages=2,
        grid=(triton.cdiv(m, bm) * triton.cdiv(n, bn),),
        warmup=False,
    )
    return out


def _hip_scalar876(a, a_scale, b, b_scale, output_dtype):
    if 128 < a.shape[1] <= 256 and a.shape[0] * b.shape[0] <= 262144:
        return _hip_short_scalar876(a, a_scale, b, b_scale, output_dtype)
    m, k = a.shape
    n = b.shape[0]
    out = torch.empty((m, n), device=a.device, dtype=output_dtype)
    if not m or not n:
        return out
    lda872, ldb872 = (k + 32, k + 32)
    ad = torch.empty((m, lda872), device=a.device, dtype=torch.bfloat16)
    bd = torch.empty((n, ldb872), device=a.device, dtype=torch.bfloat16)
    fa = torch.empty((m + n,), device=a.device, dtype=torch.int32)
    a_ptr876 = a.data_ptr()
    as_ptr876 = a_scale.data_ptr()
    b_ptr876 = b.data_ptr()
    bs_ptr876 = b_scale.data_ptr()
    out_ptr876 = out.data_ptr()
    output_t876 = (
        0
        if output_dtype == torch.float16
        else (
            1
            if output_dtype == torch.bfloat16
            else (
                2
                if output_dtype == torch.float32
                else output_type876(output_dtype)
            )
        )
    )
    ad_ptr876 = ad.data_ptr()
    bd_ptr876 = bd.data_ptr()
    f_ptr876 = fa.data_ptr()
    a_ptr876_aligned = a_ptr876 % 16 == 0
    as_ptr876_aligned = as_ptr876 % 16 == 0
    b_ptr876_aligned = b_ptr876 % 16 == 0
    bs_ptr876_aligned = bs_ptr876 % 16 == 0
    out_ptr876_aligned = out_ptr876 % 16 == 0
    ad_ptr876_aligned = ad_ptr876 % 16 == 0
    bd_ptr876_aligned = bd_ptr876 % 16 == 0
    f_ptr876_aligned = f_ptr876 % 16 == 0
    a_s0_861 = a.stride(0)
    a_s1_861 = a.stride(1)
    a_scale_s0_861 = a_scale.stride(0)
    a_scale_s1_861 = a_scale.stride(1)
    b_s0_861 = b.stride(0)
    b_s1_861 = b.stride(1)
    b_scale_s0_861 = b_scale.stride(0)
    b_scale_s1_861 = b_scale.stride(1)
    if k <= 8192:
        signbit_dequant_both_meta853_scalar876.run(
            a_ptr876,
            as_ptr876,
            b_ptr876,
            bs_ptr876,
            ad_ptr876,
            bd_ptr876,
            f_ptr876,
            (
                m,
                k,
                a_s0_861,
                a_s1_861,
                a_scale_s0_861,
                a_scale_s1_861,
                b_s0_861,
                b_s1_861,
                b_scale_s0_861,
                b_scale_s1_861,
                max(32, triton.next_power_of_2(k)),
                n,
                lda872,
                ldb872,
            ),
            (
                a_ptr876_aligned,
                as_ptr876_aligned,
                b_ptr876_aligned,
                bs_ptr876_aligned,
                ad_ptr876_aligned,
                bd_ptr876_aligned,
                f_ptr876_aligned,
            ),
            num_warps=4,
            grid=(m + n,),
            warmup=False,
        )
    else:
        signbit_dequant_rows_meta853_scalar876.run(
            a_ptr876,
            as_ptr876,
            ad_ptr876,
            f_ptr876,
            (
                m,
                k,
                a_s0_861,
                a_s1_861,
                a_scale_s0_861,
                a_scale_s1_861,
                256,
                0,
                lda872,
            ),
            (
                a_ptr876_aligned,
                as_ptr876_aligned,
                ad_ptr876_aligned,
                f_ptr876_aligned,
            ),
            num_warps=4,
            grid=(m,),
            warmup=False,
        )
        signbit_dequant_rows_meta853_scalar876.run(
            b_ptr876,
            bs_ptr876,
            bd_ptr876,
            f_ptr876,
            (
                n,
                k,
                b_s0_861,
                b_s1_861,
                b_scale_s0_861,
                b_scale_s1_861,
                256,
                m,
                ldb872,
            ),
            (
                b_ptr876_aligned,
                bs_ptr876_aligned,
                bd_ptr876_aligned,
                f_ptr876_aligned,
            ),
            num_warps=4,
            grid=(n,),
            warmup=False,
        )
    bm, bn, bk = (128, 128, 64) if m >= 512 else (64, 64, 64)
    if m >= 512 and k >= 2048:
        bm, bn, bk = (128, 64, 32)
    if m < 512 and k >= 2048:
        bm, bn, bk = (32, 64, 32)
    warps = 8 if bm == 128 else 4
    if m < 512 and k <= 1024:
        hip_guarded845_meta853_scalar876.run(
            ad_ptr876,
            bd_ptr876,
            f_ptr876,
            out_ptr876,
            a_ptr876,
            as_ptr876,
            b_ptr876,
            bs_ptr876,
            (
                m,
                n,
                k,
                bm,
                bn,
                bk,
                8,
                a_s0_861,
                a_s1_861,
                a_scale_s0_861,
                a_scale_s1_861,
                b_s0_861,
                b_s1_861,
                b_scale_s0_861,
                b_scale_s1_861,
                lda872,
                ldb872,
            ),
            (
                ad_ptr876_aligned,
                bd_ptr876_aligned,
                f_ptr876_aligned,
                out_ptr876_aligned,
                a_ptr876_aligned,
                as_ptr876_aligned,
                b_ptr876_aligned,
                bs_ptr876_aligned,
            ),
            output_t876,
            num_warps=warps,
            num_stages=3,
            grid=(triton.cdiv(m, bm) * triton.cdiv(n, bn),),
            warmup=False,
        )
    else:
        _fast_meta853_scalar876.run(
            ad_ptr876,
            bd_ptr876,
            out_ptr876,
            (
                m,
                n,
                k,
                bm,
                bn,
                bk,
                4 if m >= 512 and k >= 2048 else 8,
                lda872,
                ldb872,
            ),
            (ad_ptr876_aligned, bd_ptr876_aligned, out_ptr876_aligned),
            output_t876,
            num_warps=4 if m >= 512 and k >= 2048 else warps,
            num_stages=2 if m >= 512 and k >= 2048 else 3,
            grid=(triton.cdiv(m, bm) * triton.cdiv(n, bn),),
            warmup=False,
        )
        bm, bn, bk = (32, 64, 32)
        signbit_slow_meta853_scalar876.run(
            a_ptr876,
            as_ptr876,
            b_ptr876,
            bs_ptr876,
            f_ptr876,
            out_ptr876,
            (
                m,
                n,
                k,
                a_s0_861,
                a_s1_861,
                a_scale_s0_861,
                a_scale_s1_861,
                b_s0_861,
                b_s1_861,
                b_scale_s0_861,
                b_scale_s1_861,
                bm,
                bn,
                bk,
            ),
            (
                a_ptr876_aligned,
                as_ptr876_aligned,
                b_ptr876_aligned,
                bs_ptr876_aligned,
                f_ptr876_aligned,
                out_ptr876_aligned,
            ),
            output_t876,
            num_warps=4,
            num_stages=2,
            grid=(triton.cdiv(m, bm) * triton.cdiv(n, bn),),
            warmup=False,
        )
    return out


@triton.jit
def cast_ptr878(value, dtype: tl.constexpr, aligned: tl.constexpr):
    pointer = value.to(tl.pointer_type(dtype))
    if aligned:
        pointer = tl.multiple_of(pointer, 16)
    return pointer


@triton.jit
def _wide_index926(value, WIDE: tl.constexpr):
    return tl.cast(value, tl.int64) if WIDE else value


__all__ = ["mxfp8_block_scaled_matmul"]
