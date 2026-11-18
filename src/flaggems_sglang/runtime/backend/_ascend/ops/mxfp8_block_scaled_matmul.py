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
from triton.language import bfloat16 as bf16
from triton.language import int8 as i8
from triton.language import int32 as i32


@triton.jit
def _zero(Output, COUNT: tl.constexpr, B: tl.constexpr):
    ADDRESS64: tl.constexpr = COUNT + 2 * B >= 2147483648
    cols = (
        tl.cast(tl.program_id(0), tl.int64) if ADDRESS64 else tl.program_id(0)
    ) * B + (
        tl.cast(tl.arange(0, B), tl.int64) if ADDRESS64 else tl.arange(0, B)
    )
    tl.store(Output + cols, 0.0, cols < COUNT)


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
def _decode_linear(A, S, D, COUNT: tl.constexpr, B: tl.constexpr):
    ADDRESS64: tl.constexpr = COUNT + 2 * B >= 2147483648
    groups = (
        tl.cast(tl.program_id(0), tl.int64) if ADDRESS64 else tl.program_id(0)
    ) * (B // 16) + (
        tl.cast(tl.arange(0, B // 16), tl.int64)
        if ADDRESS64
        else tl.arange(0, B // 16)
    )
    lanes = (
        tl.cast(tl.arange(0, 16), tl.int64) if ADDRESS64 else tl.arange(0, 16)
    )
    pairs = groups[:, None] * 16 + lanes[None, :]
    packed = (
        tl.load(A + pairs, groups[:, None] < COUNT // 32, 0).to(tl.int32)
        & 65535
    )
    scale = tl.load(S + groups, groups < COUNT // 32, 127).to(tl.int32) & 255
    low = (
        _decode_halfbase(packed & 255, scale[:, None]).to(
            tl.uint32, bitcast=True
        )
        >> 16
        & 65535
    )
    high = (
        _decode_halfbase(packed >> 8, scale[:, None]).to(
            tl.uint32, bitcast=True
        )
        & 4294901760
    )
    output = tl.reshape(low | high, (B,))
    offset = (
        tl.cast(tl.program_id(0), tl.int64) if ADDRESS64 else tl.program_id(0)
    ) * B + (
        tl.cast(tl.arange(0, B), tl.int64) if ADDRESS64 else tl.arange(0, B)
    )
    tl.store(D + offset, output, offset < COUNT // 2)


@triton.jit
def _scale_flags(
    S,
    F,
    G: tl.constexpr,
    S0: tl.constexpr,
    S1: tl.constexpr,
    B: tl.constexpr,
    R: tl.constexpr,
    F_OFFSET: tl.constexpr = 0,
):
    F = F + tl.cast(F_OFFSET, tl.int64)
    ADDRESS64: tl.constexpr = (R + 1) * S0 + (
        2 * G + B
    ) * S1 >= 2147483648 or R + B >= 2147483648
    row = (
        tl.cast(tl.program_id(0), tl.int64) if ADDRESS64 else tl.program_id(0)
    )
    cols = tl.cast(tl.arange(0, B), tl.int64) if ADDRESS64 else tl.arange(0, B)
    s = tl.load(S + row * S0 + cols * S1, cols < G, 127).to(tl.int32) & 255
    safe = tl.min(((s >= 10) & (s <= 246)).to(tl.int32), 0)
    tl.store(F + row, safe)


@triton.jit
def _plain(
    A,
    B,
    Output,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
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
        acc = tl.dot(av, bv, acc, input_precision="ieee")
    tl.store(
        Output + rm[:, None] * N + rn[None, :],
        acc,
        (rm[:, None] < M) & (rn[None, :] < N),
    )


@triton.jit
def _scaled_product(ua, sa, ub, sb):
    ua = ua.to(tl.int32) & 255
    ub = ub.to(tl.int32) & 255
    sa = sa.to(tl.int32) & 255
    sb = sb.to(tl.int32) & 255
    av = _decode(ua, tl.full((), 127, tl.int32))
    bv = _decode(ub, tl.full((), 127, tl.int32))
    product = av * bv
    bits = product.to(tl.int32, bitcast=True)
    ae = av.to(tl.int32, bitcast=True) >> 23 & 255
    be = bv.to(tl.int32, bitcast=True) >> 23 & 255
    pe = (bits >> 23 & 255) + sa + sb - 254
    sign = bits & -2147483648
    normal = sign | pe << 23 | bits & 8388607
    shift = tl.minimum(tl.maximum(1 - pe, 1), 25)
    significand = bits & 8388607 | 8388608
    rounded = (
        significand + (1 << shift - 1) - 1 + (significand >> shift & 1)
        >> shift
    )
    subnormal = sign | tl.where(pe >= -23, rounded, 0)
    result = tl.where(pe > 0, normal, subnormal).to(tl.float32, bitcast=True)
    az = ua & 127 == 0
    bz = ub & 127 == 0
    ai = (sa == 255) | (ae + sa - 127 >= 255)
    bi = (sb == 255) | (be + sb - 127 >= 255)
    nan = (
        (ua & 127 == 127)
        | (ub & 127 == 127)
        | ai & bz
        | bi & az
        | (sa == 255) & az
        | (sb == 255) & bz
    )
    result = tl.where(az | bz, 0.0, result)
    infinity = (sign | 2139095040).to(tl.float32, bitcast=True)
    result = tl.where(ai | bi | (pe >= 255), infinity, result)
    return tl.where(nan, float("nan"), result)


@triton.jit
def _copy_strided_bytes(
    A,
    Output,
    R: tl.constexpr,
    K: tl.constexpr,
    S0: tl.constexpr,
    S1: tl.constexpr,
    B: tl.constexpr,
):
    ADDRESS64: tl.constexpr = (R + 1) * S0 + (
        2 * K + B
    ) * S1 >= 2147483648 or (R + 1) * K + B >= 2147483648
    for block in range(
        tl.cast(tl.program_id(0), tl.int64) if ADDRESS64 else tl.program_id(0),
        tl.cdiv(R * K, B),
        tl.num_programs(0),
    ):
        ix = (tl.cast(block, tl.int64) if ADDRESS64 else block) * B + (
            tl.cast(tl.arange(0, B), tl.int64)
            if ADDRESS64
            else tl.arange(0, B)
        )
        value = tl.load(A + ix // K * S0 + ix % K * S1, ix < R * K, 0).to(
            tl.int32
        )
        tl.store(Output + ix, value, ix < R * K)


@triton.jit
def _small_flags(
    SA,
    SB,
    Output,
    M: tl.constexpr,
    N: tl.constexpr,
    G: tl.constexpr,
    AS0: tl.constexpr,
    AS1: tl.constexpr,
    BS0: tl.constexpr,
    BS1: tl.constexpr,
    B: tl.constexpr,
):
    ADDRESS64: tl.constexpr = (
        (M + 1024) * N + (N + 1024) >= 2147483648
        or (M + 1024) * AS0 + (2 * G + B) * AS1 >= 2147483648
        or (N + 1024) * BS0 + (2 * G + B) * BS1 >= 2147483648
        or (M + N + B >= 2147483648)
    )
    cols = tl.cast(tl.arange(0, B), tl.int64) if ADDRESS64 else tl.arange(0, B)
    sa = (
        tl.load(SA + cols // G * AS0 + cols % G * AS1, cols < M * G, 127).to(
            tl.int32
        )
        & 255
    )
    sb = (
        tl.load(SB + cols // G * BS0 + cols % G * BS1, cols < N * G, 127).to(
            tl.int32
        )
        & 255
    )
    va = ((sa >= 10) & (sa <= 246)).to(tl.int32)
    vb = ((sb >= 10) & (sb <= 246)).to(tl.int32)
    value = tl.minimum(tl.min(va, 0), tl.min(vb, 0))
    tl.store(Output, value)


@triton.jit
def _range_guarded(
    A,
    AS,
    B,
    BS,
    Output,
    SAFE,
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
    BN: tl.constexpr,
):
    ADDRESS64: tl.constexpr = (
        (M + 1024) * N + (N + 4 * BN) >= 2147483648
        or (M + 1024) * K + (2 * K + 4096) >= 2147483648
        or (N + 4 * BN) * K + (2 * K + 4096) >= 2147483648
        or ((M + 1024) * A0 + (2 * K + 4096) * A1 >= 2147483648)
        or ((M + 1024) * AS0 + ((2 * K + 4096) // 32 + 1) * AS1 >= 2147483648)
        or ((N + 4 * BN) * B0 + (2 * K + 4096) * B1 >= 2147483648)
        or (
            (N + 4 * BN) * BS0 + ((2 * K + 4096) // 32 + 1) * BS1 >= 2147483648
        )
    )
    if tl.load(SAFE) == 0:
        for index in range(
            (
                tl.cast(tl.program_id(0), tl.int64)
                if ADDRESS64
                else tl.program_id(0)
            ),
            M * N,
            tl.num_programs(0),
        ):
            row = index // N
            col = index % N
            acc = tl.full((), 0.0, tl.float32)
            for block in range(tl.cdiv(K, 32)):
                ks = (
                    tl.cast(block, tl.int64) if ADDRESS64 else block
                ) * 32 + (
                    tl.cast(tl.arange(0, 32), tl.int64)
                    if ADDRESS64
                    else tl.arange(0, 32)
                )
                ua = tl.load(A + row * A0 + ks * A1, ks < K, 0)
                sa = tl.load(AS + row * AS0 + ks // 32 * AS1, ks < K, 127)
                ub = tl.load(B + col * B0 + ks * B1, ks < K, 0)
                sb = tl.load(BS + col * BS0 + ks // 32 * BS1, ks < K, 127)
                value = _scaled_product(ua, sa, ub, sb)
                acc += tl.sum(value, 0)
            tl.store(Output + row * N + col, acc)


@triton.jit
def _scale_both_binary(
    SA,
    FA,
    SB,
    FB,
    COUNT_A: tl.constexpr,
    COUNT_B: tl.constexpr,
    B: tl.constexpr,
):
    FB = FB + tl.cast(tl.cdiv(COUNT_A, B), tl.int64)
    ADDRESS64: tl.constexpr = COUNT_A + COUNT_B + 2 * B >= 2147483648
    pid = (
        tl.cast(tl.program_id(0), tl.int64) if ADDRESS64 else tl.program_id(0)
    )
    blocks_a = tl.cdiv(COUNT_A, B)
    if pid < blocks_a:
        _scale_at(SA, FA, COUNT_A, B, pid)
    else:
        _scale_at(SB, FB, COUNT_B, B, pid - blocks_a)


@triton.jit
def _scale_at(S, F, COUNT: tl.constexpr, B: tl.constexpr, block):
    ADDRESS64: tl.constexpr = COUNT + 2 * B >= 2147483648
    block = tl.cast(block, tl.int64) if ADDRESS64 else block
    pid = block
    cols = pid * B + (
        tl.cast(tl.arange(0, B), tl.int64) if ADDRESS64 else tl.arange(0, B)
    )
    s = tl.load(S + cols, cols < COUNT, 127).to(tl.int32) & 255
    safe = tl.min(((s >= 10) & (s <= 246)).to(tl.int32), 0)
    tl.store(F + pid, safe)


@triton.jit
def _range_guarded_flags(
    A,
    AS,
    B,
    BS,
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
    BN: tl.constexpr,
    FA,
    FB,
    CA: tl.constexpr,
    CB: tl.constexpr,
    FLAGS_B: tl.constexpr,
):
    FB = FB + tl.cast(CA, tl.int64)
    ADDRESS64: tl.constexpr = (
        (M + 1024) * N + (N + 4 * BN) >= 2147483648
        or (M + 1024) * K + (2 * K + 4096) >= 2147483648
        or (N + 4 * BN) * K + (2 * K + 4096) >= 2147483648
        or ((M + 1024) * A0 + (2 * K + 4096) * A1 >= 2147483648)
        or ((M + 1024) * AS0 + ((2 * K + 4096) // 32 + 1) * AS1 >= 2147483648)
        or ((N + 4 * BN) * B0 + (2 * K + 4096) * B1 >= 2147483648)
        or (
            (N + 4 * BN) * BS0 + ((2 * K + 4096) // 32 + 1) * BS1 >= 2147483648
        )
        or (CA + CB + FLAGS_B >= 2147483648)
    )
    offset = (
        tl.cast(tl.arange(0, FLAGS_B), tl.int64)
        if ADDRESS64
        else tl.arange(0, FLAGS_B)
    )
    av = tl.load(FA + offset, offset < CA, 1)
    bv = tl.load(FB + offset, offset < CB, 1)
    flag_value = tl.min(av & bv, 0)
    if flag_value == 0:
        for index in range(
            (
                tl.cast(tl.program_id(0), tl.int64)
                if ADDRESS64
                else tl.program_id(0)
            ),
            M * N,
            tl.num_programs(0),
        ):
            row = index // N
            col = index % N
            acc = tl.full((), 0.0, tl.float32)
            for block in range(tl.cdiv(K, 32)):
                ks = (
                    tl.cast(block, tl.int64) if ADDRESS64 else block
                ) * 32 + (
                    tl.cast(tl.arange(0, 32), tl.int64)
                    if ADDRESS64
                    else tl.arange(0, 32)
                )
                ua = tl.load(A + row * A0 + ks * A1, ks < K, 0)
                sa = tl.load(AS + row * AS0 + ks // 32 * AS1, ks < K, 127)
                ub = tl.load(B + col * B0 + ks * B1, ks < K, 0)
                sb = tl.load(BS + col * BS0 + ks // 32 * BS1, ks < K, 127)
                value = _scaled_product(ua, sa, ub, sb)
                acc += tl.sum(value, 0)
            tl.store(Output + row * N + col, acc)


@triton.jit
def _decode_halfbase(u, s):
    u = u.to(tl.int32) & 255
    s = s.to(tl.int32) & 255
    half_bits = (u & 128) << 8 | (u & 127) << 7
    value = half_bits.to(tl.uint16).to(tl.float16, bitcast=True).to(tl.float32)
    scale = (s + 8 << 23).to(tl.float32, bitcast=True)
    return tl.where(u & 127 == 127, float("nan"), value * scale)


@triton.jit
def _plain_full(
    A,
    B,
    Output,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
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
    if not ADDRESS64:
        ap895 = tl.make_block_ptr(
            A,
            (M, K),
            (K, 1),
            (pid // tl.cdiv(N, BN) * BM, 0),
            (BM, BK),
            (1, 0),
        )
        bp895 = tl.make_block_ptr(
            B, (K, N), (1, K), (0, pid % tl.cdiv(N, BN) * BN), (BK, BN), (0, 1)
        )
    acc = tl.zeros((BM, BN), tl.float32)
    for k in range(tl.cdiv(K, BK)):
        if ADDRESS64:
            kk = (tl.cast(k, tl.int64) if ADDRESS64 else k) * BK + rk
            if M % BM == 0 and K % BK == 0:
                av = tl.load(A + rm[:, None] * K + kk[None, :])
            else:
                av = tl.load(
                    A + rm[:, None] * K + kk[None, :],
                    (rm[:, None] < M) & (kk[None, :] < K),
                    0,
                )
            if N % BN == 0 and K % BK == 0:
                bv = tl.load(B + rn[None, :] * K + kk[:, None])
            else:
                bv = tl.load(
                    B + rn[None, :] * K + kk[:, None],
                    (rn[None, :] < N) & (kk[:, None] < K),
                    0,
                )
        else:
            if M % BM == 0 and K % BK == 0:
                av = tl.load(ap895)
            else:
                av = tl.load(
                    ap895, boundary_check=(0, 1), padding_option="zero"
                )
            if N % BN == 0 and K % BK == 0:
                bv = tl.load(bp895)
            else:
                bv = tl.load(
                    bp895, boundary_check=(0, 1), padding_option="zero"
                )
        acc = tl.dot(av, bv, acc, input_precision="ieee")
        if not ADDRESS64:
            ap895 = tl.advance(ap895, (0, BK))
            bp895 = tl.advance(bp895, (BK, 0))
    if M % BM == 0 and N % BN == 0:
        tl.store(Output + rm[:, None] * N + rn[None, :], acc)
    else:
        tl.store(
            Output + rm[:, None] * N + rn[None, :],
            acc,
            (rm[:, None] < M) & (rn[None, :] < N),
        )


@triton.jit
def _decode_linear_packedmask340(
    A, S, D, COUNT: tl.constexpr, B: tl.constexpr
):
    ADDRESS64: tl.constexpr = COUNT + 2 * B >= 2147483648
    groups = (
        tl.cast(tl.program_id(0), tl.int64) if ADDRESS64 else tl.program_id(0)
    ) * (B // 16) + (
        tl.cast(tl.arange(0, B // 16), tl.int64)
        if ADDRESS64
        else tl.arange(0, B // 16)
    )
    lanes = (
        tl.cast(tl.arange(0, 16), tl.int64) if ADDRESS64 else tl.arange(0, 16)
    )
    pairs = groups[:, None] * 16 + lanes[None, :]
    packed = (
        tl.load(A + pairs, groups[:, None] < COUNT // 32, 0).to(tl.int32)
        & 65535
    )
    scale = tl.load(S + groups, groups < COUNT // 32, 127).to(tl.int32) & 255
    lowmag = packed & 127
    highmag = packed & 32512
    lowbits = (packed & 128) << 8 | lowmag << 7
    highbits = packed & 32768 | highmag >> 1
    lv = lowbits.to(tl.uint16).to(tl.float16, bitcast=True).to(tl.float32)
    hv = highbits.to(tl.uint16).to(tl.float16, bitcast=True).to(tl.float32)
    factor = (scale + 8 << 23).to(tl.float32, bitcast=True)[:, None]
    lv = tl.where(lowmag == 127, float("nan"), lv * factor)
    hv = tl.where(highmag == 32512, float("nan"), hv * factor)
    low = lv.to(tl.uint32, bitcast=True) >> 16 & 65535
    high = hv.to(tl.uint32, bitcast=True) & 4294901760
    output = tl.reshape(low | high, (B,))
    offset = (
        tl.cast(tl.program_id(0), tl.int64) if ADDRESS64 else tl.program_id(0)
    ) * B + (
        tl.cast(tl.arange(0, B), tl.int64) if ADDRESS64 else tl.arange(0, B)
    )
    tl.store(D + offset, output, offset < COUNT // 2)


@triton.jit
def _dequant_row_at358(
    A,
    S,
    D,
    K: tl.constexpr,
    A0: tl.constexpr,
    A1: tl.constexpr,
    S0: tl.constexpr,
    S1: tl.constexpr,
    B: tl.constexpr,
    row,
):
    ADDRESS64: tl.constexpr = (
        (2 * K + B + 1024) * A1 >= 2147483648
        or ((2 * K + B + 1024) // 32 + 1) * S1 >= 2147483648
        or 2 * K + B + 1024 >= 2147483648
    )
    row = tl.cast(row, tl.int64) if ADDRESS64 else row
    cols = tl.cast(tl.arange(0, B), tl.int64) if ADDRESS64 else tl.arange(0, B)
    u = tl.load(A + row * A0 + cols * A1, cols < K, 0)
    s = (
        tl.load(S + row * S0 + cols // 32 * S1, cols < K, 127).to(tl.int32)
        & 255
    )
    value = _decode(u, s)
    tl.store(D + row * K + cols, value, cols < K)


@triton.jit
def _dequant_rows_joint358(
    A,
    AS,
    AD,
    C,
    CS,
    CD,
    M: tl.constexpr,
    K: tl.constexpr,
    A0: tl.constexpr,
    A1: tl.constexpr,
    AS0: tl.constexpr,
    AS1: tl.constexpr,
    C0: tl.constexpr,
    C1: tl.constexpr,
    CS0: tl.constexpr,
    CS1: tl.constexpr,
    B: tl.constexpr,
    N: tl.constexpr,
):
    ADDRESS64: tl.constexpr = (
        (M + 1024) * N + (N + 1024) >= 2147483648
        or (M + 1024) * K + (2 * K + 4096) >= 2147483648
        or (N + 1024) * K + (2 * K + 4096) >= 2147483648
        or ((M + 1024) * A0 + (2 * K + 4096) * A1 >= 2147483648)
        or ((M + 1024) * AS0 + ((2 * K + 4096) // 32 + 1) * AS1 >= 2147483648)
        or ((N + 1024) * C0 + (2 * K + 4096) * C1 >= 2147483648)
        or ((N + 1024) * CS0 + ((2 * K + 4096) // 32 + 1) * CS1 >= 2147483648)
        or (M + N + B >= 2147483648)
    )
    pid = (
        tl.cast(tl.program_id(0), tl.int64) if ADDRESS64 else tl.program_id(0)
    )
    if pid < M:
        _dequant_row_at358(A, AS, AD, K, A0, A1, AS0, AS1, B, pid)
    else:
        _dequant_row_at358(C, CS, CD, K, C0, C1, CS0, CS1, B, pid - M)


@triton.jit
def _zero_bounded528(Output, COUNT: tl.constexpr, B: tl.constexpr):
    ADDRESS64: tl.constexpr = COUNT + 2 * B >= 2147483648
    for _work528 in range(
        tl.cast(tl.program_id(0), tl.int64) if ADDRESS64 else tl.program_id(0),
        tl.cdiv(COUNT, B),
        tl.num_programs(0),
    ):
        cols = (tl.cast(_work528, tl.int64) if ADDRESS64 else _work528) * B + (
            tl.cast(tl.arange(0, B), tl.int64)
            if ADDRESS64
            else tl.arange(0, B)
        )
        tl.store(Output + cols, 0.0, cols < COUNT)


@triton.jit
def _decode_linear_bounded528(A, S, D, COUNT: tl.constexpr, B: tl.constexpr):
    ADDRESS64: tl.constexpr = COUNT + 2 * B >= 2147483648
    for _work528 in range(
        tl.cast(tl.program_id(0), tl.int64) if ADDRESS64 else tl.program_id(0),
        tl.cdiv(COUNT, 2 * B),
        tl.num_programs(0),
    ):
        groups = (tl.cast(_work528, tl.int64) if ADDRESS64 else _work528) * (
            B // 16
        ) + (
            tl.cast(tl.arange(0, B // 16), tl.int64)
            if ADDRESS64
            else tl.arange(0, B // 16)
        )
        lanes = (
            tl.cast(tl.arange(0, 16), tl.int64)
            if ADDRESS64
            else tl.arange(0, 16)
        )
        pairs = groups[:, None] * 16 + lanes[None, :]
        packed = (
            tl.load(A + pairs, groups[:, None] < COUNT // 32, 0).to(tl.int32)
            & 65535
        )
        scale = (
            tl.load(S + groups, groups < COUNT // 32, 127).to(tl.int32) & 255
        )
        low = (
            _decode_halfbase(packed & 255, scale[:, None]).to(
                tl.uint32, bitcast=True
            )
            >> 16
            & 65535
        )
        high = (
            _decode_halfbase(packed >> 8, scale[:, None]).to(
                tl.uint32, bitcast=True
            )
            & 4294901760
        )
        output = tl.reshape(low | high, (B,))
        offset = (
            tl.cast(_work528, tl.int64) if ADDRESS64 else _work528
        ) * B + (
            tl.cast(tl.arange(0, B), tl.int64)
            if ADDRESS64
            else tl.arange(0, B)
        )
        tl.store(D + offset, output, offset < COUNT // 2)


@triton.jit
def _decode_linear_packedmask340_bounded528(
    A, S, D, COUNT: tl.constexpr, B: tl.constexpr
):
    ADDRESS64: tl.constexpr = COUNT + 2 * B >= 2147483648
    for _work528 in range(
        tl.cast(tl.program_id(0), tl.int64) if ADDRESS64 else tl.program_id(0),
        tl.cdiv(COUNT, 2 * B),
        tl.num_programs(0),
    ):
        groups = (tl.cast(_work528, tl.int64) if ADDRESS64 else _work528) * (
            B // 16
        ) + (
            tl.cast(tl.arange(0, B // 16), tl.int64)
            if ADDRESS64
            else tl.arange(0, B // 16)
        )
        lanes = (
            tl.cast(tl.arange(0, 16), tl.int64)
            if ADDRESS64
            else tl.arange(0, 16)
        )
        pairs = groups[:, None] * 16 + lanes[None, :]
        packed = (
            tl.load(A + pairs, groups[:, None] < COUNT // 32, 0).to(tl.int32)
            & 65535
        )
        scale = (
            tl.load(S + groups, groups < COUNT // 32, 127).to(tl.int32) & 255
        )
        lowmag = packed & 127
        highmag = packed & 32512
        lowbits = (packed & 128) << 8 | lowmag << 7
        highbits = packed & 32768 | highmag >> 1
        lv = lowbits.to(tl.uint16).to(tl.float16, bitcast=True).to(tl.float32)
        hv = highbits.to(tl.uint16).to(tl.float16, bitcast=True).to(tl.float32)
        factor = (scale + 8 << 23).to(tl.float32, bitcast=True)[:, None]
        lv = tl.where(lowmag == 127, float("nan"), lv * factor)
        hv = tl.where(highmag == 32512, float("nan"), hv * factor)
        low = lv.to(tl.uint32, bitcast=True) >> 16 & 65535
        high = hv.to(tl.uint32, bitcast=True) & 4294901760
        output = tl.reshape(low | high, (B,))
        offset = (
            tl.cast(_work528, tl.int64) if ADDRESS64 else _work528
        ) * B + (
            tl.cast(tl.arange(0, B), tl.int64)
            if ADDRESS64
            else tl.arange(0, B)
        )
        tl.store(D + offset, output, offset < COUNT // 2)


@triton.jit
def _scale_both_binary_bounded528(
    SA,
    FA,
    SB,
    FB,
    COUNT_A: tl.constexpr,
    COUNT_B: tl.constexpr,
    B: tl.constexpr,
):
    FB = FB + tl.cast(tl.cdiv(COUNT_A, B), tl.int64)
    ADDRESS64: tl.constexpr = COUNT_A + COUNT_B + 2 * B >= 2147483648
    for _work528 in range(
        tl.cast(tl.program_id(0), tl.int64) if ADDRESS64 else tl.program_id(0),
        tl.cdiv(COUNT_A, B) + tl.cdiv(COUNT_B, B),
        tl.num_programs(0),
    ):
        pid = tl.cast(_work528, tl.int64) if ADDRESS64 else _work528
        blocks_a = tl.cdiv(COUNT_A, B)
        if pid < blocks_a:
            _scale_at(SA, FA, COUNT_A, B, pid)
        else:
            _scale_at(SB, FB, COUNT_B, B, pid - blocks_a)


@triton.jit
def _plain_bounded528(
    A,
    B,
    Output,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    ADDRESS64: tl.constexpr = (
        (M + 4 * BM) * N + (N + 4 * BN) >= 2147483648
        or (M + 4 * BM) * K + (K + 4 * BK) >= 2147483648
        or (N + 4 * BN) * K + (K + 4 * BK) >= 2147483648
    )
    for _work528 in range(
        tl.cast(tl.program_id(0), tl.int64) if ADDRESS64 else tl.program_id(0),
        tl.cdiv(M, BM) * tl.cdiv(N, BN),
        tl.num_programs(0),
    ):
        pid = tl.cast(_work528, tl.int64) if ADDRESS64 else _work528
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
        rk = (
            tl.cast(tl.arange(0, BK), tl.int64)
            if ADDRESS64
            else tl.arange(0, BK)
        )
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
            acc = tl.dot(av, bv, acc, input_precision="ieee")
        tl.store(
            Output + rm[:, None] * N + rn[None, :],
            acc,
            (rm[:, None] < M) & (rn[None, :] < N),
        )


@triton.jit
def _plain_full_bounded528(
    A,
    B,
    Output,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    ADDRESS64: tl.constexpr = (
        (M + 4 * BM) * N + (N + 4 * BN) >= 2147483648
        or (M + 4 * BM) * K + (K + 4 * BK) >= 2147483648
        or (N + 4 * BN) * K + (K + 4 * BK) >= 2147483648
    )
    for _work528 in range(
        tl.cast(tl.program_id(0), tl.int64) if ADDRESS64 else tl.program_id(0),
        tl.cdiv(M, BM) * tl.cdiv(N, BN),
        tl.num_programs(0),
    ):
        pid = tl.cast(_work528, tl.int64) if ADDRESS64 else _work528
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
        rk = (
            tl.cast(tl.arange(0, BK), tl.int64)
            if ADDRESS64
            else tl.arange(0, BK)
        )
        if not ADDRESS64:
            ap895 = tl.make_block_ptr(
                A,
                (M, K),
                (K, 1),
                (pid // tl.cdiv(N, BN) * BM, 0),
                (BM, BK),
                (1, 0),
            )
            bp895 = tl.make_block_ptr(
                B,
                (K, N),
                (1, K),
                (0, pid % tl.cdiv(N, BN) * BN),
                (BK, BN),
                (0, 1),
            )
        acc = tl.zeros((BM, BN), tl.float32)
        for k in range(tl.cdiv(K, BK)):
            if ADDRESS64:
                kk = (tl.cast(k, tl.int64) if ADDRESS64 else k) * BK + rk
                if M % BM == 0 and K % BK == 0:
                    av = tl.load(A + rm[:, None] * K + kk[None, :])
                else:
                    av = tl.load(
                        A + rm[:, None] * K + kk[None, :],
                        (rm[:, None] < M) & (kk[None, :] < K),
                        0,
                    )
                if N % BN == 0 and K % BK == 0:
                    bv = tl.load(B + rn[None, :] * K + kk[:, None])
                else:
                    bv = tl.load(
                        B + rn[None, :] * K + kk[:, None],
                        (rn[None, :] < N) & (kk[:, None] < K),
                        0,
                    )
            else:
                if M % BM == 0 and K % BK == 0:
                    av = tl.load(ap895)
                else:
                    av = tl.load(
                        ap895, boundary_check=(0, 1), padding_option="zero"
                    )
                if N % BN == 0 and K % BK == 0:
                    bv = tl.load(bp895)
                else:
                    bv = tl.load(
                        bp895, boundary_check=(0, 1), padding_option="zero"
                    )
            acc = tl.dot(av, bv, acc, input_precision="ieee")
            if not ADDRESS64:
                ap895 = tl.advance(ap895, (0, BK))
                bp895 = tl.advance(bp895, (BK, 0))
        if M % BM == 0 and N % BN == 0:
            tl.store(Output + rm[:, None] * N + rn[None, :], acc)
        else:
            tl.store(
                Output + rm[:, None] * N + rn[None, :],
                acc,
                (rm[:, None] < M) & (rn[None, :] < N),
            )


@triton.jit
def _dequant_row_at528(
    A,
    S,
    D,
    K: tl.constexpr,
    A0: tl.constexpr,
    A1: tl.constexpr,
    S0: tl.constexpr,
    S1: tl.constexpr,
    B: tl.constexpr,
    row,
    chunk,
):
    ADDRESS64: tl.constexpr = (
        (2 * K + B + 1024) * A1 >= 2147483648
        or ((2 * K + B + 1024) // 32 + 1) * S1 >= 2147483648
        or 2 * K + B + 1024 >= 2147483648
    )
    row = tl.cast(row, tl.int64) if ADDRESS64 else row
    cols = (tl.cast(chunk, tl.int64) if ADDRESS64 else chunk) * B + (
        tl.cast(tl.arange(0, B), tl.int64) if ADDRESS64 else tl.arange(0, B)
    )
    u = tl.load(A + row * A0 + cols * A1, cols < K, 0)
    s = (
        tl.load(S + row * S0 + cols // 32 * S1, cols < K, 127).to(tl.int32)
        & 255
    )
    value = _decode(u, s)
    tl.store(D + row * K + cols, value, cols < K)


@triton.jit
def _dequant_rows_joint528(
    A,
    AS,
    AD,
    C,
    CS,
    CD,
    M: tl.constexpr,
    K: tl.constexpr,
    A0: tl.constexpr,
    A1: tl.constexpr,
    AS0: tl.constexpr,
    AS1: tl.constexpr,
    C0: tl.constexpr,
    C1: tl.constexpr,
    CS0: tl.constexpr,
    CS1: tl.constexpr,
    B: tl.constexpr,
    N: tl.constexpr,
):
    ADDRESS64: tl.constexpr = (
        (M + 1024) * N + (N + 1024) >= 2147483648
        or (M + 1024) * K + (2 * K + 4096) >= 2147483648
        or (N + 1024) * K + (2 * K + 4096) >= 2147483648
        or ((M + 1024) * A0 + (2 * K + 4096) * A1 >= 2147483648)
        or ((M + 1024) * AS0 + ((2 * K + 4096) // 32 + 1) * AS1 >= 2147483648)
        or ((N + 1024) * C0 + (2 * K + 4096) * C1 >= 2147483648)
        or ((N + 1024) * CS0 + ((2 * K + 4096) // 32 + 1) * CS1 >= 2147483648)
        or (M + N + B >= 2147483648)
    )
    for _work528 in range(
        tl.cast(tl.program_id(0), tl.int64) if ADDRESS64 else tl.program_id(0),
        (M + N) * tl.cdiv(K, B),
        tl.num_programs(0),
    ):
        pid = (
            tl.cast(_work528, tl.int64) if ADDRESS64 else _work528
        ) // tl.cdiv(K, B)
        if pid < M:
            _dequant_row_at528(
                A,
                AS,
                AD,
                K,
                A0,
                A1,
                AS0,
                AS1,
                B,
                pid,
                _work528 % tl.cdiv(K, B),
            )
        else:
            _dequant_row_at528(
                C,
                CS,
                CD,
                K,
                C0,
                C1,
                CS0,
                CS1,
                B,
                pid - M,
                _work528 % tl.cdiv(K, B),
            )


@triton.jit
def _scale_flags_bounded528(
    S,
    F,
    G: tl.constexpr,
    S0: tl.constexpr,
    S1: tl.constexpr,
    B: tl.constexpr,
    R: tl.constexpr,
    F_OFFSET: tl.constexpr = 0,
):
    F = F + tl.cast(F_OFFSET, tl.int64)
    ADDRESS64: tl.constexpr = (R + 1) * S0 + (
        2 * G + B
    ) * S1 >= 2147483648 or R + B >= 2147483648
    for row in range(
        tl.cast(tl.program_id(0), tl.int64) if ADDRESS64 else tl.program_id(0),
        R,
        tl.num_programs(0),
    ):
        safe = tl.full((), 1, tl.int32)
        for block in range(tl.cdiv(G, B)):
            cols = (tl.cast(block, tl.int64) if ADDRESS64 else block) * B + (
                tl.cast(tl.arange(0, B), tl.int64)
                if ADDRESS64
                else tl.arange(0, B)
            )
            s = (
                tl.load(S + row * S0 + cols * S1, cols < G, 127).to(tl.int32)
                & 255
            )
            safe = tl.minimum(
                safe, tl.min(((s >= 10) & (s <= 246)).to(tl.int32), 0)
            )
        tl.store(F + row, safe)


@triton.jit
def _all_safe_bounded528(
    FA, FB, F, M: tl.constexpr, N: tl.constexpr, B: tl.constexpr
):
    FB = FB + tl.cast(M, tl.int64)
    ADDRESS64: tl.constexpr = M + N + 2 * B >= 2147483648
    safe = tl.full((), 1, tl.int32)
    for block in range(tl.cdiv(tl.maximum(M, N), B)):
        r = (tl.cast(block, tl.int64) if ADDRESS64 else block) * B + (
            tl.cast(tl.arange(0, B), tl.int64)
            if ADDRESS64
            else tl.arange(0, B)
        )
        a = tl.load(FA + r, r < M, 1)
        b = tl.load(FB + r, r < N, 1)
        safe = tl.minimum(safe, tl.min(a & b, 0))
    tl.store(F, safe)


@triton.jit(do_not_specialize=["SA", "SB", "Output"])
def _small_flags_scalar880(
    SA: tl.uint64,
    SB: tl.uint64,
    Output: tl.uint64,
    M: tl.constexpr,
    N: tl.constexpr,
    G: tl.constexpr,
    AS0: tl.constexpr,
    AS1: tl.constexpr,
    BS0: tl.constexpr,
    BS1: tl.constexpr,
    B: tl.constexpr,
    ALIGN880: tl.constexpr,
):
    SA = SA.to(tl.pointer_type(tl.int8))
    if ALIGN880[0]:
        SA = tl.multiple_of(SA, 16)
    SB = SB.to(tl.pointer_type(tl.int8))
    if ALIGN880[1]:
        SB = tl.multiple_of(SB, 16)
    Output = Output.to(tl.pointer_type(tl.int32))
    if ALIGN880[2]:
        Output = tl.multiple_of(Output, 16)
    ADDRESS64: tl.constexpr = (
        (M + 1024) * N + (N + 1024) >= 2147483648
        or (M + 1024) * AS0 + (2 * G + B) * AS1 >= 2147483648
        or (N + 1024) * BS0 + (2 * G + B) * BS1 >= 2147483648
        or (M + N + B >= 2147483648)
    )
    cols = tl.cast(tl.arange(0, B), tl.int64) if ADDRESS64 else tl.arange(0, B)
    sa = (
        tl.load(SA + cols // G * AS0 + cols % G * AS1, cols < M * G, 127).to(
            tl.int32
        )
        & 255
    )
    sb = (
        tl.load(SB + cols // G * BS0 + cols % G * BS1, cols < N * G, 127).to(
            tl.int32
        )
        & 255
    )
    va = ((sa >= 10) & (sa <= 246)).to(tl.int32)
    vb = ((sb >= 10) & (sb <= 246)).to(tl.int32)
    value = tl.minimum(tl.min(va, 0), tl.min(vb, 0))
    tl.store(Output, value)


@triton.jit(do_not_specialize=["A", "AS", "B", "BS", "Output"])
def _fused_small_scalar880(
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
    ALIGN880: tl.constexpr,
    INPUT_A880: tl.constexpr,
    INPUT_B880: tl.constexpr,
):
    A = A.to(tl.pointer_type(INPUT_A880))
    if ALIGN880[0]:
        A = tl.multiple_of(A, 16)
    AS = AS.to(tl.pointer_type(tl.int8))
    if ALIGN880[1]:
        AS = tl.multiple_of(AS, 16)
    B = B.to(tl.pointer_type(INPUT_B880))
    if ALIGN880[2]:
        B = tl.multiple_of(B, 16)
    BS = BS.to(tl.pointer_type(tl.int8))
    if ALIGN880[3]:
        BS = tl.multiple_of(BS, 16)
    Output = Output.to(tl.pointer_type(tl.bfloat16))
    if ALIGN880[4]:
        Output = tl.multiple_of(Output, 16)
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
    for block in range(tl.cdiv(K, BK)):
        kk = (tl.cast(block, tl.int64) if ADDRESS64 else block) * BK + rk
        ua = tl.load(
            A + rm[:, None] * A0 + kk[None, :] * A1,
            (rm[:, None] < M) & (kk[None, :] < K),
            0,
        )
        sa = tl.load(
            AS + rm[:, None] * AS0 + kk[None, :] // 32 * AS1,
            (rm[:, None] < M) & (kk[None, :] < K),
            127,
        )
        ub = tl.load(
            B + rn[None, :] * B0 + kk[:, None] * B1,
            (rn[None, :] < N) & (kk[:, None] < K),
            0,
        )
        sb = tl.load(
            BS + rn[None, :] * BS0 + kk[:, None] // 32 * BS1,
            (rn[None, :] < N) & (kk[:, None] < K),
            127,
        )
        av = _decode(ua, sa).to(tl.bfloat16)
        bv = _decode(ub, sb).to(tl.bfloat16)
        acc = tl.dot(av, bv, acc, input_precision="ieee")
    tl.store(
        Output + rm[:, None] * N + rn[None, :],
        acc,
        (rm[:, None] < M) & (rn[None, :] < N),
    )


@triton.jit(do_not_specialize=["A", "AS", "B", "BS", "Output", "SAFE"])
def _range_guarded_scalar880(
    A: tl.uint64,
    AS: tl.uint64,
    B: tl.uint64,
    BS: tl.uint64,
    Output: tl.uint64,
    SAFE: tl.uint64,
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
    BN: tl.constexpr,
    ALIGN880: tl.constexpr,
    INPUT_A880: tl.constexpr,
    INPUT_B880: tl.constexpr,
):
    A = A.to(tl.pointer_type(INPUT_A880))
    if ALIGN880[0]:
        A = tl.multiple_of(A, 16)
    AS = AS.to(tl.pointer_type(tl.int8))
    if ALIGN880[1]:
        AS = tl.multiple_of(AS, 16)
    B = B.to(tl.pointer_type(INPUT_B880))
    if ALIGN880[2]:
        B = tl.multiple_of(B, 16)
    BS = BS.to(tl.pointer_type(tl.int8))
    if ALIGN880[3]:
        BS = tl.multiple_of(BS, 16)
    Output = Output.to(tl.pointer_type(tl.bfloat16))
    if ALIGN880[4]:
        Output = tl.multiple_of(Output, 16)
    SAFE = SAFE.to(tl.pointer_type(tl.int32))
    if ALIGN880[5]:
        SAFE = tl.multiple_of(SAFE, 16)
    ADDRESS64: tl.constexpr = (
        (M + 1024) * N + (N + 4 * BN) >= 2147483648
        or (M + 1024) * K + (2 * K + 4096) >= 2147483648
        or (N + 4 * BN) * K + (2 * K + 4096) >= 2147483648
        or ((M + 1024) * A0 + (2 * K + 4096) * A1 >= 2147483648)
        or ((M + 1024) * AS0 + ((2 * K + 4096) // 32 + 1) * AS1 >= 2147483648)
        or ((N + 4 * BN) * B0 + (2 * K + 4096) * B1 >= 2147483648)
        or (
            (N + 4 * BN) * BS0 + ((2 * K + 4096) // 32 + 1) * BS1 >= 2147483648
        )
    )
    if tl.load(SAFE) == 0:
        for index in range(
            (
                tl.cast(tl.program_id(0), tl.int64)
                if ADDRESS64
                else tl.program_id(0)
            ),
            M * N,
            tl.num_programs(0),
        ):
            row = index // N
            col = index % N
            acc = tl.full((), 0.0, tl.float32)
            for block in range(tl.cdiv(K, 32)):
                ks = (
                    tl.cast(block, tl.int64) if ADDRESS64 else block
                ) * 32 + (
                    tl.cast(tl.arange(0, 32), tl.int64)
                    if ADDRESS64
                    else tl.arange(0, 32)
                )
                ua = tl.load(A + row * A0 + ks * A1, ks < K, 0)
                sa = tl.load(AS + row * AS0 + ks // 32 * AS1, ks < K, 127)
                ub = tl.load(B + col * B0 + ks * B1, ks < K, 0)
                sb = tl.load(BS + col * BS0 + ks // 32 * BS1, ks < K, 127)
                value = _scaled_product(ua, sa, ub, sb)
                acc += tl.sum(value, 0)
            tl.store(Output + row * N + col, acc)


def _fused_scalar880(a, a_scale, b, b_scale, output_dtype):
    m, k = a.shape
    n = b.shape[0]
    out = a.new_empty((m, n), dtype=output_dtype)
    if m == 0 or n == 0:
        return out
    if k == 0:
        (
            _zero_bounded528 if (m * n + 1024 - 1) // 1024 > 32768 else _zero
        ).run(
            out,
            m * n,
            1024,
            num_warps=4,
            grid=(min(32768, (m * n + 1024 - 1) // 1024),),
            warmup=False,
        )
        return out
    if not a.is_contiguous():
        ac = a.new_empty((m, k), dtype=torch.int32)
        _copy_strided_bytes.run(
            a.view(torch.int8),
            ac,
            m,
            k,
            a.stride(0),
            a.stride(1),
            256,
            num_warps=1,
            grid=(min(32768, (m * k + 256 - 1) // 256),),
            warmup=False,
        )
        a = ac
    if not b.is_contiguous():
        bc = b.new_empty((n, k), dtype=torch.int32)
        _copy_strided_bytes.run(
            b.view(torch.int8),
            bc,
            n,
            k,
            b.stride(0),
            b.stride(1),
            256,
            num_warps=1,
            grid=(min(32768, (n * k + 256 - 1) // 256),),
            warmup=False,
        )
        b = bc
    safe = a.new_empty((), dtype=torch.int32)
    a_ptr880 = a.data_ptr()
    as_ptr880 = a_scale.data_ptr()
    b_ptr880 = b.data_ptr()
    bs_ptr880 = b_scale.data_ptr()
    out_ptr880 = out.data_ptr()
    safe_ptr880 = safe.data_ptr()
    type_a880 = tl.int32 if a.dtype == torch.int32 else tl.int8
    type_b880 = tl.int32 if b.dtype == torch.int32 else tl.int8
    a_ptr880_aligned = a_ptr880 % 16 == 0
    as_ptr880_aligned = as_ptr880 % 16 == 0
    b_ptr880_aligned = b_ptr880 % 16 == 0
    bs_ptr880_aligned = bs_ptr880 % 16 == 0
    out_ptr880_aligned = out_ptr880 % 16 == 0
    safe_ptr880_aligned = safe_ptr880 % 16 == 0
    _small_flags_scalar880.run(
        as_ptr880,
        bs_ptr880,
        safe_ptr880,
        m,
        n,
        k // 32,
        a_scale.stride(0),
        a_scale.stride(1),
        b_scale.stride(0),
        b_scale.stride(1),
        (
            0
            if max(m, n) * (k // 32) == 0
            else 1 << (max(m, n) * (k // 32) - 1).bit_length()
        ),
        (as_ptr880_aligned, bs_ptr880_aligned, safe_ptr880_aligned),
        num_warps=4,
        grid=(1,),
        warmup=False,
    )
    _fused_small_scalar880.run(
        a_ptr880,
        as_ptr880,
        b_ptr880,
        bs_ptr880,
        out_ptr880,
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
        32,
        (
            a_ptr880_aligned,
            as_ptr880_aligned,
            b_ptr880_aligned,
            bs_ptr880_aligned,
            out_ptr880_aligned,
        ),
        type_a880,
        type_b880,
        num_warps=1,
        num_stages=1,
        grid=((m + 16 - 1) // 16 * ((n + 32 - 1) // 32),),
        warmup=False,
    )
    _range_guarded_scalar880.run(
        a_ptr880,
        as_ptr880,
        b_ptr880,
        bs_ptr880,
        out_ptr880,
        safe_ptr880,
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
        32,
        (
            a_ptr880_aligned,
            as_ptr880_aligned,
            b_ptr880_aligned,
            bs_ptr880_aligned,
            out_ptr880_aligned,
            safe_ptr880_aligned,
        ),
        type_a880,
        type_b880,
        num_warps=1,
        num_stages=1,
        grid=(min(128, m * n),),
        warmup=False,
    )
    return out


@triton.jit(do_not_specialize=["A", "S", "D"])
def _decode_linear_scalar881(
    A: tl.uint64,
    S: tl.uint64,
    D: tl.uint64,
    COUNT: tl.constexpr,
    B: tl.constexpr,
    ALIGN881: tl.constexpr,
):
    A = A.to(tl.pointer_type(tl.int16))
    if ALIGN881[0]:
        A = tl.multiple_of(A, 16)
    S = S.to(tl.pointer_type(tl.int8))
    if ALIGN881[1]:
        S = tl.multiple_of(S, 16)
    D = D.to(tl.pointer_type(tl.int32))
    if ALIGN881[2]:
        D = tl.multiple_of(D, 16)
    _decode_linear(A, S, D, COUNT, B)


@triton.jit(do_not_specialize=["A", "S", "D"])
def _decode_linear_bounded528_scalar881(
    A: tl.uint64,
    S: tl.uint64,
    D: tl.uint64,
    COUNT: tl.constexpr,
    B: tl.constexpr,
    ALIGN881: tl.constexpr,
):
    A = A.to(tl.pointer_type(tl.int16))
    if ALIGN881[0]:
        A = tl.multiple_of(A, 16)
    S = S.to(tl.pointer_type(tl.int8))
    if ALIGN881[1]:
        S = tl.multiple_of(S, 16)
    D = D.to(tl.pointer_type(tl.int32))
    if ALIGN881[2]:
        D = tl.multiple_of(D, 16)
    _decode_linear_bounded528(A, S, D, COUNT, B)


@triton.jit(do_not_specialize=["A", "S", "D"])
def _decode_linear_packedmask340_scalar881(
    A: tl.uint64,
    S: tl.uint64,
    D: tl.uint64,
    COUNT: tl.constexpr,
    B: tl.constexpr,
    ALIGN881: tl.constexpr,
):
    A = A.to(tl.pointer_type(tl.int16))
    if ALIGN881[0]:
        A = tl.multiple_of(A, 16)
    S = S.to(tl.pointer_type(tl.int8))
    if ALIGN881[1]:
        S = tl.multiple_of(S, 16)
    D = D.to(tl.pointer_type(tl.int32))
    if ALIGN881[2]:
        D = tl.multiple_of(D, 16)
    _decode_linear_packedmask340(A, S, D, COUNT, B)


@triton.jit(do_not_specialize=["A", "S", "D"])
def _decode_linear_packedmask340_bounded528_scalar881(
    A: tl.uint64,
    S: tl.uint64,
    D: tl.uint64,
    COUNT: tl.constexpr,
    B: tl.constexpr,
    ALIGN881: tl.constexpr,
):
    A = A.to(tl.pointer_type(tl.int16))
    if ALIGN881[0]:
        A = tl.multiple_of(A, 16)
    S = S.to(tl.pointer_type(tl.int8))
    if ALIGN881[1]:
        S = tl.multiple_of(S, 16)
    D = D.to(tl.pointer_type(tl.int32))
    if ALIGN881[2]:
        D = tl.multiple_of(D, 16)
    _decode_linear_packedmask340_bounded528(A, S, D, COUNT, B)


def output_type883(dtype):
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
def _plain_scalar883(
    A: tl.uint64,
    B: tl.uint64,
    Output: tl.uint64,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    META883: tl.constexpr,
):
    A = A.to(tl.pointer_type(tl.bfloat16))
    if META883[0]:
        A = tl.multiple_of(A, 16)
    B = B.to(tl.pointer_type(tl.bfloat16))
    if META883[1]:
        B = tl.multiple_of(B, 16)
    Output = Output.to(tl.pointer_type(META883[-1]))
    if META883[2]:
        Output = tl.multiple_of(Output, 16)
    _plain(A, B, Output, M, N, K, BM, BN, BK)


@triton.jit(do_not_specialize=["A", "B", "Output"])
def _plain_full_scalar883(
    A: "*bf16",
    B: "*bf16",
    Output: tl.uint64,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    META883: tl.constexpr,
):
    if META883[0]:
        A = tl.multiple_of(A, 16)
    if META883[1]:
        B = tl.multiple_of(B, 16)
    Output = Output.to(tl.pointer_type(META883[-1]))
    if META883[2]:
        Output = tl.multiple_of(Output, 16)
    _plain_full(A, B, Output, M, N, K, BM, BN, BK)


@triton.jit(do_not_specialize=["A", "B", "Output"])
def _plain_bounded528_scalar883(
    A: tl.uint64,
    B: tl.uint64,
    Output: tl.uint64,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    META883: tl.constexpr,
):
    A = A.to(tl.pointer_type(tl.bfloat16))
    if META883[0]:
        A = tl.multiple_of(A, 16)
    B = B.to(tl.pointer_type(tl.bfloat16))
    if META883[1]:
        B = tl.multiple_of(B, 16)
    Output = Output.to(tl.pointer_type(META883[-1]))
    if META883[2]:
        Output = tl.multiple_of(Output, 16)
    _plain_bounded528(A, B, Output, M, N, K, BM, BN, BK)


@triton.jit(do_not_specialize=["A", "B", "Output"])
def _plain_full_bounded528_scalar883(
    A: "*bf16",
    B: "*bf16",
    Output: tl.uint64,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    META883: tl.constexpr,
):
    if META883[0]:
        A = tl.multiple_of(A, 16)
    if META883[1]:
        B = tl.multiple_of(B, 16)
    Output = Output.to(tl.pointer_type(META883[-1]))
    if META883[2]:
        Output = tl.multiple_of(Output, 16)
    _plain_full_bounded528(A, B, Output, M, N, K, BM, BN, BK)


def _original883(a, a_scale, b, b_scale, output_dtype):
    m, k = a.shape
    n = b.shape[0]
    out = a.new_empty((m, n), dtype=output_dtype)
    if m == 0 or n == 0:
        return out
    if k == 0:
        (
            _zero_bounded528 if (m * n + 1024 - 1) // 1024 > 32768 else _zero
        ).run(
            out,
            m * n,
            1024,
            num_warps=4,
            grid=(min(32768, (m * n + 1024 - 1) // 1024),),
            warmup=False,
        )
        return out
    if not a.is_contiguous():
        ac = a.new_empty((m, k), dtype=torch.int32)
        _copy_strided_bytes.run(
            a.view(torch.int8),
            ac,
            m,
            k,
            a.stride(0),
            a.stride(1),
            256,
            num_warps=1,
            grid=(min(32768, (m * k + 256 - 1) // 256),),
            warmup=False,
        )
        a = ac
    if not b.is_contiguous():
        bc = b.new_empty((n, k), dtype=torch.int32)
        _copy_strided_bytes.run(
            b.view(torch.int8),
            bc,
            n,
            k,
            b.stride(0),
            b.stride(1),
            256,
            num_warps=1,
            grid=(min(32768, (n * k + 256 - 1) // 256),),
            warmup=False,
        )
        b = bc
    output_t883 = (
        tl.float16
        if output_dtype == torch.float16
        else (
            tl.bfloat16
            if output_dtype == torch.bfloat16
            else (
                tl.float32
                if output_dtype == torch.float32
                else output_type883(output_dtype)
            )
        )
    )
    ad = a.new_empty((m, k), dtype=torch.bfloat16)
    bd = a.new_empty((n, k), dtype=torch.bfloat16)
    a_scale_bytes828 = a_scale
    b_scale_bytes828 = b_scale
    contiguous_scales = a_scale.is_contiguous() and b_scale.is_contiguous()
    ca = max(1, (m * (k // 32) + 4096 - 1) // 4096) if contiguous_scales else m
    cb = max(1, (n * (k // 32) + 4096 - 1) // 4096) if contiguous_scales else n
    small_scales = max(m, n) * (k // 32) <= 8192
    if not small_scales:
        fa = a.new_empty((ca + cb,), dtype=torch.int32)
        fb = fa
    if (
        k >= 2048
        and contiguous_scales
        and (a.dtype != torch.int32)
        and (b.dtype != torch.int32)
        and a.is_contiguous()
        and b.is_contiguous()
        and (a.data_ptr() % 2 == 0)
        and (b.data_ptr() % 2 == 0)
    ):
        a_ptr881 = a.data_ptr()
        as_ptr881 = a_scale.data_ptr()
        ad_ptr881 = ad.data_ptr()
        b_ptr881 = b.data_ptr()
        bs_ptr881 = b_scale.data_ptr()
        bd_ptr881 = bd.data_ptr()
        align_a881 = (
            a_ptr881 % 16 == 0,
            as_ptr881 % 16 == 0,
            ad_ptr881 % 16 == 0,
        )
        align_b881 = (
            b_ptr881 % 16 == 0,
            bs_ptr881 % 16 == 0,
            bd_ptr881 % 16 == 0,
        )
        (
            (
                _decode_linear_packedmask340_bounded528_scalar881
                if m >= 4096 and n > 4096
                else _decode_linear_bounded528_scalar881
            )
            if (m * k + 8192 - 1) // 8192 > 32768
            else (
                _decode_linear_packedmask340_scalar881
                if m >= 4096 and n > 4096
                else _decode_linear_scalar881
            )
        ).run(
            a_ptr881,
            as_ptr881,
            ad_ptr881,
            m * k,
            4096,
            align_a881,
            num_warps=4,
            grid=(min(32768, (m * k + 8192 - 1) // 8192),),
            warmup=False,
        )
        (
            (
                _decode_linear_packedmask340_bounded528_scalar881
                if m >= 4096 and n > 4096
                else _decode_linear_bounded528_scalar881
            )
            if (n * k + 8192 - 1) // 8192 > 32768
            else (
                _decode_linear_packedmask340_scalar881
                if m >= 4096 and n > 4096
                else _decode_linear_scalar881
            )
        ).run(
            b_ptr881,
            bs_ptr881,
            bd_ptr881,
            n * k,
            4096,
            align_b881,
            num_warps=4,
            grid=(min(32768, (n * k + 8192 - 1) // 8192),),
            warmup=False,
        )
    else:
        (
            _dequant_rows_joint528
            if k > 4096 or m + n > 32768
            else _dequant_rows_joint358
        ).run(
            a if a.dtype == torch.int32 else a.view(torch.int8),
            a_scale_bytes828,
            ad,
            b if b.dtype == torch.int32 else b.view(torch.int8),
            b_scale_bytes828,
            bd,
            m,
            k,
            a.stride(0),
            a.stride(1),
            a_scale.stride(0),
            a_scale.stride(1),
            b.stride(0),
            b.stride(1),
            b_scale.stride(0),
            b_scale.stride(1),
            min(4096, max(32, 0 if k == 0 else 1 << (k - 1).bit_length())),
            num_warps=4,
            grid=(min(32768, (m + n) * ((k + 4096 - 1) // 4096)),),
            warmup=False,
            N=n,
        )
    if small_scales or max(ca, cb) > 4096:
        safe = a.new_empty((), dtype=torch.int32)
    if small_scales:
        _small_flags.run(
            a_scale_bytes828,
            b_scale_bytes828,
            safe,
            m,
            n,
            k // 32,
            a_scale.stride(0),
            a_scale.stride(1),
            b_scale.stride(0),
            b_scale.stride(1),
            (
                0
                if max(m, n) * (k // 32) == 0
                else 1 << (max(m, n) * (k // 32) - 1).bit_length()
            ),
            num_warps=4,
            grid=(1,),
            warmup=False,
        )
    elif contiguous_scales:
        (
            _scale_both_binary_bounded528
            if ca + cb > 32768
            else _scale_both_binary
        ).run(
            a_scale_bytes828,
            fa,
            b_scale_bytes828,
            fb,
            m * (k // 32),
            n * (k // 32),
            4096,
            num_warps=4,
            grid=(min(32768, ca + cb),),
            warmup=False,
        )
    else:
        (
            _scale_flags_bounded528
            if m > 32768 or k // 32 > 4096
            else _scale_flags
        ).run(
            a_scale_bytes828,
            fa,
            k // 32,
            a_scale.stride(0),
            a_scale.stride(1),
            min(
                4096,
                max(1, 0 if k // 32 == 0 else 1 << (k // 32 - 1).bit_length()),
            ),
            num_warps=1,
            grid=(min(32768, m),),
            warmup=False,
            R=a_scale.shape[0],
        )
        (
            _scale_flags_bounded528
            if n > 32768 or k // 32 > 4096
            else _scale_flags
        ).run(
            b_scale_bytes828,
            fb,
            k // 32,
            b_scale.stride(0),
            b_scale.stride(1),
            min(
                4096,
                max(1, 0 if k // 32 == 0 else 1 << (k // 32 - 1).bit_length()),
            ),
            num_warps=1,
            grid=(min(32768, n),),
            warmup=False,
            R=b_scale.shape[0],
            F_OFFSET=ca,
        )
    bm = min(64, max(16, 2 ** (m.bit_length() - 1)))
    bn = min(128, max(16, 2 ** (n.bit_length() - 1)))
    bk = min(64, max(32, 2 ** (k.bit_length() - 1)))
    if m >= 512 and k >= 128:
        bk = 128
        if m >= 2048:
            bm = 128
    if m >= 512 and n >= 256 and (k >= 1024):
        bn *= 2
        if m >= 2048:
            bk = 256
    ptrs883 = (ad.data_ptr(), bd.data_ptr(), out.data_ptr())
    (
        (
            _plain_full_bounded528_scalar883
            if m >= 2048
            else _plain_bounded528_scalar883
        )
        if (m + bm - 1) // bm * ((n + bn - 1) // bn) > 32768
        else _plain_full_scalar883 if m >= 2048 else _plain_scalar883
    ).run(
        ptrs883[0],
        ptrs883[1],
        ptrs883[2],
        m,
        n,
        k,
        bm,
        bn,
        bk,
        (
            ptrs883[0] % 16 == 0,
            ptrs883[1] % 16 == 0,
            ptrs883[2] % 16 == 0,
            output_t883,
        ),
        num_warps=1,
        num_stages=2,
        grid=(min(32768, (m + bm - 1) // bm * ((n + bn - 1) // bn)),),
        warmup=False,
    )
    if max(m, n) * (k // 32) > 8192:
        if max(ca, cb) <= 4096:
            repair_ptrs904 = (
                a.data_ptr(),
                a_scale_bytes828.data_ptr(),
                b.data_ptr(),
                b_scale_bytes828.data_ptr(),
                out.data_ptr(),
                fa.data_ptr(),
                fb.data_ptr(),
            )
            _range_guarded_flags_pointer904.run(
                repair_ptrs904[0],
                repair_ptrs904[1],
                repair_ptrs904[2],
                repair_ptrs904[3],
                repair_ptrs904[4],
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
                32,
                repair_ptrs904[5],
                repair_ptrs904[6],
                ca,
                cb,
                0 if max(ca, cb) == 0 else 1 << (max(ca, cb) - 1).bit_length(),
                (
                    (
                        repair_ptrs904[0] % 16 == 0,
                        repair_ptrs904[1] % 16 == 0,
                        repair_ptrs904[2] % 16 == 0,
                        repair_ptrs904[3] % 16 == 0,
                        repair_ptrs904[4] % 16 == 0,
                        repair_ptrs904[5] % 16 == 0,
                        repair_ptrs904[6] % 16 == 0,
                    ),
                    tl.int32 if a.dtype == torch.int32 else tl.int8,
                    tl.int32 if b.dtype == torch.int32 else tl.int8,
                    output_t883,
                ),
                num_warps=1,
                num_stages=1,
                grid=(min(128, m * n),),
                warmup=False,
            )
        else:
            _all_safe_bounded528.run(
                fa,
                fb,
                safe,
                ca,
                cb,
                4096,
                num_warps=4,
                grid=(1,),
                warmup=False,
            )
            _range_guarded.run(
                a if a.dtype == torch.int32 else a.view(torch.int8),
                a_scale_bytes828,
                b if b.dtype == torch.int32 else b.view(torch.int8),
                b_scale_bytes828,
                out,
                safe,
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
                32,
                num_warps=1,
                num_stages=1,
                grid=(min(128, m * n),),
                warmup=False,
            )
    else:
        _range_guarded.run(
            a if a.dtype == torch.int32 else a.view(torch.int8),
            a_scale_bytes828,
            b if b.dtype == torch.int32 else b.view(torch.int8),
            b_scale_bytes828,
            out,
            safe,
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
            32,
            num_warps=1,
            num_stages=1,
            grid=(min(128, m * n),),
            warmup=False,
        )
    return out


def mxfp8_block_scaled_matmul(a, a_scale, b, b_scale, output_dtype):
    m, k = a.shape
    n = b.shape[0]
    if (
        output_dtype == torch.bfloat16
        and (0 < m <= 256 and 0 < n <= 512 and (128 <= k <= 256))
    ) and (m + 16 - 1) // 16 * ((n + 32 - 1) // 32) <= 128:
        return _fused_scalar880(a, a_scale, b, b_scale, output_dtype)
    return _original883(a, a_scale, b, b_scale, output_dtype)


@triton.jit(do_not_specialize=["SA", "FA", "SB", "FB"])
def _scale_both_binary_pointer904(
    SA: "*i8",
    FA: "*i32",
    SB: "*i8",
    FB: "*i32",
    COUNT_A: tl.constexpr,
    COUNT_B: tl.constexpr,
    B: tl.constexpr,
    META904: tl.constexpr,
):
    if META904[0][0]:
        SA = tl.multiple_of(SA, 16)
    if META904[0][1]:
        FA = tl.multiple_of(FA, 16)
    if META904[0][2]:
        SB = tl.multiple_of(SB, 16)
    if META904[0][3]:
        FB = tl.multiple_of(FB, 16)
    _scale_both_binary(SA, FA, SB, FB, COUNT_A, COUNT_B, B)


@triton.jit(do_not_specialize=["SA", "FA", "SB", "FB"])
def _scale_both_binary_bounded528_pointer904(
    SA: "*i8",
    FA: "*i32",
    SB: "*i8",
    FB: "*i32",
    COUNT_A: tl.constexpr,
    COUNT_B: tl.constexpr,
    B: tl.constexpr,
    META904: tl.constexpr,
):
    if META904[0][0]:
        SA = tl.multiple_of(SA, 16)
    if META904[0][1]:
        FA = tl.multiple_of(FA, 16)
    if META904[0][2]:
        SB = tl.multiple_of(SB, 16)
    if META904[0][3]:
        FB = tl.multiple_of(FB, 16)
    _scale_both_binary_bounded528(SA, FA, SB, FB, COUNT_A, COUNT_B, B)


@triton.jit(do_not_specialize=["A", "AS", "B", "BS", "Output", "FA", "FB"])
def _range_guarded_flags_pointer904(
    A: tl.uint64,
    AS: "*i8",
    B: tl.uint64,
    BS: "*i8",
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
    BN: tl.constexpr,
    FA: "*i32",
    FB: "*i32",
    CA: tl.constexpr,
    CB: tl.constexpr,
    FLAGS_B: tl.constexpr,
    META904: tl.constexpr,
):
    A = A.to(tl.pointer_type(META904[1]))
    if META904[0][0]:
        A = tl.multiple_of(A, 16)
    if META904[0][1]:
        AS = tl.multiple_of(AS, 16)
    B = B.to(tl.pointer_type(META904[2]))
    if META904[0][2]:
        B = tl.multiple_of(B, 16)
    if META904[0][3]:
        BS = tl.multiple_of(BS, 16)
    Output = Output.to(tl.pointer_type(META904[3]))
    if META904[0][4]:
        Output = tl.multiple_of(Output, 16)
    if META904[0][5]:
        FA = tl.multiple_of(FA, 16)
    if META904[0][6]:
        FB = tl.multiple_of(FB, 16)
    _range_guarded_flags(
        A,
        AS,
        B,
        BS,
        Output,
        M,
        N,
        K,
        A0,
        A1,
        AS0,
        AS1,
        B0,
        B1,
        BS0,
        BS1,
        BN,
        FA,
        FB,
        CA,
        CB,
        FLAGS_B,
    )


__all__ = ["mxfp8_block_scaled_matmul"]
