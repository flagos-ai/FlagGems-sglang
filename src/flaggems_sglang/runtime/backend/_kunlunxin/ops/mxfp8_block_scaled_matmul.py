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
import triton.experimental.tle as tle
import triton.language as tl


@triton.jit
def _zero(Output, META: tl.constexpr):
    COUNT: tl.constexpr = META[0]
    B: tl.constexpr = META[1]
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
def _dequant_rows(A, S, D, META: tl.constexpr):
    K: tl.constexpr = META[0]
    A0: tl.constexpr = META[1]
    A1: tl.constexpr = META[2]
    S0: tl.constexpr = META[3]
    S1: tl.constexpr = META[4]
    B: tl.constexpr = META[5]
    R: tl.constexpr = META[6]
    ADDRESS64: tl.constexpr = (
        (R + 1) * A0 + (2 * K + B + 1024) * A1 >= 2147483648
        or (R + 1) * S0 + ((2 * K + B + 1024) // 32 + 1) * S1 >= 2147483648
        or (R + 1) * K + 2 * K + B + 1024 >= 2147483648
    )
    row = (
        tl.cast(tl.program_id(0), tl.int64) if ADDRESS64 else tl.program_id(0)
    )
    cols = tl.cast(tl.arange(0, B), tl.int64) if ADDRESS64 else tl.arange(0, B)
    u = tl.load(A + row * A0 + cols * A1, cols < K, 0)
    s = (
        tl.load(S + row * S0 + cols // 32 * S1, cols < K, 127).to(tl.int32)
        & 255
    )
    value = _decode(u, s)
    tl.store(D + row * K + cols, value, cols < K)


@triton.jit
def _decode_linear_full(A, S, D, META: tl.constexpr):
    COUNT: tl.constexpr = META[0]
    B: tl.constexpr = META[1]
    ADDRESS64: tl.constexpr = COUNT + 2 * B >= 2147483648
    pairs = (
        tl.cast(tl.program_id(0), tl.int64) if ADDRESS64 else tl.program_id(0)
    ) * B + (
        tl.cast(tl.arange(0, B), tl.int64) if ADDRESS64 else tl.arange(0, B)
    )
    packed = tl.load(A.to(tl.pointer_type(tl.uint16)) + pairs).to(tl.int32)
    s = tl.load(S + pairs // 16).to(tl.int32) & 255
    if D.dtype.element_ty == tl.float16:
        low = (
            _decode(packed & 255, s)
            .to(tl.float16)
            .to(tl.uint16, bitcast=True)
            .to(tl.uint32)
        )
        high = (
            _decode(packed >> 8, s)
            .to(tl.float16)
            .to(tl.uint16, bitcast=True)
            .to(tl.uint32)
            << 16
        )
    else:
        low = _decode(packed & 255, s).to(tl.uint32, bitcast=True) >> 16
        high = _decode(packed >> 8, s).to(tl.uint32, bitcast=True) & 4294901760
    tl.store(D.to(tl.pointer_type(tl.uint32)) + pairs, low | high)


@triton.jit
def _decode_linear(A, S, D, META: tl.constexpr):
    COUNT: tl.constexpr = META[0]
    B: tl.constexpr = META[1]
    ADDRESS64: tl.constexpr = COUNT + 2 * B >= 2147483648
    pairs = (
        tl.cast(tl.program_id(0), tl.int64) if ADDRESS64 else tl.program_id(0)
    ) * B + (
        tl.cast(tl.arange(0, B), tl.int64) if ADDRESS64 else tl.arange(0, B)
    )
    packed = tl.load(
        A.to(tl.pointer_type(tl.uint16)) + pairs, pairs < COUNT // 2, 0
    ).to(tl.int32)
    s = tl.load(S + pairs // 16, pairs < COUNT // 2, 127).to(tl.int32) & 255
    if D.dtype.element_ty == tl.float16:
        low = (
            _decode(packed & 255, s)
            .to(tl.float16)
            .to(tl.uint16, bitcast=True)
            .to(tl.uint32)
        )
        high = (
            _decode(packed >> 8, s)
            .to(tl.float16)
            .to(tl.uint16, bitcast=True)
            .to(tl.uint32)
            << 16
        )
    else:
        low = _decode(packed & 255, s).to(tl.uint32, bitcast=True) >> 16
        high = _decode(packed >> 8, s).to(tl.uint32, bitcast=True) & 4294901760
    tl.store(
        D.to(tl.pointer_type(tl.uint32)) + pairs,
        low | high,
        pairs < COUNT // 2,
    )


@triton.jit
def _scale_flags(S, F, META: tl.constexpr):
    G: tl.constexpr = META[0]
    S0: tl.constexpr = META[1]
    S1: tl.constexpr = META[2]
    B: tl.constexpr = META[3]
    R: tl.constexpr = META[4]
    ADDRESS64: tl.constexpr = (R + 1) * S0 + (
        2 * G + B
    ) * S1 >= 2147483648 or R + B >= 2147483648
    row = (
        tl.cast(tl.program_id(0), tl.int64) if ADDRESS64 else tl.program_id(0)
    )
    cols = tl.cast(tl.arange(0, B), tl.int64) if ADDRESS64 else tl.arange(0, B)
    if G == B:
        s = tl.load(S + row * S0 + cols * S1).to(tl.int32) & 255
    else:
        s = (
            tl.load(S + row * S0 + tl.minimum(cols, G - 1) * S1).to(tl.int32)
            & 255
        )
    safe = tl.min(
        tl.where(
            (s >= 112) & (s <= 134), 2, ((s >= 10) & (s <= 246)).to(tl.int32)
        ),
        0,
    )
    tl.store(F + row, safe)


@triton.jit
def _scale_chunks(S, F, COUNT: tl.constexpr, B: tl.constexpr):
    ADDRESS64: tl.constexpr = COUNT + 2 * B >= 2147483648
    pid = (
        tl.cast(tl.program_id(0), tl.int64) if ADDRESS64 else tl.program_id(0)
    )
    cols = pid * B + (
        tl.cast(tl.arange(0, B), tl.int64) if ADDRESS64 else tl.arange(0, B)
    )
    s = tl.load(S + cols, cols < COUNT, 127).to(tl.int32) & 255
    safe = tl.min(
        tl.where(
            (s >= 112) & (s <= 134), 2, ((s >= 10) & (s <= 246)).to(tl.int32)
        ),
        0,
    )
    tl.store(F + pid, safe)


@triton.jit
def _plain(A, B, Output, META: tl.constexpr):
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
        acc = tl.dot(av, bv, acc, input_precision="ieee")
    if M % BM == 0 and N % BN == 0:
        tl.store(Output + rm[:, None] * N + rn[None, :], acc)
    else:
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
def _range_matmul(A, AS, B, BS, Output, OFFSET, META: tl.constexpr):
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
    BN: tl.constexpr = META[11]
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
    OFFSET = tl.cast(OFFSET, tl.int64) if ADDRESS64 else OFFSET
    row = (
        (
            tl.cast(tl.program_id(0), tl.int64)
            if ADDRESS64
            else tl.program_id(0)
        )
        + OFFSET
    ) // N
    col = (
        (
            tl.cast(tl.program_id(0), tl.int64)
            if ADDRESS64
            else tl.program_id(0)
        )
        + OFFSET
    ) % N
    acc = tl.full((), 0.0, tl.float32)
    for block in range(tl.cdiv(K, 32)):
        ks = (tl.cast(block, tl.int64) if ADDRESS64 else block) * 32 + (
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
def _decode_full_block(A, S, D, pid, B: tl.constexpr):
    pairs = pid * B + tl.arange(0, B)
    packed = tl.load(A.to(tl.pointer_type(tl.uint16)) + pairs).to(tl.int32)
    s = tl.load(S + pairs // 16).to(tl.int32) & 255
    if D.dtype.element_ty == tl.float16:
        low = (
            _decode_halfproduct(packed & 255, s)
            .to(tl.float16)
            .to(tl.uint16, bitcast=True)
            .to(tl.uint32)
        )
        high = (
            _decode_halfproduct(packed >> 8, s)
            .to(tl.float16)
            .to(tl.uint16, bitcast=True)
            .to(tl.uint32)
            << 16
        )
    else:
        low = (
            _decode_halfbase(packed & 255, s).to(tl.uint32, bitcast=True) >> 16
        )
        high = (
            _decode_halfbase(packed >> 8, s).to(tl.uint32, bitcast=True)
            & 4294901760
        )
    tl.store(D.to(tl.pointer_type(tl.uint32)) + pairs, low | high)


@triton.jit
def _decode_both_full(
    A, AS, AD, C, CS, CD, CA: tl.constexpr, B: tl.constexpr, CB: tl.constexpr
):
    ADDRESS64: tl.constexpr = 2 * CA * B + CB + 2 * B >= 2147483648
    pid = (
        tl.cast(tl.program_id(0), tl.int64) if ADDRESS64 else tl.program_id(0)
    )
    is_a = pid < CA
    src = tl.where(is_a, A, C)
    scales = tl.where(is_a, AS, CS)
    dst = tl.where(is_a, AD, CD)
    block = tl.where(is_a, pid, pid - CA)
    _decode_full_block(src, scales, dst, block, B)


@triton.jit
def _scale_block(S, F, pid, COUNT: tl.constexpr, B: tl.constexpr):
    ADDRESS64: tl.constexpr = COUNT + 2 * B >= 2147483648
    pid = tl.cast(pid, tl.int64) if ADDRESS64 else pid
    cols = pid * B + (
        tl.cast(tl.arange(0, B), tl.int64) if ADDRESS64 else tl.arange(0, B)
    )
    s = tl.load(S + cols, cols < COUNT, 127).to(tl.int32) & 255
    safe = tl.min(
        tl.where(
            (s >= 112) & (s <= 134), 2, ((s >= 10) & (s <= 246)).to(tl.int32)
        ),
        0,
    )
    tl.store(F + pid, safe)


@triton.jit
def _scale_both(
    SA,
    FA,
    SB,
    FB,
    COUNT_A: tl.constexpr,
    COUNT_B: tl.constexpr,
    B: tl.constexpr,
):
    ADDRESS64: tl.constexpr = COUNT_A + COUNT_B + 2 * B >= 2147483648
    pid = (
        tl.cast(tl.program_id(0), tl.int64) if ADDRESS64 else tl.program_id(0)
    )
    ca = tl.cdiv(COUNT_A, B)
    is_a = pid < ca
    src = tl.where(is_a, SA, SB)
    dst = tl.where(is_a, FA, FB)
    block = tl.where(is_a, pid, pid - ca)
    count = tl.where(is_a, COUNT_A, COUNT_B)
    cols = block * B + (
        tl.cast(tl.arange(0, B), tl.int64) if ADDRESS64 else tl.arange(0, B)
    )
    s = tl.load(src + cols, cols < count, 127).to(tl.int32) & 255
    safe = tl.min(
        tl.where(
            (s >= 112) & (s <= 134), 2, ((s >= 10) & (s <= 246)).to(tl.int32)
        ),
        0,
    )
    tl.store(dst + block, safe)


@triton.jit
def _decode_halfbase(u, s):
    u = u.to(tl.int32) & 255
    s = s.to(tl.int32) & 255
    half_bits = (u & 128) << 8 | (u & 127) << 7
    value = half_bits.to(tl.uint16).to(tl.float16, bitcast=True).to(tl.float32)
    scale = (s + 8 << 23).to(tl.float32, bitcast=True)
    return tl.where(u & 127 == 127, float("nan"), value * scale)


@triton.jit
def _decode_halfproduct(u, s):
    u = u.to(tl.int32) & 255
    s = s.to(tl.int32) & 255
    half_bits = (u & 128) << 8 | (u & 127) << 7
    value = half_bits.to(tl.uint16).to(tl.float16, bitcast=True)
    scale = (s - 104 << 10).to(tl.uint16).to(tl.float16, bitcast=True)
    return tl.where(
        u & 127 == 127, tl.full((), float("nan"), tl.float16), value * scale
    )


@triton.jit
def _decode_full_bfmask335(A, S, D, pid, B: tl.constexpr):
    pairs = pid * B + tl.arange(0, B)
    packed = tl.load(A.to(tl.pointer_type(tl.uint16)) + pairs).to(tl.int32)
    s = tl.load(S.to(tl.pointer_type(tl.uint8)) + pairs // 16).to(tl.int32)
    lowmag = packed & 127
    highmag = packed & 32512
    lowbits = (packed & 128) << 8 | lowmag << 7
    highbits = packed & 32768 | highmag >> 1
    lowvalue = lowbits.to(tl.uint16).to(tl.float16, bitcast=True)
    highvalue = highbits.to(tl.uint16).to(tl.float16, bitcast=True)
    scale = (s + 8 << 23).to(tl.float32, bitcast=True)
    lv = tl.where(lowmag == 127, float("nan"), lowvalue.to(tl.float32) * scale)
    hv = tl.where(
        highmag == 32512, float("nan"), highvalue.to(tl.float32) * scale
    )
    low = lv.to(tl.uint32, bitcast=True) >> 16
    high = hv.to(tl.uint32, bitcast=True) & 4294901760
    tl.store(D.to(tl.pointer_type(tl.uint32)) + pairs, low | high)


@triton.jit
def _decode_both_bfmask335(
    A, AS, AD, C, CS, CD, CA: tl.constexpr, B: tl.constexpr, CB: tl.constexpr
):
    ADDRESS64: tl.constexpr = 2 * CA * B + CB + 2 * B >= 2147483648
    pid = (
        tl.cast(tl.program_id(0), tl.int64) if ADDRESS64 else tl.program_id(0)
    )
    is_a = pid < CA
    src = tl.where(is_a, A, C)
    scales = tl.where(is_a, AS, CS)
    dst = tl.where(is_a, AD, CD)
    block = tl.where(is_a, pid, pid - CA)
    _decode_full_bfmask335(src, scales, dst, block, B)


@triton.jit
def _dequant_rows_both354(A, AS, AD, C, CS, CD, META: tl.constexpr):
    M: tl.constexpr = META[0]
    K: tl.constexpr = META[1]
    A0: tl.constexpr = META[2]
    A1: tl.constexpr = META[3]
    AS0: tl.constexpr = META[4]
    AS1: tl.constexpr = META[5]
    C0: tl.constexpr = META[6]
    C1: tl.constexpr = META[7]
    CS0: tl.constexpr = META[8]
    CS1: tl.constexpr = META[9]
    B: tl.constexpr = META[10]
    N: tl.constexpr = META[11]
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
    first = pid < M
    row = tl.where(first, pid, pid - M)
    src = tl.where(first, A, C)
    scales = tl.where(first, AS, CS)
    dst = tl.where(first, AD, CD)
    s0 = tl.where(first, A0, C0)
    s1 = tl.where(first, A1, C1)
    ss0 = tl.where(first, AS0, CS0)
    ss1 = tl.where(first, AS1, CS1)
    cols = tl.cast(tl.arange(0, B), tl.int64) if ADDRESS64 else tl.arange(0, B)
    u = tl.load(src + row * s0 + cols * s1, cols < K, 0)
    s = tl.load(scales + row * ss0 + cols // 32 * ss1, cols < K, 127)
    value = _decode(u, s)
    tl.store(dst + row * K + cols, value, cols < K)


@triton.jit
def _all_safe529(F, Output, META: tl.constexpr):
    COUNT: tl.constexpr = META[0]
    B: tl.constexpr = META[1]
    ADDRESS64: tl.constexpr = COUNT + 2 * B >= 2147483648
    idx = tl.cast(tl.arange(0, B), tl.int64) if ADDRESS64 else tl.arange(0, B)
    flag = tl.load(F + idx, idx < COUNT, 2)
    tl.store(Output, tl.min(flag, 0))


@triton.jit
def _range_chunk563(A, AS, B, BS, Output, OFFSET, META: tl.constexpr):
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
    BN: tl.constexpr = META[11]
    BK: tl.constexpr = META[12]
    CHUNK: tl.constexpr = (
        2048
        if K >= 4096 and K % 2048 == 0
        else (
            1024
            if K >= 4096 and K % 1024 == 0
            else (
                512 if K >= 4096 and K % 512 == 0 else 256 if BK >= 256 else BK
            )
        )
    )
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
    OFFSET = tl.cast(OFFSET, tl.int64) if ADDRESS64 else OFFSET
    row = (
        (
            tl.cast(tl.program_id(0), tl.int64)
            if ADDRESS64
            else tl.program_id(0)
        )
        + OFFSET
    ) // N
    col = (
        (
            tl.cast(tl.program_id(0), tl.int64)
            if ADDRESS64
            else tl.program_id(0)
        )
        + OFFSET
    ) % N
    lanes = (
        tl.cast(tl.arange(0, CHUNK), tl.int64)
        if ADDRESS64
        else tl.arange(0, CHUNK)
    )
    acc = (
        tl.full((CHUNK,), 0.0, tl.float32)
        if K >= 4096
        else tl.full((), 0.0, tl.float32)
    )
    for block in range((K - 1) // CHUNK):
        logical = (
            tl.cast(block, tl.int64) if ADDRESS64 else block
        ) * CHUNK + lanes
        ks = logical
        ua = tl.load(A + row * A0 + ks * A1)
        sa = tl.load(AS + row * AS0 + ks // 32 * AS1)
        ub = tl.load(B + col * B0 + ks * B1)
        sb = tl.load(BS + col * BS0 + ks // 32 * BS1)
        value = _scaled_product(ua, sa, ub, sb)
        acc += value if K >= 4096 else tl.sum(value, 0)
    block = (K - 1) // CHUNK
    logical = (
        tl.cast(block, tl.int64) if ADDRESS64 else block
    ) * CHUNK + lanes
    ks = logical
    if K % CHUNK == 0:
        ua = tl.load(A + row * A0 + ks * A1)
        sa = tl.load(AS + row * AS0 + ks // 32 * AS1)
        ub = tl.load(B + col * B0 + ks * B1)
        sb = tl.load(BS + col * BS0 + ks // 32 * BS1)
    elif K <= CHUNK:
        ua = tl.load(A + row * A0 + ks * A1, logical < K, 0)
        sa = tl.load(AS + row * AS0 + ks // 32 * AS1, logical < K, 0)
        ub = tl.load(B + col * B0 + ks * B1, logical < K, 0)
        sb = tl.load(BS + col * BS0 + ks // 32 * BS1, logical < K, 0)
    else:
        ks = tl.minimum(logical, K - 1)
        ua = tl.load(A + row * A0 + ks * A1)
        sa = tl.load(AS + row * AS0 + ks // 32 * AS1)
        ub = tl.load(B + col * B0 + ks * B1)
        sb = tl.load(BS + col * BS0 + ks // 32 * BS1)
    value = _scaled_product(ua, sa, ub, sb)
    if K % CHUNK != 0:
        value = tl.where(logical < K, value, 0.0)
    acc += value if K >= 4096 else tl.sum(value, 0)
    tl.store(Output + row * N + col, tl.sum(acc, 0) if K >= 4096 else acc)


RAW632 = (
    "typedef signed char __attribute__((address_space(1))) gi8;\n"
    "typedef unsigned char __attribute__((address_space(1))) gu8;"
    "\n"
    "typedef unsigned short __attribute__((address_space(1))) gu1"
    "6;\n"
    "typedef unsigned int u32v __attribute__((ext_vector_type(16)"
    "));\n"
    "typedef unsigned short h32v __attribute__((ext_vector_type(3"
    "2)));\n"
    "typedef float f32v __attribute__((ext_vector_type(16)));\n"
    "typedef _Float16 hf32v __attribute__((ext_vector_type(32)));"
    "\n"
    "\n"
    "__attribute__((device,always_inline)) u32v halfpair638(u32v "
    "value) {\n"
    "    u32v packed = (value & 0xffu) | ((value & 0xff00u) << 8)"
    ";\n"
    "    u32v bits = (packed + (packed & 0x00800080u)) << 7;\n"
    "    return bits;\n"
    "}\n"
    "\n"
    "__attribute__((device,always_inline)) h32v pack644(f32v low,"
    " f32v high) {\n"
    "    u32v a0 = __builtin_bit_cast(u32v, low);\n"
    "    u32v b0 = __builtin_bit_cast(u32v, high);\n"
    "    u32v a1 = __builtin_xpu2_vmerge_l_u(a0, b0);\n"
    "    u32v b1 = __builtin_xpu2_vmerge_h_u(a0, b0);\n"
    "    u32v a2 = __builtin_xpu2_vmerge_l_u(a1, b1);\n"
    "    u32v b2 = __builtin_xpu2_vmerge_h_u(a1, b1);\n"
    "    u32v a3 = __builtin_xpu2_vmerge_l_u(a2, b2);\n"
    "    u32v b3 = __builtin_xpu2_vmerge_h_u(a2, b2);\n"
    "    u32v a4 = __builtin_xpu2_vmerge_l_u(a3, b3);\n"
    "    u32v b4 = __builtin_xpu2_vmerge_h_u(a3, b3);\n"
    "    return __builtin_bit_cast(h32v, (a4 >> 16) | (b4 & 0xfff"
    "f0000u));\n"
    "}\n"
    "\n"
    "__attribute__((device,always_inline)) h32v convert638(u32v w"
    "ord, unsigned int scale) {\n"
    "    hf32v half = __builtin_bit_cast(hf32v, word);\n"
    "    hf32v normalized = half * (_Float16)256;\n"
    "    u32v mag = __builtin_bit_cast(u32v, normalized) & 0x7fff"
    "7fffu;\n"
    "    unsigned int bias = ((scale - 15u) << 7) & 0x7fffu;\n"
    "    u32v bits = ((mag >> 3) & 0x0fff0fffu) + (u32v)(bias * 0"
    "x00010001u);\n"
    "    u32v nonzero = ((mag + 0x7fff7fffu) & 0x80008000u) >> 15"
    ";\n"
    "    bits &= ((nonzero << 16) - nonzero) & 0x7fff7fffu;\n"
    "    u32v nan = ((mag + 0x20802080u) & 0x80008000u) >> 15;\n"
    "    u32v nanmask = (nan << 16) - nan;\n"
    "    bits = (bits & ~nanmask) | (0x7fc07fc0u & nanmask);\n"
    "    return __builtin_bit_cast(h32v, bits | (word & 0x8000800"
    "0u));\n"
    "}\n"
    "\n"
    'extern "C" __attribute__((device)) void raw_decode632(\n'
    "    const gi8* A, const gu8* AS, gu16* AD,\n"
    "    const gi8* C, const gu8* CS, gu16* CD,\n"
    "    long long ca, long long pid) {\n"
    "    unsigned int bytes[128] __attribute__((aligned(64)));\n"
    "    unsigned char scales[16] __attribute__((aligned(64)));\n"
    "    unsigned short values[512] __attribute__((aligned(64)));"
    "\n"
    "    int core = __builtin_xpu_core_id();\n"
    "    const gi8* src = pid < ca ? A : C;\n"
    "    const gu8* scale = pid < ca ? AS : CS;\n"
    "    gu16* dst = pid < ca ? AD : CD;\n"
    "    long long block = pid < ca ? pid : pid - ca;\n"
    "    long long offset = block * 32768 + core * 512;\n"
    "    __builtin_xpu_gm2lm_v3(bytes, src + offset, 0, 512);\n"
    "    __builtin_xpu_gm2lm_v3(scales, scale + offset / 32, 0, 1"
    "6);\n"
    "    __builtin_xpu2_mfence(7);\n"
    "    for (int group = 0; group < 8; ++group) {\n"
    "        u32v packed = *((u32v*)(bytes + group * 16));\n"
    "        u32v first = halfpair638(packed);\n"
    "        u32v second = halfpair638(packed >> 16);\n"
    "        u32v low = __builtin_xpu2_vmerge_l_u(first, second);"
    "\n"
    "        u32v high = __builtin_xpu2_vmerge_h_u(first, second)"
    ";\n"
    "        *((h32v*)(values + group * 64)) = convert638(low, sc"
    "ales[2 * group]);\n"
    "        *((h32v*)(values + group * 64 + 32)) = convert638(hi"
    "gh, scales[2 * group + 1]);\n"
    "    }\n"
    "    __builtin_xpu2_mfence(7);\n"
    "    __builtin_xpu_lm2gm_v3(dst + offset, values, 0, 1024);\n"
    "    __builtin_xpu2_mfence(7);\n"
    "}\n"
)


@tle.raw.dialect("xpu3", source=RAW632)
def raw_decode632(A, AS, AD, C, CS, CD, ca, pid):
    pass


@triton.jit
def _raw_both632(A, AS, AD, C, CS, CD, META: tl.constexpr):
    CA: tl.constexpr = META[0]
    B: tl.constexpr = META[1]

    tl.static_assert(B == 16384)
    tle.raw.call(
        raw_decode632,
        (
            A,
            AS,
            AD,
            C,
            CS,
            CD,
            tl.full((), CA, tl.int64),
            tl.program_id(0).to(tl.int64),
        ),
    )


RAW_FP1670 = (
    "typedef signed char __attribute__((address_space(1))) gi8;\n"
    "typedef unsigned char __attribute__((address_space(1))) gu8;"
    "\n"
    "typedef unsigned short __attribute__((address_space(1))) gu1"
    "6;\n"
    "typedef unsigned int u32v __attribute__((ext_vector_type(16)"
    "));\n"
    "typedef unsigned short h32v __attribute__((ext_vector_type(3"
    "2)));\n"
    "typedef float f32v __attribute__((ext_vector_type(16)));\n"
    "typedef _Float16 hf32v __attribute__((ext_vector_type(32)));"
    "\n"
    "\n"
    "__attribute__((device,always_inline)) u32v halfpair638(u32v "
    "value) {\n"
    "    u32v packed = (value & 0xffu) | ((value & 0xff00u) << 8)"
    ";\n"
    "    u32v bits = (packed + (packed & 0x00800080u)) << 7;\n"
    "    u32v nanflag = ((value & 0x7f7fu) + 0x0101u) & 0x8080u;\n"
    "    u32v exponent = ((nanflag & 0x80u) << 3) | ((nanflag & 0"
    "x8000u) << 11);\n"
    "    bits |= (exponent << 5) - exponent;\n"
    "    return bits;\n"
    "}\n"
    "\n"
    "__attribute__((device,always_inline)) h32v pack644(f32v low,"
    " f32v high) {\n"
    "    u32v a0 = __builtin_bit_cast(u32v, low);\n"
    "    u32v b0 = __builtin_bit_cast(u32v, high);\n"
    "    u32v a1 = __builtin_xpu2_vmerge_l_u(a0, b0);\n"
    "    u32v b1 = __builtin_xpu2_vmerge_h_u(a0, b0);\n"
    "    u32v a2 = __builtin_xpu2_vmerge_l_u(a1, b1);\n"
    "    u32v b2 = __builtin_xpu2_vmerge_h_u(a1, b1);\n"
    "    u32v a3 = __builtin_xpu2_vmerge_l_u(a2, b2);\n"
    "    u32v b3 = __builtin_xpu2_vmerge_h_u(a2, b2);\n"
    "    u32v a4 = __builtin_xpu2_vmerge_l_u(a3, b3);\n"
    "    u32v b4 = __builtin_xpu2_vmerge_h_u(a3, b3);\n"
    "    return __builtin_bit_cast(h32v, (a4 >> 16) | (b4 & 0xfff"
    "f0000u));\n"
    "}\n"
    "\n"
    "__attribute__((device,always_inline)) h32v convert638(u32v w"
    "ord, unsigned int scale) {\n"
    "    hf32v half = __builtin_bit_cast(hf32v, word);\n"
    "    unsigned short scale_bits = (unsigned short)((scale - 10"
    "4u) << 10);\n"
    "    _Float16 factor = __builtin_bit_cast(_Float16, scale_bit"
    "s);\n"
    "    hf32v product = half * factor;\n"
    "    return __builtin_bit_cast(h32v, product);\n"
    "}\n"
    "\n"
    'extern "C" __attribute__((device)) void raw_fp1670(\n'
    "    const gi8* A, const gu8* AS, gu16* AD,\n"
    "    const gi8* C, const gu8* CS, gu16* CD,\n"
    "    long long ca, long long pid) {\n"
    "    unsigned int bytes[128] __attribute__((aligned(64)));\n"
    "    unsigned char scales[16] __attribute__((aligned(64)));\n"
    "    unsigned short values[512] __attribute__((aligned(64)));"
    "\n"
    "    int core = __builtin_xpu_core_id();\n"
    "    const gi8* src = pid < ca ? A : C;\n"
    "    const gu8* scale = pid < ca ? AS : CS;\n"
    "    gu16* dst = pid < ca ? AD : CD;\n"
    "    long long block = pid < ca ? pid : pid - ca;\n"
    "    long long offset = block * 32768 + core * 512;\n"
    "    __builtin_xpu_gm2lm_v3(bytes, src + offset, 0, 512);\n"
    "    __builtin_xpu_gm2lm_v3(scales, scale + offset / 32, 0, 1"
    "6);\n"
    "    __builtin_xpu2_mfence(7);\n"
    "    for (int group = 0; group < 8; ++group) {\n"
    "        u32v packed = *((u32v*)(bytes + group * 16));\n"
    "        u32v first = halfpair638(packed);\n"
    "        u32v second = halfpair638(packed >> 16);\n"
    "        u32v low = __builtin_xpu2_vmerge_l_u(first, second);"
    "\n"
    "        u32v high = __builtin_xpu2_vmerge_h_u(first, second)"
    ";\n"
    "        *((h32v*)(values + group * 64)) = convert638(low, sc"
    "ales[2 * group]);\n"
    "        *((h32v*)(values + group * 64 + 32)) = convert638(hi"
    "gh, scales[2 * group + 1]);\n"
    "    }\n"
    "    __builtin_xpu2_mfence(7);\n"
    "    __builtin_xpu_lm2gm_v3(dst + offset, values, 0, 1024);\n"
    "    __builtin_xpu2_mfence(7);\n"
    "}\n"
)


@tle.raw.dialect("xpu3", source=RAW_FP1670)
def raw_fp1670(A, AS, AD, C, CS, CD, ca, pid):
    pass


@triton.jit
def _raw_fp1670(A, AS, AD, C, CS, CD, META: tl.constexpr):
    CA: tl.constexpr = META[0]
    B: tl.constexpr = META[1]

    tl.static_assert(B == 16384)
    tle.raw.call(
        raw_fp1670,
        (
            A,
            AS,
            AD,
            C,
            CS,
            CD,
            tl.full((), CA, tl.int64),
            tl.program_id(0).to(tl.int64),
        ),
    )


def mxfp8_block_scaled_matmul(a, a_scale, b, b_scale, output_dtype):
    variant = "both"
    m, k = a.shape
    n = b.shape[0]
    out = torch.empty((m, n), device=a.device, dtype=output_dtype)
    if m == 0 or n == 0:
        return out
    if k == 0:
        _zero.run(
            out,
            (m * n, 1024),
            num_warps=4,
            grid=((m * n + 1024 - 1) // 1024,),
            warmup=False,
        )
        return out
    if (
        (
            m * n <= 512
            and k <= 128
            or (
                m * n <= 64
                and k <= 2048
                or (m * n <= 32 and k <= 4096)
                or (m * n <= 16 and k <= 8192)
            )
        )
        or (
            k <= 8192
            and (k <= 256 or k % 256 == 0)
            and (m * n * ((k + 256 - 1) // 256) <= 512)
        )
    ) or (4096 <= k <= 8192 and k % 512 == 0 and (m * n * (k // 512) <= 512)):
        for offset in range(0, m * n, 32768):
            _range_chunk563.run(
                a.view(torch.int8),
                a_scale,
                b.view(torch.int8),
                b_scale,
                out,
                offset,
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
                    32,
                    1 << (k - 1).bit_length(),
                ),
                num_warps=1,
                num_stages=1,
                grid=(min(32768, m * n - offset),),
                warmup=False,
            )
        return out
    contiguous_scales = a_scale.is_contiguous() and b_scale.is_contiguous()
    ca = max(1, (m * (k // 32) + 4096 - 1) // 4096) if contiguous_scales else m
    cb = max(1, (n * (k // 32) + 4096 - 1) // 4096) if contiguous_scales else n
    flags = torch.empty((ca + cb,), device=a.device, dtype=torch.int32)
    if contiguous_scales:
        _scale_joined702.run(
            a_scale,
            flags,
            b_scale,
            (m * (k // 32), n * (k // 32), 4096),
            num_warps=4,
            grid=(ca + cb,),
            warmup=False,
        )
    else:
        fa = flags[:ca]
        fb = flags[ca:]
        _scale_flags.run(
            a_scale,
            fa,
            (
                k // 32,
                a_scale.stride(0),
                a_scale.stride(1),
                max(1, 1 << (k // 32 - 1).bit_length()),
                a_scale.shape[0],
            ),
            num_warps=1,
            grid=(m,),
            warmup=False,
        )
        _scale_flags.run(
            b_scale,
            fb,
            (
                k // 32,
                b_scale.stride(0),
                b_scale.stride(1),
                max(1, 1 << (k // 32 - 1).bit_length()),
                b_scale.shape[0],
            ),
            num_warps=1,
            grid=(n,),
            warmup=False,
        )
    if ca + cb <= 8:
        safe_input = min(flags.cpu().tolist())
    else:
        safe = torch.empty((), device=a.device, dtype=torch.int32)
        _all_safe529.run(
            flags,
            safe,
            (ca + cb, 1 << (ca + cb - 1).bit_length()),
            num_warps=4,
            grid=(1,),
            warmup=False,
        )
        safe_input = int(safe.item())
    if not safe_input:
        for offset in range(0, m * n, 32768):
            _range_matmul.run(
                a.view(torch.int8),
                a_scale,
                b.view(torch.int8),
                b_scale,
                out,
                offset,
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
                    32,
                ),
                num_warps=1,
                num_stages=1,
                grid=(min(32768, m * n - offset),),
                warmup=False,
            )
        return out
    dot_n = max(n, 16)
    if n < 16:
        out = torch.empty((m, dot_n), device=a.device, dtype=output_dtype)
    intermediate_dtype = torch.float16 if safe_input == 2 else torch.bfloat16
    ad = torch.empty((m, k), device=a.device, dtype=intermediate_dtype)
    bd = torch.empty((dot_n, k), device=a.device, dtype=intermediate_dtype)
    if n < 16:
        padding = bd[n:]
        _zero.run(
            padding,
            ((dot_n - n) * k, 1024),
            num_warps=4,
            grid=(((dot_n - n) * k + 1024 - 1) // 1024,),
            warmup=False,
        )
    if (
        k >= 2048
        and contiguous_scales
        and a.is_contiguous()
        and b.is_contiguous()
        and (a.data_ptr() % 2 == 0)
        and (b.data_ptr() % 2 == 0)
    ):
        if (
            variant in ("decode", "both")
            and m * k % 32768 == 0
            and (n * k % 32768 == 0)
        ):
            if m * k % 131072 == 0 and n * k % 131072 == 0:
                (
                    _raw_bf16691
                    if intermediate_dtype == torch.bfloat16
                    else _raw_fp16691
                ).run(
                    a.view(torch.int8),
                    a_scale,
                    ad,
                    b.view(torch.int8),
                    b_scale,
                    bd,
                    (m * k // 131072, 65536, b.numel()),
                    num_warps=4,
                    isCloseMemoryAsync=False,
                    **(
                        {"unroll_num": 4}
                        if intermediate_dtype == torch.float16
                        else {}
                    ),
                    grid=((m + n) * k // 131072,),
                    warmup=False,
                )
            else:
                (
                    _raw_both632
                    if intermediate_dtype == torch.bfloat16
                    else _raw_fp1670
                ).run(
                    a.view(torch.int8),
                    a_scale,
                    ad,
                    b.view(torch.int8),
                    b_scale,
                    bd,
                    (m * k // 32768, 16384, b.numel()),
                    num_warps=4,
                    isCloseMemoryAsync=False,
                    **(
                        {"unroll_num": 4}
                        if intermediate_dtype == torch.float16
                        else {}
                    ),
                    grid=((m + n) * k // 32768,),
                    warmup=False,
                )
        else:
            decode_a = (
                _decode_linear_full if m * k % 16384 == 0 else _decode_linear
            )
            decode_b = (
                _decode_linear_full if n * k % 16384 == 0 else _decode_linear
            )
            block_a = 16384 if m * k % 32768 == 0 else 8192
            decode_a.run(
                a.view(torch.int8),
                a_scale,
                ad,
                (m * k, block_a),
                num_warps=4,
                grid=((m * k + 2 * block_a - 1) // (2 * block_a),),
                warmup=False,
            )
            block_b = 16384 if n * k % 32768 == 0 else 8192
            decode_b.run(
                b.view(torch.int8),
                b_scale,
                bd,
                (n * k, block_b),
                num_warps=4,
                grid=((n * k + 2 * block_b - 1) // (2 * block_b),),
                warmup=False,
            )
    elif max(m, n) <= 256:
        _dequant_rows_both354.run(
            a.view(torch.int8),
            a_scale,
            ad,
            b.view(torch.int8),
            b_scale,
            bd,
            (
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
                max(32, 1 << (k - 1).bit_length()),
                n,
            ),
            num_warps=4,
            grid=(m + n,),
            warmup=False,
        )
    else:
        _dequant_rows.run(
            a.view(torch.int8),
            a_scale,
            ad,
            (
                k,
                a.stride(0),
                a.stride(1),
                a_scale.stride(0),
                a_scale.stride(1),
                max(32, 1 << (k - 1).bit_length()),
                a.shape[0],
            ),
            num_warps=4,
            grid=(m,),
            warmup=False,
        )
        _dequant_rows.run(
            b.view(torch.int8),
            b_scale,
            bd,
            (
                k,
                b.stride(0),
                b.stride(1),
                b_scale.stride(0),
                b_scale.stride(1),
                max(32, 1 << (k - 1).bit_length()),
                b.shape[0],
            ),
            num_warps=4,
            grid=(n,),
            warmup=False,
        )
    bm = min(128, max(16, 2 ** (m.bit_length() - 1)))
    bn = min(256, max(16, 2 ** (n.bit_length() - 1)))
    bk = min(128, max(32, 2 ** (k.bit_length() - 1)))
    if m >= 512:
        bm, bn, bk = (
            256,
            512 if safe_input == 2 and n % 512 == 0 else 256,
            256,
        )
    bn = min(bn, max(16, 2 ** (n.bit_length() - 1)))
    _plain.run(
        ad,
        bd,
        out,
        (m, dot_n, k, bm, bn, bk),
        num_warps=1,
        num_stages=2,
        grid=((m + bm - 1) // bm * ((n + bn - 1) // bn),),
        warmup=False,
    )
    return out[:, :n] if n < 16 else out


RAW_BF16691 = (
    "typedef signed char __attribute__((address_space(1))) gi8;\n"
    "typedef unsigned char __attribute__((address_space(1))) gu8;"
    "\n"
    "typedef unsigned short __attribute__((address_space(1))) gu1"
    "6;\n"
    "typedef unsigned int u32v __attribute__((ext_vector_type(16)"
    "));\n"
    "typedef unsigned short h32v __attribute__((ext_vector_type(3"
    "2)));\n"
    "typedef float f32v __attribute__((ext_vector_type(16)));\n"
    "typedef _Float16 hf32v __attribute__((ext_vector_type(32)));"
    "\n"
    "\n"
    "__attribute__((device,always_inline)) u32v halfpair638(u32v "
    "value) {\n"
    "    u32v packed = (value & 0xffu) | ((value & 0xff00u) << 8)"
    ";\n"
    "    u32v bits = (packed + (packed & 0x00800080u)) << 7;\n"
    "    return bits;\n"
    "}\n"
    "\n"
    "__attribute__((device,always_inline)) h32v pack644(f32v low,"
    " f32v high) {\n"
    "    u32v a0 = __builtin_bit_cast(u32v, low);\n"
    "    u32v b0 = __builtin_bit_cast(u32v, high);\n"
    "    u32v a1 = __builtin_xpu2_vmerge_l_u(a0, b0);\n"
    "    u32v b1 = __builtin_xpu2_vmerge_h_u(a0, b0);\n"
    "    u32v a2 = __builtin_xpu2_vmerge_l_u(a1, b1);\n"
    "    u32v b2 = __builtin_xpu2_vmerge_h_u(a1, b1);\n"
    "    u32v a3 = __builtin_xpu2_vmerge_l_u(a2, b2);\n"
    "    u32v b3 = __builtin_xpu2_vmerge_h_u(a2, b2);\n"
    "    u32v a4 = __builtin_xpu2_vmerge_l_u(a3, b3);\n"
    "    u32v b4 = __builtin_xpu2_vmerge_h_u(a3, b3);\n"
    "    return __builtin_bit_cast(h32v, (a4 >> 16) | (b4 & 0xfff"
    "f0000u));\n"
    "}\n"
    "\n"
    "__attribute__((device,always_inline)) h32v convert638(u32v w"
    "ord, unsigned int scale) {\n"
    "    hf32v half = __builtin_bit_cast(hf32v, word);\n"
    "    hf32v normalized = half * (_Float16)256;\n"
    "    u32v mag = __builtin_bit_cast(u32v, normalized) & 0x7fff"
    "7fffu;\n"
    "    unsigned int bias = ((scale - 15u) << 7) & 0x7fffu;\n"
    "    u32v bits = ((mag >> 3) & 0x0fff0fffu) + (u32v)(bias * 0"
    "x00010001u);\n"
    "    u32v nonzero = ((mag + 0x7fff7fffu) & 0x80008000u) >> 15"
    ";\n"
    "    bits &= ((nonzero << 16) - nonzero) & 0x7fff7fffu;\n"
    "    u32v nan = ((mag + 0x20802080u) & 0x80008000u) >> 15;\n"
    "    u32v nanmask = (nan << 16) - nan;\n"
    "    bits = (bits & ~nanmask) | (0x7fc07fc0u & nanmask);\n"
    "    return __builtin_bit_cast(h32v, bits | (word & 0x8000800"
    "0u));\n"
    "}\n"
    "\n"
    'extern "C" __attribute__((device)) void raw_bf16691(\n'
    "    const gi8* A, const gu8* AS, gu16* AD,\n"
    "    const gi8* C, const gu8* CS, gu16* CD,\n"
    "    long long ca, long long pid) {\n"
    "    unsigned int bytes[512] __attribute__((aligned(64)));\n"
    "    unsigned char scales[64] __attribute__((aligned(64)));\n"
    "    unsigned short values[2048] __attribute__((aligned(64)))"
    ";\n"
    "    int core = __builtin_xpu_core_id();\n"
    "    const gi8* src = pid < ca ? A : C;\n"
    "    const gu8* scale = pid < ca ? AS : CS;\n"
    "    gu16* dst = pid < ca ? AD : CD;\n"
    "    long long block = pid < ca ? pid : pid - ca;\n"
    "    long long offset = block * 131072 + core * 2048;\n"
    "    __builtin_xpu_gm2lm_v3(bytes, src + offset, 0, 2048);\n"
    "    __builtin_xpu_gm2lm_v3(scales, scale + offset / 32, 0, 6"
    "4);\n"
    "    __builtin_xpu2_mfence(7);\n"
    "    for (int group = 0; group < 32; ++group) {\n"
    "        u32v packed = *((u32v*)(bytes + group * 16));\n"
    "        u32v first = halfpair638(packed);\n"
    "        u32v second = halfpair638(packed >> 16);\n"
    "        u32v low = __builtin_xpu2_vmerge_l_u(first, second);"
    "\n"
    "        u32v high = __builtin_xpu2_vmerge_h_u(first, second)"
    ";\n"
    "        *((h32v*)(values + group * 64)) = convert638(low, sc"
    "ales[2 * group]);\n"
    "        *((h32v*)(values + group * 64 + 32)) = convert638(hi"
    "gh, scales[2 * group + 1]);\n"
    "    }\n"
    "    __builtin_xpu2_mfence(7);\n"
    "    __builtin_xpu_lm2gm_v3(dst + offset, values, 0, 4096);\n"
    "    __builtin_xpu2_mfence(7);\n"
    "}\n"
)


@tle.raw.dialect("xpu3", source=RAW_BF16691)
def raw_bf16691(A, AS, AD, C, CS, CD, ca, pid):
    pass


@triton.jit
def _raw_bf16691(A, AS, AD, C, CS, CD, META: tl.constexpr):
    CA: tl.constexpr = META[0]
    B: tl.constexpr = META[1]

    tl.static_assert(B == 65536)
    tle.raw.call(
        raw_bf16691,
        (
            A,
            AS,
            AD,
            C,
            CS,
            CD,
            tl.full((), CA, tl.int64),
            tl.program_id(0).to(tl.int64),
        ),
    )


RAW_FP16691 = (
    "typedef signed char __attribute__((address_space(1))) gi8;\n"
    "typedef unsigned char __attribute__((address_space(1))) gu8;"
    "\n"
    "typedef unsigned short __attribute__((address_space(1))) gu1"
    "6;\n"
    "typedef unsigned int u32v __attribute__((ext_vector_type(16)"
    "));\n"
    "typedef unsigned short h32v __attribute__((ext_vector_type(3"
    "2)));\n"
    "typedef float f32v __attribute__((ext_vector_type(16)));\n"
    "typedef _Float16 hf32v __attribute__((ext_vector_type(32)));"
    "\n"
    "\n"
    "__attribute__((device,always_inline)) u32v halfpair638(u32v "
    "value) {\n"
    "    u32v packed = (value & 0xffu) | ((value & 0xff00u) << 8)"
    ";\n"
    "    u32v bits = (packed + (packed & 0x00800080u)) << 7;\n"
    "    u32v nanflag = ((value & 0x7f7fu) + 0x0101u) & 0x8080u;\n"
    "    u32v exponent = ((nanflag & 0x80u) << 3) | ((nanflag & 0"
    "x8000u) << 11);\n"
    "    bits |= (exponent << 5) - exponent;\n"
    "    return bits;\n"
    "}\n"
    "\n"
    "__attribute__((device,always_inline)) h32v pack644(f32v low,"
    " f32v high) {\n"
    "    u32v a0 = __builtin_bit_cast(u32v, low);\n"
    "    u32v b0 = __builtin_bit_cast(u32v, high);\n"
    "    u32v a1 = __builtin_xpu2_vmerge_l_u(a0, b0);\n"
    "    u32v b1 = __builtin_xpu2_vmerge_h_u(a0, b0);\n"
    "    u32v a2 = __builtin_xpu2_vmerge_l_u(a1, b1);\n"
    "    u32v b2 = __builtin_xpu2_vmerge_h_u(a1, b1);\n"
    "    u32v a3 = __builtin_xpu2_vmerge_l_u(a2, b2);\n"
    "    u32v b3 = __builtin_xpu2_vmerge_h_u(a2, b2);\n"
    "    u32v a4 = __builtin_xpu2_vmerge_l_u(a3, b3);\n"
    "    u32v b4 = __builtin_xpu2_vmerge_h_u(a3, b3);\n"
    "    return __builtin_bit_cast(h32v, (a4 >> 16) | (b4 & 0xfff"
    "f0000u));\n"
    "}\n"
    "\n"
    "__attribute__((device,always_inline)) h32v convert638(u32v w"
    "ord, unsigned int scale) {\n"
    "    hf32v half = __builtin_bit_cast(hf32v, word);\n"
    "    unsigned short scale_bits = (unsigned short)((scale - 10"
    "4u) << 10);\n"
    "    _Float16 factor = __builtin_bit_cast(_Float16, scale_bit"
    "s);\n"
    "    hf32v product = half * factor;\n"
    "    return __builtin_bit_cast(h32v, product);\n"
    "}\n"
    "\n"
    'extern "C" __attribute__((device)) void raw_fp16691(\n'
    "    const gi8* A, const gu8* AS, gu16* AD,\n"
    "    const gi8* C, const gu8* CS, gu16* CD,\n"
    "    long long ca, long long pid) {\n"
    "    unsigned int bytes[512] __attribute__((aligned(64)));\n"
    "    unsigned char scales[64] __attribute__((aligned(64)));\n"
    "    unsigned short values[2048] __attribute__((aligned(64)))"
    ";\n"
    "    int core = __builtin_xpu_core_id();\n"
    "    const gi8* src = pid < ca ? A : C;\n"
    "    const gu8* scale = pid < ca ? AS : CS;\n"
    "    gu16* dst = pid < ca ? AD : CD;\n"
    "    long long block = pid < ca ? pid : pid - ca;\n"
    "    long long offset = block * 131072 + core * 2048;\n"
    "    __builtin_xpu_gm2lm_v3(bytes, src + offset, 0, 2048);\n"
    "    __builtin_xpu_gm2lm_v3(scales, scale + offset / 32, 0, 6"
    "4);\n"
    "    __builtin_xpu2_mfence(7);\n"
    "    for (int group = 0; group < 32; ++group) {\n"
    "        u32v packed = *((u32v*)(bytes + group * 16));\n"
    "        u32v first = halfpair638(packed);\n"
    "        u32v second = halfpair638(packed >> 16);\n"
    "        u32v low = __builtin_xpu2_vmerge_l_u(first, second);"
    "\n"
    "        u32v high = __builtin_xpu2_vmerge_h_u(first, second)"
    ";\n"
    "        *((h32v*)(values + group * 64)) = convert638(low, sc"
    "ales[2 * group]);\n"
    "        *((h32v*)(values + group * 64 + 32)) = convert638(hi"
    "gh, scales[2 * group + 1]);\n"
    "    }\n"
    "    __builtin_xpu2_mfence(7);\n"
    "    __builtin_xpu_lm2gm_v3(dst + offset, values, 0, 4096);\n"
    "    __builtin_xpu2_mfence(7);\n"
    "}\n"
)


@tle.raw.dialect("xpu3", source=RAW_FP16691)
def raw_fp16691(A, AS, AD, C, CS, CD, ca, pid):
    pass


@triton.jit
def _raw_fp16691(A, AS, AD, C, CS, CD, META: tl.constexpr):
    CA: tl.constexpr = META[0]
    B: tl.constexpr = META[1]

    tl.static_assert(B == 65536)
    tle.raw.call(
        raw_fp16691,
        (
            A,
            AS,
            AD,
            C,
            CS,
            CD,
            tl.full((), CA, tl.int64),
            tl.program_id(0).to(tl.int64),
        ),
    )


@triton.jit
def _scale_joined702(SA, F, SB, META: tl.constexpr):
    COUNT_A: tl.constexpr = META[0]
    COUNT_B: tl.constexpr = META[1]
    B: tl.constexpr = META[2]
    ADDRESS64: tl.constexpr = COUNT_A + COUNT_B + 2 * B >= 2147483648
    pid = (
        tl.cast(tl.program_id(0), tl.int64) if ADDRESS64 else tl.program_id(0)
    )
    ca = tl.cdiv(COUNT_A, B)
    is_a = pid < ca
    src = tl.where(is_a, SA, SB)
    block = tl.where(is_a, pid, pid - ca)
    count = tl.where(is_a, COUNT_A, COUNT_B)
    cols = block * B + (
        tl.cast(tl.arange(0, B), tl.int64) if ADDRESS64 else tl.arange(0, B)
    )
    s = tl.load(src + cols, cols < count, 127).to(tl.int32) & 255
    safe = tl.min(
        tl.where(
            (s >= 112) & (s <= 134), 2, ((s >= 10) & (s <= 246)).to(tl.int32)
        ),
        0,
    )
    tl.store(F + pid, safe)


__all__ = ["mxfp8_block_scaled_matmul"]
