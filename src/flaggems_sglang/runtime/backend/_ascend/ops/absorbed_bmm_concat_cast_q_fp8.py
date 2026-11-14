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
from triton.language.extra.cann import extension as al


@triton.jit
def _address_index(index, WIDE: tl.constexpr):
    if WIDE:
        return index.to(tl.int64)
    else:
        return index


def _power_of_two(n):
    return 1 << (n - 1).bit_length() if n > 0 else 0


@triton.jit
def _fused_narrow(X, W, R, output_ptr, META: tl.constexpr):
    T: tl.constexpr = META[0]
    H: tl.constexpr = META[1]
    K: tl.constexpr = META[2]
    N: tl.constexpr = META[3]
    D: tl.constexpr = META[4]
    X0: tl.constexpr = META[5]
    X1: tl.constexpr = META[6]
    X2: tl.constexpr = META[7]
    W0: tl.constexpr = META[8]
    W1: tl.constexpr = META[9]
    W2: tl.constexpr = META[10]
    R0: tl.constexpr = META[11]
    R1: tl.constexpr = META[12]
    R2: tl.constexpr = META[13]
    O0: tl.constexpr = META[14]
    O1: tl.constexpr = META[15]
    O2: tl.constexpr = META[16]
    TM: tl.constexpr = META[17]
    TN: tl.constexpr = META[18]
    BM: tl.constexpr = META[19]
    BN: tl.constexpr = META[20]
    BK: tl.constexpr = META[21]
    BR: tl.constexpr = META[22]
    BIT_ROUND: tl.constexpr = META[23]
    WIDE: tl.constexpr = (
        (
            (T - 1) * X0 + (H - 1) * X1 + (K - 1) * X2 > 2147483647
            or (H - 1) * W0 + (K - 1) * W1 + (N - 1) * W2 > 2147483647
            or (T - 1) * R0 + (H - 1) * R1 + (D - 1) * R2 > 2147483647
            or ((T - 1) * O0 + (H - 1) * O1 + (N + D - 1) * O2 > 2147483647)
        )
        or X0 > 2147483647
        or X1 > 2147483647
        or (X2 > 2147483647)
        or (W0 > 2147483647)
        or (W1 > 2147483647)
        or (W2 > 2147483647)
        or (R0 > 2147483647)
        or (R1 > 2147483647)
        or (R2 > 2147483647)
        or (O0 > 2147483647)
        or (O1 > 2147483647)
        or (O2 > 2147483647)
    )
    PROGRAMS: tl.constexpr = H * TM * TN
    LOOP_WIDE: tl.constexpr = PROGRAMS - 1 + (
        PROGRAMS if PROGRAMS < 96 else 96
    ) > 2147483647 or (TM * BM - 1 > 2147483647 or TN * BN - 1 > 2147483647)
    if LOOP_WIDE:
        loop_stop = tl.full((), PROGRAMS, tl.int64)
    else:
        loop_stop = PROGRAMS
    for pid in range(
        _address_index(tl.program_id(0), LOOP_WIDE),
        loop_stop,
        _address_index(tl.num_programs(0), LOOP_WIDE),
    ):
        h = pid // (TM * TN)
        m = pid // TN % TM * BM + tl.arange(0, BM)
        n = pid % TN * BN + tl.arange(0, BN)
        k = tl.arange(0, BK)
        acc = tl.zeros((BM, BN), tl.float32)
        for ki in range(tl.cdiv(K, BK)):
            kk = ki * BK + k
            x = tl.load(
                X
                + _address_index(m[:, None], WIDE) * X0
                + _address_index(h, WIDE) * X1
                + _address_index(kk[None, :], WIDE) * X2,
                (m < T)[:, None] & (kk < K)[None, :],
                other=0,
            )
            w = tl.load(
                W
                + _address_index(h, WIDE) * W0
                + _address_index(kk[:, None], WIDE) * W1
                + _address_index(n[None, :], WIDE) * W2,
                (kk < K)[:, None] & (n < N)[None, :],
                other=0,
            )
            acc = tl.dot(x, w, acc, input_precision="ieee")
        if BIT_ROUND:
            bits = acc.to(tl.uint32, bitcast=True)
            rounded = (bits + 32767 + (bits >> 16 & 1) & 4294901760).to(
                tl.float32, bitcast=True
            )
            rounded = tl.where(bits & 2139095040 == 2139095040, acc, rounded)
        else:
            rounded = acc.to(tl.bfloat16).to(tl.float32)
        code = _encode_narrow(rounded)
        tl.store(
            output_ptr
            + _address_index(m[:, None], WIDE) * O0
            + _address_index(h, WIDE) * O1
            + _address_index(n[None, :], WIDE) * O2,
            code,
            (m < T)[:, None] & (n < N)[None, :],
        )
        if D > 0:
            if pid % TN == 0:
                ROPE_PROGRAMS: tl.constexpr = (D + BR - 1) // BR
                ROPE_WIDE: tl.constexpr = (
                    ROPE_PROGRAMS > 2147483647
                    or N + ROPE_PROGRAMS * BR - 1 > 2147483647
                )
                if ROPE_WIDE:
                    rope_start = tl.full((), 0, tl.int64)
                    rope_stop = tl.full((), ROPE_PROGRAMS, tl.int64)
                    rope_step = tl.full((), 1, tl.int64)
                else:
                    rope_start = 0
                    rope_stop = ROPE_PROGRAMS
                    rope_step = 1
                for ri in range(rope_start, rope_stop, rope_step):
                    r = ri * BR + tl.arange(0, BR)
                    value = (
                        tl.load(
                            R
                            + _address_index(m[:, None], WIDE) * R0
                            + _address_index(h, WIDE) * R1
                            + _address_index(r[None, :], WIDE) * R2,
                            (m < T)[:, None] & (r < D)[None, :],
                            other=0,
                        ).to(tl.uint32)
                        << 16
                    ).to(tl.float32, bitcast=True)
                    tl.store(
                        output_ptr
                        + _address_index(m[:, None], WIDE) * O0
                        + _address_index(h, WIDE) * O1
                        + _address_index(
                            N + _address_index(r[None, :], WIDE), WIDE
                        )
                        * O2,
                        _encode_narrow(value),
                        (m < T)[:, None] & (r < D)[None, :],
                    )


@triton.jit
def _encode_narrow(v):
    raw = v.to(tl.int32, bitcast=True)
    mag = tl.minimum(tl.abs(v).to(tl.int32, bitcast=True), 1139802112)
    exponent = tl.maximum(mag >> 23, 121)
    magic_bits = exponent * 8388616 + 167771192
    magic = magic_bits.to(tl.float32, bitcast=True)
    absolute = mag.to(tl.float32, bitcast=True)
    rounded = (absolute + magic).to(tl.int32, bitcast=True).to(tl.int16)
    sign = (raw >> 24 & 128).to(tl.int16)
    code = (rounded & 127) - sign
    return al.cast(code, tl.int8, overflow_mode="saturate").to(
        tl.uint8, bitcast=True
    )


@triton.jit
def _encode_rope_narrow_raw(value):
    bits = value.to(tl.int16)
    mag = tl.minimum(bits & 32767, tl.full((), 17392, tl.int16))
    half = mag.to(tl.bfloat16, bitcast=True).to(tl.float16)
    exponent = tl.maximum(mag >> 7, tl.full((), 121, tl.int16))
    magic_bits = exponent * tl.full((), 1032, tl.int16) + tl.full(
        (), 22584, tl.int16
    )
    magic = magic_bits.to(tl.float16, bitcast=True)
    rounded = (half + magic).to(tl.int16, bitcast=True)
    return (rounded | bits >> 8 & 128).to(tl.uint8)


@triton.jit
def _rope_rows(R, output_ptr, META: tl.constexpr):
    T: tl.constexpr = META[0]
    H: tl.constexpr = META[1]
    N: tl.constexpr = META[2]
    D: tl.constexpr = META[3]
    R0: tl.constexpr = META[4]
    R1: tl.constexpr = META[5]
    R2: tl.constexpr = META[6]
    O0: tl.constexpr = META[7]
    O1: tl.constexpr = META[8]
    O2: tl.constexpr = META[9]
    BT: tl.constexpr = META[10]
    BR: tl.constexpr = META[11]
    WIDE: tl.constexpr = (
        (
            (T - 1) * R0 + (H - 1) * R1 + (D - 1) * R2 > 2147483647
            or (T - 1) * O0 + (H - 1) * O1 + (N + D - 1) * O2 > 2147483647
        )
        or R0 > 2147483647
        or R1 > 2147483647
        or (R2 > 2147483647)
        or (O0 > 2147483647)
        or (O1 > 2147483647)
        or (O2 > 2147483647)
    )
    tm = tl.cdiv(T, BT)
    tr = tl.cdiv(D, BR)
    PROGRAMS: tl.constexpr = H * ((T + BT - 1) // BT) * ((D + BR - 1) // BR)
    LOOP_WIDE: tl.constexpr = PROGRAMS - 1 + (
        PROGRAMS if PROGRAMS < 96 else 96
    ) > 2147483647 or (
        (T + BT - 1) // BT * BT - 1 > 2147483647
        or N + (D + BR - 1) // BR * BR - 1 > 2147483647
    )
    if LOOP_WIDE:
        loop_stop = tl.full((), PROGRAMS, tl.int64)
    else:
        loop_stop = PROGRAMS
    for pid in range(
        _address_index(tl.program_id(0), LOOP_WIDE),
        loop_stop,
        _address_index(tl.num_programs(0), LOOP_WIDE),
    ):
        h = pid // (tm * tr)
        m = pid // tr % tm * BT + tl.arange(0, BT)
        r = pid % tr * BR + tl.arange(0, BR)
        raw = tl.load(
            R
            + _address_index(m[:, None], WIDE) * R0
            + _address_index(h, WIDE) * R1
            + _address_index(r[None, :], WIDE) * R2,
            (m < T)[:, None] & (r < D)[None, :],
            other=0,
        )
        tl.store(
            output_ptr
            + _address_index(m[:, None], WIDE) * O0
            + _address_index(h, WIDE) * O1
            + _address_index(N + _address_index(r[None, :], WIDE), WIDE) * O2,
            _encode_rope_narrow_raw(raw),
            (m < T)[:, None] & (r < D)[None, :],
        )


@triton.jit
def _encode(v):
    raw = v.to(tl.int32, bitcast=True)
    sign = raw >> 24 & 128
    mag = raw & 2147483647
    normal = (mag + 524287 + (mag >> 20 & 1) >> 20) - 960
    absolute = mag.to(tl.float32, bitcast=True)
    sub = (absolute + 16384.0).to(tl.int32, bitcast=True) - (141 << 23)
    code = tl.where(mag >= 1015021568, normal, sub)
    code = tl.where(absolute <= 464.0, code, 127)
    return (code | sign).to(tl.uint8)


@triton.jit
def _fused(X, W, R, output_ptr, META: tl.constexpr):
    T: tl.constexpr = META[0]
    H: tl.constexpr = META[1]
    K: tl.constexpr = META[2]
    N: tl.constexpr = META[3]
    D: tl.constexpr = META[4]
    X0: tl.constexpr = META[5]
    X1: tl.constexpr = META[6]
    X2: tl.constexpr = META[7]
    W0: tl.constexpr = META[8]
    W1: tl.constexpr = META[9]
    W2: tl.constexpr = META[10]
    R0: tl.constexpr = META[11]
    R1: tl.constexpr = META[12]
    R2: tl.constexpr = META[13]
    O0: tl.constexpr = META[14]
    O1: tl.constexpr = META[15]
    O2: tl.constexpr = META[16]
    TM: tl.constexpr = META[17]
    TN: tl.constexpr = META[18]
    BM: tl.constexpr = META[19]
    BN: tl.constexpr = META[20]
    BK: tl.constexpr = META[21]
    BR: tl.constexpr = META[22]
    BIT_ROUND: tl.constexpr = META[23]
    WIDE: tl.constexpr = (
        (
            (T - 1) * X0 + (H - 1) * X1 + (K - 1) * X2 > 2147483647
            or (H - 1) * W0 + (K - 1) * W1 + (N - 1) * W2 > 2147483647
            or (T - 1) * R0 + (H - 1) * R1 + (D - 1) * R2 > 2147483647
            or ((T - 1) * O0 + (H - 1) * O1 + (N + D - 1) * O2 > 2147483647)
        )
        or X0 > 2147483647
        or X1 > 2147483647
        or (X2 > 2147483647)
        or (W0 > 2147483647)
        or (W1 > 2147483647)
        or (W2 > 2147483647)
        or (R0 > 2147483647)
        or (R1 > 2147483647)
        or (R2 > 2147483647)
        or (O0 > 2147483647)
        or (O1 > 2147483647)
        or (O2 > 2147483647)
    )
    PROGRAMS: tl.constexpr = H * TM * TN
    LOOP_WIDE: tl.constexpr = PROGRAMS - 1 + (
        PROGRAMS if PROGRAMS < 96 else 96
    ) > 2147483647 or (TM * BM - 1 > 2147483647 or TN * BN - 1 > 2147483647)
    if LOOP_WIDE:
        loop_stop = tl.full((), PROGRAMS, tl.int64)
    else:
        loop_stop = PROGRAMS
    for pid in range(
        _address_index(tl.program_id(0), LOOP_WIDE),
        loop_stop,
        _address_index(tl.num_programs(0), LOOP_WIDE),
    ):
        h = pid // (TM * TN)
        m = pid // TN % TM * BM + tl.arange(0, BM)
        n = pid % TN * BN + tl.arange(0, BN)
        k = tl.arange(0, BK)
        acc = tl.zeros((BM, BN), tl.float32)
        for ki in range(tl.cdiv(K, BK)):
            kk = ki * BK + k
            x = tl.load(
                X
                + _address_index(m[:, None], WIDE) * X0
                + _address_index(h, WIDE) * X1
                + _address_index(kk[None, :], WIDE) * X2,
                (m < T)[:, None] & (kk < K)[None, :],
                other=0,
            )
            w = tl.load(
                W
                + _address_index(h, WIDE) * W0
                + _address_index(kk[:, None], WIDE) * W1
                + _address_index(n[None, :], WIDE) * W2,
                (kk < K)[:, None] & (n < N)[None, :],
                other=0,
            )
            acc = tl.dot(x, w, acc, input_precision="ieee")
        if BIT_ROUND:
            bits = acc.to(tl.uint32, bitcast=True)
            rounded = (bits + 32767 + (bits >> 16 & 1) & 4294901760).to(
                tl.float32, bitcast=True
            )
            rounded = tl.where(bits & 2139095040 == 2139095040, acc, rounded)
        else:
            rounded = acc.to(tl.bfloat16).to(tl.float32)
        code = _encode(rounded)
        tl.store(
            output_ptr
            + _address_index(m[:, None], WIDE) * O0
            + _address_index(h, WIDE) * O1
            + _address_index(n[None, :], WIDE) * O2,
            code,
            (m < T)[:, None] & (n < N)[None, :],
        )
        if D > 0:
            if pid % TN == 0:
                ROPE_PROGRAMS: tl.constexpr = (D + BR - 1) // BR
                ROPE_WIDE: tl.constexpr = (
                    ROPE_PROGRAMS > 2147483647
                    or N + ROPE_PROGRAMS * BR - 1 > 2147483647
                )
                if ROPE_WIDE:
                    rope_start = tl.full((), 0, tl.int64)
                    rope_stop = tl.full((), ROPE_PROGRAMS, tl.int64)
                    rope_step = tl.full((), 1, tl.int64)
                else:
                    rope_start = 0
                    rope_stop = ROPE_PROGRAMS
                    rope_step = 1
                for ri in range(rope_start, rope_stop, rope_step):
                    r = ri * BR + tl.arange(0, BR)
                    value = (
                        tl.load(
                            R
                            + _address_index(m[:, None], WIDE) * R0
                            + _address_index(h, WIDE) * R1
                            + _address_index(r[None, :], WIDE) * R2,
                            (m < T)[:, None] & (r < D)[None, :],
                            other=0,
                        ).to(tl.uint32)
                        << 16
                    ).to(tl.float32, bitcast=True)
                    tl.store(
                        output_ptr
                        + _address_index(m[:, None], WIDE) * O0
                        + _address_index(h, WIDE) * O1
                        + _address_index(
                            N + _address_index(r[None, :], WIDE), WIDE
                        )
                        * O2,
                        _encode(value),
                        (m < T)[:, None] & (r < D)[None, :],
                    )


@triton.jit
def _fallback(X, W, R, output_ptr, META: tl.constexpr):
    T: tl.constexpr = META[0]
    H: tl.constexpr = META[1]
    K: tl.constexpr = META[2]
    N: tl.constexpr = META[3]
    D: tl.constexpr = META[4]
    X0: tl.constexpr = META[5]
    X1: tl.constexpr = META[6]
    X2: tl.constexpr = META[7]
    W0: tl.constexpr = META[8]
    W1: tl.constexpr = META[9]
    W2: tl.constexpr = META[10]
    R0: tl.constexpr = META[11]
    R1: tl.constexpr = META[12]
    R2: tl.constexpr = META[13]
    O0: tl.constexpr = META[14]
    O1: tl.constexpr = META[15]
    O2: tl.constexpr = META[16]
    BK: tl.constexpr = META[17]
    BN: tl.constexpr = META[18]
    BIT_ROUND: tl.constexpr = META[19]
    WIDE: tl.constexpr = (
        (
            (T - 1) * X0 + (H - 1) * X1 + (K - 1) * X2 > 2147483647
            or (H - 1) * W0 + (K - 1) * W1 + (N - 1) * W2 > 2147483647
            or (T - 1) * R0 + (H - 1) * R1 + (D - 1) * R2 > 2147483647
            or ((T - 1) * O0 + (H - 1) * O1 + (N + D - 1) * O2 > 2147483647)
        )
        or X0 > 2147483647
        or X1 > 2147483647
        or (X2 > 2147483647)
        or (W0 > 2147483647)
        or (W1 > 2147483647)
        or (W2 > 2147483647)
        or (R0 > 2147483647)
        or (R1 > 2147483647)
        or (R2 > 2147483647)
        or (O0 > 2147483647)
        or (O1 > 2147483647)
        or (O2 > 2147483647)
    )
    PROGRAMS: tl.constexpr = T * H * ((N + BN - 1) // BN)
    LOOP_WIDE: tl.constexpr = (
        PROGRAMS - 1 + (PROGRAMS if PROGRAMS < 32768 else 32768) > 2147483647
        or (N + BN - 1) // BN * BN - 1 > 2147483647
    )
    if LOOP_WIDE:
        loop_stop = tl.full((), PROGRAMS, tl.int64)
    else:
        loop_stop = PROGRAMS
    for pid in range(
        _address_index(tl.program_id(0), LOOP_WIDE),
        loop_stop,
        _address_index(tl.num_programs(0), LOOP_WIDE),
    ):
        row = pid // tl.cdiv(N, BN)
        t = row // H
        h = row % H
        k = tl.arange(0, BK)
        n = pid % tl.cdiv(N, BN) * BN + tl.arange(0, BN)
        x = tl.load(
            X
            + _address_index(t, WIDE) * X0
            + _address_index(h, WIDE) * X1
            + _address_index(k, WIDE) * X2,
            k < K,
            other=0,
        ).to(tl.float32)
        w = tl.load(
            W
            + _address_index(h, WIDE) * W0
            + _address_index(k[:, None], WIDE) * W1
            + _address_index(n[None, :], WIDE) * W2,
            (k < K)[:, None] & (n < N)[None, :],
            other=0,
        ).to(tl.float32)
        product = x[:, None] * w
        acc = tl.sum(product, axis=0)
        if BIT_ROUND:
            bits = acc.to(tl.uint32, bitcast=True)
            rounded = (bits + 32767 + (bits >> 16 & 1) & 4294901760).to(
                tl.float32, bitcast=True
            )
            rounded = tl.where(bits & 2139095040 == 2139095040, acc, rounded)
        else:
            rounded = acc.to(tl.bfloat16).to(tl.float32)
        tl.store(
            output_ptr
            + _address_index(t, WIDE) * O0
            + _address_index(h, WIDE) * O1
            + _address_index(n, WIDE) * O2,
            _encode(rounded),
            n < N,
        )
        if D > 0:
            if pid % tl.cdiv(N, BN) == 0:
                ROPE_PROGRAMS: tl.constexpr = (D + 64 - 1) // 64
                ROPE_WIDE: tl.constexpr = (
                    ROPE_PROGRAMS > 2147483647
                    or N + ROPE_PROGRAMS * 64 - 1 > 2147483647
                )
                if ROPE_WIDE:
                    rope_start = tl.full((), 0, tl.int64)
                    rope_stop = tl.full((), ROPE_PROGRAMS, tl.int64)
                    rope_step = tl.full((), 1, tl.int64)
                else:
                    rope_start = 0
                    rope_stop = ROPE_PROGRAMS
                    rope_step = 1
                for ri in range(rope_start, rope_stop, rope_step):
                    r = ri * 64 + tl.arange(0, 64)
                    value = (
                        tl.load(
                            R
                            + _address_index(t, WIDE) * R0
                            + _address_index(h, WIDE) * R1
                            + _address_index(r, WIDE) * R2,
                            r < D,
                            other=0,
                        ).to(tl.uint32)
                        << 16
                    ).to(tl.float32, bitcast=True)
                    tl.store(
                        output_ptr
                        + _address_index(t, WIDE) * O0
                        + _address_index(h, WIDE) * O1
                        + _address_index(N + _address_index(r, WIDE), WIDE)
                        * O2,
                        _encode(value),
                        r < D,
                    )


@triton.jit
def _copy_layout(X, output_ptr, META: tl.constexpr):
    T: tl.constexpr = META[0]
    H: tl.constexpr = META[1]
    C: tl.constexpr = META[2]
    X0: tl.constexpr = META[3]
    X1: tl.constexpr = META[4]
    X2: tl.constexpr = META[5]
    O0: tl.constexpr = META[6]
    O1: tl.constexpr = META[7]
    O2: tl.constexpr = META[8]
    BM: tl.constexpr = META[9]
    BC: tl.constexpr = META[10]
    WIDE: tl.constexpr = (
        (
            (T - 1) * X0 + (H - 1) * X1 + (C - 1) * X2 > 2147483647
            or (T - 1) * O0 + (H - 1) * O1 + (C - 1) * O2 > 2147483647
        )
        or X0 > 2147483647
        or X1 > 2147483647
        or (X2 > 2147483647)
        or (O0 > 2147483647)
        or (O1 > 2147483647)
        or (O2 > 2147483647)
    )
    tm = tl.cdiv(T, BM)
    tc = tl.cdiv(C, BC)
    PROGRAMS: tl.constexpr = H * ((T + BM - 1) // BM) * ((C + BC - 1) // BC)
    LOOP_WIDE: tl.constexpr = PROGRAMS - 1 + (
        PROGRAMS if PROGRAMS < 96 else 96
    ) > 2147483647 or (
        (T + BM - 1) // BM * BM - 1 > 2147483647
        or (C + BC - 1) // BC * BC - 1 > 2147483647
    )
    if LOOP_WIDE:
        loop_stop = tl.full((), PROGRAMS, tl.int64)
    else:
        loop_stop = PROGRAMS
    for pid in range(
        _address_index(tl.program_id(0), LOOP_WIDE),
        loop_stop,
        _address_index(tl.num_programs(0), LOOP_WIDE),
    ):
        h = pid // (tm * tc)
        m = pid // tc % tm * BM + tl.arange(0, BM)
        c = pid % tc * BC + tl.arange(0, BC)
        value = tl.load(
            X
            + _address_index(m[:, None], WIDE) * X0
            + _address_index(h, WIDE) * X1
            + _address_index(c[None, :], WIDE) * X2,
            (m < T)[:, None] & (c < C)[None, :],
            other=0,
        )
        tl.store(
            output_ptr
            + _address_index(m[:, None], WIDE) * O0
            + _address_index(h, WIDE) * O1
            + _address_index(c[None, :], WIDE) * O2,
            value,
            (m < T)[:, None] & (c < C)[None, :],
        )


def _pack_rows(x, t, h, c):
    step = x.stride(2)
    if (
        0 < c <= 256
        and step > 1
        and step & step - 1
        and (t * h * c >= 131072)
        and ((c - 1) * step + 1 <= 2048)
    ):
        return _pack_gather(x, t, h, c, 8)
    packed = x.new_empty((t, h, c), dtype=x.dtype)
    source, target = (x.view(torch.uint16), packed.view(torch.uint16))
    bc = min(128, _power_of_two(c))
    _copy_layout.run(
        source,
        target,
        (t, h, c, *source.stride(), *target.stride(), 16, bc),
        num_warps=4,
        grid=(min(h * ((t + 16 - 1) // 16) * ((c + bc - 1) // bc), 96),),
        warmup=False,
    )
    return packed


@triton.jit
def _copy_gather(X, output_ptr, META: tl.constexpr):
    T: tl.constexpr = META[0]
    H: tl.constexpr = META[1]
    C: tl.constexpr = META[2]
    X0: tl.constexpr = META[3]
    X1: tl.constexpr = META[4]
    X2: tl.constexpr = META[5]
    BT: tl.constexpr = META[6]
    BC: tl.constexpr = META[7]
    SPAN: tl.constexpr = META[8]
    WIDE: tl.constexpr = (
        (
            (T - 1) * X0 + (H - 1) * X1 + (C - 1) * X2 > 2147483647
            or T * H * C > 2147483647
        )
        or X0 > 2147483647
        or X1 > 2147483647
        or (X2 > 2147483647)
    )
    tm = tl.cdiv(T, BT)
    PROGRAMS: tl.constexpr = H * ((T + BT - 1) // BT)
    LOOP_WIDE: tl.constexpr = (
        PROGRAMS - 1 + (PROGRAMS if PROGRAMS < 96 else 96) > 2147483647
        or (T + BT - 1) // BT * BT - 1 > 2147483647
    )
    if LOOP_WIDE:
        loop_stop = tl.full((), PROGRAMS, tl.int64)
    else:
        loop_stop = PROGRAMS
    for pid in range(
        _address_index(tl.program_id(0), LOOP_WIDE),
        loop_stop,
        _address_index(tl.num_programs(0), LOOP_WIDE),
    ):
        h = pid // tm
        m = pid % tm * BT + tl.arange(0, BT)
        physical = tl.arange(0, SPAN)
        values = tl.load(
            X
            + _address_index(m[:, None], WIDE) * X0
            + _address_index(h, WIDE) * X1
            + physical[None, :],
            (m < T)[:, None] & (physical < (C - 1) * X2 + 1)[None, :],
            other=0,
        )
        c = tl.arange(0, BC)
        positions = tl.broadcast_to(
            (tl.minimum(c, C - 1) * X2)[None, :], (BT, BC)
        )
        chosen = tl.gather(
            values.to(tl.float16, bitcast=True), positions, axis=1
        ).to(tl.uint16, bitcast=True)
        tl.store(
            output_ptr
            + _address_index(_address_index(m[:, None], WIDE) * H + h, WIDE)
            * C
            + c[None, :],
            chosen,
            (m < T)[:, None] & (c < C)[None, :],
        )


def _pack_gather(x, t, h, c, bt):
    packed = x.new_empty((t, h, c), dtype=x.dtype)
    source, target = (x.view(torch.uint16), packed.view(torch.uint16))
    bc = _power_of_two(c)
    span = _power_of_two((c - 1) * x.stride(2) + 1)
    _copy_gather.run(
        source,
        target,
        (t, h, c, *x.stride(), bt, bc, span),
        num_warps=4,
        grid=(min(h * ((t + bt - 1) // bt), 96),),
        warmup=False,
    )
    return packed


@triton.jit
def _prepare_q_rope(Q, R, P, output_ptr, META: tl.constexpr):
    T: tl.constexpr = META[0]
    H: tl.constexpr = META[1]
    K: tl.constexpr = META[2]
    N: tl.constexpr = META[3]
    D: tl.constexpr = META[4]
    Q0: tl.constexpr = META[5]
    Q1: tl.constexpr = META[6]
    Q2: tl.constexpr = META[7]
    R0: tl.constexpr = META[8]
    R1: tl.constexpr = META[9]
    R2: tl.constexpr = META[10]
    O0: tl.constexpr = META[11]
    O1: tl.constexpr = META[12]
    O2: tl.constexpr = META[13]
    BT: tl.constexpr = META[14]
    BK: tl.constexpr = META[15]
    BD: tl.constexpr = META[16]
    SQ: tl.constexpr = META[17]
    SR: tl.constexpr = META[18]
    COPY_Q: tl.constexpr = META[19]
    GATHER_Q: tl.constexpr = META[20]
    GATHER_R: tl.constexpr = META[21]
    WIDE: tl.constexpr = (
        (
            (T - 1) * Q0 + (H - 1) * Q1 + (K - 1) * Q2 > 2147483647
            or (T - 1) * R0 + (H - 1) * R1 + (D - 1) * R2 > 2147483647
            or (T - 1) * O0 + (H - 1) * O1 + (N + D - 1) * O2 > 2147483647
            or (T * H * K > 2147483647)
        )
        or Q0 > 2147483647
        or Q1 > 2147483647
        or (Q2 > 2147483647)
        or (R0 > 2147483647)
        or (R1 > 2147483647)
        or (R2 > 2147483647)
        or (O0 > 2147483647)
        or (O1 > 2147483647)
        or (O2 > 2147483647)
    )
    tm = tl.cdiv(T, BT)
    PROGRAMS: tl.constexpr = H * ((T + BT - 1) // BT)
    LOOP_WIDE: tl.constexpr = (
        PROGRAMS - 1 + (PROGRAMS if PROGRAMS < 96 else 96) > 2147483647
        or (T + BT - 1) // BT * BT - 1 > 2147483647
    )
    if LOOP_WIDE:
        loop_stop = tl.full((), PROGRAMS, tl.int64)
    else:
        loop_stop = PROGRAMS
    for pid in range(
        _address_index(tl.program_id(0), LOOP_WIDE),
        loop_stop,
        _address_index(tl.num_programs(0), LOOP_WIDE),
    ):
        h = pid // tm
        t = pid % tm * BT + tl.arange(0, BT)
        if COPY_Q:
            k = tl.arange(0, BK)
            if GATHER_Q:
                physical = tl.arange(0, SQ)
                raw = tl.load(
                    Q
                    + _address_index(t[:, None], WIDE) * Q0
                    + _address_index(h, WIDE) * Q1
                    + physical[None, :],
                    (t < T)[:, None] & (physical < (K - 1) * Q2 + 1)[None, :],
                    other=0,
                )
                indices = tl.broadcast_to(
                    (tl.minimum(k, K - 1) * Q2)[None, :], (BT, BK)
                )
                q = tl.gather(
                    raw.to(tl.float16, bitcast=True), indices, axis=1
                ).to(tl.uint16, bitcast=True)
            else:
                q = tl.load(
                    Q
                    + _address_index(t[:, None], WIDE) * Q0
                    + _address_index(h, WIDE) * Q1
                    + _address_index(k[None, :], WIDE) * Q2,
                    (t < T)[:, None] & (k < K)[None, :],
                    other=0,
                )
            tl.store(
                P
                + _address_index(
                    _address_index(t[:, None], WIDE) * H + h, WIDE
                )
                * K
                + k[None, :],
                q,
                (t < T)[:, None] & (k < K)[None, :],
            )
        d = tl.arange(0, BD)
        if GATHER_R:
            physical = tl.arange(0, SR)
            raw = tl.load(
                R
                + _address_index(t[:, None], WIDE) * R0
                + _address_index(h, WIDE) * R1
                + physical[None, :],
                (t < T)[:, None] & (physical < (D - 1) * R2 + 1)[None, :],
                other=0,
            )
            indices = tl.broadcast_to(
                (tl.minimum(d, D - 1) * R2)[None, :], (BT, BD)
            )
            r = tl.gather(
                raw.to(tl.float16, bitcast=True), indices, axis=1
            ).to(tl.uint16, bitcast=True)
        else:
            r = tl.load(
                R
                + _address_index(t[:, None], WIDE) * R0
                + _address_index(h, WIDE) * R1
                + _address_index(d[None, :], WIDE) * R2,
                (t < T)[:, None] & (d < D)[None, :],
                other=0,
            )
        v = (r.to(tl.uint32) << 16).to(tl.float32, bitcast=True)
        tl.store(
            output_ptr
            + _address_index(t[:, None], WIDE) * O0
            + _address_index(h, WIDE) * O1
            + _address_index(N + _address_index(d[None, :], WIDE), WIDE) * O2,
            _encode_narrow(v),
            (t < T)[:, None] & (d < D)[None, :],
        )


def _prepare(q_nope, q_rope, out, t, h, k, n, d):
    copy_q = q_nope.stride(2) != 1
    q = q_nope.new_empty((t, h, k), dtype=q_nope.dtype) if copy_q else q_nope
    qs, rs = (q_nope.stride(2), q_rope.stride(2))
    gq = (
        0 < k <= 256
        and qs > 1
        and qs & qs - 1
        and (t * h * k >= 131072)
        and ((k - 1) * qs + 1 <= 2048)
    )
    gr = (
        0 < d <= 256
        and rs > 1
        and rs & rs - 1
        and (t * h * d >= 131072)
        and ((d - 1) * rs + 1 <= 2048)
    )
    sq = _power_of_two((k - 1) * qs + 1) if gq else 1
    sr = _power_of_two((d - 1) * rs + 1) if gr else 1
    _prepare_q_rope.run(
        q_nope.view(torch.uint16),
        q_rope.view(torch.uint16),
        q.view(torch.uint16),
        out,
        (
            t,
            h,
            k,
            n,
            d,
            *q_nope.stride(),
            *q_rope.stride(),
            *out.stride(),
            8,
            _power_of_two(k),
            _power_of_two(d),
            sq,
            sr,
            copy_q,
            bool(gq),
            bool(gr),
        ),
        num_warps=4,
        grid=(min(h * ((t + 8 - 1) // 8), 96),),
        warmup=False,
    )
    return q


def _scatter_output_original(source, destination, t, h, width):
    _copy_layout.run(
        source,
        destination,
        (t, h, width, *source.stride(), *destination.stride(), 16, 128),
        num_warps=4,
        grid=(min(h * ((t + 15) // 16) * ((width + 127) // 128), 96),),
        warmup=False,
    )


def absorbed_bmm_concat_cast_q_fp8(q_fp8_pad, q_nope, w_kc, q_rope, num_heads):
    t, _, k = q_nope.shape
    h = num_heads
    n, d = (w_kc.shape[-1], q_rope.shape[-1])
    if t == 0 or h == 0:
        return q_fp8_pad
    raw = q_fp8_pad.view(torch.uint8)
    t_full = t
    width = n + d
    if width == 0:
        return q_fp8_pad
    qs, ws = (
        (q_nope.stride(), w_kc.stride()) if n > 0 else ((0, 0, 0), (0, 0, 0))
    )
    rs, os = q_rope.stride(), raw.stride()
    destination = (
        raw
        if (
            os[2] != 1
            and (n == 0 or t * h * k * n >= 65536)
            or _active_inputs_overlap(
                t,
                h,
                k,
                n,
                d,
                q_fp8_pad.data_ptr(),
                os,
                q_nope.data_ptr() if n > 0 else 0,
                qs,
                w_kc.data_ptr() if n > 0 else 0,
                ws,
                q_rope.data_ptr() if d > 0 else 0,
                rs,
            )
        )
        else None
    )
    if destination is not None:
        raw = q_nope.new_empty((t, h, width), dtype=torch.uint8)
        os = raw.stride()
    if n == 0:
        if d > 0:
            rope_raw = q_rope.view(torch.uint16)
            _rope_rows.run(
                rope_raw,
                raw,
                (t, h, 0, d, *rs, *os, 16, 64),
                num_warps=4,
                grid=(min(h * ((t + 15) // 16) * ((d + 63) // 64), 96),),
                warmup=False,
            )
        if destination is not None:
            _scatter_output(raw, destination, t_full, h, width)
        return q_fp8_pad
    prepare = (
        t >= 16
        and t * h * k * n >= 1048576
        and (k % 16 == 0)
        and (n % 64 == 0)
        and (d % 64 == 0)
        and (os[2] == 1)
    )
    if prepare and 0 < k <= 256 and (0 < d <= 256) and (qs[2] != 1):
        q_nope = _prepare(q_nope, q_rope, raw, t, h, k, n, d)
        qs = q_nope.stride()
        rope_raw = q_nope
        d = 0
    else:
        if prepare:
            if qs[2] != 1:
                q_nope = _pack_rows(q_nope, t, h, k)
                qs = q_nope.stride()
            if d > 0 and rs[2] != 1:
                q_rope = _pack_rows(q_rope, t, h, d)
                rs = q_rope.stride()
        rope_raw = q_rope.view(torch.uint16)
    fallback_bn = min(32, 1024 // _power_of_two(k)) if os[2] != 1 else 32
    compatible = (
        k % 16 == 0
        and n % 64 == 0
        and (d % 64 == 0)
        and (qs[2] == 1)
        and (d == 0 or rs[2] == 1)
        and (os[2] == 1)
    )
    if t >= 16 and t % 16 != 0 and compatible:
        whole = t - t % 16
        bm, bn = (
            64 if whole >= 64 and whole % 64 == 0 and (n % 128 == 0) else 32,
            128,
        )
        bm = min(bm, whole)
        tm, tn = ((t + bm - 1) // bm, (n + bn - 1) // bn)
        narrow = (whole * h * n >= 16777216 or n <= 128) and bm == 64
        bn = 256 if narrow and n % 256 == 0 else bn
        tn = (n + bn - 1) // bn
        split_rope = narrow and n > 128
        kernel = _fused_narrow_shift_window if narrow else _fused_shift_window
        kernel.run(
            q_nope,
            w_kc,
            rope_raw,
            raw,
            (
                t,
                h,
                k,
                n,
                0 if split_rope else d,
                *qs,
                *ws,
                *rs,
                *os,
                tm,
                tn,
                bm,
                bn,
                min(128, k & -k),
                64,
                False,
            ),
            num_warps=4,
            num_stages=1,
            grid=(min(h * tm * tn, 96),),
            warmup=False,
        )
        if split_rope and d > 0:
            _rope_rows.run(
                rope_raw,
                raw,
                (t, h, n, d, *rs, *os, 16, 64),
                num_warps=4,
                grid=(min(h * ((t + 15) // 16) * ((d + 63) // 64), 96),),
                warmup=False,
            )
        if destination is not None:
            _scatter_output(raw, destination, t_full, h, width)
        return q_fp8_pad
    if t % 16 != 0 or not compatible:
        _fallback.run(
            q_nope,
            w_kc,
            rope_raw,
            raw,
            (
                t,
                h,
                k,
                n,
                d,
                *qs,
                *ws,
                *rs,
                *os,
                _power_of_two(k),
                fallback_bn,
                False,
            ),
            num_warps=4,
            grid=(min(t * h * ((n + fallback_bn - 1) // fallback_bn), 32768),),
            warmup=False,
        )
        if destination is not None:
            _scatter_output(raw, destination, t_full, h, width)
        return q_fp8_pad
    bm, bn = (64 if t >= 64 and t % 64 == 0 and (n % 128 == 0) else 32, 128)
    tm, tn = ((t + bm - 1) // bm, (n + bn - 1) // bn)
    narrow = (t * h * n >= 16777216 or n <= 128) and bm == 64
    bn = 256 if narrow and n % 256 == 0 else bn
    tn = (n + bn - 1) // bn
    split_rope = narrow and n > 128
    kernel = _fused_narrow if narrow else _fused
    kernel.run(
        q_nope,
        w_kc,
        rope_raw,
        raw,
        (
            t,
            h,
            k,
            n,
            0 if split_rope else d,
            *qs,
            *ws,
            *rs,
            *os,
            tm,
            tn,
            bm,
            bn,
            min(128, k & -k),
            64,
            False,
        ),
        num_warps=4,
        num_stages=1,
        grid=(min(h * tm * tn, 96),),
        warmup=False,
    )
    if split_rope and d > 0:
        _rope_rows.run(
            rope_raw,
            raw,
            (t, h, n, d, *rs, *os, 16, 64),
            num_warps=4,
            grid=(min(h * ((t + 15) // 16) * ((d + 63) // 64), 96),),
            warmup=False,
        )
    if destination is not None:
        _scatter_output(raw, destination, t_full, h, width)
    return q_fp8_pad


@triton.jit
def _scatter_inverse(X, output_ptr, META: tl.constexpr):
    T: tl.constexpr = META[0]
    H: tl.constexpr = META[1]
    C: tl.constexpr = META[2]
    X0: tl.constexpr = META[3]
    X1: tl.constexpr = META[4]
    X2: tl.constexpr = META[5]
    O0: tl.constexpr = META[6]
    O1: tl.constexpr = META[7]
    O2: tl.constexpr = META[8]
    B: tl.constexpr = META[9]
    SPAN: tl.constexpr = (C - 1) * O2 + 1
    BLOCKS: tl.constexpr = (SPAN + B - 1) // B
    WIDE: tl.constexpr = (
        (T - 1) * X0 + (H - 1) * X1 + (C - 1) * X2 > 2147483647
        or (T - 1) * O0 + (H - 1) * O1 + SPAN > 2147483647
        or X0 > 2147483647
        or (X1 > 2147483647)
        or (X2 > 2147483647)
        or (O0 > 2147483647)
        or (O1 > 2147483647)
        or (O2 > 2147483647)
    )
    for pid in range(tl.program_id(0), T * H * BLOCKS, tl.num_programs(0)):
        row = pid // BLOCKS
        t = row // H
        h = row % H
        position = pid % BLOCKS * B + tl.arange(0, B)
        c = position // O2
        valid = (position < SPAN) & (position % O2 == 0)
        value = tl.load(
            X
            + _address_index(t, WIDE) * X0
            + _address_index(h, WIDE) * X1
            + _address_index(c, WIDE) * X2,
            valid,
            other=0,
        )
        tl.store(
            output_ptr
            + _address_index(t, WIDE) * O0
            + _address_index(h, WIDE) * O1
            + _address_index(position, WIDE),
            value,
            valid,
        )


def _scatter_output_prior(source, destination, t, h, width):
    if destination.stride(2) <= 1:
        return _scatter_output_original(source, destination, t, h, width)
    span = (width - 1) * destination.stride(2) + 1
    if span > 2048 or t * h * width < 4096:
        return _scatter_output_original(source, destination, t, h, width)
    block = min(2048, _power_of_two(span))
    programs = t * h * ((span + block - 1) // block)
    if span > 2147483647 or programs + min(programs, 96) > 2147483647:
        return _scatter_output_original(source, destination, t, h, width)
    _scatter_inverse.run(
        source,
        destination,
        (t, h, width, *source.stride(), *destination.stride(), block),
        num_warps=4,
        grid=(min(programs, 96),),
        warmup=False,
    )


@triton.jit
def _scatter_word(X, output_ptr, P, META: tl.constexpr):
    T: tl.constexpr = META[0]
    H: tl.constexpr = META[1]
    C: tl.constexpr = META[2]
    X0: tl.constexpr = META[3]
    X1: tl.constexpr = META[4]
    X2: tl.constexpr = META[5]
    O0: tl.constexpr = META[6]
    O1: tl.constexpr = META[7]
    SHIFT: tl.constexpr = META[8]
    B: tl.constexpr = META[9]
    WIDE: tl.constexpr = (
        (T - 1) * X0 + (H - 1) * X1 + (C - 1) * X2 > 2147483647
        or (T - 1) * O0 + (H - 1) * O1 + (C - 1) * 2 > 2147483647
        or X0 > 2147483647
        or (X1 > 2147483647)
        or (X2 > 2147483647)
        or (O0 > 2147483647)
        or (O1 > 2147483647)
    )
    for row in range(tl.program_id(0), T * H, tl.num_programs(0)):
        t = row // H
        h = row % H
        word = tl.arange(0, B)
        c = word + SHIFT
        source = (
            X + _address_index(t, WIDE) * X0 + _address_index(h, WIDE) * X1
        )
        target = (
            output_ptr
            + _address_index(t, WIDE) * O0
            + _address_index(h, WIDE) * O1
        )
        ptr = (
            P
            + _address_index(t, WIDE) * (O0 // 2)
            + _address_index(h, WIDE) * (O1 // 2)
        )
        prior = tl.load(ptr + word, word < C - 1, other=0)
        value = tl.load(
            source + _address_index(c, WIDE) * X2, word < C - 1, other=0
        ).to(tl.uint16)
        if SHIFT == 0:
            updated = prior & 65280 | value
        else:
            updated = prior & 255 | value << 8
        tl.store(ptr + word, updated, word < C - 1)
        edge: tl.constexpr = 0 if SHIFT else C - 1
        edge_value = tl.load(source + edge * X2)
        tl.store(target + edge * 2, edge_value)


def _scatter_output_word_or_prior(source, destination, t, h, width):
    o0, o1, o2 = destination.stride()
    if (
        o2 != 2
        or width < 512
        or width > 1024
        or (t * h * width < 65536)
        or o1 % 2
        or o0 % 2
        or (o1 < 2 * width)
        or (o0 < h * o1)
        or (t * h + min(t * h, 96) > 2147483647)
    ):
        return _scatter_output_prior(source, destination, t, h, width)
    shift = destination.data_ptr() % 2
    packed = destination.as_strided(
        (2 * (width - 1),), (1,), destination.storage_offset() + shift
    ).view(torch.uint16)
    _scatter_word.run(
        source,
        destination,
        packed,
        (
            t,
            h,
            width,
            *source.stride(),
            o0,
            o1,
            shift,
            _power_of_two(width - 1),
        ),
        num_warps=4,
        grid=(min(t * h, 96),),
        warmup=False,
    )


@triton.jit
def _scatter_physical(X, output_ptr, META: tl.constexpr):
    T: tl.constexpr = META[0]
    H: tl.constexpr = META[1]
    C: tl.constexpr = META[2]
    X0: tl.constexpr = META[3]
    X1: tl.constexpr = META[4]
    X2: tl.constexpr = META[5]
    O0: tl.constexpr = META[6]
    O1: tl.constexpr = META[7]
    O2: tl.constexpr = META[8]
    BC: tl.constexpr = META[9]
    BP: tl.constexpr = META[10]
    SPAN: tl.constexpr = (C - 1) * O2 + 1
    WIDE: tl.constexpr = (
        (T - 1) * X0 + (H - 1) * X1 + (C - 1) * X2 > 2147483647
        or (T - 1) * O0 + (H - 1) * O1 + SPAN > 2147483647
        or X0 > 2147483647
        or (X1 > 2147483647)
        or (X2 > 2147483647)
        or (O0 > 2147483647)
        or (O1 > 2147483647)
        or (O2 > 2147483647)
    )
    for row in range(tl.program_id(0), T * H, tl.num_programs(0)):
        t = row // H
        h = row % H
        c = tl.arange(0, BC)
        position = tl.arange(0, BP)
        value = tl.load(
            X
            + _address_index(t, WIDE) * X0
            + _address_index(h, WIDE) * X1
            + _address_index(c, WIDE) * X2,
            c < C,
            other=0,
        ).to(tl.float16)
        index = tl.minimum(position // O2, C - 1)
        expanded = tl.gather(value, index, axis=0).to(tl.uint8)
        target = (
            output_ptr
            + _address_index(t, WIDE) * O0
            + _address_index(h, WIDE) * O1
        )
        prior = tl.load(target + position, position < SPAN, other=0)
        updated = tl.where(position % O2 == 0, expanded, prior)
        tl.store(target + position, updated, position < SPAN)


def _scatter_output(source, destination, t, h, width):
    o0, o1, o2 = destination.stride()
    span = (width - 1) * o2 + 1
    if (
        o2 <= 2
        or width < 256
        or span > 4096
        or (t * h * width < 65536)
        or (o1 < span)
        or (o0 < h * o1)
        or (t * h + min(t * h, 96) > 2147483647)
    ):
        return _scatter_output_word_or_prior(source, destination, t, h, width)
    _scatter_physical.run(
        source,
        destination,
        (
            t,
            h,
            width,
            *source.stride(),
            *destination.stride(),
            _power_of_two(width),
            _power_of_two(span),
        ),
        num_warps=4,
        grid=(min(t * h, 96),),
        warmup=False,
    )


@triton.jit
def _fused_shift_window(X, W, R, output_ptr, META: tl.constexpr):
    T: tl.constexpr = META[0]
    H: tl.constexpr = META[1]
    K: tl.constexpr = META[2]
    N: tl.constexpr = META[3]
    D: tl.constexpr = META[4]
    X0: tl.constexpr = META[5]
    X1: tl.constexpr = META[6]
    X2: tl.constexpr = META[7]
    W0: tl.constexpr = META[8]
    W1: tl.constexpr = META[9]
    W2: tl.constexpr = META[10]
    R0: tl.constexpr = META[11]
    R1: tl.constexpr = META[12]
    R2: tl.constexpr = META[13]
    O0: tl.constexpr = META[14]
    O1: tl.constexpr = META[15]
    O2: tl.constexpr = META[16]
    TM: tl.constexpr = META[17]
    TN: tl.constexpr = META[18]
    BM: tl.constexpr = META[19]
    BN: tl.constexpr = META[20]
    BK: tl.constexpr = META[21]
    BR: tl.constexpr = META[22]
    BIT_ROUND: tl.constexpr = META[23]
    WIDE: tl.constexpr = (
        (
            (T - 1) * X0 + (H - 1) * X1 + (K - 1) * X2 > 2147483647
            or (H - 1) * W0 + (K - 1) * W1 + (N - 1) * W2 > 2147483647
            or (T - 1) * R0 + (H - 1) * R1 + (D - 1) * R2 > 2147483647
            or ((T - 1) * O0 + (H - 1) * O1 + (N + D - 1) * O2 > 2147483647)
        )
        or X0 > 2147483647
        or X1 > 2147483647
        or (X2 > 2147483647)
        or (W0 > 2147483647)
        or (W1 > 2147483647)
        or (W2 > 2147483647)
        or (R0 > 2147483647)
        or (R1 > 2147483647)
        or (R2 > 2147483647)
        or (O0 > 2147483647)
        or (O1 > 2147483647)
        or (O2 > 2147483647)
    )
    PROGRAMS: tl.constexpr = H * TM * TN
    LOOP_WIDE: tl.constexpr = PROGRAMS - 1 + (
        PROGRAMS if PROGRAMS < 96 else 96
    ) > 2147483647 or (TM * BM - 1 > 2147483647 or TN * BN - 1 > 2147483647)
    if LOOP_WIDE:
        loop_stop = tl.full((), PROGRAMS, tl.int64)
    else:
        loop_stop = PROGRAMS
    for pid in range(
        _address_index(tl.program_id(0), LOOP_WIDE),
        loop_stop,
        _address_index(tl.num_programs(0), LOOP_WIDE),
    ):
        h = pid // (TM * TN)
        m = tl.minimum(pid // TN % TM * BM, T - BM) + tl.arange(0, BM)
        n = pid % TN * BN + tl.arange(0, BN)
        k = tl.arange(0, BK)
        acc = tl.zeros((BM, BN), tl.float32)
        for ki in range(tl.cdiv(K, BK)):
            kk = ki * BK + k
            x = tl.load(
                X
                + _address_index(m[:, None], WIDE) * X0
                + _address_index(h, WIDE) * X1
                + _address_index(kk[None, :], WIDE) * X2,
                (kk < K)[None, :],
                other=0,
            )
            w = tl.load(
                W
                + _address_index(h, WIDE) * W0
                + _address_index(kk[:, None], WIDE) * W1
                + _address_index(n[None, :], WIDE) * W2,
                (kk < K)[:, None] & (n < N)[None, :],
                other=0,
            )
            acc = tl.dot(x, w, acc, input_precision="ieee")
        if BIT_ROUND:
            bits = acc.to(tl.uint32, bitcast=True)
            rounded = (bits + 32767 + (bits >> 16 & 1) & 4294901760).to(
                tl.float32, bitcast=True
            )
            rounded = tl.where(bits & 2139095040 == 2139095040, acc, rounded)
        else:
            rounded = acc.to(tl.bfloat16).to(tl.float32)
        code = _encode(rounded)
        tl.store(
            output_ptr
            + _address_index(m[:, None], WIDE) * O0
            + _address_index(h, WIDE) * O1
            + _address_index(n[None, :], WIDE) * O2,
            code,
            (n < N)[None, :] & (m >= pid // TN % TM * BM)[:, None],
        )
        if D > 0:
            if pid % TN == 0:
                ROPE_PROGRAMS: tl.constexpr = (D + BR - 1) // BR
                ROPE_WIDE: tl.constexpr = (
                    ROPE_PROGRAMS > 2147483647
                    or N + ROPE_PROGRAMS * BR - 1 > 2147483647
                )
                if ROPE_WIDE:
                    rope_start = tl.full((), 0, tl.int64)
                    rope_stop = tl.full((), ROPE_PROGRAMS, tl.int64)
                    rope_step = tl.full((), 1, tl.int64)
                else:
                    rope_start = 0
                    rope_stop = ROPE_PROGRAMS
                    rope_step = 1
                for ri in range(rope_start, rope_stop, rope_step):
                    r = ri * BR + tl.arange(0, BR)
                    value = (
                        tl.load(
                            R
                            + _address_index(m[:, None], WIDE) * R0
                            + _address_index(h, WIDE) * R1
                            + _address_index(r[None, :], WIDE) * R2,
                            (r < D)[None, :],
                            other=0,
                        ).to(tl.uint32)
                        << 16
                    ).to(tl.float32, bitcast=True)
                    tl.store(
                        output_ptr
                        + _address_index(m[:, None], WIDE) * O0
                        + _address_index(h, WIDE) * O1
                        + _address_index(
                            N + _address_index(r[None, :], WIDE), WIDE
                        )
                        * O2,
                        _encode(value),
                        (r < D)[None, :] & (m >= pid // TN % TM * BM)[:, None],
                    )


@triton.jit
def _fused_narrow_shift_window(X, W, R, output_ptr, META: tl.constexpr):
    T: tl.constexpr = META[0]
    H: tl.constexpr = META[1]
    K: tl.constexpr = META[2]
    N: tl.constexpr = META[3]
    D: tl.constexpr = META[4]
    X0: tl.constexpr = META[5]
    X1: tl.constexpr = META[6]
    X2: tl.constexpr = META[7]
    W0: tl.constexpr = META[8]
    W1: tl.constexpr = META[9]
    W2: tl.constexpr = META[10]
    R0: tl.constexpr = META[11]
    R1: tl.constexpr = META[12]
    R2: tl.constexpr = META[13]
    O0: tl.constexpr = META[14]
    O1: tl.constexpr = META[15]
    O2: tl.constexpr = META[16]
    TM: tl.constexpr = META[17]
    TN: tl.constexpr = META[18]
    BM: tl.constexpr = META[19]
    BN: tl.constexpr = META[20]
    BK: tl.constexpr = META[21]
    BR: tl.constexpr = META[22]
    BIT_ROUND: tl.constexpr = META[23]
    WIDE: tl.constexpr = (
        (
            (T - 1) * X0 + (H - 1) * X1 + (K - 1) * X2 > 2147483647
            or (H - 1) * W0 + (K - 1) * W1 + (N - 1) * W2 > 2147483647
            or (T - 1) * R0 + (H - 1) * R1 + (D - 1) * R2 > 2147483647
            or ((T - 1) * O0 + (H - 1) * O1 + (N + D - 1) * O2 > 2147483647)
        )
        or X0 > 2147483647
        or X1 > 2147483647
        or (X2 > 2147483647)
        or (W0 > 2147483647)
        or (W1 > 2147483647)
        or (W2 > 2147483647)
        or (R0 > 2147483647)
        or (R1 > 2147483647)
        or (R2 > 2147483647)
        or (O0 > 2147483647)
        or (O1 > 2147483647)
        or (O2 > 2147483647)
    )
    PROGRAMS: tl.constexpr = H * TM * TN
    LOOP_WIDE: tl.constexpr = PROGRAMS - 1 + (
        PROGRAMS if PROGRAMS < 96 else 96
    ) > 2147483647 or (TM * BM - 1 > 2147483647 or TN * BN - 1 > 2147483647)
    if LOOP_WIDE:
        loop_stop = tl.full((), PROGRAMS, tl.int64)
    else:
        loop_stop = PROGRAMS
    for pid in range(
        _address_index(tl.program_id(0), LOOP_WIDE),
        loop_stop,
        _address_index(tl.num_programs(0), LOOP_WIDE),
    ):
        h = pid // (TM * TN)
        m = tl.minimum(pid // TN % TM * BM, T - BM) + tl.arange(0, BM)
        n = pid % TN * BN + tl.arange(0, BN)
        k = tl.arange(0, BK)
        acc = tl.zeros((BM, BN), tl.float32)
        for ki in range(tl.cdiv(K, BK)):
            kk = ki * BK + k
            x = tl.load(
                X
                + _address_index(m[:, None], WIDE) * X0
                + _address_index(h, WIDE) * X1
                + _address_index(kk[None, :], WIDE) * X2,
                (kk < K)[None, :],
                other=0,
            )
            w = tl.load(
                W
                + _address_index(h, WIDE) * W0
                + _address_index(kk[:, None], WIDE) * W1
                + _address_index(n[None, :], WIDE) * W2,
                (kk < K)[:, None] & (n < N)[None, :],
                other=0,
            )
            acc = tl.dot(x, w, acc, input_precision="ieee")
        if BIT_ROUND:
            bits = acc.to(tl.uint32, bitcast=True)
            rounded = (bits + 32767 + (bits >> 16 & 1) & 4294901760).to(
                tl.float32, bitcast=True
            )
            rounded = tl.where(bits & 2139095040 == 2139095040, acc, rounded)
        else:
            rounded = acc.to(tl.bfloat16).to(tl.float32)
        code = _encode_narrow(rounded)
        tl.store(
            output_ptr
            + _address_index(m[:, None], WIDE) * O0
            + _address_index(h, WIDE) * O1
            + _address_index(n[None, :], WIDE) * O2,
            code,
            (n < N)[None, :] & (m >= pid // TN % TM * BM)[:, None],
        )
        if D > 0:
            if pid % TN == 0:
                ROPE_PROGRAMS: tl.constexpr = (D + BR - 1) // BR
                ROPE_WIDE: tl.constexpr = (
                    ROPE_PROGRAMS > 2147483647
                    or N + ROPE_PROGRAMS * BR - 1 > 2147483647
                )
                if ROPE_WIDE:
                    rope_start = tl.full((), 0, tl.int64)
                    rope_stop = tl.full((), ROPE_PROGRAMS, tl.int64)
                    rope_step = tl.full((), 1, tl.int64)
                else:
                    rope_start = 0
                    rope_stop = ROPE_PROGRAMS
                    rope_step = 1
                for ri in range(rope_start, rope_stop, rope_step):
                    r = ri * BR + tl.arange(0, BR)
                    value = (
                        tl.load(
                            R
                            + _address_index(m[:, None], WIDE) * R0
                            + _address_index(h, WIDE) * R1
                            + _address_index(r[None, :], WIDE) * R2,
                            (r < D)[None, :],
                            other=0,
                        ).to(tl.uint32)
                        << 16
                    ).to(tl.float32, bitcast=True)
                    tl.store(
                        output_ptr
                        + _address_index(m[:, None], WIDE) * O0
                        + _address_index(h, WIDE) * O1
                        + _address_index(
                            N + _address_index(r[None, :], WIDE), WIDE
                        )
                        * O2,
                        _encode_narrow(value),
                        (r < D)[None, :] & (m >= pid // TN % TM * BM)[:, None],
                    )


def _active_inputs_overlap(
    t,
    h,
    k,
    n,
    d,
    output_address,
    os,
    q_address,
    xs,
    w_address,
    ws,
    r_address,
    rs,
):
    if t == 0 or h == 0 or n + d == 0:
        return False
    output_end = (
        output_address
        + 1
        + (t - 1) * os[0]
        + (h - 1) * os[1]
        + (n + d - 1) * os[2]
    )
    if n > 0 and k > 0:
        if q_address < output_end and output_address < q_address + 2 * (
            1 + (t - 1) * xs[0] + (h - 1) * xs[1] + (k - 1) * xs[2]
        ):
            return True
        if w_address < output_end and output_address < w_address + 2 * (
            1 + (h - 1) * ws[0] + (k - 1) * ws[1] + (n - 1) * ws[2]
        ):
            return True
    if (
        d > 0
        and r_address < output_end
        and (
            output_address
            < r_address
            + 2 * (1 + (t - 1) * rs[0] + (h - 1) * rs[1] + (d - 1) * rs[2])
        )
    ):
        return True
    return False


def _output_overlaps_inputs(output, q, w, r, t, h):
    k, n, d = (q.shape[-1], w.shape[-1], r.shape[-1])
    if t == 0 or h == 0 or n + d == 0:
        return False
    qa, xs = (q.data_ptr(), q.stride()) if n > 0 else (0, (0, 0, 0))
    wa, ws = (w.data_ptr(), w.stride()) if n > 0 else (0, (0, 0, 0))
    ra, rs = (r.data_ptr(), r.stride()) if d > 0 else (0, (0, 0, 0))
    return _active_inputs_overlap(
        t,
        h,
        k,
        n,
        d,
        output.data_ptr(),
        output.stride(),
        qa,
        xs,
        wa,
        ws,
        ra,
        rs,
    )


__all__ = ["absorbed_bmm_concat_cast_q_fp8"]
