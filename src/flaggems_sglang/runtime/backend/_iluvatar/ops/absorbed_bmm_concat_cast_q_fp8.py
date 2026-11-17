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


def _large_matrix_prefix(whole, h, k, n):
    return (
        whole >= 256
        and whole % 128 == 0
        and (whole * h * n >= 16777216)
        and (k & -k >= 32)
    )


def _q_alignment_preparation_compatible(t, n, strides, address):
    return n % 128 == 0 and (
        strides[2] == 1
        and strides[0] % 64 == 0
        and (strides[1] % 64 == 0)
        and (address % 64 == 0 or (t % 128 == 0 and n % 128 == 0))
    )


def _matrix_loadable(tensor, width, strides=None):
    if tensor.is_contiguous():
        return tensor.data_ptr() % 256 == 0
    if strides is None:
        strides = tensor.stride()
    return (
        strides[2] == 1
        and width % 32 == 0
        and (strides[0] % 32 == 0)
        and (strides[1] % 32 == 0)
        and (strides[1] >= width)
        and (strides[0] >= tensor.shape[1] * strides[1])
        and (tensor.data_ptr() % 256 == 0)
    )


@triton.jit
def _address_index(index, WIDE: tl.constexpr):
    if WIDE:
        return index.to(tl.int64)
    else:
        return index


@triton.jit
def _encode0_group8(v):
    raw = v.to(tl.uint32, bitcast=True)
    mag = raw & 2147483647
    bf = mag >> 16
    bf = tl.minimum(bf, 17392)
    half = (bf << 16).to(tl.float32, bitcast=True).to(tl.float16)
    exponent = tl.maximum((bf >> 7).to(tl.uint16), tl.full((), 121, tl.uint16))
    magic_bits = exponent * tl.full((), 1032, tl.uint16) + tl.full(
        (), 22584, tl.uint16
    )
    magic = magic_bits.to(tl.float16, bitcast=True)
    rounded = (half + magic).to(tl.uint16, bitcast=True)
    return (rounded | (raw >> 24 & 128).to(tl.uint16)).to(tl.uint8)


@triton.jit
def _fused0_group8(X, W, R, output_ptr, META: tl.constexpr):
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
        PROGRAMS if PROGRAMS < 32768 else 32768
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
        local = pid % (TM * TN)
        head = pid // (TM * TN)
        gm = 8
        group = local // (gm * TN)
        size = tl.minimum(TM - group * gm, gm)
        mi = group * gm + local % size
        ni = local % (gm * TN) // size
        pid = head * (TM * TN) + mi * TN + ni
        h = pid // (TM * TN)
        m = pid // TN % TM * BM + tl.arange(0, BM)
        n = pid % TN * BN + tl.arange(0, BN)
        k = tl.arange(0, BK)
        acc = tl.zeros((BM, BN), tl.float32)
        for ki in range(tl.cdiv(K, BK)):
            kk = ki * BK + k
            if T % BM == 0 and K % BK == 0:
                x = tl.load(
                    X
                    + _address_index(m[:, None], WIDE) * X0
                    + _address_index(h, WIDE) * X1
                    + _address_index(kk[None, :], WIDE) * X2
                )
            else:
                x = tl.load(
                    X
                    + _address_index(m[:, None], WIDE) * X0
                    + _address_index(h, WIDE) * X1
                    + _address_index(kk[None, :], WIDE) * X2,
                    (m < T)[:, None] & (kk < K)[None, :],
                    other=0,
                )
            if K % BK == 0 and N % BN == 0:
                w = tl.load(
                    W
                    + _address_index(h, WIDE) * W0
                    + _address_index(kk[:, None], WIDE) * W1
                    + _address_index(n[None, :], WIDE) * W2
                )
            else:
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
        code = _encode0_group8(rounded)
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
                    value = tl.load(
                        R
                        + _address_index(m[:, None], WIDE) * R0
                        + _address_index(h, WIDE) * R1
                        + _address_index(r[None, :], WIDE) * R2,
                        (m < T)[:, None] & (r < D)[None, :],
                        other=0,
                    ).to(tl.float32)
                    tl.store(
                        output_ptr
                        + _address_index(m[:, None], WIDE) * O0
                        + _address_index(h, WIDE) * O1
                        + _address_index(
                            N + _address_index(r[None, :], WIDE), WIDE
                        )
                        * O2,
                        _encode0_group8(value),
                        (m < T)[:, None] & (r < D)[None, :],
                    )


@triton.jit
def _encode1_group8(v):
    raw = v.to(tl.uint32, bitcast=True)
    mag = raw & 2147483647
    bf = mag >> 16
    bf = tl.minimum(bf, 17392)
    half = (bf << 16).to(tl.float32, bitcast=True).to(tl.float16)
    exponent = tl.maximum((bf >> 7).to(tl.uint16), tl.full((), 121, tl.uint16))
    magic_bits = exponent * tl.full((), 1032, tl.uint16) + tl.full(
        (), 22584, tl.uint16
    )
    magic = magic_bits.to(tl.float16, bitcast=True)
    rounded = (half + magic).to(tl.uint16, bitcast=True)
    return (rounded | (raw >> 24 & 128).to(tl.uint16)).to(tl.uint8)


@triton.jit
def _fused1_group8(X, W, R, output_ptr, META: tl.constexpr):
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
        PROGRAMS if PROGRAMS < 32768 else 32768
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
        local = pid % (TM * TN)
        head = pid // (TM * TN)
        gm = 8
        group = local // (gm * TN)
        size = tl.minimum(TM - group * gm, gm)
        mi = group * gm + local % size
        ni = local % (gm * TN) // size
        pid = head * (TM * TN) + mi * TN + ni
        h = pid // (TM * TN)
        m = pid // TN % TM * BM + tl.arange(0, BM)
        n = pid % TN * BN + tl.arange(0, BN)
        k = tl.arange(0, BK)
        acc = tl.zeros((BM, BN), tl.float32)
        for ki in range(tl.cdiv(K, BK)):
            kk = ki * BK + k
            if T % BM == 0 and K % BK == 0:
                x = tl.load(
                    X
                    + _address_index(m[:, None], WIDE) * X0
                    + _address_index(h, WIDE) * X1
                    + _address_index(kk[None, :], WIDE) * X2
                )
            else:
                x = tl.load(
                    X
                    + _address_index(m[:, None], WIDE) * X0
                    + _address_index(h, WIDE) * X1
                    + _address_index(kk[None, :], WIDE) * X2,
                    (m < T)[:, None] & (kk < K)[None, :],
                    other=0,
                )
            if K % BK == 0 and N % BN == 0:
                w = tl.load(
                    W
                    + _address_index(h, WIDE) * W0
                    + _address_index(kk[:, None], WIDE) * W1
                    + _address_index(n[None, :], WIDE) * W2
                )
            else:
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
        code = _encode1_group8(rounded)
        tl.store(
            output_ptr
            + _address_index(m[:, None], WIDE) * O0
            + _address_index(h, WIDE) * O1
            + _address_index(n[None, :], WIDE) * O2,
            code,
            (m < T)[:, None] & (n < N)[None, :],
        )


@triton.jit
def _copyrope1_group8_tile(R, output_ptr, META: tl.constexpr, PID):
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
    B: tl.constexpr = META[10]
    WIDE: tl.constexpr = (
        (
            (T - 1) * R0 + (H - 1) * R1 + (D - 1) * R2 > 2147483647
            or (T - 1) * O0 + (H - 1) * O1 + (N + D - 1) * O2 > 2147483647
            or T * H * D > 2147483647
        )
        or R0 > 2147483647
        or R1 > 2147483647
        or (R2 > 2147483647)
        or (O0 > 2147483647)
        or (O1 > 2147483647)
        or (O2 > 2147483647)
    )
    i = _address_index(PID, WIDE) * B + tl.arange(0, B)
    r = i % D
    h = i // D % H
    t = i // (H * D)
    x = tl.load(
        R
        + _address_index(t, WIDE) * R0
        + _address_index(h, WIDE) * R1
        + _address_index(r, WIDE) * R2,
        i < T * H * D,
        other=0,
    ).to(tl.float32)
    tl.store(
        output_ptr
        + _address_index(t, WIDE) * O0
        + _address_index(h, WIDE) * O1
        + _address_index(N + _address_index(r, WIDE), WIDE) * O2,
        _encode1_group8(x),
        i < T * H * D,
    )


@triton.jit
def _pack_strided_q_tile(X, P, META: tl.constexpr, PID):
    T: tl.constexpr = META[0]
    H: tl.constexpr = META[1]
    K: tl.constexpr = META[2]
    X0: tl.constexpr = META[3]
    X1: tl.constexpr = META[4]
    X2: tl.constexpr = META[5]
    B: tl.constexpr = META[6]
    WIDE: tl.constexpr = (
        (
            (T - 1) * X0 + (H - 1) * X1 + (K - 1) * X2 > 2147483647
            or T * H * K > 2147483647
        )
        or X0 > 2147483647
        or X1 > 2147483647
        or (X2 > 2147483647)
    )
    i = _address_index(PID, WIDE) * B + tl.arange(0, B)
    k = i % K
    h = i // K % H
    t = i // (K * H)
    v = tl.load(
        X
        + _address_index(t, WIDE) * X0
        + _address_index(h, WIDE) * X1
        + _address_index(k, WIDE) * X2,
        i < T * H * K,
        other=0,
    )
    tl.store(P + i, v, i < T * H * K)


@triton.jit
def _move_byte_buffer_tile(C, R, output_ptr, META: tl.constexpr, PID):
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
    B: tl.constexpr = META[10]
    WIDE: tl.constexpr = (
        (
            (T - 1) * R0 + (H - 1) * R1 + (D - 1) * R2 > 2147483647
            or (T - 1) * O0 + (H - 1) * O1 + (N + D - 1) * O2 > 2147483647
            or T * H * D > 2147483647
            or (T * H * N > 2147483647)
        )
        or R0 > 2147483647
        or R1 > 2147483647
        or (R2 > 2147483647)
        or (O0 > 2147483647)
        or (O1 > 2147483647)
        or (O2 > 2147483647)
    )
    i = _address_index(PID, WIDE) * B + tl.arange(0, B)
    row = i // N
    n = i % N
    t = row // H
    h = row % H
    ci = i
    value = tl.load(C + ci, i < T * H * N, other=0)
    tl.store(
        output_ptr
        + _address_index(t, WIDE) * O0
        + _address_index(h, WIDE) * O1
        + _address_index(n, WIDE) * O2,
        value,
        i < T * H * N,
    )
    if D > 0:
        ROPE_PROGRAMS: tl.constexpr = (D + N - 1) // N
        ROPE_WIDE: tl.constexpr = (
            ROPE_PROGRAMS > 2147483647
            or N + ROPE_PROGRAMS * N - 1 > 2147483647
        )
        if ROPE_WIDE:
            rope_start = tl.full((), 0, tl.int64)
            rope_stop = tl.full((), ROPE_PROGRAMS, tl.int64)
            rope_step = tl.full((), 1, tl.int64)
        else:
            rope_start = 0
            rope_stop = ROPE_PROGRAMS
            rope_step = 1
        for part in range(rope_start, rope_stop, rope_step):
            r = part * N + n
            rv = tl.load(
                R
                + _address_index(t, WIDE) * R0
                + _address_index(h, WIDE) * R1
                + _address_index(r, WIDE) * R2,
                (t < T) & (r < D),
                other=0,
            ).to(tl.float32)
            tl.store(
                output_ptr
                + _address_index(t, WIDE) * O0
                + _address_index(h, WIDE) * O1
                + _address_index(N + _address_index(r, WIDE), WIDE) * O2,
                _encode1_group8(rv),
                (t < T) & (r < D),
            )


@triton.jit
def _pack_q_rope_tile(X, R, P, output_ptr, META: tl.constexpr, PID):
    T: tl.constexpr = META[0]
    H: tl.constexpr = META[1]
    K: tl.constexpr = META[2]
    N: tl.constexpr = META[3]
    D: tl.constexpr = META[4]
    X0: tl.constexpr = META[5]
    X1: tl.constexpr = META[6]
    X2: tl.constexpr = META[7]
    R0: tl.constexpr = META[8]
    R1: tl.constexpr = META[9]
    R2: tl.constexpr = META[10]
    O0: tl.constexpr = META[11]
    O1: tl.constexpr = META[12]
    O2: tl.constexpr = META[13]
    B: tl.constexpr = META[14]
    WIDE: tl.constexpr = (
        (
            (T - 1) * X0 + (H - 1) * X1 + (K - 1) * X2 > 2147483647
            or T * H * K > 2147483647
            or (T - 1) * R0 + (H - 1) * R1 + (D - 1) * R2 > 2147483647
            or ((T - 1) * O0 + (H - 1) * O1 + (N + D - 1) * O2 > 2147483647)
            or (T * H * D > 2147483647)
        )
        or X0 > 2147483647
        or X1 > 2147483647
        or (X2 > 2147483647)
        or (R0 > 2147483647)
        or (R1 > 2147483647)
        or (R2 > 2147483647)
        or (O0 > 2147483647)
        or (O1 > 2147483647)
        or (O2 > 2147483647)
    )
    qtiles: tl.constexpr = (T * H * K + B - 1) // B
    pid = PID
    if pid < qtiles:
        i = _address_index(pid, WIDE) * B + tl.arange(0, B)
        k = i % K
        qh = i // K % H
        qt = i // (K * H)
        qvalue = tl.load(
            X
            + _address_index(qt, WIDE) * X0
            + _address_index(qh, WIDE) * X1
            + _address_index(k, WIDE) * X2,
            i < T * H * K,
            other=0,
        )
        tl.store(P + i, qvalue, i < T * H * K)
    elif D > 0:
        j = _address_index(pid - qtiles, WIDE) * 512 + tl.arange(0, 512)
        d = j % D
        rh = j // D % H
        rt = j // (D * H)
        rvalue = tl.load(
            R
            + _address_index(rt, WIDE) * R0
            + _address_index(rh, WIDE) * R1
            + _address_index(d, WIDE) * R2,
            j < T * H * D,
            other=0,
        ).to(tl.float32)
        tl.store(
            output_ptr
            + _address_index(rt, WIDE) * O0
            + _address_index(rh, WIDE) * O1
            + _address_index(N + _address_index(d, WIDE), WIDE) * O2,
            _encode1_group8(rvalue),
            j < T * H * D,
        )


def _launch_prepared(q, w, r, raw, t, h, k, n, d, matrix_ready=False):
    output_address = raw.data_ptr()
    large = t * h * n >= 16777216
    bm, bn, bk = (
        (128, 128, 64)
        if t % 128 == 0 and large and q.is_contiguous()
        else (
            (16, 64, 32)
            if t < 32
            else (
                (
                    64
                    if t % 64 == 0
                    and t * h * n >= 4194304
                    and q.is_contiguous()
                    else 32
                ),
                128,
                64,
            )
        )
    )
    bk = min(bk, k & -k) if k else bk
    tm, tn = ((t + bm - 1) // bm, (n + bn - 1) // bn)
    qs, ws, rs, os = (q.stride(), w.stride(), r.stride(), raw.stride())
    kernel = _fused1_group8_address if large else _fused0_group8_address
    if not matrix_ready and (
        not _matrix_loadable(w, n, ws) or not _matrix_loadable(q, k, qs)
    ):
        (
            _fused1_group8_wmask_address
            if large
            else _fused0_group8_wmask_address
        ).run(
            q,
            w,
            r,
            output_address,
            (t, h, k, n, d, *qs, *ws, *rs, *os, tm, tn, bm, bn, bk, 64, False),
            n,
            t,
            output_address % 16 == 0,
            num_warps=8 if bm == 128 else 4,
            num_stages=1,
            grid=(min(h * tm * tn, 32768),),
            warmup=False,
        )
    else:
        kernel.run(
            q,
            w,
            r,
            output_address,
            (t, h, k, n, d, *qs, *ws, *rs, *os, tm, tn, bm, bn, bk, 64, False),
            output_address % 16 == 0,
            num_warps=8 if bm == 128 else 4,
            num_stages=1,
            grid=(min(h * tm * tn, 32768),),
            warmup=False,
        )
    if large and d > 0:
        _copyrope1_group8_address.run(
            r,
            output_address,
            (t, h, n, d, *rs, *os, 512),
            output_address % 16 == 0,
            num_warps=4,
            grid=(
                (
                    32768
                    if (t * h * d + 511) // 512 > 16777215
                    else (t * h * d + 511) // 512
                ),
            ),
            warmup=False,
        )


def _compute_entry(
    q_fp8_pad, q_nope, w_kc, q_rope, num_heads, matrix_ready=False
):
    t, _, k = q_nope.shape
    h = num_heads
    n, d = (w_kc.shape[-1], q_rope.shape[-1])
    if t == 0 or h == 0:
        return q_fp8_pad
    raw = q_fp8_pad
    output_address = raw.data_ptr()
    if n == 0:
        if d > 0:
            _copyrope1_group8_address.run(
                q_rope,
                output_address,
                (t, h, 0, d, *q_rope.stride(), *raw.stride(), 512),
                output_address % 16 == 0,
                num_warps=4,
                grid=(
                    (
                        32768
                        if (t * h * d + 511) // 512 > 16777215
                        else (t * h * d + 511) // 512
                    ),
                ),
                warmup=False,
            )
        return q_fp8_pad
    large = t * h * n >= 16777216
    if large:
        whole = t - t % 32
        prefix_ready = _large_matrix_prefix(whole, h, k, n)
        os = raw.stride()
        aligned_strides = os[2] == 1 and os[0] % 64 == 0 and (os[1] % 64 == 0)
        aligned_pointer = (
            raw.data_ptr() % 64 == 0 if aligned_strides else False
        )
        if not q_nope.is_contiguous() or (
            q_nope.data_ptr() % 256 != 0
            and prefix_ready
            and _q_alignment_preparation_compatible(t, n, os, output_address)
        ):
            packed = q_nope.new_empty((t, h, k), dtype=q_nope.dtype)
            if prefix_ready and (
                not w_kc.is_contiguous() or w_kc.data_ptr() % 256 != 0
            ):
                weights = w_kc.new_empty(w_kc.shape)
                copy_d = d if aligned_strides and aligned_pointer else 0
                w_wide = (
                    max(_layout_span(w_kc), w_kc.numel(), *w_kc.stride())
                    > 2147482623
                )
                _pack_q_weight_rope_address.run(
                    q_nope,
                    w_kc,
                    q_rope,
                    packed,
                    weights,
                    output_address,
                    (
                        t,
                        h,
                        k,
                        n,
                        copy_d,
                        *q_nope.stride(),
                        *q_rope.stride(),
                        *os,
                        1024,
                    ),
                    (*w_kc.stride(), w_wide),
                    output_address % 16 == 0,
                    num_warps=4,
                    grid=(
                        (
                            32768
                            if (t * h * k + 1023) // 1024
                            + (t * h * copy_d + 511) // 512
                            + (h * k * n + 1023) // 1024
                            > 16777215
                            else (t * h * k + 1023) // 1024
                            + (t * h * copy_d + 511) // 512
                            + (h * k * n + 1023) // 1024
                        ),
                    ),
                    warmup=False,
                )
                w_kc = weights
                if copy_d > 0:
                    d = 0
            elif aligned_strides and aligned_pointer:
                _pack_q_rope_address.run(
                    q_nope,
                    q_rope,
                    packed,
                    output_address,
                    (
                        t,
                        h,
                        k,
                        n,
                        d,
                        *q_nope.stride(),
                        *q_rope.stride(),
                        *os,
                        1024,
                    ),
                    output_address % 16 == 0,
                    num_warps=4,
                    grid=(
                        (
                            32768
                            if (t * h * k + 1023) // 1024
                            + (t * h * d + 511) // 512
                            > 16777215
                            else (t * h * k + 1023) // 1024
                            + (t * h * d + 511) // 512
                        ),
                    ),
                    warmup=False,
                )
                d = 0
            else:
                _pack_strided_q.run(
                    q_nope,
                    packed,
                    (t, h, k, *q_nope.stride(), 1024),
                    num_warps=4,
                    grid=(
                        (
                            32768
                            if (t * h * k + 1023) // 1024 > 16777215
                            else (t * h * k + 1023) // 1024
                        ),
                    ),
                    warmup=False,
                )
            q_nope = packed
        if (
            t % 128 == 0
            and n % 128 == 0
            and q_nope.is_contiguous()
            and aligned_strides
            and (not aligned_pointer)
        ):
            c = q_nope.new_empty((t, h, n), dtype=torch.uint8)
            tm, tn, bk = ((t + 127) // 128, (n + 127) // 128, min(64, k & -k))
            rs = q_rope.stride()
            if not _matrix_loadable(w_kc, n) or not _matrix_loadable(
                q_nope, k
            ):
                _fused1_group8_wmask.run(
                    q_nope,
                    w_kc,
                    q_rope,
                    c,
                    (
                        t,
                        h,
                        k,
                        n,
                        0,
                        *q_nope.stride(),
                        *w_kc.stride(),
                        *rs,
                        h * n,
                        n,
                        1,
                        tm,
                        tn,
                        128,
                        128,
                        bk,
                        64,
                        False,
                    ),
                    n,
                    t,
                    num_warps=8,
                    num_stages=1,
                    grid=(min(h * tm * tn, 32768),),
                    warmup=False,
                )
            else:
                _fused1_group8.run(
                    q_nope,
                    w_kc,
                    q_rope,
                    c,
                    (
                        t,
                        h,
                        k,
                        n,
                        0,
                        *q_nope.stride(),
                        *w_kc.stride(),
                        *rs,
                        h * n,
                        n,
                        1,
                        tm,
                        tn,
                        128,
                        128,
                        bk,
                        64,
                        False,
                    ),
                    num_warps=8,
                    num_stages=1,
                    grid=(min(h * tm * tn, 32768),),
                    warmup=False,
                )
            _move_byte_buffer_address.run(
                c,
                q_rope,
                output_address,
                (t, h, n, d, *rs, *os, 1024),
                output_address % 16 == 0,
                num_warps=4,
                grid=(
                    (
                        32768
                        if (t * h * n + 1023) // 1024 > 16777215
                        else (t * h * n + 1023) // 1024
                    ),
                ),
                warmup=False,
            )
            return q_fp8_pad
    if t >= 64 and t % 32 != 0 and large:
        _launch_prepared(
            q_nope[:whole],
            w_kc,
            q_rope[:whole],
            raw[:whole],
            whole,
            h,
            k,
            n,
            d,
            matrix_ready,
        )
        _launch_prepared(
            q_nope[whole:],
            w_kc,
            q_rope[whole:],
            raw[whole:],
            t - whole,
            h,
            k,
            n,
            d,
            matrix_ready,
        )
    else:
        _launch_prepared(
            q_nope, w_kc, q_rope, raw, t, h, k, n, d, matrix_ready
        )
    return q_fp8_pad


@triton.jit
def _copy_bounded_layout(X, output_ptr, META: tl.constexpr):
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
    i = tl.program_id(0) * B + tl.arange(0, B)
    c = i % C
    h = i // C % H
    t = i // (H * C)
    v = tl.load(X + t * X0 + h * X1 + c * X2, i < T * H * C, other=0)
    tl.store(output_ptr + t * O0 + h * O1 + c * O2, v, i < T * H * C)


def _layout_span(tensor):
    return sum(
        (
            (size - 1) * stride
            for size, stride in zip(tensor.shape, tensor.stride())
        )
    )


def _copy_partition(source, destination):
    rounded_elements = (destination.numel() + 511) // 512 * 512
    if (
        max(_layout_span(source), _layout_span(destination)) <= 2147483647
        and rounded_elements <= 2147483648
    ):
        t, h, c = destination.shape
        xs = tuple(
            (
                stride if size > 1 else 0
                for size, stride in zip(source.shape, source.stride())
            )
        )
        os = tuple(
            (
                stride if size > 1 else 0
                for size, stride in zip(
                    destination.shape, destination.stride()
                )
            )
        )
        _copy_bounded_layout.run(
            source,
            destination,
            (t, h, c, *xs, *os, 512),
            num_warps=4,
            grid=((t * h * c + 511) // 512,),
            warmup=False,
        )
        return
    dimensions = [
        (size - 1) * max(xs, os)
        for size, xs, os in zip(
            source.shape, source.stride(), destination.stride()
        )
    ]
    axis = max(range(3), key=dimensions.__getitem__)
    count = source.shape[axis]
    assert count > 1
    middle = count // 2
    left = [slice(None)] * 3
    right = [slice(None)] * 3
    left[axis] = slice(0, middle)
    right[axis] = slice(middle, count)
    _copy_partition(source[tuple(left)], destination[tuple(left)])
    _copy_partition(source[tuple(right)], destination[tuple(right)])


def absorbed_bmm_concat_cast_q_fp8(q_fp8_pad, q_nope, w_kc, q_rope, num_heads):
    t = q_nope.shape[0]
    if t == 0 or num_heads == 0:
        return q_fp8_pad
    n = w_kc.shape[-1]
    k = q_nope.shape[-1]
    matrix_ready = False
    work = t * num_heads * n
    if t % 128 == 0 and work >= 4194304 and (k & -k >= 32):
        q_bad = not q_nope.is_contiguous() or q_nope.data_ptr() % 256 != 0
        w_bad = not w_kc.is_contiguous() or w_kc.data_ptr() % 256 != 0
        if not q_bad or t >= 256:
            matrix_ready = (
                (q_bad or w_bad)
                and work < 16777216
                and (work * k <= 536870912)
                and _matrix_loadable(q_nope, k)
                and _matrix_loadable(w_kc, n)
            )
            if not matrix_ready:
                q_pack = q_bad and work < 16777216
                if q_pack and w_bad:
                    q_new = q_nope.new_empty(q_nope.shape)
                    w_new = w_kc.new_empty(w_kc.shape)
                    wide = (
                        max(
                            _layout_span(q_nope),
                            _layout_span(w_kc),
                            q_nope.numel(),
                            w_kc.numel(),
                            *q_nope.stride(),
                            *w_kc.stride(),
                        )
                        > 2147482623
                    )
                    _pack_query_weight.run(
                        q_nope.view(torch.uint16),
                        w_kc.view(torch.uint16),
                        q_new.view(torch.uint16),
                        w_new.view(torch.uint16),
                        (
                            t,
                            num_heads,
                            k,
                            n,
                            *q_nope.stride(),
                            *w_kc.stride(),
                            1024,
                            wide,
                        ),
                        num_warps=4,
                        grid=(
                            (
                                32768
                                if (q_nope.numel() + 1023) // 1024
                                + (w_kc.numel() + 1023) // 1024
                                > 16777215
                                else (q_nope.numel() + 1023) // 1024
                                + (w_kc.numel() + 1023) // 1024
                            ),
                        ),
                        warmup=False,
                    )
                    q_nope, w_kc = (q_new, w_new)
                else:
                    if q_pack:
                        packed = q_nope.new_empty(q_nope.shape)
                        _copy_partition(
                            q_nope.view(torch.uint16),
                            packed.view(torch.uint16),
                        )
                        q_nope = packed
                    if w_bad and (
                        not (
                            work >= 16777216
                            and (
                                not q_nope.is_contiguous()
                                or (
                                    q_bad
                                    and _active_output_span(
                                        q_fp8_pad, t, num_heads
                                    )
                                    <= 2147483647
                                    and _q_alignment_preparation_compatible(
                                        t,
                                        n,
                                        q_fp8_pad.stride(),
                                        q_fp8_pad.data_ptr(),
                                    )
                                )
                            )
                        )
                    ):
                        packed = w_kc.new_empty(w_kc.shape)
                        _copy_partition(
                            w_kc.view(torch.uint16), packed.view(torch.uint16)
                        )
                        w_kc = packed
    width = w_kc.shape[-1] + q_rope.shape[-1]
    if width > 0 and (
        _active_output_span(q_fp8_pad, t, num_heads) > 2147483647
        or _output_overlaps_inputs(
            q_fp8_pad, q_nope, w_kc, q_rope, t, num_heads
        )
    ):
        destination = q_fp8_pad[:t, :num_heads, :width]
        temporary = torch.empty(
            (t, num_heads, width),
            dtype=q_fp8_pad.dtype,
            device=q_fp8_pad.device,
        )
        _compute_entry(
            temporary, q_nope, w_kc, q_rope, num_heads, matrix_ready
        )
        _copy_partition(
            temporary.view(torch.uint8), destination.view(torch.uint8)
        )
        return q_fp8_pad
    return _compute_entry(
        q_fp8_pad, q_nope, w_kc, q_rope, num_heads, matrix_ready
    )


@triton.jit(do_not_specialize=["REAL_N", "REAL_T"])
def _fused0_group8_wmask(
    X, W, R, output_ptr, META: tl.constexpr, REAL_N, REAL_T
):
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
        PROGRAMS if PROGRAMS < 32768 else 32768
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
        local = pid % (TM * TN)
        head = pid // (TM * TN)
        gm = 8
        group = local // (gm * TN)
        size = tl.minimum(TM - group * gm, gm)
        mi = group * gm + local % size
        ni = local % (gm * TN) // size
        pid = head * (TM * TN) + mi * TN + ni
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
                (m < REAL_T)[:, None] & (kk < K)[None, :],
                other=0,
            )
            w = tl.load(
                W
                + _address_index(h, WIDE) * W0
                + _address_index(kk[:, None], WIDE) * W1
                + _address_index(n[None, :], WIDE) * W2,
                (kk < K)[:, None] & (n < REAL_N)[None, :],
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
        code = _encode0_group8(rounded)
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
                    value = tl.load(
                        R
                        + _address_index(m[:, None], WIDE) * R0
                        + _address_index(h, WIDE) * R1
                        + _address_index(r[None, :], WIDE) * R2,
                        (m < T)[:, None] & (r < D)[None, :],
                        other=0,
                    ).to(tl.float32)
                    tl.store(
                        output_ptr
                        + _address_index(m[:, None], WIDE) * O0
                        + _address_index(h, WIDE) * O1
                        + _address_index(
                            N + _address_index(r[None, :], WIDE), WIDE
                        )
                        * O2,
                        _encode0_group8(value),
                        (m < T)[:, None] & (r < D)[None, :],
                    )


@triton.jit(do_not_specialize=["REAL_N", "REAL_T"])
def _fused1_group8_wmask(
    X, W, R, output_ptr, META: tl.constexpr, REAL_N, REAL_T
):
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
        PROGRAMS if PROGRAMS < 32768 else 32768
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
        local = pid % (TM * TN)
        head = pid // (TM * TN)
        gm = 8
        group = local // (gm * TN)
        size = tl.minimum(TM - group * gm, gm)
        mi = group * gm + local % size
        ni = local % (gm * TN) // size
        pid = head * (TM * TN) + mi * TN + ni
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
                (m < REAL_T)[:, None] & (kk < K)[None, :],
                other=0,
            )
            w = tl.load(
                W
                + _address_index(h, WIDE) * W0
                + _address_index(kk[:, None], WIDE) * W1
                + _address_index(n[None, :], WIDE) * W2,
                (kk < K)[:, None] & (n < REAL_N)[None, :],
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
        code = _encode1_group8(rounded)
        tl.store(
            output_ptr
            + _address_index(m[:, None], WIDE) * O0
            + _address_index(h, WIDE) * O1
            + _address_index(n[None, :], WIDE) * O2,
            code,
            (m < T)[:, None] & (n < N)[None, :],
        )


@triton.jit
def _pack_query_weight_tile(X, W, P, V, META: tl.constexpr, PID):
    T: tl.constexpr = META[0]
    H: tl.constexpr = META[1]
    K: tl.constexpr = META[2]
    N: tl.constexpr = META[3]
    X0: tl.constexpr = META[4]
    X1: tl.constexpr = META[5]
    X2: tl.constexpr = META[6]
    W0: tl.constexpr = META[7]
    W1: tl.constexpr = META[8]
    W2: tl.constexpr = META[9]
    B: tl.constexpr = META[10]
    WIDE: tl.constexpr = META[11]
    qtiles: tl.constexpr = (T * H * K + B - 1) // B
    pid = PID
    if pid < qtiles:
        i = _address_index(pid, WIDE) * B + tl.arange(0, B)
        k = i % K
        h = i // K % H
        t = i // (K * H)
        value = tl.load(
            X
            + _address_index(t, WIDE) * X0
            + _address_index(h, WIDE) * X1
            + _address_index(k, WIDE) * X2,
            i < T * H * K,
            other=0,
        )
        tl.store(P + i, value, i < T * H * K)
    else:
        i = _address_index(pid - qtiles, WIDE) * B + tl.arange(0, B)
        n = i % N
        k = i // N % K
        h = i // (N * K)
        value = tl.load(
            W
            + _address_index(h, WIDE) * W0
            + _address_index(k, WIDE) * W1
            + _address_index(n, WIDE) * W2,
            i < H * K * N,
            other=0,
        )
        tl.store(V + i, value, i < H * K * N)


@triton.jit
def _pack_q_weight_rope_tile(
    X, W, R, P, V, output_ptr, META: tl.constexpr, WMETA: tl.constexpr, PID
):
    T: tl.constexpr = META[0]
    H: tl.constexpr = META[1]
    K: tl.constexpr = META[2]
    N: tl.constexpr = META[3]
    D: tl.constexpr = META[4]
    B: tl.constexpr = META[14]
    W0: tl.constexpr = WMETA[0]
    W1: tl.constexpr = WMETA[1]
    W2: tl.constexpr = WMETA[2]
    WIDE: tl.constexpr = WMETA[3]
    preceding: tl.constexpr = (T * H * K + B - 1) // B + (
        T * H * D + 511
    ) // 512
    pid = PID
    if pid < preceding:
        _pack_q_rope_tile(X, R, P, output_ptr, META, PID)
    else:
        i = _address_index(pid - preceding, WIDE) * B + tl.arange(0, B)
        n = i % N
        k = i // N % K
        h = i // (N * K)
        value = tl.load(
            W
            + _address_index(h, WIDE) * W0
            + _address_index(k, WIDE) * W1
            + _address_index(n, WIDE) * W2,
            i < H * K * N,
            other=0,
        )
        tl.store(V + i, value, i < H * K * N)


def _active_output_span(tensor, tokens, heads):
    shape, strides = (tensor.shape, tensor.stride())
    return (
        (min(tokens, shape[0]) - 1) * strides[0]
        + (min(heads, shape[1]) - 1) * strides[1]
        + (shape[2] - 1) * strides[2]
    )


@triton.jit(do_not_specialize=["output_ptr"])
def _copyrope1_group8_address(
    R, output_ptr, META: tl.constexpr, ALIGNED: tl.constexpr
):
    pointer = output_ptr.to(tl.uint64).to(tl.pointer_type(tl.uint8))
    if ALIGNED:
        pointer = tl.multiple_of(pointer, 16)
    _copyrope1_group8(R, pointer, META)


@triton.jit(do_not_specialize=["output_ptr"])
def _fused0_group8_address(
    X, W, R, output_ptr, META: tl.constexpr, ALIGNED: tl.constexpr
):
    pointer = output_ptr.to(tl.uint64).to(tl.pointer_type(tl.uint8))
    if ALIGNED:
        pointer = tl.multiple_of(pointer, 16)
    _fused0_group8(X, W, R, pointer, META)


@triton.jit(do_not_specialize=["output_ptr"])
def _fused0_group8_wmask_address(
    X,
    W,
    R,
    output_ptr,
    META: tl.constexpr,
    REAL_N,
    REAL_T,
    ALIGNED: tl.constexpr,
):
    pointer = output_ptr.to(tl.uint64).to(tl.pointer_type(tl.uint8))
    if ALIGNED:
        pointer = tl.multiple_of(pointer, 16)
    _fused0_group8_wmask(X, W, R, pointer, META, REAL_N, REAL_T)


@triton.jit(do_not_specialize=["output_ptr"])
def _fused1_group8_address(
    X, W, R, output_ptr, META: tl.constexpr, ALIGNED: tl.constexpr
):
    pointer = output_ptr.to(tl.uint64).to(tl.pointer_type(tl.uint8))
    if ALIGNED:
        pointer = tl.multiple_of(pointer, 16)
    _fused1_group8(X, W, R, pointer, META)


@triton.jit(do_not_specialize=["output_ptr"])
def _fused1_group8_wmask_address(
    X,
    W,
    R,
    output_ptr,
    META: tl.constexpr,
    REAL_N,
    REAL_T,
    ALIGNED: tl.constexpr,
):
    pointer = output_ptr.to(tl.uint64).to(tl.pointer_type(tl.uint8))
    if ALIGNED:
        pointer = tl.multiple_of(pointer, 16)
    _fused1_group8_wmask(X, W, R, pointer, META, REAL_N, REAL_T)


@triton.jit(do_not_specialize=["output_ptr"])
def _move_byte_buffer_address(
    C, R, output_ptr, META: tl.constexpr, ALIGNED: tl.constexpr
):
    pointer = output_ptr.to(tl.uint64).to(tl.pointer_type(tl.uint8))
    if ALIGNED:
        pointer = tl.multiple_of(pointer, 16)
    _move_byte_buffer(C, R, pointer, META)


@triton.jit(do_not_specialize=["output_ptr"])
def _pack_q_rope_address(
    X, R, P, output_ptr, META: tl.constexpr, ALIGNED: tl.constexpr
):
    pointer = output_ptr.to(tl.uint64).to(tl.pointer_type(tl.uint8))
    if ALIGNED:
        pointer = tl.multiple_of(pointer, 16)
    _pack_q_rope(X, R, P, pointer, META)


@triton.jit(do_not_specialize=["output_ptr"])
def _pack_q_weight_rope_address(
    X,
    W,
    R,
    P,
    V,
    output_ptr,
    META: tl.constexpr,
    WMETA: tl.constexpr,
    ALIGNED: tl.constexpr,
):
    pointer = output_ptr.to(tl.uint64).to(tl.pointer_type(tl.uint8))
    if ALIGNED:
        pointer = tl.multiple_of(pointer, 16)
    _pack_q_weight_rope(X, W, R, P, V, pointer, META, WMETA)


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


@triton.jit
def _copyrope1_group8(R, output_ptr, META: tl.constexpr):
    PROGRAMS: tl.constexpr = (
        META[0] * META[1] * META[3] + META[10] - 1
    ) // META[10]
    if PROGRAMS > 16777215:
        stop = tl.full((), PROGRAMS, tl.int64)
        for logical_pid in range(
            tl.program_id(0).to(tl.int64),
            stop,
            tl.num_programs(0).to(tl.int64),
        ):
            _copyrope1_group8_tile(R, output_ptr, META, logical_pid)
    else:
        _copyrope1_group8_tile(R, output_ptr, META, tl.program_id(0))


@triton.jit
def _pack_strided_q(X, P, META: tl.constexpr):
    PROGRAMS: tl.constexpr = (
        META[0] * META[1] * META[2] + META[6] - 1
    ) // META[6]
    if PROGRAMS > 16777215:
        stop = tl.full((), PROGRAMS, tl.int64)
        for logical_pid in range(
            tl.program_id(0).to(tl.int64),
            stop,
            tl.num_programs(0).to(tl.int64),
        ):
            _pack_strided_q_tile(X, P, META, logical_pid)
    else:
        _pack_strided_q_tile(X, P, META, tl.program_id(0))


@triton.jit
def _move_byte_buffer(C, R, output_ptr, META: tl.constexpr):
    PROGRAMS: tl.constexpr = (
        META[0] * META[1] * META[2] + META[10] - 1
    ) // META[10]
    if PROGRAMS > 16777215:
        stop = tl.full((), PROGRAMS, tl.int64)
        for logical_pid in range(
            tl.program_id(0).to(tl.int64),
            stop,
            tl.num_programs(0).to(tl.int64),
        ):
            _move_byte_buffer_tile(C, R, output_ptr, META, logical_pid)
    else:
        _move_byte_buffer_tile(C, R, output_ptr, META, tl.program_id(0))


@triton.jit
def _pack_q_rope(X, R, P, output_ptr, META: tl.constexpr):
    PROGRAMS: tl.constexpr = (
        META[0] * META[1] * META[2] + META[14] - 1
    ) // META[14] + (META[0] * META[1] * META[4] + 511) // 512
    if PROGRAMS > 16777215:
        stop = tl.full((), PROGRAMS, tl.int64)
        for logical_pid in range(
            tl.program_id(0).to(tl.int64),
            stop,
            tl.num_programs(0).to(tl.int64),
        ):
            _pack_q_rope_tile(X, R, P, output_ptr, META, logical_pid)
    else:
        _pack_q_rope_tile(X, R, P, output_ptr, META, tl.program_id(0))


@triton.jit
def _pack_query_weight(X, W, P, V, META: tl.constexpr):
    PROGRAMS: tl.constexpr = (
        META[0] * META[1] * META[2] + META[10] - 1
    ) // META[10] + (META[1] * META[2] * META[3] + META[10] - 1) // META[10]
    if PROGRAMS > 16777215:
        stop = tl.full((), PROGRAMS, tl.int64)
        for logical_pid in range(
            tl.program_id(0).to(tl.int64),
            stop,
            tl.num_programs(0).to(tl.int64),
        ):
            _pack_query_weight_tile(X, W, P, V, META, logical_pid)
    else:
        _pack_query_weight_tile(X, W, P, V, META, tl.program_id(0))


@triton.jit
def _pack_q_weight_rope(
    X, W, R, P, V, output_ptr, META: tl.constexpr, WMETA: tl.constexpr
):
    PROGRAMS: tl.constexpr = (
        (META[0] * META[1] * META[2] + META[14] - 1) // META[14]
        + (META[0] * META[1] * META[4] + 511) // 512
        + (META[1] * META[2] * META[3] + META[14] - 1) // META[14]
    )
    if PROGRAMS > 16777215:
        stop = tl.full((), PROGRAMS, tl.int64)
        for logical_pid in range(
            tl.program_id(0).to(tl.int64),
            stop,
            tl.num_programs(0).to(tl.int64),
        ):
            _pack_q_weight_rope_tile(
                X, W, R, P, V, output_ptr, META, WMETA, logical_pid
            )
    else:
        _pack_q_weight_rope_tile(
            X, W, R, P, V, output_ptr, META, WMETA, tl.program_id(0)
        )


__all__ = ["absorbed_bmm_concat_cast_q_fp8"]
