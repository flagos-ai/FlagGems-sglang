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
def _pad_copy(X, P, META: tl.constexpr):
    A: tl.constexpr = META[0]
    B: tl.constexpr = META[1]
    C: tl.constexpr = META[2]
    PA: tl.constexpr = META[3]
    PB: tl.constexpr = META[4]
    PC: tl.constexpr = META[5]
    X0: tl.constexpr = META[6]
    X1: tl.constexpr = META[7]
    X2: tl.constexpr = META[8]
    BC: tl.constexpr = triton.next_power_of_2(PC)
    WIDE: tl.constexpr = (
        (A - 1) * X0 + (B - 1) * X1 + (C - 1) * X2 > 2147483647
        or max(X0, X1, X2) > 2147483647
        or PA * PB * PC > 2147483647
    )
    row = _address_index(tl.program_id(0), WIDE)
    a = row // PB
    b = row % PB
    c = tl.arange(0, BC)
    if a < A and b < B:
        value = tl.load(
            X
            + _address_index(a, WIDE) * X0
            + _address_index(b, WIDE) * X1
            + _address_index(tl.minimum(c, C - 1), WIDE) * X2
        )
        value = tl.where(c < C, value, 0)
        tl.store(P + row * PC + c, value, c < PC)
    else:
        tl.store(P + row * PC + c, tl.full((BC,), 0, tl.uint16), c < PC)


@triton.jit
def _copy_output_shared(
    P, output_ptr, META: tl.constexpr, MASKED: tl.constexpr
):
    T: tl.constexpr = META[0]
    H: tl.constexpr = META[1]
    N: tl.constexpr = META[2]
    D: tl.constexpr = META[3]
    PN: tl.constexpr = META[4]
    PD: tl.constexpr = META[5]
    O0: tl.constexpr = META[6]
    O1: tl.constexpr = META[7]
    O2: tl.constexpr = META[8]
    tl.static_assert(D == 0 and PD == 0 and (PN % 64 == 0))
    WIDE: tl.constexpr = (
        (T - 1) * O0 + (H - 1) * O1 + (N - 1) * O2 > 2147483647
        or max(O0, O1, O2) > 2147483647
        or T * H * PN > 2147483647
    )
    row = _address_index(tl.program_id(0), WIDE)
    t = row // H
    h = row % H
    c = _address_index(tl.program_id(1), WIDE) * 64 + tl.arange(0, 64)
    if MASKED:
        value = tl.load(P + row * PN + c, c < N, other=0)
    else:
        value = tl.load(P + row * PN + c)
    tl.store(
        output_ptr
        + _address_index(t, WIDE) * O0
        + _address_index(h, WIDE) * O1
        + _address_index(c, WIDE) * O2,
        value,
        c < N,
    )


@triton.jit
def _unpad_output(P, output_ptr, META: tl.constexpr):
    _copy_output_shared(P, output_ptr, META, False)


@triton.jit
def _address_index(index, WIDE: tl.constexpr):
    if WIDE:
        return index.to(tl.int64)
    else:
        return index


@triton.jit
def _finish_rows_bf(C, output_ptr, R, META: tl.constexpr):
    _finish_rows_shared(C, output_ptr, R, META, True, True)


@triton.jit
def _encode_bf(v):
    raw = v.to(tl.uint32, bitcast=True)
    bf = (
        v.to(tl.bfloat16, fp_downcast_rounding="rtne")
        .to(tl.uint16, bitcast=True)
        .to(tl.uint32)
        & 32767
    )
    mag = tl.minimum(bf, 17392) << 16
    exponent = tl.maximum(mag >> 23, 121)
    magic_bits = exponent * 8388616 + 167771192
    absolute = mag.to(tl.float32, bitcast=True)
    rounded = (absolute + magic_bits.to(tl.float32, bitcast=True)).to(
        tl.uint32, bitcast=True
    )
    return (rounded | raw >> 24 & 128).to(tl.uint8)


@triton.jit
def _encode_double(v):
    raw = v.to(tl.uint32, bitcast=True)
    mag = raw & 2147483647
    bf = mag + 32767 + (mag >> 16 & 1) >> 16
    mag = tl.minimum(bf, 17392) << 16
    exponent = tl.maximum(mag >> 23, 121)
    magic_bits = exponent * 8388616 + 167771192
    absolute = mag.to(tl.float32, bitcast=True)
    rounded = (absolute + magic_bits.to(tl.float32, bitcast=True)).to(
        tl.uint32, bitcast=True
    )
    return (rounded | raw >> 24 & 128).to(tl.uint8)


@triton.jit
def _encode(v):
    raw = v.to(tl.uint32, bitcast=True)
    mag = tl.minimum(raw & 2147483647, 1139802112)
    exponent = tl.maximum(mag >> 23, 121)
    magic_bits = exponent * 8388616 + 167771192
    absolute = mag.to(tl.float32, bitcast=True)
    rounded = (absolute + magic_bits.to(tl.float32, bitcast=True)).to(
        tl.uint32, bitcast=True
    )
    return (rounded | raw >> 24 & 128).to(tl.uint8)


@triton.jit
def _gemm_shared(
    X, W, C, META: tl.constexpr, REAL_N: tl.constexpr, MASKED: tl.constexpr
):
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
    TM: tl.constexpr = META[10]
    TN: tl.constexpr = META[11]
    BM: tl.constexpr = META[12]
    BN: tl.constexpr = META[13]
    BK: tl.constexpr = META[14]
    WIDE: tl.constexpr = (
        (
            (T - 1) * X0 + (H - 1) * X1 + (K - 1) * X2 > 2147483647
            or (H - 1) * W0 + (K - 1) * W1 + (N - 1) * W2 > 2147483647
            or T * H * N > 2147483647
        )
        or X0 > 2147483647
        or X1 > 2147483647
        or (X2 > 2147483647)
        or (W0 > 2147483647)
        or (W1 > 2147483647)
        or (W2 > 2147483647)
    )
    pid = _address_index(tl.program_id(0), WIDE)
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
            (kk < K)[:, None] & (n < (REAL_N if MASKED else N))[None, :],
            other=0,
        )
        acc = tl.dot(x, w, acc, input_precision="ieee")
    tl.store(
        C
        + _address_index(_address_index(m[:, None], WIDE) * H + h, WIDE) * N
        + n[None, :],
        acc,
        (m < T)[:, None] & (n < N)[None, :],
    )


@triton.jit
def _gemm(X, W, C, META: tl.constexpr):
    _gemm_shared(X, W, C, META, 0, False)


@triton.jit
def _rope_tile(R, output_ptr, META: tl.constexpr, ROW, COLUMN):
    T: tl.constexpr = META[10]
    H: tl.constexpr = META[0]
    N: tl.constexpr = META[1]
    D: tl.constexpr = META[2]
    R0: tl.constexpr = META[3]
    R1: tl.constexpr = META[4]
    R2: tl.constexpr = META[5]
    O0: tl.constexpr = META[6]
    O1: tl.constexpr = META[7]
    O2: tl.constexpr = META[8]
    B: tl.constexpr = META[9]
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
    row = _address_index(ROW, WIDE)
    t = row // H
    h = row % H
    n = _address_index(COLUMN, WIDE) * B + tl.arange(0, B)
    v = tl.load(
        R
        + _address_index(t, WIDE) * R0
        + _address_index(h, WIDE) * R1
        + _address_index(n, WIDE) * R2,
        n < D,
        other=0,
    ).to(tl.float32)
    tl.store(
        output_ptr
        + _address_index(t, WIDE) * O0
        + _address_index(h, WIDE) * O1
        + _address_index(N + _address_index(n, WIDE), WIDE) * O2,
        _encode(v),
        n < D,
    )


@triton.jit
def _rope(R, output_ptr, META: tl.constexpr):
    T: tl.constexpr = META[10]
    H: tl.constexpr = META[0]
    D: tl.constexpr = META[2]
    B: tl.constexpr = META[9]
    COLUMNS: tl.constexpr = (D + B - 1) // B
    PROGRAMS: tl.constexpr = T * H * COLUMNS
    if PROGRAMS > 2147483647:
        upper = tl.full((), PROGRAMS, tl.int64)
        for tile in range(
            tl.program_id(0).to(tl.int64),
            upper,
            tl.num_programs(0).to(tl.int64),
        ):
            _rope_tile(R, output_ptr, META, tile // COLUMNS, tile % COLUMNS)
    else:
        _rope_tile(R, output_ptr, META, tl.program_id(0), tl.program_id(1))


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
        kk = tl.minimum(k, K - 1)
        nn = tl.minimum(n, N - 1)
        x = tl.load(
            X
            + _address_index(t, WIDE) * X0
            + _address_index(h, WIDE) * X1
            + _address_index(kk, WIDE) * X2
        ).to(tl.float32)
        w = tl.load(
            W
            + _address_index(h, WIDE) * W0
            + _address_index(kk[None, :], WIDE) * W1
            + _address_index(nn[:, None], WIDE) * W2
        ).to(tl.float32)
        product = tl.where(k[None, :] < K, x[None, :] * w, 0.0)
        acc = tl.sum(product, axis=1)
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
                    rr = tl.minimum(r, D - 1)
                    value = tl.load(
                        R
                        + _address_index(t, WIDE) * R0
                        + _address_index(h, WIDE) * R1
                        + _address_index(rr, WIDE) * R2
                    ).to(tl.float32)
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
def _finish_rows_shared(
    C, output_ptr, R, META: tl.constexpr, BF: tl.constexpr, ROPE: tl.constexpr
):
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
    BC: tl.constexpr = META[11]
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
    row = _address_index(tl.program_id(0), WIDE) * BT + tl.arange(0, BT)
    t = row // H
    h = row % H
    n = _address_index(tl.program_id(1), WIDE) * BC + tl.arange(0, BC)
    v = tl.load(C + _address_index(row[:, None], WIDE) * N + n[None, :])
    tl.store(
        output_ptr
        + _address_index(t[:, None], WIDE) * O0
        + _address_index(h[:, None], WIDE) * O1
        + _address_index(n[None, :], WIDE) * O2,
        _encode_bf(v) if BF else _encode_double(v),
    )
    if ROPE:
        if D > 0:
            br: tl.constexpr = min(128, D & -D)
            if _address_index(tl.program_id(1), WIDE) == 0:
                ROPE_PROGRAMS: tl.constexpr = D // br
                ROPE_WIDE: tl.constexpr = (
                    ROPE_PROGRAMS > 2147483647
                    or N + ROPE_PROGRAMS * br - 1 > 2147483647
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
                    r = ri * br + tl.arange(0, br)
                    rv = (
                        (
                            tl.load(
                                R
                                + _address_index(t[:, None], WIDE) * R0
                                + _address_index(h[:, None], WIDE) * R1
                                + _address_index(r[None, :], WIDE) * R2
                            ).to(tl.uint32)
                            << 16
                        ).to(tl.float32, bitcast=True)
                        if BF
                        else tl.load(
                            R
                            + _address_index(t[:, None], WIDE) * R0
                            + _address_index(h[:, None], WIDE) * R1
                            + _address_index(r[None, :], WIDE) * R2
                        ).to(tl.float32)
                    )
                    tl.store(
                        output_ptr
                        + _address_index(t[:, None], WIDE) * O0
                        + _address_index(h[:, None], WIDE) * O1
                        + _address_index(
                            N + _address_index(r[None, :], WIDE), WIDE
                        )
                        * O2,
                        _encode(rv),
                    )


@triton.jit
def _finish_rows(C, output_ptr, R, META: tl.constexpr):
    _finish_rows_shared(C, output_ptr, R, META, False, True)


@triton.jit
def _finish_no_rope(C, output_ptr, R, META: tl.constexpr):
    _finish_rows_shared(C, output_ptr, R, META, False, False)


@triton.jit
def _pack_x_shared(
    X, C, META: tl.constexpr, FIRST, SECOND, PARTITION: tl.constexpr
):
    T: tl.constexpr = META[6]
    H: tl.constexpr = META[0]
    K: tl.constexpr = META[1]
    X0: tl.constexpr = META[2]
    X1: tl.constexpr = META[3]
    X2: tl.constexpr = META[4]
    BK: tl.constexpr = META[5]
    WIDE: tl.constexpr = (
        (
            (T - 1) * X0 + (H - 1) * X1 + (K - 1) * X2 > 2147483647
            or T * H * K > 2147483647
        )
        or X0 > 2147483647
        or X1 > 2147483647
        or (X2 > 2147483647)
    )
    row = _address_index(FIRST, WIDE) * 16 + tl.arange(0, 16)
    t = row // H
    h = row % H
    if PARTITION:
        k = SECOND * BK + tl.arange(0, BK)
    else:
        k = _address_index(SECOND, WIDE) * BK + tl.arange(0, BK)
    value = tl.load(
        X
        + _address_index(t[:, None], WIDE) * X0
        + _address_index(h[:, None], WIDE) * X1
        + _address_index(k[None, :], WIDE) * X2
    )
    tl.store(C + _address_index(row[:, None], WIDE) * K + k[None, :], value)


@triton.jit
def _pack_x(X, C, META: tl.constexpr):
    _pack_x_shared(X, C, META, tl.program_id(0), tl.program_id(1), False)


def _positive_n(
    q_fp8_pad,
    q_nope,
    w_kc,
    q_rope,
    num_heads,
    store_n=0,
    store_t=0,
    compute_n=0,
):
    t, _, k = q_nope.shape
    h = num_heads
    n = compute_n or w_kc.shape[-1]
    d = q_rope.shape[-1]
    if t == 0 or h == 0:
        return q_fp8_pad
    whole = t - t % 16 if k % 16 == 0 and n % 64 == 0 and (d % 64 == 0) else 0
    raw = q_fp8_pad.view(torch.uint8)
    if whole < t:
        tail_q = q_nope[whole:]
        tail_r = q_rope[whole:]
        tail_out = raw[whole:]
        remainder = t - whole
        if remainder * h * k * n >= 65536 and n <= 512:
            _padded_part(q_fp8_pad[whole:], tail_q, w_kc, tail_r, h)
        else:
            _fallback.run(
                tail_q,
                w_kc,
                tail_r,
                tail_out,
                (
                    remainder,
                    h,
                    k,
                    n,
                    d,
                    *tail_q.stride(),
                    *w_kc.stride(),
                    *tail_r.stride(),
                    *tail_out.stride(),
                    triton.next_power_of_2(k),
                    32,
                    True,
                ),
                num_warps=4,
                grid=(min(remainder * h * triton.cdiv(n, 32), 32768),),
                warmup=False,
            )
    if whole == 0:
        return q_fp8_pad
    t = whole
    x = q_nope
    if q_nope.stride(2) != 1:
        x = torch.empty((t, h, k), dtype=q_nope.dtype, device=q_nope.device)
        bk = min(32, k & -k)
        if (
            d == 64
            and (n % 128 != 0 or q_rope.stride(2) != 1)
            and (t * h * k >= 32768)
        ):
            _pack_x_rope.run(
                q_nope,
                x,
                q_rope,
                raw,
                (h, k, *q_nope.stride(), bk, t),
                (t, h, n, *q_rope.stride(), *raw.stride()),
                num_warps=4,
                groups_per_cluster=16,
                grid=(t * h // 16, k // bk),
                warmup=False,
            )
            q_rope = q_rope[:, :, :0]
            d = 0
        else:
            _pack_x.run(
                q_nope,
                x,
                (h, k, *q_nope.stride(), bk, t),
                num_warps=4,
                groups_per_cluster=16,
                grid=(t * h // 16, k // bk),
                warmup=False,
            )
    c = torch.empty((t, h, n), dtype=torch.float32, device=q_nope.device)
    bm, bn, bk = (min(128, t & -t), min(256, n & -n), min(128, k & -k))
    if compute_n and bn == n:
        bn = bn // 2
    tm, tn = (t // bm, n // bn)
    if compute_n:
        _gemm_logical_masked.run(
            x,
            w_kc,
            c,
            (t, h, k, n, *x.stride(), *w_kc.stride(), tm, tn, bm, bn, bk),
            w_kc.shape[-1],
            num_warps=4,
            num_stages=1,
            grid=(h * tm * tn,),
            warmup=False,
        )
    else:
        _gemm.run(
            x,
            w_kc,
            c,
            (t, h, k, n, *x.stride(), *w_kc.stride(), tm, tn, bm, bn, bk),
            num_warps=4,
            num_stages=1,
            grid=(h * tm * tn,),
            warmup=False,
        )
    bt = 64 if t >= 512 and t * h % 64 == 0 else 16
    bc = min(512, n & -n)
    if store_n:
        if store_t:
            whole_rows = store_t * h // bt * bt
            if whole_rows and store_n >= 64:
                _finish_trim_full.run(
                    c,
                    raw,
                    (store_t, h, n, store_n, *raw.stride(), bt),
                    num_warps=4,
                    groups_per_cluster=bt,
                    grid=(whole_rows // bt, store_n // 64),
                    warmup=False,
                )
            fringe = (whole_rows if store_n % 64 else 0) + (
                store_t * h - whole_rows
            ) * ((store_n + 63) // 64)
            if fringe:
                _finish_fringe.run(
                    c,
                    raw,
                    (store_t, h, n, store_n, *raw.stride(), whole_rows),
                    num_warps=4,
                    groups_per_cluster=16,
                    grid=(fringe,),
                    warmup=False,
                )
        elif (
            bt == 16 and store_n >= 64 and (t * h // 16 * (store_n // 64) <= 6)
        ):
            _finish_partitioned.run(
                c,
                raw,
                (t, h, n, store_n, *raw.stride(), 16),
                num_warps=4,
                groups_per_cluster=16,
                grid=(t * h // 16 * (store_n // 64) + t * h,),
                warmup=False,
            )
        else:
            if store_n >= 64:
                _finish_trim_full.run(
                    c,
                    raw,
                    (t, h, n, store_n, *raw.stride(), bt),
                    num_warps=4,
                    groups_per_cluster=bt,
                    grid=(t * h // bt, store_n // 64),
                    warmup=False,
                )
            _finish_trim.run(
                c,
                raw,
                (t, h, n, store_n, *raw.stride(), bt, 64),
                num_warps=4,
                groups_per_cluster=16,
                grid=(t * h, 1),
                warmup=False,
            )
    elif n % 128 == 0 and q_rope.stride(2) == 1 and (d <= 64):
        if t * h * n >= 4194304 and bt == 64 and (bc == 512):
            _finish_rows_bf.run(
                c,
                raw,
                q_rope.view(torch.uint16),
                (t, h, n, d, *q_rope.stride(), *raw.stride(), bt, bc),
                num_warps=4,
                groups_per_cluster=bt,
                isCloseDtypeConvert=True,
                grid=(t * h // bt, n // bc),
                warmup=False,
            )
        else:
            _finish_rows.run(
                c,
                raw,
                q_rope,
                (t, h, n, d, *q_rope.stride(), *raw.stride(), bt, bc),
                num_warps=4,
                groups_per_cluster=bt,
                grid=(t * h // bt, n // bc),
                warmup=False,
            )
    else:
        _finish_no_rope.run(
            c,
            raw,
            q_rope,
            (t, h, n, d, *q_rope.stride(), *raw.stride(), bt, bc),
            num_warps=4,
            groups_per_cluster=bt,
            grid=(t * h // bt, n // bc),
            warmup=False,
        )
        if d > 0:
            _rope.run(
                q_rope,
                raw,
                (h, n, d, *q_rope.stride(), *raw.stride(), 64, t),
                num_warps=4,
                grid=(
                    (32768,)
                    if t * h * (d // 64) > 2147483647
                    else (t * h, d // 64)
                ),
                warmup=False,
            )
    return q_fp8_pad


def _compute_entry(q_fp8_pad, q_nope, w_kc, q_rope, num_heads):
    if w_kc.shape[-1] != 0:
        return _positive_n(q_fp8_pad, q_nope, w_kc, q_rope, num_heads)
    t, h, d = (q_nope.shape[0], num_heads, q_rope.shape[-1])
    if t > 0 and h > 0 and (d > 0):
        raw = q_fp8_pad.view(torch.uint8)
        _rope.run(
            q_rope,
            raw,
            (h, 0, d, *q_rope.stride(), *raw.stride(), 64, t),
            num_warps=4,
            grid=(
                (32768,)
                if t * h * triton.cdiv(d, 64) > 2147483647
                else (t * h, triton.cdiv(d, 64))
            ),
            warmup=False,
        )
    return q_fp8_pad


def _padded_tensor(source, shape):
    target = source.new_empty(shape)
    elements = shape[0] * shape[1] * shape[2]
    if elements:
        _pad_copy.run(
            source.view(torch.uint16),
            target.view(torch.uint16),
            (*source.shape, *shape, *source.stride()),
            num_warps=4,
            groups_per_cluster=16,
            grid=(shape[0] * shape[1],),
            warmup=False,
        )
    return target


def _padded_part(destination, q, w, r, h):
    t, k = (q.shape[0], q.shape[2])
    n, d = (w.shape[2], r.shape[2])
    pt, pn = ((t + 15) // 16 * 16, (n + 63) // 64 * 64)
    if pn != n and n >= 64 and (k % 16 == 0) and (w.stride(2) == 1):
        _logical_weight_part(destination, q, w, r, h, pt, pn)
        return
    if (
        pt != t
        and t >= 16
        and (pn != n)
        and (pn <= 512)
        and (k % 16 == 0)
        and (w.stride(2) == 1)
        and (d == 64)
        and (t * h * k >= 32768)
    ):
        pq, pw = _padded_all(q, w, pn, pt, r, destination, t, h, k, n)
        r = r[:, :, :0]
        d = 0
    elif (
        pt == t
        and pn != n
        and (pn <= 512)
        and (k % 16 == 0)
        and (w.stride(2) == 1)
        and (q.stride(2) != 1)
        and (d == 64)
        and (t * h * k >= 32768)
        and (t * h * 64 >= 4096)
    ):
        pq, pw = _padded_weight_query_rope(
            q, w, pn, r, destination, t, h, k, n
        )
        r = r[:, :, :0]
        d = 0
    elif (
        pt == t
        and pn != n
        and (pn <= 512)
        and (k % 16 == 0)
        and (w.stride(2) == 1)
        and (q.stride(2) != 1)
        and (d != 64)
        and (t * h * k >= 32768)
        and (t * h * 64 >= 4096)
    ):
        pq, pw = _padded_weight_query(q, w, pn, t, h, k, n)
    else:
        if (
            pt != t
            and k % 16 == 0
            and (t >= 16)
            and (t * h * k >= 32768)
            and (d == 64)
        ):
            pq = _padded_q_rope(q, pt, h, k, r, destination, n)
            r = r[:, :, :0]
            d = 0
        else:
            pq = q if pt == t else _padded_q(q, pt, h, k)
        if (
            pt == t
            and pn != n
            and (pn <= 512)
            and (k % 16 == 0)
            and (w.stride(2) == 1)
            and (d == 64)
            and (t * h * 64 >= 4096)
        ):
            pw = _padded_weights_rope(w, pn, r, destination, t, h, k, n)
            r = r[:, :, :0]
            d = 0
        else:
            pw = _padded_weights(w, pn) if pn != n else w
    if pt != t:
        _positive_n(destination, pq, pw, r[:, :, :0], h, n, t)
        if d:
            raw = destination.view(torch.uint8)
            _rope.run(
                r,
                raw,
                (h, n, d, *r.stride(), *raw.stride(), 64, t),
                num_warps=4,
                grid=(
                    (32768,)
                    if t * h * ((d + 63) // 64) > 2147483647
                    else (t * h, (d + 63) // 64)
                ),
                warmup=False,
            )
        return
    if pt == t and pn != n:
        _positive_n(destination, pq, pw, r[:, :, :0], h, n)
        if d:
            raw = destination.view(torch.uint8)
            _rope.run(
                r,
                raw,
                (h, n, d, *r.stride(), *raw.stride(), 64, t),
                num_warps=4,
                grid=(
                    (32768,)
                    if t * h * ((d + 63) // 64) > 2147483647
                    else (t * h, (d + 63) // 64)
                ),
                warmup=False,
            )
        return
    output = (
        destination
        if pt == t and pn == n
        else destination.new_empty((pt, h, pn))
    )
    _compute_entry(output, pq, pw, r[:, :, :0], h)
    raw = destination.view(torch.uint8)
    if pt != t or pn != n:
        _unpad_output.run(
            output.view(torch.uint8),
            raw,
            (t, h, n, 0, pn, 0, *raw.stride()),
            num_warps=4,
            groups_per_cluster=16,
            grid=(t * h, (n + 63) // 64),
            warmup=False,
        )
    if d:
        _rope.run(
            r,
            raw,
            (h, n, d, *r.stride(), *raw.stride(), 64, t),
            num_warps=4,
            grid=(
                (32768,)
                if t * h * ((d + 63) // 64) > 2147483647
                else (t * h, (d + 63) // 64)
            ),
            warmup=False,
        )


@triton.jit
def _pad_weight(X, P, META: tl.constexpr):
    H: tl.constexpr = META[0]
    K: tl.constexpr = META[1]
    N: tl.constexpr = META[2]
    PN: tl.constexpr = META[3]
    X0: tl.constexpr = META[4]
    X1: tl.constexpr = META[5]
    X2: tl.constexpr = META[6]
    WIDE: tl.constexpr = (
        (H - 1) * X0 + (K - 1) * X1 + (N - 1) * X2 > 2147483647
        or max(X0, X1, X2) > 2147483647
        or H * K * PN > 2147483647
    )
    h = _address_index(tl.program_id(0), WIDE)
    k = _address_index(tl.program_id(1), WIDE) * 16 + tl.arange(0, 16)
    c = tl.arange(0, PN)
    value = tl.load(
        X
        + _address_index(h, WIDE) * X0
        + _address_index(k[:, None], WIDE) * X1
        + _address_index(tl.minimum(c[None, :], N - 1), WIDE) * X2
    )
    value = tl.where(c[None, :] < N, value, 0)
    tl.store(
        P + (_address_index(h, WIDE) * K + k[:, None]) * PN + c[None, :], value
    )


def _padded_weights(source, pn):
    h, k, n = source.shape
    if pn > 512 or k % 16 or source.stride(2) != 1:
        return _padded_tensor(source, (h, k, pn))
    target = source.new_empty((h, k, pn))
    if pn & pn - 1:
        bc = pn & -pn
        _pad_weight_chunks.run(
            source.view(torch.uint16),
            target.view(torch.uint16),
            (h, k, n, pn, *source.stride(), bc),
            num_warps=4,
            groups_per_cluster=16,
            grid=(h * (pn // bc), k // 16),
            warmup=False,
        )
    else:
        _pad_weight.run(
            source.view(torch.uint16),
            target.view(torch.uint16),
            (h, k, n, pn, *source.stride()),
            num_warps=4,
            groups_per_cluster=16,
            grid=(h, k // 16),
            warmup=False,
        )
    return target


@triton.jit
def _finish_trim(C, output_ptr, META: tl.constexpr):
    _trim_tail_store(C, output_ptr, META, tl.program_id(0))


@triton.jit
def _finish_trim_full(C, output_ptr, META: tl.constexpr):
    T: tl.constexpr = META[0]
    H: tl.constexpr = META[1]
    PN: tl.constexpr = META[2]
    N: tl.constexpr = META[3]
    O0: tl.constexpr = META[4]
    O1: tl.constexpr = META[5]
    O2: tl.constexpr = META[6]
    BT: tl.constexpr = META[7]
    WIDE: tl.constexpr = (
        T * H * PN > 2147483647
        or (T - 1) * O0 + (H - 1) * O1 + (N - 1) * O2 > 2147483647
        or O0 > 2147483647
        or (O1 > 2147483647)
        or (O2 > 2147483647)
    )
    row = _address_index(tl.program_id(0), WIDE) * BT + tl.arange(0, BT)
    t = row // H
    h = row % H
    n = _address_index(tl.program_id(1), WIDE) * 64 + tl.arange(0, 64)
    value = tl.load(C + _address_index(row[:, None], WIDE) * PN + n[None, :])
    tl.store(
        output_ptr
        + _address_index(t[:, None], WIDE) * O0
        + _address_index(h[:, None], WIDE) * O1
        + _address_index(n[None, :], WIDE) * O2,
        _encode_double(value),
    )


@triton.jit
def _finish_fringe(C, output_ptr, META: tl.constexpr):
    T: tl.constexpr = META[0]
    H: tl.constexpr = META[1]
    PN: tl.constexpr = META[2]
    N: tl.constexpr = META[3]
    O0: tl.constexpr = META[4]
    O1: tl.constexpr = META[5]
    O2: tl.constexpr = META[6]
    FULL_ROWS: tl.constexpr = META[7]
    TAIL_COL_ROWS: tl.constexpr = FULL_ROWS if N % 64 else 0
    COL_BLOCKS: tl.constexpr = (N + 63) // 64
    WIDE: tl.constexpr = (
        T * H * PN > 2147483647
        or (T - 1) * O0 + (H - 1) * O1 + (N - 1) * O2 > 2147483647
        or O0 > 2147483647
        or (O1 > 2147483647)
        or (O2 > 2147483647)
    )
    pid = _address_index(tl.program_id(0), WIDE)
    if TAIL_COL_ROWS:
        row = tl.where(
            pid < TAIL_COL_ROWS,
            pid,
            FULL_ROWS + (pid - TAIL_COL_ROWS) // COL_BLOCKS,
        )
        block = tl.where(
            pid < TAIL_COL_ROWS, N // 64, (pid - TAIL_COL_ROWS) % COL_BLOCKS
        )
    else:
        row = FULL_ROWS + pid // COL_BLOCKS
        block = pid % COL_BLOCKS
    t = row // H
    h = row % H
    n = block * 64 + tl.arange(0, 64)
    value = tl.load(C + row * PN + n)
    tl.store(
        output_ptr
        + _address_index(t, WIDE) * O0
        + _address_index(h, WIDE) * O1
        + _address_index(n, WIDE) * O2,
        _encode_double(value),
        n < N,
    )


@triton.jit
def _trim_tail_store(C, output_ptr, META: tl.constexpr, ROW):
    T: tl.constexpr = META[0]
    H: tl.constexpr = META[1]
    PN: tl.constexpr = META[2]
    N: tl.constexpr = META[3]
    O0: tl.constexpr = META[4]
    O1: tl.constexpr = META[5]
    O2: tl.constexpr = META[6]
    WIDE: tl.constexpr = (
        T * H * PN > 2147483647
        or (T - 1) * O0 + (H - 1) * O1 + (N - 1) * O2 > 2147483647
        or O0 > 2147483647
        or (O1 > 2147483647)
        or (O2 > 2147483647)
    )
    row = _address_index(ROW, WIDE)
    t = row // H
    h = row % H
    n = N // 64 * 64 + _address_index(tl.arange(0, 64), WIDE)
    value = tl.load(C + row * PN + n)
    tl.store(
        output_ptr
        + _address_index(t, WIDE) * O0
        + _address_index(h, WIDE) * O1
        + _address_index(n, WIDE) * O2,
        _encode_double(value),
        n < N,
    )


@triton.jit
def _finish_partitioned(C, output_ptr, META: tl.constexpr):
    T: tl.constexpr = META[0]
    H: tl.constexpr = META[1]
    PN: tl.constexpr = META[2]
    N: tl.constexpr = META[3]
    O0: tl.constexpr = META[4]
    O1: tl.constexpr = META[5]
    O2: tl.constexpr = META[6]
    BT: tl.constexpr = META[7]
    FULL: tl.constexpr = N // 64
    MAIN: tl.constexpr = T * H // BT * FULL
    WIDE: tl.constexpr = (
        T * H * PN > 2147483647
        or (T - 1) * O0 + (H - 1) * O1 + (N - 1) * O2 > 2147483647
        or O0 > 2147483647
        or (O1 > 2147483647)
        or (O2 > 2147483647)
    )
    pid = _address_index(tl.program_id(0), WIDE)
    if FULL > 0:
        if pid < MAIN:
            row = pid // FULL * BT + tl.arange(0, BT)
            t = row // H
            h = row % H
            n = pid % FULL * 64 + tl.arange(0, 64)
            value = tl.load(
                C + _address_index(row[:, None], WIDE) * PN + n[None, :]
            )
            tl.store(
                output_ptr
                + _address_index(t[:, None], WIDE) * O0
                + _address_index(h[:, None], WIDE) * O1
                + _address_index(n[None, :], WIDE) * O2,
                _encode_double(value),
            )
        else:
            _trim_tail_store(C, output_ptr, META, pid - MAIN)
    else:
        _trim_tail_store(C, output_ptr, META, pid)


@triton.jit
def _pad_weight_chunks_shared(
    X, P, META: tl.constexpr, FIRST, SECOND, PARTITION: tl.constexpr
):
    H: tl.constexpr = META[0]
    K: tl.constexpr = META[1]
    N: tl.constexpr = META[2]
    PN: tl.constexpr = META[3]
    X0: tl.constexpr = META[4]
    X1: tl.constexpr = META[5]
    X2: tl.constexpr = META[6]
    WIDE: tl.constexpr = (
        (H - 1) * X0 + (K - 1) * X1 + (N - 1) * X2 > 2147483647
        or max(X0, X1, X2) > 2147483647
        or H * K * PN > 2147483647
    )
    BC: tl.constexpr = META[7]
    NC: tl.constexpr = PN // BC
    if PARTITION:
        h = FIRST // NC
    else:
        h = _address_index(FIRST, WIDE) // NC
    if PARTITION:
        k = SECOND * 16 + tl.arange(0, 16)
    else:
        k = _address_index(SECOND, WIDE) * 16 + tl.arange(0, 16)
    if PARTITION:
        c = FIRST % NC * BC + tl.arange(0, BC)
    else:
        c = _address_index(FIRST, WIDE) % NC * BC + tl.arange(0, BC)
    value = tl.load(
        X
        + _address_index(h, WIDE) * X0
        + _address_index(k[:, None], WIDE) * X1
        + _address_index(tl.minimum(c[None, :], N - 1), WIDE) * X2
    )
    value = tl.where(c[None, :] < N, value, 0)
    tl.store(
        P + (_address_index(h, WIDE) * K + k[:, None]) * PN + c[None, :], value
    )


@triton.jit
def _pad_weight_chunks(X, P, META: tl.constexpr):
    _pad_weight_chunks_shared(
        X, P, META, tl.program_id(0), tl.program_id(1), False
    )


@triton.jit
def _pad_q_grouped_shared(
    X, P, META: tl.constexpr, PID, PARTITION: tl.constexpr
):
    T: tl.constexpr = META[0]
    H: tl.constexpr = META[1]
    K: tl.constexpr = META[2]
    PT: tl.constexpr = META[3]
    X0: tl.constexpr = META[4]
    X1: tl.constexpr = META[5]
    X2: tl.constexpr = META[6]
    BC: tl.constexpr = META[7]
    NT: tl.constexpr = T // 16
    NK: tl.constexpr = K // BC
    MAIN: tl.constexpr = NT * H * NK
    WIDE: tl.constexpr = (
        (T - 1) * X0 + (H - 1) * X1 + (K - 1) * X2 > 2147483647
        or X0 > 2147483647
        or X1 > 2147483647
        or (X2 > 2147483647)
        or (PT * H * K > 2147483647)
    )
    pid = _address_index(PID, WIDE)
    if pid < MAIN:
        h = pid // (NT * NK)
        t = pid // NK % NT * 16 + tl.arange(0, 16)
        k = pid % NK * BC + tl.arange(0, BC)
        value = tl.load(
            X
            + _address_index(t[:, None], WIDE) * X0
            + _address_index(h, WIDE) * X1
            + _address_index(k[None, :], WIDE) * X2
        )
        tl.store(
            P + (_address_index(t[:, None], WIDE) * H + h) * K + k[None, :],
            value,
        )
    else:
        _pad_q_tail(X, P, META, pid)


@triton.jit
def _pad_q_grouped(X, P, META: tl.constexpr):
    _pad_q_grouped_shared(X, P, META, tl.program_id(0), False)


def _padded_q(source, pt, h, k):
    t = source.shape[0]
    if k % 16 or t < 16 or t * h * k < 32768:
        return _padded_tensor(source, (pt, h, k))
    target = source.new_empty((pt, h, k))
    bc = min(128, k & -k)
    nt = t // 16
    nk = k // bc
    _pad_q_grouped.run(
        source.view(torch.uint16),
        target.view(torch.uint16),
        (t, h, k, pt, *source.stride(), bc),
        num_warps=4,
        groups_per_cluster=16,
        grid=(nt * h * nk + (pt - nt * 16) * h * nk,),
        warmup=False,
    )
    return target


@triton.jit
def _pad_q_tail(X, P, META: tl.constexpr, PID):
    T: tl.constexpr = META[0]
    H: tl.constexpr = META[1]
    K: tl.constexpr = META[2]
    PT: tl.constexpr = META[3]
    X0: tl.constexpr = META[4]
    X1: tl.constexpr = META[5]
    X2: tl.constexpr = META[6]
    BC: tl.constexpr = META[7]
    NT: tl.constexpr = T // 16
    NK: tl.constexpr = K // BC
    MAIN: tl.constexpr = NT * H * NK
    WIDE: tl.constexpr = (
        (T - 1) * X0 + (H - 1) * X1 + (K - 1) * X2 > 2147483647
        or X0 > 2147483647
        or X1 > 2147483647
        or (X2 > 2147483647)
        or (PT * H * K > 2147483647)
    )
    pid = _address_index(PID, WIDE)
    tail = pid - MAIN
    t = NT * 16 + tail // (H * NK)
    h = tail // NK % H
    k = tail % NK * BC + tl.arange(0, BC)
    if t < T:
        value = tl.load(
            X
            + _address_index(t, WIDE) * X0
            + _address_index(h, WIDE) * X1
            + _address_index(k, WIDE) * X2
        )
        tl.store(P + (_address_index(t, WIDE) * H + h) * K + k, value)
    else:
        tl.store(
            P + (_address_index(t, WIDE) * H + h) * K + k,
            tl.full((BC,), 0, tl.uint16),
        )


@triton.jit
def _prepare_rope_full(R, output_ptr, META: tl.constexpr, TOKEN, HEAD):
    T: tl.constexpr = META[0]
    H: tl.constexpr = META[1]
    N: tl.constexpr = META[2]
    R0: tl.constexpr = META[3]
    R1: tl.constexpr = META[4]
    R2: tl.constexpr = META[5]
    O0: tl.constexpr = META[6]
    O1: tl.constexpr = META[7]
    O2: tl.constexpr = META[8]
    WIDE: tl.constexpr = (
        (T - 1) * R0 + (H - 1) * R1 + 63 * R2 > 2147483647
        or (T - 1) * O0 + (H - 1) * O1 + (N + 63) * O2 > 2147483647
        or R0 > 2147483647
        or (R1 > 2147483647)
        or (R2 > 2147483647)
        or (O0 > 2147483647)
        or (O1 > 2147483647)
        or (O2 > 2147483647)
    )
    t = TOKEN + tl.arange(0, 16)
    r = tl.arange(0, 64)
    value = tl.load(
        R
        + _address_index(t[:, None], WIDE) * R0
        + _address_index(HEAD, WIDE) * R1
        + _address_index(r[None, :], WIDE) * R2
    ).to(tl.float32)
    tl.store(
        output_ptr
        + _address_index(t[:, None], WIDE) * O0
        + _address_index(HEAD, WIDE) * O1
        + _address_index(N + _address_index(r[None, :], WIDE), WIDE) * O2,
        _encode(value),
    )


@triton.jit
def _prepare_rope_tail(R, output_ptr, META: tl.constexpr, TOKEN, HEAD):
    T: tl.constexpr = META[0]
    H: tl.constexpr = META[1]
    N: tl.constexpr = META[2]
    R0: tl.constexpr = META[3]
    R1: tl.constexpr = META[4]
    R2: tl.constexpr = META[5]
    O0: tl.constexpr = META[6]
    O1: tl.constexpr = META[7]
    O2: tl.constexpr = META[8]
    WIDE: tl.constexpr = (
        (T - 1) * R0 + (H - 1) * R1 + 63 * R2 > 2147483647
        or (T - 1) * O0 + (H - 1) * O1 + (N + 63) * O2 > 2147483647
        or R0 > 2147483647
        or (R1 > 2147483647)
        or (R2 > 2147483647)
        or (O0 > 2147483647)
        or (O1 > 2147483647)
        or (O2 > 2147483647)
    )
    r = tl.arange(0, 64)
    value = tl.load(
        R
        + _address_index(TOKEN, WIDE) * R0
        + _address_index(HEAD, WIDE) * R1
        + _address_index(r, WIDE) * R2
    ).to(tl.float32)
    tl.store(
        output_ptr
        + _address_index(TOKEN, WIDE) * O0
        + _address_index(HEAD, WIDE) * O1
        + _address_index(N + _address_index(r, WIDE), WIDE) * O2,
        _encode(value),
    )


@triton.jit
def _prepare_q_rope_shared(
    X,
    P,
    R,
    output_ptr,
    Q_META: tl.constexpr,
    R_META: tl.constexpr,
    PID,
    PARTITION: tl.constexpr,
):
    if PARTITION:
        _pad_q_grouped_partition(X, P, Q_META, PID)
    else:
        _pad_q_grouped(X, P, Q_META)
    T: tl.constexpr = Q_META[0]
    H: tl.constexpr = Q_META[1]
    K: tl.constexpr = Q_META[2]
    BC: tl.constexpr = Q_META[7]
    NT: tl.constexpr = T // 16
    NK: tl.constexpr = K // BC
    MAIN: tl.constexpr = NT * H * NK
    pid = _address_index(PID, Q_META[3] * H * K > 2147483647)
    if pid < MAIN:
        if pid % NK == 0:
            _prepare_rope_full(
                R, output_ptr, R_META, pid // NK % NT * 16, pid // (NT * NK)
            )
    else:
        tail = pid - MAIN
        token = NT * 16 + tail // (H * NK)
        if token < T:
            if tail % NK == 0:
                _prepare_rope_tail(
                    R, output_ptr, R_META, token, tail // NK % H
                )


@triton.jit
def _prepare_q_rope(
    X, P, R, output_ptr, Q_META: tl.constexpr, R_META: tl.constexpr
):
    _prepare_q_rope_shared(
        X, P, R, output_ptr, Q_META, R_META, tl.program_id(0), False
    )


def _padded_q_rope(source, pt, h, k, rope, destination, n):
    t = source.shape[0]
    target = source.new_empty((pt, h, k))
    bc = min(128, k & -k)
    nt = t // 16
    nk = k // bc
    raw = destination.view(torch.uint8)
    _prepare_q_rope.run(
        source.view(torch.uint16),
        target.view(torch.uint16),
        rope,
        raw,
        (t, h, k, pt, *source.stride(), bc),
        (t, h, n, *rope.stride(), *raw.stride()),
        num_warps=4,
        groups_per_cluster=16,
        grid=(nt * h * nk + (pt - nt * 16) * h * nk,),
        warmup=False,
    )
    return target


@triton.jit
def _pack_x_rope_shared(
    X,
    P,
    R,
    output_ptr,
    Q_META: tl.constexpr,
    R_META: tl.constexpr,
    FIRST,
    SECOND,
    PARTITION: tl.constexpr,
):
    if PARTITION:
        _pack_x_partition(X, P, Q_META, FIRST, SECOND)
    else:
        _pack_x(X, P, Q_META)
    if SECOND == 0:
        T: tl.constexpr = R_META[0]
        H: tl.constexpr = R_META[1]
        N: tl.constexpr = R_META[2]
        R0: tl.constexpr = R_META[3]
        R1: tl.constexpr = R_META[4]
        R2: tl.constexpr = R_META[5]
        O0: tl.constexpr = R_META[6]
        O1: tl.constexpr = R_META[7]
        O2: tl.constexpr = R_META[8]
        WIDE: tl.constexpr = (
            (T - 1) * R0 + (H - 1) * R1 + 63 * R2 > 2147483647
            or (T - 1) * O0 + (H - 1) * O1 + (N + 63) * O2 > 2147483647
            or T * H > 2147483647
            or (R0 > 2147483647)
            or (R1 > 2147483647)
            or (R2 > 2147483647)
            or (O0 > 2147483647)
            or (O1 > 2147483647)
            or (O2 > 2147483647)
        )
        row = _address_index(FIRST, WIDE) * 16 + tl.arange(0, 16)
        t = row // H
        h = row % H
        r = tl.arange(0, 64)
        value = tl.load(
            R
            + _address_index(t[:, None], WIDE) * R0
            + _address_index(h[:, None], WIDE) * R1
            + _address_index(r[None, :], WIDE) * R2
        ).to(tl.float32)
        tl.store(
            output_ptr
            + _address_index(t[:, None], WIDE) * O0
            + _address_index(h[:, None], WIDE) * O1
            + _address_index(N + _address_index(r[None, :], WIDE), WIDE) * O2,
            _encode(value),
        )


@triton.jit
def _pack_x_rope(
    X, P, R, output_ptr, Q_META: tl.constexpr, R_META: tl.constexpr
):
    _pack_x_rope_shared(
        X,
        P,
        R,
        output_ptr,
        Q_META,
        R_META,
        tl.program_id(0),
        tl.program_id(1),
        False,
    )


@triton.jit
def _pad_weight_rope(
    W, P, R, output_ptr, W_META: tl.constexpr, R_META: tl.constexpr
):
    K: tl.constexpr = W_META[1]
    PN: tl.constexpr = W_META[3]
    BC: tl.constexpr = W_META[7]
    NC: tl.constexpr = PN // BC
    if tl.program_id(1) < K // 16:
        if PN & PN - 1:
            _pad_weight_chunks(W, P, W_META)
        else:
            _pad_weight(W, P, W_META)
    elif tl.program_id(0) % NC == 0:
        _prepare_rope_full(
            R,
            output_ptr,
            R_META,
            (tl.program_id(1) - K // 16) * 16,
            tl.program_id(0) // NC,
        )


def _padded_weights_rope(w, pn, r, destination, t, h, k, n):
    target = w.new_empty((h, k, pn))
    raw = destination.view(torch.uint8)
    bc = pn & -pn
    _pad_weight_rope.run(
        w.view(torch.uint16),
        target.view(torch.uint16),
        r,
        raw,
        (h, k, n, pn, *w.stride(), bc),
        (t, h, n, *r.stride(), *raw.stride()),
        num_warps=4,
        groups_per_cluster=16,
        grid=(h * (pn // bc), k // 16 + t // 16),
        warmup=False,
    )
    return target


@triton.jit
def _pack_x_partition(X, C, META: tl.constexpr, FIRST, SECOND):
    _pack_x_shared(X, C, META, FIRST, SECOND, True)


@triton.jit
def _pack_x_rope_partition(
    X,
    P,
    R,
    output_ptr,
    Q_META: tl.constexpr,
    R_META: tl.constexpr,
    FIRST,
    SECOND,
):
    _pack_x_rope_shared(
        X, P, R, output_ptr, Q_META, R_META, FIRST, SECOND, True
    )


@triton.jit
def _pad_weight_chunks_partition(X, P, META: tl.constexpr, FIRST, SECOND):
    _pad_weight_chunks_shared(X, P, META, FIRST, SECOND, True)


@triton.jit
def _prepare_weight_query_rope(
    Q,
    W,
    PQ,
    PW,
    R,
    output_ptr,
    Q_META: tl.constexpr,
    W_META: tl.constexpr,
    R_META: tl.constexpr,
):
    H: tl.constexpr = W_META[0]
    K: tl.constexpr = W_META[1]
    PN: tl.constexpr = W_META[3]
    BC: tl.constexpr = W_META[7]
    WK: tl.constexpr = K // 16
    QK: tl.constexpr = K // Q_META[5]
    WEIGHT: tl.constexpr = H * (PN // BC) * WK
    pid = tl.program_id(0)
    if pid < WEIGHT:
        _pad_weight_chunks_partition(W, PW, W_META, pid // WK, pid % WK)
    else:
        _pack_x_rope_partition(
            Q,
            PQ,
            R,
            output_ptr,
            Q_META,
            R_META,
            (pid - WEIGHT) // QK,
            (pid - WEIGHT) % QK,
        )


def _padded_weight_query_rope(q, w, pn, r, destination, t, h, k, n):
    pq = q.new_empty((t, h, k))
    pw = w.new_empty((h, k, pn))
    raw = destination.view(torch.uint8)
    bk = min(32, k & -k)
    bc = pn & -pn
    programs = h * (pn // bc) * (k // 16) + t * h // 16 * (k // bk)
    _prepare_weight_query_rope.run(
        q,
        w.view(torch.uint16),
        pq,
        pw.view(torch.uint16),
        r,
        raw,
        (h, k, *q.stride(), bk, t),
        (h, k, n, pn, *w.stride(), bc),
        (t, h, n, *r.stride(), *raw.stride()),
        num_warps=4,
        groups_per_cluster=16,
        grid=(programs,),
        warmup=False,
    )
    return (pq, pw)


@triton.jit
def _prepare_weight_query(
    Q, W, PQ, PW, Q_META: tl.constexpr, W_META: tl.constexpr
):
    H: tl.constexpr = W_META[0]
    K: tl.constexpr = W_META[1]
    PN: tl.constexpr = W_META[3]
    BC: tl.constexpr = W_META[7]
    WK: tl.constexpr = K // 16
    QK: tl.constexpr = K // Q_META[5]
    WEIGHT: tl.constexpr = H * (PN // BC) * WK
    pid = tl.program_id(0)
    if pid < WEIGHT:
        _pad_weight_chunks_partition(W, PW, W_META, pid // WK, pid % WK)
    else:
        _pack_x_partition(
            Q, PQ, Q_META, (pid - WEIGHT) // QK, (pid - WEIGHT) % QK
        )


def _padded_weight_query(q, w, pn, t, h, k, n):
    pq = q.new_empty((t, h, k))
    pw = w.new_empty((h, k, pn))
    bk = min(32, k & -k)
    bc = pn & -pn
    programs = h * (pn // bc) * (k // 16) + t * h // 16 * (k // bk)
    _prepare_weight_query.run(
        q,
        w.view(torch.uint16),
        pq,
        pw.view(torch.uint16),
        (h, k, *q.stride(), bk, t),
        (h, k, n, pn, *w.stride(), bc),
        num_warps=4,
        groups_per_cluster=16,
        grid=(programs,),
        warmup=False,
    )
    return (pq, pw)


@triton.jit
def _pad_q_grouped_partition(X, P, META: tl.constexpr, PID):
    _pad_q_grouped_shared(X, P, META, PID, True)


def _padded_all(q, w, pn, pt, r, destination, t, h, k, n):
    pq = q.new_empty((pt, h, k))
    pw = w.new_empty((h, k, pn))
    raw = destination.view(torch.uint8)
    bk = min(128, k & -k)
    bc = pn & -pn
    nt = t // 16
    nk = k // bk
    programs = (
        h * (pn // bc) * (k // 16) + nt * h * nk + (pt - nt * 16) * h * nk
    )
    _prepare_padded_all.run(
        q.view(torch.uint16),
        w.view(torch.uint16),
        pq.view(torch.uint16),
        pw.view(torch.uint16),
        r,
        raw,
        (t, h, k, pt, *q.stride(), bk),
        (h, k, n, pn, *w.stride(), bc),
        (t, h, n, *r.stride(), *raw.stride()),
        num_warps=4,
        groups_per_cluster=16,
        grid=(programs,),
        warmup=False,
    )
    return (pq, pw)


@triton.jit
def _prepare_padded_all(
    Q,
    W,
    PQ,
    PW,
    R,
    output_ptr,
    Q_META: tl.constexpr,
    W_META: tl.constexpr,
    R_META: tl.constexpr,
):
    H: tl.constexpr = W_META[0]
    K: tl.constexpr = W_META[1]
    PN: tl.constexpr = W_META[3]
    BC: tl.constexpr = W_META[7]
    WK: tl.constexpr = K // 16
    WEIGHT: tl.constexpr = H * (PN // BC) * WK
    pid = tl.program_id(0)
    if pid < WEIGHT:
        _pad_weight_chunks_partition(W, PW, W_META, pid // WK, pid % WK)
    else:
        _prepare_q_rope_partition(
            Q, PQ, R, output_ptr, Q_META, R_META, pid - WEIGHT
        )


@triton.jit
def _prepare_q_rope_partition(
    X, P, R, output_ptr, Q_META: tl.constexpr, R_META: tl.constexpr, PID
):
    _prepare_q_rope_shared(X, P, R, output_ptr, Q_META, R_META, PID, True)


def _logical_weight_part(destination, q, w, r, h, pt, pn):
    t, k = (q.shape[0], q.shape[2])
    n, d = (w.shape[2], r.shape[2])
    if pt != t:
        if t >= 16 and t * h * k >= 32768 and (d == 64):
            pq = _padded_q_rope(q, pt, h, k, r, destination, n)
            r = r[:, :, :0]
            d = 0
        else:
            pq = _padded_q(q, pt, h, k)
    elif q.stride(2) != 1 and d == 64 and (t * h * k >= 32768):
        pq = torch.empty((t, h, k), dtype=q.dtype, device=q.device)
        bk = min(32, k & -k)
        _pack_x_rope.run(
            q,
            pq,
            r,
            destination.view(torch.uint8),
            (h, k, *q.stride(), bk, t),
            (t, h, n, *r.stride(), *destination.stride()),
            num_warps=4,
            groups_per_cluster=16,
            grid=(t * h // 16, k // bk),
            warmup=False,
        )
        r = r[:, :, :0]
        d = 0
    else:
        pq = q
    _positive_n(destination, pq, w, r[:, :, :0], h, n, t if pt != t else 0, pn)
    if d:
        raw = destination.view(torch.uint8)
        _rope.run(
            r,
            raw,
            (h, n, d, *r.stride(), *raw.stride(), 64, t),
            num_warps=4,
            grid=(
                (32768,)
                if t * h * ((d + 63) // 64) > 2147483647
                else (t * h, (d + 63) // 64)
            ),
            warmup=False,
        )


@triton.jit
def _gemm_logical_masked(X, W, C, META: tl.constexpr, REAL_N: tl.constexpr):
    _gemm_shared(X, W, C, META, REAL_N, True)


def absorbed_bmm_concat_cast_q_fp8(q_fp8_pad, q_nope, w_kc, q_rope, num_heads):
    t = q_nope.shape[0]
    width = w_kc.shape[-1] + q_rope.shape[-1]
    if t == 0 or num_heads == 0 or width == 0:
        return q_fp8_pad
    if _output_overlaps_inputs(q_fp8_pad, q_nope, w_kc, q_rope, t, num_heads):
        padded_width = (width + 63) // 64 * 64
        temporary = torch.empty(
            (t, num_heads, padded_width),
            dtype=q_fp8_pad.dtype,
            device=q_fp8_pad.device,
        )
        _compute_entry(temporary, q_nope, w_kc, q_rope, num_heads)
        _copy_alias_padded(
            temporary.view(torch.uint8),
            q_fp8_pad[:t, :num_heads, :width].view(torch.uint8),
            width,
        )
        return q_fp8_pad
    return _compute_entry(q_fp8_pad, q_nope, w_kc, q_rope, num_heads)


def _copy_alias_padded(source, destination, width):
    t, h = destination.shape[:2]
    if t * h > 32768:
        axis = 0 if t > 1 else 1
        count = t if axis == 0 else h
        middle = count // 2
        left = [slice(None)] * 3
        right = [slice(None)] * 3
        left[axis], right[axis] = (slice(0, middle), slice(middle, count))
        _copy_alias_padded(
            source[tuple(left)], destination[tuple(left)], width
        )
        _copy_alias_padded(
            source[tuple(right)], destination[tuple(right)], width
        )
        return
    if (width + 63) // 64 > 65535:
        middle = (width // 2 + 63) // 64 * 64
        _copy_alias_padded(
            source[:, :, :middle], destination[:, :, :middle], middle
        )
        _copy_alias_padded(
            source[:, :, middle:], destination[:, :, middle:], width - middle
        )
        return
    _copy_alias_output.run(
        source,
        destination,
        (t, h, width, 0, source.stride(1), 0, *destination.stride()),
        num_warps=4,
        groups_per_cluster=16,
        grid=(t * h, (width + 63) // 64),
        warmup=False,
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


@triton.jit
def _copy_alias_output(P, output_ptr, META: tl.constexpr):
    _copy_output_shared(P, output_ptr, META, True)


__all__ = ["absorbed_bmm_concat_cast_q_fp8"]
