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
import triton.language.core as core


@core.extern
def _byte_perm(left, right, selector, _semantic=None):
    return core.extern_elementwise(
        "",
        "",
        [left, right, selector],
        {
            (core.dtype("int32"), core.dtype("int32"), core.dtype("int32")): (
                "llvm.mxc.byte.perm",
                core.dtype("int32"),
            )
        },
        is_pure=True,
        _semantic=_semantic,
    )


@triton.jit
def _pack_four_bytes(raw):
    grouped = tl.reshape(raw, (raw.shape[0], raw.shape[1] // 4, 2, 2))
    even, odd = tl.split(grouped)
    a, c = tl.split(even)
    b, d = tl.split(odd)
    ab = _byte_perm(a.to(tl.int32), b.to(tl.int32), 262148)
    cd = _byte_perm(c.to(tl.int32), d.to(tl.int32), 262148)
    return _byte_perm(ab, cd, 16778500).to(tl.uint32)


@triton.jit
def _address_index(index, WIDE: tl.constexpr):
    if WIDE:
        return index.to(tl.int64)
    else:
        return index


@triton.jit
def _encode_column(v, RAW: tl.constexpr = False):
    raw = v.to(tl.uint32, bitcast=True)
    sign = raw >> 24 & 128
    mag = tl.minimum(raw & 2147483647, 1139802112)
    exponent = tl.maximum(mag >> 23, 121)
    magic_bits = exponent * 8388616 + 167771192
    absolute = mag.to(tl.float32, bitcast=True)
    rounded = (absolute + magic_bits.to(tl.float32, bitcast=True)).to(
        tl.uint32, bitcast=True
    )
    if RAW:
        return rounded | sign
    else:
        return (rounded | sign).to(tl.uint8)


@triton.jit
def _fused_transposed(X, W, R, output_ptr, META: tl.constexpr):
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
        gm = TM
        group = local // (gm * TN)
        size = tl.minimum(TM - group * gm, gm)
        mi = group * gm + local % size
        ni = local % (gm * TN) // size
        pid = head * (TM * TN) + mi * TN + ni
        h = pid // (TM * TN)
        m = pid // TN % TM * BM + tl.arange(0, BM)
        n = pid % TN * BN + tl.arange(0, BN)
        k = tl.arange(0, BK)
        acc = tl.zeros((BN, BM), tl.float32)
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
            acc = tl.dot(tl.trans(w), tl.trans(x), acc, input_precision="ieee")
        acc = tl.trans(acc)
        if BIT_ROUND:
            bits = acc.to(tl.uint32, bitcast=True)
            rounded = (bits + 32767 + (bits >> 16 & 1) & 4294901760).to(
                tl.float32, bitcast=True
            )
            rounded = tl.where(bits & 2139095040 == 2139095040, acc, rounded)
        else:
            rounded = acc.to(tl.bfloat16).to(tl.float32)
        code = _encode_column(rounded)
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
                        _encode_column(value),
                        (m < T)[:, None] & (r < D)[None, :],
                    )


@triton.jit
def _fused_column(X, W, R, output_ptr, META: tl.constexpr):
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
        gm = TM
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
        code = _encode_column(rounded)
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
                        _encode_column(value),
                        (m < T)[:, None] & (r < D)[None, :],
                    )


def _positive_n(
    out,
    t,
    h,
    k,
    n,
    d,
    q_nope_address,
    w_kc_address,
    q_rope_address,
    output_address,
    xs,
    ws,
    rs,
    os,
):
    bm, bn, bk = (16, 64, 32) if t < 32 else (64, 64, 64)
    tm, tn = ((t + bm - 1) // bm, (n + bn - 1) // bn)
    paired = (
        t * h * n >= 16777216
        and k % 64 == 0
        and (n % 128 == 0)
        and (xs[2] == 1)
        and (ws[2] == 1)
        and (os[2] == 1)
        and (q_nope_address % 16 == 0)
        and (w_kc_address % 16 == 0)
        and (output_address % 16 == 0)
        and (xs[0] % 8 == 0)
        and (xs[1] % 8 == 0)
        and (ws[0] % 8 == 0)
        and (ws[1] % 8 == 0)
        and (os[0] % 16 == 0)
        and (os[1] % 16 == 0)
        and (
            d == 0
            or (
                rs[2] == 1
                and q_rope_address % 16 == 0
                and (rs[0] % 8 == 0)
                and (rs[1] % 8 == 0)
            )
        )
    )
    kernel = (
        _fused_paired_n_all_addresses
        if paired
        else (
            _fused_transposed_all_addresses
            if t * h * n >= 16777216
            else _fused_column_all_addresses
        )
    )
    kernel.run(
        q_nope_address,
        w_kc_address,
        q_rope_address,
        output_address,
        (t, h, k, n, d, *xs, *ws, *rs, *os, tm, tn, bm, bn, bk, 64, False),
        (
            q_nope_address % 16 == 0,
            w_kc_address % 16 == 0,
            q_rope_address % 16 == 0,
            output_address % 16 == 0,
        ),
        num_warps=4,
        num_stages=1,
        grid=(
            (
                min(h * ((tm + 3) // 4) * ((tn + 1) // 2), 32768)
                if paired
                else min(h * tm * tn, 32768)
            ),
        ),
        warmup=False,
    )
    return out


@triton.jit
def _copyrope(R, output_ptr, META: tl.constexpr):
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
    i = _address_index(tl.program_id(0), WIDE) * B + tl.arange(0, B)
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
        _encode_column(x),
        i < T * H * D,
    )


def _compute_entry(out, t, h, k, n, d, qa, wa, ra, oa, xs, ws, rs, os):
    if n != 0:
        return _positive_n(out, t, h, k, n, d, qa, wa, ra, oa, xs, ws, rs, os)
    if d > 0:
        _copyrope_all_addresses.run(
            ra,
            oa,
            (t, h, 0, d, *rs, *os, 512),
            (ra % 16 == 0, oa % 16 == 0),
            num_warps=4,
            grid=((t * h * d + 512 - 1) // 512,),
            warmup=False,
        )
    return out


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


def _compact_wide_tensor(tensor):
    if tensor.is_contiguous():
        return tensor
    return tensor.contiguous() if _layout_span(tensor) > 2147483647 else tensor


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
        _copy_bounded_layout[(t * h * c + 511) // 512,](
            source, destination, (t, h, c, *xs, *os, 512), num_warps=4
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
    t, _, k = q_nope.shape
    h = num_heads
    if t == 0 or h == 0:
        return q_fp8_pad
    q_nope = _compact_wide_tensor(q_nope)
    w_kc = _compact_wide_tensor(w_kc)
    q_rope = _compact_wide_tensor(q_rope)
    n, d = (w_kc.shape[-1], q_rope.shape[-1])
    width = n + d
    if width == 0:
        return q_fp8_pad
    qa, xs = (q_nope.data_ptr(), q_nope.stride()) if n > 0 else (0, (0, 0, 0))
    wa, ws = (w_kc.data_ptr(), w_kc.stride()) if n > 0 else (0, (0, 0, 0))
    ra, rs = (q_rope.data_ptr(), q_rope.stride())
    oa, os = (q_fp8_pad.data_ptr(), q_fp8_pad.stride())
    if _active_output_span(
        q_fp8_pad.shape, os, t, h
    ) > 2147483647 or _active_inputs_overlap(
        t, h, k, n, d, oa, os, qa, xs, wa, ws, ra, rs
    ):
        destination = q_fp8_pad[:t, :h, :width]
        temporary = torch.empty(
            (t, h, width), dtype=q_fp8_pad.dtype, device=q_fp8_pad.device
        )
        _compute_entry(
            temporary,
            t,
            h,
            k,
            n,
            d,
            qa,
            wa,
            ra,
            temporary.data_ptr(),
            xs,
            ws,
            rs,
            temporary.stride(),
        )
        _copy_partition(
            temporary.view(torch.uint8), destination.view(torch.uint8)
        )
        return q_fp8_pad
    return _compute_entry(
        q_fp8_pad, t, h, k, n, d, qa, wa, ra, oa, xs, ws, rs, os
    )


@triton.jit(do_not_specialize=["output_ptr"])
def _fused_column_address(
    X, W, R, output_ptr, META: tl.constexpr, ALIGNED: tl.constexpr
):
    pointer = output_ptr.to(tl.uint64).to(tl.pointer_type(tl.uint8))
    if ALIGNED:
        pointer = tl.multiple_of(pointer, 16)
    _fused_column(X, W, R, pointer, META)


@triton.jit(do_not_specialize=["output_ptr"])
def _fused_transposed_address(
    X, W, R, output_ptr, META: tl.constexpr, ALIGNED: tl.constexpr
):
    pointer = output_ptr.to(tl.uint64).to(tl.pointer_type(tl.uint8))
    if ALIGNED:
        pointer = tl.multiple_of(pointer, 16)
    _fused_transposed(X, W, R, pointer, META)


@triton.jit(do_not_specialize=["output_ptr"])
def _copyrope_address(
    R, output_ptr, META: tl.constexpr, ALIGNED: tl.constexpr
):
    pointer = output_ptr.to(tl.uint64).to(tl.pointer_type(tl.uint8))
    if ALIGNED:
        pointer = tl.multiple_of(pointer, 16)
    _copyrope(R, pointer, META)


@triton.jit(do_not_specialize=["X", "W", "R", "output_ptr"])
def _fused_column_all_addresses(
    X, W, R, output_ptr, META: tl.constexpr, ALIGNED: tl.constexpr
):
    x = X.to(tl.uint64).to(tl.pointer_type(tl.bfloat16))
    if ALIGNED[0]:
        x = tl.multiple_of(x, 16)
    w = W.to(tl.uint64).to(tl.pointer_type(tl.bfloat16))
    if ALIGNED[1]:
        w = tl.multiple_of(w, 16)
    r = R.to(tl.uint64).to(tl.pointer_type(tl.bfloat16))
    if ALIGNED[2]:
        r = tl.multiple_of(r, 16)
    _fused_column_address(x, w, r, output_ptr, META, ALIGNED[3])


@triton.jit(do_not_specialize=["X", "W", "R", "output_ptr"])
def _fused_transposed_all_addresses(
    X, W, R, output_ptr, META: tl.constexpr, ALIGNED: tl.constexpr
):
    x = X.to(tl.uint64).to(tl.pointer_type(tl.bfloat16))
    if ALIGNED[0]:
        x = tl.multiple_of(x, 16)
    w = W.to(tl.uint64).to(tl.pointer_type(tl.bfloat16))
    if ALIGNED[1]:
        w = tl.multiple_of(w, 16)
    r = R.to(tl.uint64).to(tl.pointer_type(tl.bfloat16))
    if ALIGNED[2]:
        r = tl.multiple_of(r, 16)
    _fused_transposed_address(x, w, r, output_ptr, META, ALIGNED[3])


@triton.jit(do_not_specialize=["R", "output_ptr"])
def _copyrope_all_addresses(
    R, output_ptr, META: tl.constexpr, ALIGNED: tl.constexpr
):
    r = R.to(tl.uint64).to(tl.pointer_type(tl.bfloat16))
    if ALIGNED[0]:
        r = tl.multiple_of(r, 16)
    _copyrope_address(r, output_ptr, META, ALIGNED[1])


def _active_output_span(shape, strides, tokens, heads):
    return (
        (min(tokens, shape[0]) - 1) * strides[0]
        + (min(heads, shape[1]) - 1) * strides[1]
        + (shape[2] - 1) * strides[2]
    )


@triton.jit
def _fused_paired_n(X, W, R, output_ptr, META: tl.constexpr):
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
    PROGRAMS: tl.constexpr = H * ((TM + 3) // 4) * ((TN + 1) // 2)
    LOOP_WIDE: tl.constexpr = PROGRAMS - 1 + (
        PROGRAMS if PROGRAMS < 32768 else 32768
    ) > 2147483647 or (
        TM * BM - 1 > 2147483647 or (TN + 1) // 2 * 2 * BN - 1 > 2147483647
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
        local = pid % ((TM + 3) // 4 * ((TN + 1) // 2))
        h = pid // ((TM + 3) // 4 * ((TN + 1) // 2))
        pair_m = local % ((TM + 3) // 4)
        ni = local // ((TM + 3) // 4) * 2
        n = ni * BN + tl.arange(0, BN)
        n1 = n + BN
        k = tl.arange(0, BK)
        weights0 = ()
        weights1 = ()
        for wi in tl.static_range((K + BK - 1) // BK):
            wk = wi * BK + k
            if K % BK == 0 and N % BN == 0:
                weight0 = tl.load(
                    W
                    + _address_index(h, WIDE) * W0
                    + _address_index(wk[:, None], WIDE) * W1
                    + _address_index(n[None, :], WIDE) * W2
                )
            else:
                weight0 = tl.load(
                    W
                    + _address_index(h, WIDE) * W0
                    + _address_index(wk[:, None], WIDE) * W1
                    + _address_index(n[None, :], WIDE) * W2,
                    (wk < K)[:, None] & (n < N)[None, :],
                    other=0,
                )
            if K % BK == 0 and N % (2 * BN) == 0:
                weight1 = tl.load(
                    W
                    + _address_index(h, WIDE) * W0
                    + _address_index(wk[:, None], WIDE) * W1
                    + _address_index(n1[None, :], WIDE) * W2
                )
            else:
                weight1 = tl.load(
                    W
                    + _address_index(h, WIDE) * W0
                    + _address_index(wk[:, None], WIDE) * W1
                    + _address_index(n1[None, :], WIDE) * W2,
                    (wk < K)[:, None] & (n1 < N)[None, :],
                    other=0,
                )
            weights0 += (weight0,)
            weights1 += (weight1,)
        for member in tl.static_range(4):
            mi = pair_m * 4 + member
            if TM % 4 == 0 or mi < TM:
                m = mi * BM + tl.arange(0, BM)
                acc = tl.zeros((BN, BM), tl.float32)
                acc1 = tl.zeros((BN, BM), tl.float32)
                for ki in tl.static_range((K + BK - 1) // BK):
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
                    w = weights0[ki]
                    acc = tl.dot(
                        tl.trans(w), tl.trans(x), acc, input_precision="ieee"
                    )
                    w1 = weights1[ki]
                    acc1 = tl.dot(
                        tl.trans(w1), tl.trans(x), acc1, input_precision="ieee"
                    )
                acc = tl.trans(acc)
                if BIT_ROUND:
                    bits = acc.to(tl.uint32, bitcast=True)
                    rounded = (
                        bits + 32767 + (bits >> 16 & 1) & 4294901760
                    ).to(tl.float32, bitcast=True)
                    rounded = tl.where(
                        bits & 2139095040 == 2139095040, acc, rounded
                    )
                else:
                    rounded = acc.to(tl.bfloat16).to(tl.float32)
                code = _pack_four_bytes(_encode_column(rounded, True))
                nq = ni * (BN // 4) + tl.arange(0, BN // 4)
                tl.store(
                    output_ptr.to(tl.pointer_type(tl.uint32))
                    + _address_index(m[:, None], WIDE) * (O0 // 4)
                    + _address_index(h, WIDE) * (O1 // 4)
                    + _address_index(nq[None, :], WIDE),
                    code,
                    (m < T)[:, None],
                )
                acc1 = tl.trans(acc1)
                if BIT_ROUND:
                    bits1 = acc1.to(tl.uint32, bitcast=True)
                    rounded1 = (
                        bits1 + 32767 + (bits1 >> 16 & 1) & 4294901760
                    ).to(tl.float32, bitcast=True)
                    rounded1 = tl.where(
                        bits1 & 2139095040 == 2139095040, acc1, rounded1
                    )
                else:
                    rounded1 = acc1.to(tl.bfloat16).to(tl.float32)
                code1 = _pack_four_bytes(_encode_column(rounded1, True))
                nq1 = (ni + 1) * (BN // 4) + tl.arange(0, BN // 4)
                tl.store(
                    output_ptr.to(tl.pointer_type(tl.uint32))
                    + _address_index(m[:, None], WIDE) * (O0 // 4)
                    + _address_index(h, WIDE) * (O1 // 4)
                    + _address_index(nq1[None, :], WIDE),
                    code1,
                    (m < T)[:, None],
                )
                if D > 0:
                    if ni == 0:
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
                                _encode_column(value),
                                (m < T)[:, None] & (r < D)[None, :],
                            )


@triton.jit(do_not_specialize=["output_ptr"])
def _fused_paired_n_address(
    X, W, R, output_ptr, META: tl.constexpr, ALIGNED: tl.constexpr
):
    pointer = output_ptr.to(tl.uint64).to(tl.pointer_type(tl.uint8))
    if ALIGNED:
        pointer = tl.multiple_of(pointer, 16)
    _fused_paired_n(X, W, R, pointer, META)


@triton.jit(do_not_specialize=["X", "W", "R", "output_ptr"])
def _fused_paired_n_all_addresses(
    X, W, R, output_ptr, META: tl.constexpr, ALIGNED: tl.constexpr
):
    x = X.to(tl.uint64).to(tl.pointer_type(tl.bfloat16))
    if ALIGNED[0]:
        x = tl.multiple_of(x, 16)
    w = W.to(tl.uint64).to(tl.pointer_type(tl.bfloat16))
    if ALIGNED[1]:
        w = tl.multiple_of(w, 16)
    r = R.to(tl.uint64).to(tl.pointer_type(tl.bfloat16))
    if ALIGNED[2]:
        r = tl.multiple_of(r, 16)
    _fused_paired_n_address(x, w, r, output_ptr, META, ALIGNED[3])


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
