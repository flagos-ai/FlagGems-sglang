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
def _address_index(index, WIDE: tl.constexpr):
    if WIDE:
        return index.to(tl.int64)
    else:
        return index


@triton.jit
def _encode(v):
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
def _encode_double(v):
    raw = v.to(tl.uint32, bitcast=True)
    mag = raw & 2147483647
    bf = mag + 32767 + (mag >> 16 & 1) >> 16
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
    COMBINED: tl.constexpr = META[24]
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
        if COMBINED or BIT_ROUND:
            code = _encode_double(acc)
        else:
            if BIT_ROUND:
                bits = acc.to(tl.uint32, bitcast=True)
                rounded = (bits + 32767 + (bits >> 16 & 1) & 4294901760).to(
                    tl.float32, bitcast=True
                )
                rounded = tl.where(
                    bits & 2139095040 == 2139095040, acc, rounded
                )
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
                        _encode(value),
                        (m < T)[:, None] & (r < D)[None, :],
                    )


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
        _encode(x),
        i < T * H * D,
    )


def absorbed_bmm_concat_cast_q_fp8(q_fp8_pad, q_nope, w_kc, q_rope, num_heads):
    n = w_kc.shape[-1]
    if n != 0:
        out = q_fp8_pad
        t, _, k = q_nope.shape
        h = num_heads
        d = q_rope.shape[-1]
        if t == 0 or h == 0:
            return out
        bm, bn, bk = (
            (64, 128 if n % 128 == 0 else 64, 64)
            if t % 64 == 0 and t * h * n >= 16777216 and q_nope.is_contiguous()
            else (16 if t <= 16 else 32, 128, 64)
        )
        tm, tn = ((t + bm - 1) // bm, (n + bn - 1) // bn)
        raw = out
        output_address = raw.data_ptr()
        separate_rope = bm == 64 and t * h * n >= 16777216 and (d > 0)
        q_nope_address = q_nope.data_ptr()
        w_kc_address = w_kc.data_ptr()
        q_rope_address = q_rope.data_ptr()
        xs, ws, rs, os = (
            q_nope.stride(),
            w_kc.stride(),
            q_rope.stride(),
            raw.stride(),
        )
        needs_temporary = _active_inputs_overlap(
            t,
            h,
            k,
            n,
            d,
            output_address,
            os,
            q_nope_address,
            xs,
            w_kc_address,
            ws,
            q_rope_address,
            rs,
        )
        if needs_temporary:
            original_address, original_strides = (output_address, os)
            raw = torch.empty(
                (t, h, n + d), dtype=q_fp8_pad.dtype, device=q_fp8_pad.device
            )
            output_address, os = (raw.data_ptr(), raw.stride())
        if separate_rope:
            _roles_all_addresses.run(
                q_nope_address,
                w_kc_address,
                q_rope_address,
                output_address,
                (
                    t,
                    h,
                    k,
                    n,
                    0 if separate_rope else d,
                    *xs,
                    *ws,
                    *rs,
                    *os,
                    bm,
                ),
                (t, h, n, d, *rs, *os),
                (
                    q_nope_address % 16 == 0,
                    w_kc_address % 16 == 0,
                    q_rope_address % 16 == 0,
                    output_address % 16 == 0,
                ),
                num_warps=4,
                num_stages=1,
                kpack=2 if bm == 64 else 1,
                grid=(
                    min(h * tm * tn, 32768)
                    + min((t * h * d + 4095) // 4096, 32768),
                ),
                warmup=False,
            )
        else:
            _fused_all_addresses.run(
                q_nope_address,
                w_kc_address,
                q_rope_address,
                output_address,
                (
                    t,
                    h,
                    k,
                    n,
                    0 if separate_rope else d,
                    *xs,
                    *ws,
                    *rs,
                    *os,
                    bm,
                ),
                (
                    q_nope_address % 16 == 0,
                    w_kc_address % 16 == 0,
                    q_rope_address % 16 == 0,
                    output_address % 16 == 0,
                ),
                num_warps=4,
                num_stages=1,
                kpack=2 if bm == 64 else 1,
                grid=(min(h * tm * tn, 32768),),
                warmup=False,
            )
        if needs_temporary:
            _copy_active_output.run(
                raw.view(torch.uint8),
                original_address,
                (t, h, n + d, *original_strides),
                num_warps=4,
                grid=(min((t * h * (n + d) + 511) // 512, 32768),),
                warmup=False,
            )
        return out
    t, h, d = (q_nope.shape[0], num_heads, q_rope.shape[-1])
    if t > 0 and h > 0 and (d > 0):
        raw = q_fp8_pad
        output_address = raw.data_ptr()
        q_rope_address = q_rope.data_ptr()
        rs, os = (q_rope.stride(), raw.stride())
        needs_temporary = _active_inputs_overlap(
            t,
            h,
            0,
            0,
            d,
            output_address,
            os,
            0,
            (0, 0, 0),
            0,
            (0, 0, 0),
            q_rope_address,
            rs,
        )
        if needs_temporary:
            original_address, original_strides = (output_address, os)
            raw = torch.empty(
                (t, h, d), dtype=q_fp8_pad.dtype, device=q_fp8_pad.device
            )
            output_address, os = (raw.data_ptr(), raw.stride())
        _copyrope_all_addresses.run(
            q_rope_address,
            output_address,
            (t, h, 0, d, *rs, *os),
            (q_rope_address % 16 == 0, output_address % 16 == 0),
            num_warps=4,
            grid=((t * h * d + 512 - 1) // 512,),
            warmup=False,
        )
        if needs_temporary:
            _copy_active_output.run(
                raw.view(torch.uint8),
                original_address,
                (t, h, d, *original_strides),
                num_warps=4,
                grid=(min((t * h * d + 511) // 512, 32768),),
                warmup=False,
            )
    return q_fp8_pad


@triton.jit(do_not_specialize=["output_ptr"])
def _fused_address(
    X, W, R, output_ptr, META: tl.constexpr, ALIGNED: tl.constexpr
):
    pointer = output_ptr.to(tl.uint64).to(tl.pointer_type(tl.uint8))
    if ALIGNED:
        pointer = tl.multiple_of(pointer, 16)
    _fused(X, W, R, pointer, META)


@triton.jit(do_not_specialize=["output_ptr"])
def _copyrope_address(
    R, output_ptr, META: tl.constexpr, ALIGNED: tl.constexpr
):
    pointer = output_ptr.to(tl.uint64).to(tl.pointer_type(tl.uint8))
    if ALIGNED:
        pointer = tl.multiple_of(pointer, 16)
    _copyrope(R, pointer, META)


@triton.jit(do_not_specialize=["X", "W", "R", "output_ptr"])
def _fused_all_addresses(
    X, W, R, output_ptr, META: tl.constexpr, ALIGNED: tl.constexpr
):
    DERIVED_BN: tl.constexpr = (
        (128 if META[3] % 128 == 0 else 64) if META[17] == 64 else 128
    )
    FULL_META: tl.constexpr = (
        META[0],
        META[1],
        META[2],
        META[3],
        META[4],
        META[5],
        META[6],
        META[7],
        META[8],
        META[9],
        META[10],
        META[11],
        META[12],
        META[13],
        META[14],
        META[15],
        META[16],
        (META[0] + META[17] - 1) // META[17],
        (META[3] + DERIVED_BN - 1) // DERIVED_BN,
        META[17],
        DERIVED_BN,
        64,
        64,
        True,
        META[17] == 64,
    )
    x = X.to(tl.uint64).to(tl.pointer_type(tl.bfloat16))
    if ALIGNED[0]:
        x = tl.multiple_of(x, 16)
    w = W.to(tl.uint64).to(tl.pointer_type(tl.bfloat16))
    if ALIGNED[1]:
        w = tl.multiple_of(w, 16)
    r = R.to(tl.uint64).to(tl.pointer_type(tl.bfloat16))
    if ALIGNED[2]:
        r = tl.multiple_of(r, 16)
    _fused_address(x, w, r, output_ptr, FULL_META, ALIGNED[3])


@triton.jit(do_not_specialize=["R", "output_ptr"])
def _copyrope_all_addresses(
    R, output_ptr, META: tl.constexpr, ALIGNED: tl.constexpr
):
    FULL_META: tl.constexpr = (
        META[0],
        META[1],
        META[2],
        META[3],
        META[4],
        META[5],
        META[6],
        META[7],
        META[8],
        META[9],
        512,
    )
    r = R.to(tl.uint64).to(tl.pointer_type(tl.bfloat16))
    if ALIGNED[0]:
        r = tl.multiple_of(r, 16)
    _copyrope_address(r, output_ptr, FULL_META, ALIGNED[1])


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


@triton.jit(do_not_specialize=["output_ptr"])
def _copy_active_output(X, output_ptr, META: tl.constexpr):
    T: tl.constexpr = META[0]
    H: tl.constexpr = META[1]
    C: tl.constexpr = META[2]
    O0: tl.constexpr = META[3]
    O1: tl.constexpr = META[4]
    O2: tl.constexpr = META[5]
    B: tl.constexpr = 512
    source = X.to(tl.pointer_type(tl.uint8))
    output = output_ptr.to(tl.uint64).to(tl.pointer_type(tl.uint8))
    total = tl.full((), T * H * C, tl.int64)
    blocks = tl.full((), (T * H * C + B - 1) // B, tl.int64)
    for block in range(
        tl.program_id(0).to(tl.int64), blocks, tl.num_programs(0).to(tl.int64)
    ):
        index = block * B + tl.arange(0, B).to(tl.int64)
        column = index % C
        head = index // C % H
        token = index // (H * C)
        value = tl.load(source + index, index < total, other=0)
        tl.store(
            output + token * O0 + head * O1 + column * O2, value, index < total
        )


@triton.jit
def _fused_role(
    X, W, R, output_ptr, META: tl.constexpr, PID, GRID: tl.constexpr
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
    COMBINED: tl.constexpr = META[24]
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
        _address_index(PID, LOOP_WIDE),
        loop_stop,
        _address_index(tl.full((), GRID, tl.int32), LOOP_WIDE),
    ):
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
        if COMBINED or BIT_ROUND:
            code = _encode_double(acc)
        else:
            if BIT_ROUND:
                bits = acc.to(tl.uint32, bitcast=True)
                rounded = (bits + 32767 + (bits >> 16 & 1) & 4294901760).to(
                    tl.float32, bitcast=True
                )
                rounded = tl.where(
                    bits & 2139095040 == 2139095040, acc, rounded
                )
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
                        _encode(value),
                        (m < T)[:, None] & (r < D)[None, :],
                    )


@triton.jit
def _rope_role(R, output_ptr, META: tl.constexpr, PID, GRID: tl.constexpr):
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
    if WIDE:
        stop = tl.full((), (T * H * D + B - 1) // B, tl.int64)
    else:
        stop = (T * H * D + B - 1) // B
    for logical_pid in range(
        _address_index(PID, WIDE),
        stop,
        _address_index(tl.full((), GRID, tl.int32), WIDE),
    ):
        i = _address_index(logical_pid, WIDE) * B + tl.arange(0, B)
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
            _encode(x),
            i < T * H * D,
        )


@triton.jit(do_not_specialize=["X", "W", "R", "output_ptr"])
def _roles_all_addresses(
    X,
    W,
    R,
    output_ptr,
    META: tl.constexpr,
    ROPE_META: tl.constexpr,
    ALIGNED: tl.constexpr,
):
    DERIVED_BN: tl.constexpr = (
        (128 if META[3] % 128 == 0 else 64) if META[17] == 64 else 128
    )
    FULL_META: tl.constexpr = (
        META[0],
        META[1],
        META[2],
        META[3],
        META[4],
        META[5],
        META[6],
        META[7],
        META[8],
        META[9],
        META[10],
        META[11],
        META[12],
        META[13],
        META[14],
        META[15],
        META[16],
        (META[0] + META[17] - 1) // META[17],
        (META[3] + DERIVED_BN - 1) // DERIVED_BN,
        META[17],
        DERIVED_BN,
        64,
        64,
        True,
        META[17] == 64,
    )
    FULL_ROPE_META: tl.constexpr = (
        ROPE_META[0],
        ROPE_META[1],
        ROPE_META[2],
        ROPE_META[3],
        ROPE_META[4],
        ROPE_META[5],
        ROPE_META[6],
        ROPE_META[7],
        ROPE_META[8],
        ROPE_META[9],
        4096,
    )
    MATRIX_GRID: tl.constexpr = min(
        FULL_META[1] * FULL_META[17] * FULL_META[18], 32768
    )
    ROPE_GRID: tl.constexpr = min(
        (
            FULL_ROPE_META[0] * FULL_ROPE_META[1] * FULL_ROPE_META[3]
            + FULL_ROPE_META[10]
            - 1
        )
        // FULL_ROPE_META[10],
        32768,
    )
    x = X.to(tl.uint64).to(tl.pointer_type(tl.bfloat16))
    w = W.to(tl.uint64).to(tl.pointer_type(tl.bfloat16))
    r = R.to(tl.uint64).to(tl.pointer_type(tl.bfloat16))
    o = output_ptr.to(tl.uint64).to(tl.pointer_type(tl.uint8))
    if ALIGNED[0]:
        x = tl.multiple_of(x, 16)
    if ALIGNED[1]:
        w = tl.multiple_of(w, 16)
    if ALIGNED[2]:
        r = tl.multiple_of(r, 16)
    if ALIGNED[3]:
        o = tl.multiple_of(o, 16)
    pid = tl.program_id(0)
    if pid < MATRIX_GRID:
        _fused_role(x, w, r, o, FULL_META, pid, MATRIX_GRID)
    else:
        _rope_role(r, o, FULL_ROPE_META, pid - MATRIX_GRID, ROPE_GRID)


__all__ = ["absorbed_bmm_concat_cast_q_fp8"]
