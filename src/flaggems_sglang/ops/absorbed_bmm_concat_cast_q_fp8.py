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
def _encode_pair_bf16(value):
    if value.dtype == tl.bfloat16:
        raw = value.to(tl.uint16, bitcast=True)
    else:
        raw = (value.to(tl.uint32, bitcast=True) >> 16).to(tl.uint16)
    return tl.inline_asm_elementwise(
        (
            "{\n"
            " .reg .b32 one, minimum, cap, mag0, bounded0, exponent0, magic0, rounded0;\n"
            " .reg .b32 mag1, bounded1, exponent1, magic1, rounded1, packed, signs;\n"
            " mov.b32 one, 0x3f803f80;\n"
            " mov.b32 minimum, 0x3c803c80;\n"
            " mov.b32 cap, 0x43f043f0;\n"
            " and.b32 mag0, $1, 0x7fff7fff;\n"
            " min.f16x2 mag0, mag0, cap;\n"
            " max.f16x2 bounded0, mag0, minimum;\n"
            " shr.u32 exponent0, bounded0, 7;\n"
            " and.b32 exponent0, exponent0, 0x00ff00ff;\n"
            " mad.lo.u32 magic0, exponent0, 136, 0xfe37fe38;\n"
            " fma.rn.bf16x2 rounded0, mag0, one, magic0;\n"
            " and.b32 mag1, $2, 0x7fff7fff;\n"
            " min.f16x2 mag1, mag1, cap;\n"
            " max.f16x2 bounded1, mag1, minimum;\n"
            " shr.u32 exponent1, bounded1, 7;\n"
            " and.b32 exponent1, exponent1, 0x00ff00ff;\n"
            " mad.lo.u32 magic1, exponent1, 136, 0xfe37fe38;\n"
            " fma.rn.bf16x2 rounded1, mag1, one, magic1;\n"
            " prmt.b32 packed, rounded0, rounded1, 0x6420;\n"
            " prmt.b32 signs, $1, $2, 0x7531;\n"
            " lop3.b32 $0, packed, signs, 0x80808080, 0xd8;\n"
            "}"
        ),
        constraints="=r,r,r",
        args=[raw],
        dtype=tl.uint8,
        is_pure=True,
        pack=4,
    )


@triton.jit
def _encode_pair_select(value, PACKED: tl.constexpr):
    if PACKED:
        return _encode_pair_bf16(value)
    return _encode(value.to(tl.float32))


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
    BM: tl.constexpr = META[17]
    PACKED: tl.constexpr = META[18]
    BN: tl.constexpr = 64 if not PACKED and BM == 64 else 128
    BK: tl.constexpr = 64
    BR: tl.constexpr = 64
    TM: tl.constexpr = (T + BM - 1) // BM
    TN: tl.constexpr = (N + BN - 1) // BN
    BIT_ROUND: tl.constexpr = False
    HEAD_ORDER: tl.constexpr = T * H * N >= 16777216
    COMBINED: tl.constexpr = (
        not PACKED and BM == 64 and (BN == 64) and (T * H * N >= 16777216)
    )
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
        if HEAD_ORDER:
            local = pid // H
            h = pid % H
            ni = local % TN
            m = local // TN * BM + tl.arange(0, BM)
            n = ni * BN + tl.arange(0, BN)
        else:
            h = pid // (TM * TN)
            m = pid // TN % TM * BM + tl.arange(0, BM)
            n = pid % TN * BN + tl.arange(0, BN)
            ni = pid % TN
        k = tl.arange(0, BK)
        if WIDE:
            m = m.to(tl.int64)
            h = h.to(tl.int64)
            n = n.to(tl.int64)
            k = k.to(tl.int64)
        acc = tl.zeros((BM, BN), tl.float32)
        for ki in range(tl.cdiv(K, BK)):
            kk = ki * BK + k
            if T % BM == 0 and K % BK == 0:
                x = tl.load(X + m[:, None] * X0 + h * X1 + kk[None, :] * X2)
            else:
                x = tl.load(
                    X + m[:, None] * X0 + h * X1 + kk[None, :] * X2,
                    (m < T)[:, None] & (kk < K)[None, :],
                    other=0,
                )
            if K % BK == 0 and N % BN == 0:
                w = tl.load(W + h * W0 + kk[:, None] * W1 + n[None, :] * W2)
            else:
                w = tl.load(
                    W + h * W0 + kk[:, None] * W1 + n[None, :] * W2,
                    (kk < K)[:, None] & (n < N)[None, :],
                    other=0,
                )
            acc = tl.dot(x, w, acc, input_precision="ieee")
        if COMBINED:
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
                rounded = acc.to(tl.bfloat16)
            code = _encode_pair_select(rounded, PACKED)
        tl.store(
            output_ptr + m[:, None] * O0 + h * O1 + n[None, :] * O2,
            code,
            (m < T)[:, None] & (n < N)[None, :],
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
                    if WIDE:
                        r = r.to(tl.int64)
                    value = tl.load(
                        R + m[:, None] * R0 + h * R1 + r[None, :] * R2,
                        (m < T)[:, None] & (r < D)[None, :],
                        other=0,
                    )
                    tl.store(
                        output_ptr
                        + m[:, None] * O0
                        + h * O1
                        + (N + r[None, :]) * O2,
                        _encode_pair_select(value, PACKED),
                        (m < T)[:, None] & (r < D)[None, :],
                    )


@triton.jit
def _copyrope(R, output_ptr, META: tl.constexpr):
    T: tl.constexpr = META[0]
    H: tl.constexpr = META[1]
    D: tl.constexpr = META[2]
    R0: tl.constexpr = META[3]
    R1: tl.constexpr = META[4]
    R2: tl.constexpr = META[5]
    O0: tl.constexpr = META[6]
    O1: tl.constexpr = META[7]
    O2: tl.constexpr = META[8]
    PACKED: tl.constexpr = META[9]
    N: tl.constexpr = 0
    B: tl.constexpr = 512
    WIDE: tl.constexpr = (
        (
            T * H * D > 2147483647
            or (T - 1) * R0 + (H - 1) * R1 + (D - 1) * R2 > 2147483647
            or (T - 1) * O0 + (H - 1) * O1 + (N + D - 1) * O2 > 2147483647
        )
        or R0 > 2147483647
        or R1 > 2147483647
        or (R2 > 2147483647)
        or (O0 > 2147483647)
        or (O1 > 2147483647)
        or (O2 > 2147483647)
    )
    if WIDE:
        i = tl.program_id(0).to(tl.int64) * B + tl.arange(0, B)
    else:
        i = tl.program_id(0) * B + tl.arange(0, B)
    r = i % D
    h = i // D % H
    t = i // (H * D)
    x = tl.load(R + t * R0 + h * R1 + r * R2, i < T * H * D, other=0)
    tl.store(
        output_ptr + t * O0 + h * O1 + (N + r) * O2,
        _encode_pair_select(x, PACKED),
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
        if torch.version.hip is not None:
            bm, bn, bk = (
                (64, 64, 64)
                if t % 64 == 0
                and t * h * n >= 4194304
                and q_nope.is_contiguous()
                else (32, 128, 64)
            )
        else:
            bm, bn, bk = (64, 128, 64)
        tm, tn = ((t + bm - 1) // bm, (n + bn - 1) // bn)
        raw = out
        output_address = raw.data_ptr()
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
        if (
            torch.version.hip is None
            and t * h * n >= 16777216
            and (k % bk == 0)
            and (k <= 2 * bk)
        ):
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
                    d,
                    *xs,
                    *ws,
                    *rs,
                    *os,
                    bm,
                    torch.version.hip is None,
                ),
                (
                    q_nope_address % 16 == 0,
                    w_kc_address % 16 == 0,
                    q_rope_address % 16 == 0,
                    output_address % 16 == 0,
                ),
                num_warps=4,
                num_stages=2,
                grid=(min(h * tm * tn, 32768),),
                warmup=False,
                maxnreg=128,
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
                    d,
                    *xs,
                    *ws,
                    *rs,
                    *os,
                    bm,
                    torch.version.hip is None,
                ),
                (
                    q_nope_address % 16 == 0,
                    w_kc_address % 16 == 0,
                    q_rope_address % 16 == 0,
                    output_address % 16 == 0,
                ),
                num_warps=4,
                num_stages=2,
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
            (t, h, d, *rs, *os, torch.version.hip is None),
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
    x = X.to(tl.uint64).to(tl.pointer_type(tl.bfloat16))
    w = W.to(tl.uint64).to(tl.pointer_type(tl.bfloat16))
    r = R.to(tl.uint64).to(tl.pointer_type(tl.bfloat16))
    if ALIGNED[0]:
        x = tl.multiple_of(x, 16)
    if ALIGNED[1]:
        w = tl.multiple_of(w, 16)
    if ALIGNED[2]:
        r = tl.multiple_of(r, 16)
    _fused_address(x, w, r, output_ptr, META, ALIGNED[3])


@triton.jit(do_not_specialize=["R", "output_ptr"])
def _copyrope_all_addresses(
    R, output_ptr, META: tl.constexpr, ALIGNED: tl.constexpr
):
    r = R.to(tl.uint64).to(tl.pointer_type(tl.bfloat16))
    if ALIGNED[0]:
        r = tl.multiple_of(r, 16)
    _copyrope_address(r, output_ptr, META, ALIGNED[1])


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


__all__ = ["absorbed_bmm_concat_cast_q_fp8"]
