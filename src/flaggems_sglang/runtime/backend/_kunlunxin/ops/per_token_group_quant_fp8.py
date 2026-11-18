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
import triton as tr
import triton.language as tl


@tr.jit
def _magnitude(x):
    return tl.abs(x).to(tl.int32, bitcast=True)


@tr.jit
def _encode(v):
    bits = v.to(tl.int32, bitcast=True)
    mag = tl.minimum(bits & 0x7FFFFFFF, 0x43E00000)
    normal = (mag + 0x0407FFFF + ((mag >> 20) & 1)) >> 20
    subnormal = (tl.abs(v) + 16384.0).to(tl.int32, bitcast=True) - 0x46800000
    code = tl.where(mag < 0x3C800000, subnormal, normal)
    code = tl.where((bits & 0x7FFFFFFF) > 0x7F800000, 127, code)
    return (code | ((bits >> 24) & 128)).to(tl.uint8)


@tr.jit
def _load_bits(X, offsets, mask, F: tl.constexpr):
    if F == 7 or F == 8:
        raw = tl.load(X.to(tl.pointer_type(tl.int16)) + offsets, mask, 0)
    elif F == 9:
        raw = tl.load(X.to(tl.pointer_type(tl.int32)) + offsets, mask, 0)
    else:
        raw = tl.load(X + offsets, mask, 0)
    if F == 7:
        return raw.to(tl.int32) << 16
    elif F == 9:
        return raw.to(tl.int32)
    elif F == 8:
        bits = raw.to(tl.int32)
        exponent = (bits >> 10) & 31
        fraction = bits & 1023
        p = tl.full(fraction.shape, 0, tl.int32)
        for shift in tl.static_range(1, 10):
            p = tl.where(fraction >= (1 << shift), shift, p)
        sub = ((p + 103) << 23) | ((fraction - (1 << p)) << (23 - p))
        sub = tl.where(fraction == 0, 0, sub)
        code = ((exponent + 112) << 23) | (fraction << 13)
        code = tl.where(exponent == 0, sub, code)
        code = tl.where(exponent == 31, 0x7F800000 | (fraction << 13), code)
        return code | ((bits & 32768) << 16)
    elif F == 6:
        bits = raw.to(tl.int32)
        code = tl.where(bits == 0, 0x00400000, bits << 23)
        return tl.where(bits == 255, 0x7FC00000, code)
    elif F == 5:
        bits = raw.to(tl.int64)
        exponent = (bits >> 52) & 2047
        fraction = bits & 4503599627370495
        significand = fraction | 4503599627370496
        shift = tl.minimum(tl.maximum(926 - exponent, 29), 53)
        rounded = significand >> shift
        remainder = significand & ((1 << shift) - 1)
        halfway = 1 << (shift - 1)
        rounded += (
            (remainder > halfway)
            | ((remainder == halfway) & ((rounded & 1) != 0))
        ).to(tl.int64)
        code = tl.where(
            exponent >= 897, ((exponent - 897) << 23) + rounded, rounded
        )
        code = tl.where(exponent < 873, 0, code)
        code = tl.where(exponent > 1150, 2139095040, code)
        code = tl.where((exponent == 2047) & (fraction != 0), 2143289344, code)
        code = code.to(tl.int32) | ((bits >> 32).to(tl.int32) & -2147483648)
        return code
    elif F == 0:
        return raw.to(tl.float32).to(tl.int32, bitcast=True)
    else:
        bits = raw.to(tl.int32)
        mag = bits & 127
        if F == 1 or F == 3:
            M: tl.constexpr = 3
            BIAS: tl.constexpr = 7 if F == 1 else 8
        else:
            M: tl.constexpr = 2
            BIAS: tl.constexpr = 15 if F == 2 else 16
        exponent = mag >> M
        mantissa = mag & ((1 << M) - 1)
        p = tl.where(mantissa >= 4, 2, tl.where(mantissa >= 2, 1, 0))
        sub = ((128 - BIAS - M + p) << 23) | (
            (mantissa - (1 << p)) << (23 - p)
        )
        sub = tl.where(mantissa == 0, 0, sub)
        code = ((exponent + 127 - BIAS) << 23) | (mantissa << (23 - M))
        code = tl.where(exponent == 0, sub, code)
        if F == 1:
            code = tl.where(mag == 127, 0x7FC00000, code)
        elif F == 2:
            code = tl.where(
                exponent == 31,
                tl.where(mantissa == 0, 0x7F800000, 0x7FC00000),
                code,
            )
        else:
            code = tl.where(bits == 128, 0x7FC00000, code)
        return code | ((bits & 128) << 24)


@tr.jit
def _quant(
    X,
    Q,
    S,
    G: tl.constexpr,
    N: tl.constexpr,
    R: tl.constexpr,
    B: tl.constexpr,
    F: tl.constexpr,
):
    if N >= 2147483648:
        rows = tl.program_id(0).to(tl.int64) * R + tl.arange(0, R)
    else:
        rows = tl.program_id(0) * R + tl.arange(0, R)
    if N * G >= 2147483648:
        rows = rows.to(tl.int64)
    cols = tl.arange(0, B)
    if G <= B:
        offsets = rows[:, None] * G + cols[None, :]
        mask = (rows[:, None] < N) & (cols[None, :] < G)
        bits = _load_bits(X, offsets, mask, F)
        mag = bits & 0x7FFFFFFF
        maximum = tl.maximum(tl.max(mag, 1), 786163455)
        s = maximum.to(tl.float32, bitcast=True) / 448.0
        x = bits.to(tl.float32, bitcast=True)
        q = tl.minimum(tl.maximum(x / s[:, None], -448.0), 448.0)
        code = tl.where(
            (mag >= 0x7F800000) | (maximum[:, None] > 0x7F800000),
            127,
            _encode(q),
        ).to(tl.uint8)
        tl.store(Q + offsets, code, mask)
    else:
        a = tl.full((R, B), 0, tl.int32)
        for start in range(tr.cdiv(G, B)):
            k = start * B + cols
            offsets = rows[:, None] * G + k[None, :]
            bits = _load_bits(
                X, offsets, (rows[:, None] < N) & (k[None, :] < G), F
            )
            a = tl.maximum(a, bits & 0x7FFFFFFF)
        maximum = tl.maximum(tl.max(a, 1), 786163455)
        s = maximum.to(tl.float32, bitcast=True) / 448.0
        for start in range(tr.cdiv(G, B)):
            k = start * B + cols
            offsets = rows[:, None] * G + k[None, :]
            mask = (rows[:, None] < N) & (k[None, :] < G)
            bits = _load_bits(X, offsets, mask, F)
            mag = bits & 0x7FFFFFFF
            x = bits.to(tl.float32, bitcast=True)
            q = tl.minimum(tl.maximum(x / s[:, None], -448.0), 448.0)
            code = tl.where(
                (mag >= 0x7F800000) | (maximum[:, None] > 0x7F800000),
                127,
                _encode(q),
            ).to(tl.uint8)
            tl.store(Q + offsets, code, mask)
    tl.store(S + rows, s, rows < N)


@tr.jit
def _packed(
    X, Q, S, G: tl.constexpr, N: tl.constexpr, R: tl.constexpr, P: tl.constexpr
):
    rows = tl.program_id(0) * R + tl.arange(0, R)
    cols = tl.arange(0, G // P)
    offsets = rows[:, None] * (G // P) + cols[None, :]
    if P == 2:
        raw = tl.load(
            X.to(tl.pointer_type(tl.uint32)) + offsets,
            offset_state_policy="continuous",
        )
    else:
        raw = tl.load(
            X.to(tl.pointer_type(tl.uint64)) + offsets,
            offset_state_policy="continuous",
        )
    low = raw.to(tl.int32)
    a = (low << 16).to(tl.float32, bitcast=True)
    b = (low & -65536).to(tl.float32, bitcast=True)
    magnitude = tl.maximum(_magnitude(a), _magnitude(b))
    if P == 4:
        high = (raw >> 32).to(tl.int32)
        c = (high << 16).to(tl.float32, bitcast=True)
        d = (high & -65536).to(tl.float32, bitcast=True)
        magnitude = tl.maximum(
            magnitude, tl.maximum(_magnitude(c), _magnitude(d))
        )
    scale = (
        tl.maximum(tl.max(magnitude, 1), 786163455).to(
            tl.float32, bitcast=True
        )
        / 448.0
    )
    qa = _encode(a / scale[:, None]).to(tl.int32)
    qb = _encode(b / scale[:, None]).to(tl.int32)
    word = qa | (qb << 8)
    if P == 4:
        qc = _encode(c / scale[:, None]).to(tl.int32)
        qd = _encode(d / scale[:, None]).to(tl.int32)
        word = word | (qc << 16) | (qd << 24)
        ptr = Q.to(tl.pointer_type(tl.int32))
    else:
        ptr = Q.to(tl.pointer_type(tl.int16))
    tl.store(ptr + offsets, word, offset_state_policy="continuous")
    tl.store(S + rows, scale, offset_state_policy="continuous")


def per_token_group_quant_fp8(x, group_size):
    dtype = x.dtype
    kind = 0
    if dtype == torch.bfloat16:
        kind = 7
    elif dtype == torch.float16:
        kind = 8
    elif dtype == torch.float32:
        kind = 9
    elif dtype == torch.float8_e4m3fn:
        kind = 1
    elif dtype == torch.float8_e5m2:
        kind = 2
    elif dtype == torch.float8_e4m3fnuz:
        kind = 3
    elif dtype == torch.float8_e5m2fnuz:
        kind = 4
    elif dtype == torch.float64:
        kind = 5
    elif dtype == getattr(torch, "float8_e8m0fnu", None):
        kind = 6
    shape = x.shape
    q = x.new_empty(shape, dtype=torch.uint8)
    s = x.new_empty(
        shape[:-1] + (shape[-1] // group_size,), dtype=torch.float32
    )
    groups = x.numel() // group_size
    if (
        dtype == torch.bfloat16
        and groups >= 1024
        and groups % 64 == 0
        and 4 <= group_size <= 128
        and group_size & (group_size - 1) == 0
        and groups * group_size < 2147483648
        and x.data_ptr() % 8 == 0
        and q.data_ptr() % 4 == 0
    ):
        _packed.run(
            x,
            q,
            s,
            group_size,
            groups,
            64,
            4,
            grid=(groups // 64,),
            warmup=False,
            num_warps=4,
            enable_fp_fusion=False,
        )
    elif groups:
        block = min(tr.next_power_of_2(group_size), 4096)
        rows = min(32, max(1, 4096 // block))
        _quant.run(
            (
                x.view(torch.int64)
                if kind == 5
                else x.view(torch.uint8) if 1 <= kind <= 6 else x
            ),
            q,
            s,
            group_size,
            groups,
            rows,
            block,
            kind,
            num_warps=4,
            enable_fp_fusion=False,
            grid=(tr.cdiv(groups, rows),),
            warmup=False,
        )
    return q.view(torch.float8_e4m3fn), s


__all__ = ["per_token_group_quant_fp8"]
