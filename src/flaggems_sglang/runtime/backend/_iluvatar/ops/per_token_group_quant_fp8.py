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
def _input_scale_ptr(X, S):
    if (
        S.dtype == tl.int64
        or S.dtype == tl.uint64
        or S.dtype == tl.int32
        or S.dtype == tl.uint32
    ):
        S = S.to(tl.pointer_type(tl.float32))
    if (
        X.dtype == tl.int64
        or X.dtype == tl.uint64
        or X.dtype == tl.int32
        or X.dtype == tl.uint32
    ):
        X = tl.multiple_of(X.to(tl.pointer_type(tl.bfloat16)), 16)
    return X, S


@tr.jit
def _magnitude(x):
    return x.to(tl.int32, bitcast=True) & 0x7FFFFFFF


@tr.jit
def _encode(v):
    bits = v.to(tl.uint32, bitcast=True)
    mag = tl.minimum(bits & 0x7FFFFFFF, 0x43E00000)
    normal = (mag + 0x0407FFFF + ((mag >> 20) & 1)) >> 20
    subnormal = (tl.abs(v) + 16384.0).to(tl.uint32, bitcast=True) - 0x46800000
    code = tl.where(mag < 0x3C800000, subnormal, normal)
    code = tl.where((bits & 0x7FFFFFFF) > 0x7F800000, 127, code)
    return (code | ((bits >> 24) & 128)).to(tl.uint8)


@tr.jit
def _load_value(X, offsets, mask, F: tl.constexpr):
    raw = tl.load(X + offsets, mask, 0)
    if F == 0:
        return raw.to(tl.float32)
    elif F == 6:
        bits = raw.to(tl.int32)
        code = tl.where(bits == 0, 0x00400000, bits << 23)
        code = tl.where(bits == 255, 0x7FC00000, code)
        return code.to(tl.float32, bitcast=True)
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
        return code.to(tl.float32, bitcast=True)
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
        normal = (((exponent + 127 - BIAS) << 23) | (mantissa << (23 - M))).to(
            tl.float32, bitcast=True
        )
        value = tl.where(
            exponent == 0,
            mantissa.to(tl.float32) * (2.0 ** (1 - BIAS - M)),
            normal,
        )
        value = tl.where((bits & 128) != 0, -value, value)
        if F == 1:
            value = tl.where(mag == 127, float("nan"), value)
        elif F == 2:
            value = tl.where(
                exponent == 31,
                tl.where(
                    mantissa == 0,
                    tl.where(bits & 128, -float("inf"), float("inf")),
                    float("nan"),
                ),
                value,
            )
        else:
            value = tl.where(bits == 128, float("nan"), value)
        return value


@tr.jit
def _output_ptr(Q):
    if (
        Q.dtype == tl.int64
        or Q.dtype == tl.uint64
        or Q.dtype == tl.int32
        or Q.dtype == tl.uint32
    ):
        return tl.multiple_of(Q.to(tl.pointer_type(tl.uint8)), 16)
    else:
        return Q


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
    X, S = _input_scale_ptr(X, S)
    Q = _output_ptr(Q)
    if N >= 2147483648:
        rows = tl.program_id(0).to(tl.int64) * R + tl.arange(0, R)
    else:
        rows = tl.program_id(0) * R + tl.arange(0, R)
    index_rows = rows.to(tl.int64) if N * G >= 2147483648 else rows
    cols = tl.arange(0, B)
    if G <= B:
        offsets = index_rows[:, None] * G + cols[None, :]
        if N * G >= 8388608 and N * G < 16777216:
            x = _load_cached(
                X, offsets, (rows[:, None] < N) & (cols[None, :] < G), F
            )
        else:
            x = _load_value(
                X, offsets, (rows[:, None] < N) & (cols[None, :] < G), F
            )
        a = tl.maximum(tl.max(_magnitude(x), 1), 786163455).to(
            tl.float32, bitcast=True
        )
        s = a / 448.0
        q = x / s[:, None]
        tl.store(
            Q + offsets,
            _encode(q),
            (rows[:, None] < N) & (cols[None, :] < G),
            cache_modifier=(".wt" if N * G >= 8388608 else ""),
        )
    else:
        a = tl.full((R, B), 0, tl.int32)
        for start in range(tr.cdiv(G, B)):
            k = start * B + cols
            offsets = index_rows[:, None] * G + k[None, :]
            x = _load_value(
                X, offsets, (rows[:, None] < N) & (k[None, :] < G), F
            )
            a = tl.maximum(a, _magnitude(x))
        s = (
            tl.maximum(tl.max(a, 1), 786163455).to(tl.float32, bitcast=True)
            / 448.0
        )
        for start in range(tr.cdiv(G, B)):
            k = start * B + cols
            offsets = index_rows[:, None] * G + k[None, :]
            x = _load_value(
                X, offsets, (rows[:, None] < N) & (k[None, :] < G), F
            )
            q = x / s[:, None]
            tl.store(
                Q + offsets,
                _encode(q),
                (rows[:, None] < N) & (k[None, :] < G),
                cache_modifier=(".wt" if N * G >= 8388608 else ""),
            )
    tl.store(S + rows, s, rows < N)


@tr.jit
def _fast(X, Q, S, G: tl.constexpr, N: tl.constexpr):
    X, S = _input_scale_ptr(X, S)
    B: tl.constexpr = tr.next_power_of_2(G) if G < 4096 else 4096
    R: tl.constexpr = (
        (8 if B <= 128 else 1024 // B if B <= 1024 else 1)
        if N >= 8192
        or (G == 128 and N >= 512 and X.dtype.element_ty == tl.bfloat16)
        else (4 if B <= 256 else 1024 // B if B <= 1024 else 1)
    )
    _quant(X, Q, S, G, N, R, B, 0)


def _general(x, group_size):
    dtype = x.dtype
    kind = 0
    if (
        dtype == torch.bfloat16
        or dtype == torch.float16
        or dtype == torch.float32
    ):
        kind = 0
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
    q = x.new_empty(shape, dtype=torch.float8_e4m3fn)
    address = q.data_ptr()
    out = address if address % 16 == 0 else q.view(torch.uint8)
    s = x.new_empty(
        shape[:-1] + (shape[-1] // group_size,), dtype=torch.float32
    )
    scale_address = s.data_ptr()
    input_address = x.data_ptr() if dtype == torch.bfloat16 else 0
    inp = input_address if input_address and input_address % 16 == 0 else x
    groups = x.numel() // group_size
    if groups:
        rows = (
            (
                8
                if group_size <= 128
                else 4 if group_size <= 256 else 2 if group_size <= 512 else 1
            )
            if groups >= 8192
            or (
                group_size == 128 and groups >= 512 and dtype == torch.bfloat16
            )
            else (4 if group_size <= 256 else 2 if group_size <= 512 else 1)
        )
        if kind == 0:
            _fast.run(
                inp,
                out,
                scale_address,
                group_size,
                groups,
                grid=((groups + rows - 1) // rows,),
                warmup=False,
                enable_fp_fusion=False,
            )
        else:
            block = min(tr.next_power_of_2(group_size), 4096)
            rows = (
                min(8, max(1, 1024 // block))
                if groups >= 8192
                else min(4, max(1, 1024 // block))
            )
            _quant.run(
                (
                    x.view(torch.int64)
                    if kind == 5
                    else x.view(torch.uint8) if kind else x
                ),
                out,
                scale_address,
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
    return q, s


@tr.jit
def _g128(X, Q, S, N: tl.constexpr):
    _fast(X, Q, S, 128, N)


def per_token_group_quant_fp8(x, group_size):
    if group_size != 128 or x.dtype != torch.bfloat16:
        return _general(x, group_size)
    shape = x.shape
    elements = x.numel()
    groups = elements // 128
    q = x.new_empty(shape, dtype=torch.float8_e4m3fn)
    s = x.new_empty(shape[:-1] + (shape[-1] // 128,), dtype=torch.float32)
    address = q.data_ptr()
    out = address if address % 16 == 0 else q.view(torch.uint8)
    input_address = x.data_ptr()
    inp = input_address if input_address % 16 == 0 else x
    scale_address = s.data_ptr()
    if (
        8388608 <= elements < 2147483648
        and groups % 16 == 0
        and address % 4 == 0
    ):
        if elements < 33554432:
            _packed_pair4.run(
                inp,
                out,
                scale_address,
                grid=(groups // 4,),
                num_warps=1,
                warmup=False,
                enable_fp_fusion=False,
            )
        else:
            _packed_narrow.run(
                inp,
                out,
                scale_address,
                grid=(groups // 8,),
                num_warps=1,
                warmup=False,
                enable_fp_fusion=False,
            )
    elif groups:
        rows = (16 if groups >= 8192 else 8) if groups >= 512 else 4
        if (
            elements < 8388608
            and groups % rows == 0
            and input_address % 16 == 0
            and address % 16 == 0
        ):
            if rows == 16:
                _complete_paired8.run(
                    inp,
                    out,
                    scale_address,
                    grid=(groups // 8,),
                    num_warps=1,
                    warmup=False,
                    enable_fp_fusion=False,
                )
            elif rows == 4:
                _complete0.run(
                    inp,
                    out,
                    scale_address,
                    grid=(groups // rows,),
                    warmup=False,
                    enable_fp_fusion=False,
                )
            else:
                _complete1.run(
                    inp,
                    out,
                    scale_address,
                    grid=(groups // rows,),
                    warmup=False,
                    enable_fp_fusion=False,
                )
        else:
            if (
                8388608 <= elements < 2147483648
                and groups % 16 == 0
                and address % 4 == 0
            ):
                if elements < 2147483648:
                    if elements < 33554432:
                        _packed_pair4.run(
                            inp,
                            out,
                            scale_address,
                            grid=(groups // 4,),
                            num_warps=1,
                            warmup=False,
                            enable_fp_fusion=False,
                        )
                    else:
                        _packed_narrow.run(
                            inp,
                            out,
                            scale_address,
                            grid=(groups // 8,),
                            num_warps=1,
                            warmup=False,
                            enable_fp_fusion=False,
                        )
                else:
                    _packed_store.run(
                        inp,
                        out,
                        scale_address,
                        groups,
                        grid=(groups // 16,),
                        num_warps=1,
                        warmup=False,
                        enable_fp_fusion=False,
                    )
            else:
                _g128.run(
                    inp,
                    out,
                    scale_address,
                    groups,
                    grid=(
                        (groups + (7 if groups >= 512 else 3))
                        // (8 if groups >= 512 else 4),
                    ),
                    warmup=False,
                    enable_fp_fusion=False,
                )
    return q, s


@tr.jit
def _load_cached(X, offsets, mask, F: tl.constexpr):
    raw = tl.load(X + offsets, mask, 0, cache_modifier=".cg")
    if F == 0:
        return raw.to(tl.float32)
    elif F == 6:
        bits = raw.to(tl.int32)
        code = tl.where(bits == 0, 0x00400000, bits << 23)
        code = tl.where(bits == 255, 0x7FC00000, code)
        return code.to(tl.float32, bitcast=True)
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
        return code.to(tl.float32, bitcast=True)
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
        normal = (((exponent + 127 - BIAS) << 23) | (mantissa << (23 - M))).to(
            tl.float32, bitcast=True
        )
        value = tl.where(
            exponent == 0,
            mantissa.to(tl.float32) * (2.0 ** (1 - BIAS - M)),
            normal,
        )
        value = tl.where((bits & 128) != 0, -value, value)
        if F == 1:
            value = tl.where(mag == 127, float("nan"), value)
        elif F == 2:
            value = tl.where(
                exponent == 31,
                tl.where(
                    mantissa == 0,
                    tl.where(bits & 128, -float("inf"), float("inf")),
                    float("nan"),
                ),
                value,
            )
        else:
            value = tl.where(bits == 128, float("nan"), value)
        return value


@tr.jit
def _complete(
    X, Q, S, R: tl.constexpr, STREAM: tl.constexpr, FOLD: tl.constexpr
):
    if X.dtype == tl.int64 or X.dtype == tl.uint64:
        X = tl.multiple_of(X.to(tl.pointer_type(tl.bfloat16)), 16)
    if S.dtype == tl.int64 or S.dtype == tl.uint64:
        S = S.to(tl.pointer_type(tl.float32))
    Q = _output_ptr(Q)
    rows = tl.program_id(0) * R + tl.arange(0, R)
    offsets = rows[:, None].to(tl.int64) * 128 + tl.arange(0, 128)[None, :]
    x = tl.load(X + offsets).to(tl.float32)
    a = tl.maximum(tl.max(_magnitude(x), 1), 786163455).to(
        tl.float32, bitcast=True
    )
    s = a / 448.0
    q = x / s[:, None]
    tl.store(Q + offsets, _encode(q), cache_modifier=".wt" if STREAM else "")
    tl.store(S + rows, s)


@tr.jit
def _complete0(X, Q, S):
    _complete(X, Q, S, 4, False, False)


@tr.jit
def _complete1(X, Q, S):
    _complete(X, Q, S, 8, False, False)


@tr.jit
def _packed_store(X, Q, S, N: tl.constexpr):
    X, S = _input_scale_ptr(X, S)
    Q = _output_ptr(Q)
    rows = tl.program_id(0) * 16 + tl.arange(0, 16)
    cols = tl.arange(0, 128)
    offsets = rows[:, None] * 128 + cols[None, :]
    x = tl.load(
        X + offsets, cache_modifier=".cg" if N * 128 < 16777216 else ""
    ).to(tl.float32)
    a = tl.maximum(tl.max(_magnitude(x), 1), 786163455).to(
        tl.float32, bitcast=True
    )
    s = a / 448.0
    r = tl.div_rn(1.0, s)
    low = tl.fma(-r, s, 1.0) * r
    low = tl.where(s == float("inf"), 0.0, low)
    q = tl.fma(x, r[:, None], x * low[:, None])
    code = (
        _encode(q).to(tl.uint32)
        | ((x.to(tl.uint32, bitcast=True) >> 24) & 128)
    ).reshape(16, 64, 2)
    shifts = tl.arange(0, 2) * 8
    packed = tl.sum(code << shifts[None, None, :], 2).to(tl.uint32)
    target = rows[:, None] * 64 + tl.arange(0, 64)[None, :]
    tl.store(
        Q.to(tl.pointer_type(tl.uint16)) + target, packed, cache_modifier=""
    )
    tl.store(S + rows, s)


@tr.jit
def _packed_narrow(X, Q, S):
    X, S = _input_scale_ptr(X, S)
    Q = _output_ptr(Q)
    rows = tl.program_id(0) * 8 + tl.arange(0, 8)
    cols = tl.arange(0, 128)
    offsets = rows[:, None] * 128 + cols[None, :]
    x = tl.load(X + offsets).to(tl.float32)
    x0, x1 = tl.split(x.reshape(8, 64, 2))
    m = tl.maximum(_magnitude(x0), _magnitude(x1))
    a = tl.maximum(tl.max(m, 1), 786163455).to(tl.float32, bitcast=True)
    s = tl.fma(a, 0.0022321343421936035156, a * 8.5149496342751263001e-09)
    c0 = _encode_pair_bounded(x0 / s[:, None]).to(tl.uint16)
    c1 = _encode_pair_bounded(x1 / s[:, None]).to(tl.uint16)
    target = rows[:, None] * 64 + tl.arange(0, 64)[None, :]
    tl.store(Q.to(tl.pointer_type(tl.uint16)) + target, c0 | (c1 << 8))
    tl.store(S + rows, s)


@tr.jit
def _complete_paired8(X, Q, S):
    _complete_direct(X, Q, S)


@tr.jit
def _complete_direct(X, Q, S):
    X, S = _input_scale_ptr(X, S)
    Q = _output_ptr(Q)
    rows = tl.program_id(0) * 8 + tl.arange(0, 8)
    offsets = rows[:, None] * 64 + tl.arange(0, 64)[None, :]
    native_offsets = rows[:, None] * 128 + tl.arange(0, 128)[None, :]
    x = tl.load(X + native_offsets).to(tl.float32)
    x0, x1 = tl.split(x.reshape(8, 64, 2))
    m = tl.maximum(_magnitude(x0), _magnitude(x1))
    a = tl.maximum(tl.max(m, 1), 786163455).to(tl.float32, bitcast=True)
    s = a / 448.0
    c0 = _encode_pair_bounded(x0 / s[:, None]).to(tl.uint16)
    c1 = _encode_pair_bounded(x1 / s[:, None]).to(tl.uint16)
    tl.store(Q.to(tl.pointer_type(tl.uint16)) + offsets, c0 | (c1 << 8))
    tl.store(S + rows, s)


@tr.jit
def _encode_pair_bounded(v):
    bits = v.to(tl.uint32, bitcast=True)
    mag = bits & 0x7FFFFFFF
    normal = (mag + 0x0407FFFF + ((mag >> 20) & 1)) >> 20
    subnormal = (tl.abs(v) + 16384.0).to(tl.uint32, bitcast=True) - 0x46800000
    code = tl.where(mag < 0x3C800000, subnormal, normal)
    code = tl.where((bits & 0x7FFFFFFF) > 0x7F800000, 127, code)
    return (code | ((bits >> 24) & 128)).to(tl.uint8)


@tr.jit
def _packed_pair4(X, Q, S):
    X, S = _input_scale_ptr(X, S)
    Q = _output_ptr(Q)
    rows = tl.program_id(0) * 4 + tl.arange(0, 4)
    offsets = (
        rows[:, None, None] * 128
        + tl.arange(0, 2)[None, :, None]
        + tl.arange(0, 64)[None, None, :] * 2
    )
    x = tl.load(X + offsets).to(tl.float32)
    x0, x1 = tl.split(x.trans(0, 2, 1))
    m = tl.maximum(_magnitude(x0), _magnitude(x1))
    a = tl.maximum(tl.max(m, 1), 786163455).to(tl.float32, bitcast=True)
    s = tl.fma(a, 0.0022321343421936035156, a * 8.5149496342751263001e-09)
    c0 = _encode_pair_bounded(x0 / s[:, None]).to(tl.uint8)
    c1 = _encode_pair_bounded(x1 / s[:, None]).to(tl.uint8)
    encoded = tl.join(c0, c1).trans(0, 2, 1)
    tl.store(Q.to(tl.pointer_type(tl.uint8)) + offsets, encoded)
    tl.store(S + rows, s)


__all__ = ["per_token_group_quant_fp8"]
