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
from triton.language.extra import libdevice


@tr.jit
def _input_scale_ptr(X, S):
    if (
        S.dtype == tl.int64
        or S.dtype == tl.uint64
        or S.dtype == tl.int32
        or S.dtype == tl.uint32
    ):
        S = tl.multiple_of(S.to(tl.pointer_type(tl.float32)), 16)
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
    return tl.abs(x).to(tl.int32, bitcast=True)


@tr.jit
def _encode(v, original):
    bits = v.to(tl.uint32, bitcast=True)
    mag = bits
    normal = (mag + 0x0407FFFF + ((mag >> 20) & 1)) >> 20
    subnormal = (v + 16384.0).to(tl.uint32, bitcast=True) - 0x46800000
    code = tl.where(mag < 0x3C800000, subnormal, normal)
    code = tl.where(bits > 0x7F800000, 127, code)
    return (code | ((original.to(tl.uint32, bitcast=True) >> 24) & 128)).to(
        tl.uint8
    )


@tr.jit
def _load_value(X, offsets, mask, F: tl.constexpr):
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
def _coeff_1d(s):
    bits = s.to(tl.int32, bitcast=True)
    factor = ((254 - (bits >> 23)) << 23).to(tl.float32, bitcast=True)
    factor = tl.where(s == float("inf"), 0.0, tl.where(s != s, s, factor))
    sn = ((bits & 0x7FFFFF) | 0x3F800000).to(tl.float32, bitcast=True)
    r = 1.0 / sn
    sh = (sn.to(tl.int32, bitcast=True) & -4096).to(tl.float32, bitcast=True)
    sl = sn - sh
    high = (r.to(tl.int32, bitcast=True) & -4096).to(tl.float32, bitcast=True)
    error = (1.0 - high * sh) - high * sl
    low = error * r
    high = high * factor
    low = low * factor
    return high, low


@tr.jit
def _divide_short(x, s, BF: tl.constexpr, FINITE: tl.constexpr):
    if BF:
        high, low = _coeff_1d(s)
        ax = tl.abs(x)
        q = ax * high + ax * low
        return q
    return x / s


@tr.jit
def _divide(x, s, BF: tl.constexpr, FINITE: tl.constexpr):
    if BF:
        r = libdevice.fast_dividef(1.0, s)
        q = x * r
        if FINITE:
            error = tl.fma(-q, tl.minimum(s, 3.4028234663852886e38), x)
        else:
            error = tl.fma(-q, s, x)
        corrected = tl.fma(error, r, q)
        if FINITE:
            return corrected
        return tl.where(s == float("inf"), q, corrected)
    return x / s


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
    cols = tl.arange(0, B)
    if G <= B:
        offsets = rows[:, None].to(tl.int64) * G + cols[None, :]
        x = _load_value(
            X, offsets, (rows[:, None] < N) & (cols[None, :] < G), F
        )
        a = tl.maximum(tl.max(_magnitude(x), 1), 786163455).to(
            tl.float32, bitcast=True
        )
        s = tl.fma(a, 0.002232142724096775, a * 1.3304608803554885e-10)
        if (
            X.dtype.element_ty == tl.bfloat16
            and N * G >= 16777216
            and N * G < 33554432
        ):
            q = _divide_short(tl.abs(x), s[:, None], True, False)
        else:
            q = _divide(
                tl.abs(x),
                s[:, None],
                X.dtype.element_ty == tl.bfloat16,
                N * G >= 33554432,
            )
        tl.store(
            Q + offsets,
            _encode(q, x),
            (rows[:, None] < N) & (cols[None, :] < G),
            cache_modifier=".cs",
        )
    else:
        a = tl.full((R, B), 0, tl.int32)
        for start in range(tr.cdiv(G, B)):
            k = start * B + cols
            offsets = rows[:, None].to(tl.int64) * G + k[None, :]
            x = _load_value(
                X, offsets, (rows[:, None] < N) & (k[None, :] < G), F
            )
            a = tl.maximum(a, _magnitude(x))
        a = tl.maximum(tl.max(a, 1), 786163455).to(tl.float32, bitcast=True)
        s = tl.fma(a, 0.002232142724096775, a * 1.3304608803554885e-10)
        for start in range(tr.cdiv(G, B)):
            k = start * B + cols
            offsets = rows[:, None].to(tl.int64) * G + k[None, :]
            x = _load_value(
                X, offsets, (rows[:, None] < N) & (k[None, :] < G), F
            )
            if (
                X.dtype.element_ty == tl.bfloat16
                and N * G >= 16777216
                and N * G < 33554432
            ):
                q = _divide_short(tl.abs(x), s[:, None], True, False)
            else:
                q = _divide(
                    tl.abs(x),
                    s[:, None],
                    X.dtype.element_ty == tl.bfloat16,
                    N * G >= 33554432,
                )
            tl.store(
                Q + offsets,
                _encode(q, x),
                (rows[:, None] < N) & (k[None, :] < G),
                cache_modifier=".cs",
            )
    tl.store(S + rows, s, rows < N)


@tr.jit
def _fast(X, Q, S, G: tl.constexpr, N: tl.constexpr):
    X, S = _input_scale_ptr(X, S)
    B: tl.constexpr = tr.next_power_of_2(G) if G < 4096 else 4096
    R: tl.constexpr = (
        64
        if X.dtype.element_ty == tl.bfloat16 and N >= 131072 and G <= 128
        else (
            (8 if B <= 128 else 1024 // B if B <= 1024 else 1)
            if N >= 8192
            else (4 if B <= 256 else 1024 // B if B <= 1024 else 1)
        )
    )
    _quant(X, Q, S, G, N, R, B, 0)


@tr.jit
def _raw_encoding(v, sign):
    bits = v.to(tl.uint32, bitcast=True)
    exponent = tl.maximum(bits >> 23, 121)
    bias = exponent * 8388616 + 167771192
    code = (v + bias.to(tl.float32, bitcast=True)).to(tl.uint32, bitcast=True)
    code = tl.where(bits > 0x7F800000, 127, code)
    return (code | sign).to(tl.uint8)


@tr.jit
def _raw_quant(X, Q, S, G: tl.constexpr, N: tl.constexpr, R: tl.constexpr):
    X, S = _input_scale_ptr(X, S)
    Q = _output_ptr(Q)
    rows = tl.program_id(0) * R + tl.arange(0, R)
    cols = tl.arange(0, G)
    offsets = rows[:, None].to(tl.int64) * G + cols[None, :]
    raw = tl.load(X + offsets, cache_modifier=".cg").to(
        tl.uint16, bitcast=True
    )
    sign = (raw >> 8) & 128
    ax = (raw & 32767).to(tl.bfloat16, bitcast=True).to(tl.float32)
    a = tl.maximum(tl.max(ax.to(tl.int32, bitcast=True), 1), 786163455).to(
        tl.float32, bitcast=True
    )
    s = tl.fma(a, 0.002232142724096775, a * 1.3304608803554885e-10)
    if (N * G >= 33554432) & (
        tl.max(a.to(tl.int32, bitcast=True), 0) < 2139095040
    ):
        high = libdevice.fast_dividef(1.0, s)
        low = tl.fma(-high, s, 1.0) * high
        low = tl.where(s == float("inf"), 0.0, low)
        q = tl.fma(ax, high[:, None], ax * low[:, None])
        tl.store(Q + offsets, _finite_encoding(q, sign), cache_modifier=".cs")
    else:
        high = libdevice.fast_dividef(1.0, s)
        low = tl.fma(-high, s, 1.0) * high
        low = tl.where(s == float("inf"), 0.0, low)
        q = tl.fma(ax, high[:, None], ax * low[:, None])
        tl.store(Q + offsets, _raw_encoding(q, sign), cache_modifier=".cs")
    tl.store(S + rows, s)


def per_token_group_quant_fp8(x, group_size):
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
    elements = x.numel()
    scale_address = s.data_ptr()
    scale_address = (
        scale_address if elements < 33554432 and scale_address % 16 == 0 else s
    )
    input_address = (
        x.data_ptr() if dtype == torch.bfloat16 and elements < 33554432 else 0
    )
    inp = input_address if input_address and input_address % 16 == 0 else x
    groups = elements // group_size
    if groups:
        rows = (
            (
                8
                if group_size <= 128
                else 4 if group_size <= 256 else 2 if group_size <= 512 else 1
            )
            if groups >= 8192
            else (4 if group_size <= 256 else 2 if group_size <= 512 else 1)
        )
        warps = 1 if groups >= 8192 else 4
        if dtype == torch.bfloat16 and groups >= 131072 and group_size <= 128:
            rows = 64
            warps = 8
        if (
            dtype == torch.bfloat16
            and group_size == 128
            and groups >= 131072
            and groups % 64 == 0
            and x.data_ptr() % 16 == 0
        ):
            warps = 4 if elements < 33554432 else 8
            _raw_quant.run(
                inp,
                out,
                scale_address,
                group_size,
                groups,
                64,
                grid=(groups // 64,),
                num_warps=warps,
                warmup=False,
                enable_fp_fusion=False,
            )
        elif kind == 0:
            _fast.run(
                inp,
                out,
                scale_address,
                group_size,
                groups,
                grid=((groups + rows - 1) // rows,),
                warmup=False,
                num_warps=warps,
                enable_fp_fusion=not (
                    dtype == torch.bfloat16 and 16777216 <= elements < 33554432
                ),
            )
        else:
            block = min(tr.next_power_of_2(group_size), 4096)
            rows = (
                min(8, max(1, 1024 // block))
                if groups >= 8192
                else min(4, max(1, 1024 // block))
            )
            warps = 1 if groups >= 8192 else 4
            if (
                dtype == torch.bfloat16
                and groups >= 131072
                and group_size <= 128
            ):
                rows = 64
                warps = 8
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
                num_warps=warps,
                grid=(tr.cdiv(groups, rows),),
                warmup=False,
            )
    return q, s


@tr.jit
def _finite_encoding(v, sign):
    bits = v.to(tl.uint32, bitcast=True)
    exponent = tl.maximum(bits >> 23, 121)
    bias = exponent * 8388616 + 167771192
    code = (v + bias.to(tl.float32, bitcast=True)).to(tl.uint32, bitcast=True)
    return (code | sign).to(tl.uint8)


__all__ = ["per_token_group_quant_fp8"]
