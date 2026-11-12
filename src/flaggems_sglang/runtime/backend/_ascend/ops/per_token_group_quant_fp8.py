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
def _max_nan(a, b):
    return tl.maximum(a, b, propagate_nan=tl.PropagateNan.ALL)


@tr.jit
def _reduce_nan(x, axis: tl.constexpr):
    return tl.reduce(x, axis, _max_nan)


@tr.jit
def _min_nan(a, b):
    return tl.minimum(a, b, propagate_nan=tl.PropagateNan.ALL)


@tr.jit
def _encode(v):
    bits = v.to(tl.int32, bitcast=True)
    mag = bits & 0x7FFFFFFF
    normal = (mag + 0x0407FFFF + ((mag >> 20) & 1)) >> 20
    subnormal = (tl.abs(v) + 16384.0).to(tl.int32, bitcast=True) - 0x46800000
    code = tl.where(mag < 0x3C800000, subnormal, normal)
    code = tl.where(v != v, 127, code)
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
        shift = _min_nan(_max_nan(926 - exponent, 29), 53)
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
def _boundary_code(x, s):
    ax = tl.abs(x)
    v = ax / s
    mag = v.to(tl.int32, bitcast=True) & 0x7FFFFFFF
    lower = (mag >> 20) - 960
    midpoint = ((mag & 0x7FF00000) | 0x80000).to(tl.float32, bitcast=True)
    sub = (v * 512.0).to(tl.int32)
    midpoint = tl.where(
        mag < 0x3C800000, (sub.to(tl.float32) + 0.5) * 0.001953125, midpoint
    )
    lower = tl.where(mag < 0x3C800000, sub, lower)
    exponent = (midpoint.to(tl.int32, bitcast=True) >> 23) & 255
    half_ulp = ((exponent - 24) << 23).to(tl.float32, bitcast=True)
    high = (s.to(tl.int32, bitcast=True) & -256).to(tl.float32, bitcast=True)
    low = s - high
    residual = (ax - midpoint * high) - midpoint * low
    threshold = s * half_ulp
    up = tl.where(
        (lower & 1) != 0, residual >= -threshold, residual > threshold
    )
    code = tl.minimum(lower + up.to(tl.int32), 126)
    code = tl.where(s == float("inf"), 0, code)
    code = tl.where((s != s) | (ax == float("inf")), 127, code)
    sign = (x.to(tl.int32, bitcast=True) >> 24) & 128
    return (code | sign).to(tl.uint8)


@tr.jit
def _group_code(x, s, BF: tl.constexpr):
    if not BF:
        return _boundary_code(x, s)
    bits = s.to(tl.int32, bitcast=True)
    factor = ((254 - (bits >> 23)) << 23).to(tl.float32, bitcast=True)
    factor = tl.where(s == float("inf"), 0.0, tl.where(s != s, s, factor))
    sn = ((bits & 0x7FFFFF) | 0x3F800000).to(tl.float32, bitcast=True)
    r = 1.0 / sn
    sh = (sn.to(tl.int32, bitcast=True) & -4096).to(tl.float32, bitcast=True)
    sl = sn - sh
    rh = (r.to(tl.int32, bitcast=True) & -4096).to(tl.float32, bitcast=True)
    rl = r - rh
    error = ((1.0 - rh * sh) - rh * sl) - rl * sn
    high = (r.to(tl.int32, bitcast=True) & -256).to(tl.float32, bitcast=True)
    low = (r - high) + error * r
    ax = tl.abs(x)
    high = high * factor
    low = low * factor
    q = ax * high + ax * low
    mag = q.to(tl.int32, bitcast=True)
    exponent = tl.maximum(mag >> 23, 121)
    bias_bits = exponent * 8388616 + 167771192
    rounded = (q + bias_bits.to(tl.float32, bitcast=True)).to(
        tl.int32, bitcast=True
    )
    code = rounded.to(tl.uint16)
    code = tl.where(q != q, tl.full((), 127, tl.uint16), code)
    sign = ((x.to(tl.int32, bitcast=True) >> 24) & 128).to(tl.uint16)
    return (code | sign).to(tl.uint8)


@tr.jit
def _tile(
    X,
    Q,
    S,
    START,
    G: tl.constexpr,
    N: tl.constexpr,
    R: tl.constexpr,
    B: tl.constexpr,
    F: tl.constexpr,
):
    X, S = _input_scale_ptr(X, S)
    if N >= 2147483648:
        rows = START.to(tl.int64) * R + tl.arange(0, R)
    else:
        rows = START * R + tl.arange(0, R)
    if N * G >= 2147483648:
        rows = rows.to(tl.int64)
    cols = tl.arange(0, B)
    if G <= B:
        offsets = rows[:, None] * G + cols[None, :]
        x = _load_value(
            X,
            offsets,
            (
                True
                if G == B and N % R == 0
                else (
                    (offsets < N * G)
                    if G == B
                    else ((rows[:, None] < N) & (cols[None, :] < G))
                )
            ),
            F,
        )
        a = _max_nan(_reduce_nan(tl.abs(x), 1), 1.0e-10)
        s = a / 448.0
        tl.store(
            Q + offsets,
            _group_code(x, s[:, None], X.dtype.element_ty == tl.bfloat16),
            (
                True
                if G == B and N % R == 0
                else (
                    (offsets < N * G)
                    if G == B
                    else ((rows[:, None] < N) & (cols[None, :] < G))
                )
            ),
        )
    else:
        a = tl.full((R, B), 0, tl.float32)
        for start in range(tr.cdiv(G, B)):
            k = start * B + cols
            offsets = rows[:, None] * G + k[None, :]
            x = _load_value(
                X, offsets, (rows[:, None] < N) & (k[None, :] < G), F
            )
            a = _max_nan(a, tl.abs(x))
        s = _max_nan(_reduce_nan(a, 1), 1.0e-10) / 448.0
        for start in range(tr.cdiv(G, B)):
            k = start * B + cols
            offsets = rows[:, None] * G + k[None, :]
            x = _load_value(
                X, offsets, (rows[:, None] < N) & (k[None, :] < G), F
            )
            tl.store(
                Q + offsets,
                _group_code(x, s[:, None], X.dtype.element_ty == tl.bfloat16),
                (rows[:, None] < N) & (k[None, :] < G),
            )
    tl.store(S + rows, s, rows < N)


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
    for tile in range(tl.program_id(0), tr.cdiv(N, R), tl.num_programs(0)):
        _tile(X, Q, S, tile, G, N, R, B, F)


@tr.jit
def _fast(X, Q, S, G: tl.constexpr, N: tl.constexpr):
    X, S = _input_scale_ptr(X, S)
    B: tl.constexpr = tr.next_power_of_2(G) if G < 128 else 128
    R: tl.constexpr = (
        (32 if X.dtype.element_ty == tl.bfloat16 and N % 32 == 0 else 16)
        if G <= 128
        else 4
    )
    _quant(X, Q, S, G, N, R, B, 0)


@tr.jit
def _coeff_1d(s):
    r = 1.0 / s
    sh = (s.to(tl.int32, bitcast=True) & -4096).to(tl.float32, bitcast=True)
    sl = s - sh
    high = (r.to(tl.int32, bitcast=True) & -4096).to(tl.float32, bitcast=True)
    error = (1.0 - high * sh) - high * sl
    low = error * r
    low = tl.where(s == float("inf"), 0.0, low)
    return high, low


@tr.jit
def _encode_coeff_1d(ax, sign, high, low):
    q = tl.fma(ax, high, ax * low)
    mag = q.to(tl.int32, bitcast=True)
    exponent = tl.maximum(mag >> 23, 121)
    bias_bits = exponent * 8388616 + 167771192
    rounded = (q + bias_bits.to(tl.float32, bitcast=True)).to(
        tl.int32, bitcast=True
    )
    code = rounded.to(tl.uint16)
    code = tl.where(q != q, tl.full((), 127, tl.uint16), code)
    return (code | sign).to(tl.uint8)


@tr.jit
def _split_columns(X, Q, S, G: tl.constexpr, N: tl.constexpr, R: tl.constexpr):
    X, S = _input_scale_ptr(X, S)
    Q = _output_ptr(Q)
    cols = tl.arange(0, G)
    for tile in range(tl.program_id(0), tr.cdiv(N, R), tl.num_programs(0)):
        rows = tile * R + tl.arange(0, R)
        if N * G >= 2147483648:
            rows = tile.to(tl.int64) * R + tl.arange(0, R)
        offsets = rows[:, None] * G + cols[None, :]
        raw = tl.load(X + offsets).to(tl.uint16, bitcast=True)
        sign = (raw >> 8) & 128
        ax = (raw & 32767).to(tl.bfloat16, bitcast=True).to(tl.float32)
        a = _max_nan(_reduce_nan(ax, 1), 1.0e-10)
        s = a / 448.0
        high, low = _coeff_1d(s)
        code = _encode_coeff_1d(ax, sign, high[:, None], low[:, None])
        tl.store(Q + offsets, code)
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
    scale_address = s.data_ptr()
    input_address = (
        x.data_ptr() if dtype == torch.bfloat16 and x.numel() < 8388608 else 0
    )
    inp = input_address if input_address and input_address % 16 == 0 else x
    groups = x.numel() // group_size
    if groups:
        if (
            dtype == torch.bfloat16
            and group_size == 128
            and groups < 65536
            and groups % 16 == 0
        ):
            rows = 32 if groups % 32 == 0 else 16
            _single_short.run(
                inp,
                out,
                scale_address,
                group_size,
                groups,
                rows,
                grid=(groups // rows,),
                warmup=False,
            )
        elif (
            dtype == torch.bfloat16
            and group_size == 128
            and groups >= 65536
            and groups % 64 == 0
        ):
            if 16777216 <= groups * group_size < 33554432:
                _finite_tile.run(
                    inp,
                    out,
                    scale_address,
                    group_size,
                    groups,
                    64,
                    grid=(40,),
                    warmup=False,
                )
            else:
                _split_columns.run(
                    inp,
                    out,
                    scale_address,
                    group_size,
                    groups,
                    64,
                    grid=(40,),
                    warmup=False,
                )
        elif kind == 0:
            rows = (
                (32 if dtype == torch.bfloat16 and groups % 32 == 0 else 16)
                if group_size <= 128
                else 4
            )
            programs = min(40, (groups + rows - 1) // rows)
            _fast.run(
                inp,
                out,
                scale_address,
                group_size,
                groups,
                grid=(programs,),
                warmup=False,
            )
        else:
            block = min(tr.next_power_of_2(group_size), 128)
            rows = (
                32
                if group_size <= 128
                and dtype == torch.bfloat16
                and groups >= 8192
                else 16 if group_size <= 128 and kind == 0 else 4
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
                grid=(min(40, tr.cdiv(groups, rows)),),
                warmup=False,
            )
    return q, s


@tr.jit
def _coeff_finite(s):
    r = 1.0 / s
    sh = (s.to(tl.int32, bitcast=True) & -4096).to(tl.float32, bitcast=True)
    sl = s - sh
    high = (r.to(tl.int32, bitcast=True) & -4096).to(tl.float32, bitcast=True)
    error = (1.0 - high * sh) - high * sl
    low = error * r
    return high, low


@tr.jit
def _encode_finite(ax, sign, high, low):
    q = tl.fma(ax, high, ax * low)
    mag = q.to(tl.int32, bitcast=True)
    exponent = tl.maximum(mag >> 23, 121)
    bias_bits = exponent * 8388616 + 167771192
    rounded = (q + bias_bits.to(tl.float32, bitcast=True)).to(
        tl.int32, bitcast=True
    )
    code = rounded.to(tl.uint16)
    return (code | sign).to(tl.uint8)


@tr.jit
def _finite_tile(X, Q, S, G: tl.constexpr, N: tl.constexpr, R: tl.constexpr):
    X, S = _input_scale_ptr(X, S)
    Q = _output_ptr(Q)
    cols = tl.arange(0, G)
    for tile in range(tl.program_id(0), tr.cdiv(N, R), tl.num_programs(0)):
        rows = tile * R + tl.arange(0, R)
        if N * G >= 2147483648:
            rows = tile.to(tl.int64) * R + tl.arange(0, R)
        offsets = rows[:, None] * G + cols[None, :]
        raw = tl.load(X + offsets).to(tl.uint16, bitcast=True)
        sign = (raw >> 8) & 128
        ax = (raw & 32767).to(tl.bfloat16, bitcast=True).to(tl.float32)
        a = _max_nan(_reduce_nan(ax, 1), 1.0e-10)
        s = a / 448.0
        tile_max = _reduce_nan(a, 0)
        if tile_max < float("inf"):
            high, low = _coeff_finite(s)
            code = _encode_finite(ax, sign, high[:, None], low[:, None])
        else:
            high, low = _coeff_1d(s)
            code = _encode_coeff_1d(ax, sign, high[:, None], low[:, None])
        tl.store(Q + offsets, code)
        tl.store(S + rows, s)


@tr.jit
def _single_short(X, Q, S, G: tl.constexpr, N: tl.constexpr, R: tl.constexpr):
    X, S = _input_scale_ptr(X, S)
    Q = _output_ptr(Q)
    cols = tl.arange(0, G)
    tile = tl.program_id(0)
    rows = tile * R + tl.arange(0, R)
    if N * G >= 2147483648:
        rows = tile.to(tl.int64) * R + tl.arange(0, R)
    offsets = rows[:, None] * G + cols[None, :]
    raw = tl.load(X + offsets).to(tl.uint16, bitcast=True)
    sign = (raw >> 8) & 128
    ax = (raw & 32767).to(tl.bfloat16, bitcast=True).to(tl.float32)
    a = _max_nan(_reduce_nan(ax, 1), 1.0e-10)
    s = a / 448.0
    high, low = _coeff_1d(s)
    code = _encode_coeff_1d(ax, sign, high[:, None], low[:, None])
    tl.store(Q + offsets, code)
    tl.store(S + rows, s)


__all__ = ["per_token_group_quant_fp8"]
