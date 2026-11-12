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

_HIP = tl.constexpr(torch.version.hip is not None)


@tr.jit
def _magnitude(x):
    return tl.abs(x).to(tl.int32, bitcast=True)


@tr.jit
def _encode(v, original, FOLD: tl.constexpr = False):
    bits = v.to(tl.uint32, bitcast=True)
    mag = bits
    exponent = tl.maximum(mag >> 23, 121)
    if FOLD:
        bias = exponent * 8388616 + 167771192
        code = (v + bias.to(tl.float32, bitcast=True)).to(
            tl.uint32, bitcast=True
        )
    else:
        bias = (exponent + 20) << 23
        code = (
            (v + bias.to(tl.float32, bitcast=True)).to(tl.uint32, bitcast=True)
            - bias
            + ((exponent - 121) << 3)
        )
    code = tl.where(bits > 0x7F800000, 127, code)
    return (code | ((original.to(tl.uint32, bitcast=True) >> 24) & 128)).to(
        tl.uint8
    )


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
def _divide(x, s, BF: tl.constexpr):
    if BF:
        if _HIP:
            r = libdevice.fast_dividef(1.0, s)
        else:
            r = tl.div_rn(1.0, s)
        q = x * r
        if _HIP:
            error = tl.fma(-q, tl.minimum(s, 3.4028234663852886e38), x)
        else:
            error = tl.fma(-q, s, x)
        corrected = tl.fma(error, r, q)
        if _HIP:
            return corrected
        return tl.where((s == float("inf")), q, corrected)
    return tl.div_rn(x, s)


@tr.jit
def _scale(a):
    return tl.fma(a, 0.002232142724096775, a * 1.3304608803554885e-10)


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
    Q = _output_ptr(Q)
    if (
        X.dtype == tl.int64
        or X.dtype == tl.uint64
        or X.dtype == tl.int32
        or X.dtype == tl.uint32
    ):
        X = tl.multiple_of(X.to(tl.pointer_type(tl.bfloat16)), 16)
    if (
        S.dtype == tl.int64
        or S.dtype == tl.uint64
        or S.dtype == tl.int32
        or S.dtype == tl.uint32
    ):
        S = S.to(tl.pointer_type(tl.float32))
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
        s = _scale(a)
        q = _divide(tl.abs(x), s[:, None], X.dtype.element_ty == tl.bfloat16)
        tl.store(
            Q + offsets,
            _encode(
                q, x, (16777216 <= N * G) and (N * G < 33554432) and not _HIP
            ),
            (rows[:, None] < N) & (cols[None, :] < G),
            cache_modifier=(".cs" if N * G >= 16777216 else ""),
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
        s = _scale(
            tl.maximum(tl.max(a, 1), 786163455).to(tl.float32, bitcast=True)
        )
        for start in range(tr.cdiv(G, B)):
            k = start * B + cols
            offsets = rows[:, None].to(tl.int64) * G + k[None, :]
            x = _load_value(
                X, offsets, (rows[:, None] < N) & (k[None, :] < G), F
            )
            q = _divide(
                tl.abs(x), s[:, None], X.dtype.element_ty == tl.bfloat16
            )
            tl.store(
                Q + offsets,
                _encode(
                    q,
                    x,
                    (16777216 <= N * G) and (N * G < 33554432) and not _HIP,
                ),
                (rows[:, None] < N) & (k[None, :] < G),
                cache_modifier=(".cs" if N * G >= 16777216 else ""),
            )
    tl.store(S + rows, s, rows < N)


@tr.jit(do_not_specialize=["X", "Q", "S"])
def _fast(X, Q, S, G: tl.constexpr, N: tl.constexpr):
    B: tl.constexpr = tr.next_power_of_2(G) if G < 4096 else 4096
    R: tl.constexpr = (
        (32 if B <= 128 else 4096 // B)
        if N >= 8192
        else (4 if B <= 256 else 1024 // B if B <= 1024 else 1)
    )
    _quant(X, Q, S, G, N, R, B, 0)


@tr.jit
def _packed(
    X, Q, S, G: tl.constexpr, N: tl.constexpr, R: tl.constexpr, P: tl.constexpr
):
    Q = _output_ptr(Q)
    if (
        X.dtype == tl.int64
        or X.dtype == tl.uint64
        or X.dtype == tl.int32
        or X.dtype == tl.uint32
    ):
        X = tl.multiple_of(X.to(tl.pointer_type(tl.bfloat16)), 16)
    if (
        S.dtype == tl.int64
        or S.dtype == tl.uint64
        or S.dtype == tl.int32
        or S.dtype == tl.uint32
    ):
        S = S.to(tl.pointer_type(tl.float32))
    rows = tl.program_id(0) * R + tl.arange(0, R)
    cols = tl.arange(0, G // P)
    offsets = rows[:, None] * (G // P) + cols[None, :]
    if P == 2:
        if G == 128 and N * G >= 16777216:
            base = X.to(tl.pointer_type(tl.uint32)) + tl.program_id(0) * R * (
                G // P
            )
            tile = tl.make_block_ptr(
                base,
                shape=(R, G // P),
                strides=(G // P, 1),
                offsets=(0, 0),
                block_shape=(R, G // P),
                order=(1, 0),
            )
            raw = tl.load(tile)
        else:
            raw = tl.load(X.to(tl.pointer_type(tl.uint32)) + offsets)
    else:
        raw = tl.load(X.to(tl.pointer_type(tl.uint64)) + offsets)
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
    scale = _scale(
        tl.maximum(tl.max(magnitude, 1), 786163455).to(
            tl.float32, bitcast=True
        )
    )
    r = libdevice.fast_dividef(1.0, scale)
    hi = r
    lo = tl.fma(-r, scale, 1.0) * r
    lo = tl.where(scale == float("inf"), 0.0, lo)
    qa = _encode(
        tl.fma(tl.abs(a), hi[:, None], tl.abs(a) * lo[:, None]), a, True
    ).to(tl.int32)
    qb = _encode(
        tl.fma(tl.abs(b), hi[:, None], tl.abs(b) * lo[:, None]), b, True
    ).to(tl.int32)
    word = qa | (qb << 8)
    if P == 4:
        qc = _encode(
            tl.fma(tl.abs(c), hi[:, None], tl.abs(c) * lo[:, None]), c, True
        ).to(tl.int32)
        qd = _encode(
            tl.fma(tl.abs(d), hi[:, None], tl.abs(d) * lo[:, None]), d, True
        ).to(tl.int32)
        word = word | (qc << 16) | (qd << 24)
        ptr = Q.to(tl.pointer_type(tl.int32))
    else:
        ptr = Q.to(tl.pointer_type(tl.int16))
    if G == 128 and N * G >= 16777216:
        out_base = ptr + tl.program_id(0) * R * (G // P)
        out_tile = tl.make_block_ptr(
            out_base,
            shape=(R, G // P),
            strides=(G // P, 1),
            offsets=(0, 0),
            block_shape=(R, G // P),
            order=(1, 0),
        )
        tl.store(
            out_tile,
            word.to(ptr.dtype.element_ty),
            cache_modifier=(".cs" if N * G >= 16777216 else ""),
        )
    else:
        tl.store(
            ptr + offsets,
            word,
            cache_modifier=(".cs" if N * G >= 16777216 else ""),
        )
    tl.store(S + rows, scale)


@tr.jit
def _packed_fast(X, Q, S, G: tl.constexpr, N: tl.constexpr):
    R: tl.constexpr = (
        16
        if N * G < 16777216
        else (64 if N * G < 33554432 and N % 64 == 0 else 32)
    )
    _packed(X, Q, S, G, N, R, 2)


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
    input_address = x.data_ptr() if dtype == torch.bfloat16 else 0
    inp = input_address if input_address and input_address % 16 == 0 else x
    scale_address = s.data_ptr()
    groups = x.numel() // group_size
    if (
        torch.version.hip is not None
        and dtype == torch.bfloat16
        and 8388608 <= x.numel() < 2147483648
        and groups % 32 == 0
        and 4 <= group_size <= 128
        and group_size == tr.next_power_of_2(group_size)
        and x.data_ptr() % 8 == 0
        and address % 4 == 0
    ):
        rows = (
            16
            if x.numel() < 16777216
            else (64 if x.numel() < 33554432 and groups % 64 == 0 else 32)
        )
        _packed_fast.run(
            inp,
            out,
            scale_address,
            group_size,
            groups,
            grid=(groups // rows,),
            warmup=False,
            enable_fp_fusion=False,
        )
    elif groups:
        if groups >= 8192:
            rows = (
                32
                if group_size <= 128
                else (
                    4096 // tr.next_power_of_2(group_size)
                    if group_size < 4096
                    else 1
                )
            )
        else:
            rows = 4 if group_size <= 256 else 2 if group_size <= 512 else 1
        if kind == 0:
            _fast.run(
                inp,
                out,
                scale_address,
                group_size,
                groups,
                grid=((groups + rows - 1) // rows,),
                warmup=False,
            )
        else:
            block = min(tr.next_power_of_2(group_size), 4096)
            _quant.run(
                x.view(torch.int64) if kind == 5 else x.view(torch.uint8),
                out,
                scale_address,
                group_size,
                groups,
                rows,
                block,
                kind,
                grid=(tr.cdiv(groups, rows),),
                warmup=False,
                num_warps=4,
                enable_fp_fusion=False,
            )
    return q, s


@tr.jit(do_not_specialize=["X", "Q", "S"])
def _g128(X, Q, S, N: tl.constexpr):
    _fast(X, Q, S, 128, N)


@tr.jit(do_not_specialize=["X", "Q", "S"])
def _packed128(X, Q, S, N: tl.constexpr):
    if N >= 262144:
        _packed_store789(X, Q, S, 128, N, 32, 2)
    else:
        _packed_fast(X, Q, S, 128, N)


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
    if groups:
        if (
            torch.version.hip is None
            and 1048576 <= elements < 2097152
            and groups % 16 == 0
            and input_address % 16 == 0
            and address % 16 == 0
        ):
            _cuda_scale_early.run(
                inp, out, scale_address, grid=(groups // 16,), warmup=False
            )
        elif (
            torch.version.hip is None
            and (
                1048576 <= elements < 4194304 or 8388608 <= elements < 16777216
            )
            and groups % 16 == 0
            and input_address % 16 == 0
            and address % 16 == 0
        ):
            _cuda_small.run(
                inp, out, scale_address, grid=(groups // 16,), warmup=False
            )
        elif (
            torch.version.hip is None
            and 4194304 <= elements < 8388608
            and groups % 4 == 0
            and input_address % 16 == 0
            and address % 16 == 0
        ):
            _cuda_quad.run(
                inp,
                out,
                scale_address,
                grid=(groups // 4,),
                num_warps=1,
                warmup=False,
            )
        elif (
            torch.version.hip is not None
            and 524288 <= elements < 8388608
            and groups % 4 == 0
            and input_address % 16 == 0
            and address % 16 == 0
        ):
            _small_complete.run(
                inp,
                out,
                scale_address,
                grid=(groups // 4,),
                num_warps=1,
                warmup=False,
            )
        elif (
            torch.version.hip is not None
            and 8388608 <= elements < 2147483648
            and groups % 32 == 0
            and input_address % 8 == 0
            and address % 4 == 0
        ):
            rows = (
                16
                if elements < 16777216
                else (64 if elements < 33554432 and groups % 64 == 0 else 32)
            )
            if 16777216 <= elements < 33554432 and groups % 64 == 0:
                _compact64.run(
                    inp,
                    out,
                    scale_address,
                    grid=(groups // 64,),
                    warmup=False,
                    enable_fp_fusion=False,
                    waves_per_eu=16,
                )
            else:
                _packed128.run(
                    inp,
                    out,
                    scale_address,
                    groups,
                    grid=(groups // rows,),
                    warmup=False,
                    enable_fp_fusion=False,
                    waves_per_eu=(16 if elements >= 33554432 else 0),
                )
        else:
            rows = 32 if groups >= 8192 else 4
            _g128.run(
                inp,
                out,
                scale_address,
                groups,
                grid=((groups + rows - 1) // rows,),
                warmup=False,
            )
    return q, s


@tr.jit
def _packed_complete(
    X, Q, S, R: tl.constexpr, STREAM: tl.constexpr, FOLD: tl.constexpr = True
):
    G: tl.constexpr = 128
    P: tl.constexpr = 2
    Q = _output_ptr(Q)
    if (
        X.dtype == tl.int64
        or X.dtype == tl.uint64
        or X.dtype == tl.int32
        or X.dtype == tl.uint32
    ):
        X = tl.multiple_of(X.to(tl.pointer_type(tl.bfloat16)), 16)
    if (
        S.dtype == tl.int64
        or S.dtype == tl.uint64
        or S.dtype == tl.int32
        or S.dtype == tl.uint32
    ):
        S = S.to(tl.pointer_type(tl.float32))
    rows = tl.program_id(0) * R + tl.arange(0, R)
    cols = tl.arange(0, G // P)
    offsets = rows[:, None] * (G // P) + cols[None, :]
    if P == 2:
        base = X.to(tl.pointer_type(tl.uint32)) + tl.program_id(0) * R * (
            G // P
        )
        tile = tl.make_block_ptr(
            base,
            shape=(R, G // P),
            strides=(G // P, 1),
            offsets=(0, 0),
            block_shape=(R, G // P),
            order=(1, 0),
        )
        raw = tl.load(tile)
    else:
        raw = tl.load(X.to(tl.pointer_type(tl.uint64)) + offsets)
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
    scale = _scale(
        tl.maximum(tl.max(magnitude, 1), 786163455).to(
            tl.float32, bitcast=True
        )
    )
    if _HIP:
        r = libdevice.fast_dividef(1.0, scale)
    else:
        r = tl.div_rn(1.0, scale)
    hi = r
    lo = tl.fma(-r, scale, 1.0) * r
    lo = tl.where(scale == float("inf"), 0.0, lo)
    qa = _encode(
        tl.fma(tl.abs(a), hi[:, None], tl.abs(a) * lo[:, None]), a, FOLD
    ).to(tl.int32)
    qb = _encode(
        tl.fma(tl.abs(b), hi[:, None], tl.abs(b) * lo[:, None]), b, FOLD
    ).to(tl.int32)
    word = qa | (qb << 8)
    if P == 4:
        qc = _encode(
            tl.fma(tl.abs(c), hi[:, None], tl.abs(c) * lo[:, None]), c, FOLD
        ).to(tl.int32)
        qd = _encode(
            tl.fma(tl.abs(d), hi[:, None], tl.abs(d) * lo[:, None]), d, FOLD
        ).to(tl.int32)
        word = word | (qc << 16) | (qd << 24)
        ptr = Q.to(tl.pointer_type(tl.int32))
    else:
        ptr = Q.to(tl.pointer_type(tl.int16))
    out_base = ptr + tl.program_id(0) * R * (G // P)
    out_tile = tl.make_block_ptr(
        out_base,
        shape=(R, G // P),
        strides=(G // P, 1),
        offsets=(0, 0),
        block_shape=(R, G // P),
        order=(1, 0),
    )
    tl.store(
        out_tile,
        word.to(ptr.dtype.element_ty),
        cache_modifier=(".cs" if STREAM else ""),
    )
    tl.store(S + rows, scale)


@tr.jit(do_not_specialize=["X", "Q", "S"])
def _compact64(X, Q, S):
    _packed_complete(X, Q, S, 64, True)


@tr.jit(do_not_specialize=["X", "Q", "S"])
def _small_complete(X, Q, S):
    _packed_complete(X, Q, S, 4, False, True)


@tr.jit
def _packed_scale_schedule(
    X, Q, S, R: tl.constexpr, STREAM: tl.constexpr, FOLD: tl.constexpr = True
):
    G: tl.constexpr = 128
    P: tl.constexpr = 2
    Q = _output_ptr(Q)
    if (
        X.dtype == tl.int64
        or X.dtype == tl.uint64
        or X.dtype == tl.int32
        or X.dtype == tl.uint32
    ):
        X = tl.multiple_of(X.to(tl.pointer_type(tl.bfloat16)), 16)
    if (
        S.dtype == tl.int64
        or S.dtype == tl.uint64
        or S.dtype == tl.int32
        or S.dtype == tl.uint32
    ):
        S = S.to(tl.pointer_type(tl.float32))
    rows = tl.program_id(0) * R + tl.arange(0, R)
    cols = tl.arange(0, G // P)
    offsets = rows[:, None] * (G // P) + cols[None, :]
    if P == 2:
        base = X.to(tl.pointer_type(tl.uint32)) + tl.program_id(0) * R * (
            G // P
        )
        tile = tl.make_block_ptr(
            base,
            shape=(R, G // P),
            strides=(G // P, 1),
            offsets=(0, 0),
            block_shape=(R, G // P),
            order=(1, 0),
        )
        raw = tl.load(tile)
    else:
        raw = tl.load(X.to(tl.pointer_type(tl.uint64)) + offsets)
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
    scale = _scale(
        tl.maximum(tl.max(magnitude, 1), 786163455).to(
            tl.float32, bitcast=True
        )
    )
    a = tl.inline_asm_elementwise(
        asm="{ .reg .pred p; setp.ne.u32 p, $4, 0; mov.b32 $0, $1; @p st.global.f32 [$2], $3; }",
        constraints="=r,r,l,f,r",
        args=[
            a.to(tl.int32, bitcast=True),
            S + rows[:, None],
            scale[:, None],
            (cols[None, :] == 0).to(tl.int32),
        ],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    ).to(tl.float32, bitcast=True)
    if _HIP:
        r = libdevice.fast_dividef(1.0, scale)
    else:
        r = tl.div_rn(1.0, scale)
    hi = r
    lo = tl.fma(-r, scale, 1.0) * r
    lo = tl.where(scale == float("inf"), 0.0, lo)
    qa = _encode(
        tl.fma(tl.abs(a), hi[:, None], tl.abs(a) * lo[:, None]), a, FOLD
    ).to(tl.int32)
    qb = _encode(
        tl.fma(tl.abs(b), hi[:, None], tl.abs(b) * lo[:, None]), b, FOLD
    ).to(tl.int32)
    word = qa | (qb << 8)
    if P == 4:
        qc = _encode(
            tl.fma(tl.abs(c), hi[:, None], tl.abs(c) * lo[:, None]), c, FOLD
        ).to(tl.int32)
        qd = _encode(
            tl.fma(tl.abs(d), hi[:, None], tl.abs(d) * lo[:, None]), d, FOLD
        ).to(tl.int32)
        word = word | (qc << 16) | (qd << 24)
        ptr = Q.to(tl.pointer_type(tl.int32))
    else:
        ptr = Q.to(tl.pointer_type(tl.int16))
    out_base = ptr + tl.program_id(0) * R * (G // P)
    out_tile = tl.make_block_ptr(
        out_base,
        shape=(R, G // P),
        strides=(G // P, 1),
        offsets=(0, 0),
        block_shape=(R, G // P),
        order=(1, 0),
    )
    tl.store(
        out_tile,
        word.to(ptr.dtype.element_ty),
        cache_modifier=(".cs" if STREAM else ""),
    )


@tr.jit(do_not_specialize=["X", "Q", "S"])
def _cuda_small(X, Q, S):
    _packed_complete(X, Q, S, 16, False, True)


@tr.jit
def _fold_p4(
    X, Q, S, R: tl.constexpr, STREAM: tl.constexpr, FOLD: tl.constexpr = False
):
    G: tl.constexpr = 128
    P: tl.constexpr = 4
    Q = _output_ptr(Q)
    if (
        X.dtype == tl.int64
        or X.dtype == tl.uint64
        or X.dtype == tl.int32
        or X.dtype == tl.uint32
    ):
        X = tl.multiple_of(X.to(tl.pointer_type(tl.bfloat16)), 16)
    if (
        S.dtype == tl.int64
        or S.dtype == tl.uint64
        or S.dtype == tl.int32
        or S.dtype == tl.uint32
    ):
        S = S.to(tl.pointer_type(tl.float32))
    rows = tl.program_id(0) * R + tl.arange(0, R)
    base = X.to(tl.pointer_type(tl.uint64)) + tl.program_id(0) * R * (G // P)
    tile = tl.make_block_ptr(
        base,
        shape=(R, G // P),
        strides=(G // P, 1),
        offsets=(0, 0),
        block_shape=(R, G // P),
        order=(1, 0),
    )
    raw = tl.load(tile)
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
    scale = _scale(
        tl.maximum(tl.max(magnitude, 1), 786163455).to(
            tl.float32, bitcast=True
        )
    )
    if _HIP:
        r = libdevice.fast_dividef(1.0, scale)
    else:
        r = tl.div_rn(1.0, scale)
    hi = r
    lo = tl.fma(-r, scale, 1.0) * r
    lo = tl.where(scale == float("inf"), 0.0, lo)
    qa = _encode(
        tl.fma(tl.abs(a), hi[:, None], tl.abs(a) * lo[:, None]), a, FOLD
    ).to(tl.int32)
    qb = _encode(
        tl.fma(tl.abs(b), hi[:, None], tl.abs(b) * lo[:, None]), b, FOLD
    ).to(tl.int32)
    word = qa | (qb << 8)
    if P == 4:
        qc = _encode(
            tl.fma(tl.abs(c), hi[:, None], tl.abs(c) * lo[:, None]), c, FOLD
        ).to(tl.int32)
        qd = _encode(
            tl.fma(tl.abs(d), hi[:, None], tl.abs(d) * lo[:, None]), d, FOLD
        ).to(tl.int32)
        word = word | (qc << 16) | (qd << 24)
        ptr = Q.to(tl.pointer_type(tl.int32))
    else:
        ptr = Q.to(tl.pointer_type(tl.int16))
    out_base = ptr + tl.program_id(0) * R * (G // P)
    out_tile = tl.make_block_ptr(
        out_base,
        shape=(R, G // P),
        strides=(G // P, 1),
        offsets=(0, 0),
        block_shape=(R, G // P),
        order=(1, 0),
    )
    tl.store(
        out_tile,
        word.to(ptr.dtype.element_ty),
        cache_modifier=(".cs" if STREAM else ""),
    )
    tl.store(S + rows, scale)


@tr.jit(do_not_specialize=["X", "Q", "S"])
def _cuda_quad(X, Q, S):
    _fold_p4(X, Q, S, 4, False, True)


@tr.jit(do_not_specialize=["X", "Q", "S"])
def _cuda_scale_early(X, Q, S):
    _packed_scale_schedule(X, Q, S, 16, False, True)


@tr.jit
def _packed_store789(
    X, Q, S, G: tl.constexpr, N: tl.constexpr, R: tl.constexpr, P: tl.constexpr
):
    Q = _output_ptr(Q)
    if (
        X.dtype == tl.int64
        or X.dtype == tl.uint64
        or X.dtype == tl.int32
        or X.dtype == tl.uint32
    ):
        X = tl.multiple_of(X.to(tl.pointer_type(tl.bfloat16)), 16)
    if (
        S.dtype == tl.int64
        or S.dtype == tl.uint64
        or S.dtype == tl.int32
        or S.dtype == tl.uint32
    ):
        S = S.to(tl.pointer_type(tl.float32))
    rows = tl.program_id(0) * R + tl.arange(0, R)
    cols = tl.arange(0, G // P)
    offsets = rows[:, None] * (G // P) + cols[None, :]
    if P == 2:
        if G == 128 and N * G >= 16777216:
            base = X.to(tl.pointer_type(tl.uint32)) + tl.program_id(0) * R * (
                G // P
            )
            tile = tl.make_block_ptr(
                base,
                shape=(R, G // P),
                strides=(G // P, 1),
                offsets=(0, 0),
                block_shape=(R, G // P),
                order=(1, 0),
            )
            raw = tl.load(tile)
        else:
            raw = tl.load(X.to(tl.pointer_type(tl.uint32)) + offsets)
    else:
        raw = tl.load(X.to(tl.pointer_type(tl.uint64)) + offsets)
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
    scale = _scale(
        tl.maximum(tl.max(magnitude, 1), 786163455).to(
            tl.float32, bitcast=True
        )
    )
    tl.store(S + rows, scale)
    r = libdevice.fast_dividef(1.0, scale)
    hi = r
    lo = tl.fma(-r, scale, 1.0) * r
    lo = tl.where(scale == float("inf"), 0.0, lo)
    qa = _encode(
        tl.fma(tl.abs(a), hi[:, None], tl.abs(a) * lo[:, None]), a, True
    ).to(tl.int32)
    qb = _encode(
        tl.fma(tl.abs(b), hi[:, None], tl.abs(b) * lo[:, None]), b, True
    ).to(tl.int32)
    word = qa | (qb << 8)
    if P == 4:
        qc = _encode(
            tl.fma(tl.abs(c), hi[:, None], tl.abs(c) * lo[:, None]), c, True
        ).to(tl.int32)
        qd = _encode(
            tl.fma(tl.abs(d), hi[:, None], tl.abs(d) * lo[:, None]), d, True
        ).to(tl.int32)
        word = word | (qc << 16) | (qd << 24)
        ptr = Q.to(tl.pointer_type(tl.int32))
    else:
        ptr = Q.to(tl.pointer_type(tl.int16))
    if G == 128 and N * G >= 16777216:
        out_base = ptr + tl.program_id(0) * R * (G // P)
        out_tile = tl.make_block_ptr(
            out_base,
            shape=(R, G // P),
            strides=(G // P, 1),
            offsets=(0, 0),
            block_shape=(R, G // P),
            order=(1, 0),
        )
        tl.store(
            out_tile,
            word.to(ptr.dtype.element_ty),
            cache_modifier=(".cs" if N * G >= 16777216 else ""),
        )
    else:
        tl.store(
            ptr + offsets,
            word,
            cache_modifier=(".cs" if N * G >= 16777216 else ""),
        )


__all__ = ["per_token_group_quant_fp8"]
