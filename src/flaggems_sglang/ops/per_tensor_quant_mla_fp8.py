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
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied. See the License for the specific language governing
# permissions and limitations under the License.

import torch
import triton
import triton.language as tl


@triton.jit
def _encode_fp8_S(value):
    bits = value.to(tl.int32, bitcast=True)
    sign = (bits >> 24) & 128
    magnitude = bits & 0x7FFFFFFF
    absolute = magnitude.to(tl.float32, bitcast=True)
    normal = ((magnitude + 0x7FFFF + ((magnitude >> 20) & 1)) >> 20) - 960
    subnormal = (absolute + 16384.0).to(tl.int32, bitcast=True) - 0x46800000
    code = tl.where(magnitude < 0x3C800000, subnormal, normal)
    code = tl.where(
        (magnitude > 0x43E80000) | (magnitude >= 0x7F800000), 127, code
    )
    return (code | sign).to(tl.uint8)


@triton.jit
def _offset_S(
    i,
    S: tl.constexpr,
    D: tl.constexpr,
    X0: tl.constexpr,
    X1: tl.constexpr,
    X2: tl.constexpr,
):
    head = i // (S * D)
    rem = i % (S * D)
    return head * X0 + (rem // D) * X1 + (rem % D) * X2


@triton.jit
def _partial_max_S(
    X,
    P,
    N: tl.constexpr,
    S: tl.constexpr,
    D: tl.constexpr,
    X0: tl.constexpr,
    X1: tl.constexpr,
    X2: tl.constexpr,
    B: tl.constexpr,
):
    program = tl.program_id(0).to(tl.int64)
    lane = tl.arange(0, B)
    maximum = tl.full((B,), 0.0, tl.float32)
    for base in range(program * B, N, tl.num_programs(0) * B):
        i = base + lane
        x = tl.load(X + _offset_S(i, S, D, X0, X1, X2), i < N, other=0).to(
            tl.float32
        )
        maximum = tl.maximum(maximum, tl.abs(x))
    result = tl.max(maximum, 0)
    tl.store(P + program, result)


@triton.jit
def _finish_scale_S(
    P, Scale, PARTS: tl.constexpr, EPS: tl.constexpr, B: tl.constexpr
):
    i = tl.arange(0, B)
    x = tl.load(P + i, i < PARTS, other=0.0)
    maximum = tl.maximum(tl.max(x, 0), EPS)
    tl.store(Scale, maximum / 448.0)


@triton.jit
def _finish_and_quant_S(
    X,
    P,
    Scale,
    Y,
    N: tl.constexpr,
    S: tl.constexpr,
    D: tl.constexpr,
    X0: tl.constexpr,
    X1: tl.constexpr,
    X2: tl.constexpr,
    PARTS: tl.constexpr,
    EPS: tl.constexpr,
    BP: tl.constexpr,
    B: tl.constexpr,
):
    j = tl.arange(0, BP)
    partial = tl.load(P + j, j < PARTS, other=0.0)
    scale = tl.maximum(tl.max(partial, 0), EPS) / 448.0
    program = tl.program_id(0).to(tl.int64)
    if program == 0:
        tl.store(Scale, scale)
    lane = tl.arange(0, B)
    first = program * B
    step = tl.num_programs(0).to(tl.int64) * B
    for base in range(first, N, step):
        i = base + lane
        x = tl.load(X + _offset_S(i, S, D, X0, X1, X2), i < N, other=0).to(
            tl.float32
        )
        value = tl.minimum(tl.maximum(tl.div_rn(x, scale), -448.0), 448.0)
        tl.store(Y + i, _encode_fp8_S(value), i < N)


@triton.jit
def _quantize_S(
    X,
    Scale,
    Y,
    N: tl.constexpr,
    S: tl.constexpr,
    D: tl.constexpr,
    X0: tl.constexpr,
    X1: tl.constexpr,
    X2: tl.constexpr,
    B: tl.constexpr,
):
    program = tl.program_id(0).to(tl.int64)
    lane = tl.arange(0, B)
    scale = tl.load(Scale).to(tl.float32)
    for base in range(program * B, N, tl.num_programs(0) * B):
        i = base + lane
        x = tl.load(X + _offset_S(i, S, D, X0, X1, X2), i < N, other=0).to(
            tl.float32
        )
        value = tl.minimum(tl.maximum(tl.div_rn(x, scale), -448.0), 448.0)
        tl.store(Y + i, _encode_fp8_S(value), i < N)


@triton.jit
def _small_fused_S(
    X,
    Y,
    Scale,
    N: tl.constexpr,
    S: tl.constexpr,
    D: tl.constexpr,
    X0: tl.constexpr,
    X1: tl.constexpr,
    X2: tl.constexpr,
    EPS: tl.constexpr,
    B: tl.constexpr,
):
    i = tl.arange(0, B)
    x = tl.load(X + _offset_S(i, S, D, X0, X1, X2), i < N, other=0.0).to(
        tl.float32
    )
    scale = tl.maximum(tl.max(tl.abs(x), 0), EPS) / 448.0
    value = tl.minimum(tl.maximum(tl.div_rn(x, scale), -448.0), 448.0)
    tl.store(Y + i, _encode_fp8_S(value), i < N)
    tl.store(Scale, scale)


@triton.jit
def _redundant_quant_S(
    X,
    Y,
    Scale,
    N: tl.constexpr,
    S: tl.constexpr,
    D: tl.constexpr,
    X0: tl.constexpr,
    X1: tl.constexpr,
    X2: tl.constexpr,
    EPS: tl.constexpr,
    BR: tl.constexpr,
    BQ: tl.constexpr,
):
    all_i = tl.arange(0, BR)
    values = tl.load(
        X + _offset_S(all_i, S, D, X0, X1, X2), all_i < N, other=0
    ).to(tl.float32)
    scale = tl.maximum(tl.max(tl.abs(values), 0), EPS) / 448.0
    program = tl.program_id(0).to(tl.int64)
    if program == 0:
        tl.store(Scale, scale)
    i = program * BQ + tl.arange(0, BQ)
    x = tl.load(X + _offset_S(i, S, D, X0, X1, X2), i < N, other=0).to(
        tl.float32
    )
    value = tl.minimum(tl.maximum(tl.div_rn(x, scale), -448.0), 448.0)
    tl.store(Y + i, _encode_fp8_S(value), i < N)


def per_tensor_quant_mla_fp8_small(x, eps=1e-12):
    h, s, d = x.shape
    n = x.numel()
    output = torch.empty((h, s, d), dtype=torch.float8_e4m3fn, device=x.device)
    scale = torch.empty((1,), dtype=torch.float32, device=x.device)
    if 0 < n <= 4096:
        _small_fused_S[(1,)](
            x,
            output.view(torch.uint8),
            scale,
            n,
            s,
            d,
            *x.stride(),
            eps,
            triton.next_power_of_2(n),
            num_warps=4,
            num_stages=1,
            enable_fp_fusion=False,
        )
        return output, scale
    if 4096 < n <= 16384 and n % 1024 == 0:
        bq = triton.next_power_of_2(triton.cdiv(n, 4))
        _redundant_quant_S[(4,)](
            x,
            output.view(torch.uint8),
            scale,
            n,
            s,
            d,
            *x.stride(),
            eps,
            triton.next_power_of_2(n),
            bq,
            num_warps=4,
            num_stages=1,
            enable_fp_fusion=False,
        )
        return output, scale
    parts = max(1, min(triton.cdiv(n, 1024), 1024))
    partial = torch.empty((parts,), dtype=torch.float32, device=x.device)
    strides = x.stride()
    _partial_max_S[(parts,)](
        x,
        partial,
        n,
        s,
        d,
        *strides,
        1024,
        num_warps=4,
        num_stages=1,
        enable_fp_fusion=False,
    )
    if n and n <= 1048576:
        _finish_and_quant_S[(min(triton.cdiv(n, 1024), 512),)](
            x,
            partial,
            scale,
            output.view(torch.uint8),
            n,
            s,
            d,
            *strides,
            parts,
            eps,
            triton.next_power_of_2(parts),
            1024,
            num_warps=4,
            num_stages=1,
            enable_fp_fusion=False,
        )
    else:
        _finish_scale_S[(1,)](
            partial,
            scale,
            parts,
            eps,
            triton.next_power_of_2(parts),
            num_warps=4,
            num_stages=1,
            enable_fp_fusion=False,
        )
        if n:
            _quantize_S[(min(triton.cdiv(n, 1024), 65535),)](
                x,
                scale,
                output.view(torch.uint8),
                n,
                s,
                d,
                *strides,
                1024,
                num_warps=4,
                num_stages=1,
                enable_fp_fusion=False,
            )
    return output, scale


@triton.jit
def _encode_dynamic_bf16_L(value, original, scale):
    raw = value.to(tl.uint32, bitcast=True)
    sign = original.to(tl.uint32, bitcast=True) >> 24 & 128
    magnitude = raw & 2147483647
    exponent = tl.maximum(magnitude >> 23, 121)
    magic_bits = exponent * 8388616 + 167771192
    rounded = (tl.abs(value) + magic_bits.to(tl.float32, bitcast=True)).to(
        tl.uint32, bitcast=True
    )
    if scale < float("inf"):
        result = rounded | sign
    else:
        result = tl.where(value != value, 127, rounded | sign)
    return result.to(tl.uint8)


@triton.jit
def _epsilon_floor_L(value, EPS: tl.constexpr):
    if EPS != EPS:
        return value
    return _maximum_nan_L(value, tl.full((), EPS, tl.float32))


@triton.jit
def _maximum_nan_L(a, b):
    return tl.maximum(a, b, propagate_nan=tl.PropagateNan.ALL)


@triton.jit
def _max_nan_L(values):
    return tl.reduce(values, 0, _maximum_nan_L)


@triton.jit
def _clamp_L(v):
    return tl.minimum(
        tl.maximum(v, -448.0, propagate_nan=tl.PropagateNan.ALL),
        448.0,
        propagate_nan=tl.PropagateNan.ALL,
    )


@triton.jit
def _encode_L(v):
    raw = v.to(tl.uint32, bitcast=True)
    sign = raw >> 24 & 128
    mag = raw & 2147483647
    normal = (mag + 524287 + (mag >> 20 & 1) >> 20) - 960
    sub = (tl.abs(v) + 16384.0).to(tl.uint32, bitcast=True) - 1182793728
    code = tl.where(tl.abs(v) >= 0.015625, normal, sub)
    code = tl.where(v != v, 127, code)
    return (code | sign).to(tl.uint8)


@triton.jit
def _encode_bf16_L(v, original):
    raw = v.to(tl.uint32, bitcast=True)
    sign = original.to(tl.uint32, bitcast=True) >> 24 & 128
    mag = raw & 2147483647
    normal = (mag + 524287 + (mag >> 20 & 1) >> 20) - 960
    sub = (tl.abs(v) + 16384.0).to(tl.uint32, bitcast=True) - 1182793728
    code = tl.where(tl.abs(v) >= 0.015625, normal, sub)
    return tl.where(v != v, 127, code | sign).to(tl.uint8)


@triton.jit
def _divide_L(v, scale, BF16: tl.constexpr):
    if BF16:
        inv = tl.div_rn(1.0, scale)
        coarse = v * inv
        residual = tl.fma(
            -coarse, tl.where(scale < float("inf"), scale, 1.0), v
        )
        return tl.fma(residual, inv, coarse)
    return tl.div_rn(v, scale)


@triton.jit
def _linear_index_L(block, B: tl.constexpr, N: tl.constexpr):
    if N >= 2147483648:
        block = block.to(tl.int64)
    return block * B + tl.arange(0, B)


@triton.jit
def _offset_L(i, META: tl.constexpr):
    S: tl.constexpr = META[1]
    D: tl.constexpr = META[2]
    S0: tl.constexpr = META[3]
    S1: tl.constexpr = META[4]
    S2: tl.constexpr = META[5]
    if S2 == 1 and S1 == D and (S0 == S * D):
        return i
    H: tl.constexpr = META[0] // (S * D)
    if (
        S0 >= 2147483648
        or S1 >= 2147483648
        or S2 >= 2147483648
        or ((H - 1) * S0 + (S - 1) * S1 + (D - 1) * S2 >= 2147483648)
    ):
        i = i.to(tl.int64)
    return i // (S * D) * S0 + i // D % S * S1 + i % D * S2


@triton.jit
def _single_L(
    X, Q, OS, META: tl.constexpr, EPS: tl.constexpr, B: tl.constexpr
):
    N: tl.constexpr = META[0]
    i = tl.arange(0, B)
    v = tl.load(X + _offset_L(i, META), i < N, other=0).to(tl.float32)
    maximum = _epsilon_floor_L(_max_nan_L(tl.abs(v)), EPS)
    scale = maximum * 0.0022321429569274187
    q = _divide_L(v, scale, META[6])
    if META[6]:
        if META[7]:
            code = _encode_dynamic_bf16_L(q, v, scale)
        else:
            code = _encode_bf16_L(q, v)
    else:
        q = _clamp_L(q)
        code = _encode_L(q)
    tl.store(Q + i, code, i < N)
    tl.store(OS, scale)


@triton.jit(do_not_specialize=["X", "Q", "OS"])
def _single_all_address_L(
    X, Q, OS, META: tl.constexpr, EPS: tl.constexpr, B: tl.constexpr
):
    X = tl.multiple_of(X.to(tl.uint64).to(tl.pointer_type(tl.bfloat16)), 16)
    Q = tl.multiple_of(Q.to(tl.uint64).to(tl.pointer_type(tl.uint8)), 16)
    OS = tl.multiple_of(OS.to(tl.uint64).to(tl.pointer_type(tl.float32)), 16)
    _single_L(X, Q, OS, META, EPS, B)


@triton.jit(do_not_specialize=["Q"])
def _single_address_L(
    X, Q, OS, META: tl.constexpr, EPS: tl.constexpr, B: tl.constexpr
):
    Q = tl.multiple_of(Q.to(tl.uint64).to(tl.pointer_type(tl.uint8)), 16)
    _single_L(X, Q, OS, META, EPS, B)


@triton.jit
def _partial_L(X, P, META: tl.constexpr, B: tl.constexpr):
    N: tl.constexpr = META[0]
    i = _linear_index_L(tl.program_id(0), B, N)
    v = tl.load(X + _offset_L(i, META), i < N, other=0).to(tl.float32)
    tl.store(P + tl.program_id(0), _max_nan_L(tl.abs(v)))


@triton.jit(do_not_specialize=["X", "P"])
def _partial_all_address_L(X, P, META: tl.constexpr, B: tl.constexpr):
    X = tl.multiple_of(X.to(tl.uint64).to(tl.pointer_type(tl.bfloat16)), 16)
    P = tl.multiple_of(P.to(tl.uint64).to(tl.pointer_type(tl.float32)), 16)
    _partial_L(X, P, META, B)


@triton.jit
def _scale_L(P, OS, NP: tl.constexpr, PB: tl.constexpr, EPS: tl.constexpr):
    r = tl.arange(0, PB)
    a = tl.load(P + r, r < NP, other=0)
    scale = _epsilon_floor_L(_max_nan_L(a), EPS) * 0.0022321429569274187
    tl.store(OS, scale)


@triton.jit(do_not_specialize=["P", "OS"])
def _scale_all_address_L(
    P, OS, NP: tl.constexpr, PB: tl.constexpr, EPS: tl.constexpr
):
    P = tl.multiple_of(P.to(tl.uint64).to(tl.pointer_type(tl.float32)), 16)
    OS = tl.multiple_of(OS.to(tl.uint64).to(tl.pointer_type(tl.float32)), 16)
    _scale_L(P, OS, NP, PB, EPS)


@triton.jit
def _quant_L(
    X,
    P,
    Q,
    OS,
    META: tl.constexpr,
    EPS: tl.constexpr,
    B: tl.constexpr,
    NP: tl.constexpr,
    PB: tl.constexpr,
    REDUCE: tl.constexpr,
):
    N: tl.constexpr = META[0]
    if REDUCE:
        r = tl.arange(0, PB)
        a = tl.load(P + r, r < NP, other=0)
        scale = _epsilon_floor_L(_max_nan_L(a), EPS) * 0.0022321429569274187
        if tl.program_id(0) == 0:
            tl.store(OS, scale)
    else:
        scale = tl.load(OS)
    i = _linear_index_L(tl.program_id(0), B, N)
    v = tl.load(
        X + _offset_L(i, META),
        i < N,
        other=0,
        cache_modifier=".cg" if META[7] and META[0] >= 67108864 else "",
    ).to(tl.float32)
    q = _divide_L(v, scale, META[6])
    if META[6]:
        if META[7]:
            code = _encode_dynamic_bf16_L(q, v, scale)
        else:
            code = _encode_bf16_L(q, v)
    else:
        q = _clamp_L(q)
        code = _encode_L(q)
    tl.store(Q + i, code, i < N, cache_modifier=".cs" if META[7] else "")


@triton.jit(do_not_specialize=["X", "P", "Q", "OS"])
def _quant_all_address_L(
    X,
    P,
    Q,
    OS,
    META: tl.constexpr,
    EPS: tl.constexpr,
    B: tl.constexpr,
    NP: tl.constexpr,
    PB: tl.constexpr,
    REDUCE: tl.constexpr,
):
    X = tl.multiple_of(X.to(tl.uint64).to(tl.pointer_type(tl.bfloat16)), 16)
    P = tl.multiple_of(P.to(tl.uint64).to(tl.pointer_type(tl.float32)), 16)
    Q = tl.multiple_of(Q.to(tl.uint64).to(tl.pointer_type(tl.uint8)), 16)
    OS = tl.multiple_of(OS.to(tl.uint64).to(tl.pointer_type(tl.float32)), 16)
    _quant_L(X, P, Q, OS, META, EPS, B, NP, PB, REDUCE)


@triton.jit(do_not_specialize=["Q"])
def _quant_address_L(
    X,
    P,
    Q,
    OS,
    META: tl.constexpr,
    EPS: tl.constexpr,
    B: tl.constexpr,
    NP: tl.constexpr,
    PB: tl.constexpr,
    REDUCE: tl.constexpr,
):
    Q = tl.multiple_of(Q.to(tl.uint64).to(tl.pointer_type(tl.uint8)), 16)
    _quant_L(X, P, Q, OS, META, EPS, B, NP, PB, REDUCE)


@triton.jit
def _replicated_L(
    X,
    Q,
    OS,
    META: tl.constexpr,
    EPS: tl.constexpr,
    B: tl.constexpr,
    RB: tl.constexpr,
):
    N: tl.constexpr = META[0]
    r = tl.arange(0, RB)
    maximum = tl.full((RB,), 0.0, tl.float32)
    for base in range(triton.cdiv(N, RB)):
        j = base * RB + r
        v = tl.load(X + _offset_L(j, META), j < N, other=0).to(tl.float32)
        maximum = _maximum_nan_L(maximum, tl.abs(v))
    scale = _epsilon_floor_L(_max_nan_L(maximum), EPS) * 0.0022321429569274187
    i = _linear_index_L(tl.program_id(0), B, N)
    v = tl.load(X + _offset_L(i, META), i < N, other=0).to(tl.float32)
    q = _divide_L(v, scale, META[6])
    if META[6]:
        if META[7]:
            code = _encode_dynamic_bf16_L(q, v, scale)
        else:
            code = _encode_bf16_L(q, v)
    else:
        q = _clamp_L(q)
        code = _encode_L(q)
    tl.store(Q + i, code, i < N)
    if tl.program_id(0) == 0:
        tl.store(OS, scale)


@triton.jit(do_not_specialize=["X", "Q", "OS"])
def _replicated_all_address_L(
    X,
    Q,
    OS,
    META: tl.constexpr,
    EPS: tl.constexpr,
    B: tl.constexpr,
    RB: tl.constexpr,
):
    X = tl.multiple_of(X.to(tl.uint64).to(tl.pointer_type(tl.bfloat16)), 16)
    Q = tl.multiple_of(Q.to(tl.uint64).to(tl.pointer_type(tl.uint8)), 16)
    OS = tl.multiple_of(OS.to(tl.uint64).to(tl.pointer_type(tl.float32)), 16)
    _replicated_L(X, Q, OS, META, EPS, B, RB)


@triton.jit(do_not_specialize=["Q"])
def _replicated_address_L(
    X,
    Q,
    OS,
    META: tl.constexpr,
    EPS: tl.constexpr,
    B: tl.constexpr,
    RB: tl.constexpr,
):
    Q = tl.multiple_of(Q.to(tl.uint64).to(tl.pointer_type(tl.uint8)), 16)
    _replicated_L(X, Q, OS, META, EPS, B, RB)


def per_tensor_quant_mla_fp8_large(x, eps=1e-12):
    h, s, d = x.shape
    n = h * s * d
    out = x.new_empty((h, s, d), dtype=torch.float8_e4m3fn)
    output = out.data_ptr()
    assert output % 16 == 0
    scale = x.new_empty((1,), dtype=torch.float32)
    input_address = x.data_ptr()
    scale_address = scale.data_ptr()
    raw = (
        x.dtype == torch.bfloat16
        and (input_address | scale_address | output) % 16 == 0
    )
    meta = (
        n,
        s,
        d,
        *x.stride(),
        x.dtype == torch.bfloat16 and eps >= 1e-30,
        torch.version.hip is not None,
    )
    if 1024 < n <= 16384:
        if raw:
            _replicated_all_address_L.run(
                input_address,
                output,
                scale_address,
                meta,
                eps,
                1024,
                min(2048, 1 << (n - 1).bit_length() if n else 0),
                num_warps=4,
                enable_fp_fusion=False,
                grid=((n + 1024 - 1) // 1024,),
                warmup=False,
            )
        else:
            _replicated_address_L.run(
                x,
                output,
                scale,
                meta,
                eps,
                1024,
                min(2048, 1 << (n - 1).bit_length() if n else 0),
                num_warps=4,
                enable_fp_fusion=False,
                grid=((n + 1024 - 1) // 1024,),
                warmup=False,
            )
    elif n <= 1024:
        if raw:
            _single_all_address_L.run(
                input_address,
                output,
                scale_address,
                meta,
                eps,
                1 << (n - 1).bit_length() if n else 0,
                num_warps=4,
                enable_fp_fusion=False,
                grid=(1,),
                warmup=False,
            )
        else:
            _single_address_L.run(
                x,
                output,
                scale,
                meta,
                eps,
                1 << (n - 1).bit_length() if n else 0,
                num_warps=4,
                enable_fp_fusion=False,
                grid=(1,),
                warmup=False,
            )
    else:
        partial_block = (
            16384 if n >= 1048576 and torch.version.hip is None else 4096
        )
        partial_warps = 8 if n >= 1048576 and torch.version.hip is None else 4
        quant_block = (
            4096
            if n >= 8388608 and torch.version.hip is None and x.is_contiguous()
            else 1024
        )
        blocks = (n + partial_block - 1) // partial_block
        partial = x.new_empty((blocks,), dtype=torch.float32)
        partial_address = partial.data_ptr()
        assert partial_address % 16 == 0
        if raw:
            _partial_all_address_L.run(
                input_address,
                partial_address,
                meta,
                partial_block,
                num_warps=partial_warps,
                grid=(blocks,),
                warmup=False,
            )
        else:
            _partial_L.run(
                x,
                partial,
                meta,
                partial_block,
                num_warps=partial_warps,
                grid=(blocks,),
                warmup=False,
            )
        reduce = blocks <= 256
        if not reduce:
            if raw:
                _scale_all_address_L.run(
                    partial_address,
                    scale_address,
                    blocks,
                    1 << (blocks - 1).bit_length() if blocks else 0,
                    eps,
                    num_warps=4,
                    enable_fp_fusion=False,
                    grid=(1,),
                    warmup=False,
                )
            else:
                _scale_L.run(
                    partial,
                    scale,
                    blocks,
                    1 << (blocks - 1).bit_length() if blocks else 0,
                    eps,
                    num_warps=4,
                    enable_fp_fusion=False,
                    grid=(1,),
                    warmup=False,
                )
        if raw:
            _quant_all_address_L.run(
                input_address,
                partial_address,
                output,
                scale_address,
                meta,
                eps,
                quant_block,
                blocks,
                1 << (blocks - 1).bit_length() if blocks else 0,
                reduce,
                num_warps=4,
                enable_fp_fusion=False,
                grid=((n + quant_block - 1) // quant_block,),
                warmup=False,
            )
        else:
            _quant_address_L.run(
                x,
                partial,
                output,
                scale,
                meta,
                eps,
                quant_block,
                blocks,
                1 << (blocks - 1).bit_length() if blocks else 0,
                reduce,
                num_warps=4,
                enable_fp_fusion=False,
                grid=((n + quant_block - 1) // quant_block,),
                warmup=False,
            )
    return (out, scale)


def per_tensor_quant_mla_fp8(x, eps=1e-12):
    if x.numel() <= 512:
        return per_tensor_quant_mla_fp8_small(x, eps)
    return per_tensor_quant_mla_fp8_large(x, eps)


__all__ = ["per_tensor_quant_mla_fp8"]
