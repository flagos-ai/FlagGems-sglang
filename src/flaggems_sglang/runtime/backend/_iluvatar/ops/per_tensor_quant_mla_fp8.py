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
def _maximum_nan_S(a, b):
    return tl.maximum(a, b, propagate_nan=tl.PropagateNan.ALL)


@triton.jit
def _max_nan_S(values):
    return tl.max(values.to(tl.uint32, bitcast=True), 0).to(
        tl.float32, bitcast=True
    )


@triton.jit
def _clamp_S(v):
    return tl.minimum(
        tl.maximum(v, -448.0, propagate_nan=tl.PropagateNan.ALL),
        448.0,
        propagate_nan=tl.PropagateNan.ALL,
    )


@triton.jit
def _encode_S(v):
    raw = v.to(tl.uint32, bitcast=True)
    sign = raw >> 24 & 128
    mag = raw & 2147483647
    normal = (mag + 524287 + (mag >> 20 & 1) >> 20) - 960
    sub = (tl.abs(v) + 16384.0).to(tl.uint32, bitcast=True) - 1182793728
    code = tl.where(tl.abs(v) >= 0.015625, normal, sub)
    code = tl.where(v != v, 127, code)
    return (code | sign).to(tl.uint8)


@triton.jit
def _encode_bf16_S(v, original):
    raw = v.to(tl.uint32, bitcast=True)
    sign = original.to(tl.uint32, bitcast=True) >> 24 & 128
    mag = raw & 2147483647
    normal = (mag + 524287 + (mag >> 20 & 1) >> 20) - 960
    sub = (tl.abs(v) + 16384.0).to(tl.uint32, bitcast=True) - 1182793728
    code = tl.where(tl.abs(v) >= 0.015625, normal, sub)
    return tl.where(v != v, 127, code | sign).to(tl.uint8)


@triton.jit
def _divide_S(v, scale, BF16: tl.constexpr, SMALL: tl.constexpr):
    if SMALL:
        sb = scale.to(tl.uint32, bitcast=True)
        raw = v.to(tl.uint32, bitcast=True)
        magnitude = raw & 2147483647
        exponent = magnitude >> 23
        subnormal_scaled = (magnitude & 8388607).to(
            tl.float32
        ) * 2.350988701644575e-38
        normal_scaled_bits = magnitude + 201326592
        scaled_bits = (
            tl.where(
                exponent == 0,
                subnormal_scaled.to(tl.uint32, bitcast=True),
                normal_scaled_bits,
            )
            | raw & 2147483648
        )
        scale_scaled = sb.to(tl.float32) * 2.350988701644575e-38
        numerator_bits = tl.where(sb < 8388608, scaled_bits, raw)
        denominator_bits = tl.where(
            sb < 8388608, scale_scaled.to(tl.uint32, bitcast=True), sb
        )
        q = tl.div_rn(
            numerator_bits.to(tl.float32, bitcast=True),
            denominator_bits.to(tl.float32, bitcast=True),
        )
        return q
    if BF16:
        inv = tl.div_rn(1.0, scale)
        coarse = v * inv
        residual = tl.fma(
            -coarse, tl.where(scale < float("inf"), scale, 1.0), v
        )
        return tl.fma(residual, inv, coarse)
    return tl.div_rn(v, scale)


@triton.jit
def _linear_index_S(block, B: tl.constexpr, N: tl.constexpr):
    if N >= 2147483648:
        block = block.to(tl.int64)
    return block * B + tl.arange(0, B)


@triton.jit
def _offset_S(i, META: tl.constexpr):
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
def _single_S(
    X, Q, OS, META: tl.constexpr, EPS: tl.constexpr, B: tl.constexpr
):
    N: tl.constexpr = META[0]
    i = tl.arange(0, B)
    v = tl.load(X + _offset_S(i, META), i < N, other=0).to(tl.float32)
    maximum = _maximum_nan_S(
        _max_nan_S(tl.abs(v)), tl.full((), EPS, tl.float32)
    )
    scale = maximum * 0.0022321429569274187
    q = _divide_S(
        v,
        scale,
        X.dtype.element_ty == tl.bfloat16 and EPS >= 1e-30,
        EPS < 1e-30,
    )
    if not (
        X.dtype.element_ty == tl.bfloat16
        and EPS >= 1e-30
        and (EPS < float("inf"))
    ):
        q = _clamp_S(q)
    if X.dtype.element_ty == tl.bfloat16 and EPS >= 1e-30:
        code = _encode_bf16_S(q, v)
    else:
        code = _encode_S(q)
    tl.store(Q + i, code, i < N)
    tl.store(OS, scale)


@triton.jit(do_not_specialize=["X", "Q", "OS"])
def _single_all_address_S(
    X, Q, OS, META: tl.constexpr, EPS: tl.constexpr, B: tl.constexpr
):
    X = tl.multiple_of(X.to(tl.uint64).to(tl.pointer_type(tl.bfloat16)), 16)
    Q = tl.multiple_of(Q.to(tl.uint64).to(tl.pointer_type(tl.uint8)), 16)
    OS = tl.multiple_of(OS.to(tl.uint64).to(tl.pointer_type(tl.float32)), 16)
    _single_S(X, Q, OS, META, EPS, B)


@triton.jit(do_not_specialize=["Q"])
def _single_address_S(
    X, Q, OS, META: tl.constexpr, EPS: tl.constexpr, B: tl.constexpr
):
    Q = tl.multiple_of(Q.to(tl.uint64).to(tl.pointer_type(tl.uint8)), 16)
    _single_S(X, Q, OS, META, EPS, B)


@triton.jit
def _partial_S(X, P, META: tl.constexpr, B: tl.constexpr):
    N: tl.constexpr = META[0]
    i = _linear_index_S(tl.program_id(0), B, N)
    v = tl.load(X + _offset_S(i, META), i < N, other=0).to(tl.float32)
    tl.store(P + tl.program_id(0), _max_nan_S(tl.abs(v)))


@triton.jit(do_not_specialize=["X", "P"])
def _partial_all_address_S(X, P, META: tl.constexpr, B: tl.constexpr):
    X = tl.multiple_of(X.to(tl.uint64).to(tl.pointer_type(tl.bfloat16)), 16)
    P = tl.multiple_of(P.to(tl.uint64).to(tl.pointer_type(tl.float32)), 16)
    _partial_S(X, P, META, B)


@triton.jit
def _partial_persistent_S(X, P, META: tl.constexpr, B: tl.constexpr):
    N: tl.constexpr = META[0]
    maximum = tl.full((B,), 0, tl.uint32)
    for tile in range(tl.program_id(0), triton.cdiv(N, B), tl.num_programs(0)):
        i = _linear_index_S(tile, B, N)
        v = tl.load(X + _offset_S(i, META), i < N, other=0).to(tl.float32)
        maximum = tl.maximum(maximum, tl.abs(v).to(tl.uint32, bitcast=True))
    tl.store(
        P + tl.program_id(0), tl.max(maximum, 0).to(tl.float32, bitcast=True)
    )


@triton.jit(do_not_specialize=["X", "P"])
def _partial_persistent_all_address_S(
    X, P, META: tl.constexpr, B: tl.constexpr
):
    X = tl.multiple_of(X.to(tl.uint64).to(tl.pointer_type(tl.bfloat16)), 16)
    P = tl.multiple_of(P.to(tl.uint64).to(tl.pointer_type(tl.float32)), 16)
    _partial_persistent_S(X, P, META, B)


@triton.jit
def _scale_S(P, OS, NP: tl.constexpr, PB: tl.constexpr, EPS: tl.constexpr):
    r = tl.arange(0, PB)
    a = tl.load(P + r, r < NP, other=0)
    scale = (
        _maximum_nan_S(_max_nan_S(a), tl.full((), EPS, tl.float32))
        * 0.0022321429569274187
    )
    tl.store(OS, scale)


@triton.jit(do_not_specialize=["P", "OS"])
def _scale_all_address_S(
    P, OS, NP: tl.constexpr, PB: tl.constexpr, EPS: tl.constexpr
):
    P = tl.multiple_of(P.to(tl.uint64).to(tl.pointer_type(tl.float32)), 16)
    OS = tl.multiple_of(OS.to(tl.uint64).to(tl.pointer_type(tl.float32)), 16)
    _scale_S(P, OS, NP, PB, EPS)


@triton.jit
def _quant_S(
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
        scale = (
            _maximum_nan_S(_max_nan_S(a), tl.full((), EPS, tl.float32))
            * 0.0022321429569274187
        )
        if tl.program_id(0) == 0:
            tl.store(OS, scale)
    else:
        scale = tl.load(OS)
    i = _linear_index_S(tl.program_id(0), B, N)
    v = tl.load(X + _offset_S(i, META), i < N, other=0).to(tl.float32)
    q = _divide_S(
        v,
        scale,
        X.dtype.element_ty == tl.bfloat16 and EPS >= 1e-30,
        EPS < 1e-30,
    )
    if not (
        X.dtype.element_ty == tl.bfloat16
        and EPS >= 1e-30
        and (EPS < float("inf"))
    ):
        q = _clamp_S(q)
    if X.dtype.element_ty == tl.bfloat16 and EPS >= 1e-30:
        code = _encode_bf16_S(q, v)
    else:
        code = _encode_S(q)
    tl.store(Q + i, code, i < N)


@triton.jit(do_not_specialize=["X", "P", "Q", "OS"])
def _quant_all_address_S(
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
    _quant_S(X, P, Q, OS, META, EPS, B, NP, PB, REDUCE)


@triton.jit(do_not_specialize=["Q"])
def _quant_address_S(
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
    _quant_S(X, P, Q, OS, META, EPS, B, NP, PB, REDUCE)


@triton.jit
def _replicated_S(
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
    maximum = tl.full((RB,), 0, tl.uint32)
    for base in range(triton.cdiv(N, RB)):
        j = base * RB + r
        v = tl.load(X + _offset_S(j, META), j < N, other=0).to(tl.float32)
        maximum = tl.maximum(maximum, tl.abs(v).to(tl.uint32, bitcast=True))
    scale = (
        _maximum_nan_S(
            tl.max(maximum, 0).to(tl.float32, bitcast=True),
            tl.full((), EPS, tl.float32),
        )
        * 0.0022321429569274187
    )
    i = _linear_index_S(tl.program_id(0), B, N)
    v = tl.load(X + _offset_S(i, META), i < N, other=0).to(tl.float32)
    q = _divide_S(
        v,
        scale,
        X.dtype.element_ty == tl.bfloat16 and EPS >= 1e-30,
        EPS < 1e-30,
    )
    if not (
        X.dtype.element_ty == tl.bfloat16
        and EPS >= 1e-30
        and (EPS < float("inf"))
    ):
        q = _clamp_S(q)
    if X.dtype.element_ty == tl.bfloat16 and EPS >= 1e-30:
        code = _encode_bf16_S(q, v)
    else:
        code = _encode_S(q)
    tl.store(Q + i, code, i < N)
    if tl.program_id(0) == 0:
        tl.store(OS, scale)


@triton.jit(do_not_specialize=["X", "Q", "OS"])
def _replicated_all_address_S(
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
    _replicated_S(X, Q, OS, META, EPS, B, RB)


@triton.jit(do_not_specialize=["Q"])
def _replicated_address_S(
    X,
    Q,
    OS,
    META: tl.constexpr,
    EPS: tl.constexpr,
    B: tl.constexpr,
    RB: tl.constexpr,
):
    Q = tl.multiple_of(Q.to(tl.uint64).to(tl.pointer_type(tl.uint8)), 16)
    _replicated_S(X, Q, OS, META, EPS, B, RB)


@triton.jit
def _scale_preinverse_S(
    P, OS, NP: tl.constexpr, PB: tl.constexpr, EPS: tl.constexpr
):
    r = tl.arange(0, PB)
    a = tl.load(P + r, r < NP, other=0)
    scale = (
        _maximum_nan_S(_max_nan_S(a), tl.full((), EPS, tl.float32))
        * 0.0022321429569274187
    )
    tl.store(OS, scale)
    inv = tl.div_rn(1.0, scale)
    tl.store(P, inv)


@triton.jit
def _quant_preinverse_S(
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
        scale = (
            _maximum_nan_S(_max_nan_S(a), tl.full((), EPS, tl.float32))
            * 0.0022321429569274187
        )
        if tl.program_id(0) == 0:
            tl.store(OS, scale)
    else:
        scale = tl.load(OS)
    i = _linear_index_S(tl.program_id(0), B, N)
    v = tl.load(X + _offset_S(i, META), i < N, other=0).to(tl.float32)
    inv = tl.load(P)
    coarse = v * inv
    residual = tl.fma(-coarse, tl.where(scale < float("inf"), scale, 1.0), v)
    q = tl.fma(residual, inv, coarse)
    if not (
        X.dtype.element_ty == tl.bfloat16
        and EPS >= 1e-30
        and (EPS < float("inf"))
    ):
        q = _clamp_S(q)
    if X.dtype.element_ty == tl.bfloat16 and EPS >= 1e-30:
        code = _encode_bf16_S(q, v)
    else:
        code = _encode_S(q)
    tl.store(Q + i, code, i < N)


@triton.jit(do_not_specialize=["P", "OS"])
def _scale_preinverse_all_address_S(
    P, OS, NP: tl.constexpr, PB: tl.constexpr, EPS: tl.constexpr
):
    P = tl.multiple_of(P.to(tl.uint64).to(tl.pointer_type(tl.float32)), 16)
    OS = tl.multiple_of(OS.to(tl.uint64).to(tl.pointer_type(tl.float32)), 16)
    _scale_preinverse_S(P, OS, NP, PB, EPS)


@triton.jit(do_not_specialize=["X", "P", "Q", "OS"])
def _quant_preinverse_all_address_S(
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
    _quant_preinverse_S(X, P, Q, OS, META, EPS, B, NP, PB, REDUCE)


@triton.jit(do_not_specialize=["Q"])
def _quant_preinverse_address_S(
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
    _quant_preinverse_S(X, P, Q, OS, META, EPS, B, NP, PB, REDUCE)


def per_tensor_quant_mla_fp8_small(x, eps=1e-12):
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
    meta = (n, s, d, *x.stride())
    if 4096 < n <= 65536:
        reduction_block = (
            8192
            if n >= 32768
            else min(2048, 1 << (n - 1).bit_length() if n else 0)
        )
        if raw:
            _replicated_all_address_S.run(
                input_address,
                output,
                scale_address,
                meta,
                eps,
                1024,
                reduction_block,
                num_warps=4,
                enable_fp_fusion=False,
                grid=((n + 1024 - 1) // 1024,),
                warmup=False,
            )
        else:
            _replicated_address_S.run(
                x,
                output,
                scale,
                meta,
                eps,
                1024,
                reduction_block,
                num_warps=4,
                enable_fp_fusion=False,
                grid=((n + 1024 - 1) // 1024,),
                warmup=False,
            )
    elif n <= 4096:
        if raw:
            _single_all_address_S.run(
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
            _single_address_S.run(
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
        persistent = (
            n >= 8388608 and x.dtype == torch.bfloat16 and x.is_contiguous()
        )
        preinverse = persistent and eps >= 1e-30
        blocks = (
            min(1024, (n + 4096 - 1) // 4096)
            if persistent
            else (n + 4096 - 1) // 4096
        )
        partial = x.new_empty((blocks,), dtype=torch.float32)
        partial_address = partial.data_ptr()
        assert partial_address % 16 == 0
        if persistent:
            if raw:
                _partial_persistent_all_address_S.run(
                    input_address,
                    partial_address,
                    meta,
                    4096,
                    num_warps=4,
                    grid=(blocks,),
                    warmup=False,
                )
            else:
                _partial_persistent_S.run(
                    x,
                    partial,
                    meta,
                    4096,
                    num_warps=4,
                    grid=(blocks,),
                    warmup=False,
                )
        elif raw:
            _partial_all_address_S.run(
                input_address,
                partial_address,
                meta,
                4096,
                num_warps=4,
                grid=(blocks,),
                warmup=False,
            )
        else:
            _partial_S.run(
                x,
                partial,
                meta,
                4096,
                num_warps=4,
                grid=(blocks,),
                warmup=False,
            )
        reduce = blocks <= 256
        if not reduce:
            if raw:
                if preinverse:
                    _scale_preinverse_all_address_S.run(
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
                    _scale_all_address_S.run(
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
            elif preinverse:
                _scale_preinverse_S.run(
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
            else:
                _scale_S.run(
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
            if preinverse:
                _quant_preinverse_all_address_S.run(
                    input_address,
                    partial_address,
                    output,
                    scale_address,
                    meta,
                    eps,
                    1024,
                    blocks,
                    1 << (blocks - 1).bit_length() if blocks else 0,
                    reduce,
                    num_warps=4,
                    enable_fp_fusion=False,
                    grid=((n + 1024 - 1) // 1024,),
                    warmup=False,
                )
            else:
                _quant_all_address_S.run(
                    input_address,
                    partial_address,
                    output,
                    scale_address,
                    meta,
                    eps,
                    1024,
                    blocks,
                    1 << (blocks - 1).bit_length() if blocks else 0,
                    reduce,
                    num_warps=4,
                    enable_fp_fusion=False,
                    grid=((n + 1024 - 1) // 1024,),
                    warmup=False,
                )
        elif preinverse:
            _quant_preinverse_address_S.run(
                x,
                partial,
                output,
                scale,
                meta,
                eps,
                1024,
                blocks,
                1 << (blocks - 1).bit_length() if blocks else 0,
                reduce,
                num_warps=4,
                enable_fp_fusion=False,
                grid=((n + 1024 - 1) // 1024,),
                warmup=False,
            )
        else:
            _quant_address_S.run(
                x,
                partial,
                output,
                scale,
                meta,
                eps,
                1024,
                blocks,
                1 << (blocks - 1).bit_length() if blocks else 0,
                reduce,
                num_warps=4,
                enable_fp_fusion=False,
                grid=((n + 1024 - 1) // 1024,),
                warmup=False,
            )
    return (out, scale)


@triton.jit
def _encode_e4m3fn_L(x):

    u32 = x.to(tl.int32, bitcast=True)
    u = u32 & 0x7FFFFFFF
    rounded = u + 0x7FFFF + ((u >> 20) & 1)
    normal = (rounded >> 20) - 960
    sub = (tl.abs(x) + 16384.0).to(tl.int32, bitcast=True) - 0x46800000
    code = tl.where(u < 0x3C800000, sub, normal)
    return (code | ((u32 >> 24) & 0x80)).to(tl.uint8)


@triton.jit
def _encode_saturating_L(x):

    isnan = x != x
    x = tl.minimum(tl.maximum(x, -448.0), 448.0)
    code = _encode_e4m3fn_L(x)
    return tl.where(isnan, 0x7F, code)


@triton.jit
def _flat_absmax_kernel_L(
    x_ptr,
    pmax_ptr,
    pnan_ptr,
    total,
    NB,
    BLOCK: tl.constexpr,
):

    pid = tl.program_id(0).to(tl.int64)
    nprog = tl.num_programs(0).to(tl.int64)
    m = tl.zeros((BLOCK,), dtype=tl.float32)

    nan_acc = tl.zeros((BLOCK,), dtype=tl.int32)
    for i in range(pid, NB, nprog):
        offs = i * BLOCK + tl.arange(0, BLOCK)
        v = tl.load(x_ptr + offs, mask=offs < total, other=0.0).to(tl.float32)
        m = tl.maximum(m, tl.abs(v))
        nan_acc = tl.maximum(nan_acc, (v != v).to(tl.int32))
    tl.store(pmax_ptr + pid, tl.max(m))
    tl.store(pnan_ptr + pid, tl.max(nan_acc))


@triton.jit
def _reduce_final_kernel_L(
    pmax_ptr,
    pnan_ptr,
    scale_ptr,
    P,
    eps,
    BLOCK: tl.constexpr,
):

    offs = tl.arange(0, BLOCK)
    m = tl.load(pmax_ptr + offs, mask=offs < P, other=0.0)
    nan = tl.load(pnan_ptr + offs, mask=offs < P, other=0)
    amax = tl.max(m)
    any_nan = tl.max(nan) > 0
    tl.store(
        scale_ptr,
        tl.where(any_nan, float("nan"), tl.maximum(amax, eps) / 448.0),
    )


@triton.jit
def _quant_kernel_L(
    x_ptr,
    scale_ptr,
    out_u8_ptr,
    total,
    NB,
    BLOCK: tl.constexpr,
):

    pid = tl.program_id(0).to(tl.int64)
    nprog = tl.num_programs(0).to(tl.int64)
    scale = tl.load(scale_ptr)
    for i in range(pid, NB, nprog):
        offs = i * BLOCK + tl.arange(0, BLOCK)
        m = offs < total
        v = tl.load(x_ptr + offs, mask=m, other=0.0).to(tl.float32)
        q = v / scale
        tl.store(out_u8_ptr + offs, _encode_saturating_L(q), mask=m)


def per_tensor_quant_mla_fp8_large(x, eps=1e-12):
    H, S, D = x.shape
    out = torch.empty((H, S, D), dtype=torch.float8_e4m3fn, device=x.device)
    scale = torch.empty((1,), dtype=torch.float32, device=x.device)
    n = H * S
    if n <= 0 or D <= 0:
        scale.fill_(0.0)
        return out, scale
    if not x.is_contiguous():
        x = x.contiguous()
    fblock = 1024
    total = n * D
    NB = triton.cdiv(total, fblock)
    p = min(NB, 4096)
    pmax = torch.empty((p,), dtype=torch.float32, device=x.device)
    pnan = torch.empty((p,), dtype=torch.int32, device=x.device)
    _flat_absmax_kernel_L[(p,)](
        x,
        pmax,
        pnan,
        total,
        NB,
        BLOCK=fblock,
        num_warps=4,
    )
    _reduce_final_kernel_L[(1,)](
        pmax,
        pnan,
        scale,
        p,
        eps,
        BLOCK=4096,
        num_warps=4,
    )
    qblock = 1024
    _quant_kernel_L[(min(triton.cdiv(total, qblock), 32768),)](
        x,
        scale,
        out.view(torch.uint8),
        total,
        triton.cdiv(total, qblock),
        BLOCK=qblock,
        num_warps=4,
    )
    return out, scale


def per_tensor_quant_mla_fp8(x, eps=1e-12):
    if x.numel() <= 33554432:
        return per_tensor_quant_mla_fp8_small(x, eps)
    return per_tensor_quant_mla_fp8_large(x, eps)


__all__ = ["per_tensor_quant_mla_fp8"]
