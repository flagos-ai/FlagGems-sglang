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

import struct

import torch
import triton
import triton.language as tl
from triton.language import bfloat16 as bf16
from triton.language import float32 as fp32
from triton.language import int8 as i8


@triton.jit
def _normalize_finite194(x, a):
    finite = a.to(tl.int32, bitcast=True) < 2139095040
    return tl.where(finite, x * (1.0 / a * 127.0), 0.0)


@triton.jit
def _maximum_bits189(a, b):
    ai = a.to(tl.int32, bitcast=True) & 2147483647
    bi = b.to(tl.int32, bitcast=True) & 2147483647
    return tl.maximum(ai, bi).to(tl.float32, bitcast=True)


@triton.jit
def _reduce_bits189(value, AXIS: tl.constexpr):
    bits = value.to(tl.int32, bitcast=True) & 2147483647
    return tl.max(bits, AXIS).to(tl.float32, bitcast=True)


@triton.jit
def _clamp_min189(value):
    return tl.where(value < 1e-10, 1e-10, value)


@triton.jit
def _nan_code_zero189(value):
    bits = value.to(tl.int32, bitcast=True) & 2147483647
    return tl.where(bits > 2139095040, 0.0, value)


@triton.jit
def _wide_index(value, WIDE: tl.constexpr):
    if WIDE:
        return value.to(tl.int64)
    else:
        return value


@triton.jit
def _quant(X, Q, S, U, META: tl.constexpr):
    M: tl.constexpr = META[0]
    N: tl.constexpr = META[1]
    SUM: tl.constexpr = META[2]
    B: tl.constexpr = triton.next_power_of_2(N)
    WIDE: tl.constexpr = M * N > 2147483647
    c = tl.arange(0, B)
    for row in range(
        _wide_index(tl.program_id(0), WIDE), M, tl.num_programs(0)
    ):
        x = tl.load(X + row * N + c, c < N, 0).to(tl.float32)
        a = _clamp_min189(_reduce_bits189(tl.abs(x), 0))
        v = _normalize_finite194(x, a)
        q = tl.floor(tl.abs(v) + 0.5)
        q = tl.where(v < 0, -q, q).to(tl.int8)
        tl.store(Q + row * N + c, q, c < N)
        tl.store(S + row, a / 127.0)
        if SUM:
            tl.store(U + row, tl.sum(x, 0))


@triton.jit
def _stats(
    X,
    A,
    S,
    U,
    M: tl.constexpr,
    N: tl.constexpr,
    SUM: tl.constexpr,
    B: tl.constexpr,
    WIDE: tl.constexpr,
):
    c = tl.arange(0, B)
    for row in range(
        _wide_index(tl.program_id(0), WIDE), M, tl.num_programs(0)
    ):
        maxima = tl.full((B,), 0.0, tl.float32)
        totals = tl.full((B,), 0.0, tl.float32)
        for start in range(tl.cdiv(N, B)):
            col = _wide_index(start, WIDE) * B + c
            x = tl.load(X + row * N + col, col < N, 0).to(tl.float32)
            maxima = _maximum_bits189(maxima, tl.abs(x))
            if SUM:
                totals += x
        a = _clamp_min189(_reduce_bits189(maxima, 0))
        tl.store(A + row, a)
        tl.store(S + row, a / 127.0)
        if SUM:
            tl.store(U + row, tl.sum(totals, 0))


@triton.jit
def _finish(
    X,
    A,
    Q,
    M: tl.constexpr,
    N: tl.constexpr,
    B: tl.constexpr,
    WIDE: tl.constexpr,
):
    c = _wide_index(tl.program_id(1), WIDE) * B + tl.arange(0, B)
    for row in range(
        _wide_index(tl.program_id(0), WIDE), M, tl.num_programs(0)
    ):
        a = tl.load(A + row)
        x = tl.load(X + row * N + c, c < N, 0).to(tl.float32)
        v = _normalize_finite194(x, a)
        q = tl.floor(tl.abs(v) + 0.5)
        q = tl.where(v < 0, -q, q).to(tl.int8)
        tl.store(Q + row * N + c, q, c < N)


@triton.jit
def _quant_half(X, Q, S, U, META: tl.constexpr):
    M: tl.constexpr = META[0]
    N: tl.constexpr = META[1]
    SUM: tl.constexpr = META[2]
    B: tl.constexpr = triton.next_power_of_2(N)
    WIDE: tl.constexpr = M * N > 2147483647
    c = tl.arange(0, B)
    for row in range(
        _wide_index(tl.program_id(0), WIDE), M, tl.num_programs(0)
    ):
        x = tl.load(X + row * N + c, c < N, 0).to(tl.float32)
        a = _clamp_min189(_reduce_half351(x, 0))
        if a.to(tl.int32, bitcast=True) < 2139095040:
            v = x * (1.0 / a * 127.0)
            q = tl.where(v < 0, v - 0.5, v + 0.5).to(tl.int32).to(tl.int8)
            tl.store(Q + row * N + c, q, c < N)
        else:
            tl.store(Q + row * N + c, tl.full((B,), 0, tl.int8), c < N)
        tl.store(S + row, a / 127.0)
        if SUM:
            tl.store(U + row, tl.sum(x, 0))


def official(x, scale_dtype, cal_sum):
    n = x.shape[-1]
    m = x.numel() // n
    q = x.new_empty(x.shape, dtype=torch.int8)
    scales = x.new_empty((*x.shape[:-1], 1), dtype=scale_dtype)
    sums = x.new_empty(x.shape[:-1]) if cal_sum else scales
    if m:
        if n <= 4096:
            low_sum = not cal_sum or x.dtype in (torch.float16, torch.bfloat16)
            kernel = (
                _quant_half if n > 2048 and m >= 1024 and low_sum else _quant
            )
            kernel[min(m, 40 if m >= 1024 and low_sum else 65535),](
                x, q, scales, sums, (m, n, cal_sum), enable_fp_fusion=False
            )
        else:
            a = x.new_empty((m,), dtype=torch.float32)
            _stats[min(m, 65535),](
                x,
                a,
                scales,
                sums,
                m,
                n,
                cal_sum,
                2048,
                enable_fp_fusion=False,
                WIDE=m * n > 2147483647,
            )
            blocks = triton.cdiv(n, 1024)
            _finish[min(m, max(1, 65535 // blocks)), blocks](
                x,
                a,
                q,
                m,
                n,
                1024,
                enable_fp_fusion=False,
                WIDE=m * n > 2147483647,
            )
    if cal_sum:
        return (q, scales, sums)
    return (q, scales)


@triton.jit
def _fused_split(X, Q, S, U, META: tl.constexpr):
    M: tl.constexpr = META[0]
    N: tl.constexpr = META[1]
    SUM: tl.constexpr = META[2]
    B: tl.constexpr = 2048
    WIDE: tl.constexpr = M * N > 2147483647
    c = tl.arange(0, B)
    for row in range(
        _wide_index(tl.program_id(0), WIDE), M, tl.num_programs(0)
    ):
        maxima = tl.full((B,), 0, tl.int32)
        totals = tl.full((B,), 0.0, tl.float32)
        if (
            N <= 8192
            and M >= 512
            and (
                not SUM
                or X.dtype.element_ty == tl.float16
                or X.dtype.element_ty == tl.bfloat16
            )
        ):
            for start in tl.static_range((N + B - 1) // B):
                col = _wide_index(start, WIDE) * B + c
                x = tl.load(X + row * N + col, col < N, 0).to(tl.float32)
                maxima = tl.maximum(
                    maxima, tl.abs(x).to(tl.int32, bitcast=True)
                )
                if SUM:
                    totals += x
        else:
            for start in range(tl.cdiv(N, B)):
                col = _wide_index(start, WIDE) * B + c
                x = tl.load(X + row * N + col, col < N, 0).to(tl.float32)
                maxima = tl.maximum(
                    maxima, tl.abs(x).to(tl.int32, bitcast=True)
                )
                if SUM:
                    totals += x
        a = _clamp_min189(tl.max(maxima, 0).to(tl.float32, bitcast=True))
        tl.store(S + row, a / 127.0)
        if SUM:
            tl.store(U + row, tl.sum(totals, 0))
        if (
            N <= 8192
            and M >= 512
            and (
                not SUM
                or X.dtype.element_ty == tl.float16
                or X.dtype.element_ty == tl.bfloat16
            )
        ):
            for start in tl.static_range((N + B - 1) // B):
                col = _wide_index(start, WIDE) * B + c
                x = tl.load(X + row * N + col, col < N, 0).to(tl.float32)
                v = _normalize_finite194(x, a)
                q = tl.floor(tl.abs(v) + 0.5)
                q = tl.where(v < 0, -q, q).to(tl.int8)
                tl.store(Q + row * N + col, q, col < N)
        else:
            for start in range(tl.cdiv(N, B)):
                col = _wide_index(start, WIDE) * B + c
                x = tl.load(X + row * N + col, col < N, 0).to(tl.float32)
                v = _normalize_finite194(x, a)
                q = tl.floor(tl.abs(v) + 0.5)
                q = tl.where(v < 0, -q, q).to(tl.int8)
                tl.store(Q + row * N + col, q, col < N)


@triton.jit
def _maximum_nan321(a, b):
    return tl.maximum(a, b, propagate_nan=tl.PropagateNan.ALL)


@triton.jit
def _resident_split168(X, Q, S, U, META: tl.constexpr):
    M: tl.constexpr = META[0]
    N: tl.constexpr = META[1]
    SUM: tl.constexpr = META[2]
    B: tl.constexpr = 2048
    WIDE: tl.constexpr = M * N > 2147483647
    row = _wide_index(tl.program_id(0), WIDE)
    c = tl.arange(0, B)
    x0 = tl.load(X + row * N + c, c < N, 0).to(tl.float32)
    x1 = tl.load(X + row * N + B + c, B + c < N, 0).to(tl.float32)
    x2 = tl.load(X + row * N + 2 * B + c, 2 * B + c < N, 0).to(tl.float32)
    if N > 3 * B:
        x3 = tl.load(X + row * N + 3 * B + c, 3 * B + c < N, 0).to(tl.float32)
    maxima = tl.full((B,), 0, tl.int32)
    maxima = tl.maximum(maxima, x0.to(tl.int32, bitcast=True) & 2147483647)
    maxima = tl.maximum(maxima, x1.to(tl.int32, bitcast=True) & 2147483647)
    maxima = tl.maximum(maxima, x2.to(tl.int32, bitcast=True) & 2147483647)
    if N > 3 * B:
        maxima = tl.maximum(maxima, x3.to(tl.int32, bitcast=True) & 2147483647)
    reduced321 = tl.reduce(
        maxima.to(tl.float32, bitcast=True), 0, _maximum_nan321
    )
    positive321 = reduced321.to(tl.int32, bitcast=True) & 2147483647
    a = _clamp_min189(positive321.to(tl.float32, bitcast=True))
    tl.store(S + row, a / 127.0)
    if SUM:
        totals = tl.full((B,), 0.0, tl.float32)
        totals += x0
        totals += x1
        totals += x2
        if N > 3 * B:
            totals += x3
        tl.store(U + row, tl.sum(totals, 0))
    if X.dtype.element_ty == tl.bfloat16:
        if a.to(tl.int32, bitcast=True) < 2139095040:
            v0 = x0 * (1.0 / a * 127.0)
            q0 = v0
            q0 = tl.where(v0 < 0, q0 - 0.5, q0 + 0.5).to(tl.int32).to(tl.int8)
            tl.store(Q + row * N + c, q0, c < N)
            v1 = x1 * (1.0 / a * 127.0)
            q1 = v1
            q1 = tl.where(v1 < 0, q1 - 0.5, q1 + 0.5).to(tl.int32).to(tl.int8)
            tl.store(Q + row * N + B + c, q1, B + c < N)
            v2 = x2 * (1.0 / a * 127.0)
            q2 = v2
            q2 = tl.where(v2 < 0, q2 - 0.5, q2 + 0.5).to(tl.int32).to(tl.int8)
            tl.store(Q + row * N + 2 * B + c, q2, 2 * B + c < N)
            if N > 3 * B:
                v3 = x3 * (1.0 / a * 127.0)
                q3 = v3
                q3 = (
                    tl.where(v3 < 0, q3 - 0.5, q3 + 0.5)
                    .to(tl.int32)
                    .to(tl.int8)
                )
                tl.store(Q + row * N + 3 * B + c, q3, 3 * B + c < N)
        else:
            zero = tl.full((B,), 0, tl.int8)
            tl.store(Q + row * N + c, zero, c < N)
            tl.store(Q + row * N + B + c, zero, B + c < N)
            tl.store(Q + row * N + 2 * B + c, zero, 2 * B + c < N)
            if N > 3 * B:
                tl.store(Q + row * N + 3 * B + c, zero, 3 * B + c < N)
    else:
        v0 = _normalize_finite194(x0, a)
        q0 = v0
        q0 = tl.where(v0 < 0, q0 - 0.5, q0 + 0.5).to(tl.int32).to(tl.int8)
        tl.store(Q + row * N + c, q0, c < N)
        v1 = _normalize_finite194(x1, a)
        q1 = v1
        q1 = tl.where(v1 < 0, q1 - 0.5, q1 + 0.5).to(tl.int32).to(tl.int8)
        tl.store(Q + row * N + B + c, q1, B + c < N)
        v2 = _normalize_finite194(x2, a)
        q2 = v2
        q2 = tl.where(v2 < 0, q2 - 0.5, q2 + 0.5).to(tl.int32).to(tl.int8)
        tl.store(Q + row * N + 2 * B + c, q2, 2 * B + c < N)
        if N > 3 * B:
            v3 = _normalize_finite194(x3, a)
            q3 = v3
            q3 = tl.where(v3 < 0, q3 - 0.5, q3 + 0.5).to(tl.int32).to(tl.int8)
            tl.store(Q + row * N + 3 * B + c, q3, 3 * B + c < N)


def per_token_quant_int8_sum(x, scale_dtype, cal_sum):
    shape = x.shape
    n = shape[-1]
    m = x.numel() // n
    q = x.new_empty(shape, dtype=torch.int8)
    scales = x.new_empty(shape[:-1] + (1,), dtype=scale_dtype)
    sums = x.new_empty(shape[:-1]) if cal_sum else scales
    if m:
        typed460 = x.dtype == torch.bfloat16 and scale_dtype == torch.float32
        grid536 = (
            min(
                m,
                (
                    40
                    if m >= 1024
                    and (
                        not cal_sum
                        or x.dtype in (torch.float16, torch.bfloat16)
                    )
                    else 65535
                ),
            )
            if n <= 4096
            else min(m, 65535)
        )
        carrier536 = typed460 and grid536 >= m
        if typed460:
            x_ptr460 = x.data_ptr()
            q_ptr460 = q.data_ptr()
            s_ptr460 = scales.data_ptr()
            u_ptr460 = sums.data_ptr()
            align460 = (
                x_ptr460 % 16 == 0,
                q_ptr460 % 16 == 0,
                s_ptr460 % 16 == 0,
                u_ptr460 % 16 == 0,
            )
            meta460 = (m, n, cal_sum, align460)
            if carrier536:
                x_arg536, q_arg536, s_arg536, u_arg536 = struct.unpack(
                    "=4d",
                    struct.pack("=4Q", x_ptr460, q_ptr460, s_ptr460, u_ptr460),
                )
        if n <= 4096:
            low_sum = not cal_sum or x.dtype in (torch.float16, torch.bfloat16)
            kernel = (
                _quant_half if n > 2048 and m >= 1024 and low_sum else _quant
            )
            if typed460:
                if carrier536:
                    (
                        _quant_half_carrier536
                        if kernel is _quant_half
                        else _quant_carrier536
                    ).run(
                        x_arg536,
                        q_arg536,
                        s_arg536,
                        u_arg536,
                        meta460,
                        enable_fp_fusion=False,
                        grid=(grid536,),
                        warmup=False,
                    )
                else:
                    (
                        _quant_half_pointer460
                        if kernel is _quant_half
                        else _quant_pointer460
                    ).run(
                        x_ptr460,
                        q_ptr460,
                        s_ptr460,
                        u_ptr460,
                        meta460,
                        enable_fp_fusion=False,
                        grid=(grid536,),
                        warmup=False,
                    )
            else:
                kernel.run(
                    x,
                    q,
                    scales,
                    sums,
                    (m, n, cal_sum),
                    enable_fp_fusion=False,
                    grid=(grid536,),
                    warmup=False,
                )
        elif typed460:
            if carrier536:
                (
                    _resident_split168_carrier536
                    if n <= 8192
                    and 512 <= m <= 65535
                    and (
                        not cal_sum
                        or x.dtype in (torch.float16, torch.bfloat16)
                    )
                    else _fused_split_carrier536
                ).run(
                    x_arg536,
                    q_arg536,
                    s_arg536,
                    u_arg536,
                    meta460,
                    enable_fp_fusion=False,
                    grid=(grid536,),
                    warmup=False,
                )
            else:
                (
                    _resident_split168_pointer460
                    if n <= 8192
                    and 512 <= m <= 65535
                    and (
                        not cal_sum
                        or x.dtype in (torch.float16, torch.bfloat16)
                    )
                    else _fused_split_pointer460
                ).run(
                    x_ptr460,
                    q_ptr460,
                    s_ptr460,
                    u_ptr460,
                    meta460,
                    enable_fp_fusion=False,
                    grid=(grid536,),
                    warmup=False,
                )
        else:
            (
                _resident_split168
                if n <= 8192
                and 512 <= m <= 65535
                and (not cal_sum or x.dtype in (torch.float16, torch.bfloat16))
                else _fused_split
            ).run(
                x,
                q,
                scales,
                sums,
                (m, n, cal_sum),
                enable_fp_fusion=False,
                grid=(grid536,),
                warmup=False,
            )
    return (q, scales, sums) if cal_sum else (q, scales)


@triton.jit
def _reduce_half351(value, AXIS: tl.constexpr):
    bits = value.to(tl.int32, bitcast=True) & 2147483647
    reduced347 = tl.reduce(
        bits.to(tl.float32, bitcast=True), AXIS, _maximum_nan321
    )
    positive347 = reduced347.to(tl.int32, bitcast=True) & 2147483647
    return positive347.to(tl.float32, bitcast=True)


def _type460(dtype):
    if dtype == torch.float32:
        return tl.float32
    if dtype == torch.bfloat16:
        return tl.bfloat16
    if dtype == torch.float16:
        return tl.float16
    if dtype == torch.float64:
        return tl.float64
    return None


@triton.jit
def _cast_pointers460(X, Q, S, U, META: tl.constexpr):
    if META[3][0]:
        X = tl.multiple_of(X, 16)
    if META[3][1]:
        Q = tl.multiple_of(Q, 16)
    if META[3][2]:
        S = tl.multiple_of(S, 16)
    if META[3][3]:
        U = tl.multiple_of(U, 16)
    return (X, Q, S, U)


@triton.jit(do_not_specialize=["X", "Q", "S", "U"])
def _quant_pointer460(
    X: "*bf16", Q: "*i8", S: "*fp32", U: "*bf16", META: tl.constexpr
):
    X, Q, S, U = _cast_pointers460(X, Q, S, U, META)
    _quant(X, Q, S, U, META)


@triton.jit(do_not_specialize=["X", "Q", "S", "U"])
def _quant_half_pointer460(
    X: "*bf16", Q: "*i8", S: "*fp32", U: "*bf16", META: tl.constexpr
):
    X, Q, S, U = _cast_pointers460(X, Q, S, U, META)
    _quant_half(X, Q, S, U, META)


@triton.jit(do_not_specialize=["X", "Q", "S", "U"])
def _resident_split168_pointer460(
    X: "*bf16", Q: "*i8", S: "*fp32", U: "*bf16", META: tl.constexpr
):
    X, Q, S, U = _cast_pointers460(X, Q, S, U, META)
    _resident_split168(X, Q, S, U, META)


@triton.jit(do_not_specialize=["X", "Q", "S", "U"])
def _fused_split_pointer460(
    X: "*bf16", Q: "*i8", S: "*fp32", U: "*bf16", META: tl.constexpr
):
    X, Q, S, U = _cast_pointers460(X, Q, S, U, META)
    _fused_split(X, Q, S, U, META)


@triton.jit
def _decode_pointers536(X, Q, S, U, META: tl.constexpr):
    X = X.to(tl.uint64, bitcast=True).to(tl.pointer_type(tl.bfloat16))
    Q = Q.to(tl.uint64, bitcast=True).to(tl.pointer_type(tl.int8))
    S = S.to(tl.uint64, bitcast=True).to(tl.pointer_type(tl.float32))
    U = U.to(tl.uint64, bitcast=True).to(tl.pointer_type(tl.bfloat16))
    return _cast_pointers460(X, Q, S, U, META)


@triton.jit(do_not_specialize=["X", "Q", "S", "U"])
def _quant_carrier536(
    X: tl.float64,
    Q: tl.float64,
    S: tl.float64,
    U: tl.float64,
    META: tl.constexpr,
):
    X, Q, S, U = _decode_pointers536(X, Q, S, U, META)
    _quant(X, Q, S, U, META)


@triton.jit(do_not_specialize=["X", "Q", "S", "U"])
def _quant_half_carrier536(
    X: tl.float64,
    Q: tl.float64,
    S: tl.float64,
    U: tl.float64,
    META: tl.constexpr,
):
    X, Q, S, U = _decode_pointers536(X, Q, S, U, META)
    _quant_half(X, Q, S, U, META)


@triton.jit(do_not_specialize=["X", "Q", "S", "U"])
def _resident_split168_carrier536(
    X: tl.float64,
    Q: tl.float64,
    S: tl.float64,
    U: tl.float64,
    META: tl.constexpr,
):
    X, Q, S, U = _decode_pointers536(X, Q, S, U, META)
    _resident_split168(X, Q, S, U, META)


@triton.jit(do_not_specialize=["X", "Q", "S", "U"])
def _fused_split_carrier536(
    X: tl.float64,
    Q: tl.float64,
    S: tl.float64,
    U: tl.float64,
    META: tl.constexpr,
):
    X, Q, S, U = _decode_pointers536(X, Q, S, U, META)
    _fused_split(X, Q, S, U, META)


__all__ = ["per_token_quant_int8_sum"]
