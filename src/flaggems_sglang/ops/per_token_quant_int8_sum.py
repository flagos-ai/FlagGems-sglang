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


@triton.jit
def _normalize_finite194(x, a):
    finite = a.to(tl.int32, bitcast=True) < 2139095040
    return tl.where(finite, x * (tl.div_rn(1.0, a) * 127.0), 0.0)


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
    N: tl.constexpr = META[0]
    SUM: tl.constexpr = U is not None
    WIDE: tl.constexpr = META[1]
    B: tl.constexpr = triton.next_power_of_2(N)
    row = _wide_index(tl.program_id(0), WIDE)
    c = tl.arange(0, B)
    x = tl.load(X + row * N + c, c < N, 0).to(tl.float32)
    a = _clamp_min189(_reduce_bits189(tl.abs(x), 0))
    v = _normalize_finite194(x, a)
    q = tl.where(v < 0, v - 0.5, v + 0.5).to(tl.int32).to(tl.int8)
    tl.store(Q + row * N + c, q, c < N, cache_modifier=".cs")
    tl.store(S + row, a / 127.0)
    if SUM:
        tl.store(U + row, tl.sum(x, 0))


@triton.jit
def _max_sum_state384(left, right):
    left_max = (left >> 32).to(tl.int32)
    right_max = (right >> 32).to(tl.int32)
    left_sum = left.to(tl.uint32).to(tl.float32, bitcast=True)
    right_sum = right.to(tl.uint32).to(tl.float32, bitcast=True)
    maximum = tl.maximum(left_max, right_max)
    total = left_sum + right_sum
    return maximum.to(tl.uint64) << 32 | total.to(tl.uint32, bitcast=True).to(
        tl.uint64
    )


@triton.jit
def _max_sum_plain164(a_max, a_sum, b_max, b_sum):
    return tl.maximum(a_max, b_max), a_sum + b_sum


@triton.jit
def _quant_scalar411(
    X: tl.float64,
    Q: tl.float64,
    S: tl.float64,
    U: tl.float64,
    META: tl.constexpr,
):
    X = X.to(tl.uint64, bitcast=True).to(
        tl.pointer_type(
            (tl.float32, tl.bfloat16, tl.float16, tl.float64)[META[3]]
        )
    )
    Q = Q.to(tl.uint64, bitcast=True).to(tl.pointer_type(tl.int8))
    S = S.to(tl.uint64, bitcast=True).to(
        tl.pointer_type(
            (tl.float32, tl.bfloat16, tl.float16, tl.float64)[META[2]]
        )
    )
    if META[-2]:
        U = U.to(tl.uint64, bitcast=True).to(
            tl.pointer_type(X.dtype.element_ty)
        )
    else:
        U = None
    if META[-1][0]:
        Q = tl.multiple_of(Q, 16)
    if META[-1][1]:
        S = tl.multiple_of(S, 16)
    if U is not None:
        if META[-1][2]:
            U = tl.multiple_of(U, 16)
    if META[-1][3]:
        X = tl.multiple_of(X, 16)
    N: tl.constexpr = META[0]
    SUM: tl.constexpr = U is not None
    WIDE: tl.constexpr = META[1]
    B: tl.constexpr = triton.next_power_of_2(N)
    row = _wide_index(tl.program_id(0), WIDE)
    c = tl.arange(0, B)
    x = tl.load(X + row * N + c, c < N, 0).to(tl.float32)
    PACKED: tl.constexpr = (
        torch.version.hip is not None
        and SUM
        and (N > 2048)
        and (N <= 4096)
        and (
            X.dtype.element_ty == tl.float16
            or X.dtype.element_ty == tl.bfloat16
        )
    )
    if PACKED:
        magnitude = tl.abs(x).to(tl.int32, bitcast=True).to(tl.uint64)
        value_bits = x.to(tl.uint32, bitcast=True).to(tl.uint64)
        state = tl.reduce(magnitude << 32 | value_bits, 0, _max_sum_state384)
        a = _clamp_min189(
            (state >> 32).to(tl.int32).to(tl.float32, bitcast=True)
        )
        total = state.to(tl.uint32).to(tl.float32, bitcast=True)
    else:
        if (
            torch.version.hip is not None
            and SUM
            and N == 8192
            and X.dtype.element_ty == tl.bfloat16
        ):
            magnitude = tl.abs(x).to(tl.int32, bitcast=True) & 2147483647
            maximum, total = tl.reduce((magnitude, x), 0, _max_sum_plain164)
            a = _clamp_min189(maximum.to(tl.float32, bitcast=True))
        else:
            a = _clamp_min189(_reduce_bits189(tl.abs(x), 0))
    v = _normalize_finite194(x, a)
    q = tl.where(v < 0, v - 0.5, v + 0.5).to(tl.int32).to(tl.int8)
    tl.store(Q + row * N + c, q, c < N, cache_modifier=".cs")
    tl.store(S + row, a / 127.0)
    if SUM:
        if PACKED:
            tl.store(U + row, total)
        else:
            if (
                torch.version.hip is not None
                and N == 8192
                and X.dtype.element_ty == tl.bfloat16
            ):
                tl.store(U + row, total)
            else:
                tl.store(U + row, tl.sum(x, 0))


@triton.jit
def _rows_kernel(X, Q, S, SUM, META: tl.constexpr):
    M: tl.constexpr = META[0]
    N: tl.constexpr = META[1]
    B: tl.constexpr = META[2]
    R: tl.constexpr = META[3]
    CAL_SUM: tl.constexpr = SUM is not None
    WIDE: tl.constexpr = META[4]
    rows = _wide_index(tl.program_id(0), WIDE) * R + tl.arange(0, R)
    cols = tl.arange(0, B)
    if M % R == 0 and N == B:
        x = tl.load(X + rows[:, None] * N + cols[None, :]).to(tl.float32)
    else:
        x = tl.load(
            X + rows[:, None] * N + cols[None, :],
            (rows[:, None] < M) & (cols[None, :] < N),
            0,
        ).to(tl.float32)
    amax = _clamp_min189(_reduce_bits189(tl.abs(x), 1))
    y = _normalize_finite194(x, amax[:, None])
    q = tl.where(y < 0, -1.0, 1.0) * tl.floor(tl.abs(y) + 0.5)
    if M % R == 0 and N == B:
        tl.store(Q + rows[:, None] * N + cols[None, :], q.to(tl.int8))
    else:
        tl.store(
            Q + rows[:, None] * N + cols[None, :],
            q.to(tl.int8),
            (rows[:, None] < M) & (cols[None, :] < N),
        )
    tl.store(S + rows, amax / 127.0, rows < M)
    if CAL_SUM:
        tl.store(SUM + rows, tl.sum(x, 1), rows < M)


@triton.jit
def _sum_order(X, U, META: tl.constexpr):
    N: tl.constexpr = META[0]
    TX: tl.constexpr = META[1]
    TY: tl.constexpr = META[2]
    row = tl.program_id(0).to(tl.int64)
    t = tl.arange(0, TX * TY)
    v0 = tl.full((TX * TY,), 0, tl.float32)
    v1 = tl.full((TX * TY,), 0, tl.float32)
    v2 = tl.full((TX * TY,), 0, tl.float32)
    v3 = tl.full((TX * TY,), 0, tl.float32)
    for k in range(tl.cdiv(N, TX * TY * 4)):
        sum_c = (k * TX * TY + t) * 4
        v0 += tl.load(X + row * N + sum_c, sum_c < N, 0).to(tl.float32)
        v1 += tl.load(X + row * N + sum_c + 1, sum_c + 1 < N, 0).to(tl.float32)
        v2 += tl.load(X + row * N + sum_c + 2, sum_c + 2 < N, 0).to(tl.float32)
        v3 += tl.load(X + row * N + sum_c + 3, sum_c + 3 < N, 0).to(tl.float32)
    value = v0 + v1 + v2 + v3
    for log_offset in tl.static_range(3, -1, -1):
        offset = 1 << log_offset
        if TY > offset:
            other = tl.gather(
                value, tl.minimum(t + offset * TX, TX * TY - 1), 0
            )
            value = tl.where(t // TX < offset, value + other, value)
    for log_offset in tl.static_range(8, 4, -1):
        offset = 1 << log_offset
        if TX > offset:
            other = tl.gather(value, tl.minimum(t + offset, TX * TY - 1), 0)
            value = tl.where(t % TX < offset, value + other, value)
    for log_offset in tl.static_range(0, 5):
        offset = 1 << log_offset
        other = tl.gather(value, tl.minimum(t + offset, TX * TY - 1), 0)
        value = tl.where(t % TX < 32, value + other, value)
    total = tl.sum(tl.where(t == 0, value, 0.0), 0)
    tl.store(U + row, total)


@triton.jit
def _quant_ordered(X, Q, S, U, META: tl.constexpr):
    N: tl.constexpr = META[0]
    TX: tl.constexpr = META[1]
    TY: tl.constexpr = META[2]
    B: tl.constexpr = triton.next_power_of_2(N)
    row = tl.program_id(0).to(tl.int64)
    c = tl.arange(0, B)
    x = tl.load(X + row * N + c, c < N, 0).to(tl.float32)
    a = _clamp_min189(_reduce_bits189(tl.abs(x), 0))
    v = _normalize_finite194(x, a)
    q = tl.where(v < 0, v - 0.5, v + 0.5).to(tl.int32).to(tl.int8)
    tl.store(Q + row * N + c, q, c < N)
    tl.store(S + row, a / 127.0)
    t = tl.arange(0, TX * TY)
    v0 = tl.full((TX * TY,), 0, tl.float32)
    v1 = tl.full((TX * TY,), 0, tl.float32)
    v2 = tl.full((TX * TY,), 0, tl.float32)
    v3 = tl.full((TX * TY,), 0, tl.float32)
    for k in range(tl.cdiv(N, TX * TY * 4)):
        tile_c = k * TX * TY * 4 + tl.arange(0, TX * TY * 4)
        tile = tl.load(X + row * N + tile_c, tile_c < N, 0).to(tl.float32)
        even, odd = tl.split(tl.reshape(tile, (TX * TY, 2, 2)))
        tile0, tile2 = tl.split(even)
        tile1, tile3 = tl.split(odd)
        v0 += tile0
        v1 += tile1
        v2 += tile2
        v3 += tile3
    value = v0 + v1 + v2 + v3
    for log_offset in tl.static_range(3, -1, -1):
        offset = 1 << log_offset
        if TY > offset:
            other = tl.gather(
                value, tl.minimum(t + offset * TX, TX * TY - 1), 0
            )
            value = tl.where(t // TX < offset, value + other, value)
    for log_offset in tl.static_range(8, 4, -1):
        offset = 1 << log_offset
        if TX > offset:
            other = tl.gather(value, tl.minimum(t + offset, TX * TY - 1), 0)
            value = tl.where(t % TX < offset, value + other, value)
    for log_offset in tl.static_range(0, 5):
        offset = 1 << log_offset
        other = tl.gather(value, tl.minimum(t + offset, TX * TY - 1), 0)
        value = tl.where(t % TX < 32, value + other, value)
    total = tl.sum(tl.where(t == 0, value, 0.0), 0)
    tl.store(U + row, total)


def _type411(
    dtype,
    f32=torch.float32,
    bf16=torch.bfloat16,
    f16=torch.float16,
    f64=torch.float64,
):
    if dtype == f32:
        return 0
    if dtype == bf16:
        return 1
    if dtype == f16:
        return 2
    if dtype == f64:
        return 3
    return None


def per_token_quant_int8_sum(x, scale_dtype, cal_sum):
    shape = x.shape
    row_shape = shape[:-1]
    n = shape[-1]
    m = row_shape.numel()
    b = triton.next_power_of_2(n) if n <= 512 else 0
    r = min(16, 2048 // b) if n <= 512 else 1
    rows = (
        n <= 512
        and m >= 512
        and (not cal_sum or x.dtype in (torch.float16, torch.bfloat16))
        and (triton.cdiv(m, r) <= 32768)
    )
    ordered_sum = False
    tx = 32
    ty = 1
    if (
        m
        and cal_sum
        and (n > 128)
        and (n % 4 == 0)
        and (
            x.dtype == torch.float32
            and x.data_ptr() % 16 == 0
            or x.dtype == torch.float64
        )
    ):
        hip = torch.version.hip is not None
        dim0 = min(512, 1 << (n // 4).bit_length() - 1)
        dim1 = min(512, 1 << m.bit_length() - 1)
        tx = min(dim0, 32)
        bh = min(dim1, 512 // tx)
        tx = min(dim0, 512 // bh)
        ty = bh if triton.cdiv(n, tx) >= min(bh * 16, 256) else 1
        if hip and triton.cdiv(n, tx) < 1024:
            ty = 1
        ordered_sum = triton.cdiv(n, tx * ty) < 256 or (
            hip and ty == 1 and (triton.cdiv(n, tx) < 1024)
        )
    fused = ordered_sum and n <= 32768
    q = x.new_empty(shape, dtype=torch.int8)
    scales = x.new_empty(row_shape + (1,), dtype=scale_dtype)
    sums = x.new_empty(row_shape) if cal_sum else None
    if m:
        if fused:
            _quant_ordered.run(
                x,
                q,
                scales,
                sums,
                (n, tx, ty),
                grid=(m,),
                warmup=False,
                num_warps=8,
                enable_fp_fusion=False,
            )
        elif rows:
            _rows_kernel.run(
                x,
                q,
                scales,
                sums,
                (m, n, b, r, m * n > 2147483647),
                grid=(triton.cdiv(m, r),),
                warmup=False,
                num_warps=4,
                num_stages=1,
                enable_fp_fusion=False,
            )
        else:
            scale_type411 = _type411(scale_dtype)
            input_type411 = _type411(x.dtype)
            if scale_type411 is not None and input_type411 is not None:
                q_ptr412 = q.data_ptr()
                s_ptr412 = scales.data_ptr()
                u_ptr412 = (
                    None if ordered_sum or sums is None else sums.data_ptr()
                )
                x_ptr412 = x.data_ptr()
                x_arg528, q_arg528, s_arg528, u_arg528 = struct.unpack(
                    "=4d",
                    struct.pack(
                        "=4Q",
                        x_ptr412,
                        q_ptr412,
                        s_ptr412,
                        0 if u_ptr412 is None else u_ptr412,
                    ),
                )
                _quant_scalar411.run(
                    x_arg528,
                    q_arg528,
                    s_arg528,
                    u_arg528,
                    (
                        n,
                        m * n > 2147483647,
                        scale_type411,
                        input_type411,
                        u_ptr412 is not None,
                        (
                            q_ptr412 % 16 == 0,
                            s_ptr412 % 16 == 0,
                            u_ptr412 is not None and u_ptr412 % 16 == 0,
                            x_ptr412 % 16 == 0,
                        ),
                    ),
                    grid=(m,),
                    warmup=False,
                    num_warps=4 if n <= 2048 else 8,
                    enable_fp_fusion=False,
                )
            else:
                _quant.run(
                    x,
                    q,
                    scales,
                    None if ordered_sum else sums,
                    (n, m * n > 2147483647),
                    grid=(m,),
                    warmup=False,
                    num_warps=4 if n <= 2048 else 8,
                    enable_fp_fusion=False,
                )
    if ordered_sum and (not fused):
        _sum_order.run(
            x,
            sums,
            (n, tx, ty),
            grid=(m,),
            warmup=False,
            num_warps=4,
            enable_fp_fusion=False,
        )
    return (q, scales, sums) if cal_sum else (q, scales)


__all__ = ["per_token_quant_int8_sum"]
