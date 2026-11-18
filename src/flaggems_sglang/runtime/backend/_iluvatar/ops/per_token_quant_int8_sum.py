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
def _quant(X, Q, S, U, N: tl.constexpr, WIDE: tl.constexpr):
    SUM: tl.constexpr = U is not None
    B: tl.constexpr = triton.next_power_of_2(N)
    row = _wide_index(tl.program_id(0), WIDE)
    c = tl.arange(0, B)
    x = tl.load(X + row * N + c, c < N, 0).to(tl.float32)
    a = _clamp_min189(_reduce_bits189(tl.abs(x), 0))
    v = _normalize_finite194(x, a)
    q = tl.where(v < 0, v - 0.5, v + 0.5).to(tl.int32).to(tl.int8)
    tl.store(Q + row * N + c, q, c < N)
    tl.store(S + row, a / 127.0)
    if SUM:
        tl.store(U + row, tl.sum(x, 0))


@triton.jit
def _max_sum_combine(a_max, a_sum, b_max, b_sum):
    return (tl.maximum(a_max, b_max), a_sum + b_sum)


@triton.jit
def _quant_tuple(X, Q, S, U, N: tl.constexpr, WIDE: tl.constexpr):
    SUM: tl.constexpr = U is not None
    B: tl.constexpr = triton.next_power_of_2(N)
    row = _wide_index(tl.program_id(0), WIDE)
    c = tl.arange(0, B)
    x = tl.load(X + row * N + c, c < N, 0).to(tl.float32)
    if SUM:
        a_bits196, total = tl.reduce(
            (tl.abs(x).to(tl.int32, bitcast=True), x), 0, _max_sum_combine
        )
        a = a_bits196.to(tl.float32, bitcast=True)
    else:
        a = _reduce_bits189(tl.abs(x), 0)
    a = _clamp_min189(a)
    v = _normalize_finite194(x, a)
    q = tl.where(v < 0, v - 0.5, v + 0.5).to(tl.int32).to(tl.int8)
    tl.store(Q + row * N + c, q, c < N)
    tl.store(S + row, a / 127.0)
    if SUM:
        tl.store(U + row, total)


@triton.jit
def _rows_kernel(
    X,
    Q,
    S,
    SUM,
    M: tl.constexpr,
    N: tl.constexpr,
    B: tl.constexpr,
    R: tl.constexpr,
    CAL_SUM: tl.constexpr,
    WIDE: tl.constexpr,
):
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
    q = x.new_empty(shape, dtype=torch.int8)
    scales = x.new_empty(row_shape + (1,), dtype=scale_dtype)
    sums = x.new_empty(row_shape) if cal_sum else None
    if m:
        if rows:
            _rows_kernel.run(
                x,
                q,
                scales,
                sums,
                m,
                n,
                b,
                r,
                cal_sum,
                grid=(triton.cdiv(m, r),),
                warmup=False,
                num_warps=4,
                num_stages=1,
                enable_fp_fusion=False,
                WIDE=m * n > 2147483647,
            )
        else:
            low_sum = not cal_sum or x.dtype in (torch.float16, torch.bfloat16)
            kernel = (
                _quant_tuple if low_sum and (n > 1024 or cal_sum) else _quant
            )
            if low_sum:
                warps = 4 if n <= 1024 else 8 if n <= 2048 else 16
            else:
                warps = 4 if n <= 2048 else 8
            kernel.run(
                x,
                q,
                scales,
                sums,
                n,
                grid=(m,),
                warmup=False,
                num_warps=warps,
                enable_fp_fusion=False,
                WIDE=m * n > 2147483647,
            )
    return (q, scales, sums) if cal_sum else (q, scales)


__all__ = ["per_token_quant_int8_sum"]
