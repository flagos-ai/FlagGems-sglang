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
    ai = a.to(tl.int32, bitcast=True)
    bi = b.to(tl.int32, bitcast=True)
    return tl.maximum(ai, bi).to(tl.float32, bitcast=True)


@triton.jit
def _reduce_bits189(value, AXIS: tl.constexpr):
    bits = value.to(tl.int32, bitcast=True)
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
def _store_float(P, i, value, mask):
    if P.dtype.element_ty == tl.bfloat16:
        bits = value.to(tl.float32).to(tl.uint32, bitcast=True)
        rounded = bits + 32767 + (bits >> 16 & 1) >> 16
        nan = (bits & 2139095040 == 2139095040) & (bits & 8388607 != 0)
        encoded = tl.where(nan, bits >> 16 | 64, rounded).to(tl.uint16)
        tl.store(P.to(tl.pointer_type(tl.uint16)) + i, encoded, mask)
    else:
        tl.store(P + i, value, mask)


@triton.jit(do_not_specialize=["X", "Q", "S", "U"])
def _quant(X, Q, S, U, META: tl.constexpr):
    N: tl.constexpr = META[0]
    SUM: tl.constexpr = META[1]
    WIDE: tl.constexpr = META[2]
    B: tl.constexpr = triton.next_power_of_2(N)
    row = _wide_index(tl.program_id(0), WIDE)
    c = tl.arange(0, B)
    if N == B:
        x = tl.load(X + row * N + c).to(tl.float32)
    else:
        x = tl.load(X + row * N + c, c < N, 0).to(tl.float32)
    a = _clamp_min189(_reduce_bits189(tl.abs(x), 0))
    v = _normalize_finite194(x, a)
    q = tl.floor(tl.abs(v) + 0.5)
    q = tl.where(v < 0, -q, q).to(tl.int8)
    tl.store(Q + row * N + c, q, c < N)
    _store_float(S, row, a / 127.0, True)
    if SUM:
        _store_float(U, row, tl.sum(x, 0), True)


def official(x, scale_dtype, cal_sum):
    n = x.shape[-1]
    m = x.numel() // n
    q = x.new_empty(x.shape, dtype=torch.int8)
    scales = x.new_empty((*x.shape[:-1], 1), dtype=scale_dtype)
    sums = x.new_empty(x.shape[:-1]) if cal_sum else scales
    if m:
        _quant.run(
            x,
            q,
            scales,
            sums,
            (n, cal_sum, m * n > 2147483647),
            enable_fp_fusion=False,
            grid=(m,),
            warmup=False,
        )
    if cal_sum:
        return (q, scales, sums)
    return (q, scales)


@triton.jit
def _max_sum_combine(a_max, a_sum, b_max, b_sum):
    return (tl.maximum(a_max, b_max), a_sum + b_sum)


@triton.jit(do_not_specialize=["X", "Q", "S", "SUM"])
def _rows_kernel(X, Q, S, SUM, META: tl.constexpr):
    M: tl.constexpr = META[0]
    N: tl.constexpr = META[1]
    R: tl.constexpr = META[2]
    CAL_SUM: tl.constexpr = META[3]
    B: tl.constexpr = triton.next_power_of_2(N)
    WIDE: tl.constexpr = M * N > 2147483647
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
    if CAL_SUM:
        absmax_bits, total = tl.reduce(
            (tl.abs(x).to(tl.int32, bitcast=True), x), 1, _max_sum_combine
        )
        absmax = absmax_bits.to(tl.float32, bitcast=True)
    else:
        absmax = _reduce_bits189(tl.abs(x), 1)
    amax = _clamp_min189(absmax)
    y = _normalize_finite194(x, amax[:, None])
    q = tl.where(y < 0, y - 0.5, y + 0.5).to(tl.int32).to(tl.int8)
    if M % R == 0 and N == B:
        tl.store(Q + rows[:, None] * N + cols[None, :], q.to(tl.int8))
    else:
        tl.store(
            Q + rows[:, None] * N + cols[None, :],
            q.to(tl.int8),
            (rows[:, None] < M) & (cols[None, :] < N),
        )
    if M % R == 0:
        _store_float(S, rows, amax / 127.0, True)
    else:
        _store_float(S, rows, amax / 127.0, rows < M)
    if CAL_SUM:
        if M % R == 0:
            _store_float(SUM, rows, total, True)
        else:
            _store_float(SUM, rows, total, rows < M)


def rows_path(x, scale_dtype, cal_sum):
    n = x.shape[-1]
    rows = x.numel() // n
    if (
        n > 8192
        or rows < 32
        or (cal_sum and x.dtype not in (torch.float16, torch.bfloat16))
    ):
        return official(x, scale_dtype, cal_sum)
    block = triton.next_power_of_2(n)
    if (
        not cal_sum
        and 4096 <= n <= 8192
        and (n == block)
        and (rows >= 1024)
        and (rows % 8 == 0)
        and (triton.cdiv(rows, max(1, min(16, 8192 // block))) <= 32768)
    ):
        r = 8
    elif n <= 256 and rows >= 1024:
        r = min(64, 16384 // block)
    elif 1024 < n <= 2048:
        r = min(16, 32768 // block)
    else:
        r = max(1, min(16, 8192 // block))
    if triton.cdiv(rows, r) > 32768:
        return official(x, scale_dtype, cal_sum)
    q = x.new_empty(x.shape, dtype=torch.int8)
    scales = x.new_empty(x.shape[:-1] + (1,), dtype=scale_dtype)
    sums = x.new_empty(x.shape[:-1]) if cal_sum else scales
    _rows_kernel.run(
        x,
        q,
        scales,
        sums,
        (rows, n, r, cal_sum),
        num_warps=4,
        num_stages=1,
        enable_fp_fusion=False,
        grid=(triton.cdiv(rows, r),),
        warmup=False,
        groups_per_cluster=(
            8
            if not cal_sum
            and 4096 <= n <= 8192
            and (n == block)
            and (rows >= 1024)
            and (rows % 8 == 0)
            and (triton.cdiv(rows, max(1, min(16, 8192 // block))) <= 32768)
            else 1
        ),
        unroll_num=(
            4
            if not cal_sum
            and 4096 <= n <= 8192
            and (n == block)
            and (rows >= 1024)
            and (rows % 8 == 0)
            and (triton.cdiv(rows, max(1, min(16, 8192 // block))) <= 32768)
            else 2
        ),
    )
    if cal_sum:
        return (q, scales, sums)
    return (q, scales)


@triton.jit
def _stats_row(
    X,
    A,
    S,
    U,
    N: tl.constexpr,
    B: tl.constexpr,
    SUM: tl.constexpr,
    WIDE: tl.constexpr,
):
    row = _wide_index(tl.program_id(0), WIDE)
    c = tl.arange(0, B)
    if N == B:
        x = tl.load(X + row * N + c).to(tl.float32)
    else:
        x = tl.load(X + row * N + tl.minimum(c, N - 1)).to(tl.float32)
        x = tl.where(c < N, x, 0.0)
    a = _clamp_min189(_reduce_bits189(tl.abs(x), 0))
    tl.store(A + row, a)
    _store_float(S, row, a / 127.0, True)
    if SUM:
        _store_float(U, row, tl.sum(x, 0), True)


@triton.jit
def _quant_blocks(
    X, A, Q, N: tl.constexpr, B: tl.constexpr, WIDE: tl.constexpr
):
    row = _wide_index(tl.program_id(0), WIDE)
    col = _wide_index(tl.program_id(1), WIDE) * B + tl.arange(0, B)
    a = tl.load(A + row)
    x = tl.load(X + row * N + col, col < N, 0).to(tl.float32)
    v = _normalize_finite194(x, a)
    q = tl.floor(tl.abs(v) + 0.5)
    q = tl.where(v < 0, -q, q).to(tl.int8)
    tl.store(Q + row * N + col, q, col < N)


@triton.jit
def _quant_loop(X, A, Q, N: tl.constexpr, B: tl.constexpr, WIDE: tl.constexpr):
    row = _wide_index(tl.program_id(0), WIDE)
    c = tl.arange(0, B)
    a = tl.load(A + row)
    for start in range(tl.cdiv(N, B)):
        col = _wide_index(start, WIDE) * B + c
        x = tl.load(X + row * N + col, col < N, 0).to(tl.float32)
        v = _normalize_finite194(x, a)
        q = tl.where(v < 0, v - 0.5, v + 0.5).to(tl.int32).to(tl.int8)
        tl.store(Q + row * N + col, q, col < N)


def wide_path(x, scale_dtype, cal_sum):
    n = x.shape[-1]
    m = x.numel() // n
    q = x.new_empty(x.shape, dtype=torch.int8)
    scales = x.new_empty(x.shape[:-1] + (1,), dtype=scale_dtype)
    sums = x.new_empty(x.shape[:-1]) if cal_sum else scales
    a = x.new_empty((m,), dtype=torch.float32)
    if m:
        _stats_row.run(
            x,
            a,
            scales,
            sums,
            n,
            triton.next_power_of_2(n),
            cal_sum,
            enable_fp_fusion=False,
            grid=(m,),
            warmup=False,
            WIDE=m * n > 2147483647,
        )
        _quant_loop.run(
            x,
            a,
            q,
            n,
            8192,
            enable_fp_fusion=False,
            grid=(m,),
            warmup=False,
            WIDE=m * n > 2147483647,
        )
    return (q, scales, sums) if cal_sum else (q, scales)


def original(x, scale_dtype, cal_sum):
    if x.shape[-1] > 8192:
        return wide_path(x, scale_dtype, cal_sum)
    return rows_path(x, scale_dtype, cal_sum)


@triton.jit
def _full_stats(
    X,
    P,
    U,
    N: tl.constexpr,
    C: tl.constexpr,
    B: tl.constexpr,
    SUM: tl.constexpr,
    WIDE: tl.constexpr,
):
    row = _wide_index(tl.program_id(0), WIDE)
    part = _wide_index(tl.program_id(1), WIDE)
    col = part * B + tl.arange(0, B)
    x = tl.load(X + row * N + col).to(tl.float32)
    tl.store(P + row * C + part, _reduce_bits189(tl.abs(x), 0))
    if SUM:
        tl.store(U + row * C + part, tl.sum(x, 0))


@triton.jit
def _tail_stats(
    X,
    P,
    U,
    N: tl.constexpr,
    C: tl.constexpr,
    F: tl.constexpr,
    T: tl.constexpr,
    B: tl.constexpr,
    SUM: tl.constexpr,
    WIDE: tl.constexpr,
):
    row = _wide_index(tl.program_id(0), WIDE)
    c = tl.arange(0, B)
    x = tl.load(X + row * N + F * 8192 + c, c < T, 0).to(tl.float32)
    tl.store(P + row * C + F, _reduce_bits189(tl.abs(x), 0))
    if SUM:
        tl.store(U + row * C + F, tl.sum(x, 0))


@triton.jit
def _combine_parts(
    P,
    U,
    A,
    S,
    SUMS,
    C: tl.constexpr,
    CB: tl.constexpr,
    SUM: tl.constexpr,
    WIDE: tl.constexpr,
):
    row = _wide_index(tl.program_id(0), WIDE)
    ci = tl.arange(0, CB)
    maxima = tl.load(P + row * C + ci, ci < C, 0)
    a = _clamp_min189(_reduce_bits189(maxima, 0))
    tl.store(A + row, a)
    _store_float(S, row, a / 127.0, True)
    if SUM:
        values = tl.load(U + row * C + ci, ci < C, 0)
        _store_float(SUMS, row, tl.sum(values, 0), True)


def selected(x, scale_dtype, cal_sum):
    n = x.shape[-1]
    if n <= 8192 or n & n - 1 == 0 or (cal_sum and x.dtype == torch.float32):
        return original(x, scale_dtype, cal_sum)
    m = x.numel() // n
    full = n // 8192
    tail = n % 8192
    c = full + (tail != 0)
    q = x.new_empty(x.shape, dtype=torch.int8)
    scales = x.new_empty(x.shape[:-1] + (1,), dtype=scale_dtype)
    sums = x.new_empty(x.shape[:-1]) if cal_sum else scales
    maxima = x.new_empty((m, c), dtype=torch.float32)
    partial_sums = (
        x.new_empty((m, c), dtype=torch.float32) if cal_sum else maxima
    )
    amax = x.new_empty((m,), dtype=torch.float32)
    if m:
        _full_stats.run(
            x,
            maxima,
            partial_sums,
            n,
            c,
            8192,
            cal_sum,
            enable_fp_fusion=False,
            grid=(m, full),
            warmup=False,
            WIDE=m * n > 2147483647,
        )
        if tail:
            _tail_stats.run(
                x,
                maxima,
                partial_sums,
                n,
                c,
                full,
                tail,
                triton.next_power_of_2(tail),
                cal_sum,
                enable_fp_fusion=False,
                grid=(m,),
                warmup=False,
                WIDE=m * n > 2147483647,
            )
        _combine_parts.run(
            maxima,
            partial_sums,
            amax,
            scales,
            sums,
            c,
            triton.next_power_of_2(c),
            cal_sum,
            enable_fp_fusion=False,
            grid=(m,),
            warmup=False,
            WIDE=m * n > 2147483647,
        )
        _quant_loop.run(
            x,
            amax,
            q,
            n,
            8192,
            enable_fp_fusion=False,
            grid=(m,),
            warmup=False,
            WIDE=m * n > 2147483647,
        )
    return (q, scales, sums) if cal_sum else (q, scales)


def _ceil_div(x, y):
    return (x + y - 1) // y


def per_token_quant_int8_sum(x, scale_dtype, cal_sum):
    shape = x.shape
    n = shape[-1]
    if n > 8192:
        return selected(x, scale_dtype, cal_sum)
    row_shape = shape[:-1]
    m = row_shape.numel()
    q = x.new_empty(shape, dtype=torch.int8)
    scales = x.new_empty(row_shape + (1,), dtype=scale_dtype)
    sums = x.new_empty(row_shape) if cal_sum else scales
    if m == 0:
        return (q, scales, sums) if cal_sum else (q, scales)
    b = 1 << (n - 1).bit_length()
    grouped = (
        (not cal_sum or x.dtype in (torch.float16, torch.bfloat16))
        and (2048 if cal_sum else 1024) <= n <= 8192
        and (n == b)
        and (m >= 1024)
        and (m % 8 == 0)
        and (
            _ceil_div(m, 8 if n == 2048 else max(1, min(16, 8192 // b)))
            <= 32768
        )
    )
    if grouped:
        r = 8
    elif n <= 256 and m >= 1024:
        r = min(64, 16384 // b)
    elif 1024 < n <= 2048:
        r = min(16, 32768 // b)
    else:
        r = max(1, min(16, 8192 // b))
    batched = m >= 32 and (
        not cal_sum or x.dtype in (torch.float16, torch.bfloat16)
    )
    programs = _ceil_div(m, r) if batched else 0
    batched = batched and programs <= 32768
    if batched:
        _rows_kernel.run(
            x,
            q,
            scales,
            sums,
            (m, n, r, cal_sum),
            grid=(programs,),
            warmup=False,
            num_warps=4,
            num_stages=1,
            enable_fp_fusion=False,
            groups_per_cluster=8 if grouped else 1,
            unroll_num=(
                8
                if grouped and cal_sum
                else 4 if grouped or (cal_sum and n > 512) else 2
            ),
        )
    else:
        _quant.run(
            x,
            q,
            scales,
            sums,
            (n, cal_sum, m * n > 2147483647),
            grid=(m,),
            warmup=False,
            enable_fp_fusion=False,
        )
    return (q, scales, sums) if cal_sum else (q, scales)


__all__ = ["per_token_quant_int8_sum"]
