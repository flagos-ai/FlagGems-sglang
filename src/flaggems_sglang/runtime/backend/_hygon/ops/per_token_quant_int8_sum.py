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
import triton.experimental.tle.language as tle
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
    WIDE: tl.constexpr = META[1]
    SUM: tl.constexpr = U is not None
    B: tl.constexpr = triton.next_power_of_2(N)
    row = _wide_index(tl.program_id(0), WIDE)
    c = tl.arange(0, B)
    x = tl.load(X + row * N + c, c < N, 0).to(tl.float32)
    a = _clamp_min189(_reduce_bits189(tl.abs(x), 0))
    finite286 = a.to(tl.int32, bitcast=True) < 2139095040
    if finite286:
        v = x * (tl.div_rn(1.0, a) * 127.0)
        q = (
            (
                v
                + (v.to(tl.uint32, bitcast=True) & 2147483648 | 1056964608).to(
                    tl.float32, bitcast=True
                )
            )
            .to(tl.int32)
            .to(tl.int8)
        )
        tl.store(Q + row * N + c, q, c < N)
    else:
        q = tl.full((B,), 0, tl.int8)
        tl.store(Q + row * N + c, q, c < N)
    tl.store(S + row, a / 127.0)
    if SUM:
        tl.store(U + row, tl.sum(x, 0))


@triton.jit
def _quant_scalar416(X, Q, S, U, META: tl.constexpr):
    X = X.to(tl.uint64).to(
        tl.pointer_type(
            (tl.float32, tl.bfloat16, tl.float16, tl.float64)[META[3]]
        )
    )
    Q = Q.to(tl.uint64).to(tl.pointer_type(tl.int8))
    S = S.to(tl.uint64).to(
        tl.pointer_type(
            (tl.float32, tl.bfloat16, tl.float16, tl.float64)[META[2]]
        )
    )
    if U is not None:
        U = U.to(tl.uint64).to(tl.pointer_type(X.dtype.element_ty))
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
    WIDE: tl.constexpr = META[1]
    SUM: tl.constexpr = U is not None
    B: tl.constexpr = triton.next_power_of_2(N)
    row = _wide_index(tl.program_id(0), WIDE)
    c = tl.arange(0, B)
    x = tl.load(X + row * N + c, c < N, 0).to(tl.float32)
    JOINT: tl.constexpr = (
        SUM
        and N == B
        and N > 2048
        and N <= 8192
        and not WIDE
        and META[2] == 0
        and META[3] == 1
        and META[-1][0]
        and META[-1][1]
        and META[-1][2]
        and META[-1][3]
    )
    if JOINT:
        wave_x = tl.reshape(x, (8, B // 8))
        wave_bits = tl.abs(wave_x).to(tl.int32, bitcast=True) & 2147483647
        wave_max = tl.max(wave_bits, 1)
        wave_sum = tl.sum(wave_x, 1)
        max_exchange = tle.gpu.alloc(
            [8],
            dtype=tl.int32,
            layout=None,
            scope=tle.gpu.smem,
            init_value=wave_max,
            nv_mma_shared_layout=False,
        )
        sum_exchange = tle.gpu.alloc(
            [8],
            dtype=tl.float32,
            layout=None,
            scope=tle.gpu.smem,
            init_value=wave_sum,
            nv_mma_shared_layout=False,
        )
        tl.debug_barrier()
        partial_indices = tl.arange(0, 8)
        max_partials = tl.load(
            tle.gpu.local_ptr(max_exchange, (partial_indices,))
        )
        a = _clamp_min189(tl.max(max_partials, 0).to(tl.float32, bitcast=True))
    else:
        a = _clamp_min189(_reduce_bits189(tl.abs(x), 0))
    finite286 = a.to(tl.int32, bitcast=True) < 2139095040
    if finite286:
        v = x * (tl.div_rn(1.0, a) * 127.0)
        q = (
            (
                v
                + (v.to(tl.uint32, bitcast=True) & 2147483648 | 1056964608).to(
                    tl.float32, bitcast=True
                )
            )
            .to(tl.int32)
            .to(tl.int8)
        )
        tl.store(Q + row * N + c, q, c < N)
    else:
        q = tl.full((B,), 0, tl.int8)
        tl.store(Q + row * N + c, q, c < N)
    tl.store(S + row, a / 127.0)
    if SUM:
        if JOINT:
            sum_partials = tl.load(
                tle.gpu.local_ptr(sum_exchange, (partial_indices,))
            )
            tl.store(U + row, tl.sum(sum_partials, 0))
        else:
            tl.store(U + row, tl.sum(x, 0))


@triton.jit
def _rows_kernel(X, Q, S, SUM, META: tl.constexpr):
    M: tl.constexpr = META[0]
    N: tl.constexpr = META[1]
    B: tl.constexpr = META[2]
    R: tl.constexpr = META[3]
    CAL_SUM: tl.constexpr = META[4]
    WIDE: tl.constexpr = META[5]
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
                (m, n, b, r, cal_sum, m * n > 2147483647),
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
                u_ptr412 = None if sums is None else sums.data_ptr()
                x_ptr412 = x.data_ptr()
                _quant_scalar416.run(
                    x_ptr412,
                    q_ptr412,
                    s_ptr412,
                    u_ptr412,
                    (
                        n,
                        m * n > 2147483647,
                        scale_type411,
                        input_type411,
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
                    sums,
                    (n, m * n > 2147483647),
                    grid=(m,),
                    warmup=False,
                    num_warps=4 if n <= 2048 else 8,
                    enable_fp_fusion=False,
                )
    return (q, scales, sums) if cal_sum else (q, scales)


__all__ = ["per_token_quant_int8_sum"]
