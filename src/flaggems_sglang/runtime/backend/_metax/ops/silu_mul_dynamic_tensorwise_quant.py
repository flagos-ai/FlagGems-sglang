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
from triton.language.extra import libdevice


@triton.jit
def _scale_divide448207(a):
    if a < float("inf"):
        reciprocal = 0.0022321429569274187
        quotient = a * reciprocal
        negative = (quotient.to(tl.uint32, bitcast=True) ^ 2147483648).to(
            tl.float32, bitcast=True
        )
        residual = tl.fma(negative, 448.0, a)
        result = tl.fma(residual, reciprocal, quotient)
    else:
        result = a / 448.0
    return result


@triton.jit
def _parts207(
    index,
    D: tl.constexpr,
    T: tl.constexpr,
    K: tl.constexpr,
    B: tl.constexpr,
    I32: tl.constexpr,
):
    if (
        I32
        and D > 0
        and (T > 0)
        and (K > 0)
        and (B > 0)
        and (T * D + K * B + B <= 2147483647)
    ):
        row = (index.to(tl.uint32) // D).to(tl.int32)
        col = (index.to(tl.uint32) % D).to(tl.int32)
    else:
        row = index // D
        col = index % D
    return (row, col)


@triton.jit
def _inverse207(scale):
    if tl.abs(scale) < float("inf"):
        reciprocal = _reciprocal207(scale)
        residual = tl.fma(-reciprocal, scale, 1.0)
        result = tl.fma(residual, reciprocal, reciprocal)
    else:
        result = libdevice.div_rn(1.0, scale)
    return result


@triton.jit
def _normalize207(value, scale, inverse_scale):
    if tl.abs(scale) < float("inf"):
        quotient = value * inverse_scale
        residual = tl.fma(-quotient, scale, value)
        corrected = tl.fma(residual, inverse_scale, quotient)
        result = corrected
        sign_source = value
    else:
        result = libdevice.div_rn(value, scale)
        sign_source = result
    return (result, sign_source)


@tl.core.extern
def _reciprocal207(value, _semantic=None):
    return tl.core.extern_elementwise(
        "",
        "",
        [value],
        {(tl.float32,): ("llvm.mxc.rcp.f32", tl.float32)},
        is_pure=True,
        _semantic=_semantic,
    )


@triton.jit
def _encode(v, sign_source):
    raw = sign_source.to(tl.int32, bitcast=True)
    sign = raw >> 24 & 128
    a = tl.abs(v)
    mag = a.to(tl.int32, bitcast=True)
    exponent = tl.maximum(mag & 2139095040, 1015021568)
    bias = exponent + 167772160
    biased = a + bias.to(tl.float32, bitcast=True)
    code = biased.to(tl.int32, bitcast=True) - bias + (bias >> 20) - 1128
    code = tl.where(a <= 464.0, code, 127)
    return (code | sign).to(tl.uint8)


@triton.jit
def _activate(
    X,
    A,
    P,
    NT,
    D: tl.constexpr,
    K: tl.constexpr,
    SX: tl.constexpr,
    CX: tl.constexpr,
    B: tl.constexpr,
    BUFFER: tl.constexpr,
    T: tl.constexpr,
):
    n = tl.load(NT)
    lane = tl.arange(0, B)
    maximum = tl.full((B,), 0.0, tl.float32)
    if BUFFER:
        first = tl.program_id(0) * B + lane
        first_row, first_col = _parts207(
            first, D, T, K, B, NT.dtype.element_ty == tl.int32
        )
        current_gate = tl.load(
            X + first_row * SX + first_col * CX, first < n * D, other=0
        )
        current_up = tl.load(
            X + first_row * SX + (first_col + D) * CX, first < n * D, other=0
        )
        for start in range(tl.program_id(0) * B, n * D, K * B):
            next_index = start + K * B + lane
            next_row, next_col = _parts207(
                next_index, D, T, K, B, NT.dtype.element_ty == tl.int32
            )
            next_gate = tl.load(
                X + next_row * SX + next_col * CX, next_index < n * D, other=0
            )
            next_up = tl.load(
                X + next_row * SX + (next_col + D) * CX,
                next_index < n * D,
                other=0,
            )
            gate = current_gate.to(tl.float32)
            up = current_up.to(tl.float32)
            value = (
                gate
                * (
                    _reciprocal207(
                        tl.fma(
                            tl.exp(-gate),
                            2.3283064365386963e-10,
                            2.3283064365386963e-10,
                        )
                    )
                    * 2.3283064365386963e-10
                )
                * up
            )
            maximum = tl.maximum(maximum, tl.abs(value))
            index = start + lane
            tl.store(A + index, value, index < n * D)
            current_gate = next_gate
            current_up = next_up
    else:
        for start in range(tl.program_id(0) * B, n * D, K * B):
            index = start + lane
            row, col = _parts207(
                index, D, T, K, B, NT.dtype.element_ty == tl.int32
            )
            gate = tl.load(X + row * SX + col * CX, index < n * D, other=0).to(
                tl.float32
            )
            up = tl.load(
                X + row * SX + (col + D) * CX, index < n * D, other=0
            ).to(tl.float32)
            value = (
                gate
                * (
                    _reciprocal207(
                        tl.fma(
                            tl.exp(-gate),
                            2.3283064365386963e-10,
                            2.3283064365386963e-10,
                        )
                    )
                    * 2.3283064365386963e-10
                )
                * up
            )
            maximum = tl.maximum(maximum, tl.abs(value))
            if BUFFER:
                tl.store(A + index, value, index < n * D)
    tl.store(P + tl.program_id(0), tl.max(maximum, 0))


@triton.jit
def _quant(
    X,
    A,
    Q: tl.pointer_type(tl.uint8),
    P,
    S,
    NT,
    D: tl.constexpr,
    T: tl.constexpr,
    SO: tl.constexpr,
    CO: tl.constexpr,
    K: tl.constexpr,
    BK: tl.constexpr,
    B: tl.constexpr,
    SX: tl.constexpr,
    CX: tl.constexpr,
    BUFFER: tl.constexpr,
):
    n = tl.load(NT)
    r = tl.arange(0, BK)
    partial = tl.load(P + r, r < K, other=0)
    scale = _scale_divide448207(tl.maximum(tl.max(partial, 0), 1e-10))
    inverse_scale = _inverse207(scale)
    if tl.program_id(0) == 0:
        tl.store(S, scale)
    lane = tl.arange(0, B)
    if BUFFER:
        first = tl.program_id(0) * B + lane
        current_value = tl.load(A + first, first < n * D, other=0)
    for start in range(tl.program_id(0) * B, n * D, K * B):
        index = start + lane
        row, col = _parts207(
            index, D, T, K, B, NT.dtype.element_ty == tl.int32
        )
        if BUFFER:
            next_index = index + K * B
            next_value = tl.load(A + next_index, next_index < n * D, other=0)
            value = current_value
        else:
            gate = tl.load(X + row * SX + col * CX, index < n * D, other=0).to(
                tl.float32
            )
            up = tl.load(
                X + row * SX + (col + D) * CX, index < n * D, other=0
            ).to(tl.float32)
            value = (
                gate
                * (
                    _reciprocal207(
                        tl.fma(
                            tl.exp(-gate),
                            2.3283064365386963e-10,
                            2.3283064365386963e-10,
                        )
                    )
                    * 2.3283064365386963e-10
                )
                * up
            )
        normalized, sign_source = _normalize207(value, scale, inverse_scale)
        code = _encode(normalized, sign_source)
        flat_output: tl.constexpr = (
            BUFFER
            and NT.dtype.element_ty == tl.int32
            and (SO == D * CO)
            and (CO >= 0)
            and (T > 0)
            and (D > 0)
            and (T * D + K * B <= 2147483647)
            and ((T * D + B - 1) // B * B <= 2147483647)
            and ((T * D + B - 1) // B * B * CO <= 2147483647)
        )
        if flat_output:
            tl.store(Q + index * CO, code, index < n * D)
        else:
            tl.store(Q + row * SO + col * CO, code, index < n * D)
        if BUFFER:
            current_value = next_value


def silu_mul_dynamic_tensorwise_quant(
    input,
    output,
    scale,
    num_tokens_tensor,
    expected_num_tokens,
    intermediate_size,
):
    rows, width = (input.shape[0], intermediate_size)
    result = output
    result_scale = scale
    sx, cx = input.stride()
    so, co = output.stride()
    block = 1024
    programs = min(1024, (rows * width + block - 1) // block)
    partial = torch.empty(
        (programs,), device=input.device, dtype=torch.float32
    )
    buffered = rows * width >= 4194304
    activation = (
        torch.empty((rows, width), device=input.device, dtype=torch.float32)
        if buffered
        else None
    )
    _activate.run(
        input,
        activation,
        partial,
        num_tokens_tensor,
        width,
        programs,
        sx,
        cx,
        block,
        buffered,
        rows,
        num_warps=4,
        enable_fp_fusion=False,
        grid=(programs,),
        warmup=False,
    )
    _quant.run(
        None if buffered else input,
        activation,
        result,
        partial,
        result_scale,
        num_tokens_tensor,
        width,
        rows,
        so,
        co,
        programs,
        programs and 1 << (programs - 1).bit_length(),
        block,
        sx,
        cx,
        buffered,
        num_warps=4,
        enable_fp_fusion=False,
        grid=(programs,),
        warmup=False,
    )
    return (result, result_scale)


__all__ = ["silu_mul_dynamic_tensorwise_quant"]
