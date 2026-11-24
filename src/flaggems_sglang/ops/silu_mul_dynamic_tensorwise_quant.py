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

if torch.version.hip is None:

    @triton.jit
    def _positive_max(x):
        return tl.max(x.to(tl.uint32, bitcast=True), 0).to(
            tl.float32, bitcast=True
        )

    @triton.jit
    def _activation_exp(x):
        return libdevice.fast_expf(x)

    @triton.jit
    def _activation_reciprocal(d):
        scaled = tl.fma(d, 2.3283064365386963e-10, 2.3283064365386963e-10)
        reciprocal = tl.inline_asm_elementwise(
            "rcp.approx.ftz.f32 $0, $1;",
            constraints="=f,f",
            args=[scaled],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )
        return reciprocal * 2.3283064365386963e-10

    @triton.jit
    def _normalization_divide207(a, b):
        return a / b

    @triton.jit
    def _normalization_multiply207(a, b, reciprocal):
        return a * reciprocal

    @triton.jit
    def _address_parts207(
        index, D: tl.constexpr, T: tl.constexpr, B: tl.constexpr
    ):
        return (index // D, index % D)

    @triton.jit
    def _encode(v):
        raw = v.to(tl.int32, bitcast=True)
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
    def _scale_divide448207(a):
        return a / 448.0

else:

    @triton.jit
    def _positive_max(x):
        return tl.max(x, 0)

    @triton.jit
    def _activation_exp(x):
        return libdevice.fast_expf(x)

    @triton.jit
    def _activation_reciprocal(d):
        d = 1.0 + d
        return (
            libdevice.fast_dividef(1.0, d * 2.3283064365386963e-10)
            * 2.3283064365386963e-10
        )

    @triton.jit
    def _normalization_divide207(a, b):
        if tl.abs(b) < float("inf"):
            reciprocal = libdevice.fast_dividef(1.0, b)
            quotient = a * reciprocal
            residual = tl.fma(
                (quotient.to(tl.uint32, bitcast=True) ^ 2147483648).to(
                    tl.float32, bitcast=True
                ),
                b,
                a,
            )
            result = tl.fma(residual, reciprocal, quotient)
            result = libdevice.copysign(result, a)
        else:
            result = a / b
        return result

    @triton.jit
    def _normalization_multiply207(a, b, reciprocal):
        quotient = a * reciprocal
        if tl.abs(b) < float("inf"):
            residual = tl.fma(
                (quotient.to(tl.uint32, bitcast=True) ^ 2147483648).to(
                    tl.float32, bitcast=True
                ),
                b,
                a,
            )
            quotient = tl.fma(residual, reciprocal, quotient)
            quotient = libdevice.copysign(quotient, a)
        return quotient

    @triton.jit
    def _address_parts207(
        index, D: tl.constexpr, T: tl.constexpr, B: tl.constexpr
    ):
        if D > 0 and T * D > 0 and (T * D <= 2147483647 - 1024 * B):
            row = (index.to(tl.uint32) // D).to(tl.int32)
            col = (index.to(tl.uint32) % D).to(tl.int32)
        else:
            row = index // D
            col = index % D
        return (row, col)

    @triton.jit
    def _encode(v):
        raw = v.to(tl.int32, bitcast=True)
        sign = raw >> 24 & 128
        a = tl.abs(v)
        mag = a.to(tl.int32, bitcast=True)
        exponent = tl.maximum(mag & 2139095040, 1015021568)
        bias = exponent + 167772160
        biased = a + bias.to(tl.float32, bitcast=True)
        code = biased.to(tl.int32, bitcast=True) + (bias >> 20) - 1128
        code = tl.where(a <= 464.0, code, 127)
        return (code | sign).to(tl.uint8)

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
def _activate(
    X,
    P,
    NT,
    D: tl.constexpr,
    T: tl.constexpr,
    SX: tl.constexpr,
    CX: tl.constexpr,
    B: tl.constexpr,
):
    n = tl.load(NT)
    lane = tl.arange(0, B)
    maximum = tl.full((B,), 0.0, tl.float32)
    for start in range(tl.program_id(0) * B, T * D, tl.num_programs(0) * B):
        index = start + lane
        row, col = _address_parts207(index, D, T, B)
        gate = tl.load(
            X + row * SX + col * CX,
            index < n * D,
            other=0,
            cache_modifier=".cg",
        ).to(tl.float32)
        up = tl.load(
            X + row * SX + (col + D) * CX,
            index < n * D,
            other=0,
            cache_modifier=".cg",
        ).to(tl.float32)
        value = gate * _activation_reciprocal(_activation_exp(-gate)) * up
        maximum = tl.maximum(maximum, tl.abs(value))
    tl.store(P + tl.program_id(0), _positive_max(maximum))


@triton.jit
def _quant(
    X,
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
):
    n = tl.load(NT)
    r = tl.arange(0, BK)
    partial = tl.load(P + r, r < K, other=0)
    scale = _scale_divide448207(tl.maximum(_positive_max(partial), 1e-10))
    if (T * D + B - 1) // B > K:
        inverse_scale = _normalization_divide207(1.0, scale)
    if tl.program_id(0) == 0:
        tl.store(S, scale)
    lane = tl.arange(0, B)
    for start in range(tl.program_id(0) * B, T * D, tl.num_programs(0) * B):
        index = start + lane
        row, col = _address_parts207(index, D, T, B)
        gate = tl.load(
            X + row * SX + col * CX,
            index < n * D,
            other=0,
            cache_modifier=".cg",
        ).to(tl.float32)
        up = tl.load(
            X + row * SX + (col + D) * CX,
            index < n * D,
            other=0,
            cache_modifier=".cg",
        ).to(tl.float32)
        value = gate * _activation_reciprocal(_activation_exp(-gate)) * up
        if (T * D + B - 1) // B > K:
            code = _encode(
                _normalization_multiply207(value, scale, inverse_scale)
            )
        else:
            code = _encode(_normalization_divide207(value, scale))
        tl.store(Q + row * SO + col * CO, code, index < n * D)


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
    block = 1024
    programs = min(1024, (rows * width + block - 1) // block)
    sx, cx = input.stride()
    so, co = output.stride()
    partial = input.new_empty((programs,), dtype=torch.float32)
    _activate.run(
        input,
        partial,
        num_tokens_tensor,
        width,
        rows,
        sx,
        cx,
        block,
        num_warps=4,
        enable_fp_fusion=False,
        grid=(programs,),
        warmup=False,
    )
    partial_block = programs and 1 << (programs - 1).bit_length()
    _quant.run(
        input,
        result,
        partial,
        result_scale,
        num_tokens_tensor,
        width,
        rows,
        so,
        co,
        programs,
        partial_block,
        block,
        sx,
        cx,
        num_warps=4,
        enable_fp_fusion=False,
        grid=(programs,),
        warmup=False,
    )
    return (result, result_scale)


__all__ = ["silu_mul_dynamic_tensorwise_quant"]
