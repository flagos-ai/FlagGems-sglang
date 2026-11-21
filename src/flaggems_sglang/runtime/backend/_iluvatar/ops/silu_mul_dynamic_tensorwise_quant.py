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
def _encode(v):
    raw = v.to(tl.int32, bitcast=True)
    sign = raw >> 24 & 128
    a = tl.abs(v)
    mag = a.to(tl.int32, bitcast=True)
    exponent = tl.maximum(mag & 2139095040, 1015021568)
    bias = exponent + 167772160
    biased = a + bias.to(tl.float32, bitcast=True)
    code = biased.to(tl.int32, bitcast=True) - bias + (bias >> 20) - 1128
    code = tl.where(v != v, 127, code)
    return (code | sign).to(tl.uint8)


@triton.jit
def _activate(X, P, NT, META: tl.constexpr):
    D: tl.constexpr = META[0]
    T: tl.constexpr = META[1]
    K: tl.constexpr = META[2]
    SX: tl.constexpr = META[3]
    CX: tl.constexpr = META[4]
    B: tl.constexpr = META[5]
    n = tl.load(NT)
    pid = tl.program_id(0)
    tl.assume(pid >= 0)
    tl.assume(pid < K)
    lane = tl.arange(0, B)
    maximum = tl.full((B,), 0.0, tl.float32)
    for start in range(pid * B, T * D, K * B):
        if start < n * D:
            index = start + lane
            row = index // D
            col = index % D
            gate = tl.load(X + row * SX + col * CX, index < n * D, other=0).to(
                tl.float32
            )
            up = tl.load(
                X + row * SX + (col + D) * CX, index < n * D, other=0
            ).to(tl.float32)
            value = (
                gate
                * libdevice.fast_dividef(
                    2.3283064365386963e-10,
                    (1.0 + tl.exp(-gate)) * 2.3283064365386963e-10,
                )
                * up
            )
            maximum = tl.maximum(maximum, tl.abs(value))
    tl.store(P + pid, tl.max(maximum, 0))


@triton.jit
def _quant_loop(
    X,
    Q: tl.pointer_type(tl.uint8),
    n,
    scale,
    inverse_scale,
    D: tl.constexpr,
    T: tl.constexpr,
    K: tl.constexpr,
    SO: tl.constexpr,
    CO: tl.constexpr,
    B: tl.constexpr,
    SX: tl.constexpr,
    CX: tl.constexpr,
    REFINE: tl.constexpr,
):
    pid = tl.program_id(0)
    tl.assume(pid >= 0)
    tl.assume(pid < K)
    lane = tl.arange(0, B)
    use_scalar: tl.constexpr = (
        D % B == 0 and K * B % D == 0 and (T * D >= 2 * K * B)
    )
    for start in range(pid * B, T * D, K * B):
        index = start + lane
        row = index // D
        col = index % D
        gate = tl.load(
            X + row * SX + col * CX,
            (start if use_scalar else index) < n * D,
            other=0,
        ).to(tl.float32)
        up = tl.load(
            X + row * SX + (col + D) * CX,
            (start if use_scalar else index) < n * D,
            other=0,
        ).to(tl.float32)
        value = (
            gate
            * libdevice.fast_dividef(
                2.3283064365386963e-10,
                (1.0 + tl.exp(-gate)) * 2.3283064365386963e-10,
            )
            * up
        )
        if REFINE:
            quotient = value * inverse_scale
            residual = tl.fma(-quotient, scale, value)
            corrected = tl.fma(residual, inverse_scale, quotient)
            code = _encode(corrected)
            sign = (value.to(tl.int32, bitcast=True) >> 24 & 128).to(tl.uint8)
            code = code & 127 | sign
        else:
            code = _encode(value / scale)
        tl.store(
            Q + row * SO + col * CO,
            code,
            (start if use_scalar else index) < n * D,
        )


@triton.jit
def _quant(X, Q: tl.pointer_type(tl.uint8), P, S, NT, META: tl.constexpr):
    D: tl.constexpr = META[0]
    T: tl.constexpr = META[1]
    SO: tl.constexpr = META[2]
    CO: tl.constexpr = META[3]
    K: tl.constexpr = META[4]
    BK: tl.constexpr = META[5]
    B: tl.constexpr = META[6]
    SX: tl.constexpr = META[7]
    CX: tl.constexpr = META[8]
    n = tl.load(NT)
    r = tl.arange(0, BK)
    partial = tl.load(P + r, r < K, other=0)
    scale = tl.maximum(tl.max(partial, 0), 1e-10) / 448.0
    if tl.program_id(0) == 0:
        tl.store(S, scale)
    if T * D <= K * B:
        _quant_loop(X, Q, n, scale, 0.0, D, T, K, SO, CO, B, SX, CX, False)
    elif tl.abs(scale) < float("inf"):
        inverse_scale = tl.div_rn(1.0, scale)
        _quant_loop(
            X, Q, n, scale, inverse_scale, D, T, K, SO, CO, B, SX, CX, True
        )
    else:
        _quant_loop(X, Q, n, scale, 0.0, D, T, K, SO, CO, B, SX, CX, False)


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
    _activate.run(
        input,
        partial,
        num_tokens_tensor,
        (width, rows, programs, sx, cx, block),
        num_warps=4,
        enable_fp_fusion=False,
        grid=(programs,),
        warmup=False,
    )
    _quant.run(
        input,
        result,
        partial,
        result_scale,
        num_tokens_tensor,
        (
            width,
            rows,
            so,
            co,
            programs,
            programs and 1 << (programs - 1).bit_length(),
            block,
            sx,
            cx,
        ),
        num_warps=8 if rows * width >= 4194304 else 4,
        enable_fp_fusion=False,
        grid=(programs,),
        warmup=False,
    )
    return (result, result_scale)


__all__ = ["silu_mul_dynamic_tensorwise_quant"]
