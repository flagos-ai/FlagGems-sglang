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
def _parts207(index, D: tl.constexpr, U: tl.constexpr):
    if U:
        row = (index.to(tl.uint32) // D).to(tl.int32)
        col = (index.to(tl.uint32) % D).to(tl.int32)
    else:
        row = index // D
        col = index % D
    return (row, col)


@triton.jit
def _denominator_pair(e0, e1):
    low = e0.to(tl.uint32, bitcast=True).to(tl.uint64)
    high = e1.to(tl.uint32, bitcast=True).to(tl.uint64)
    packed = low | high << 32
    coefficient = tl.full((), 3422735717598494720, tl.uint64)
    result = tl.inline_asm_elementwise(
        "v_pk_fma_f32 $0, $1, $2, $2;",
        constraints="=v,v,v",
        args=[packed, coefficient],
        dtype=tl.uint64,
        is_pure=True,
        pack=1,
    )
    first = result.to(tl.uint32).to(tl.float32, bitcast=True)
    second = (result >> 32).to(tl.uint32).to(tl.float32, bitcast=True)
    return (first, second)


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
def _activate(X, P, NT, META: tl.constexpr):
    D: tl.constexpr = META[0]
    SX: tl.constexpr = META[1]
    CX: tl.constexpr = META[2]
    B: tl.constexpr = META[3]
    n = tl.load(NT)
    lane = tl.arange(0, B)
    maximum = tl.full((B,), 0.0, tl.float32)
    for start in range(tl.program_id(0) * B, n * D, tl.num_programs(0) * B):
        index = start + lane
        row = index // D
        col = index % D
        gate = tl.load(X + row * SX + col * CX, index < n * D, other=0).to(
            tl.float32
        )
        up = tl.load(X + row * SX + (col + D) * CX, index < n * D, other=0).to(
            tl.float32
        )
        value = (
            gate
            * libdevice.fast_dividef(
                2.3283064365386963e-10,
                tl.map_elementwise(_denominator_pair, tl.exp(-gate), pack=2)[
                    0
                ],
            )
            * up
        )
        maximum = tl.maximum(maximum, tl.abs(value))
    tl.store(P + tl.program_id(0), tl.max(maximum, 0))


@triton.jit
def _quant(
    X,
    Q: tl.pointer_type(tl.uint8),
    P,
    S,
    NT,
    META: tl.constexpr,
    U: tl.constexpr,
):
    D: tl.constexpr = META[0]
    SO: tl.constexpr = META[1]
    CO: tl.constexpr = META[2]
    K: tl.constexpr = META[3]
    BK: tl.constexpr = META[4]
    B: tl.constexpr = META[5]
    SX: tl.constexpr = META[6]
    CX: tl.constexpr = META[7]
    n = tl.load(NT)
    r = tl.arange(0, BK)
    partial = tl.load(P + r, r < K, other=0)
    scale = tl.maximum(tl.max(partial, 0), 1e-10) / 448.0
    if tl.program_id(0) == 0:
        tl.store(S, scale)
    lane = tl.arange(0, B)
    for start in range(tl.program_id(0) * B, n * D, tl.num_programs(0) * B):
        index = start + lane
        row, col = _parts207(index, D, U and NT.dtype.element_ty == tl.int32)
        gate = tl.load(X + row * SX + col * CX, index < n * D, other=0).to(
            tl.float32
        )
        up = tl.load(X + row * SX + (col + D) * CX, index < n * D, other=0).to(
            tl.float32
        )
        value = (
            gate
            * libdevice.fast_dividef(
                2.3283064365386963e-10,
                tl.map_elementwise(_denominator_pair, tl.exp(-gate), pack=2)[
                    0
                ],
            )
            * up
        )
        code = _encode(value / scale)
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
    sx, cx = input.stride()
    so, co = output.stride()
    block = 4096 if rows * width >= 1048576 else 1024
    programs = min(256, (rows * width + block - 1) // block)
    unsigned = (
        rows > 0
        and width > 0
        and (programs > 0)
        and (block > 0)
        and (rows * width + programs * block + block <= 2147483647)
    )
    partial = torch.empty(
        (programs,), device=input.device, dtype=torch.float32
    )
    _activate.run(
        input,
        partial,
        num_tokens_tensor,
        (width, sx, cx, block),
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
            so,
            co,
            programs,
            programs and 1 << (programs - 1).bit_length(),
            block,
            sx,
            cx,
        ),
        unsigned,
        num_warps=4,
        enable_fp_fusion=False,
        grid=(programs,),
        warmup=False,
    )
    return (result, result_scale)


__all__ = ["silu_mul_dynamic_tensorwise_quant"]
