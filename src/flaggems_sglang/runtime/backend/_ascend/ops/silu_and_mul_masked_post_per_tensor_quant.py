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
def _encode(value):
    is_nan = value != value
    raw = value.to(tl.int32, bitcast=True)
    sign = (raw >> 24).to(tl.int16) & 128
    magnitude = tl.minimum(
        tl.abs(value).to(tl.int32, bitcast=True), 1138753536
    )
    value = magnitude.to(tl.float32, bitcast=True)
    exponent = tl.maximum(magnitude >> 23, 121)
    magic_bits = exponent * 8388616 + 167771192
    rounded = value + magic_bits.to(tl.float32, bitcast=True)
    selected = tl.where(is_nan, 1.000015139579773, rounded)
    return selected.to(tl.int32, bitcast=True).to(tl.int16) | sign


@triton.jit
def _encode_pair(value0, value1):
    is_nan0 = value0 != value0
    is_nan1 = value1 != value1
    raw0 = value0.to(tl.int32, bitcast=True)
    raw1 = value1.to(tl.int32, bitcast=True)
    sign0 = (raw0 >> 24).to(tl.int16) & 128
    sign1 = (raw1 >> 24).to(tl.int16) & 128
    magnitude0 = tl.minimum(
        tl.abs(value0).to(tl.int32, bitcast=True), 1138753536
    )
    magnitude1 = tl.minimum(
        tl.abs(value1).to(tl.int32, bitcast=True), 1138753536
    )
    value0 = magnitude0.to(tl.float32, bitcast=True)
    value1 = magnitude1.to(tl.float32, bitcast=True)
    exponent0 = tl.maximum(magnitude0 >> 23, 121)
    exponent1 = tl.maximum(magnitude1 >> 23, 121)
    magic_bits0 = exponent0 * 8388616 + 167771192
    magic_bits1 = exponent1 * 8388616 + 167771192
    rounded0 = value0 + magic_bits0.to(tl.float32, bitcast=True)
    rounded1 = value1 + magic_bits1.to(tl.float32, bitcast=True)
    selected0 = tl.where(is_nan0, 1.000015139579773, rounded0)
    selected1 = tl.where(is_nan1, 1.000015139579773, rounded1)
    return (
        selected0.to(tl.int32, bitcast=True).to(tl.int16)
        | sign0
        | (selected1.to(tl.int32, bitcast=True).to(tl.int16) | sign1) << 8
    )


@triton.jit
def _encode_pair_scaled(value0, value1, scale_shift):
    is_nan0 = value0 != value0
    is_nan1 = value1 != value1
    raw0 = value0.to(tl.int32, bitcast=True)
    raw1 = value1.to(tl.int32, bitcast=True)
    sign0 = (raw0 >> 24).to(tl.int16) & 128
    sign1 = (raw1 >> 24).to(tl.int16) & 128
    magnitude0 = tl.minimum(
        tl.abs(value0).to(tl.int32, bitcast=True),
        1138753536 + scale_shift * 8388608,
    )
    magnitude1 = tl.minimum(
        tl.abs(value1).to(tl.int32, bitcast=True),
        1138753536 + scale_shift * 8388608,
    )
    value0 = magnitude0.to(tl.float32, bitcast=True)
    value1 = magnitude1.to(tl.float32, bitcast=True)
    exponent0 = tl.maximum(magnitude0 >> 23, 121 + scale_shift)
    exponent1 = tl.maximum(magnitude1 >> 23, 121 + scale_shift)
    magic_bits0 = exponent0 * 8388616 + (167771192 - scale_shift * 8)
    magic_bits1 = exponent1 * 8388616 + (167771192 - scale_shift * 8)
    rounded0 = value0 + magic_bits0.to(tl.float32, bitcast=True)
    rounded1 = value1 + magic_bits1.to(tl.float32, bitcast=True)
    selected0 = tl.where(is_nan0, 1.000015139579773, rounded0)
    selected1 = tl.where(is_nan1, 1.000015139579773, rounded1)
    return (
        selected0.to(tl.int32, bitcast=True).to(tl.int16)
        | sign0
        | (selected1.to(tl.int32, bitcast=True).to(tl.int16) | sign1) << 8
    )


@triton.jit
def _kernel(X, M, S, Q, META: tl.constexpr):
    TOTAL: tl.constexpr = META[0]
    T: tl.constexpr = META[1]
    H: tl.constexpr = META[2]
    SX0: tl.constexpr = META[3]
    SX1: tl.constexpr = META[4]
    SX2: tl.constexpr = META[5]
    SM: tl.constexpr = META[6]
    B: tl.constexpr = META[7]
    Q = Q.to(tl.pointer_type(tl.uint8))
    tiles: tl.constexpr = triton.cdiv(H, B)
    scale = tl.load(S).to(tl.float32)
    for pid in range(tl.program_id(0), TOTAL, tl.num_programs(0)):
        row = pid // tiles
        expert = row // T
        token = row % T
        col = pid % tiles * B + tl.arange(0, B)
        n = tl.load(M + expert * SM)
        base = X + expert * SX0 + token * SX1 + col * SX2
        gate = tl.load(base, col < H, other=0).to(tl.float32)
        up = tl.load(base + H * SX2, col < H, other=0).to(tl.float32)
        value = gate * (1.0 / (1.0 + tl.exp(gate * -1.0))) * up / scale
        encoded = tl.where(token < n, _encode(value), 0).to(tl.uint8)
        tl.store(Q + row * H + col, encoded, col < H)


@triton.jit
def _paired(X, M, S, Q, META: tl.constexpr):
    TOTAL: tl.constexpr = META[0]
    T: tl.constexpr = META[1]
    H: tl.constexpr = META[2]
    SX0: tl.constexpr = META[3]
    SX1: tl.constexpr = META[4]
    SM: tl.constexpr = META[5]
    B: tl.constexpr = META[6]
    tiles: tl.constexpr = triton.cdiv(H // 2, B)
    scale = tl.load(S).to(tl.float32)
    scale_shift = (scale.to(tl.int32, bitcast=True) >> 23) - 127
    if (
        (scale >= 7.888609052210118e-31)
        & (scale <= 6.338253001141147e29)
        & (scale.to(tl.int32, bitcast=True) & 8388607 == 0)
    ):
        for pid in range(tl.program_id(0), TOTAL, tl.num_programs(0)):
            row = pid // tiles
            expert = row // T
            token = row % T
            col = pid % tiles * B + tl.arange(0, B)
            n = tl.load(M + expert * SM)
            base = X + expert * SX0 + token * SX1 + col
            gb = tl.load(base, col < H // 2, other=0).to(tl.int32)
            g0 = (gb << 16).to(tl.float32, bitcast=True)
            g1 = (gb & -65536).to(tl.float32, bitcast=True)
            d0 = 1.0 + tl.exp(g0 * -1.0)
            d1 = 1.0 + tl.exp(g1 * -1.0)
            s0 = 1.0 / d0
            s1 = 1.0 / d1
            p0 = g0 * s0
            p1 = g1 * s1
            ub = tl.load(base + H // 2, col < H // 2, other=0).to(tl.int32)
            u0 = (ub << 16).to(tl.float32, bitcast=True)
            u1 = (ub & -65536).to(tl.float32, bitcast=True)
            y0 = p0 * u0
            y1 = p1 * u1
            v0 = y0
            v1 = y1
            code = tl.where(
                token < n, _encode_pair_scaled(v0, v1, scale_shift), 0
            ).to(tl.int16)
            tl.store(Q + row * (H // 2) + col, code, col < H // 2)
    else:
        for pid in range(tl.program_id(0), TOTAL, tl.num_programs(0)):
            row = pid // tiles
            expert = row // T
            token = row % T
            col = pid % tiles * B + tl.arange(0, B)
            n = tl.load(M + expert * SM)
            base = X + expert * SX0 + token * SX1 + col
            gb = tl.load(base, col < H // 2, other=0).to(tl.int32)
            g0 = (gb << 16).to(tl.float32, bitcast=True)
            g1 = (gb & -65536).to(tl.float32, bitcast=True)
            d0 = 1.0 + tl.exp(g0 * -1.0)
            d1 = 1.0 + tl.exp(g1 * -1.0)
            s0 = 1.0 / d0
            s1 = 1.0 / d1
            p0 = g0 * s0
            p1 = g1 * s1
            ub = tl.load(base + H // 2, col < H // 2, other=0).to(tl.int32)
            u0 = (ub << 16).to(tl.float32, bitcast=True)
            u1 = (ub & -65536).to(tl.float32, bitcast=True)
            y0 = p0 * u0
            y1 = p1 * u1
            v0 = y0 / scale
            v1 = y1 / scale
            code = tl.where(token < n, _encode_pair(v0, v1), 0).to(tl.int16)
            tl.store(Q + row * (H // 2) + col, code, col < H // 2)


@triton.jit
def _paired_two(X, M, S, Q, META: tl.constexpr):
    TOTAL: tl.constexpr = META[0]
    T: tl.constexpr = META[1]
    H: tl.constexpr = META[2]
    SX0: tl.constexpr = META[3]
    SX1: tl.constexpr = META[4]
    SM: tl.constexpr = META[5]
    B: tl.constexpr = META[6]
    tiles: tl.constexpr = triton.cdiv(H // 2, B)
    scale = tl.load(S).to(tl.float32)
    scale_shift = (scale.to(tl.int32, bitcast=True) >> 23) - 127
    if (
        (scale >= 7.888609052210118e-31)
        & (scale <= 6.338253001141147e29)
        & (scale.to(tl.int32, bitcast=True) & 8388607 == 0)
    ):
        for pid in range(tl.program_id(0), TOTAL, tl.num_programs(0)):
            row = pid // tiles
            expert = row // triton.cdiv(T, 2)
            token = row % triton.cdiv(T, 2) * 2 + tl.arange(0, 2)
            col = pid % tiles * B + tl.arange(0, B)
            n = tl.load(M + expert * SM)
            base = X + expert * SX0 + token[:, None] * SX1 + col[None, :]
            gb = tl.load(
                base, (token[:, None] < T) & (col[None, :] < H // 2), other=0
            ).to(tl.int32)
            g0 = (gb << 16).to(tl.float32, bitcast=True)
            g1 = (gb & -65536).to(tl.float32, bitcast=True)
            d0 = 1.0 + tl.exp(g0 * -1.0)
            d1 = 1.0 + tl.exp(g1 * -1.0)
            s0 = 1.0 / d0
            s1 = 1.0 / d1
            p0 = g0 * s0
            p1 = g1 * s1
            ub = tl.load(
                base + H // 2,
                (token[:, None] < T) & (col[None, :] < H // 2),
                other=0,
            ).to(tl.int32)
            u0 = (ub << 16).to(tl.float32, bitcast=True)
            u1 = (ub & -65536).to(tl.float32, bitcast=True)
            y0 = p0 * u0
            y1 = p1 * u1
            v0 = y0
            v1 = y1
            code = tl.where(
                token[:, None] < n, _encode_pair_scaled(v0, v1, scale_shift), 0
            ).to(tl.int16)
            tl.store(
                Q + (expert * T + token[:, None]) * (H // 2) + col[None, :],
                code,
                (token[:, None] < T) & (col[None, :] < H // 2),
            )
    else:
        for pid in range(tl.program_id(0), TOTAL, tl.num_programs(0)):
            row = pid // tiles
            expert = row // triton.cdiv(T, 2)
            token = row % triton.cdiv(T, 2) * 2 + tl.arange(0, 2)
            col = pid % tiles * B + tl.arange(0, B)
            n = tl.load(M + expert * SM)
            base = X + expert * SX0 + token[:, None] * SX1 + col[None, :]
            gb = tl.load(
                base, (token[:, None] < T) & (col[None, :] < H // 2), other=0
            ).to(tl.int32)
            g0 = (gb << 16).to(tl.float32, bitcast=True)
            g1 = (gb & -65536).to(tl.float32, bitcast=True)
            d0 = 1.0 + tl.exp(g0 * -1.0)
            d1 = 1.0 + tl.exp(g1 * -1.0)
            s0 = 1.0 / d0
            s1 = 1.0 / d1
            p0 = g0 * s0
            p1 = g1 * s1
            ub = tl.load(
                base + H // 2,
                (token[:, None] < T) & (col[None, :] < H // 2),
                other=0,
            ).to(tl.int32)
            u0 = (ub << 16).to(tl.float32, bitcast=True)
            u1 = (ub & -65536).to(tl.float32, bitcast=True)
            y0 = p0 * u0
            y1 = p1 * u1
            v0 = y0 / scale
            v1 = y1 / scale
            code = tl.where(token[:, None] < n, _encode_pair(v0, v1), 0).to(
                tl.int16
            )
            tl.store(
                Q + (expert * T + token[:, None]) * (H // 2) + col[None, :],
                code,
                (token[:, None] < T) & (col[None, :] < H // 2),
            )


def silu_and_mul_masked_post_per_tensor_quant(input, masked_m, scale):
    experts, tokens, doubled = input.shape
    hidden = doubled // 2
    strides = input.stride()
    output = input.new_empty(
        (experts, tokens, hidden), dtype=torch.float8_e4m3fn
    )
    if experts * tokens * hidden:
        if (
            strides[2] == 1
            and hidden % 2 == 0
            and (input.storage_offset() % 2 == 0)
            and (strides[0] % 2 == 0)
            and (strides[1] % 2 == 0)
        ):
            words = input.view(torch.int32)
            qwords = output.view(torch.int16)
            block = min(2048, 1 << max(0, (hidden // 2 - 1).bit_length()))
            rows = 2 if experts * tokens >= 4096 else 1
            kernel = _paired_two if rows == 2 else _paired
            grid = (
                experts
                * ((tokens + rows - 1) // rows)
                * ((hidden // 2 + block - 1) // block)
            )
            kernel.run(
                words,
                masked_m,
                scale,
                qwords,
                (
                    grid,
                    tokens,
                    hidden,
                    strides[0] // 2,
                    strides[1] // 2,
                    masked_m.stride(0),
                    block,
                ),
                num_warps=4,
                enable_fp_fusion=False,
                num_stages=2 if rows == 2 else 1,
                grid=(min(grid, 40),),
                warmup=False,
            )
        else:
            block = min(4096, 1 << max(0, (hidden - 1).bit_length()))
            grid = experts * tokens * ((hidden + block - 1) // block)
            _kernel.run(
                input,
                masked_m,
                scale,
                output.view(torch.uint8),
                (grid, tokens, hidden, *strides, masked_m.stride(0), block),
                num_warps=4,
                enable_fp_fusion=False,
                num_stages=1,
                grid=(min(grid, 40),),
                warmup=False,
            )
    return output


__all__ = ["silu_and_mul_masked_post_per_tensor_quant"]
