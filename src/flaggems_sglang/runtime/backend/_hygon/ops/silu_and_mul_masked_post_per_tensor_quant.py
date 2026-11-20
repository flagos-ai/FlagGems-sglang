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
from triton.language.extra import libdevice


@triton.jit
def _encode_scaled(value, shift):
    raw = value.to(tl.int32, bitcast=True)
    sign = raw >> 24 & 128
    magnitude = tl.minimum(raw & 2147483647, 1138753536 + shift * 8388608)
    value = magnitude.to(tl.float32, bitcast=True)
    exponent = tl.maximum(magnitude >> 23, 121 + shift)
    magic_bits = exponent * 8388616 + (167771192 - shift * 8) | sign
    rounded = (value + magic_bits.to(tl.float32, bitcast=True)).to(
        tl.int32, bitcast=True
    )
    return tl.where(
        raw & 2147483647 > 2139095040, 127 | sign, rounded & 255
    ).to(tl.uint8)


@triton.jit
def _encode(value):
    raw = value.to(tl.int32, bitcast=True)
    sign = raw >> 24 & 128
    magnitude = tl.minimum(raw & 2147483647, 1138753536)
    value = magnitude.to(tl.float32, bitcast=True)
    exponent = tl.maximum(magnitude >> 23, 121)
    magic_bits = exponent * 8388616 + 167771192 | sign
    rounded = (value + magic_bits.to(tl.float32, bitcast=True)).to(
        tl.int32, bitcast=True
    )
    return tl.where(
        raw & 2147483647 > 2139095040, 127 | sign, rounded & 255
    ).to(tl.uint8)


@triton.jit
def _kernel(X, M, S, Q, META: tl.constexpr):
    Q = Q.to(tl.pointer_type(tl.uint8))
    T: tl.constexpr = META[0]
    H: tl.constexpr = META[1]
    SX0: tl.constexpr = META[2]
    SX1: tl.constexpr = META[3]
    SX2: tl.constexpr = META[4]
    SM: tl.constexpr = META[5]
    B: tl.constexpr = META[6]
    pid = tl.program_id(0)
    tiles: tl.constexpr = triton.cdiv(H, B)
    row = pid // tiles
    expert = row // T
    token = row % T
    col = pid % tiles * B + tl.arange(0, B)
    n = tl.load(M + expert * SM)
    scale = tl.load(S).to(tl.float32)
    base = X + expert * SX0 + token * SX1 + col * SX2
    gate = tl.load(base, col < H, other=0, cache_modifier=".cg").to(tl.float32)
    up = tl.load(base + H * SX2, col < H, other=0, cache_modifier=".cg").to(
        tl.float32
    )
    if (
        (scale >= 7.888609052210118e-31)
        & (scale <= 6.338253001141147e29)
        & (scale.to(tl.int32, bitcast=True) & 8388607 == 0)
    ):
        shift = (scale.to(tl.int32, bitcast=True) >> 23) - 127
        value = (
            gate
            * (
                libdevice.fast_dividef(
                    1.0, (1.0 + tl.exp(-gate)) * 5.960464477539063e-08
                )
                * 5.960464477539063e-08
            )
            * up
        )
        encoded = tl.where(token < n, _encode_scaled(value, shift), 0).to(
            tl.uint8
        )
        tl.store(Q + row * H + col, encoded, col < H)
    else:
        if (tl.abs(scale) >= 7.888609052210118e-31) & (
            tl.abs(scale) <= 1.2676506002282294e30
        ):
            value = (
                gate
                * (
                    libdevice.fast_dividef(
                        1.0, (1.0 + tl.exp(-gate)) * 5.960464477539063e-08
                    )
                    * 5.960464477539063e-08
                )
                * up
                * (1.0 / scale)
            )
        else:
            value = (
                gate
                * (
                    libdevice.fast_dividef(
                        1.0, (1.0 + tl.exp(-gate)) * 5.960464477539063e-08
                    )
                    * 5.960464477539063e-08
                )
                * up
                / scale
            )
        encoded = tl.where(token < n, _encode(value), 0).to(tl.uint8)
        tl.store(Q + row * H + col, encoded, col < H)


@triton.jit
def _paired(X, M, S, Q, META: tl.constexpr):
    Q = Q.to(tl.pointer_type(tl.int16))
    T: tl.constexpr = META[0]
    H: tl.constexpr = META[1]
    SX0: tl.constexpr = META[2]
    SX1: tl.constexpr = META[3]
    SM: tl.constexpr = META[4]
    B: tl.constexpr = META[5]
    tiles: tl.constexpr = triton.cdiv(H // 2, B)
    pid = tl.program_id(0)
    row = pid // tiles
    expert = row // T
    token = row % T
    col = pid % tiles * B + tl.arange(0, B)
    ptr = X + expert * SX0 + token * SX1 + col
    gb = tl.load(ptr, col < H // 2, other=0, cache_modifier=".cg").to(
        tl.uint32
    )
    ub = tl.load(ptr + H // 2, col < H // 2, other=0, cache_modifier=".cg").to(
        tl.uint32
    )
    g0 = (gb << 16).to(tl.float32, bitcast=True)
    g1 = (gb & 4294901760).to(tl.float32, bitcast=True)
    u0 = (ub << 16).to(tl.float32, bitcast=True)
    u1 = (ub & 4294901760).to(tl.float32, bitcast=True)
    scale = tl.load(S).to(tl.float32)
    if (
        (scale >= 7.888609052210118e-31)
        & (scale <= 6.338253001141147e29)
        & (scale.to(tl.int32, bitcast=True) & 8388607 == 0)
    ):
        shift = (scale.to(tl.int32, bitcast=True) >> 23) - 127
        v0 = (
            g0
            * (
                libdevice.fast_dividef(
                    1.0, (1.0 + tl.exp(-g0)) * 5.960464477539063e-08
                )
                * 5.960464477539063e-08
            )
            * u0
        )
        v1 = (
            g1
            * (
                libdevice.fast_dividef(
                    1.0, (1.0 + tl.exp(-g1)) * 5.960464477539063e-08
                )
                * 5.960464477539063e-08
            )
            * u1
        )
        n = tl.load(M + expert * SM)
        packed = (
            _encode_scaled(v0, shift).to(tl.uint32)
            | _encode_scaled(v1, shift).to(tl.uint32) << 8
        )
        code = tl.where(token < n, packed, 0).to(tl.int16)
        tl.store(Q + row * (H // 2) + col, code, col < H // 2)
    else:
        if (tl.abs(scale) >= 7.888609052210118e-31) & (
            tl.abs(scale) <= 1.2676506002282294e30
        ):
            v0 = (
                g0
                * (
                    libdevice.fast_dividef(
                        1.0, (1.0 + tl.exp(-g0)) * 5.960464477539063e-08
                    )
                    * 5.960464477539063e-08
                )
                * u0
                * (1.0 / scale)
            )
            v1 = (
                g1
                * (
                    libdevice.fast_dividef(
                        1.0, (1.0 + tl.exp(-g1)) * 5.960464477539063e-08
                    )
                    * 5.960464477539063e-08
                )
                * u1
                * (1.0 / scale)
            )
        else:
            v0 = (
                g0
                * (
                    libdevice.fast_dividef(
                        1.0, (1.0 + tl.exp(-g0)) * 5.960464477539063e-08
                    )
                    * 5.960464477539063e-08
                )
                * u0
                / scale
            )
            v1 = (
                g1
                * (
                    libdevice.fast_dividef(
                        1.0, (1.0 + tl.exp(-g1)) * 5.960464477539063e-08
                    )
                    * 5.960464477539063e-08
                )
                * u1
                / scale
            )
        n = tl.load(M + expert * SM)
        packed = _encode(v0).to(tl.uint32) | _encode(v1).to(tl.uint32) << 8
        code = tl.where(token < n, packed, 0).to(tl.int16)
        tl.store(Q + row * (H // 2) + col, code, col < H // 2)


def silu_and_mul_masked_post_per_tensor_quant(input, masked_m, scale):
    experts, tokens, doubled = input.shape
    hidden = doubled // 2
    output = torch.empty(
        (experts, tokens, hidden),
        dtype=torch.float8_e4m3fn,
        device=input.device,
    )
    if output.numel():
        if (
            input.stride(2) == 1
            and hidden % 2 == 0
            and (input.storage_offset() % 2 == 0)
            and (input.stride(0) % 2 == 0)
            and (input.stride(1) % 2 == 0)
        ):
            words = input.view(torch.int32)
            qwords = output
            block = min(2048, triton.next_power_of_2(hidden // 2))
            grid = experts * tokens * triton.cdiv(hidden // 2, block)
            _paired.run(
                words,
                masked_m,
                scale,
                qwords,
                (
                    tokens,
                    hidden,
                    words.stride(0),
                    words.stride(1),
                    masked_m.stride(0),
                    block,
                ),
                grid=(grid,),
                warmup=False,
                num_warps=8,
                enable_fp_fusion=False,
            )
        else:
            block = min(1024, triton.next_power_of_2(hidden))
            grid = experts * tokens * triton.cdiv(hidden, block)
            _kernel.run(
                input,
                masked_m,
                scale,
                output,
                (tokens, hidden, *input.stride(), masked_m.stride(0), block),
                grid=(grid,),
                warmup=False,
                num_warps=4,
                enable_fp_fusion=False,
            )
    return output


__all__ = ["silu_and_mul_masked_post_per_tensor_quant"]
