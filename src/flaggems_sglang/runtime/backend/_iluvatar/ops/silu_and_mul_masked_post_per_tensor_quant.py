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
def _kernel(X, M, S, Q, META: tl.constexpr):
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
    gate = tl.load(base, col < H, other=0, cache_modifier=META[7]).to(
        tl.float32
    )
    up = tl.load(base + H * SX2, col < H, other=0, cache_modifier=META[7]).to(
        tl.float32
    )
    if META[8] and (scale >= 7.888609052210118e-31) & (
        scale <= 6.338253001141147e29
    ) & (scale.to(tl.int32, bitcast=True) & 8388607 == 0):
        value = gate * (1.0 / (1.0 + tl.exp(-gate))) * up
        shift = (scale.to(tl.int32, bitcast=True) >> 23) - 127
        encoded = tl.where(token < n, _encode_scaled(value, shift), 0).to(
            tl.uint8
        )
    else:
        value = gate * (1.0 / (1.0 + tl.exp(-gate))) * up / scale
        encoded = tl.where(token < n, _encode(value), 0).to(tl.uint8)
    tl.store(Q + row * H + col, encoded, col < H)


def silu_and_mul_masked_post_per_tensor_quant(input, masked_m, scale):
    experts, tokens, doubled = input.shape
    hidden = doubled // 2
    output = torch.empty(
        (experts, tokens, hidden), dtype=torch.uint8, device=input.device
    )
    if output.numel():
        block = min(512, triton.next_power_of_2(hidden))
        grid = experts * tokens * triton.cdiv(hidden, block)
        _kernel.run(
            input,
            masked_m,
            scale,
            output,
            (
                tokens,
                hidden,
                *input.stride(),
                masked_m.stride(0),
                block,
                ".cv" if experts * tokens * doubled <= 16777216 else "",
                experts * tokens * doubled > 16777216,
            ),
            num_warps=4,
            enable_fp_fusion=False,
            grid=(grid,),
            warmup=False,
        )
    return output.view(torch.float8_e4m3fn)


__all__ = ["silu_and_mul_masked_post_per_tensor_quant"]
