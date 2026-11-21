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
def _kernel(
    X,
    M,
    S,
    Q,
    T: tl.constexpr,
    H: tl.constexpr,
    SX0: tl.constexpr,
    SX1: tl.constexpr,
    SX2: tl.constexpr,
    SM: tl.constexpr,
    B: tl.constexpr,
    E: tl.constexpr,
):
    X = X.to(tl.pointer_type(tl.int16))
    pid = tl.program_id(0)
    tiles: tl.constexpr = triton.cdiv(H, B)
    col = pid % tiles * B + tl.arange(0, B)
    scale = tl.load(S).to(tl.float32)
    for local in range(4):
        row = pid // tiles * 4 + local
        if E * T % 4 == 0 or row < E * T:
            expert = row // T
            token = row % T
            n = tl.load(M + expert * SM)
            base = X + expert * SX0 + token * SX1 + col * SX2
            gate = (tl.load(base, col < H, other=0).to(tl.int32) << 16).to(
                tl.float32, bitcast=True
            )
            up = (
                tl.load(base + H * SX2, col < H, other=0).to(tl.int32) << 16
            ).to(tl.float32, bitcast=True)
            value = gate * (1.0 / (1.0 + tl.exp(-gate))) * up / scale
            encoded = tl.where(token < n, _encode(value), 0).to(tl.uint8)
            tl.store(Q + row * H + col, encoded, col < H)


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
def _paired(
    X,
    M,
    S,
    Q,
    T: tl.constexpr,
    H: tl.constexpr,
    SX0: tl.constexpr,
    SX1: tl.constexpr,
    SM: tl.constexpr,
    B: tl.constexpr,
    E: tl.constexpr,
    R: tl.constexpr,
):
    pid = tl.program_id(0)
    tiles: tl.constexpr = triton.cdiv(H // 2, B)
    groups: tl.constexpr = triton.cdiv(T, R)
    expert = pid // tiles // groups
    first_token = pid // tiles % groups * R
    col = pid % tiles * B + tl.arange(0, B)
    if R > 8:
        scale = tl.load(S).to(tl.float32)
        n = tl.load(M + expert * SM)
        fold_scale = (
            (scale >= 7.888609052210118e-31)
            & (scale <= 6.338253001141147e29)
            & (scale.to(tl.int32, bitcast=True) & 8388607 == 0)
        )
        scale_shift = tl.where(
            fold_scale, (scale.to(tl.int32, bitcast=True) >> 23) - 127, 0
        )
    token = first_token + tl.arange(0, R)
    base = X + expert * SX0 + token[:, None] * SX1 + col[None, :]
    if H // 2 % B == 0 and T % R == 0:
        gb = tl.load(base).to(tl.uint32)
    else:
        gb = tl.load(
            base, (col[None, :] < H // 2) & (token[:, None] < T), other=0
        ).to(tl.uint32)
    g0 = (gb << 16).to(tl.float32, bitcast=True)
    g1 = (gb & 4294901760).to(tl.float32, bitcast=True)
    p0 = g0 * (1.0 / (1.0 + tl.exp(-g0)))
    p1 = g1 * (1.0 / (1.0 + tl.exp(-g1)))
    if H // 2 % B == 0 and T % R == 0:
        ub = tl.load(base + H // 2).to(tl.uint32)
    else:
        ub = tl.load(
            base + H // 2,
            (col[None, :] < H // 2) & (token[:, None] < T),
            other=0,
        ).to(tl.uint32)
    u0 = (ub << 16).to(tl.float32, bitcast=True)
    u1 = (ub & 4294901760).to(tl.float32, bitcast=True)
    v0 = p0 * u0
    v1 = p1 * u1
    if R <= 8:
        scale = tl.load(S).to(tl.float32)
        fold_scale = (
            (scale >= 7.888609052210118e-31)
            & (scale <= 6.338253001141147e29)
            & (scale.to(tl.int32, bitcast=True) & 8388607 == 0)
        )
        scale_shift = tl.where(
            fold_scale, (scale.to(tl.int32, bitcast=True) >> 23) - 127, 0
        )
    if not fold_scale:
        v0 = v0 / scale
        v1 = v1 / scale
    if R <= 8:
        n = tl.load(M + expert * SM)
    packed = (
        _encode_scaled(v0, scale_shift).to(tl.uint32)
        | _encode_scaled(v1, scale_shift).to(tl.uint32) << 8
    )
    packed = tl.where(token[:, None] < n, packed, 0).to(tl.int16)
    tl.store(
        Q + (expert * T + token[:, None]) * (H // 2) + col[None, :],
        packed,
        (col[None, :] < H // 2) & (token[:, None] < T),
    )


def silu_and_mul_masked_post_per_tensor_quant(input, masked_m, scale):
    experts, tokens, doubled = input.shape
    hidden = doubled // 2
    output = torch.empty(
        (experts, tokens, hidden), dtype=torch.uint8, device=input.device
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
            qwords = output.view(torch.int16)
            block = min(2048, triton.next_power_of_2(hidden // 2))
            rows = (
                16
                if experts
                * triton.cdiv(tokens, 16)
                * triton.cdiv(hidden // 2, block)
                >= 128
                else 8
            )
            grid = (
                experts
                * triton.cdiv(tokens, rows)
                * triton.cdiv(hidden // 2, block)
            )
            _paired[grid,](
                words,
                masked_m,
                scale,
                qwords,
                tokens,
                hidden,
                words.stride(0),
                words.stride(1),
                masked_m.stride(0),
                block,
                experts,
                rows,
                num_warps=4,
                enable_fp_fusion=False,
                budget_tiling=False,
                unroll_num=4,
                isCloseMemoryAsync=False,
                groups_per_cluster=rows,
            )
        else:
            block = min(4096, triton.next_power_of_2(hidden))
            grid = triton.cdiv(experts * tokens, 4) * triton.cdiv(
                hidden, block
            )
            _kernel[grid,](
                input,
                masked_m,
                scale,
                output,
                tokens,
                hidden,
                *input.stride(),
                masked_m.stride(0),
                block,
                experts,
                num_warps=4,
                enable_fp_fusion=False,
            )
    return output.view(torch.float8_e4m3fn)


__all__ = ["silu_and_mul_masked_post_per_tensor_quant"]
