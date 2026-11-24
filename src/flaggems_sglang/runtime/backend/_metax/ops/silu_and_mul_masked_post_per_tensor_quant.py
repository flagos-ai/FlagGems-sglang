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


@tl.core.extern
def _native_rcp(arg0, _semantic=None):
    return tl.core.extern_elementwise(
        "",
        "",
        [arg0],
        {(tl.float32,): ("llvm.mxc.rcp.f32", tl.float32)},
        is_pure=True,
        _semantic=_semantic,
    )


@triton.jit
def _divide_scale(value, scale):
    if (tl.abs(scale) < 7.888609052210118e-31) | (
        tl.abs(scale) > 1.2676506002282294e30
    ):
        return tl.div_rn(value, scale)
    return value / scale


@triton.jit
def _encode(value):
    raw = value.to(tl.uint32, bitcast=True)
    sign = raw >> 24 & 128
    magnitude = tl.minimum(raw & 2147483647, 1138753536)
    value = magnitude.to(tl.float32, bitcast=True)
    exponent = tl.maximum(magnitude >> 23, 121)
    magic_bits = exponent * 8388616 + 167771192 | sign
    rounded = (value + magic_bits.to(tl.float32, bitcast=True)).to(
        tl.uint32, bitcast=True
    )
    return tl.where(
        raw & 2147483647 > 2139095040, 127 | sign, rounded & 255
    ).to(tl.uint8)


@triton.jit
def _encode_scaled(value, shift):
    raw = value.to(tl.uint32, bitcast=True)
    sign = raw >> 24 & 128
    magnitude = tl.minimum(raw & 2147483647, 1138753536 + shift * 8388608)
    value = magnitude.to(tl.float32, bitcast=True)
    exponent = tl.maximum(magnitude >> 23, 121 + shift)
    magic_bits = exponent * 8388616 + (167771192 - shift * 8) | sign
    rounded = (value + magic_bits.to(tl.float32, bitcast=True)).to(
        tl.uint32, bitcast=True
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
    N: tl.constexpr = META[6]
    B: tl.constexpr = META[7]
    R: tl.constexpr = META[8]
    tiles: tl.constexpr = tl.cdiv(H, B)
    tile = tl.program_id(0)
    item = tile * R + tl.arange(0, R)
    row = item // tiles
    expert = row // T
    token = row % T
    col = item[:, None] % tiles * B + tl.arange(0, B)[None, :]
    base = X + expert[:, None] * SX0 + token[:, None] * SX1 + col * SX2
    if N % R == 0 and H % B == 0:
        mask = tl.full((R, B), True, tl.int1)
    else:
        mask = (item[:, None] < N) & (col < H)
    gate = tl.load(base, mask, 0).to(tl.float32)
    up = tl.load(base + H * SX2, mask, 0).to(tl.float32)
    scale = tl.load(S).to(tl.float32)
    if (
        (scale >= 7.888609052210118e-31)
        & (scale <= 6.338253001141147e29)
        & (scale.to(tl.int32, bitcast=True) & 8388607 == 0)
    ):
        shift = (scale.to(tl.int32, bitcast=True) >> 23) - 127
        value = (
            gate
            * (
                _native_rcp(
                    tl.fma(
                        tl.exp2(-gate * 1.4426950408889634),
                        5.960464477539063e-08,
                        5.960464477539063e-08,
                    )
                )
                * 5.960464477539063e-08
            )
            * up
        )
    elif (tl.abs(scale) >= 7.888609052210118e-31) & (
        tl.abs(scale) <= 1.2676506002282294e30
    ):
        shift = 0
        value = (
            gate
            * (
                _native_rcp(
                    tl.fma(
                        tl.exp2(-gate * 1.4426950408889634),
                        5.960464477539063e-08,
                        5.960464477539063e-08,
                    )
                )
                * 5.960464477539063e-08
            )
            * up
            / scale
        )
    else:
        shift = 0
        value = tl.div_rn(
            gate
            * (
                _native_rcp(
                    tl.fma(
                        tl.exp2(-gate * 1.4426950408889634),
                        5.960464477539063e-08,
                        5.960464477539063e-08,
                    )
                )
                * 5.960464477539063e-08
            )
            * up,
            scale,
        )
    n = tl.load(M + expert * SM, item < N, 0)
    code = tl.where(
        token[:, None] < n[:, None], _encode_scaled(value, shift), 0
    ).to(tl.uint8)
    tl.store(Q + row[:, None] * H + col, code, mask)


@triton.jit
def _paired(X, M, S, Q, META: tl.constexpr):
    T: tl.constexpr = META[0]
    H: tl.constexpr = META[1]
    SX0: tl.constexpr = META[2]
    SX1: tl.constexpr = META[3]
    SM: tl.constexpr = META[4]
    N: tl.constexpr = META[5]
    B: tl.constexpr = META[6]
    R: tl.constexpr = META[7]
    tiles: tl.constexpr = tl.cdiv(H // 2, B)
    item = tl.program_id(0) * R + tl.arange(0, R)
    row = item // tiles
    expert = row // T
    token = row % T
    col = item[:, None] % tiles * B + tl.arange(0, B)[None, :]
    base = X + expert[:, None] * SX0 + token[:, None] * SX1 + col
    mask = (item[:, None] < N) & (col < H // 2)
    gb = tl.load(base, mask, 0).to(tl.uint32)
    ub = tl.load(base + H // 2, mask, 0).to(tl.uint32)
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
                _native_rcp(
                    tl.fma(
                        tl.exp2(-g0 * 1.4426950408889634),
                        5.960464477539063e-08,
                        5.960464477539063e-08,
                    )
                )
                * 5.960464477539063e-08
            )
            * u0
        )
        v1 = (
            g1
            * (
                _native_rcp(
                    tl.fma(
                        tl.exp2(-g1 * 1.4426950408889634),
                        5.960464477539063e-08,
                        5.960464477539063e-08,
                    )
                )
                * 5.960464477539063e-08
            )
            * u1
        )
    elif (tl.abs(scale) >= 7.888609052210118e-31) & (
        tl.abs(scale) <= 1.2676506002282294e30
    ):
        shift = 0
        v0 = (
            g0
            * (
                _native_rcp(
                    tl.fma(
                        tl.exp2(-g0 * 1.4426950408889634),
                        5.960464477539063e-08,
                        5.960464477539063e-08,
                    )
                )
                * 5.960464477539063e-08
            )
            * u0
            / scale
        )
        v1 = (
            g1
            * (
                _native_rcp(
                    tl.fma(
                        tl.exp2(-g1 * 1.4426950408889634),
                        5.960464477539063e-08,
                        5.960464477539063e-08,
                    )
                )
                * 5.960464477539063e-08
            )
            * u1
            / scale
        )
    else:
        shift = 0
        v0 = tl.div_rn(
            g0
            * (
                _native_rcp(
                    tl.fma(
                        tl.exp2(-g0 * 1.4426950408889634),
                        5.960464477539063e-08,
                        5.960464477539063e-08,
                    )
                )
                * 5.960464477539063e-08
            )
            * u0,
            scale,
        )
        v1 = tl.div_rn(
            g1
            * (
                _native_rcp(
                    tl.fma(
                        tl.exp2(-g1 * 1.4426950408889634),
                        5.960464477539063e-08,
                        5.960464477539063e-08,
                    )
                )
                * 5.960464477539063e-08
            )
            * u1,
            scale,
        )
    n = tl.load(M + expert * SM, item < N, 0)
    valid = token[:, None] < n[:, None]
    packed = (
        _encode_scaled(v0, shift).to(tl.uint32)
        | _encode_scaled(v1, shift).to(tl.uint32) << 8
    )
    packed = tl.where(valid, packed, 0).to(tl.int16)
    tl.store(Q + row[:, None] * (H // 2) + col, packed, mask)


@triton.jit
def _paired_scalar(X, M, S, Q, META: tl.constexpr):
    T: tl.constexpr = META[0]
    H: tl.constexpr = META[1]
    SX0: tl.constexpr = META[2]
    SX1: tl.constexpr = META[3]
    SM: tl.constexpr = META[4]
    B: tl.constexpr = META[6]
    R: tl.constexpr = META[7]
    tiles: tl.constexpr = triton.cdiv(H // 2, B)
    row = tl.program_id(0) // (tiles // R)
    expert = row // T
    token = row % T
    col = (tl.program_id(0) % (tiles // R) * R + tl.arange(0, R))[
        :, None
    ] * B + tl.arange(0, B)[None, :]
    base = X + expert * SX0 + token * SX1 + col
    mask = col < H // 2
    gb = tl.load(base, mask, 0).to(tl.uint32)
    ub = tl.load(base + H // 2, mask, 0).to(tl.uint32)
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
                _native_rcp(
                    tl.fma(
                        tl.exp2(-g0 * 1.4426950408889634),
                        5.960464477539063e-08,
                        5.960464477539063e-08,
                    )
                )
                * 5.960464477539063e-08
            )
            * u0
        )
        v1 = (
            g1
            * (
                _native_rcp(
                    tl.fma(
                        tl.exp2(-g1 * 1.4426950408889634),
                        5.960464477539063e-08,
                        5.960464477539063e-08,
                    )
                )
                * 5.960464477539063e-08
            )
            * u1
        )
    elif (tl.abs(scale) >= 7.888609052210118e-31) & (
        tl.abs(scale) <= 1.2676506002282294e30
    ):
        shift = 0
        v0 = (
            g0
            * (
                _native_rcp(
                    tl.fma(
                        tl.exp2(-g0 * 1.4426950408889634),
                        5.960464477539063e-08,
                        5.960464477539063e-08,
                    )
                )
                * 5.960464477539063e-08
            )
            * u0
            / scale
        )
        v1 = (
            g1
            * (
                _native_rcp(
                    tl.fma(
                        tl.exp2(-g1 * 1.4426950408889634),
                        5.960464477539063e-08,
                        5.960464477539063e-08,
                    )
                )
                * 5.960464477539063e-08
            )
            * u1
            / scale
        )
    else:
        shift = 0
        v0 = tl.div_rn(
            g0
            * (
                _native_rcp(
                    tl.fma(
                        tl.exp2(-g0 * 1.4426950408889634),
                        5.960464477539063e-08,
                        5.960464477539063e-08,
                    )
                )
                * 5.960464477539063e-08
            )
            * u0,
            scale,
        )
        v1 = tl.div_rn(
            g1
            * (
                _native_rcp(
                    tl.fma(
                        tl.exp2(-g1 * 1.4426950408889634),
                        5.960464477539063e-08,
                        5.960464477539063e-08,
                    )
                )
                * 5.960464477539063e-08
            )
            * u1,
            scale,
        )
    n = tl.load(M + expert * SM)
    valid = token < n
    packed = (
        _encode_scaled(v0, shift).to(tl.uint32)
        | _encode_scaled(v1, shift).to(tl.uint32) << 8
    )
    packed = tl.where(valid, packed, 0).to(tl.int16)
    tl.store(Q + row * (H // 2) + col, packed, mask)


def silu_and_mul_masked_post_per_tensor_quant(input, masked_m, scale):
    experts, tokens, doubled = input.shape
    hidden = doubled // 2
    output = input.new_empty((experts, tokens, hidden), dtype=torch.uint8)
    if output.numel():
        if (
            input.dtype == torch.bfloat16
            and input.stride(2) == 1
            and (hidden % 2 == 0)
            and (input.storage_offset() % 2 == 0)
            and (input.stride(0) % 2 == 0)
            and (input.stride(1) % 2 == 0)
        ):
            words = input.view(torch.int32)
            qwords = output.view(torch.int16)
            block = min(512, triton.next_power_of_2(hidden // 2))
            rows = 4
            kernel = (
                _paired_scalar
                if triton.cdiv(hidden // 2, block) % rows == 0
                else _paired
            )
            count = experts * tokens * triton.cdiv(hidden // 2, block)
            kernel.run(
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
                    count,
                    block,
                    rows,
                ),
                num_warps=4,
                enable_fp_fusion=False,
                grid=(triton.cdiv(count, rows),),
                warmup=False,
            )
        else:
            block = min(1024, triton.next_power_of_2(hidden))
            count = experts * tokens * triton.cdiv(hidden, block)
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
                    count,
                    block,
                    1,
                ),
                num_warps=4,
                enable_fp_fusion=False,
                grid=(count,),
                warmup=False,
            )
    return output.view(torch.float8_e4m3fn)


__all__ = ["silu_and_mul_masked_post_per_tensor_quant"]
