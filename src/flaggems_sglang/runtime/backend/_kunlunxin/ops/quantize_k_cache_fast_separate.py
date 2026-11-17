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

"""quantize_k_cache_fast_separate implementation for kunlunxin."""

import torch
import triton
import triton.language as tl


@triton.jit
def _quantize_k_cache_fast_separate_encode_e4m3fn(value):
    bits = value.to(tl.int32, bitcast=True)
    magnitude = bits & 2147483647
    sign = bits >> 24 & 128
    rounded = (magnitude + 524287 + (magnitude >> 20 & 1) >> 20) - 960
    absolute = tl.abs(value)
    subnormal = (absolute + 16384.0).to(tl.int32, bitcast=True) - 1182793728
    code = tl.where(absolute < 0.015625, subnormal, rounded)
    code = tl.minimum(code, 126)
    code = tl.where(value != value, 127, code)
    return (code | sign).to(tl.int8)


@triton.jit
def _quantize_k_cache_fast_separate_pack_kernel(
    k_nope_ptr,
    k_rope_ptr,
    quant_ptr,
    scale_ptr,
    rope_out_ptr,
    NUM_TOKENS: tl.constexpr,
    NOPE_DIM: tl.constexpr,
    ROPE_DIM: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    STRIDE_NOPE_TOKEN: tl.constexpr,
    STRIDE_NOPE_COL: tl.constexpr,
    STRIDE_ROPE_TOKEN: tl.constexpr,
    STRIDE_ROPE_COL: tl.constexpr,
    OUTPUT_WIDTH: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    GROUPS: tl.constexpr,
    STRIDE_ROPE_OUT: tl.constexpr,
):
    group = tl.program_id(0).to(tl.int64) * GROUPS + tl.arange(0, GROUPS)
    row = group // NUM_GROUPS
    tile = group % NUM_GROUPS
    cols = tl.arange(0, BLOCK_SIZE)
    mask = (row[:, None] < NUM_TOKENS) & (cols[None, :] < GROUP_SIZE)
    scaled_value = tl.load(
        k_nope_ptr
        + row[:, None] * STRIDE_NOPE_TOKEN
        + (tile[:, None] * GROUP_SIZE + cols[None, :]) * STRIDE_NOPE_COL,
        mask,
        other=0,
    ).to(tl.float32)
    amax_bits = tl.max(
        scaled_value.to(tl.int32, bitcast=True) & 2147483647, axis=1
    )
    amax = amax_bits.to(tl.float32, bitcast=True)
    scale = amax / 448.0
    tiny = (scale > 0.0) & (scale < 7.888609052210118e-31)
    denominator = tl.where(tiny, scale * 1.8446744073709552e19, scale)
    inv_scale = 1.0 / denominator
    numerator = tl.where(
        tiny[:, None], scaled_value * 1.8446744073709552e19, scaled_value
    )
    value = numerator * inv_scale[:, None]
    value = tl.where(value < -448.0, -448.0, value)
    value = tl.where(value > 448.0, 448.0, value)
    encoded = _quantize_k_cache_fast_separate_encode_e4m3fn(value)
    tl.store(
        quant_ptr
        + row[:, None] * OUTPUT_WIDTH
        + tile[:, None] * GROUP_SIZE
        + cols[None, :],
        encoded,
        mask,
    )
    tl.store(
        scale_ptr + row * (OUTPUT_WIDTH // 4) + tile, scale, row < NUM_TOKENS
    )
    rope_cols = tl.arange(0, BLOCK_ROWS)
    rope_mask = (
        (row[:, None] < NUM_TOKENS)
        & (tile[:, None] == 0)
        & (rope_cols[None, :] < ROPE_DIM)
    )
    rope = tl.load(
        k_rope_ptr
        + row[:, None] * STRIDE_ROPE_TOKEN
        + rope_cols[None, :] * STRIDE_ROPE_COL,
        rope_mask,
        other=0,
    )
    tl.store(
        rope_out_ptr + row[:, None] * STRIDE_ROPE_OUT + rope_cols[None, :],
        rope,
        rope_mask,
    )


def quantize_k_cache_fast_separate(
    k_nope,
    k_rope,
    group_size=128,
):
    (num_tokens, nope_dim) = k_nope.shape
    rope_dim = k_rope.shape[1]
    num_groups = nope_dim // group_size
    width = nope_dim + num_groups * 4
    nope_part = torch.empty(
        (num_tokens, width), dtype=torch.uint8, device=k_nope.device
    )
    rope_part = torch.empty_strided(
        (num_tokens, rope_dim * 2),
        (max(2, rope_dim * 2), 1),
        dtype=torch.uint8,
        device=k_rope.device,
    )
    if num_tokens:
        raw = nope_part.view(torch.int8)
        scales = raw[:, nope_dim:].view(torch.float32)
        rope = rope_part.view(torch.int16)
        group_batch = max(
            1, min(8, 1024 // triton.next_power_of_2(group_size))
        )
        _quantize_k_cache_fast_separate_pack_kernel[
            (triton.cdiv(num_tokens * num_groups, group_batch),)
        ](
            k_nope,
            k_rope.view(torch.int16),
            raw,
            scales,
            rope,
            num_tokens,
            nope_dim,
            rope_dim,
            num_groups,
            group_size,
            k_nope.stride(0),
            k_nope.stride(1),
            k_rope.stride(0),
            k_rope.stride(1),
            width,
            triton.next_power_of_2(group_size),
            triton.next_power_of_2(max(1, rope_dim)),
            group_batch,
            rope_dim,
            num_warps=4,
        )
    return (nope_part.unsqueeze(1), rope_part.unsqueeze(1))


__all__ = ["quantize_k_cache_fast_separate"]
