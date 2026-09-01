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


@triton.jit
def _mrope_kernel(
    input_ptr,
    output_ptr,
    cos_sin_cache_ptr,
    positions_ptr,
    input_stride,
    cache_stride,
    position_axis_stride,
    position_token_stride,
    NUM_HEADS: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    ROTARY_DIM: tl.constexpr,
    SECTION_T: tl.constexpr,
    SECTION_H: tl.constexpr,
    BLOCK_HEADS: tl.constexpr,
    BLOCK_PAIRS: tl.constexpr,
    BLOCK_TAIL: tl.constexpr,
):
    token = tl.program_id(0)
    half_rotary: tl.constexpr = ROTARY_DIM // 2

    temporal_position = tl.load(positions_ptr + token * position_token_stride)
    height_position = tl.load(
        positions_ptr + position_axis_stride + token * position_token_stride
    )
    width_position = tl.load(
        positions_ptr
        + 2 * position_axis_stride
        + token * position_token_stride
    )

    pair = tl.arange(0, BLOCK_PAIRS)
    pair_mask = pair < half_rotary
    position = tl.where(
        pair < SECTION_T,
        temporal_position,
        tl.where(
            pair < SECTION_T + SECTION_H,
            height_position,
            width_position,
        ),
    )
    cache_base = cos_sin_cache_ptr + position * cache_stride
    cos = tl.load(cache_base + pair, mask=pair_mask, other=0.0).to(tl.float32)
    sin = tl.load(
        cache_base + half_rotary + pair,
        mask=pair_mask,
        other=0.0,
    ).to(tl.float32)

    head = tl.arange(0, BLOCK_HEADS)
    head_pair_mask = (head[:, None] < NUM_HEADS) & pair_mask[None, :]
    row_base = token * input_stride + head[:, None] * HEAD_SIZE
    first = tl.load(
        input_ptr + row_base + pair[None, :],
        mask=head_pair_mask,
        other=0.0,
    ).to(tl.float32)
    second = tl.load(
        input_ptr + row_base + half_rotary + pair[None, :],
        mask=head_pair_mask,
        other=0.0,
    ).to(tl.float32)
    tl.store(
        output_ptr + row_base + pair[None, :],
        first * cos[None, :] - second * sin[None, :],
        mask=head_pair_mask,
    )
    tl.store(
        output_ptr + row_base + half_rotary + pair[None, :],
        second * cos[None, :] + first * sin[None, :],
        mask=head_pair_mask,
    )

    if HEAD_SIZE > ROTARY_DIM:
        tail = tl.arange(0, BLOCK_TAIL)
        tail_mask = (head[:, None] < NUM_HEADS) & (
            tail[None, :] < HEAD_SIZE - ROTARY_DIM
        )
        tail_offsets = (
            token * input_stride
            + head[:, None] * HEAD_SIZE
            + ROTARY_DIM
            + tail[None, :]
        )
        tail_values = tl.load(input_ptr + tail_offsets, mask=tail_mask)
        tl.store(output_ptr + tail_offsets, tail_values, mask=tail_mask)


def _apply_mrope(
    x,
    cos_sin_cache,
    positions,
    mrope_section,
    head_size,
    rotary_dim,
):
    num_tokens, width = x.shape
    num_heads = width // head_size
    output = torch.empty_like(x)
    block_heads = triton.next_power_of_2(num_heads)
    block_pairs = triton.next_power_of_2(rotary_dim // 2)
    block_tail = triton.next_power_of_2(max(1, head_size - rotary_dim))
    _mrope_kernel[(num_tokens,)](
        x,
        output,
        cos_sin_cache,
        positions,
        x.stride(0),
        cos_sin_cache.stride(0),
        positions.stride(0),
        positions.stride(1),
        NUM_HEADS=num_heads,
        HEAD_SIZE=head_size,
        ROTARY_DIM=rotary_dim,
        SECTION_T=int(mrope_section[0]),
        SECTION_H=int(mrope_section[1]),
        BLOCK_HEADS=block_heads,
        BLOCK_PAIRS=block_pairs,
        BLOCK_TAIL=block_tail,
        num_warps=4,
    )
    return output


def mrope_fused(
    q,
    k,
    cos_sin_cache,
    positions,
    mrope_section,
    head_size,
    rotary_dim,
):
    positions_i32 = positions.to(torch.int32)
    q_out = _apply_mrope(
        q,
        cos_sin_cache,
        positions_i32,
        mrope_section,
        head_size,
        rotary_dim,
    )
    k_out = _apply_mrope(
        k,
        cos_sin_cache,
        positions_i32,
        mrope_section,
        head_size,
        rotary_dim,
    )
    return q_out, k_out


__all__ = ["mrope_fused"]
