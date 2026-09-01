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
):
    token = tl.program_id(0)
    head_group = tl.program_id(1) * BLOCK_HEADS
    pair = tl.program_id(2)
    half_rotary: tl.constexpr = ROTARY_DIM // 2

    axis = tl.where(
        pair < SECTION_T,
        0,
        tl.where(pair < SECTION_T + SECTION_H, 1, 2),
    )
    position = tl.load(
        positions_ptr
        + axis * position_axis_stride
        + token * position_token_stride
    )
    cache_base = cos_sin_cache_ptr + position * cache_stride
    cos = tl.load(cache_base + pair).to(tl.float32)
    sin = tl.load(cache_base + half_rotary + pair).to(tl.float32)

    head = head_group + tl.arange(0, BLOCK_HEADS)
    head_mask = head < NUM_HEADS
    row_base = token * input_stride + head * HEAD_SIZE
    first = tl.load(
        input_ptr + row_base + pair,
        mask=head_mask,
        other=0.0,
    ).to(tl.float32)
    second = tl.load(
        input_ptr + row_base + half_rotary + pair,
        mask=head_mask,
        other=0.0,
    ).to(tl.float32)
    tl.store(
        output_ptr + row_base + pair,
        first * cos - second * sin,
        mask=head_mask,
    )
    tl.store(
        output_ptr + row_base + half_rotary + pair,
        second * cos + first * sin,
        mask=head_mask,
    )


@triton.jit
def _copy_tail_kernel(
    input_ptr,
    output_ptr,
    input_stride,
    NUM_HEADS: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    ROTARY_DIM: tl.constexpr,
    BLOCK_HEADS: tl.constexpr,
):
    token = tl.program_id(0)
    head_group = tl.program_id(1) * BLOCK_HEADS
    tail = tl.program_id(2)
    head = head_group + tl.arange(0, BLOCK_HEADS)
    head_mask = head < NUM_HEADS
    offset = token * input_stride + head * HEAD_SIZE + ROTARY_DIM + tail
    tl.store(
        output_ptr + offset,
        tl.load(input_ptr + offset, mask=head_mask),
        mask=head_mask,
    )


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
    block_heads = 8
    half_rotary = rotary_dim // 2
    head_groups = triton.cdiv(num_heads, block_heads)
    grid = (num_tokens, head_groups, half_rotary)
    _mrope_kernel[grid](
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
        num_warps=1,
    )
    tail_size = head_size - rotary_dim
    if tail_size:
        _copy_tail_kernel[(num_tokens, head_groups, tail_size)](
            x,
            output,
            x.stride(0),
            NUM_HEADS=num_heads,
            HEAD_SIZE=head_size,
            ROTARY_DIM=rotary_dim,
            BLOCK_HEADS=block_heads,
            num_warps=1,
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
    q_out = _apply_mrope(
        q,
        cos_sin_cache,
        positions,
        mrope_section,
        head_size,
        rotary_dim,
    )
    k_out = _apply_mrope(
        k,
        cos_sin_cache,
        positions,
        mrope_section,
        head_size,
        rotary_dim,
    )
    return q_out, k_out


__all__ = ["mrope_fused"]
