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
def _ernie45_rope_qk_xpu(
    q_ptr,
    k_ptr,
    q_out_ptr,
    k_out_ptr,
    cos_sin_cache_ptr,
    positions_ptr,
    q_stride,
    k_stride,
    cache_stride,
    position_axis_stride,
    position_token_stride,
    NUM_Q_HEADS: tl.constexpr,
    NUM_K_HEADS: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    ROTARY_DIM: tl.constexpr,
    SECTION_HW: tl.constexpr,
    BLOCK_PAIRS: tl.constexpr,
):
    token = tl.program_id(0)
    half_rotary: tl.constexpr = ROTARY_DIM // 2
    pair = tl.arange(0, BLOCK_PAIRS)
    pair_mask = pair < half_rotary

    temporal_position = tl.load(positions_ptr + token * position_token_stride)
    height_position = tl.load(
        positions_ptr + position_axis_stride + token * position_token_stride
    )
    width_position = tl.load(
        positions_ptr
        + 2 * position_axis_stride
        + token * position_token_stride
    )
    cos_t = tl.load(
        cos_sin_cache_ptr + temporal_position * cache_stride + pair,
        mask=pair_mask,
        other=0.0,
    ).to(tl.float32)
    sin_t = tl.load(
        cos_sin_cache_ptr
        + temporal_position * cache_stride
        + half_rotary
        + pair,
        mask=pair_mask,
        other=0.0,
    ).to(tl.float32)
    cos_h = tl.load(
        cos_sin_cache_ptr + height_position * cache_stride + pair,
        mask=pair_mask,
        other=0.0,
    ).to(tl.float32)
    sin_h = tl.load(
        cos_sin_cache_ptr
        + height_position * cache_stride
        + half_rotary
        + pair,
        mask=pair_mask,
        other=0.0,
    ).to(tl.float32)
    cos_w = tl.load(
        cos_sin_cache_ptr + width_position * cache_stride + pair,
        mask=pair_mask,
        other=0.0,
    ).to(tl.float32)
    sin_w = tl.load(
        cos_sin_cache_ptr + width_position * cache_stride + half_rotary + pair,
        mask=pair_mask,
        other=0.0,
    ).to(tl.float32)
    use_hw = pair < SECTION_HW
    use_height = pair % 2 == 0
    cos = tl.where(use_hw, tl.where(use_height, cos_h, cos_w), cos_t)
    sin = tl.where(use_hw, tl.where(use_height, sin_h, sin_w), sin_t)

    for h in range(0, NUM_Q_HEADS):
        row = token * q_stride + h * HEAD_SIZE
        q_first = tl.load(q_ptr + row + pair, mask=pair_mask, other=0.0).to(
            tl.float32
        )
        q_second = tl.load(
            q_ptr + row + half_rotary + pair, mask=pair_mask, other=0.0
        ).to(tl.float32)
        tl.store(
            q_out_ptr + row + pair,
            q_first * cos - q_second * sin,
            mask=pair_mask,
        )
        tl.store(
            q_out_ptr + row + half_rotary + pair,
            q_second * cos + q_first * sin,
            mask=pair_mask,
        )

    for h in range(0, NUM_K_HEADS):
        row = token * k_stride + h * HEAD_SIZE
        k_first = tl.load(k_ptr + row + pair, mask=pair_mask, other=0.0).to(
            tl.float32
        )
        k_second = tl.load(
            k_ptr + row + half_rotary + pair, mask=pair_mask, other=0.0
        ).to(tl.float32)
        tl.store(
            k_out_ptr + row + pair,
            k_first * cos - k_second * sin,
            mask=pair_mask,
        )
        tl.store(
            k_out_ptr + row + half_rotary + pair,
            k_second * cos + k_first * sin,
            mask=pair_mask,
        )


@triton.jit
def _copy_tail(
    src_ptr,
    dst_ptr,
    stride,
    NUM_HEADS: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    ROTARY_DIM: tl.constexpr,
    BLOCK_TAIL: tl.constexpr,
):
    token = tl.program_id(0)
    h = tl.program_id(1)
    tail = tl.arange(0, BLOCK_TAIL)
    mask = (h < NUM_HEADS) & (tail < HEAD_SIZE - ROTARY_DIM)
    off = token * stride + h * HEAD_SIZE + ROTARY_DIM + tail
    tl.store(dst_ptr + off, tl.load(src_ptr + off, mask=mask), mask=mask)


def ernie45_rope_fused(
    q,
    k,
    cos_sin_cache,
    positions,
    mrope_section,
    head_size,
    rotary_dim,
):
    num_tokens = q.shape[0]
    num_q_heads = q.shape[1] // head_size
    num_k_heads = k.shape[1] // head_size
    q_out = torch.empty_like(q)
    k_out = torch.empty_like(k)
    block_pairs = 1 << ((rotary_dim // 2) - 1).bit_length()
    _ernie45_rope_qk_xpu[(num_tokens,)](
        q,
        k,
        q_out,
        k_out,
        cos_sin_cache,
        positions,
        q.stride(0),
        k.stride(0),
        cos_sin_cache.stride(0),
        positions.stride(0),
        positions.stride(1),
        NUM_Q_HEADS=num_q_heads,
        NUM_K_HEADS=num_k_heads,
        HEAD_SIZE=head_size,
        ROTARY_DIM=rotary_dim,
        SECTION_HW=int(mrope_section[0]) + int(mrope_section[1]),
        BLOCK_PAIRS=block_pairs,
        num_warps=1,
    )
    tail_size = head_size - rotary_dim
    if tail_size > 0:
        block_tail = 1 << (tail_size - 1).bit_length()
        _copy_tail[(num_tokens, num_q_heads)](
            q,
            q_out,
            q.stride(0),
            NUM_HEADS=num_q_heads,
            HEAD_SIZE=head_size,
            ROTARY_DIM=rotary_dim,
            BLOCK_TAIL=block_tail,
            num_warps=1,
        )
        _copy_tail[(num_tokens, num_k_heads)](
            k,
            k_out,
            k.stride(0),
            NUM_HEADS=num_k_heads,
            HEAD_SIZE=head_size,
            ROTARY_DIM=rotary_dim,
            BLOCK_TAIL=block_tail,
            num_warps=1,
        )
    return q_out, k_out


__all__ = ["ernie45_rope_fused"]
