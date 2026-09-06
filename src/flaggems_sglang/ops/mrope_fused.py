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
def _mrope_qk_kernel(
    q_ptr,
    k_ptr,
    q_out_ptr,
    k_out_ptr,
    cos_sin_cache_ptr,
    positions_ptr,
    NUM_Q_HEADS: tl.constexpr,
    NUM_K_HEADS: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    ROTARY_DIM: tl.constexpr,
    SECTION_T: tl.constexpr,
    SECTION_H: tl.constexpr,
    NUM_TOKENS: tl.constexpr,
    BLOCK_HEADS: tl.constexpr,
    BLOCK_PAIRS: tl.constexpr,
    BLOCK_TAIL: tl.constexpr,
):
    token = tl.program_id(0)
    half_rotary: tl.constexpr = ROTARY_DIM // 2
    q_stride: tl.constexpr = NUM_Q_HEADS * HEAD_SIZE
    k_stride: tl.constexpr = NUM_K_HEADS * HEAD_SIZE
    position_axis_stride: tl.constexpr = NUM_TOKENS

    temporal_position = tl.load(positions_ptr + token)
    height_position = tl.load(positions_ptr + position_axis_stride + token)
    width_position = tl.load(positions_ptr + 2 * position_axis_stride + token)

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
    cache_base = cos_sin_cache_ptr + position * ROTARY_DIM
    cos = tl.load(cache_base + pair, mask=pair_mask, other=0.0).to(tl.float32)
    sin = tl.load(
        cache_base + half_rotary + pair,
        mask=pair_mask,
        other=0.0,
    ).to(tl.float32)

    head = tl.arange(0, BLOCK_HEADS)

    # Rotate q.
    q_head_mask = (head[:, None] < NUM_Q_HEADS) & pair_mask[None, :]
    q_row_base = token * q_stride + head[:, None] * HEAD_SIZE
    q_first = tl.load(
        q_ptr + q_row_base + pair[None, :], mask=q_head_mask, other=0.0
    ).to(tl.float32)
    q_second = tl.load(
        q_ptr + q_row_base + half_rotary + pair[None, :],
        mask=q_head_mask,
        other=0.0,
    ).to(tl.float32)
    tl.store(
        q_out_ptr + q_row_base + pair[None, :],
        q_first * cos[None, :] - q_second * sin[None, :],
        mask=q_head_mask,
    )
    tl.store(
        q_out_ptr + q_row_base + half_rotary + pair[None, :],
        q_second * cos[None, :] + q_first * sin[None, :],
        mask=q_head_mask,
    )

    # Rotate k (shares cos/sin with q).
    k_head_mask = (head[:, None] < NUM_K_HEADS) & pair_mask[None, :]
    k_row_base = token * k_stride + head[:, None] * HEAD_SIZE
    k_first = tl.load(
        k_ptr + k_row_base + pair[None, :], mask=k_head_mask, other=0.0
    ).to(tl.float32)
    k_second = tl.load(
        k_ptr + k_row_base + half_rotary + pair[None, :],
        mask=k_head_mask,
        other=0.0,
    ).to(tl.float32)
    tl.store(
        k_out_ptr + k_row_base + pair[None, :],
        k_first * cos[None, :] - k_second * sin[None, :],
        mask=k_head_mask,
    )
    tl.store(
        k_out_ptr + k_row_base + half_rotary + pair[None, :],
        k_second * cos[None, :] + k_first * sin[None, :],
        mask=k_head_mask,
    )

    if HEAD_SIZE > ROTARY_DIM:
        tail = tl.arange(0, BLOCK_TAIL)
        tail_span = tail[None, :] < HEAD_SIZE - ROTARY_DIM
        q_tail_mask = (head[:, None] < NUM_Q_HEADS) & tail_span
        q_tail_offsets = (
            token * q_stride
            + head[:, None] * HEAD_SIZE
            + ROTARY_DIM
            + tail[None, :]
        )
        q_tail = tl.load(q_ptr + q_tail_offsets, mask=q_tail_mask)
        tl.store(q_out_ptr + q_tail_offsets, q_tail, mask=q_tail_mask)

        k_tail_mask = (head[:, None] < NUM_K_HEADS) & tail_span
        k_tail_offsets = (
            token * k_stride
            + head[:, None] * HEAD_SIZE
            + ROTARY_DIM
            + tail[None, :]
        )
        k_tail = tl.load(k_ptr + k_tail_offsets, mask=k_tail_mask)
        tl.store(k_out_ptr + k_tail_offsets, k_tail, mask=k_tail_mask)


def mrope_fused(
    q,
    k,
    cos_sin_cache,
    positions,
    mrope_section,
    head_size,
    rotary_dim,
):
    num_tokens, q_width = q.shape
    k_width = k.shape[1]
    num_q_heads = q_width // head_size
    num_k_heads = k_width // head_size

    q_out = torch.empty_like(q)
    k_out = torch.empty_like(k)

    block_heads = triton.next_power_of_2(max(num_q_heads, num_k_heads))
    block_pairs = triton.next_power_of_2(rotary_dim // 2)
    block_tail = triton.next_power_of_2(max(1, head_size - rotary_dim))

    _mrope_qk_kernel[(num_tokens,)](
        q,
        k,
        q_out,
        k_out,
        cos_sin_cache,
        positions,
        NUM_Q_HEADS=num_q_heads,
        NUM_K_HEADS=num_k_heads,
        HEAD_SIZE=head_size,
        ROTARY_DIM=rotary_dim,
        SECTION_T=int(mrope_section[0]),
        SECTION_H=int(mrope_section[1]),
        NUM_TOKENS=num_tokens,
        BLOCK_HEADS=block_heads,
        BLOCK_PAIRS=block_pairs,
        BLOCK_TAIL=block_tail,
        num_warps=4,
    )
    return q_out, k_out


__all__ = ["mrope_fused"]
