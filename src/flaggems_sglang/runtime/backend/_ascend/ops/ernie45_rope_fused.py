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

PERSIST_MIN_S = 128
PERSIST_GRID = 640


@triton.jit
def _ernie45_rope_qk_persist(
    q_ptr,
    k_ptr,
    q_out_ptr,
    k_out_ptr,
    cos_sin_cache_ptr,
    positions_ptr,
    NUM_TOKENS: tl.constexpr,
    NUM_PROGRAMS: tl.constexpr,
    NUM_Q_HEADS: tl.constexpr,
    NUM_K_HEADS: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    ROTARY_DIM: tl.constexpr,
    SECTION_HW: tl.constexpr,
    Q_STRIDE: tl.constexpr,
    K_STRIDE: tl.constexpr,
    CACHE_STRIDE: tl.constexpr,
    POS_AXIS_STRIDE: tl.constexpr,
    POS_TOKEN_STRIDE: tl.constexpr,
    BLOCK_HEADS: tl.constexpr,
    BLOCK_PAIRS: tl.constexpr,
    BLOCK_TAIL: tl.constexpr,
    EXACT: tl.constexpr,
):
    pid = tl.program_id(0)
    half_rotary: tl.constexpr = ROTARY_DIM // 2
    pair = tl.arange(0, BLOCK_PAIRS)
    pair_c = tl.minimum(pair, half_rotary - 1)
    head = tl.arange(0, BLOCK_HEADS)
    q_ok = head < NUM_Q_HEADS
    k_ok = head < NUM_K_HEADS

    for token in tl.range(pid, NUM_TOKENS, NUM_PROGRAMS, num_stages=1):
        temporal_position = tl.load(positions_ptr + token * POS_TOKEN_STRIDE)
        height_position = tl.load(
            positions_ptr + POS_AXIS_STRIDE + token * POS_TOKEN_STRIDE
        )
        width_position = tl.load(
            positions_ptr + 2 * POS_AXIS_STRIDE + token * POS_TOKEN_STRIDE
        )
        if EXACT:
            cos_t = tl.load(
                cos_sin_cache_ptr + temporal_position * CACHE_STRIDE + pair
            ).to(tl.float32)
            sin_t = tl.load(
                cos_sin_cache_ptr
                + temporal_position * CACHE_STRIDE
                + half_rotary
                + pair
            ).to(tl.float32)
            cos_h = tl.load(
                cos_sin_cache_ptr + height_position * CACHE_STRIDE + pair
            ).to(tl.float32)
            sin_h = tl.load(
                cos_sin_cache_ptr
                + height_position * CACHE_STRIDE
                + half_rotary
                + pair
            ).to(tl.float32)
            cos_w = tl.load(
                cos_sin_cache_ptr + width_position * CACHE_STRIDE + pair
            ).to(tl.float32)
            sin_w = tl.load(
                cos_sin_cache_ptr
                + width_position * CACHE_STRIDE
                + half_rotary
                + pair
            ).to(tl.float32)
        else:
            cos_t = tl.load(
                cos_sin_cache_ptr + temporal_position * CACHE_STRIDE + pair_c
            ).to(tl.float32)
            sin_t = tl.load(
                cos_sin_cache_ptr
                + temporal_position * CACHE_STRIDE
                + half_rotary
                + pair_c
            ).to(tl.float32)
            cos_h = tl.load(
                cos_sin_cache_ptr + height_position * CACHE_STRIDE + pair_c
            ).to(tl.float32)
            sin_h = tl.load(
                cos_sin_cache_ptr
                + height_position * CACHE_STRIDE
                + half_rotary
                + pair_c
            ).to(tl.float32)
            cos_w = tl.load(
                cos_sin_cache_ptr + width_position * CACHE_STRIDE + pair_c
            ).to(tl.float32)
            sin_w = tl.load(
                cos_sin_cache_ptr
                + width_position * CACHE_STRIDE
                + half_rotary
                + pair_c
            ).to(tl.float32)

        use_hw = pair < SECTION_HW
        use_height = pair % 2 == 0
        cos = tl.where(use_hw, tl.where(use_height, cos_h, cos_w), cos_t)
        sin = tl.where(use_hw, tl.where(use_height, sin_h, sin_w), sin_t)

        q_row_base = token * Q_STRIDE + head[:, None] * HEAD_SIZE
        k_row_base = token * K_STRIDE + head[:, None] * HEAD_SIZE
        if EXACT:
            q_first = tl.load(q_ptr + q_row_base + pair[None, :]).to(
                tl.float32
            )
            q_second = tl.load(
                q_ptr + q_row_base + half_rotary + pair[None, :]
            ).to(tl.float32)
            tl.store(
                q_out_ptr + q_row_base + pair[None, :],
                q_first * cos[None, :] - q_second * sin[None, :],
                mask=q_ok[:, None],
            )
            tl.store(
                q_out_ptr + q_row_base + half_rotary + pair[None, :],
                q_second * cos[None, :] + q_first * sin[None, :],
                mask=q_ok[:, None],
            )
            k_mask = k_ok[:, None]
            k_first = tl.load(
                k_ptr + k_row_base + pair[None, :], mask=k_mask, other=0.0
            ).to(tl.float32)
            k_second = tl.load(
                k_ptr + k_row_base + half_rotary + pair[None, :],
                mask=k_mask,
                other=0.0,
            ).to(tl.float32)
            tl.store(
                k_out_ptr + k_row_base + pair[None, :],
                k_first * cos[None, :] - k_second * sin[None, :],
                mask=k_mask,
            )
            tl.store(
                k_out_ptr + k_row_base + half_rotary + pair[None, :],
                k_second * cos[None, :] + k_first * sin[None, :],
                mask=k_mask,
            )
        else:
            pair_mask = pair < half_rotary
            q_mask = q_ok[:, None] & pair_mask[None, :]
            q_first = tl.load(
                q_ptr + q_row_base + pair[None, :], mask=q_mask, other=0.0
            ).to(tl.float32)
            q_second = tl.load(
                q_ptr + q_row_base + half_rotary + pair[None, :],
                mask=q_mask,
                other=0.0,
            ).to(tl.float32)
            tl.store(
                q_out_ptr + q_row_base + pair[None, :],
                q_first * cos[None, :] - q_second * sin[None, :],
                mask=q_mask,
            )
            tl.store(
                q_out_ptr + q_row_base + half_rotary + pair[None, :],
                q_second * cos[None, :] + q_first * sin[None, :],
                mask=q_mask,
            )
            k_mask = k_ok[:, None] & pair_mask[None, :]
            k_first = tl.load(
                k_ptr + k_row_base + pair[None, :], mask=k_mask, other=0.0
            ).to(tl.float32)
            k_second = tl.load(
                k_ptr + k_row_base + half_rotary + pair[None, :],
                mask=k_mask,
                other=0.0,
            ).to(tl.float32)
            tl.store(
                k_out_ptr + k_row_base + pair[None, :],
                k_first * cos[None, :] - k_second * sin[None, :],
                mask=k_mask,
            )
            tl.store(
                k_out_ptr + k_row_base + half_rotary + pair[None, :],
                k_second * cos[None, :] + k_first * sin[None, :],
                mask=k_mask,
            )
            if HEAD_SIZE > ROTARY_DIM:
                tail = tl.arange(0, BLOCK_TAIL)
                tail_span = tail[None, :] < HEAD_SIZE - ROTARY_DIM
                q_tail_mask = q_ok[:, None] & tail_span
                q_tail_offsets = (
                    token * Q_STRIDE
                    + head[:, None] * HEAD_SIZE
                    + ROTARY_DIM
                    + tail[None, :]
                )
                tl.store(
                    q_out_ptr + q_tail_offsets,
                    tl.load(q_ptr + q_tail_offsets, mask=q_tail_mask),
                    mask=q_tail_mask,
                )
                k_tail_mask = k_ok[:, None] & tail_span
                k_tail_offsets = (
                    token * K_STRIDE
                    + head[:, None] * HEAD_SIZE
                    + ROTARY_DIM
                    + tail[None, :]
                )
                tl.store(
                    k_out_ptr + k_tail_offsets,
                    tl.load(k_ptr + k_tail_offsets, mask=k_tail_mask),
                    mask=k_tail_mask,
                )


def ernie45_rope_fused(
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
    block_heads = 1 << (max(num_q_heads, num_k_heads) - 1).bit_length()
    block_pairs = 1 << ((rotary_dim // 2) - 1).bit_length()
    tail = max(1, head_size - rotary_dim)
    block_tail = 1 << (tail - 1).bit_length()
    exact = int(
        block_heads == num_q_heads
        and block_pairs == rotary_dim // 2
        and head_size == rotary_dim
    )
    if num_tokens >= PERSIST_MIN_S:
        num_programs = (
            PERSIST_GRID if num_tokens > PERSIST_GRID else num_tokens
        )
    else:
        num_programs = num_tokens if num_tokens > 0 else 1
    _ernie45_rope_qk_persist[(num_programs,)](
        q,
        k,
        q_out,
        k_out,
        cos_sin_cache,
        positions,
        NUM_TOKENS=num_tokens,
        NUM_PROGRAMS=num_programs,
        NUM_Q_HEADS=num_q_heads,
        NUM_K_HEADS=num_k_heads,
        HEAD_SIZE=head_size,
        ROTARY_DIM=rotary_dim,
        SECTION_HW=int(mrope_section[0]) + int(mrope_section[1]),
        Q_STRIDE=q.stride(0),
        K_STRIDE=k.stride(0),
        CACHE_STRIDE=cos_sin_cache.stride(0),
        POS_AXIS_STRIDE=positions.stride(0),
        POS_TOKEN_STRIDE=positions.stride(1),
        BLOCK_HEADS=block_heads,
        BLOCK_PAIRS=block_pairs,
        BLOCK_TAIL=block_tail,
        EXACT=exact,
        num_warps=4,
    )
    return q_out, k_out


__all__ = ["ernie45_rope_fused"]
