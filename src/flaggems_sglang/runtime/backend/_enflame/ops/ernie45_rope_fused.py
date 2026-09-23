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
def _ernie45_rope_qk_one(
    q_ptr,
    k_ptr,
    q_out_ptr,
    k_out_ptr,
    cos_sin_cache_ptr,
    positions_ptr,
    NUM_TOKENS: tl.constexpr,
    NUM_Q_HEADS: tl.constexpr,
    NUM_K_HEADS: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    ROTARY_DIM: tl.constexpr,
    SECTION_HW: tl.constexpr,
    BLOCK_Q_HEADS: tl.constexpr,
    BLOCK_K_HEADS: tl.constexpr,
    BLOCK_PAIRS: tl.constexpr,
    BLOCK_TAIL: tl.constexpr,
):
    token = tl.program_id(0)
    half_rotary: tl.constexpr = ROTARY_DIM // 2
    q_stride: tl.constexpr = NUM_Q_HEADS * HEAD_SIZE
    k_stride: tl.constexpr = NUM_K_HEADS * HEAD_SIZE
    cache_stride: tl.constexpr = ROTARY_DIM
    pos_axis: tl.constexpr = NUM_TOKENS
    tpos = tl.load(positions_ptr + token)
    hpos = tl.load(positions_ptr + pos_axis + token)
    wpos = tl.load(positions_ptr + 2 * pos_axis + token)
    pair = tl.arange(0, BLOCK_PAIRS)
    pair_mask = pair < half_rotary
    cos_t = tl.load(
        cos_sin_cache_ptr + tpos * cache_stride + pair,
        mask=pair_mask,
        other=0.0,
    ).to(tl.float32)
    sin_t = tl.load(
        cos_sin_cache_ptr + tpos * cache_stride + half_rotary + pair,
        mask=pair_mask,
        other=0.0,
    ).to(tl.float32)
    cos_h = tl.load(
        cos_sin_cache_ptr + hpos * cache_stride + pair,
        mask=pair_mask,
        other=0.0,
    ).to(tl.float32)
    sin_h = tl.load(
        cos_sin_cache_ptr + hpos * cache_stride + half_rotary + pair,
        mask=pair_mask,
        other=0.0,
    ).to(tl.float32)
    cos_w = tl.load(
        cos_sin_cache_ptr + wpos * cache_stride + pair,
        mask=pair_mask,
        other=0.0,
    ).to(tl.float32)
    sin_w = tl.load(
        cos_sin_cache_ptr + wpos * cache_stride + half_rotary + pair,
        mask=pair_mask,
        other=0.0,
    ).to(tl.float32)
    use_hw = pair < SECTION_HW
    use_height = pair % 2 == 0
    cos = tl.where(use_hw, tl.where(use_height, cos_h, cos_w), cos_t)
    sin = tl.where(use_hw, tl.where(use_height, sin_h, sin_w), sin_t)

    q_heads = tl.arange(0, BLOCK_Q_HEADS)
    q_mask = (q_heads[:, None] < NUM_Q_HEADS) & pair_mask[None, :]
    q_row = token * q_stride + q_heads[:, None] * HEAD_SIZE
    q1 = tl.load(q_ptr + q_row + pair[None, :], mask=q_mask, other=0.0).to(
        tl.float32
    )
    q2 = tl.load(
        q_ptr + q_row + half_rotary + pair[None, :], mask=q_mask, other=0.0
    ).to(tl.float32)
    tl.store(
        q_out_ptr + q_row + pair[None, :],
        q1 * cos[None, :] - q2 * sin[None, :],
        mask=q_mask,
    )
    tl.store(
        q_out_ptr + q_row + half_rotary + pair[None, :],
        q2 * cos[None, :] + q1 * sin[None, :],
        mask=q_mask,
    )

    k_heads = tl.arange(0, BLOCK_K_HEADS)
    k_mask = (k_heads[:, None] < NUM_K_HEADS) & pair_mask[None, :]
    k_row = token * k_stride + k_heads[:, None] * HEAD_SIZE
    k1 = tl.load(k_ptr + k_row + pair[None, :], mask=k_mask, other=0.0).to(
        tl.float32
    )
    k2 = tl.load(
        k_ptr + k_row + half_rotary + pair[None, :], mask=k_mask, other=0.0
    ).to(tl.float32)
    tl.store(
        k_out_ptr + k_row + pair[None, :],
        k1 * cos[None, :] - k2 * sin[None, :],
        mask=k_mask,
    )
    tl.store(
        k_out_ptr + k_row + half_rotary + pair[None, :],
        k2 * cos[None, :] + k1 * sin[None, :],
        mask=k_mask,
    )

    if HEAD_SIZE > ROTARY_DIM:
        tail = tl.arange(0, BLOCK_TAIL)
        span = tail < HEAD_SIZE - ROTARY_DIM
        q_tm = (q_heads[:, None] < NUM_Q_HEADS) & span[None, :]
        q_toff = q_row + ROTARY_DIM + tail[None, :]
        tl.store(
            q_out_ptr + q_toff, tl.load(q_ptr + q_toff, mask=q_tm), mask=q_tm
        )
        k_tm = (k_heads[:, None] < NUM_K_HEADS) & span[None, :]
        k_toff = k_row + ROTARY_DIM + tail[None, :]
        tl.store(
            k_out_ptr + k_toff, tl.load(k_ptr + k_toff, mask=k_tm), mask=k_tm
        )


def _pow2(n):
    n = max(int(n), 1)
    return 1 << (n - 1).bit_length()


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
    q_out = torch.empty_like(q)
    k_out = torch.empty_like(k)
    positions_i32 = positions.to(torch.int32)
    _ernie45_rope_qk_one[(num_tokens,)](
        q,
        k,
        q_out,
        k_out,
        cos_sin_cache,
        positions_i32,
        NUM_TOKENS=num_tokens,
        NUM_Q_HEADS=q.shape[1] // head_size,
        NUM_K_HEADS=k.shape[1] // head_size,
        HEAD_SIZE=head_size,
        ROTARY_DIM=rotary_dim,
        SECTION_HW=int(mrope_section[0]) + int(mrope_section[1]),
        BLOCK_Q_HEADS=_pow2(q.shape[1] // head_size),
        BLOCK_K_HEADS=_pow2(k.shape[1] // head_size),
        BLOCK_PAIRS=_pow2(rotary_dim // 2),
        BLOCK_TAIL=_pow2(max(1, head_size - rotary_dim)),
        num_warps=8,
        num_stages=1,
    )
    return q_out, k_out


__all__ = ["ernie45_rope_fused"]
