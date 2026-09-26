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

"""Enflame candidate: coarser tiles with a bounded one-dimensional grid.

Standard Triton only. This file is standalone and selected by its chip suffix.
Hardware performance must be established by the official evaluation.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _concat_mla_k_kernel_enflame(
    out_ptr,
    nope_ptr,
    rope_ptr,
    H: tl.constexpr,
    N: tl.constexpr,
    R: tl.constexpr,
    NS0: tl.constexpr,
    NS1: tl.constexpr,
    NS2: tl.constexpr,
    RS0: tl.constexpr,
    RS2: tl.constexpr,
    HEAD_TILES: tl.constexpr,
    TOTAL_TILES: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_R: tl.constexpr,
    PROGRAMS: tl.constexpr,
    STEPS: tl.constexpr,
):
    for step in range(STEPS):
        tile = tl.program_id(0) + step * PROGRAMS
        token = tile // HEAD_TILES
        heads = (tile % HEAD_TILES) * BLOCK_H + tl.arange(0, BLOCK_H)
        valid = tile < TOTAL_TILES
        nope_cols = tl.arange(0, BLOCK_N)
        rope_cols = tl.arange(0, BLOCK_R)

        nope = tl.load(
            nope_ptr + token * NS0 + heads[:, None] * NS1 + nope_cols[None, :] * NS2,
            mask=valid & (heads[:, None] < H) & (nope_cols[None, :] < N),
            other=0,
        )
        tl.store(
            out_ptr + token * H * (N + R) + heads[:, None] * (N + R) + nope_cols[None, :],
            nope,
            mask=valid & (heads[:, None] < H) & (nope_cols[None, :] < N),
        )
        rope = tl.load(
            rope_ptr + token * RS0 + rope_cols[None, :] * RS2,
            mask=valid & (rope_cols[None, :] < R),
            other=0,
        )
        rope = tl.broadcast_to(rope, (BLOCK_H, BLOCK_R))
        tl.store(
            out_ptr + token * H * (N + R) + heads[:, None] * (N + R) + N + rope_cols[None, :],
            rope,
            mask=valid & (heads[:, None] < H) & (rope_cols[None, :] < R),
        )


def concat_mla_k(k, k_nope, k_rope):
    """Copy NoPE and broadcast RoPE into new storage; all inputs are read-only."""
    tokens, heads = k.shape[:2]
    nope_dim, rope_dim = k_nope.shape[2], k_rope.shape[2]
    out = torch.empty((tokens, heads, nope_dim + rope_dim), device=k.device, dtype=k.dtype)
    if tokens == 0 or heads == 0:
        return out
    block_h = 8 if tokens < 16 else 32
    # One-parameter experiment on the measured v6 copy kernel. Retain its
    # small-input tiling; production-sized large batches use a whole head row.
    if tokens >= 256 and heads == 128 and nope_dim == 128 and rope_dim == 64:
        block_h = 128
    head_tiles = triton.cdiv(heads, block_h)
    total_tiles = tokens * head_tiles
    programs = min(total_tiles, 65535)
    grid = (programs,)
    _concat_mla_k_kernel_enflame[grid](
        out, k_nope, k_rope,
        H=heads, N=nope_dim, R=rope_dim,
        NS0=k_nope.stride(0), NS1=k_nope.stride(1), NS2=k_nope.stride(2),
        RS0=k_rope.stride(0), RS2=k_rope.stride(2),
        HEAD_TILES=head_tiles, TOTAL_TILES=total_tiles,
        BLOCK_H=block_h,
        BLOCK_N=triton.next_power_of_2(max(1, nope_dim)),
        BLOCK_R=triton.next_power_of_2(max(1, rope_dim)),
        PROGRAMS=programs,
        STEPS=triton.cdiv(total_tiles, programs),
        num_warps=4,
    )
    return out


def reference(k, k_nope, k_rope):
    return concat_mla_k(k, k_nope, k_rope)

__all__ = ["concat_mla_k"]
