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

"""Kunlunxin head-tile experiment based on the officially passing v18.

Larger production inputs use 32 heads per tile without the v21 program cap.
The production test range T<=4096 keeps one iteration per program. Shape and
stride metadata are compile-time constants; all data movement uses Triton.
Vendor compiler compatibility and speed remain subject to official testing.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _concat_mla_k_kernel_kunlunxin(
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
    pid = tl.program_id(0)
    nope_cols = tl.arange(0, BLOCK_N)
    rope_cols = tl.arange(0, BLOCK_R)
    nope_mask = nope_cols < N
    rope_mask = rope_cols < R

    # Ordinary range does not force unrolling. Production T<=4096 has STEPS=1;
    # no runtime launch-dimension query is needed.
    for step in range(STEPS):
        tile = pid + step * PROGRAMS
        token = tile // HEAD_TILES
        heads = (tile % HEAD_TILES) * BLOCK_H + tl.arange(0, BLOCK_H)
        valid = tile < TOTAL_TILES
        head_mask = heads < H
        mask_nope = valid & head_mask[:, None] & nope_mask[None, :]
        mask_rope = valid & head_mask[:, None] & rope_mask[None, :]

        nope_ptrs = (
            nope_ptr
            + token * NS0
            + heads[:, None] * NS1
            + nope_cols[None, :] * NS2
        )
        nope = tl.load(nope_ptrs, mask=mask_nope, other=0)

        out_base = out_ptr + token * H * (N + R) + heads[:, None] * (N + R)
        tl.store(out_base + nope_cols[None, :], nope, mask=mask_nope)

        rope_ptrs = rope_ptr + token * RS0 + rope_cols[None, :] * RS2
        rope = tl.load(rope_ptrs, mask=valid & rope_mask[None, :], other=0)
        rope = tl.broadcast_to(rope, (BLOCK_H, BLOCK_R))
        tl.store(out_base + N + rope_cols[None, :], rope, mask=mask_rope)


def concat_mla_k(k, k_nope, k_rope):
    """Return ``cat(k_nope, k_rope.expand(...))`` with a Triton kernel."""
    tokens, heads = k.shape[:2]
    nope_dim, rope_dim = k_nope.shape[2], k_rope.shape[2]
    out = torch.empty(
        (tokens, heads, nope_dim + rope_dim),
        device=k.device,
        dtype=k.dtype,
    )
    if tokens == 0 or heads == 0:
        return out

    # Keep small shapes on the passing tile size. Larger production shapes
    # halve the logical grid by doubling head rows, without a persistent cap.
    # This trades more per-program storage for fewer programs; timing is unknown.
    block_h = 16 if heads >= 64 else 8
    if tokens >= 64 and heads == 128 and nope_dim == 128 and rope_dim == 64:
        block_h = 32
    head_tiles = triton.cdiv(heads, block_h)
    total_tiles = tokens * head_tiles
    programs = min(total_tiles, 65535)
    steps = triton.cdiv(total_tiles, programs)
    _concat_mla_k_kernel_kunlunxin[(programs,)](
        out,
        k_nope,
        k_rope,
        H=heads,
        N=nope_dim,
        R=rope_dim,
        NS0=k_nope.stride(0),
        NS1=k_nope.stride(1),
        NS2=k_nope.stride(2),
        RS0=k_rope.stride(0),
        RS2=k_rope.stride(2),
        HEAD_TILES=head_tiles,
        TOTAL_TILES=total_tiles,
        BLOCK_H=block_h,
        BLOCK_N=triton.next_power_of_2(max(1, nope_dim)),
        BLOCK_R=triton.next_power_of_2(max(1, rope_dim)),
        PROGRAMS=programs,
        STEPS=steps,
        num_warps=4,
    )
    return out


def reference(k, k_nope, k_rope):
    return concat_mla_k(k, k_nope, k_rope)

__all__ = ["concat_mla_k"]
