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

"""Hygon specialization: compile-time shape and stride metadata.

The first argument is part of the public benchmark signature and supplies the
output shape/dtype.  It is intentionally not modified: the result is written
to a newly allocated tensor, matching the benchmark reference implementation.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _concat_mla_k_kernel(
    out_ptr,
    nope_ptr,
    rope_ptr,
    total_tokens: tl.constexpr,
    num_heads: tl.constexpr,
    nope_dim: tl.constexpr,
    rope_dim: tl.constexpr,
    out_s0: tl.constexpr,
    out_s1: tl.constexpr,
    out_s2: tl.constexpr,
    nope_s0: tl.constexpr,
    nope_s1: tl.constexpr,
    nope_s2: tl.constexpr,
    rope_s0: tl.constexpr,
    rope_s1: tl.constexpr,
    rope_s2: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    """Copy a token tile and reuse its single RoPE row for every head."""
    head_tiles = tl.cdiv(num_heads, BLOCK_H)
    total_tiles = total_tokens * head_tiles
    nope_cols = tl.arange(0, BLOCK_N)
    rope_cols = tl.arange(0, BLOCK_R)
    nope_mask = nope_cols < nope_dim
    rope_mask = rope_cols < rope_dim

    # Each tile belongs to exactly one program, including tiles beyond grid.x.
    for tile in range(tl.program_id(0), total_tiles, tl.num_programs(0)):
        token = tile // head_tiles
        head_tile = tile % head_tiles
        heads = head_tile * BLOCK_H + tl.arange(0, BLOCK_H)
        head_mask = heads < num_heads

        nope_ptrs = (
            nope_ptr
            + token * nope_s0
            + heads[:, None] * nope_s1
            + nope_cols[None, :] * nope_s2
        )
        nope = tl.load(nope_ptrs, mask=head_mask[:, None] & nope_mask[None, :])
        out_nope_ptrs = (
            out_ptr
            + token * out_s0
            + heads[:, None] * out_s1
            + nope_cols[None, :] * out_s2
        )
        tl.store(out_nope_ptrs, nope, mask=head_mask[:, None] & nope_mask[None, :])

        rope_ptrs = rope_ptr + token * rope_s0 + rope_cols[None, :] * rope_s2
        rope = tl.load(rope_ptrs, mask=rope_mask[None, :])
        rope = tl.broadcast_to(rope, (BLOCK_H, BLOCK_R))
        out_rope_ptrs = (
            out_ptr
            + token * out_s0
            + heads[:, None] * out_s1
            + (nope_dim + rope_cols)[None, :] * out_s2
        )
        tl.store(out_rope_ptrs, rope, mask=head_mask[:, None] & rope_mask[None, :])


def concat_mla_k(k, k_nope, k_rope):
    """Return ``cat(k_nope, k_rope.expand(...))`` using a Triton kernel."""
    total_tokens = k.shape[0]
    num_heads = k.shape[1]
    nope_dim = k_nope.shape[2]
    rope_dim = k_rope.shape[2]
    out = torch.empty(
        (total_tokens, num_heads, nope_dim + rope_dim),
        device=k.device,
        dtype=k.dtype,
    )
    # Enflame limits grid.y to 255; Ascend also limits total coreDim to 65535.
    # A bounded 1D launch satisfies both; the kernel loops over remaining tiles.
    total_tiles = total_tokens * triton.cdiv(num_heads, 8)
    if total_tiles == 0:
        return out
    grid = (min(total_tiles, 65535),)
    _concat_mla_k_kernel[grid](
        out,
        k_nope,
        k_rope,
        total_tokens,
        num_heads,
        nope_dim,
        rope_dim,
        out.stride(0),
        out.stride(1),
        out.stride(2),
        k_nope.stride(0),
        k_nope.stride(1),
        k_nope.stride(2),
        k_rope.stride(0),
        k_rope.stride(1),
        k_rope.stride(2),
        BLOCK_H=8,
        BLOCK_N=128,
        BLOCK_R=64,
        num_warps=4,
    )
    return out


def reference(k, k_nope, k_rope):
    """Compatibility entry point with the benchmark reference signature."""
    return concat_mla_k(k, k_nope, k_rope)

__all__ = ["concat_mla_k"]
