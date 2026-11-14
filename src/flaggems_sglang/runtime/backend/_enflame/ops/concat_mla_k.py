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

"""Enflame candidate: exact-shape persistent kernel plus a bounded fallback.

The exact kernel serves contiguous production layouts: one persistent
program writes whole ``N + R`` output rows as a single contiguous masked
store (no write expansion at the 384-byte row pitch), with NoPE and RoPE
values blended by a bitwise OR of complementary constant-masked loads —
bit-exact for every payload including ``-0.0`` and NaN.  The launch stays
at ``min(48, tiles)`` programs with ``num_warps=1`` (GCU400 wide-vector
streaming sweet spot).  Anything irregular (non-contiguous NoPE, odd head
counts, oversized tiles, very long per-program loops) falls back to the
original bounded-grid masked kernel.

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
            nope_ptr
            + token * NS0
            + heads[:, None] * NS1
            + nope_cols[None, :] * NS2,
            mask=valid & (heads[:, None] < H) & (nope_cols[None, :] < N),
            other=0,
        )
        tl.store(
            out_ptr
            + token * H * (N + R)
            + heads[:, None] * (N + R)
            + nope_cols[None, :],
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
            out_ptr
            + token * H * (N + R)
            + heads[:, None] * (N + R)
            + N
            + rope_cols[None, :],
            rope,
            mask=valid & (heads[:, None] < H) & (rope_cols[None, :] < R),
        )


@triton.jit
def _concat_mla_k_exact_kernel(
    out_ptr,
    nope_ptr,
    rope_ptr,
    H: tl.constexpr,
    N: tl.constexpr,
    R: tl.constexpr,
    RS0: tl.constexpr,
    W: tl.constexpr,
    ROWS: tl.constexpr,
    TILES,
    PROGRAMS,
    IT: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = tl.arange(0, ROWS)
    cols = tl.arange(0, W)
    mask_nope = cols < N
    mask_rope = (cols >= N) & (cols < N + R)
    mask_out = cols < N + R
    for tile in tl.range(pid, TILES, PROGRAMS):
        r0 = tile * ROWS
        token = r0 // H
        nope = tl.load(
            nope_ptr + (r0 + rows)[:, None] * N + cols[None, :],
            mask=mask_nope[None, :],
            other=0,
        )
        rope = tl.load(
            rope_ptr + token * RS0 + (cols[None, :] - N),
            mask=mask_rope[None, :],
            other=0,
        )
        blended = (
            (nope.to(IT, bitcast=True) | rope.to(IT, bitcast=True))
            .to(nope.dtype, bitcast=True)
        )
        tl.store(
            out_ptr + (r0 + rows)[:, None] * (N + R) + cols[None, :],
            blended,
            mask=mask_out[None, :],
        )


def _launch_exact(out, k_nope, k_rope, tokens, heads, nope_dim, rope_dim):
    if not k_nope.is_contiguous() or k_rope.stride(2) != 1:
        return False
    if not (out.dtype.is_floating_point and k_nope.dtype == out.dtype
            and k_rope.dtype == out.dtype):
        return False
    elem = out.element_size()
    if elem not in (2, 4):
        return False
    int_type = tl.int16 if elem == 2 else tl.int32
    if tokens * heads * (nope_dim + rope_dim) >= 2**31:
        return False
    width = triton.next_power_of_2(nope_dim + rope_dim)
    if width > 1024:
        return False
    rows = 128
    while rows > 8 and (heads % rows != 0 or rows * width * elem > 65536):
        rows //= 2
    if heads % rows != 0 or rows * width * elem > 65536:
        return False
    tiles = (tokens * heads) // rows
    while tiles < 8 and rows > 16:
        rows //= 2
        tiles = (tokens * heads) // rows
    programs = min(48, tiles)
    if tiles // programs > 256:
        return False
    _concat_mla_k_exact_kernel[(programs,)](
        out,
        k_nope,
        k_rope,
        H=heads,
        N=nope_dim,
        R=rope_dim,
        RS0=k_rope.stride(0),
        W=width,
        ROWS=rows,
        TILES=tiles,
        PROGRAMS=programs,
        IT=int_type,
        num_warps=1,
    )
    return True


def concat_mla_k(k, k_nope, k_rope):
    """Copy NoPE and broadcast RoPE into new storage; all inputs are read-only."""
    tokens, heads = k.shape[:2]
    nope_dim, rope_dim = k_nope.shape[2], k_rope.shape[2]
    out = torch.empty(
        (tokens, heads, nope_dim + rope_dim), device=k.device, dtype=k.dtype
    )
    if tokens == 0 or heads == 0:
        return out
    if _launch_exact(out, k_nope, k_rope, tokens, heads, nope_dim, rope_dim):
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
        STEPS=triton.cdiv(total_tiles, programs),
        num_warps=4,
    )
    return out


def reference(k, k_nope, k_rope):
    return concat_mla_k(k, k_nope, k_rope)


__all__ = ["concat_mla_k"]
