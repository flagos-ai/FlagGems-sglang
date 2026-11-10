# Copyright 2026, The FlagOS Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Triton implementation of kv_cache/quantize_k_cache_fast.

A packed DSA KV-cache row holds three segments back to back::

    [dim_nope fp8 bytes | num_tiles fp32 scales | dim_rope bf16 elements]

with ``num_tiles = dim_nope // group_size``, so a row is
``dim_nope + num_tiles * 4 + dim_rope * 2`` bytes wide. This is the write
side of the layout that ``dequantize_k_cache_fast`` reads back.

The NoPE segment is quantized one ``group_size`` tile at a time with a single
fp32 scale per tile, ``scale = amax(|x|) / 448.0`` where 448 is the largest
magnitude ``float8_e4m3fn`` can represent. The RoPE segment is not quantized
at all: its bytes are copied through into the row.
"""

import torch
import triton
import triton.language as tl

__all__ = ["quantize_k_cache_fast"]

# The largest finite magnitude of float8_e4m3fn. The scale is chosen so that
# the widest element in each tile lands exactly on it.
_FP8_MAX = 448.0
_FP8_MIN = -448.0


@triton.jit
def _quantize_kernel(
    k_nope_ptr,
    k_rope_ptr,
    nope_q_ptr,
    nope_s_ptr,
    rope_out_ptr,
    k_nope_s0,
    k_rope_s0,
    nope_q_s0,
    nope_s_s0,
    rope_out_s0,
    NUM_NOPE_BLOCKS: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    DIM_NOPE: tl.constexpr,
    DIM_ROPE: tl.constexpr,
    FP8_MIN: tl.constexpr,
    FP8_MAX: tl.constexpr,
):
    """One program handles one tile of one row.

    The first ``NUM_NOPE_BLOCKS`` block ids quantize a NoPE tile and write its
    scale; the remaining ids copy a RoPE tile, which is already in the output
    dtype and needs no arithmetic.
    """
    token_id = tl.program_id(0).to(tl.int64)
    raw_block = tl.program_id(1)

    if raw_block < NUM_NOPE_BLOCKS:
        offs = raw_block * GROUP_SIZE + tl.arange(0, GROUP_SIZE)
        mask = offs < DIM_NOPE

        x = tl.load(
            k_nope_ptr + token_id * k_nope_s0 + offs, mask=mask, other=0.0
        ).to(tl.float32)

        # The upstream reference divides by an unguarded scale, so a tile that
        # is entirely zero produces nan rather than zero here as well. Guarding
        # it would be friendlier but would stop matching the contract.
        scale = tl.max(tl.abs(x)) / FP8_MAX
        inv_scale = 1.0 / scale

        # Mathematically this clamp is a no-op: every |x| is at most the tile
        # maximum the scale was derived from. It is kept because the reference
        # kernel keeps it, covering the case where the fp32 round-off pushes a
        # value a hair past 448.
        q = tl.minimum(tl.maximum(x * inv_scale, FP8_MIN), FP8_MAX).to(
            nope_q_ptr.dtype.element_ty
        )

        tl.store(nope_q_ptr + token_id * nope_q_s0 + offs, q, mask=mask)
        tl.store(nope_s_ptr + token_id * nope_s_s0 + raw_block, scale)
    else:
        block = raw_block - NUM_NOPE_BLOCKS
        offs = block * GROUP_SIZE + tl.arange(0, GROUP_SIZE)
        mask = offs < DIM_ROPE

        data = tl.load(
            k_rope_ptr + token_id * k_rope_s0 + offs, mask=mask, other=0.0
        )
        tl.store(rope_out_ptr + token_id * rope_out_s0 + offs, data, mask=mask)


def _packed_row_width(dim_nope, dim_rope, group_size, rope_bytes):
    return dim_nope + (dim_nope // group_size) * 4 + rope_bytes * dim_rope


def quantize_k_cache_fast(k_nope, k_rope, group_size=128):
    """Quantize the NoPE keys to fp8 and pack the rotary keys alongside.

    Args:
        k_nope: ``(tokens, dim_nope)`` or ``(tokens, 1, dim_nope)`` bf16.
            ``dim_nope`` must be a multiple of ``group_size``.
        k_rope: ``(tokens, dim_rope)`` or ``(tokens, 1, dim_rope)`` bf16, same
            token count. A zero-width rotary part is allowed.
        group_size: NoPE columns covered by one fp32 scale. Must be a power of
            two, since one tile is loaded with a single ``tl.arange``.

    Returns:
        A ``float8_e4m3fn`` tensor whose rows are
        ``[dim_nope fp8 | num_tiles fp32 scales | dim_rope bf16 bytes]``, so the
        byte width is ``dim_nope + num_tiles * 4 + dim_rope * 2``. The rank of
        the input is preserved: a 3-D input comes back 3-D with a singleton
        head axis, which is the shape ``dequantize_k_cache_fast`` and the
        MLA KV-buffer writer both expect.
    """
    if k_nope.ndim not in (2, 3):
        raise ValueError(
            "k_nope must be 2D (tokens, dim_nope) or 3D (tokens, 1, dim_nope)"
        )
    if k_rope.ndim not in (2, 3):
        raise ValueError(
            "k_rope must be 2D (tokens, dim_rope) or 3D (tokens, 1, dim_rope)"
        )
    if k_nope.dtype != torch.bfloat16 or k_rope.dtype != torch.bfloat16:
        raise ValueError("k_nope and k_rope must both be bfloat16")
    if k_nope.device != k_rope.device:
        raise ValueError("k_nope and k_rope must be on the same device")

    k_nope_2d = k_nope.squeeze(1) if k_nope.ndim == 3 else k_nope
    k_rope_2d = k_rope.squeeze(1) if k_rope.ndim == 3 else k_rope
    if k_nope.ndim == 3 and k_nope.shape[1] != 1:
        raise ValueError(f"k_nope head axis must be 1, got {k_nope.shape[1]}")
    if k_rope.ndim == 3 and k_rope.shape[1] != 1:
        raise ValueError(f"k_rope head axis must be 1, got {k_rope.shape[1]}")

    if isinstance(group_size, bool) or not isinstance(group_size, int):
        raise ValueError(f"group_size must be an int, got {group_size!r}")
    if group_size <= 0 or (group_size & (group_size - 1)) != 0:
        raise ValueError(
            f"group_size must be a power of two, got {group_size}"
        )

    num_tokens, dim_nope = k_nope_2d.shape
    if k_rope_2d.shape[0] != num_tokens:
        raise ValueError(
            f"k_nope has {num_tokens} tokens, k_rope has {k_rope_2d.shape[0]}"
        )
    dim_rope = int(k_rope_2d.shape[1])
    if dim_nope % group_size != 0:
        raise ValueError(
            f"dim_nope ({dim_nope}) must be a multiple of "
            f"group_size ({group_size})"
        )
    if k_nope_2d.stride(1) != 1 or k_rope_2d.stride(1) != 1:
        raise ValueError(
            "k_nope and k_rope must be contiguous on the last dim"
        )

    num_tiles = dim_nope // group_size
    rope_bytes = k_rope_2d.element_size()
    packed_width = _packed_row_width(
        dim_nope, dim_rope, group_size, rope_bytes
    )

    # One fp8-typed allocation holds all three segments; the scale and RoPE
    # segments are re-typed views over their byte ranges, the same way the
    # reader reinterprets them.
    packed = torch.empty(
        (num_tokens, packed_width),
        dtype=torch.float8_e4m3fn,
        device=k_nope.device,
    )
    nope_q = packed[:, :dim_nope]
    nope_s = packed[:, dim_nope : dim_nope + num_tiles * 4].view(torch.float32)
    if dim_rope > 0:
        rope_out = packed[:, dim_nope + num_tiles * 4 :].view(torch.bfloat16)
    else:
        rope_out = torch.empty(
            (num_tokens, 0), dtype=torch.bfloat16, device=k_nope.device
        )

    # A zero-width rotary part contributes no block; otherwise its blocks come
    # after the NoPE ones so a single grid dimension covers both segments.
    num_blocks = num_tiles + (
        triton.cdiv(dim_rope, group_size) if dim_rope else 0
    )

    if num_tokens > 0 and num_blocks > 0:
        _quantize_kernel[(num_tokens, num_blocks)](
            k_nope_2d,
            k_rope_2d,
            nope_q,
            nope_s,
            rope_out,
            k_nope_2d.stride(0),
            k_rope_2d.stride(0) if dim_rope > 0 else 0,
            nope_q.stride(0),
            nope_s.stride(0),
            rope_out.stride(0) if dim_rope > 0 else 0,
            NUM_NOPE_BLOCKS=num_tiles,
            GROUP_SIZE=group_size,
            DIM_NOPE=dim_nope,
            DIM_ROPE=dim_rope,
            FP8_MIN=_FP8_MIN,
            FP8_MAX=_FP8_MAX,
        )

    if k_nope.ndim == 3:
        return packed.view(num_tokens, 1, packed_width)
    return packed
