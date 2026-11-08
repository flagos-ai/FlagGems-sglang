# Copyright 2026, The FlagOS Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied. See the License for the specific language governing
# permissions and limitations under the License.

"""Dequantize a packed DSA KV-cache row back to bf16.

A row packs three segments back to back::

    [dim_nope fp8 bytes | num_tiles fp32 scales | dim_rope bf16 elements]

with ``num_tiles = dim_nope // group_size``. One program handles one
``group_size`` slice of one row: the first ``num_tiles`` block ids
dequantize NoPE with the fp32 scale covering them, the remaining block ids
copy the RoPE segment, which is already stored as bf16. The two segments
have different element widths, so they cannot share a single typed
pointer; each segment gets its own base pointer and row stride.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _dequantize_k_cache_fast_kernel(
    out_ptr,
    q_ptr,
    s_ptr,
    r_ptr,
    out_s0,
    q_s0,
    s_s0,
    r_s0,
    NUM_NOPE_BLOCKS: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    DIM_NOPE: tl.constexpr,
    DIM_ROPE: tl.constexpr,
):
    token_id = tl.program_id(0).to(tl.int64)
    block_id = tl.program_id(1)

    if block_id < NUM_NOPE_BLOCKS:
        offs = block_id * GROUP_SIZE + tl.arange(0, GROUP_SIZE)
        mask = offs < DIM_NOPE
        q = tl.load(q_ptr + token_id * q_s0 + offs, mask=mask, other=0.0)
        scale = tl.load(s_ptr + token_id * s_s0 + block_id)
        y = (q.to(tl.float32) * scale).to(out_ptr.dtype.element_ty)
        tl.store(out_ptr + token_id * out_s0 + offs, y, mask=mask)
    else:
        rope_block = block_id - NUM_NOPE_BLOCKS
        offs = rope_block * GROUP_SIZE + tl.arange(0, GROUP_SIZE)
        mask = offs < DIM_ROPE
        src = r_ptr + token_id * r_s0 + offs
        dst = out_ptr + token_id * out_s0 + DIM_NOPE + offs
        tl.store(dst, tl.load(src, mask=mask).to(tl.bfloat16), mask=mask)


def dequantize_k_cache_fast(quant_k_cache, dim_nope, dim_rope, group_size=128):
    """Dequantize packed DSA k-cache rows with direct row addressing.

    Args:
        quant_k_cache: ``float8_e4m3fn`` tensor, either
            ``[tokens, dim_quant]`` or ``[blocks, block_size, dim_quant]``.
        dim_nope: NoPE width; must be a multiple of ``group_size``.
        dim_rope: RoPE width stored uncompressed in the row.
        group_size: NoPE columns covered by one fp32 scale.

    Returns:
        ``bfloat16`` tensor of shape ``[..., dim_nope + dim_rope]``.
    """
    if quant_k_cache.dtype != torch.float8_e4m3fn:
        raise ValueError(
            "dequantize_k_cache_fast expects a float8_e4m3fn tensor, "
            f"got {quant_k_cache.dtype}"
        )
    if dim_nope % group_size != 0:
        raise ValueError(
            f"dim_nope ({dim_nope}) must be a multiple of "
            f"group_size ({group_size})"
        )

    dim_quant = quant_k_cache.shape[-1]
    num_tiles = dim_nope // group_size
    expected = dim_nope + num_tiles * 4 + dim_rope * 2
    if dim_quant != expected:
        raise ValueError(
            f"packed row width {dim_quant} does not match dim_nope="
            f"{dim_nope}, dim_rope={dim_rope}, group_size={group_size}"
        )

    rows = quant_k_cache.reshape(-1, dim_quant)
    num_tokens = rows.shape[0]
    out = torch.empty(
        (num_tokens, dim_nope + dim_rope),
        dtype=torch.bfloat16,
        device=rows.device,
    )

    nope_q = rows[:, :dim_nope]
    nope_s = rows[:, dim_nope : dim_nope + num_tiles * 4].view(torch.float32)
    rope = rows[:, dim_nope + num_tiles * 4 :].view(torch.bfloat16)

    num_blocks = num_tiles + triton.cdiv(dim_rope, group_size)
    _dequantize_k_cache_fast_kernel[(num_tokens, num_blocks)](
        out,
        nope_q,
        nope_s,
        rope,
        out.stride(0),
        nope_q.stride(0),
        nope_s.stride(0),
        rope.stride(0),
        NUM_NOPE_BLOCKS=num_tiles,
        GROUP_SIZE=group_size,
        DIM_NOPE=dim_nope,
        DIM_ROPE=dim_rope,
    )

    return out.view(*quant_k_cache.shape[:-1], dim_nope + dim_rope)


__all__ = ["dequantize_k_cache_fast"]
