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

"""Dequantize paged DSA KV-cache tokens back to bf16.

Same packed row layout as ``dequantize_k_cache_fast``::

    [dim_nope fp8 bytes | num_tiles fp32 scales | dim_rope bf16 elements]

The difference is that the source row is addressed indirectly through
``page_table_1_flattened``: the read index is ``page_table[token_id]``
while the write index is ``token_id``. ``num_tokens`` may exceed the
number of cache rows, because several sequences can share KV slots through
prefix caching, so the page table is not assumed to be injective.
"""

import torch
import triton
import triton.language as tl


def _infer_dims(dim_quant, group_size=128):
    """Recover ``(dim_nope, dim_rope)`` from a packed row width."""
    for dim_rope in (0, 64):
        scaled_width = (dim_quant - 2 * dim_rope) * group_size
        if scaled_width % (group_size + 4):
            continue
        dim_nope = scaled_width // (group_size + 4)
        if dim_nope > 0 and dim_nope % group_size == 0:
            return dim_nope, dim_rope
    raise ValueError(f"Unsupported packed DSA KV row width: {dim_quant}")


@triton.jit
def _dequantize_k_cache_paged_kernel(
    out_ptr,
    q_ptr,
    s_ptr,
    r_ptr,
    page_ptr,
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
    src_row = tl.load(page_ptr + token_id).to(tl.int64)
    block_id = tl.program_id(1)

    if block_id < NUM_NOPE_BLOCKS:
        offs = block_id * GROUP_SIZE + tl.arange(0, GROUP_SIZE)
        mask = offs < DIM_NOPE
        q = tl.load(q_ptr + src_row * q_s0 + offs, mask=mask, other=0.0)
        scale = tl.load(s_ptr + src_row * s_s0 + block_id)
        y = (q.to(tl.float32) * scale).to(out_ptr.dtype.element_ty)
        tl.store(out_ptr + token_id * out_s0 + offs, y, mask=mask)
    else:
        rope_block = block_id - NUM_NOPE_BLOCKS
        offs = rope_block * GROUP_SIZE + tl.arange(0, GROUP_SIZE)
        mask = offs < DIM_ROPE
        src = r_ptr + src_row * r_s0 + offs
        dst = out_ptr + token_id * out_s0 + DIM_NOPE + offs
        tl.store(dst, tl.load(src, mask=mask).to(tl.bfloat16), mask=mask)


def dequantize_k_cache_paged(
    quant_k_cache, page_table_1_flattened, group_size=128
):
    """Dequantize paged DSA k-cache tokens.

    Args:
        quant_k_cache: ``float8_e4m3fn`` tensor in paged layout, either
            ``[total_tokens, 1, dim_quant]`` or
            ``[num_blocks, block_size, 1, dim_quant]``.
        page_table_1_flattened: 1-D integer tensor of shape ``[num_tokens]``
            holding the source row of each token.
        group_size: NoPE columns covered by one fp32 scale.

    Returns:
        ``bfloat16`` tensor of shape ``[num_tokens, 1, dim_nope + dim_rope]``.
    """
    if quant_k_cache.dtype != torch.float8_e4m3fn:
        raise ValueError(
            "dequantize_k_cache_paged expects a float8_e4m3fn tensor, "
            f"got {quant_k_cache.dtype}"
        )

    dim_quant = quant_k_cache.shape[-1]
    dim_nope, dim_rope = _infer_dims(dim_quant, group_size)
    num_tiles = dim_nope // group_size
    num_tokens = page_table_1_flattened.shape[0]

    rows = quant_k_cache.reshape(-1, dim_quant)
    out = torch.empty(
        (num_tokens, dim_nope + dim_rope),
        dtype=torch.bfloat16,
        device=rows.device,
    )

    nope_q = rows[:, :dim_nope]
    nope_s = rows[:, dim_nope : dim_nope + num_tiles * 4].view(torch.float32)
    rope = rows[:, dim_nope + num_tiles * 4 :].view(torch.bfloat16)

    num_blocks = num_tiles + triton.cdiv(dim_rope, group_size)
    _dequantize_k_cache_paged_kernel[(num_tokens, num_blocks)](
        out,
        nope_q,
        nope_s,
        rope,
        page_table_1_flattened,
        out.stride(0),
        nope_q.stride(0),
        nope_s.stride(0),
        rope.stride(0),
        NUM_NOPE_BLOCKS=num_tiles,
        GROUP_SIZE=group_size,
        DIM_NOPE=dim_nope,
        DIM_ROPE=dim_rope,
    )

    return out.view(num_tokens, 1, dim_nope + dim_rope)


__all__ = ["dequantize_k_cache_paged"]
