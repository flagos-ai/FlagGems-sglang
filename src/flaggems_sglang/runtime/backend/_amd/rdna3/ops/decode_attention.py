# Copyright 2026 FlagOS Contributors
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

"""decode_attention for RDNA3 (gfx1100) — one decode step of paged attention
over the KV cache (Triton).

For 16-bit inputs the sequence is handed to the shared split-KV kernels, which
fill the 96 compute units even when there are few sequences and few KV heads.
float32 keeps the kernel without tiles below: its products run on the FMA
units, where a block of query heads only pays for itself once the block is
full, and the generic operator's tiled kernel asks for 68 KB of LDS, which is
more than gfx1100 has.

This kernel is written in portable Triton only. It must NOT call any
pre-compiled / vendor-specific cached operator (no
``F.scaled_dot_product_attention``, no ``_compiled`` handles, no
``torch.ops.*`` attention) — the whole computation is done inside the Triton
kernel.
"""

import torch
import triton
import triton.language as tl

from ._decode_splitkv import (
    decode_splitkv,
    needs_int64,
    supported,
    unit_head_stride,
)


@triton.jit
def _decode_attention_kernel(
    q,
    k_buffer,
    v_buffer,
    kv_indptr,
    kv_indices,
    output,
    sm_scale,
    total_work,
    stride_qb: tl.constexpr,
    stride_qh: tl.constexpr,
    stride_kt: tl.constexpr,
    stride_kh: tl.constexpr,
    stride_vt: tl.constexpr,
    stride_vh: tl.constexpr,
    H_Q: tl.constexpr,
    H_KV: tl.constexpr,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_N: tl.constexpr,
    INDEX64: tl.constexpr,
):
    program_id = tl.program_id(0)
    program_count = tl.num_programs(0)
    dim_offsets = tl.arange(0, BLOCK_D)
    dim_mask = dim_offsets < D
    token_lanes = tl.arange(0, BLOCK_N)
    for batch_head in tl.range(program_id, total_work, program_count):
        batch = batch_head // H_Q
        query_head = batch_head - batch * H_Q
        kv_head = query_head // (H_Q // H_KV)
        start = tl.load(kv_indptr + batch)
        end = tl.load(kv_indptr + batch + 1)
        sequence_length = end - start
        query = tl.load(
            q + batch * stride_qb + query_head * stride_qh + dim_offsets,
            mask=dim_mask,
            other=0.0,
        ).to(tl.float32)
        running_max = -float("inf")
        running_sum = 0.0
        accumulator = tl.zeros((BLOCK_D,), dtype=tl.float32)
        for block_start in tl.range(0, sequence_length, BLOCK_N):
            token_offsets = block_start + token_lanes
            token_mask = token_offsets < sequence_length
            token_indices = tl.load(
                kv_indices + start + token_offsets,
                mask=token_mask,
                other=0,
            )
            if INDEX64:
                token_indices = token_indices.to(tl.int64)
            keys = tl.load(
                k_buffer
                + token_indices[:, None] * stride_kt
                + kv_head * stride_kh
                + dim_offsets[None, :],
                mask=token_mask[:, None] & dim_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            logits = tl.sum(keys * query[None, :], axis=1) * sm_scale
            logits = tl.where(token_mask, logits, -float("inf"))
            block_max = tl.max(logits, axis=0)
            next_max = tl.maximum(running_max, block_max)
            correction = tl.exp(running_max - next_max)
            probabilities = tl.exp(logits - next_max)
            running_sum = running_sum * correction + tl.sum(
                probabilities, axis=0
            )
            values = tl.load(
                v_buffer
                + token_indices[:, None] * stride_vt
                + kv_head * stride_vh
                + dim_offsets[None, :],
                mask=token_mask[:, None] & dim_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            accumulator = accumulator * correction + tl.sum(
                probabilities[:, None] * values,
                axis=0,
            )
            running_max = next_max
        tl.store(
            output + batch_head * D + dim_offsets,
            accumulator / running_sum,
            mask=dim_mask,
        )


def _launch_direct(q, k_buffer, v_buffer, kv_indptr, kv_indices, sm_scale):
    q, k_buffer, v_buffer = unit_head_stride(q, k_buffer, v_buffer)
    batch_size, query_heads, head_dim = q.shape
    kv_heads = k_buffer.shape[1]
    block_dim = triton.next_power_of_2(head_dim)
    block_tokens = 16 if block_dim >= 512 else 32 if block_dim >= 128 else 64
    # a 512-wide head leaves little for a warp to do beyond its own row, unless
    # the row is float32 and twice the size
    wide = block_dim >= 512 and q.dtype != torch.float32
    total_work = batch_size * query_heads
    output = torch.empty(
        (batch_size, query_heads, v_buffer.shape[-1]),
        dtype=torch.float32,
        device=q.device,
    )
    q_strides, k_strides, v_strides = (
        q.stride(),
        k_buffer.stride(),
        v_buffer.stride(),
    )
    _decode_attention_kernel[(total_work,)](
        q,
        k_buffer,
        v_buffer,
        kv_indptr,
        kv_indices,
        output,
        sm_scale,
        total_work,
        q_strides[0],
        q_strides[1],
        k_strides[0],
        k_strides[1],
        v_strides[0],
        v_strides[1],
        query_heads,
        kv_heads,
        head_dim,
        block_dim,
        block_tokens,
        needs_int64(k_buffer, v_buffer, k_strides, v_strides),
        num_warps=2 if wide else 4,
        num_stages=2,
    )
    return output


def decode_attention(q, k_buffer, v_buffer, kv_indptr, kv_indices, sm_scale):
    query_heads = q.shape[1]
    group = query_heads // k_buffer.shape[1]
    if supported(q, k_buffer, v_buffer, group):
        return decode_splitkv(
            q, k_buffer, v_buffer, kv_indptr, kv_indices, sm_scale
        )
    return _launch_direct(
        q, k_buffer, v_buffer, kv_indptr, kv_indices, sm_scale
    )


__all__ = ["decode_attention"]
