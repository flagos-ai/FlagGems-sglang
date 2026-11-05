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

"""decode_grouped_attention for RDNA3 (gfx1100) — one decode step of paged
attention for a large query group a KV head (Triton).

The sequence is handed to the shared split-KV kernels, which read every K/V
tile once for a whole block of query heads instead of once a head, and which
fill the 96 compute units even when there are few sequences. float32 products
run on the FMA units but a large group still fills the tile, so they take the
same path; the kernel below is left for the head sizes and layouts the tiles
do not cover.

This kernel is written in portable Triton only. It must NOT call any
pre-compiled / vendor-specific cached operator (no
``F.scaled_dot_product_attention``, no ``_compiled`` handles, no
``torch.ops.*`` attention) — the whole computation is done inside the Triton
kernel.
"""

import torch
import triton
import triton.language as tl

from ._decode_splitkv import decode_splitkv, supported


@triton.jit
def _row_kernel(
    q,
    k_buffer,
    v_buffer,
    kv_indptr,
    kv_indices,
    output,
    sm_scale,
    H_Q: tl.constexpr,
    H_KV: tl.constexpr,
    D: tl.constexpr,
    D_V: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_DV: tl.constexpr,
):
    batch_id = tl.program_id(0)
    query_head = tl.program_id(1)
    kv_head = query_head // GROUP_SIZE
    d_offsets = tl.arange(0, BLOCK_D)
    dv_offsets = tl.arange(0, BLOCK_DV)
    q_values = tl.load(
        q + (batch_id * H_Q + query_head) * D + d_offsets,
        mask=d_offsets < D,
        other=0.0,
    ).to(tl.float32)
    start = tl.load(kv_indptr + batch_id)
    end = tl.load(kv_indptr + batch_id + 1)
    running_max = -float("inf")
    running_sum = 0.0
    accumulator = tl.zeros((BLOCK_DV,), dtype=tl.float32)
    for position in tl.range(start, end):
        page = tl.load(kv_indices + position)
        k_values = tl.load(
            k_buffer + (page * H_KV + kv_head) * D + d_offsets,
            mask=d_offsets < D,
            other=0.0,
        ).to(tl.float32)
        score = tl.sum(q_values * k_values, axis=0) * sm_scale
        next_max = tl.maximum(running_max, score)
        old_scale = tl.exp(running_max - next_max)
        probability = tl.exp(score - next_max)
        v_values = tl.load(
            v_buffer + (page * H_KV + kv_head) * D_V + dv_offsets,
            mask=dv_offsets < D_V,
            other=0.0,
        ).to(tl.float32)
        accumulator = accumulator * old_scale + probability * v_values
        running_sum = running_sum * old_scale + probability
        running_max = next_max
    output_offsets = (batch_id * H_Q + query_head) * D_V + dv_offsets
    tl.store(
        output + output_offsets,
        accumulator / running_sum,
        mask=dv_offsets < D_V,
    )


def decode_grouped_attention(
    q, k_buffer, v_buffer, kv_indptr, kv_indices, sm_scale
):
    batch_size, query_heads, head_size = q.shape
    kv_heads = k_buffer.shape[1]
    value_size = v_buffer.shape[2]
    group_size = query_heads // kv_heads
    if supported(q, k_buffer, v_buffer, group_size):
        return decode_splitkv(
            q, k_buffer, v_buffer, kv_indptr, kv_indices, sm_scale
        )
    output = torch.empty(
        (batch_size, query_heads, value_size),
        dtype=torch.float32,
        device=q.device,
    )
    _row_kernel[(batch_size, query_heads)](
        q,
        k_buffer,
        v_buffer,
        kv_indptr,
        kv_indices,
        output,
        sm_scale,
        query_heads,
        kv_heads,
        head_size,
        value_size,
        group_size,
        triton.next_power_of_2(head_size),
        triton.next_power_of_2(value_size),
        # a program here owns one query head, and a 512-wide head leaves little
        # for more than one warp to do
        num_warps=(
            1
            if max(head_size, value_size) >= 512
            else 8 if max(head_size, value_size) >= 256 else 4
        ),
        num_stages=1,
    )
    return output


__all__ = ["decode_grouped_attention"]
