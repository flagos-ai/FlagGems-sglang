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

import torch
import triton
import triton.language as tl


@triton.jit
def _per_group_transpose_kernel(
    input_ptr,
    output_ptr,
    expert_offsets_ptr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    expert = tl.program_id(0)
    k_block = tl.program_id(1)

    group_start = tl.load(expert_offsets_ptr + expert)
    group_end = tl.load(expert_offsets_ptr + expert + 1)
    group_rows = group_end - group_start
    group_base = group_start * K

    k_offsets = k_block * BLOCK_K + tl.arange(0, BLOCK_K)
    k_mask = k_offsets < K

    for row_start in tl.range(0, group_rows, BLOCK_M):
        row_offsets = row_start + tl.arange(0, BLOCK_M)
        row_mask = row_offsets < group_rows

        input_offsets = (
            group_base + row_offsets[:, None] * K + k_offsets[None, :]
        )
        values = tl.load(
            input_ptr + input_offsets,
            mask=row_mask[:, None] & k_mask[None, :],
            other=0,
        )

        output_offsets = (
            group_base + k_offsets[:, None] * group_rows + row_offsets[None, :]
        )
        tl.store(
            output_ptr + output_offsets,
            tl.trans(values),
            mask=k_mask[:, None] & row_mask[None, :],
        )


def per_group_transpose(a, expert_offsets, m_alignment=1):
    _, k = a.shape
    num_experts = expert_offsets.numel() - 1
    output = torch.empty_like(a)

    # gfx1100: from 2^20 elements on, larger tiles (64x128 for 16-bit
    # elements, 64x64 for 32-bit ones) with 8 warps, unless that leaves fewer
    # than 16 programs; the other element sizes keep the generic launch.
    block_m, block_k, num_warps = 32, 32, 4
    wide = {2: 128, 4: 64}.get(a.element_size())
    if (
        wide
        and a.numel() >= 1 << 20
        and num_experts * triton.cdiv(k, wide) >= 16
    ):
        block_m, block_k, num_warps = 64, wide, 8
    grid = (num_experts, triton.cdiv(k, block_k))
    _per_group_transpose_kernel[grid](
        a,
        output,
        expert_offsets,
        K=k,
        BLOCK_M=block_m,
        BLOCK_K=block_k,
        num_warps=num_warps,
    )
    return output


__all__ = ["per_group_transpose"]
