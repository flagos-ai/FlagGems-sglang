# Copyright 2026, The FlagOS Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License")
# you may not use this file except in compliance with the License
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
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
    k: tl.constexpr,
    M_ALIGNMENT: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    expert_id = tl.program_id(0)
    m_program_id = tl.program_id(1)
    k_program_id = tl.program_id(2)

    start = tl.load(expert_offsets_ptr + expert_id)
    end = tl.load(expert_offsets_ptr + expert_id + 1)
    rows = end - start
    tl.multiple_of(start, M_ALIGNMENT)
    tl.multiple_of(end, M_ALIGNMENT)

    input_base = input_ptr + start * k
    output_base = output_ptr + start * k
    k_offsets = k_program_id * BLOCK_K + tl.arange(0, BLOCK_K)
    k_mask = k_offsets < k

    for m_start in tl.range(0, rows, BLOCK_M * tl.num_programs(1)):
        m_offsets = m_start + m_program_id * BLOCK_M + tl.arange(0, BLOCK_M)
        mask = (m_offsets[:, None] < rows) & k_mask[None, :]
        values = tl.load(
            input_base + m_offsets[:, None] * k + k_offsets[None, :],
            mask=mask,
        )
        tl.store(
            output_base + m_offsets[:, None] + k_offsets[None, :] * rows,
            values,
            mask=mask,
        )


def _cdiv(a, b):
    return (a + b - 1) // b


def per_group_transpose(a, expert_offsets, m_alignment=1):
    m, k = a.shape
    output = torch.empty_like(a)
    num_experts = expert_offsets.numel() - 1
    average_rows = (m + num_experts - 1) // num_experts
    grid = (
        num_experts,
        max(1, _cdiv(average_rows, 16)),
        _cdiv(k, 8),
    )
    _per_group_transpose_kernel[grid](
        a,
        output,
        expert_offsets,
        k=k,
        M_ALIGNMENT=m_alignment,
        BLOCK_M=16,
        BLOCK_K=8,
    )
    return output


__all__ = ["per_group_transpose"]
