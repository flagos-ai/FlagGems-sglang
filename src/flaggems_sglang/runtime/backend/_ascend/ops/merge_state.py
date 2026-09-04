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
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied. See the License for the specific language governing
# permissions and limitations under the License.

import math

import torch
import triton
import triton.language as tl


@triton.jit
def _merge_state_kernel(
    prefix_output,
    prefix_lse,
    suffix_output,
    suffix_lse,
    output,
    output_lse,
    row_count,
    HEAD_DIM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    ROWS_PER_PROGRAM: tl.constexpr,
):
    program = tl.program_id(0)
    rows = program * ROWS_PER_PROGRAM + tl.arange(0, ROWS_PER_PROGRAM)
    cols = tl.arange(0, BLOCK_SIZE)
    row_mask = rows < row_count

    p_lse = tl.load(prefix_lse + rows, mask=row_mask, other=-math.inf).to(
        tl.float32
    )
    s_lse = tl.load(suffix_lse + rows, mask=row_mask, other=-math.inf).to(
        tl.float32
    )
    p_lse = tl.where(p_lse == math.inf, -math.inf, p_lse)
    s_lse = tl.where(s_lse == math.inf, -math.inf, s_lse)

    p_is_max = p_lse >= s_lse
    max_lse = tl.maximum(p_lse, s_lse)
    min_lse = tl.minimum(p_lse, s_lse)
    low_se = tl.exp2((min_lse - max_lse) * 1.4426950408889634)
    out_se = 1.0 + low_se
    merged_lse = tl.log2(out_se) * 0.6931471805599453 + max_lse
    high_scale = 1.0 / out_se
    low_scale = low_se * high_scale
    p_scale = tl.where(p_is_max, high_scale, low_scale)
    s_scale = tl.where(p_is_max, low_scale, high_scale)
    tl.store(output_lse + rows, merged_lse, mask=row_mask)

    offsets = rows[:, None] * HEAD_DIM + cols[None, :]
    mask = row_mask[:, None] & (cols[None, :] < HEAD_DIM)
    p_value = tl.load(prefix_output + offsets, mask=mask, other=0.0).to(
        tl.float32
    )
    s_value = tl.load(suffix_output + offsets, mask=mask, other=0.0).to(
        tl.float32
    )
    merged = p_value * p_scale[:, None] + s_value * s_scale[:, None]
    tl.store(output + offsets, merged, mask=mask)


def merge_state(prefix_output, prefix_lse, suffix_output, suffix_lse):
    output = torch.empty_like(prefix_output)
    output_lse = torch.empty_like(prefix_lse)
    head_dim = prefix_output.shape[-1]
    row_count = prefix_output.numel() // head_dim
    block_size = triton.next_power_of_2(head_dim)

    if row_count >= 2048:
        rows_per_program = 32 if head_dim <= 128 else 16
    elif row_count >= 256:
        rows_per_program = 32
    else:
        rows_per_program = 1
    num_warps = 4 if block_size >= 64 else 1

    _merge_state_kernel[(triton.cdiv(row_count, rows_per_program),)](
        prefix_output,
        prefix_lse,
        suffix_output,
        suffix_lse,
        output,
        output_lse,
        row_count,
        HEAD_DIM=head_dim,
        BLOCK_SIZE=block_size,
        ROWS_PER_PROGRAM=rows_per_program,
        num_warps=num_warps,
    )
    return output, output_lse


__all__ = ["merge_state"]
