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
    FAST_LOG: tl.constexpr,
):
    rows = tl.program_id(0) * ROWS_PER_PROGRAM + tl.arange(0, ROWS_PER_PROGRAM)
    columns = tl.arange(0, BLOCK_SIZE)
    row_mask = rows < row_count

    prefix_lse_value = tl.load(
        prefix_lse + rows, mask=row_mask, other=-math.inf
    ).to(tl.float32)
    suffix_lse_value = tl.load(
        suffix_lse + rows, mask=row_mask, other=-math.inf
    ).to(tl.float32)
    prefix_lse_value = tl.where(
        prefix_lse_value == math.inf, -math.inf, prefix_lse_value
    )
    suffix_lse_value = tl.where(
        suffix_lse_value == math.inf, -math.inf, suffix_lse_value
    )

    prefix_is_high = prefix_lse_value >= suffix_lse_value
    high_lse = tl.maximum(prefix_lse_value, suffix_lse_value)
    delta = tl.minimum(prefix_lse_value, suffix_lse_value) - high_lse
    low_se = tl.exp(delta)
    inverse_sum = 1.0 / (1.0 + low_se)
    low_scale = low_se * inverse_sum
    if FAST_LOG:
        transformed = low_se / (2.0 + low_se)
        transformed_squared = transformed * transformed
        log_sum = (
            2.0
            * transformed
            * (
                1.0
                + transformed_squared
                * (
                    0.3333333333333333
                    + transformed_squared
                    * (0.2 + transformed_squared * 0.14285714285714285)
                )
            )
        )
    else:
        log_sum = tl.log(1.0 + low_se)
    prefix_scale = tl.where(prefix_is_high, inverse_sum, low_scale)
    suffix_scale = tl.where(prefix_is_high, low_scale, inverse_sum)
    tl.store(output_lse + rows, high_lse + log_sum, mask=row_mask)

    offsets = rows[:, None] * HEAD_DIM + columns[None, :]
    mask = row_mask[:, None] & (columns[None, :] < HEAD_DIM)
    prefix_value = tl.load(prefix_output + offsets, mask=mask, other=0.0).to(
        tl.float32
    )
    suffix_value = tl.load(suffix_output + offsets, mask=mask, other=0.0).to(
        tl.float32
    )
    merged = (
        prefix_value * prefix_scale[:, None]
        + suffix_value * suffix_scale[:, None]
    )
    tl.store(output + offsets, merged, mask=mask)


def merge_state(prefix_output, prefix_lse, suffix_output, suffix_lse):
    output_elements = prefix_output.numel()
    lse_storage_elements = (
        prefix_lse.numel()
        * prefix_lse.element_size()
        // prefix_output.element_size()
    )
    storage = torch.empty(
        output_elements + lse_storage_elements,
        device=prefix_output.device,
        dtype=prefix_output.dtype,
    )
    output = storage[:output_elements].view_as(prefix_output)
    output_lse = (
        storage[output_elements:].view(prefix_lse.dtype).view_as(prefix_lse)
    )

    head_dim = prefix_output.shape[-1]
    row_count = output_elements // head_dim
    block_size = triton.next_power_of_2(head_dim)
    if row_count <= 64:
        rows_per_program = 64
        num_warps = 1
        fast_log = False
    elif row_count <= 512:
        rows_per_program = 64
        num_warps = 1
        fast_log = True
    elif row_count <= 4096:
        rows_per_program = 256
        num_warps = 2
        fast_log = False
    else:
        rows_per_program = 256
        num_warps = 1
        fast_log = False

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
        FAST_LOG=fast_log,
        num_warps=num_warps,
    )
    return output, output_lse


__all__ = ["merge_state"]
