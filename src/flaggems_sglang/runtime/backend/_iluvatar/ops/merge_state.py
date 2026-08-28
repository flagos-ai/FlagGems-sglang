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
    total_elements,
    HEAD_DIM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements
    rows = offsets // HEAD_DIM
    columns = offsets - rows * HEAD_DIM

    prefix_lse_value = tl.load(
        prefix_lse + rows, mask=mask, other=-math.inf
    ).to(tl.float32)
    suffix_lse_value = tl.load(
        suffix_lse + rows, mask=mask, other=-math.inf
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
    low_se = tl.exp2(delta * 1.4426950408889634)
    inverse_sum = 1.0 / (1.0 + low_se)
    low_scale = low_se * inverse_sum
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

    prefix_value = tl.load(prefix_output + offsets, mask=mask, other=0.0).to(
        tl.float32
    )
    suffix_value = tl.load(suffix_output + offsets, mask=mask, other=0.0).to(
        tl.float32
    )
    high_value = tl.where(prefix_is_high, prefix_value, suffix_value)
    low_value = tl.where(prefix_is_high, suffix_value, prefix_value)
    merged = high_value + (low_value - high_value) * low_scale
    tl.store(output + offsets, merged, mask=mask)
    tl.store(
        output_lse + rows,
        high_lse + log_sum,
        mask=mask & (columns == 0),
    )


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
    if row_count <= 2048:
        num_warps = 8
    elif row_count <= 16384:
        num_warps = 4
    else:
        num_warps = 2
    block_size = 512
    _merge_state_kernel[(triton.cdiv(output_elements, block_size),)](
        prefix_output,
        prefix_lse,
        suffix_output,
        suffix_lse,
        output,
        output_lse,
        output_elements,
        HEAD_DIM=head_dim,
        BLOCK_SIZE=block_size,
        num_warps=num_warps,
    )
    return output, output_lse


reference = merge_state

__all__ = ["merge_state"]
