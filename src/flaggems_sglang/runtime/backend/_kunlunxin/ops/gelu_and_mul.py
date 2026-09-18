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

"""Kunlunxin exact-erf persistent route for FlagOS Task 29.

Derived from the Task 29 C004 SGLang-based implementation and the fixed
official FlagGems Kunlunxin erf/pointwise sources at commit
bca7a6994a1750c04177dfeb8c24f900ffbd7d4a.  Modifications: route exact erf
through XPU libdevice and use the official one-dimensional pointwise topology:
12 CTAs, one power-of-two tile per CTA, the 4/8/16-warp heuristic, and a 2048
buffer-size limit.  Runtime strides, float32 intermediates, tail masks,
formula order, and the one-argument entrypoint are retained.

This source is a local candidate only.  It has not been compiled or executed
on Kunlunxin hardware.

Copyright 2023-2024 SGLang Team
Copyright 2026 FlagOS Contributors
SPDX-License-Identifier: Apache-2.0
"""

import torch
import triton
import triton.language as tl
from triton.language.extra.xpu.libdevice import erf as _erf

__all__ = ["gelu_and_mul"]


@triton.jit
def _gelu_and_mul_kernel(
    hidden_states_ptr,
    output_ptr,
    hidden_size,
    batch_size,
    input_stride_batch,
    input_stride_hidden,
    output_stride_batch,
    output_stride_hidden,
    BLOCK_SIZE: tl.constexpr,
):
    total_outputs = batch_size * hidden_size
    flat_offsets = tl.program_id(axis=0) * BLOCK_SIZE + tl.arange(
        0, BLOCK_SIZE
    )
    mask = flat_offsets < total_outputs
    batch_index = flat_offsets // hidden_size
    columns = flat_offsets - batch_index * hidden_size

    input_row = batch_index * input_stride_batch
    gate = tl.load(
        hidden_states_ptr + input_row + columns * input_stride_hidden,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    up = tl.load(
        hidden_states_ptr
        + input_row
        + (hidden_size + columns) * input_stride_hidden,
        mask=mask,
        other=0.0,
    ).to(tl.float32)

    activated = gate * (0.5 * (1.0 + _erf(gate * 0.7071067811865475)))
    output_offsets = (
        batch_index * output_stride_batch + columns * output_stride_hidden
    )
    tl.store(output_ptr + output_offsets, activated * up, mask=mask)


def gelu_and_mul(hidden_states):
    batch_size, doubled_hidden_size = hidden_states.shape
    hidden_size = doubled_hidden_size // 2
    output = torch.empty(
        (batch_size, hidden_size),
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )
    tile_size = triton.next_power_of_2(
        triton.cdiv(batch_size * hidden_size, 12)
    )
    num_warps = min(16, max(4, tile_size // 256))
    grid = (12,)
    _gelu_and_mul_kernel[grid](
        hidden_states,
        output,
        hidden_size,
        batch_size,
        hidden_states.stride(0),
        hidden_states.stride(1),
        output.stride(0),
        output.stride(1),
        tile_size,
        num_warps=num_warps,
        buffer_size_limit=2048,
    )
    return output
