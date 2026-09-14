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

"""Stride-aware Triton candidate for FlagOS Task 29 ``gelu_and_mul``.

Derived from SGLang's official Apache-2.0 ``activation.py`` and
``activation.cuh`` at commit 09ecb9aaaab960a9d2d5792938803487c4efd404.
Modifications: exact one-argument FlagOS entrypoint, Triton masked tiles,
runtime strides, and removal of SGLang/JIT/device-capability dependencies.

Copyright 2023-2024 SGLang Team
SPDX-License-Identifier: Apache-2.0
"""

import torch
import triton
import triton.language as tl

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
    flat_offsets = tl.program_id(axis=0) * BLOCK_SIZE + tl.arange(
        0, BLOCK_SIZE
    )
    total_outputs = batch_size * hidden_size
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

    activated = gate * (0.5 * (1.0 + tl.erf(gate * 0.7071067811865475)))
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
    grid = (triton.cdiv(batch_size * hidden_size, 256),)
    _gelu_and_mul_kernel[grid](
        hidden_states,
        output,
        hidden_size,
        batch_size,
        hidden_states.stride(0),
        hidden_states.stride(1),
        output.stride(0),
        output.stride(1),
        256,
        num_warps=4,
        num_stages=2,
    )
    return output
