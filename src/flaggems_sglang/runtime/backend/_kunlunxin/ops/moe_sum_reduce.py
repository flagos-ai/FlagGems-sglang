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
def _moe_sum_reduce_kernel(
    input_ptr,
    output_ptr,
    top_k: tl.constexpr,
    hidden_dim: tl.constexpr,
    input_stride_token,
    input_stride_topk,
    input_stride_hidden,
    output_stride_token,
    output_stride_hidden,
    routed_scaling_factor,
    BLOCK_N: tl.constexpr,
):
    token = tl.program_id(0)
    hidden_block = tl.program_id(1)
    offsets_n = hidden_block * BLOCK_N + tl.arange(0, BLOCK_N)
    input_offsets = (
        token * input_stride_token + offsets_n * input_stride_hidden
    )
    mask = offsets_n < hidden_dim
    reduced = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for topk_index in range(0, top_k):
        values = tl.load(
            input_ptr + input_offsets + topk_index * input_stride_topk,
            mask=mask,
            other=0.0,
        )
        reduced += values.to(tl.float32)
    reduced *= routed_scaling_factor
    output_offsets = (
        token * output_stride_token + offsets_n * output_stride_hidden
    )
    tl.store(output_ptr + output_offsets, reduced, mask=mask)


def moe_sum_reduce(
    input: torch.Tensor, routed_scaling_factor: float
) -> torch.Tensor:
    num_tokens, top_k, hidden_dim = input.shape
    output = torch.empty(
        (num_tokens, hidden_dim), dtype=input.dtype, device=input.device
    )
    block_n = min(128, triton.next_power_of_2(hidden_dim))
    _moe_sum_reduce_kernel[(num_tokens, triton.cdiv(hidden_dim, block_n))](
        input,
        output,
        top_k,
        hidden_dim,
        *input.stride(),
        *output.stride(),
        routed_scaling_factor,
        BLOCK_N=block_n,
        num_warps=1,
        num_stages=1,
    )
    return output


__all__ = ["moe_sum_reduce"]
