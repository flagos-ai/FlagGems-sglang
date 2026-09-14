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
    num_tokens: tl.constexpr,
    top_k: tl.constexpr,
    hidden_dim: tl.constexpr,
    input_stride_token,
    input_stride_topk,
    input_stride_hidden,
    output_stride_token,
    output_stride_hidden,
    routed_scaling_factor,
    BLOCK_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    token = tl.program_id(0)
    hidden_block = tl.program_id(1)
    offsets_k = tl.arange(0, BLOCK_K)
    offsets_n = hidden_block * BLOCK_N + tl.arange(0, BLOCK_N)
    input_offsets = (
        token * input_stride_token
        + offsets_k[:, None] * input_stride_topk
        + offsets_n[None, :] * input_stride_hidden
    )
    mask = (offsets_k[:, None] < top_k) & (offsets_n[None, :] < hidden_dim)
    values = tl.load(input_ptr + input_offsets, mask=mask, other=0.0)
    reduced = tl.sum(values.to(tl.float32), axis=0)
    reduced *= routed_scaling_factor
    output_offsets = (
        token * output_stride_token + offsets_n * output_stride_hidden
    )
    tl.store(output_ptr + output_offsets, reduced, mask=offsets_n < hidden_dim)


def moe_sum_reduce(
    input: torch.Tensor, routed_scaling_factor: float
) -> torch.Tensor:
    num_tokens, top_k, hidden_dim = input.shape
    output = torch.empty(
        (num_tokens, hidden_dim), dtype=input.dtype, device=input.device
    )
    block_k = triton.next_power_of_2(top_k)
    max_block_n = max(128, 16384 // block_k)
    block_n = min(1024, max_block_n, triton.next_power_of_2(hidden_dim))
    _moe_sum_reduce_kernel[(num_tokens, triton.cdiv(hidden_dim, block_n))](
        input,
        output,
        num_tokens,
        top_k,
        hidden_dim,
        *input.stride(),
        *output.stride(),
        routed_scaling_factor,
        BLOCK_K=block_k,
        BLOCK_N=block_n,
        num_warps=2,
    )
    return output


__all__ = ["moe_sum_reduce"]
