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
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    offsets_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    offsets_n = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (offsets_m[:, None] < num_tokens) & (
        offsets_n[None, :] < hidden_dim
    )
    input_offsets = (
        offsets_m[:, None] * input_stride_token
        + offsets_n[None, :] * input_stride_hidden
    )
    reduced = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for topk_index in range(0, top_k):
        values = tl.load(
            input_ptr + input_offsets + topk_index * input_stride_topk,
            mask=mask,
            other=0.0,
        )
        reduced += values.to(tl.float32)
    reduced *= routed_scaling_factor
    output_offsets = (
        offsets_m[:, None] * output_stride_token
        + offsets_n[None, :] * output_stride_hidden
    )
    tl.store(output_ptr + output_offsets, reduced, mask=mask)


def moe_sum_reduce(
    input: torch.Tensor, routed_scaling_factor: float
) -> torch.Tensor:
    num_tokens, top_k, hidden_dim = input.shape
    output = torch.empty(
        (num_tokens, hidden_dim), dtype=input.dtype, device=input.device
    )
    if num_tokens <= 64:
        block_m, block_n, num_warps = 2, 512, 4
    else:
        block_m, block_n, num_warps = 1, 2048, 2
    block_n = min(block_n, triton.next_power_of_2(hidden_dim))
    _moe_sum_reduce_kernel[
        (triton.cdiv(num_tokens, block_m), triton.cdiv(hidden_dim, block_n))
    ](
        input,
        output,
        num_tokens,
        top_k,
        hidden_dim,
        *input.stride(),
        *output.stride(),
        routed_scaling_factor,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        num_warps=num_warps,
        num_stages=1,
    )
    return output


__all__ = ["moe_sum_reduce"]
