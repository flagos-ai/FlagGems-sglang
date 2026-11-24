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

import torch
import triton
import triton.language as tl


@triton.jit
def _dsv3_router_gemm_pad_input_kernel(
    x_ptr,
    out_ptr,
    NUM_ROWS: tl.constexpr,
    HIDDEN_SIZE: tl.constexpr,
    STRIDE_X_ROW: tl.constexpr,
    STRIDE_X_COL: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row_indices = tl.program_id(0)
    cols = tl.program_id(1) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    if row_indices < NUM_ROWS:
        value = tl.load(
            x_ptr + row_indices * STRIDE_X_ROW + cols * STRIDE_X_COL,
            cols < HIDDEN_SIZE,
            other=0,
        )
        tl.store(
            out_ptr + row_indices * HIDDEN_SIZE + cols,
            value,
            cols < HIDDEN_SIZE,
        )
    else:
        tl.store(
            out_ptr + row_indices * HIDDEN_SIZE + cols,
            tl.full((BLOCK_SIZE,), 0, tl.float32),
            cols < HIDDEN_SIZE,
        )


@triton.jit
def _dsv3_router_gemm_matmul_kernel(
    x_ptr,
    weight_ptr,
    out_ptr,
    NUM_EXPERTS: tl.constexpr,
    HIDDEN_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    row_index = tl.arange(0, BLOCK_M)
    expert_index = tl.program_id(0) * 32 + tl.arange(0, 32)
    k_offsets = tl.arange(0, 1024)
    accumulator = tl.zeros((BLOCK_M, 32), tl.float32)
    x_data_ptr = x_ptr + row_index[:, None] * HIDDEN_SIZE + k_offsets[None, :]
    weight_data_ptr = (
        weight_ptr + expert_index[None, :] * HIDDEN_SIZE + k_offsets[:, None]
    )
    for start in range(HIDDEN_SIZE // 1024):
        inputs = tl.load(x_data_ptr)
        weights = tl.load(weight_data_ptr)
        accumulator = tl.dot(
            inputs.to(tl.float32),
            weights.to(tl.float32),
            accumulator,
            input_precision="ieee",
        )
        x_data_ptr += 1024
        weight_data_ptr += 1024
    tl.store(
        out_ptr + row_index[:, None] * NUM_EXPERTS + expert_index[None, :],
        accumulator,
    )


def dsv3_router_gemm(
    hidden_states,
    router_weights,
    out_dtype,
):
    hidden_states = hidden_states.contiguous()
    router_weights = router_weights.contiguous()
    num_rows, hidden_size = hidden_states.shape
    num_experts = router_weights.shape[0]
    block_m = triton.next_power_of_2(num_rows)
    padded_input = hidden_states
    if num_rows != block_m:
        padded_input = torch.empty(
            (block_m, hidden_size),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        _dsv3_router_gemm_pad_input_kernel[block_m, 1](
            hidden_states,
            padded_input,
            num_rows,
            hidden_size,
            hidden_states.stride(0),
            hidden_states.stride(1),
            triton.next_power_of_2(hidden_size),
        )
    out = torch.empty(
        (block_m, num_experts), dtype=out_dtype, device=hidden_states.device
    )
    _dsv3_router_gemm_matmul_kernel[num_experts // 32,](
        padded_input,
        router_weights,
        out,
        num_experts,
        hidden_size,
        block_m,
        num_warps=4,
        num_stages=1,
    )
    if block_m == num_rows:
        return out
    return out[:num_rows]


__all__ = ["dsv3_router_gemm"]
