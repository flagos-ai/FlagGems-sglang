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
def _dsv3_router_gemm_row_tiles_kernel(
    x_ptr,
    weight_ptr,
    out_ptr,
    NUM_ROWS: tl.constexpr,
    NUM_EXPERTS: tl.constexpr,
    HIDDEN_SIZE: tl.constexpr,
    STRIDE_X_ROW: tl.constexpr,
    STRIDE_X_COL: tl.constexpr,
    STRIDE_WEIGHT_ROW: tl.constexpr,
    STRIDE_WEIGHT_COL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.program_id(1)
    k_offsets = tl.arange(0, BLOCK_K)
    accumulator = tl.zeros((BLOCK_M, BLOCK_K), tl.float32)
    for base in range(tl.cdiv(HIDDEN_SIZE, BLOCK_K)):
        k_block = base * BLOCK_K + k_offsets
        x = tl.load(
            x_ptr
            + rows[:, None] * STRIDE_X_ROW
            + k_block[None, :] * STRIDE_X_COL,
            (rows[:, None] < NUM_ROWS) & (k_block[None, :] < HIDDEN_SIZE),
            other=0,
        ).to(tl.float32)
        weights = tl.load(
            weight_ptr
            + cols * STRIDE_WEIGHT_ROW
            + k_block * STRIDE_WEIGHT_COL,
            k_block < HIDDEN_SIZE,
            other=0,
        ).to(tl.float32)
        accumulator += x * weights[None, :]
    out = tl.sum(accumulator, 1)
    tl.store(out_ptr + rows * NUM_EXPERTS + cols, out, rows < NUM_ROWS)


@triton.jit
def _dsv3_router_gemm_matmul_kernel(
    x_ptr,
    weight_ptr,
    partial_ptr,
    NUM_ROWS: tl.constexpr,
    NUM_EXPERTS: tl.constexpr,
    HIDDEN_SIZE: tl.constexpr,
    STRIDE_X_ROW: tl.constexpr,
    STRIDE_X_COL: tl.constexpr,
    STRIDE_WEIGHT_ROW: tl.constexpr,
    STRIDE_WEIGHT_COL: tl.constexpr,
):
    cols = tl.program_id(0) * 32 + tl.arange(0, 32)
    rows = tl.arange(0, 16)
    split_index = tl.program_id(1)
    k_offsets = split_index * 512 + tl.arange(0, 512)
    inputs = tl.load(
        x_ptr
        + rows[:, None] * STRIDE_X_ROW
        + k_offsets[None, :] * STRIDE_X_COL,
        rows[:, None] < NUM_ROWS,
        other=0,
    )
    weights = tl.load(
        weight_ptr
        + cols[None, :] * STRIDE_WEIGHT_ROW
        + k_offsets[:, None] * STRIDE_WEIGHT_COL
    )
    accumulator = tl.dot(inputs, weights)
    tl.store(
        partial_ptr
        + split_index * NUM_ROWS * NUM_EXPERTS
        + rows[:, None] * NUM_EXPERTS
        + cols[None, :],
        accumulator,
        rows[:, None] < NUM_ROWS,
    )


@triton.jit
def _dsv3_router_gemm_reduce_partials_kernel(
    partial_ptr,
    out_ptr,
    NUM_ELEMENTS: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    BLOCK_SPLITS: tl.constexpr,
):
    index = tl.program_id(0) * 32 + tl.arange(0, 32)
    split_index = tl.arange(0, BLOCK_SPLITS)
    values = tl.load(
        partial_ptr + split_index[:, None] * NUM_ELEMENTS + index[None, :],
        (split_index[:, None] < NUM_SPLITS) & (index[None, :] < NUM_ELEMENTS),
        other=0,
    )
    value = tl.sum(values, 0)
    tl.store(out_ptr + index, value, index < NUM_ELEMENTS)


def dsv3_router_gemm(
    hidden_states,
    router_weights,
    out_dtype,
):
    num_rows, hidden_size = hidden_states.shape
    num_experts = router_weights.shape[0]
    if num_rows <= 4:
        block_m = triton.next_power_of_2(num_rows)
        block_k = 2048
        out = torch.empty(
            (num_rows, num_experts),
            dtype=out_dtype,
            device=hidden_states.device,
        )
        _dsv3_router_gemm_row_tiles_kernel[
            (triton.cdiv(num_rows, block_m), num_experts)
        ](
            hidden_states,
            router_weights,
            out,
            num_rows,
            num_experts,
            hidden_size,
            hidden_states.stride(0),
            hidden_states.stride(1),
            router_weights.stride(0),
            router_weights.stride(1),
            block_m,
            block_k,
            num_warps=4,
            enable_fp_fusion=False,
        )
        return out
    splits = hidden_size // 512
    partial = torch.empty(
        (splits, num_rows, num_experts),
        device=hidden_states.device,
        dtype=torch.float32,
    )
    out = torch.empty(
        (num_rows, num_experts), device=hidden_states.device, dtype=out_dtype
    )
    _dsv3_router_gemm_matmul_kernel[(num_experts // 32, splits)](
        hidden_states,
        router_weights,
        partial,
        num_rows,
        num_experts,
        hidden_size,
        hidden_states.stride(0),
        hidden_states.stride(1),
        router_weights.stride(0),
        router_weights.stride(1),
        num_warps=2,
        num_stages=2,
    )
    _dsv3_router_gemm_reduce_partials_kernel[
        (triton.cdiv(num_rows * num_experts, 32),)
    ](
        partial,
        out,
        num_rows * num_experts,
        splits,
        triton.next_power_of_2(splits),
        num_warps=1,
    )
    return out


__all__ = ["dsv3_router_gemm"]
