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
def _dsv3_router_gemm_small_batch_kernel(
    x_ptr,
    weight_ptr,
    out_ptr,
    NUM_EXPERTS: tl.constexpr,
    HIDDEN_SIZE: tl.constexpr,
    STRIDE_X_ROW: tl.constexpr,
    STRIDE_X_COL: tl.constexpr,
    STRIDE_WEIGHT_ROW: tl.constexpr,
    STRIDE_WEIGHT_COL: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    rows = tl.program_id(0)
    cols = tl.program_id(1)
    k_offsets = tl.arange(0, BLOCK_K)
    accumulator = tl.zeros((BLOCK_K,), tl.float32)
    for base in range(tl.cdiv(HIDDEN_SIZE, BLOCK_K)):
        k_block = base * BLOCK_K + k_offsets
        x = tl.load(
            x_ptr + rows * STRIDE_X_ROW + k_block * STRIDE_X_COL,
            k_block < HIDDEN_SIZE,
            other=0,
        ).to(tl.float32)
        weights = tl.load(
            weight_ptr
            + cols * STRIDE_WEIGHT_ROW
            + k_block * STRIDE_WEIGHT_COL,
            k_block < HIDDEN_SIZE,
            other=0,
            cache_modifier=".cg",
        ).to(tl.float32)
        accumulator += weights * x
    out = tl.sum(accumulator, 0)
    tl.store(out_ptr + rows * NUM_EXPERTS + cols, out)


@triton.jit
def _dsv3_router_gemm_large_batch_kernel(
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
    rows = tl.arange(0, BLOCK_M)
    cols = tl.program_id(0)
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
            cache_modifier=".cg",
        ).to(tl.float32)
        accumulator += x * weights[None, :]
    out = tl.sum(accumulator, 1)
    tl.store(out_ptr + rows * NUM_EXPERTS + cols, out, rows < NUM_ROWS)


def dsv3_router_gemm(
    hidden_states,
    router_weights,
    out_dtype,
):
    num_rows, hidden_size = hidden_states.shape
    num_experts = router_weights.shape[0]
    out = torch.empty(
        (num_rows, num_experts), dtype=out_dtype, device=hidden_states.device
    )
    if num_rows != 16:
        _dsv3_router_gemm_small_batch_kernel[(num_rows, num_experts)](
            hidden_states,
            router_weights,
            out,
            num_experts,
            hidden_size,
            hidden_states.stride(0),
            hidden_states.stride(1),
            router_weights.stride(0),
            router_weights.stride(1),
            triton.next_power_of_2(hidden_size),
            num_warps=1 if num_rows <= 4 else 4,
            enable_fp_fusion=False,
        )
    else:
        _dsv3_router_gemm_large_batch_kernel[(num_experts,)](
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
            triton.next_power_of_2(num_rows),
            1024,
            num_warps=8,
            enable_fp_fusion=False,
        )
    return out


__all__ = ["dsv3_router_gemm"]
