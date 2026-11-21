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
    cols = tl.program_id(1) * 32 + tl.arange(0, 32)
    rows = tl.arange(0, 16)
    split_index = tl.program_id(0)
    k_offsets = split_index * 128 + tl.arange(0, 128)
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
        + k_offsets[:, None] * STRIDE_WEIGHT_COL,
        eviction_policy="evict_last",
    )
    accumulator = tl.dot(inputs, weights)
    tl.store(
        partial_ptr
        + (tl.program_id(1) * (HIDDEN_SIZE // 128) + split_index)
        * NUM_ROWS
        * 32
        + rows[:, None] * 32
        + tl.arange(0, 32)[None, :],
        accumulator,
        rows[:, None] < NUM_ROWS,
    )


@triton.jit
def _dsv3_router_gemm_reduce_partials_kernel(
    partial_ptr,
    out_ptr,
    NUM_ROWS: tl.constexpr,
    NUM_EXPERTS: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    BLOCK_SPLITS: tl.constexpr,
):
    tile = tl.program_id(0)
    rows = tl.program_id(1)
    index = tl.arange(0, 32)
    split_index = tl.arange(0, BLOCK_SPLITS)
    values = tl.load(
        partial_ptr
        + (tile * NUM_SPLITS + split_index[:, None]) * NUM_ROWS * 32
        + rows * 32
        + index[None, :],
        split_index[:, None] < NUM_SPLITS,
        other=0,
    )
    value = tl.sum(values, 0)
    tl.store(out_ptr + rows * NUM_EXPERTS + tile * 32 + index, value)


@triton.jit
def _dsv3_router_gemm_packed_rows_kernel(
    x_ptr,
    weight_ptr,
    out_ptr,
    NUM_EXPERTS: tl.constexpr,
    HIDDEN_SIZE: tl.constexpr,
    BLOCK_K: tl.constexpr,
    NUM_ROWS: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.program_id(1)
    k_offsets = tl.arange(0, BLOCK_K)
    x = tl.load(
        x_ptr + rows[:, None] * HIDDEN_SIZE + k_offsets[None, :],
        (rows[:, None] < NUM_ROWS) & (k_offsets[None, :] < HIDDEN_SIZE),
        other=0,
    )
    weights = tl.load(
        weight_ptr + cols * HIDDEN_SIZE + k_offsets,
        k_offsets < HIDDEN_SIZE,
        other=0,
        cache_modifier="" if NUM_ROWS <= BLOCK_M else ".cg",
        eviction_policy="evict_last" if NUM_ROWS <= BLOCK_M else "",
    )
    x_low = (x << 16).to(tl.float32, bitcast=True)
    x_high = (x & -65536).to(tl.float32, bitcast=True)
    weight_low = (weights << 16).to(tl.float32, bitcast=True)
    weight_high = (weights & -65536).to(tl.float32, bitcast=True)
    value = tl.sum(
        weight_low[None, :] * x_low + weight_high[None, :] * x_high, 1
    )
    tl.store(out_ptr + rows * NUM_EXPERTS + cols, value, rows < NUM_ROWS)


def dsv3_router_gemm(
    hidden_states,
    router_weights,
    out_dtype,
):
    num_rows, hidden_size = hidden_states.shape
    num_experts = router_weights.shape[0]
    if (
        num_rows <= 4
        and hidden_states.is_contiguous()
        and router_weights.is_contiguous()
        and (num_rows > 1 or hidden_states.stride(0) % 2 == 0)
        and (hidden_states.storage_offset() % 2 == 0)
        and (router_weights.storage_offset() % 2 == 0)
    ):
        x = hidden_states.view(torch.int32)
        weights = router_weights.view(torch.int32)
        out = hidden_states.new_empty((num_rows, num_experts), dtype=out_dtype)
        _dsv3_router_gemm_packed_rows_kernel[
            triton.cdiv(num_rows, min(2, triton.next_power_of_2(num_rows))),
            num_experts,
        ](
            x,
            weights,
            out,
            num_experts,
            hidden_size // 2,
            triton.next_power_of_2(hidden_size // 2),
            num_rows,
            min(2, triton.next_power_of_2(num_rows)),
            num_warps=4,
            enable_fp_fusion=False,
        )
        return out
    if num_rows <= 4:
        out = torch.empty(
            (num_rows, num_experts),
            dtype=out_dtype,
            device=hidden_states.device,
        )
        _dsv3_router_gemm_small_batch_kernel[num_rows, num_experts](
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
        return out
    splits = hidden_size // 128
    partial = torch.empty(
        (num_experts // 32, splits, num_rows, 32),
        device=hidden_states.device,
        dtype=torch.float32,
    )
    out = torch.empty(
        (num_rows, num_experts), device=hidden_states.device, dtype=out_dtype
    )
    _dsv3_router_gemm_matmul_kernel[splits, num_experts // 32](
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
    _dsv3_router_gemm_reduce_partials_kernel[num_experts // 32, num_rows](
        partial,
        out,
        num_rows,
        num_experts,
        splits,
        triton.next_power_of_2(splits),
        num_warps=1,
    )
    return out


__all__ = ["dsv3_router_gemm"]
