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

import triton
import triton.experimental.tle.language as tle
import triton.language as tl


@triton.jit(do_not_specialize=["x_ptr", "weight_ptr", "out_ptr"])
def _dsv3_router_gemm_two_rows_kernel(
    x_ptr,
    weight_ptr,
    out_ptr,
    NUM_EXPERTS: tl.constexpr,
    HIDDEN_SIZE: tl.constexpr,
):
    rows = tl.program_id(0)
    cols = tl.program_id(1) * 8 + tl.arange(0, 8)
    row_indices = tl.arange(0, 512)
    acc_0 = tl.zeros((8, 512), tl.float32)
    acc_1 = tl.zeros((8, 512), tl.float32)
    x_low = tle.load(x_ptr + rows * HIDDEN_SIZE + row_indices, is_async=True)
    x_high = tle.load(
        x_ptr + rows * HIDDEN_SIZE + row_indices + 512, is_async=True
    )
    weight_low = tle.load(
        weight_ptr + cols[:, None] * HIDDEN_SIZE + row_indices[None, :],
        is_async=True,
    )
    weight_high = tle.load(
        weight_ptr + cols[:, None] * HIDDEN_SIZE + row_indices[None, :] + 512,
        is_async=True,
    )
    for base in range(HIDDEN_SIZE // 1024 - 1):
        k_low = (base + 1) * 1024 + row_indices
        k_high = k_low + 512
        nx_0 = tle.load(x_ptr + rows * HIDDEN_SIZE + k_low, is_async=True)
        nx_1 = tle.load(x_ptr + rows * HIDDEN_SIZE + k_high, is_async=True)
        nw_0 = tle.load(
            weight_ptr + cols[:, None] * HIDDEN_SIZE + k_low[None, :],
            is_async=True,
        )
        nw_1 = tle.load(
            weight_ptr + cols[:, None] * HIDDEN_SIZE + k_high[None, :],
            is_async=True,
        )
        acc_0 += weight_low.to(tl.float32) * x_low[None, :].to(tl.float32)
        acc_1 += weight_high.to(tl.float32) * x_high[None, :].to(tl.float32)
        x_low = nx_0
        x_high = nx_1
        weight_low = nw_0
        weight_high = nw_1
    acc_0 += weight_low.to(tl.float32) * x_low[None, :].to(tl.float32)
    acc_1 += weight_high.to(tl.float32) * x_high[None, :].to(tl.float32)
    value = tl.sum(acc_0 + acc_1, 1)
    tl.store(out_ptr + rows * NUM_EXPERTS + cols, value)


@triton.jit(do_not_specialize=["x_ptr", "weight_ptr", "out_ptr"])
def _dsv3_router_gemm_row_tiles_kernel(
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
    cols = tl.program_id(1) * 8 + tl.arange(0, 8)
    k_offsets = tl.arange(0, BLOCK_K)
    accumulator = tl.zeros((8, BLOCK_K), tl.float32)
    for base in range(tl.cdiv(HIDDEN_SIZE, BLOCK_K)):
        k_block = base * BLOCK_K + k_offsets
        x = tl.load(
            x_ptr + rows * STRIDE_X_ROW + k_block * STRIDE_X_COL,
            k_block < HIDDEN_SIZE,
            other=0,
        ).to(tl.float32)
        weights = tl.load(
            weight_ptr
            + cols[:, None] * STRIDE_WEIGHT_ROW
            + k_block[None, :] * STRIDE_WEIGHT_COL,
            k_block[None, :] < HIDDEN_SIZE,
            other=0,
        ).to(tl.float32)
        accumulator += weights * x[None, :]
    out = tl.sum(accumulator, 1)
    tl.store(out_ptr + rows * NUM_EXPERTS + cols, out)


def dsv3_router_gemm(
    hidden_states,
    router_weights,
    out_dtype,
):
    num_rows, hidden_size = hidden_states.shape
    num_experts = router_weights.shape[0]
    out = hidden_states.new_empty((num_rows, num_experts), dtype=out_dtype)
    if hidden_states.is_contiguous() and router_weights.is_contiguous():
        _dsv3_router_gemm_two_rows_kernel.run(
            hidden_states,
            router_weights,
            out,
            num_experts,
            hidden_size,
            grid=(num_rows, num_experts // 8),
            warmup=False,
            num_warps=4,
            enable_fp_fusion=False,
        )
        return out
    _dsv3_router_gemm_row_tiles_kernel.run(
        hidden_states,
        router_weights,
        out,
        num_experts,
        hidden_size,
        hidden_states.stride(0),
        hidden_states.stride(1),
        router_weights.stride(0),
        router_weights.stride(1),
        (
            1024
            if hidden_states.stride(1) == router_weights.stride(1) == 1
            else 128
        ),
        num_warps=4,
        enable_fp_fusion=False,
        grid=(num_rows, num_experts // 8),
        warmup=False,
    )
    return out


__all__ = ["dsv3_router_gemm"]
