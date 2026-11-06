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

"""gemma_dual_rmsnorm_residual_scalar implementation for Kunlunxin."""

import torch
import triton
import triton.language as tl


@triton.jit
def _gemma_dual_rmsnorm_residual_scalar_sum_pair(
    first_lhs, second_lhs, first_rhs, second_rhs
):
    return (first_lhs + first_rhs, second_lhs + second_rhs)


@triton.jit
def _gemma_dual_rmsnorm_residual_scalar_tiled_kernel(
    x1_ptr,
    weight1_ptr,
    x2_ptr,
    weight2_ptr,
    weight3_ptr,
    residual_ptr,
    scalar_ptr,
    out_ptr,
    NUM_ROWS: tl.constexpr,
    D: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
):
    col = tl.arange(0, D)
    w1_raw = tl.load(weight1_ptr + col)
    w2_raw = tl.load(weight2_ptr + col)
    w3_raw = tl.load(weight3_ptr + col)
    scalar = tl.load(scalar_ptr).to(tl.float32)
    for group in range(
        tl.program_id(0), NUM_ROWS // BLOCK_ROWS, tl.num_programs(0)
    ):
        row = group * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
        offset = row[:, None] * D + col[None, :]
        first_input = tl.load(
            x1_ptr + offset, offset_state_policy="continuous"
        ).to(tl.float32)
        second_input = tl.load(
            x2_ptr + offset, offset_state_policy="continuous"
        ).to(tl.float32)
        (first_sum_sq, second_sum_sq) = tl.reduce(
            (first_input * first_input, second_input * second_input),
            1,
            _gemma_dual_rmsnorm_residual_scalar_sum_pair,
        )
        first_rstd = tl.rsqrt(first_sum_sq / D + EPS[0])
        w1 = w1_raw.to(tl.float32)
        first_norm = first_input * first_rstd[:, None] * w1[None, :]
        second_rstd = tl.rsqrt(second_sum_sq / D + EPS[1])
        w2 = w2_raw.to(tl.float32)
        combined = (
            first_norm + second_input * second_rstd[:, None] * w2[None, :]
        )
        output_rstd = tl.rsqrt(tl.sum(combined * combined, 1) / D + EPS[2])
        w3 = w3_raw.to(tl.float32)
        residual = tl.load(
            residual_ptr + offset, offset_state_policy="continuous"
        ).to(tl.float32)
        out = (
            combined * output_rstd[:, None] * w3[None, :] + residual
        ) * scalar
        tl.store(out_ptr + offset, out, offset_state_policy="continuous")


@triton.jit
def _gemma_dual_rmsnorm_residual_scalar_wide_kernel(
    x1_ptr,
    weight1_ptr,
    x2_ptr,
    weight2_ptr,
    weight3_ptr,
    residual_ptr,
    scalar_ptr,
    out_ptr,
    NUM_ROWS: tl.constexpr,
    D: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
):
    col = tl.arange(0, D)
    w1_raw = tl.load(weight1_ptr + col)
    w2_raw = tl.load(weight2_ptr + col)
    w3_raw = tl.load(weight3_ptr + col)
    scalar = tl.load(scalar_ptr).to(tl.float32)
    for group in range(
        tl.program_id(0), NUM_ROWS // BLOCK_ROWS, tl.num_programs(0)
    ):
        row = group
        offset = row * D + col
        residual_raw = tl.load(
            residual_ptr + offset, offset_state_policy="continuous"
        )
        first_input = tl.load(
            x1_ptr + offset, offset_state_policy="continuous"
        ).to(tl.float32)
        second_input = tl.load(
            x2_ptr + offset, offset_state_policy="continuous"
        ).to(tl.float32)
        (first_sum_sq, second_sum_sq) = tl.reduce(
            (first_input * first_input, second_input * second_input),
            0,
            _gemma_dual_rmsnorm_residual_scalar_sum_pair,
        )
        first_rstd = tl.rsqrt(first_sum_sq / D + EPS[0])
        second_rstd = tl.rsqrt(second_sum_sq / D + EPS[1])
        w1 = w1_raw.to(tl.float32)
        first_norm = first_input * first_rstd * w1
        w2 = w2_raw.to(tl.float32)
        combined = first_norm + second_input * second_rstd * w2
        output_rstd = tl.rsqrt(tl.sum(combined * combined, 0) / D + EPS[2])
        w3 = w3_raw.to(tl.float32)
        residual = residual_raw.to(tl.float32)
        out = (combined * output_rstd * w3 + residual) * scalar
        tl.store(out_ptr + offset, out, offset_state_policy="continuous")


@triton.jit
def _gemma_dual_rmsnorm_residual_scalar_contiguous_kernel(
    x1_ptr,
    weight1_ptr,
    x2_ptr,
    weight2_ptr,
    weight3_ptr,
    residual_ptr,
    scalar_ptr,
    out_ptr,
    D: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK_D)
    first_input = tl.where(
        col < D,
        tl.load(x1_ptr + row * D + col * 1, col < D, other=0).to(tl.float32),
        0,
    )
    second_input = tl.where(
        col < D,
        tl.load(x2_ptr + row * D + col * 1, col < D, other=0).to(tl.float32),
        0,
    )
    w1 = tl.load(weight1_ptr + col * 1, col < D, other=0).to(tl.float32)
    w2 = tl.load(weight2_ptr + col * 1, col < D, other=0).to(tl.float32)
    (first_sum_sq, second_sum_sq) = tl.reduce(
        (first_input * first_input, second_input * second_input),
        0,
        _gemma_dual_rmsnorm_residual_scalar_sum_pair,
    )
    first_rstd = tl.rsqrt(first_sum_sq / D + EPS[0])
    second_rstd = tl.rsqrt(second_sum_sq / D + EPS[1])
    combined = tl.where(
        col < D,
        first_input * first_rstd * w1 + second_input * second_rstd * w2,
        0,
    )
    output_rstd = tl.rsqrt(tl.sum(combined * combined, 0) / D + EPS[2])
    w3 = tl.load(weight3_ptr + col * 1, col < D, other=0).to(tl.float32)
    residual = tl.load(residual_ptr + row * D + col * 1, col < D, other=0).to(
        tl.float32
    )
    scalar = tl.load(scalar_ptr).to(tl.float32)
    tl.store(
        out_ptr + row * D + col,
        (combined * output_rstd * w3 + residual) * scalar,
        col < D,
    )


@triton.jit
def _gemma_dual_rmsnorm_residual_scalar_chunk_wide_kernel(
    x1_ptr,
    weight1_ptr,
    x2_ptr,
    weight2_ptr,
    weight3_ptr,
    residual_ptr,
    scalar_ptr,
    out_ptr,
    NUM_ROWS: tl.constexpr,
    D: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    NUM_PROGRAMS: tl.constexpr,
):
    col = tl.arange(0, D)
    w1_raw = tl.load(weight1_ptr + col)
    w2_raw = tl.load(weight2_ptr + col)
    w3_raw = tl.load(weight3_ptr + col)
    scalar = tl.load(scalar_ptr).to(tl.float32)
    for group in range(
        tl.program_id(0) * (NUM_ROWS // BLOCK_ROWS) // NUM_PROGRAMS,
        (tl.program_id(0) + 1) * (NUM_ROWS // BLOCK_ROWS) // NUM_PROGRAMS,
    ):
        row = group
        offset = row * D + col
        residual_raw = tl.load(
            residual_ptr + offset, offset_state_policy="continuous"
        )
        first_input = tl.load(
            x1_ptr + offset, offset_state_policy="continuous"
        ).to(tl.float32)
        second_input = tl.load(
            x2_ptr + offset, offset_state_policy="continuous"
        ).to(tl.float32)
        (first_sum_sq, second_sum_sq) = tl.reduce(
            (first_input * first_input, second_input * second_input),
            0,
            _gemma_dual_rmsnorm_residual_scalar_sum_pair,
        )
        first_rstd = tl.rsqrt(first_sum_sq / D + EPS[0])
        second_rstd = tl.rsqrt(second_sum_sq / D + EPS[1])
        w1 = w1_raw.to(tl.float32)
        first_norm = first_input * first_rstd * w1
        w2 = w2_raw.to(tl.float32)
        combined = first_norm + second_input * second_rstd * w2
        output_rstd = tl.rsqrt(tl.sum(combined * combined, 0) / D + EPS[2])
        w3 = w3_raw.to(tl.float32)
        residual = residual_raw.to(tl.float32)
        out = (combined * output_rstd * w3 + residual) * scalar
        tl.store(out_ptr + offset, out, offset_state_policy="continuous")


def gemma_dual_rmsnorm_residual_scalar(
    x1, weight1, x2, weight2, weight3, residual, scalar, eps1, eps2, eps3
):
    num_rows, hidden_size = x1.shape
    out = torch.empty(
        (num_rows, hidden_size), device=x1.device, dtype=x1.dtype
    )
    eps = (eps1, eps2, eps3)
    block_cols = triton.next_power_of_2(hidden_size)
    block_rows = 1 if hidden_size >= 4096 else 8
    if (
        num_rows < 64
        or num_rows % block_rows != 0
        or hidden_size != block_cols
    ):
        _gemma_dual_rmsnorm_residual_scalar_contiguous_kernel[(num_rows,)](
            x1,
            weight1,
            x2,
            weight2,
            weight3,
            residual,
            scalar,
            out,
            hidden_size,
            eps,
            block_cols,
            num_warps=4,
            num_stages=1,
            enable_fp_fusion=False,
            isCloseMemoryAsync=False,
        )
    else:
        grid = min(triton.cdiv(num_rows, block_rows), 12)
        if hidden_size < 4096:
            _gemma_dual_rmsnorm_residual_scalar_tiled_kernel[(grid,)](
                x1,
                weight1,
                x2,
                weight2,
                weight3,
                residual,
                scalar,
                out,
                num_rows,
                hidden_size,
                eps,
                block_rows,
                groups_per_cluster=block_rows,
                buffer_size_limit=1024,
                isCloseUnrollControl=True,
                num_stages=1,
                enable_fp_fusion=False,
            )
        elif hidden_size == 4096 and num_rows >= 384:
            _gemma_dual_rmsnorm_residual_scalar_chunk_wide_kernel[(grid,)](
                x1,
                weight1,
                x2,
                weight2,
                weight3,
                residual,
                scalar,
                out,
                num_rows,
                hidden_size,
                eps,
                block_rows,
                grid,
                groups_per_cluster=block_rows,
                buffer_size_limit=1024,
                num_stages=1,
                enable_fp_fusion=False,
                isCloseMemoryAsync=False,
            )
        else:
            _gemma_dual_rmsnorm_residual_scalar_wide_kernel[(grid,)](
                x1,
                weight1,
                x2,
                weight2,
                weight3,
                residual,
                scalar,
                out,
                num_rows,
                hidden_size,
                eps,
                block_rows,
                groups_per_cluster=block_rows,
                buffer_size_limit=1024,
                num_stages=1,
                enable_fp_fusion=False,
                isCloseMemoryAsync=False,
            )
    return out


__all__ = ["gemma_dual_rmsnorm_residual_scalar"]
