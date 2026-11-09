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

"""gemma_dual_rmsnorm_residual_scalar implementation for AMD."""

import torch
import triton
import triton.language as tl


@triton.jit
def _gemma_dual_rmsnorm_residual_scalar_sum_pair(lhs1, lhs2, rhs1, rhs2):
    return (lhs1 + rhs1, lhs2 + rhs2)


@triton.jit
def _gemma_dual_rmsnorm_residual_scalar_strided_kernel(
    x1_ptr,
    weight1_ptr,
    x2_ptr,
    weight2_ptr,
    weight3_ptr,
    residual_ptr,
    scalar_ptr,
    out_ptr,
    METADATA: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, METADATA[13])
    x1 = tl.load(
        x1_ptr + row * METADATA[1] + cols * METADATA[2],
        cols < METADATA[0],
        other=0,
        cache_modifier=".cg",
    ).to(tl.float32)
    x2 = tl.load(
        x2_ptr + row * METADATA[3] + cols * METADATA[4],
        cols < METADATA[0],
        other=0,
        cache_modifier=".cg",
    ).to(tl.float32)
    w1 = tl.load(
        weight1_ptr + cols * METADATA[7], cols < METADATA[0], other=0
    ).to(tl.float32)
    w2 = tl.load(
        weight2_ptr + cols * METADATA[8], cols < METADATA[0], other=0
    ).to(tl.float32)
    (x1_sum_sq, x2_sum_sq) = tl.reduce(
        (x1 * x1, x2 * x2), 0, _gemma_dual_rmsnorm_residual_scalar_sum_pair
    )
    x1_norm = x1 * tl.rsqrt(x1_sum_sq / METADATA[0] + METADATA[10]) * w1
    x2_norm = x2 * tl.rsqrt(x2_sum_sq / METADATA[0] + METADATA[11]) * w2
    combined = x1_norm + x2_norm
    w3 = tl.load(
        weight3_ptr + cols * METADATA[9], cols < METADATA[0], other=0
    ).to(tl.float32)
    out_norm = (
        combined
        * tl.rsqrt(tl.sum(combined * combined, 0) / METADATA[0] + METADATA[12])
        * w3
    )
    residual = tl.load(
        residual_ptr + row * METADATA[5] + cols * METADATA[6],
        cols < METADATA[0],
        other=0,
        cache_modifier=".cg",
    ).to(tl.float32)
    scalar = tl.load(scalar_ptr).to(tl.float32)
    y = (out_norm + residual) * scalar
    tl.store(out_ptr + row * METADATA[0] + cols, y, cols < METADATA[0])


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
    METADATA: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, METADATA[4])
    x1 = tl.load(
        x1_ptr + row * METADATA[0] + cols * 1,
        cols < METADATA[0],
        other=0,
        cache_modifier=".cg",
    ).to(tl.float32)
    x2 = tl.load(
        x2_ptr + row * METADATA[0] + cols * 1,
        cols < METADATA[0],
        other=0,
        cache_modifier=".cg",
    ).to(tl.float32)
    w1 = tl.load(weight1_ptr + cols * 1, cols < METADATA[0], other=0).to(
        tl.float32
    )
    w2 = tl.load(weight2_ptr + cols * 1, cols < METADATA[0], other=0).to(
        tl.float32
    )
    (x1_sum_sq, x2_sum_sq) = tl.reduce(
        (x1 * x1, x2 * x2), 0, _gemma_dual_rmsnorm_residual_scalar_sum_pair
    )
    x1_norm = x1 * tl.rsqrt(x1_sum_sq / METADATA[0] + METADATA[1]) * w1
    x2_norm = x2 * tl.rsqrt(x2_sum_sq / METADATA[0] + METADATA[2]) * w2
    combined = x1_norm + x2_norm
    w3 = tl.load(weight3_ptr + cols * 1, cols < METADATA[0], other=0).to(
        tl.float32
    )
    out_norm = (
        combined
        * tl.rsqrt(tl.sum(combined * combined, 0) / METADATA[0] + METADATA[3])
        * w3
    )
    residual = tl.load(
        residual_ptr + row * METADATA[0] + cols * 1,
        cols < METADATA[0],
        other=0,
        cache_modifier=".cg",
    ).to(tl.float32)
    scalar = tl.load(scalar_ptr).to(tl.float32)
    y = (out_norm + residual) * scalar
    tl.store(
        out_ptr + row * METADATA[0] + cols,
        y,
        cols < METADATA[0],
        cache_modifier=".cs" if METADATA[0] == 4096 and METADATA[5] else "",
    )


@triton.jit
def _gemma_dual_rmsnorm_residual_scalar_overlap_kernel(
    x1_ptr,
    weight1_ptr,
    x2_ptr,
    weight2_ptr,
    weight3_ptr,
    residual_ptr,
    scalar_ptr,
    out_ptr,
    METADATA: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, METADATA[4])
    x1 = tl.load(
        x1_ptr + row * METADATA[0] + cols * 1,
        cols < METADATA[0],
        other=0,
        cache_modifier=".cg",
    ).to(tl.float32)
    x2 = tl.load(
        x2_ptr + row * METADATA[0] + cols * 1,
        cols < METADATA[0],
        other=0,
        cache_modifier=".cg",
    ).to(tl.float32)
    (x1_sum_sq, x2_sum_sq) = tl.reduce(
        (x1 * x1, x2 * x2), 0, _gemma_dual_rmsnorm_residual_scalar_sum_pair
    )
    w1 = tl.load(weight1_ptr + cols * 1, cols < METADATA[0], other=0).to(
        tl.float32
    )
    w2 = tl.load(weight2_ptr + cols * 1, cols < METADATA[0], other=0).to(
        tl.float32
    )
    x1_norm = x1 * tl.rsqrt(x1_sum_sq / METADATA[0] + METADATA[1]) * w1
    x2_norm = x2 * tl.rsqrt(x2_sum_sq / METADATA[0] + METADATA[2]) * w2
    combined = x1_norm + x2_norm
    w3 = tl.load(weight3_ptr + cols * 1, cols < METADATA[0], other=0).to(
        tl.float32
    )
    residual_raw = tl.load(
        residual_ptr + row * METADATA[0] + cols * 1,
        cols < METADATA[0],
        other=0,
        cache_modifier=".cg",
    )
    out_norm = (
        combined
        * tl.rsqrt(tl.sum(combined * combined, 0) / METADATA[0] + METADATA[3])
        * w3
    )
    residual = residual_raw.to(tl.float32)
    scalar = tl.load(scalar_ptr).to(tl.float32)
    y = (out_norm + residual) * scalar
    tl.store(
        out_ptr + row * METADATA[0] + cols,
        y,
        cols < METADATA[0],
        cache_modifier=".cs" if METADATA[0] == 4096 and METADATA[5] else "",
    )


def gemma_dual_rmsnorm_residual_scalar(
    x1,
    weight1,
    x2,
    weight2,
    weight3,
    residual,
    scalar,
    eps1,
    eps2,
    eps3,
):
    (num_rows, hidden_size) = x1.shape
    out = torch.empty_like(x1, memory_format=torch.contiguous_format)
    if num_rows:
        block_d = triton.next_power_of_2(hidden_size)
        num_warps = 16 if block_d >= 8192 else 4
        if (
            x1.is_contiguous()
            and weight1.is_contiguous()
            and x2.is_contiguous()
            and weight2.is_contiguous()
            and weight3.is_contiguous()
            and residual.is_contiguous()
        ):
            num_warps = (
                1
                if hidden_size == 1024
                and num_rows >= 1024
                and (torch.version.hip is not None)
                else (
                    2
                    if hidden_size == 1024
                    and num_rows >= 512
                    and (torch.version.hip is None)
                    else num_warps
                )
            )
            if (
                hidden_size == 4096
                and num_rows >= 2048
                and (torch.version.hip is not None)
            ):
                _gemma_dual_rmsnorm_residual_scalar_contiguous_kernel[
                    num_rows,
                ](
                    x1,
                    weight1,
                    x2,
                    weight2,
                    weight3,
                    residual,
                    scalar,
                    out,
                    (hidden_size, eps1, eps2, eps3, block_d, num_rows >= 2048),
                    num_warps=num_warps,
                    enable_fp_fusion=False,
                    waves_per_eu=8,
                )
            else:
                kernel = (
                    _gemma_dual_rmsnorm_residual_scalar_overlap_kernel
                    if num_rows >= 512
                    and hidden_size in (4096, 8192)
                    and (torch.version.hip is None)
                    else _gemma_dual_rmsnorm_residual_scalar_contiguous_kernel
                )
                kernel[num_rows,](
                    x1,
                    weight1,
                    x2,
                    weight2,
                    weight3,
                    residual,
                    scalar,
                    out,
                    (hidden_size, eps1, eps2, eps3, block_d, num_rows >= 2048),
                    num_warps=num_warps,
                    enable_fp_fusion=False,
                )
        else:
            _gemma_dual_rmsnorm_residual_scalar_strided_kernel[num_rows,](
                x1,
                weight1,
                x2,
                weight2,
                weight3,
                residual,
                scalar,
                out,
                (
                    hidden_size,
                    x1.stride(0),
                    x1.stride(1),
                    x2.stride(0),
                    x2.stride(1),
                    residual.stride(0),
                    residual.stride(1),
                    weight1.stride(0),
                    weight2.stride(0),
                    weight3.stride(0),
                    eps1,
                    eps2,
                    eps3,
                    block_d,
                ),
                num_warps=num_warps,
                enable_fp_fusion=False,
            )
    return out


__all__ = ["gemma_dual_rmsnorm_residual_scalar"]
