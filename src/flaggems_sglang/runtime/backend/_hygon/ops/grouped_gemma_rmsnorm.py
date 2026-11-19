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
def _grouped_gemma_rmsnorm_single_group_kernel(
    x_ptr,
    weight_ptr,
    out_ptr,
    GROUP_SIZE: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    EPS: tl.constexpr,
):
    unit_index = tl.program_id(0)
    base = unit_index.to(tl.int64) * GROUP_SIZE
    cols = tl.arange(0, GROUP_SIZE)
    x = tl.load(x_ptr + base + cols).to(tl.float32)
    weight_values = tl.load(
        weight_ptr + (unit_index % NUM_GROUPS) * GROUP_SIZE + cols
    ).to(tl.float32)
    rstd = tl.rsqrt(tl.sum(x * x, 0) / GROUP_SIZE + EPS)
    tl.store(out_ptr + base + cols, x * rstd * (1.0 + weight_values))


@triton.jit
def _grouped_gemma_rmsnorm_tiled_kernel(
    x_ptr,
    weight_ptr,
    out_ptr,
    GROUP_SIZE: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
):
    group_offsets = tl.program_id(0)
    row_offsets = tl.arange(0, BLOCK_ROWS)
    cols = tl.arange(0, GROUP_SIZE)
    unit_index = group_offsets * BLOCK_ROWS + row_offsets
    out_offsets = unit_index.to(tl.int64)[:, None] * GROUP_SIZE + cols[None, :]
    value = (unit_index % NUM_GROUPS)[:, None] * GROUP_SIZE + cols[None, :]
    x = tl.load(x_ptr + out_offsets).to(tl.float32)
    weight_values = tl.load(weight_ptr + value).to(tl.float32)
    rstd = tl.rsqrt(tl.sum(x * x, 1) / GROUP_SIZE + EPS)
    tl.store(out_ptr + out_offsets, x * rstd[:, None] * (1.0 + weight_values))


@triton.jit
def _grouped_gemma_rmsnorm_split_group_kernel(
    x_ptr,
    weight_ptr,
    out_ptr,
    GROUP_PREFIX: tl.constexpr,
    GROUP_SUFFIX: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    EPS: tl.constexpr,
):
    unit_index = tl.program_id(0)
    base = unit_index.to(tl.int64) * (GROUP_PREFIX + GROUP_SUFFIX)
    prefix_cols = tl.arange(0, GROUP_PREFIX)
    suffix_cols = tl.arange(0, GROUP_SUFFIX)
    prefix_values = tl.load(
        x_ptr + base + prefix_cols, cache_modifier=".cg"
    ).to(tl.float32)
    suffix_values = tl.load(
        x_ptr + base + GROUP_PREFIX + suffix_cols, cache_modifier=".cg"
    ).to(tl.float32)
    group_offset = (unit_index % NUM_GROUPS) * (GROUP_PREFIX + GROUP_SUFFIX)
    prefix_weight = tl.load(
        weight_ptr + group_offset + prefix_cols, cache_modifier=".ca"
    ).to(tl.float32)
    suffix_weight = tl.load(
        weight_ptr + group_offset + GROUP_PREFIX + suffix_cols,
        cache_modifier=".ca",
    ).to(tl.float32)
    rstd = tl.rsqrt(
        (
            tl.sum(prefix_values * prefix_values, 0)
            + tl.sum(suffix_values * suffix_values, 0)
        )
        / (GROUP_PREFIX + GROUP_SUFFIX)
        + EPS
    )
    tl.store(
        out_ptr + base + prefix_cols,
        prefix_values * rstd * (1.0 + prefix_weight),
        cache_modifier=".cs",
    )
    tl.store(
        out_ptr + base + GROUP_PREFIX + suffix_cols,
        suffix_values * rstd * (1.0 + suffix_weight),
        cache_modifier=".cs",
    )


@triton.jit
def _grouped_gemma_rmsnorm_padded_group_kernel(
    x_ptr,
    weight_ptr,
    out_ptr,
    GROUP_SIZE: tl.constexpr,
    BLOCK_GROUP: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    EPS: tl.constexpr,
):
    unit_index = tl.program_id(0)
    base = unit_index.to(tl.int64) * GROUP_SIZE
    cols = tl.arange(0, BLOCK_GROUP)
    mask = cols < GROUP_SIZE
    x = tl.load(x_ptr + base + cols, mask, other=0.0).to(tl.float32)
    weight_values = tl.load(
        weight_ptr + (unit_index % NUM_GROUPS) * GROUP_SIZE + cols,
        mask,
        other=0.0,
    ).to(tl.float32)
    rstd = tl.rsqrt(tl.sum(x * x, 0) / GROUP_SIZE + EPS)
    tl.store(out_ptr + base + cols, x * rstd * (1.0 + weight_values), mask)


def grouped_gemma_rmsnorm(
    x,
    weight,
    group_size,
    eps=1e-6,
):
    out = torch.empty_like(x)
    units = x.numel() // group_size
    groups = x.shape[-1] // group_size
    if not group_size & group_size - 1:
        if group_size >= 1024 and units >= 1024 and units % 4 == 0:
            _grouped_gemma_rmsnorm_tiled_kernel.run(
                x,
                weight,
                out,
                group_size,
                groups,
                eps,
                4,
                num_warps=4,
                enable_fp_fusion=False,
                grid=(units // 4,),
                warmup=False,
            )
            return out
        _grouped_gemma_rmsnorm_single_group_kernel.run(
            x,
            weight,
            out,
            group_size,
            groups,
            eps,
            num_warps=1 if group_size <= 512 else 4,
            enable_fp_fusion=False,
            grid=(units,),
            warmup=False,
        )
        return out
    head = 1 << group_size.bit_length() - 1
    tail = group_size - head
    if not tail & tail - 1:
        _grouped_gemma_rmsnorm_split_group_kernel.run(
            x,
            weight,
            out,
            head,
            tail,
            groups,
            eps,
            num_warps=4,
            enable_fp_fusion=False,
            grid=(units,),
            warmup=False,
        )
        return out
    _grouped_gemma_rmsnorm_padded_group_kernel.run(
        x,
        weight,
        out,
        group_size,
        triton.next_power_of_2(group_size),
        groups,
        eps,
        num_warps=4,
        enable_fp_fusion=False,
        grid=(units,),
        warmup=False,
    )
    return out


__all__ = ["grouped_gemma_rmsnorm"]
