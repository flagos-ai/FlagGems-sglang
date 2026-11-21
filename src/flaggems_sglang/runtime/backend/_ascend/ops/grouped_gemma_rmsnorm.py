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


@triton.jit(
    do_not_specialize=["x_ptr", "weight_ptr", "out_ptr"],
    do_not_specialize_on_alignment=["x_ptr", "weight_ptr", "out_ptr"],
)
def _grouped_gemma_rmsnorm_single_group_kernel(
    x_ptr,
    weight_ptr,
    out_ptr,
    GROUP_SIZE: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    EPS: tl.constexpr,
    NUM_UNITS: tl.constexpr,
):
    cols = tl.arange(0, GROUP_SIZE)
    for unit_index in tl.range(
        tl.program_id(0), NUM_UNITS, tl.num_programs(0), num_stages=2
    ):
        base = unit_index.to(tl.int64) * GROUP_SIZE
        x = tl.load(x_ptr + base + cols).to(tl.float32)
        weight_values = tl.load(
            weight_ptr + (unit_index % NUM_GROUPS) * GROUP_SIZE + cols
        ).to(tl.float32)
        rstd = tl.rsqrt(tl.sum(x * x, 0) / GROUP_SIZE + EPS)
        tl.store(out_ptr + base + cols, x * rstd * (1.0 + weight_values))


@triton.jit(
    do_not_specialize=["x_ptr", "weight_ptr", "out_ptr"],
    do_not_specialize_on_alignment=["x_ptr", "weight_ptr", "out_ptr"],
)
def _grouped_gemma_rmsnorm_split_group_kernel(
    x_ptr,
    weight_ptr,
    out_ptr,
    GROUP_PREFIX: tl.constexpr,
    GROUP_SUFFIX: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    EPS: tl.constexpr,
    NUM_UNITS: tl.constexpr,
):
    prefix_cols = tl.arange(0, GROUP_PREFIX)
    suffix_cols = tl.arange(0, GROUP_SUFFIX)
    for unit_index in tl.range(
        tl.program_id(0), NUM_UNITS, tl.num_programs(0), num_stages=2
    ):
        base = unit_index.to(tl.int64) * (GROUP_PREFIX + GROUP_SUFFIX)
        prefix_values = tl.load(x_ptr + base + prefix_cols).to(tl.float32)
        suffix_values = tl.load(x_ptr + base + GROUP_PREFIX + suffix_cols).to(
            tl.float32
        )
        group_offset = (unit_index % NUM_GROUPS) * (
            GROUP_PREFIX + GROUP_SUFFIX
        )
        prefix_weight = tl.load(weight_ptr + group_offset + prefix_cols).to(
            tl.float32
        )
        suffix_weight = tl.load(
            weight_ptr + group_offset + GROUP_PREFIX + suffix_cols
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
        )
        tl.store(
            out_ptr + base + GROUP_PREFIX + suffix_cols,
            suffix_values * rstd * (1.0 + suffix_weight),
        )


@triton.jit(
    do_not_specialize=["x_ptr", "weight_ptr", "out_ptr"],
    do_not_specialize_on_alignment=["x_ptr", "weight_ptr", "out_ptr"],
)
def _grouped_gemma_rmsnorm_padded_group_kernel(
    x_ptr,
    weight_ptr,
    out_ptr,
    GROUP_SIZE: tl.constexpr,
    BLOCK_GROUP: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    EPS: tl.constexpr,
    NUM_UNITS: tl.constexpr,
):
    cols = tl.arange(0, BLOCK_GROUP)
    mask = cols < GROUP_SIZE
    for unit_index in tl.range(
        tl.program_id(0), NUM_UNITS, tl.num_programs(0), num_stages=2
    ):
        base = unit_index.to(tl.int64) * GROUP_SIZE
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
    num_programs = units if units < 40 else 40
    if not group_size & group_size - 1:
        _grouped_gemma_rmsnorm_single_group_kernel.run(
            x,
            weight,
            out,
            group_size,
            groups,
            eps,
            units,
            num_warps=4 if group_size <= 2048 else 8,
            num_stages=1,
            enable_fp_fusion=False,
            grid=(num_programs,),
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
            units,
            num_warps=4 if group_size <= 2048 else 8,
            num_stages=1,
            enable_fp_fusion=False,
            grid=(num_programs,),
            warmup=False,
        )
        return out
    block_size = triton.next_power_of_2(group_size)
    _grouped_gemma_rmsnorm_padded_group_kernel.run(
        x,
        weight,
        out,
        group_size,
        block_size,
        groups,
        eps,
        units,
        num_warps=4 if block_size <= 2048 else 8,
        num_stages=1,
        enable_fp_fusion=False,
        grid=(num_programs,),
        warmup=False,
    )
    return out


__all__ = ["grouped_gemma_rmsnorm"]
