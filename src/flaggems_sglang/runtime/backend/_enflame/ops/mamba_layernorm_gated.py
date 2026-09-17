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
def _fallback_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    z_ptr,
    output_ptr,
    row_count,
    group_count,
    group_size: tl.constexpr,
    eps,
    has_bias: tl.constexpr,
    has_z: tl.constexpr,
    norm_before_gate: tl.constexpr,
    is_rms_norm: tl.constexpr,
    block_size: tl.constexpr,
):
    program = tl.program_id(0)
    program_count = tl.num_programs(0)
    columns = tl.arange(0, block_size)
    mask = columns < group_size
    for row in tl.range(program, row_count, program_count):
        offsets = row * group_size + columns
        feature_offsets = (row % group_count) * group_size + columns
        values = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        if has_z:
            gate = tl.load(z_ptr + offsets, mask=mask, other=0.0).to(
                tl.float32
            )
            silu_gate = gate * tl.sigmoid(gate)
            if not norm_before_gate:
                values *= silu_gate
        if is_rms_norm:
            centered = values
        else:
            mean = tl.sum(values, axis=0) / group_size
            centered = values - mean
        variance = tl.sum(centered * centered, axis=0) / group_size
        normalized = centered * tl.rsqrt(variance + eps)
        weight = tl.load(
            weight_ptr + feature_offsets, mask=mask, other=0.0
        ).to(tl.float32)
        output = normalized * weight
        if has_bias:
            output += tl.load(
                bias_ptr + feature_offsets, mask=mask, other=0.0
            ).to(tl.float32)
        if has_z and norm_before_gate:
            output *= silu_gate
        tl.store(output_ptr + offsets, output, mask=mask)


@triton.jit
def _performance_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    z_ptr,
    output_ptr,
    row_count,
    task_count,
    eps,
    hidden: tl.constexpr,
    group_count: tl.constexpr,
    group_size: tl.constexpr,
    rows_per_program: tl.constexpr,
):
    program = tl.program_id(0)
    program_count = tl.num_programs(0)
    row_lanes = tl.arange(0, rows_per_program)
    columns = tl.arange(0, group_size)
    for task in tl.range(program, task_count, program_count):
        group = task % group_count
        row_block = task // group_count
        rows = row_block * rows_per_program + row_lanes
        mask = rows[:, None] < row_count
        offsets = (
            rows[:, None] * hidden + group * group_size + columns[None, :]
        )
        feature_offsets = group * group_size + columns
        values = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        gate = tl.load(z_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        weight = tl.load(weight_ptr + feature_offsets).to(tl.float32)
        bias = tl.load(bias_ptr + feature_offsets).to(tl.float32)
        variance = tl.sum(values * values, axis=1) / group_size
        normalized = values * tl.rsqrt(variance[:, None] + eps)
        output = (
            (normalized * weight[None, :] + bias[None, :])
            * gate
            * tl.sigmoid(gate)
        )
        tl.store(output_ptr + offsets, output, mask=mask)


@triton.jit
def _performance_multi_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    z_ptr,
    output_ptr,
    row_count,
    task_count,
    eps,
    hidden: tl.constexpr,
    group_count: tl.constexpr,
    group_size: tl.constexpr,
    rows_per_program: tl.constexpr,
    groups_per_program: tl.constexpr,
):
    program = tl.program_id(0)
    program_count = tl.num_programs(0)
    row_lanes = tl.arange(0, rows_per_program)
    group_lanes = tl.arange(0, groups_per_program)
    columns = tl.arange(0, group_size)
    group_block_count = group_count // groups_per_program
    for task in tl.range(program, task_count, program_count):
        group_block = task % group_block_count
        row_block = task // group_block_count
        rows = row_block * rows_per_program + row_lanes
        groups = group_block * groups_per_program + group_lanes
        mask = rows[:, None, None] < row_count
        offsets = (
            rows[:, None, None] * hidden
            + groups[None, :, None] * group_size
            + columns[None, None, :]
        )
        feature_offsets = groups[:, None] * group_size + columns[None, :]
        values = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        gate = tl.load(z_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        weight = tl.load(weight_ptr + feature_offsets).to(tl.float32)
        bias = tl.load(bias_ptr + feature_offsets).to(tl.float32)
        variance = tl.sum(values * values, axis=2) / group_size
        normalized = values * tl.rsqrt(variance[:, :, None] + eps)
        result = (
            (normalized * weight[None, :, :] + bias[None, :, :])
            * gate
            * tl.sigmoid(gate)
        )
        tl.store(output_ptr + offsets, result, mask=mask)


@triton.jit
def _performance_multi_mixed_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    z_ptr,
    output_ptr,
    row_count: tl.constexpr,
    task_count: tl.constexpr,
    eps: tl.constexpr,
    hidden: tl.constexpr,
    group_count: tl.constexpr,
    group_size: tl.constexpr,
    rows_per_program: tl.constexpr,
    groups_per_program: tl.constexpr,
):
    program = tl.program_id(0)
    program_count = tl.num_programs(0)
    row_lanes = tl.arange(0, rows_per_program)
    group_lanes = tl.arange(0, groups_per_program)
    columns = tl.arange(0, group_size)
    group_block_count = group_count // groups_per_program
    for task in tl.range(program, task_count, program_count):
        group_block = task % group_block_count
        row_block = task // group_block_count
        rows = row_block * rows_per_program + row_lanes
        groups = group_block * groups_per_program + group_lanes
        mask = rows[:, None, None] < row_count
        offsets = (
            rows[:, None, None] * hidden
            + groups[None, :, None] * group_size
            + columns[None, None, :]
        )
        feature_offsets = groups[:, None] * group_size + columns[None, :]
        values = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        weight = tl.load(weight_ptr + feature_offsets)
        bias = tl.load(bias_ptr + feature_offsets)
        variance = tl.sum(values * values, axis=2) * (1.0 / group_size)
        normalized = values * tl.rsqrt(variance[:, :, None] + eps)
        affine = normalized * weight[None, :, :] + bias[None, :, :]
        gate_raw = tl.load(z_ptr + offsets, mask=mask, other=0.0)
        gate = gate_raw.to(tl.float32)
        result = affine * gate_raw * tl.sigmoid(gate)
        tl.store(output_ptr + offsets, result, mask=mask)


@triton.jit
def _performance_fixed_multi_mixed_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    z_ptr,
    output_ptr,
    row_count: tl.constexpr,
    row_block_count: tl.constexpr,
    eps: tl.constexpr,
    hidden: tl.constexpr,
    group_count: tl.constexpr,
    group_size: tl.constexpr,
    rows_per_program: tl.constexpr,
    groups_per_program: tl.constexpr,
    replicas: tl.constexpr,
    use_fp16_affine: tl.constexpr,
):
    program = tl.program_id(0)
    group_block_count = group_count // groups_per_program
    group_block = program % group_block_count
    replica = program // group_block_count
    row_lanes = tl.arange(0, rows_per_program)
    group_lanes = tl.arange(0, groups_per_program)
    columns = tl.arange(0, group_size)
    groups = group_block * groups_per_program + group_lanes
    feature_offsets = groups[:, None] * group_size + columns[None, :]
    weight = tl.load(weight_ptr + feature_offsets)
    bias = tl.load(bias_ptr + feature_offsets)
    for row_block in tl.range(replica, row_block_count, replicas):
        rows = row_block * rows_per_program + row_lanes
        mask = rows[:, None, None] < row_count
        offsets = (
            rows[:, None, None] * hidden
            + groups[None, :, None] * group_size
            + columns[None, None, :]
        )
        values = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        variance = tl.sum(values * values, axis=2) * (1.0 / group_size)
        normalized = values * tl.rsqrt(variance[:, :, None] + eps)
        if use_fp16_affine:
            affine = (
                normalized.to(tl.float16) * weight.to(tl.float16)[None, :, :]
                + bias.to(tl.float16)[None, :, :]
            )
        else:
            affine = normalized * weight[None, :, :] + bias[None, :, :]
        gate_raw = tl.load(z_ptr + offsets, mask=mask, other=0.0)
        gate = gate_raw.to(tl.float32)
        result = affine * gate_raw * tl.sigmoid(gate)
        tl.store(output_ptr + offsets, result, mask=mask)


def mamba_layernorm_gated(
    x,
    weight,
    bias,
    eps,
    z=None,
    group_size=None,
    norm_before_gate=True,
    is_rms_norm=True,
):
    row_count, hidden = x.shape
    if group_size is None:
        group_size = hidden
    group_count = hidden // group_size
    total_rows = row_count * group_count
    output = torch.empty_like(x)
    if (
        hidden == 4096
        and group_size == 128
        and z is not None
        and bias is not None
        and norm_before_gate
        and is_rms_norm
    ):
        if x.dtype == torch.bfloat16 and row_count >= 512:
            if row_count == 512:
                rows_per_program, groups_per_program, replicas = 128, 2, 3
                num_stages, fp_fusion = 2, True
            else:
                rows_per_program, groups_per_program, replicas = 16, 32, 48
                num_stages, fp_fusion = 3, False
            group_block_count = group_count // groups_per_program
            row_block_count = triton.cdiv(row_count, rows_per_program)
            _performance_fixed_multi_mixed_kernel[
                (group_block_count * replicas,)
            ](
                x,
                weight,
                bias,
                z,
                output,
                row_count,
                row_block_count,
                eps,
                hidden,
                group_count,
                group_size,
                rows_per_program,
                groups_per_program,
                replicas,
                True,
                num_warps=1,
                num_stages=num_stages,
                enable_fp_fusion=fp_fusion,
            )
            return output
        if x.dtype == torch.bfloat16 and row_count in (8, 64):
            if row_count == 8:
                rows_per_program, groups_per_program, replicas = 8, 4, 1
            else:
                rows_per_program, groups_per_program, replicas = 64, 2, 1
            group_block_count = group_count // groups_per_program
            row_block_count = triton.cdiv(row_count, rows_per_program)
            _performance_fixed_multi_mixed_kernel[
                (group_block_count * replicas,)
            ](
                x,
                weight,
                bias,
                z,
                output,
                row_count,
                row_block_count,
                eps,
                hidden,
                group_count,
                group_size,
                rows_per_program,
                groups_per_program,
                replicas,
                True,
                num_warps=1,
                num_stages=1,
                enable_fp_fusion=False,
            )
            return output
        if x.dtype == torch.bfloat16 and row_count == 1:
            rows_per_program, groups_per_program, grid_cap = 1, 8, 64
            num_stages, fp_fusion = 1, False
        elif x.dtype == torch.bfloat16 and row_count <= 8:
            rows_per_program, groups_per_program, grid_cap = 8, 4, 64
            num_stages, fp_fusion = 3, False
        elif x.dtype == torch.bfloat16 and row_count <= 64:
            rows_per_program, groups_per_program, grid_cap = 64, 2, 48
            num_stages, fp_fusion = 1, True
        elif row_count <= 8:
            rows_per_program, groups_per_program, grid_cap = 8, 1, 64
            num_stages, fp_fusion = 2, True
        else:
            rows_per_program, groups_per_program, grid_cap = 128, 1, 48
            num_stages, fp_fusion = 2, True
        num_warps = 1
        group_block_count = group_count // groups_per_program
        task_count = (
            triton.cdiv(row_count, rows_per_program) * group_block_count
        )
        if x.dtype == torch.bfloat16:
            _performance_multi_mixed_kernel[(min(task_count, grid_cap),)](
                x,
                weight,
                bias,
                z,
                output,
                row_count,
                task_count,
                eps,
                hidden,
                group_count,
                group_size,
                rows_per_program,
                groups_per_program,
                num_warps=num_warps,
                num_stages=num_stages,
                enable_fp_fusion=fp_fusion,
            )
        elif groups_per_program == 1:
            _performance_kernel[(min(task_count, grid_cap),)](
                x,
                weight,
                bias,
                z,
                output,
                row_count,
                task_count,
                eps,
                hidden,
                group_count,
                group_size,
                rows_per_program,
                num_warps=num_warps,
                num_stages=num_stages,
            )
        else:
            _performance_multi_kernel[(min(task_count, grid_cap),)](
                x,
                weight,
                bias,
                z,
                output,
                row_count,
                task_count,
                eps,
                hidden,
                group_count,
                group_size,
                rows_per_program,
                groups_per_program,
                num_warps=num_warps,
                num_stages=num_stages,
            )
        return output
    block_size = triton.next_power_of_2(group_size)
    _fallback_kernel[(min(total_rows, 32768),)](
        x,
        weight,
        bias,
        z,
        output,
        total_rows,
        group_count,
        group_size,
        eps,
        bias is not None,
        z is not None,
        norm_before_gate,
        is_rms_norm,
        block_size,
        num_warps=4,
        num_stages=2,
    )
    return output


__all__ = ["mamba_layernorm_gated"]
