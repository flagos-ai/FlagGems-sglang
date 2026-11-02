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

"""Fused RMS normalization, tanh gating, and residual addition."""

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit
def _rmsnorm_weight(x, inv, weight):
    # Preserve both BF16 rounding stages from the original implementation.
    norm = (x * inv).to(tl.bfloat16).to(tl.float32)
    return (norm * weight).to(tl.bfloat16).to(tl.float32)


@triton.jit
def _apply_gate(gate, norm):
    gate = libdevice.tanh(gate).to(tl.bfloat16).to(tl.float32)
    return (gate * norm).to(tl.bfloat16).to(tl.float32)


@triton.jit
def _row_offset(row, ROW_SHAPE: tl.constexpr, ROW_STRIDES: tl.constexpr):
    offset = tl.full((), 0, tl.int64)
    for index in tl.static_range(len(ROW_SHAPE) - 1, -1, -1):
        offset += (row % ROW_SHAPE[index]).to(tl.int64) * ROW_STRIDES[index]
        row = row // ROW_SHAPE[index]
    return offset


@triton.jit
def _rmsnorm_tanh_residual_strided_kernel(
    x_ptr,
    gate_ptr,
    residual_ptr,
    weight_ptr,
    out_ptr,
    NUM_ROWS: tl.constexpr,
    D: tl.constexpr,
    ROWS_PER_GATE: tl.constexpr,
    ROW_SHAPE: tl.constexpr,
    X_STRIDES: tl.constexpr,
    GATE_SHAPE: tl.constexpr,
    GATE_STRIDES: tl.constexpr,
    RESIDUAL_STRIDES: tl.constexpr,
    W_S: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    cols = tl.arange(0, BLOCK_SIZE)
    w = tl.load(weight_ptr + cols * W_S, cols < D, 0).to(tl.float32)
    for row in range(tl.program_id(0), NUM_ROWS, tl.num_programs(0)):
        x = tl.load(
            x_ptr
            + _row_offset(row, ROW_SHAPE, X_STRIDES)
            + cols * X_STRIDES[len(ROW_SHAPE)],
            cols < D,
            0,
        ).to(tl.float32)
        g = tl.load(
            gate_ptr
            + _row_offset(row // ROWS_PER_GATE, GATE_SHAPE, GATE_STRIDES)
            + cols * GATE_STRIDES[len(GATE_SHAPE)],
            cols < D,
            0,
        ).to(tl.float32)
        r = tl.load(
            residual_ptr
            + _row_offset(row, ROW_SHAPE, RESIDUAL_STRIDES)
            + cols * RESIDUAL_STRIDES[len(ROW_SHAPE)],
            cols < D,
            0,
        ).to(tl.float32)
        inv = tl.rsqrt(tl.sum(x * x, 0) / D + EPS)
        norm = _rmsnorm_weight(x, inv, w)
        product = _apply_gate(g, norm)
        tl.store(out_ptr + row * D + cols, r + product, cols < D)


@triton.jit
def _rmsnorm_tanh_residual_row_kernel(
    x_ptr,
    gate_ptr,
    residual_ptr,
    weight_ptr,
    out_ptr,
    D: tl.constexpr,
    ROWS_PER_GATE: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    CACHE_MODIFIER: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    x = tl.load(
        x_ptr + row * D + cols, cols < D, 0, cache_modifier=CACHE_MODIFIER
    ).to(tl.float32)
    inv = tl.rsqrt(tl.sum(x * x, 0) / D + EPS)
    w = tl.load(
        weight_ptr + cols, cols < D, 0, cache_modifier=CACHE_MODIFIER
    ).to(tl.float32)
    norm = _rmsnorm_weight(x, inv, w)
    g = tl.load(
        gate_ptr + row // ROWS_PER_GATE * D + cols,
        cols < D,
        0,
        cache_modifier=CACHE_MODIFIER,
    ).to(tl.float32)
    product = _apply_gate(g, norm)
    r = tl.load(
        residual_ptr + row * D + cols,
        cols < D,
        0,
        cache_modifier=CACHE_MODIFIER,
    ).to(tl.float32)
    tl.store(out_ptr + row * D + cols, r + product, cols < D)


@triton.jit
def _rmsnorm_tanh_residual_split_kernel(
    x_ptr,
    gate_ptr,
    residual_ptr,
    weight_ptr,
    out_ptr,
    D: tl.constexpr,
    ROWS_PER_GATE: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    PARTS: tl.constexpr,
    NUM_COLS: tl.constexpr,
):
    row = tl.program_id(0) // PARTS
    part = tl.program_id(0) % PARTS
    full_c = tl.arange(0, BLOCK_SIZE)
    full_x = tl.load(x_ptr + row * D + full_c, full_c < D, 0).to(tl.float32)
    inv = tl.rsqrt(tl.sum(full_x * full_x, 0) / D + EPS)
    cols = part * NUM_COLS + tl.arange(0, NUM_COLS)
    x = tl.load(x_ptr + row * D + cols, cols < D, 0).to(tl.float32)
    w = tl.load(weight_ptr + cols, cols < D, 0).to(tl.float32)
    norm = _rmsnorm_weight(x, inv, w)
    g = tl.load(gate_ptr + row // ROWS_PER_GATE * D + cols, cols < D, 0).to(
        tl.float32
    )
    product = _apply_gate(g, norm)
    r = tl.load(residual_ptr + row * D + cols, cols < D, 0).to(tl.float32)
    tl.store(out_ptr + row * D + cols, r + product, cols < D)


def rmsnorm_tanh_residual(x, gate, residual, weight, eps):
    hidden_size = x.shape[-1]
    num_rows = x.numel() // hidden_size
    out = torch.empty_like(x, memory_format=torch.contiguous_format)
    if num_rows == 0:
        return out

    is_hip = torch.version.hip is not None
    rows_per_gate = num_rows // (gate.numel() // hidden_size)
    block_size = triton.next_power_of_2(hidden_size)
    num_warps = 4 if block_size <= 2048 else 8
    dense = all(t.is_contiguous() for t in (x, gate, residual, weight))

    if not dense or num_rows > 65535:
        _rmsnorm_tanh_residual_strided_kernel[min(num_rows, 65535),](
            x,
            gate,
            residual,
            weight,
            out,
            num_rows,
            hidden_size,
            rows_per_gate,
            tuple(x.shape[:-1]),
            tuple(x.stride()),
            tuple(gate.shape[:-1]),
            tuple(gate.stride()),
            tuple(residual.stride()),
            weight.stride(0),
            eps,
            block_size,
            num_warps=num_warps,
            enable_fp_fusion=False,
        )
    elif num_rows <= 64 and block_size >= 1024:
        num_parts = 4
        _rmsnorm_tanh_residual_split_kernel[num_rows * num_parts,](
            x,
            gate,
            residual,
            weight,
            out,
            hidden_size,
            rows_per_gate,
            eps,
            block_size,
            num_parts,
            block_size // num_parts,
            num_warps=4,
            enable_fp_fusion=False,
        )
    else:
        cache_modifier = ".cg" if is_hip and num_rows >= 4096 else ""
        _rmsnorm_tanh_residual_row_kernel[num_rows,](
            x,
            gate,
            residual,
            weight,
            out,
            hidden_size,
            rows_per_gate,
            eps,
            block_size,
            cache_modifier,
            num_warps=num_warps,
            enable_fp_fusion=False,
        )
    return out


__all__ = ["rmsnorm_tanh_residual"]
