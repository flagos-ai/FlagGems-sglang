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
def _round_to_dtype(value, DTYPE: tl.constexpr):
    if DTYPE == tl.bfloat16:
        bits = value.to(tl.uint32, bitcast=True)
        rounded = ((bits + 32767 + ((bits >> 16) & 1)) & 4294901760).to(
            tl.float32, bitcast=True
        )
        return tl.where((bits & 2139095040) == 2139095040, value, rounded)
    else:
        return value.to(DTYPE).to(tl.float32)


@triton.jit
def _get_row_offset(row, SHAPE: tl.constexpr, STRIDES: tl.constexpr):
    offset = tl.full((), 0, tl.int64)
    for j in tl.static_range(len(SHAPE) - 1, -1, -1):
        offset += (row % SHAPE[j]).to(tl.int64) * STRIDES[j]
        row = row // SHAPE[j]
    return offset


@triton.jit
def _rmsnorm_tanh_residual_kernel(
    x_ptr,
    gate_ptr,
    residual_ptr,
    weight_ptr,
    out_ptr,
    NUM_ROWS: tl.constexpr,
    D: tl.constexpr,
    REPEAT: tl.constexpr,
    X_SHAPE: tl.constexpr,
    X_STRIDES: tl.constexpr,
    GATE_SHAPE: tl.constexpr,
    GATE_STRIDES: tl.constexpr,
    RESIDUAL_STRIDES: tl.constexpr,
    W_S: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    cols = tl.arange(0, BLOCK_D)
    w = tl.load(weight_ptr + cols * W_S, cols < D, other=0).to(tl.float32)
    for row in range(tl.program_id(0), NUM_ROWS, tl.num_programs(0)):
        x = tl.load(
            x_ptr
            + _get_row_offset(row, X_SHAPE, X_STRIDES)
            + cols * X_STRIDES[len(X_SHAPE)],
            cols < D,
            other=0,
        ).to(tl.float32)
        gate = tl.load(
            gate_ptr
            + _get_row_offset(row // REPEAT, GATE_SHAPE, GATE_STRIDES)
            + cols * GATE_STRIDES[len(GATE_SHAPE)],
            cols < D,
            other=0,
        ).to(tl.float32)
        residual = tl.load(
            residual_ptr
            + _get_row_offset(row, X_SHAPE, RESIDUAL_STRIDES)
            + cols * RESIDUAL_STRIDES[len(X_SHAPE)],
            cols < D,
            other=0,
        ).to(tl.float32)
        rstd = tl.rsqrt(tl.sum(x * x, 0) / D + EPS)
        norm = _round_to_dtype(x * rstd, out_ptr.dtype.element_ty)
        norm = _round_to_dtype(norm * w, out_ptr.dtype.element_ty)
        abs_gate = tl.abs(gate)
        exp_gate = tl.exp(-2.0 * abs_gate)
        tanh_gate = (1.0 - exp_gate) / (1.0 + exp_gate)
        tanh_gate = tl.where(gate < 0, -tanh_gate, tanh_gate)
        tanh_gate = tl.where(
            abs_gate < 0.01, gate * (1.0 - gate * gate / 3.0), tanh_gate
        )
        tanh_gate = _round_to_dtype(tanh_gate, out_ptr.dtype.element_ty)
        product = _round_to_dtype(tanh_gate * norm, out_ptr.dtype.element_ty)
        tl.store(out_ptr + row * D + cols, residual + product, cols < D)


def rmsnorm_tanh_residual(x, gate, residual, weight, eps):
    hidden_size = x.shape[-1]
    out = torch.empty(x.shape, dtype=x.dtype, device=x.device)
    if x.numel() == 0:
        return out
    num_rows = x.numel() // hidden_size
    rows_per_gate = num_rows // (gate.numel() // hidden_size)
    block_d = max(64, triton.next_power_of_2(hidden_size))
    _rmsnorm_tanh_residual_kernel[min(num_rows, 65535),](
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
        float(eps),
        block_d,
        num_warps=1,
        num_stages=1,
        enable_fp_fusion=False,
    )
    return out


__all__ = ["rmsnorm_tanh_residual"]
