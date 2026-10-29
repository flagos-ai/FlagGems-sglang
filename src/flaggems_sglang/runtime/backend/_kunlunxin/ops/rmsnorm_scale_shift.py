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
def _round_bfloat16(value):
    # Round to nearest-even bfloat16 using integer operations.
    bits = value.to(tl.uint32, bitcast=True)
    bits = (bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000
    return bits.to(tl.float32, bitcast=True)


@triton.jit
def _rmsnorm_scale_shift_kernel(
    x_ptr,
    stride_x,
    weight_ptr,
    scale_ptr,
    stride_scale,
    shift_ptr,
    stride_shift,
    out_ptr,
    num_rows,
    hidden_size,
    num_col_blocks,
    rows_per_program,
    rows_per_scale,
    eps,
    BLOCK_D: tl.constexpr,
):
    # Reduce each row in one-dimensional column tiles.
    pid = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK_D)
    for i in range(0, rows_per_program):
        row = tl.minimum(pid * rows_per_program + i, num_rows - 1).to(tl.int64)
        scale_row = row // rows_per_scale
        x_offset = row * stride_x
        out_offset = row * hidden_size
        acc = tl.zeros([BLOCK_D], dtype=tl.float32)
        for col_block in range(0, num_col_blocks):
            offs = col_block * BLOCK_D + cols
            mask = offs < hidden_size
            value = tl.load(x_ptr + x_offset + offs, mask=mask, other=0.0).to(
                tl.float32
            )
            acc += value * value
        rstd = tl.math.rsqrt(tl.sum(acc, axis=0) / hidden_size + eps)
        for col_block in range(0, num_col_blocks):
            offs = col_block * BLOCK_D + cols
            mask = offs < hidden_size
            value = tl.load(x_ptr + x_offset + offs, mask=mask, other=0.0).to(
                tl.float32
            )
            weight = tl.load(weight_ptr + offs, mask=mask, other=0.0).to(
                tl.float32
            )
            scale = tl.load(
                scale_ptr + scale_row * stride_scale + offs,
                mask=mask,
                other=0.0,
            ).to(tl.float32)
            shift = tl.load(
                shift_ptr + scale_row * stride_shift + offs,
                mask=mask,
                other=0.0,
            ).to(tl.float32)
            y = _round_bfloat16(
                value * rstd * weight
            )  # single rounding of the norm
            y = _round_bfloat16(
                y * _round_bfloat16(1.0 + scale)
            )  # bf16 modulation
            y = _round_bfloat16(y + shift)
            tl.store(
                out_ptr + out_offset + offs,
                y.to(out_ptr.dtype.element_ty),
                mask=mask,
            )


def _flatten_rows(tensor, hidden_size):
    rows = tensor.reshape(-1, hidden_size)
    if rows.stride(-1) != 1:
        rows = rows.contiguous()
    return rows


def rmsnorm_scale_shift(x, weight, scale, shift, eps):
    hidden_size = x.shape[-1]
    x_rows = _flatten_rows(x, hidden_size)
    scale_rows = _flatten_rows(scale, hidden_size)
    shift_rows = _flatten_rows(shift, hidden_size)
    weight_tensor = weight if weight.is_contiguous() else weight.contiguous()
    num_rows = x_rows.shape[0]
    out = torch.empty((num_rows, hidden_size), dtype=x.dtype, device=x.device)
    if num_rows == 0 or hidden_size == 0:
        return out.reshape(x.shape)
    rows_per_scale = max(1, num_rows // max(scale_rows.shape[0], 1))
    block_d = max(32, min(2048, triton.next_power_of_2(hidden_size)))
    num_col_blocks = triton.cdiv(hidden_size, block_d)
    num_programs = min(num_rows, 65535)
    rows_per_program = triton.cdiv(num_rows, num_programs)
    _rmsnorm_scale_shift_kernel[(num_programs,)](
        x_rows,
        x_rows.stride(0),
        weight_tensor,
        scale_rows,
        scale_rows.stride(0),
        shift_rows,
        shift_rows.stride(0),
        out,
        num_rows,
        hidden_size,
        num_col_blocks,
        rows_per_program,
        rows_per_scale,
        float(eps),
        BLOCK_D=block_d,
        num_warps=min(16, max(4, block_d // 256)),
    )
    return out.reshape(x.shape)


__all__ = ["rmsnorm_scale_shift"]
