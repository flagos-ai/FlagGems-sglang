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
from triton.language.extra import libdevice as lib


@triton.jit
def _silu_and_mul_masked_post_quant_scaled_silu(gate):
    denominator = 1.0 + tl.exp(-gate)
    direct = lib.fast_dividef(
        gate * 5.960464477539063e-08, denominator * 5.960464477539063e-08
    )
    return tl.where(tl.abs(gate) < 7.888609052210118e-31, gate * 0.5, direct)


@triton.jit
def _silu_and_mul_masked_post_quant_divide_refined(
    value,
    scale,
    inv_scale,
):
    initial = value * inv_scale
    residual = tl.fma(-initial, scale, value)
    return tl.fma(residual, inv_scale, initial)


@triton.jit
def _silu_and_mul_masked_post_quant_encode_fp8(value):
    raw = value.to(tl.uint32, bitcast=True)
    sign = raw >> 24 & 128
    magnitude = raw & 2147483647
    exponent = tl.maximum(magnitude >> 23, 121)
    magic_bits = exponent * 8388616 + 167771192 | sign
    rounded = (tl.abs(value) + magic_bits.to(tl.float32, bitcast=True)).to(
        tl.uint32, bitcast=True
    )
    return tl.where(value != value, 127 | sign, rounded & 255).to(tl.uint8)


@triton.jit
def _silu_and_mul_masked_post_quant_kernel(
    x_ptr,
    masked_m_ptr,
    out_ptr,
    scale_ptr,
    NUM_TOKENS: tl.constexpr,
    HIDDEN_SIZE: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    STRIDE_X_EXPERT: tl.constexpr,
    STRIDE_X_TOKEN: tl.constexpr,
    STRIDE_X_COL: tl.constexpr,
    STRIDE_MASKED_M: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
):
    group = tl.program_id(0) % tl.cdiv(
        GROUP_SIZE, BLOCK_ROWS
    ) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    row = tl.program_id(0) // tl.cdiv(GROUP_SIZE, BLOCK_ROWS)
    expert = row // NUM_TOKENS
    token = row % NUM_TOKENS
    group_cols = tl.arange(0, BLOCK_SIZE)
    col = group[:, None] * (HIDDEN_SIZE // GROUP_SIZE) + group_cols[None, :]
    base = expert * STRIDE_X_EXPERT + token * STRIDE_X_TOKEN
    if (
        GROUP_SIZE % BLOCK_ROWS == 0
        and BLOCK_SIZE == HIDDEN_SIZE // GROUP_SIZE
    ):
        mask = tl.full((BLOCK_ROWS, BLOCK_SIZE), True, tl.int1)
    else:
        mask = (group[:, None] < GROUP_SIZE) & (
            group_cols[None, :] < HIDDEN_SIZE // GROUP_SIZE
        )
    gate = tl.load(x_ptr + base + col * STRIDE_X_COL, mask, 0).to(tl.float32)
    up = tl.load(
        x_ptr + base + (col + HIDDEN_SIZE) * STRIDE_X_COL, mask, 0
    ).to(tl.float32)
    value = _silu_and_mul_masked_post_quant_scaled_silu(gate) * up
    maximum = tl.max(tl.abs(value), 1)
    maximum = tl.maximum(maximum, 1e-10)
    scale = maximum * 0.002232142857142857
    inv_scale = tl.div_rn(1.0, scale)
    quantized = tl.minimum(
        tl.maximum(
            _silu_and_mul_masked_post_quant_divide_refined(
                value, scale[:, None], inv_scale[:, None]
            ),
            -448.0,
        ),
        448.0,
    )
    valid = token < tl.load(masked_m_ptr + expert * STRIDE_MASKED_M)
    code = tl.where(
        valid, _silu_and_mul_masked_post_quant_encode_fp8(quantized), 0
    ).to(tl.uint8)
    tl.store(out_ptr + row * HIDDEN_SIZE + col, code, mask)
    if GROUP_SIZE % BLOCK_ROWS == 0:
        tl.store(
            scale_ptr + row * GROUP_SIZE + group, tl.where(valid, scale, 0.0)
        )
    else:
        tl.store(
            scale_ptr + row * GROUP_SIZE + group,
            tl.where(valid, scale, 0.0),
            group < GROUP_SIZE,
        )


@triton.jit
def _silu_and_mul_masked_post_quant_packed_group_rows_kernel(
    x_ptr,
    masked_m_ptr,
    out_ptr,
    scale_ptr,
    NUM_TOKENS: tl.constexpr,
    HIDDEN_SIZE: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    STRIDE_X_EXPERT: tl.constexpr,
    STRIDE_X_TOKEN: tl.constexpr,
    STRIDE_MASKED_M: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
):
    x_ptr = x_ptr.to(tl.pointer_type(tl.int32))
    out_ptr = out_ptr.to(tl.pointer_type(tl.int16))
    group = tl.program_id(0) % (
        (GROUP_SIZE + BLOCK_ROWS - 1) // BLOCK_ROWS
    ) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    row = tl.program_id(0) // ((GROUP_SIZE + BLOCK_ROWS - 1) // BLOCK_ROWS)
    expert = row // NUM_TOKENS
    token = row % NUM_TOKENS
    group_cols = tl.arange(0, BLOCK_SIZE)
    col = (
        group[:, None] * (HIDDEN_SIZE // GROUP_SIZE // 2) + group_cols[None, :]
    )
    base = expert * STRIDE_X_EXPERT + token * STRIDE_X_TOKEN
    mask = (group[:, None] < GROUP_SIZE) & (
        group_cols[None, :] < HIDDEN_SIZE // GROUP_SIZE // 2
    )
    gate_bits = tl.load(x_ptr + base + col, mask, 0).to(tl.uint32)
    up_bits = tl.load(x_ptr + base + col + HIDDEN_SIZE // 2, mask, 0).to(
        tl.uint32
    )
    gate_0 = (gate_bits << 16).to(tl.float32, bitcast=True)
    gate_1 = (gate_bits & 4294901760).to(tl.float32, bitcast=True)
    up_0 = (up_bits << 16).to(tl.float32, bitcast=True)
    up_1 = (up_bits & 4294901760).to(tl.float32, bitcast=True)
    value_0 = _silu_and_mul_masked_post_quant_scaled_silu(gate_0) * up_0
    value_1 = _silu_and_mul_masked_post_quant_scaled_silu(gate_1) * up_1
    maximum = tl.max(tl.maximum(tl.abs(value_0), tl.abs(value_1)), 1)
    scale = tl.maximum(maximum, 1e-10) * 0.002232142857142857
    inv_scale = tl.div_rn(1.0, scale)
    quant_0 = tl.minimum(
        tl.maximum(
            _silu_and_mul_masked_post_quant_divide_refined(
                value_0, scale[:, None], inv_scale[:, None]
            ),
            -448.0,
        ),
        448.0,
    )
    quant_1 = tl.minimum(
        tl.maximum(
            _silu_and_mul_masked_post_quant_divide_refined(
                value_1, scale[:, None], inv_scale[:, None]
            ),
            -448.0,
        ),
        448.0,
    )
    code_0 = _silu_and_mul_masked_post_quant_encode_fp8(quant_0).to(tl.uint32)
    code_1 = _silu_and_mul_masked_post_quant_encode_fp8(quant_1).to(tl.uint32)
    valid = token < tl.load(masked_m_ptr + expert * STRIDE_MASKED_M)
    packed = tl.where(valid, code_0 | code_1 << 8, 0).to(tl.int16)
    tl.store(out_ptr + row * (HIDDEN_SIZE // 2) + col, packed, mask)
    tl.store(
        scale_ptr + row * GROUP_SIZE + group,
        tl.where(valid, scale, 0.0),
        group < GROUP_SIZE,
    )


def silu_and_mul_masked_post_quant(
    input,
    quant_group_size,
    masked_m,
):
    experts, tokens, width = input.shape
    hidden = width // 2
    groups = hidden // quant_group_size
    quantized = torch.empty(
        (experts, tokens, hidden), dtype=torch.uint8, device=input.device
    )
    scales = torch.empty(
        (experts, tokens, groups), dtype=torch.float32, device=input.device
    )
    count = experts * tokens * groups
    if count:
        if (
            input.stride(-1) == 1
            and hidden % 2 == 0
            and (quant_group_size % 2 == 0)
            and (input.storage_offset() % 2 == 0)
            and (input.stride(0) % 2 == 0)
            and (input.stride(1) % 2 == 0)
        ):
            rows = 8
            _silu_and_mul_masked_post_quant_packed_group_rows_kernel.run(
                input,
                masked_m,
                quantized,
                scales,
                tokens,
                hidden,
                groups,
                input.stride(0) // 2,
                input.stride(1) // 2,
                masked_m.stride(0),
                triton.next_power_of_2(quant_group_size // 2),
                count,
                rows,
                num_warps=1,
                enable_fp_fusion=False,
                grid=(experts * tokens * triton.cdiv(groups, rows),),
                warmup=False,
            )
        else:
            rows = 8
            _silu_and_mul_masked_post_quant_kernel.run(
                input,
                masked_m,
                quantized,
                scales,
                tokens,
                hidden,
                groups,
                *input.stride(),
                masked_m.stride(0),
                triton.next_power_of_2(quant_group_size),
                count,
                rows,
                num_warps=1,
                enable_fp_fusion=False,
                grid=(experts * tokens * triton.cdiv(groups, rows),),
                warmup=False,
            )
    return (quantized.view(torch.float8_e4m3fn), scales)


__all__ = ["silu_and_mul_masked_post_quant"]
