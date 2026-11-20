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
    out_ptr: tl.pointer_type(tl.uint8),
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
    out_ptr = out_ptr.to(tl.pointer_type(tl.uint8))
    group = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    row = group // GROUP_SIZE
    expert = row // NUM_TOKENS
    token = row % NUM_TOKENS
    group_cols = tl.arange(0, BLOCK_SIZE)
    col = (
        group[:, None] % GROUP_SIZE * (HIDDEN_SIZE // GROUP_SIZE)
        + group_cols[None, :]
    )
    base = expert * STRIDE_X_EXPERT + token * STRIDE_X_TOKEN
    if (
        NUM_GROUPS % BLOCK_ROWS == 0
        and BLOCK_SIZE == HIDDEN_SIZE // GROUP_SIZE
    ):
        mask = tl.full((BLOCK_ROWS, BLOCK_SIZE), True, tl.int1)
    else:
        mask = (group[:, None] < NUM_GROUPS) & (
            group_cols[None, :] < HIDDEN_SIZE // GROUP_SIZE
        )
    gate = tl.load(
        x_ptr + base[:, None] + col * STRIDE_X_COL,
        mask,
        0,
        cache_modifier=".cg",
    ).to(tl.float32)
    up = tl.load(
        x_ptr + base[:, None] + (col + HIDDEN_SIZE) * STRIDE_X_COL,
        mask,
        0,
        cache_modifier=".cg",
    ).to(tl.float32)
    value = gate * (1.0 / (1.0 + tl.exp(-gate))) * up
    maximum = tl.max(tl.abs(value), 1)
    maximum = tl.maximum(maximum, 1e-10)
    scale = maximum / 448.0
    inv_scale = 448.0 / maximum
    quantized = value * inv_scale[:, None]
    if NUM_GROUPS % BLOCK_ROWS == 0:
        valid = token < tl.load(masked_m_ptr + expert * STRIDE_MASKED_M)
    else:
        valid = token < tl.load(
            masked_m_ptr + expert * STRIDE_MASKED_M, group < NUM_GROUPS, 0
        )
    code = tl.where(
        valid[:, None],
        _silu_and_mul_masked_post_quant_encode_fp8(quantized),
        0,
    ).to(tl.uint8)
    tl.store(out_ptr + row[:, None] * HIDDEN_SIZE + col, code, mask)
    if NUM_GROUPS % BLOCK_ROWS == 0:
        tl.store(scale_ptr + group, tl.where(valid, scale, 0.0))
    else:
        tl.store(
            scale_ptr + group, tl.where(valid, scale, 0.0), group < NUM_GROUPS
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
        (experts, tokens, hidden),
        dtype=torch.float8_e4m3fn,
        device=input.device,
    )
    scales = torch.empty(
        (experts, tokens, groups), dtype=torch.float32, device=input.device
    )
    count = experts * tokens * groups
    if count:
        rows = 4
        _silu_and_mul_masked_post_quant_kernel[triton.cdiv(count, rows),](
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
            num_warps=2,
            enable_fp_fusion=False,
        )
    return (quantized, scales)


__all__ = ["silu_and_mul_masked_post_quant"]
