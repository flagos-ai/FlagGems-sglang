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
    x_ptr = x_ptr.to(tl.pointer_type(tl.int16))
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
    gate_bits = tl.load(
        x_ptr + base[:, None] + col * STRIDE_X_COL, mask, 0
    ).to(tl.int32)
    up_bits = tl.load(
        x_ptr + base[:, None] + (col + HIDDEN_SIZE) * STRIDE_X_COL, mask, 0
    ).to(tl.int32)
    gate = (gate_bits << 16).to(tl.float32, bitcast=True)
    up = (up_bits << 16).to(tl.float32, bitcast=True)
    value = gate / (1.0 + tl.exp(-gate)) * up
    maximum = tl.max(tl.abs(value), 1)
    scale = tl.div_rn(tl.maximum(maximum, 1e-10), 448.0)
    quantized = tl.minimum(
        tl.maximum(tl.div_rn(value, scale[:, None]), -448.0), 448.0
    )
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


@triton.jit
def _silu_and_mul_masked_post_quant_packed_groups_kernel(
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
    lane = tl.arange(0, BLOCK_ROWS)
    if BLOCK_ROWS > 64:
        group = (
            tl.program_id(0) * BLOCK_ROWS
            + lane % 64 * (BLOCK_ROWS // 64)
            + lane // 64
        )
    else:
        group = tl.program_id(0) * BLOCK_ROWS + lane
    row = group // GROUP_SIZE
    expert = row // NUM_TOKENS
    token = row % NUM_TOKENS
    group_cols = tl.arange(0, BLOCK_SIZE)
    col = (
        group[:, None] % GROUP_SIZE * (HIDDEN_SIZE // GROUP_SIZE // 2)
        + group_cols[None, :]
    )
    if STRIDE_X_EXPERT == NUM_TOKENS * STRIDE_X_TOKEN:
        base = row * STRIDE_X_TOKEN
    else:
        base = expert * STRIDE_X_EXPERT + token * STRIDE_X_TOKEN
    if (
        NUM_GROUPS % BLOCK_ROWS == 0
        and BLOCK_SIZE == HIDDEN_SIZE // GROUP_SIZE // 2
    ):
        mask = tl.full((BLOCK_ROWS, BLOCK_SIZE), True, tl.int1)
    else:
        mask = (group[:, None] < NUM_GROUPS) & (
            group_cols[None, :] < HIDDEN_SIZE // GROUP_SIZE // 2
        )
    gate_bits = tl.load(x_ptr + base[:, None] + col, mask, 0).to(tl.uint32)
    up_bits = tl.load(
        x_ptr + base[:, None] + col + HIDDEN_SIZE // 2, mask, 0
    ).to(tl.uint32)
    gate_0 = (gate_bits << 16).to(tl.float32, bitcast=True)
    gate_1 = (gate_bits & 4294901760).to(tl.float32, bitcast=True)
    up_0 = (up_bits << 16).to(tl.float32, bitcast=True)
    up_1 = (up_bits & 4294901760).to(tl.float32, bitcast=True)
    value_0 = gate_0 / (1.0 + tl.exp(-gate_0)) * up_0
    value_1 = gate_1 / (1.0 + tl.exp(-gate_1)) * up_1
    maximum = tl.max(tl.maximum(tl.abs(value_0), tl.abs(value_1)), 1)
    scale = tl.div_rn(tl.maximum(maximum, 1e-10), 448.0)
    quant_0 = tl.minimum(
        tl.maximum(tl.div_rn(value_0, scale[:, None]), -448.0), 448.0
    )
    quant_1 = tl.minimum(
        tl.maximum(tl.div_rn(value_1, scale[:, None]), -448.0), 448.0
    )
    code_0 = _silu_and_mul_masked_post_quant_encode_fp8(quant_0).to(tl.uint32)
    code_1 = _silu_and_mul_masked_post_quant_encode_fp8(quant_1).to(tl.uint32)
    if NUM_GROUPS % BLOCK_ROWS == 0:
        valid = token < tl.load(masked_m_ptr + expert * STRIDE_MASKED_M)
    else:
        valid = token < tl.load(
            masked_m_ptr + expert * STRIDE_MASKED_M, group < NUM_GROUPS, 0
        )
    packed = tl.where(valid[:, None], code_0 | code_1 << 8, 0).to(tl.int16)
    if HIDDEN_SIZE % GROUP_SIZE == 0:
        tl.store(
            out_ptr
            + group[:, None] * (HIDDEN_SIZE // GROUP_SIZE // 2)
            + group_cols[None, :],
            packed,
            mask,
        )
    else:
        tl.store(
            out_ptr + row[:, None] * (HIDDEN_SIZE // 2) + col, packed, mask
        )
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
        (experts, tokens, hidden), dtype=torch.uint8, device=input.device
    )
    scales = torch.empty(
        (experts, tokens, groups), dtype=torch.float32, device=input.device
    )
    count = experts * tokens * groups
    if count:
        block_size = triton.next_power_of_2(quant_group_size)
        rows = min(64, max(1, 8192 // block_size))
        if (
            input.stride(-1) == 1
            and hidden % 2 == 0
            and (quant_group_size % 2 == 0)
            and (input.storage_offset() % 2 == 0)
            and (input.stride(0) % 2 == 0)
            and (input.stride(1) % 2 == 0)
        ):
            rows = 256 if block_size <= 128 and count % 256 == 0 else rows
            words = input.view(torch.int32)
            output_words = quantized.view(torch.int16)
            _silu_and_mul_masked_post_quant_packed_groups_kernel[
                triton.cdiv(count, rows),
            ](
                words,
                masked_m,
                output_words,
                scales,
                tokens,
                hidden,
                groups,
                words.stride(0),
                words.stride(1),
                masked_m.stride(0),
                triton.next_power_of_2(quant_group_size // 2),
                count,
                rows,
                num_warps=4,
                enable_fp_fusion=False,
                isCloseMemoryAsync=False,
                unroll_num=4,
            )
        else:
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
                block_size,
                count,
                rows,
                num_warps=4,
                enable_fp_fusion=False,
                isCloseMemoryAsync=False,
                unroll_num=4,
            )
    return (quantized.view(torch.float8_e4m3fn), scales)


__all__ = ["silu_and_mul_masked_post_quant"]
