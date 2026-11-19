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
    sign = (raw >> 24 & 128).to(tl.int16)
    magnitude = tl.abs(value)
    exponent = tl.maximum(raw & 2139095040, 1015021568)
    bias = exponent + 167772160
    rounded = (magnitude + bias.to(tl.float32, bitcast=True)).to(
        tl.uint32, bitcast=True
    )
    code = rounded.to(tl.int16) + (bias >> 20).to(tl.int16) - 1128
    return (tl.where(value != value, 127, code) | sign).to(tl.uint8)


@triton.jit
def _silu_and_mul_masked_post_quant_quantize_groups(
    x_ptr,
    masked_m_ptr,
    out_ptr,
    scale_ptr,
    METADATA: tl.constexpr,
    ACTIVE: tl.constexpr,
):
    NUM_TOKENS: tl.constexpr = METADATA[0]
    HIDDEN_SIZE: tl.constexpr = METADATA[1]
    GROUP_SIZE: tl.constexpr = METADATA[2]
    STRIDE_X_EXPERT: tl.constexpr = METADATA[3]
    STRIDE_X_TOKEN: tl.constexpr = METADATA[4]
    STRIDE_X_COL: tl.constexpr = METADATA[5]
    STRIDE_MASKED_M: tl.constexpr = METADATA[6]
    BLOCK_SIZE: tl.constexpr = METADATA[7]
    NUM_GROUPS: tl.constexpr = METADATA[8]
    BLOCK_ROWS: tl.constexpr = METADATA[9]
    out_ptr = out_ptr.to(tl.pointer_type(tl.uint8))
    for tile in range(
        tl.program_id(0),
        NUM_GROUPS // GROUP_SIZE * tl.cdiv(GROUP_SIZE, BLOCK_ROWS),
        tl.num_programs(0),
    ):
        group = tile % tl.cdiv(
            GROUP_SIZE, BLOCK_ROWS
        ) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
        row = tile // tl.cdiv(GROUP_SIZE, BLOCK_ROWS)
        expert = row // NUM_TOKENS
        token = row % NUM_TOKENS
        group_cols = tl.arange(0, BLOCK_SIZE)
        col = (
            group[:, None] * (HIDDEN_SIZE // GROUP_SIZE) + group_cols[None, :]
        )
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
        gate = tl.load(x_ptr + base + col * STRIDE_X_COL, mask, 0).to(
            tl.float32
        )
        up = tl.load(
            x_ptr + base + (col + HIDDEN_SIZE) * STRIDE_X_COL, mask, 0
        ).to(tl.float32)
        value = gate * (1.0 / (1.0 + tl.exp(-gate))) * up
        maximum = tl.max(tl.abs(value), 1)
        scale = tl.maximum(maximum, 1e-10) / 448.0
        quantized = value / scale[:, None]
        if ACTIVE:
            valid = True
        else:
            valid = token < tl.load(masked_m_ptr + expert * STRIDE_MASKED_M)
        code = tl.where(
            valid, _silu_and_mul_masked_post_quant_encode_fp8(quantized), 0
        ).to(tl.uint8)
        tl.store(out_ptr + row * HIDDEN_SIZE + col, code, mask)
        if GROUP_SIZE % BLOCK_ROWS == 0:
            tl.store(
                scale_ptr + row * GROUP_SIZE + group,
                tl.where(valid, scale, 0.0),
            )
        else:
            tl.store(
                scale_ptr + row * GROUP_SIZE + group,
                tl.where(valid, scale, 0.0),
                group < GROUP_SIZE,
            )


@triton.jit
def _silu_and_mul_masked_post_quant_kernel(
    x_ptr,
    masked_m_ptr,
    out_ptr,
    scale_ptr,
    METADATA: tl.constexpr,
):
    NUM_EXPERTS: tl.constexpr = METADATA[8] // METADATA[2] // METADATA[0]
    if (
        NUM_EXPERTS <= 64
        and METADATA[8] // METADATA[2] * tl.cdiv(METADATA[2], METADATA[9])
        >= 4096
    ):
        indices = tl.arange(0, triton.next_power_of_2(NUM_EXPERTS))
        counts = tl.load(
            masked_m_ptr + indices * METADATA[6],
            indices < NUM_EXPERTS,
            METADATA[0],
        )
        active = tl.min((counts >= METADATA[0]).to(tl.int32), 0)
        if active:
            _silu_and_mul_masked_post_quant_quantize_groups(
                x_ptr, masked_m_ptr, out_ptr, scale_ptr, METADATA, True
            )
        else:
            _silu_and_mul_masked_post_quant_quantize_groups(
                x_ptr, masked_m_ptr, out_ptr, scale_ptr, METADATA, False
            )
    else:
        _silu_and_mul_masked_post_quant_quantize_groups(
            x_ptr, masked_m_ptr, out_ptr, scale_ptr, METADATA, False
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
        rows = min(
            32, max(1, 4096 // triton.next_power_of_2(quant_group_size))
        )
        _silu_and_mul_masked_post_quant_kernel.run(
            input,
            masked_m,
            quantized.view(torch.uint8),
            scales,
            (
                tokens,
                hidden,
                groups,
                *input.stride(),
                masked_m.stride(0),
                triton.next_power_of_2(quant_group_size),
                count,
                rows,
            ),
            num_warps=4,
            enable_fp_fusion=False,
            num_stages=1,
            grid=(min(experts * tokens * triton.cdiv(groups, rows), 40),),
            warmup=False,
        )
    return (quantized, scales)


__all__ = ["silu_and_mul_masked_post_quant"]
