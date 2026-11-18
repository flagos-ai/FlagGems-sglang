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

"""fp8_per_token_to_per_tensor_quant implementation for kunlunxin."""

import torch
import triton
import triton.language as tl


def _fp8_per_token_to_per_tensor_quant_requires_wide_indices(
    x,
    x_scale,
    masked_m,
):
    limit = 2147483647
    if x.numel() > limit:
        return True
    for tensor in (x, x_scale, masked_m):
        strides = tensor.stride()
        if max(strides) > limit:
            return True
        if tensor.is_contiguous():
            continue
        if (
            sum(
                (
                    (size - 1) * stride
                    for (size, stride) in zip(tensor.shape, strides)
                )
            )
            > limit
        ):
            return True
    return False


@triton.jit
def _fp8_per_token_to_per_tensor_quant_wide_index(index, WIDE: tl.constexpr):
    if WIDE:
        return index.to(tl.int64)
    else:
        return index


@triton.jit
def _fp8_per_token_to_per_tensor_quant_encode_magnitude(
    value, DEFERRED: tl.constexpr = False
):
    raw = value.to(tl.uint32, bitcast=True)
    magnitude = raw
    magnitude = tl.minimum(magnitude, 1139802112)
    exponent = tl.maximum(magnitude & 2139095040, 1015021568)
    magic_bits = exponent + 167772160
    magic = magic_bits.to(tl.float32, bitcast=True)
    absolute = magnitude.to(tl.float32, bitcast=True)
    rounded = (absolute + magic).to(tl.uint32, bitcast=True) - magic_bits
    code = rounded + (exponent >> 20) - (0 if DEFERRED else 968)
    return (code & 127).to(tl.uint8)


@triton.jit
def _fp8_per_token_to_per_tensor_quant_decode(raw):
    raw_bits = raw.to(tl.uint32)
    magnitude = raw_bits & 127
    normal = ((magnitude << 20) + (120 << 23)).to(tl.float32, bitcast=True)
    value = tl.where(magnitude < 8, tl.fma(normal, 2.0, -0.015625), normal)
    return value


@triton.jit
def _fp8_per_token_to_per_tensor_quant_persistent_kernel(
    x_ptr,
    x_scale_ptr,
    masked_m_ptr,
    out_scale_ptr,
    out_ptr,
    METADATA: tl.constexpr,
    WIDE: tl.constexpr = False,
):
    NUM_TOKENS: tl.constexpr = METADATA[0]
    CHANNELS: tl.constexpr = METADATA[1]
    STRIDE_X_0: tl.constexpr = METADATA[2]
    STRIDE_X_1: tl.constexpr = METADATA[3]
    STRIDE_X_2: tl.constexpr = METADATA[4]
    STRIDE_SCALE_0: tl.constexpr = METADATA[5]
    STRIDE_SCALE_1: tl.constexpr = METADATA[6]
    STRIDE_SCALE_2: tl.constexpr = METADATA[7]
    STRIDE_MASKED_M: tl.constexpr = METADATA[8]
    BLOCK_SIZE: tl.constexpr = METADATA[9]
    NUM_PROGRAMS: tl.constexpr = METADATA[10]
    tiles: tl.constexpr = triton.cdiv(CHANNELS, BLOCK_SIZE)
    program_id = _fp8_per_token_to_per_tensor_quant_wide_index(
        tl.program_id(0), WIDE
    )
    batch_index = (program_id // (NUM_PROGRAMS * tiles)).to(tl.int64)
    start = program_id // tiles % NUM_PROGRAMS
    cols = (program_id % tiles).to(tl.int64) * BLOCK_SIZE + tl.arange(
        0, BLOCK_SIZE
    ).to(tl.int64)
    count = tl.minimum(
        tl.load(masked_m_ptr + batch_index * STRIDE_MASKED_M), NUM_TOKENS
    )
    inv_scale = tl.div_rn(1.0, tl.load(out_scale_ptr).to(tl.float32))
    magnitude_inv = tl.abs(inv_scale)
    for token in range(start, count, NUM_PROGRAMS):
        raw = tl.load(
            x_ptr
            + batch_index * STRIDE_X_0
            + token * STRIDE_X_1
            + cols * STRIDE_X_2,
            cols < CHANNELS,
            other=0,
        )
        scale = tl.load(
            x_scale_ptr
            + batch_index * STRIDE_SCALE_0
            + token * STRIDE_SCALE_1
            + cols // 128 * STRIDE_SCALE_2,
            cols < CHANNELS,
            other=0,
        )
        magnitude_scale = tl.abs(scale)
        value = (
            _fp8_per_token_to_per_tensor_quant_decode(raw) * magnitude_scale
        )
        value = value * magnitude_inv
        sign = (
            raw.to(tl.uint32) << 24
            ^ scale.to(tl.uint32, bitcast=True)
            ^ inv_scale.to(tl.uint32, bitcast=True)
        ) >> 24 & 128
        encoded = (
            _fp8_per_token_to_per_tensor_quant_encode_magnitude(value).to(
                tl.uint32
            )
            | sign
        ).to(tl.uint8)
        encoded = (
            encoded.to(tl.uint32)
            | tl.where(raw.to(tl.uint32) & 127 == 127, 127, 0)
        ).to(tl.uint8)
        tl.store(
            out_ptr + (batch_index * NUM_TOKENS + token) * CHANNELS + cols,
            encoded,
            cols < CHANNELS,
        )


def _fp8_per_token_to_per_tensor_quant_baseline(
    x,
    x_scale,
    masked_m,
    output_scale,
):
    (batch_index, num_tokens, cols) = x.shape
    out = torch.empty(
        (batch_index, num_tokens, cols), dtype=torch.uint8, device=x.device
    )
    if batch_index * num_tokens * cols:
        num_programs = min(num_tokens, 32)
        _fp8_per_token_to_per_tensor_quant_persistent_kernel[
            (batch_index * num_programs * triton.cdiv(cols, 4096),)
        ](
            x.view(torch.uint8),
            x_scale,
            masked_m,
            output_scale,
            out,
            (
                num_tokens,
                cols,
                *x.stride(),
                *x_scale.stride(),
                masked_m.stride(0),
                4096,
                num_programs,
            ),
            num_warps=4,
            enable_fp_fusion=False,
            WIDE=_fp8_per_token_to_per_tensor_quant_requires_wide_indices(
                x, x_scale, masked_m
            ),
        )
    return out.view(torch.float8_e4m3fn)


@triton.jit
def _fp8_per_token_to_per_tensor_quant_rows_packed_kernel(
    x_ptr,
    x_scale_ptr,
    masked_m_ptr,
    out_scale_ptr,
    out_ptr,
    METADATA: tl.constexpr,
    WIDE: tl.constexpr = False,
):
    NUM_TOKENS: tl.constexpr = METADATA[0]
    CHANNELS: tl.constexpr = METADATA[1]
    STRIDE_MASKED_M: tl.constexpr = METADATA[2]
    BLOCK_SIZE: tl.constexpr = METADATA[3]
    BLOCK_ROWS: tl.constexpr = METADATA[4]
    batch_index = _fp8_per_token_to_per_tensor_quant_wide_index(
        tl.program_id(1), WIDE
    ).to(tl.int64)
    rows = _fp8_per_token_to_per_tensor_quant_wide_index(
        tl.program_id(0), WIDE
    ) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    words = _fp8_per_token_to_per_tensor_quant_wide_index(
        tl.program_id(2), WIDE
    ) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offsets = (batch_index * NUM_TOKENS + rows[:, None]) * (
        CHANNELS // 4
    ) + words[None, :]
    raw = tl.load(x_ptr + offsets, offset_state_policy="continuous").to(
        tl.uint32
    )
    scale = tl.load(
        x_scale_ptr
        + (batch_index * NUM_TOKENS + rows[:, None]) * (CHANNELS // 128)
        + words[None, :] // 32
    )
    magnitude_scale = tl.abs(scale)
    inv_scale = tl.div_rn(1.0, tl.load(out_scale_ptr).to(tl.float32))
    magnitude_inv = tl.abs(inv_scale)
    count = tl.load(masked_m_ptr + batch_index * STRIDE_MASKED_M)
    byte_0 = _fp8_per_token_to_per_tensor_quant_decode(raw) * magnitude_scale
    byte_1 = (
        _fp8_per_token_to_per_tensor_quant_decode(raw >> 8) * magnitude_scale
    )
    byte_2 = (
        _fp8_per_token_to_per_tensor_quant_decode(raw >> 16) * magnitude_scale
    )
    byte_3 = (
        _fp8_per_token_to_per_tensor_quant_decode(raw >> 24) * magnitude_scale
    )
    byte_0 = byte_0 * magnitude_inv
    byte_1 = byte_1 * magnitude_inv
    byte_2 = byte_2 * magnitude_inv
    byte_3 = byte_3 * magnitude_inv
    code = _fp8_per_token_to_per_tensor_quant_encode_magnitude(
        byte_0, True
    ).to(tl.uint32)
    code = (
        code
        | _fp8_per_token_to_per_tensor_quant_encode_magnitude(byte_1, True).to(
            tl.uint32
        )
        << 8
    )
    code = (
        code
        | _fp8_per_token_to_per_tensor_quant_encode_magnitude(byte_2, True).to(
            tl.uint32
        )
        << 16
    )
    code = (
        code
        | _fp8_per_token_to_per_tensor_quant_encode_magnitude(byte_3, True).to(
            tl.uint32
        )
        << 24
    )
    code = code + 943208504 & 2139062143
    factor_sign = (
        (
            scale.to(tl.int32, bitcast=True)
            ^ inv_scale.to(tl.int32, bitcast=True)
        )
        >> 31
    ).to(tl.uint32)
    code = code | (raw ^ factor_sign) & 2155905152
    nan_bits = (raw & 2139062143) + 16843009 & 2155905152
    code = code | nan_bits - (nan_bits >> 7)
    tl.store(
        out_ptr + offsets,
        tl.where(rows[:, None] < count, code, 0),
        offset_state_policy="continuous",
    )


def fp8_per_token_to_per_tensor_quant(
    x,
    x_scale,
    masked_m,
    output_scale,
):
    (batch_index, num_tokens, cols) = x.shape
    if (
        batch_index > 0
        and batch_index <= 65535
        and (num_tokens >= 16)
        and (num_tokens % 16 == 0)
        and (1024 <= cols <= 8192)
        and (cols & cols - 1 == 0)
        and x.is_contiguous()
        and x_scale.is_contiguous()
        and ((x.data_ptr() | x_scale.data_ptr()) & 63 == 0)
    ):
        byte_1 = min(512, cols // 4)
        block_rows = min(32, num_tokens & -num_tokens)
        out = torch.empty(
            (batch_index, num_tokens, cols // 4),
            dtype=torch.int32,
            device=x.device,
        )
        _fp8_per_token_to_per_tensor_quant_rows_packed_kernel[
            (num_tokens // block_rows, batch_index, cols // (4 * byte_1))
        ](
            x.view(torch.int32),
            x_scale,
            masked_m,
            output_scale,
            out,
            (num_tokens, cols, masked_m.stride(0), byte_1, block_rows),
            num_warps=1,
            groups_per_cluster=block_rows,
            enable_fp_fusion=False,
            WIDE=_fp8_per_token_to_per_tensor_quant_requires_wide_indices(
                x, x_scale, masked_m
            ),
            isCloseMemoryAsync=False,
        )
        return out.view(torch.float8_e4m3fn)
    return _fp8_per_token_to_per_tensor_quant_baseline(
        x, x_scale, masked_m, output_scale
    )


__all__ = ["fp8_per_token_to_per_tensor_quant"]
