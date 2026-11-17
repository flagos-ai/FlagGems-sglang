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

"""per_tensor_quant_mla_fp8 implementation for kunlunxin."""

import torch
import triton
import triton.language as tl


@triton.jit
def _per_tensor_quant_mla_fp8_epsilon_scale_bits(maximum, EPS: tl.constexpr):
    if EPS != EPS:
        return _per_tensor_quant_mla_fp8_make_scale_bits(maximum, 0.0)
    return _per_tensor_quant_mla_fp8_make_scale_bits(maximum, EPS)


@triton.jit
def _per_tensor_quant_mla_fp8_make_scale_bits(maximum, EPS: tl.constexpr):
    maximum_mantissa = maximum.to(tl.uint32, bitcast=True)
    epsilon = tl.full((), EPS, tl.float32)
    epsilon_bits = epsilon.to(tl.uint32, bitcast=True)
    normal = tl.div_rn(tl.maximum(maximum, epsilon), 448.0)
    nan = (maximum_mantissa > 2139095040) | (
        epsilon_bits & 2147483647 > 2139095040
    )
    result = tl.where(
        nan,
        tl.full((), 2143289344, tl.uint32),
        normal.to(tl.uint32, bitcast=True),
    )
    if EPS < 1e-30:
        positive_epsilon_bits = tl.where(
            epsilon_bits & 2147483648 == 0, epsilon_bits, 0
        )
        maximum_bits = tl.maximum(maximum_mantissa, positive_epsilon_bits)
        exponent = maximum_bits >> 23
        mantissa = maximum_bits & 8388607 | tl.where(exponent > 0, 8388608, 0)
        shift = tl.minimum(tl.maximum(exponent.to(tl.int32) - 1, 0), 8)
        numerator = mantissa << shift
        quotient = numerator // 448
        remainder = numerator - quotient * 448
        rounded = quotient + (
            (remainder > 224) | (remainder == 224) & (quotient & 1 != 0)
        ).to(tl.uint32)
        result = tl.where(maximum_bits < 81788928, rounded, result)
    return result


@triton.jit
def _per_tensor_quant_mla_fp8_quant_code(
    value,
    scale_bits,
    FAST: tl.constexpr,
    SMALL: tl.constexpr,
):
    scale = scale_bits.to(tl.float32, bitcast=True)
    if SMALL:
        raw = value.to(tl.uint32, bitcast=True)
        magnitude = raw & 2147483647
        exponent = magnitude >> 23
        subnormal_scaled = (magnitude & 8388607).to(
            tl.float32
        ) * 2.350988701644575e-38
        normal_scaled_bits = magnitude + 201326592
        scaled_bits = (
            tl.where(
                exponent == 0,
                subnormal_scaled.to(tl.uint32, bitcast=True),
                normal_scaled_bits,
            )
            | raw & 2147483648
        )
        scale_scaled = scale_bits.to(tl.float32) * 2.350988701644575e-38
        numerator_bits = tl.where(scale_bits < 8388608, scaled_bits, raw)
        denominator_bits = tl.where(
            scale_bits < 8388608,
            scale_scaled.to(tl.uint32, bitcast=True),
            scale_bits,
        )
        quantized = tl.div_rn(
            numerator_bits.to(tl.float32, bitcast=True),
            denominator_bits.to(tl.float32, bitcast=True),
        )
    else:
        quantized = tl.div_rn(value, scale)
    if not FAST:
        quantized = _per_tensor_quant_mla_fp8_clamp(quantized)
    code = _per_tensor_quant_mla_fp8_encode(quantized, FAST)
    if FAST:
        nan = scale_bits > 2139095040
    else:
        magnitude = value.to(tl.uint32, bitcast=True) & 2147483647
        nan = (
            (scale_bits > 2139095040)
            | (magnitude >= 2139095040)
            | (magnitude == 0) & (scale_bits == 0)
        )
    return tl.where(nan, 127, code).to(tl.uint8)


@triton.jit
def _per_tensor_quant_mla_fp8_max_nan(values):
    return tl.max(values.to(tl.uint32, bitcast=True), 0).to(
        tl.float32, bitcast=True
    )


@triton.jit
def _per_tensor_quant_mla_fp8_clamp(value):
    return tl.minimum(tl.maximum(value, -448.0), 448.0)


@triton.jit
def _per_tensor_quant_mla_fp8_encode(value, CHECK_NAN: tl.constexpr):
    raw = value.to(tl.uint32, bitcast=True)
    magnitude = raw & 2147483647
    if CHECK_NAN:
        exponent = tl.maximum(magnitude >> 23, 121)
        magic_bits = exponent * 8388616 + 167771192 | raw >> 24 & 128
        rounded = (
            magnitude.to(tl.float32, bitcast=True)
            + magic_bits.to(tl.float32, bitcast=True)
        ).to(tl.uint32, bitcast=True)
        code = rounded.to(tl.uint8)
        return tl.where(magnitude > 2139095040, 127, code).to(tl.uint8)
    else:
        exponent = tl.maximum(magnitude & 2139095040, 1015021568)
        magic = (exponent + 167772160).to(tl.float32, bitcast=True)
        absolute = magnitude.to(tl.float32, bitcast=True)
        rounded = (absolute + magic).to(tl.uint32, bitcast=True)
        code = rounded + (exponent >> 20) - 968
        code = (code | raw >> 24 & 128).to(tl.uint8)
        return code


@triton.jit
def _per_tensor_quant_mla_fp8_linear_index(
    block,
    BLOCK_SIZE: tl.constexpr,
    NUM_ELEMENTS: tl.constexpr,
):
    if NUM_ELEMENTS >= 2147483648:
        block = block.to(tl.int64)
    return block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)


@triton.jit
def _per_tensor_quant_mla_fp8_offset(index, METADATA: tl.constexpr):
    NUM_TOKENS: tl.constexpr = METADATA[1]
    HEAD_DIM: tl.constexpr = METADATA[2]
    STRIDE_SCALE_0: tl.constexpr = METADATA[3]
    STRIDE_SCALE_1: tl.constexpr = METADATA[4]
    STRIDE_SCALE_2: tl.constexpr = METADATA[5]
    if (
        STRIDE_SCALE_2 == 1
        and STRIDE_SCALE_1 == HEAD_DIM
        and (STRIDE_SCALE_0 == NUM_TOKENS * HEAD_DIM)
    ):
        return index
    NUM_HEADS: tl.constexpr = METADATA[0] // (NUM_TOKENS * HEAD_DIM)
    if (
        STRIDE_SCALE_0 >= 2147483648
        or STRIDE_SCALE_1 >= 2147483648
        or STRIDE_SCALE_2 >= 2147483648
        or (
            (NUM_HEADS - 1) * STRIDE_SCALE_0
            + (NUM_TOKENS - 1) * STRIDE_SCALE_1
            + (HEAD_DIM - 1) * STRIDE_SCALE_2
            >= 2147483648
        )
    ):
        index = index.to(tl.int64)
    return (
        index // (NUM_TOKENS * HEAD_DIM) * STRIDE_SCALE_0
        + index // HEAD_DIM % NUM_TOKENS * STRIDE_SCALE_1
        + index % HEAD_DIM * STRIDE_SCALE_2
    )


@triton.jit
def _per_tensor_quant_mla_fp8_load_value(
    x_ptr,
    index,
    METADATA: tl.constexpr,
    FULL: tl.constexpr,
):
    NUM_ELEMENTS: tl.constexpr = METADATA[0]
    if FULL:
        offsets = _per_tensor_quant_mla_fp8_offset(index, METADATA)
    else:
        offsets = _per_tensor_quant_mla_fp8_offset(
            tl.minimum(index, NUM_ELEMENTS - 1), METADATA
        )
    if x_ptr.dtype.element_ty == tl.float16:
        raw = tl.load(x_ptr.to(tl.pointer_type(tl.uint16)) + offsets).to(
            tl.uint32
        )
        magnitude = raw & 32767
        exponent = magnitude >> 10
        mantissa = magnitude & 1023
        normal = ((magnitude << 13) + 939524096).to(tl.float32, bitcast=True)
        subnormal = mantissa.to(tl.float32) * 5.960464477539063e-08
        special = (magnitude << 13 | 2139095040).to(tl.float32, bitcast=True)
        absolute = tl.where(
            exponent == 0, subnormal, tl.where(exponent == 31, special, normal)
        )
        bits = absolute.to(tl.uint32, bitcast=True) | (raw & 32768) << 16
        value = bits.to(tl.float32, bitcast=True)
    else:
        value = tl.load(x_ptr + offsets).to(tl.float32)
    if FULL:
        return value
    return tl.where(index < NUM_ELEMENTS, value, 0.0)


@triton.jit
def _per_tensor_quant_mla_fp8_single_kernel(
    x_ptr,
    out_ptr,
    out_scale_ptr,
    METADATA: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    NUM_ELEMENTS: tl.constexpr = METADATA[0]
    index = tl.arange(0, BLOCK_SIZE)
    if (
        x_ptr.dtype.element_ty == tl.bfloat16
        and NUM_ELEMENTS > 2048
        and (NUM_ELEMENTS % BLOCK_SIZE != 0)
    ):
        offset = _per_tensor_quant_mla_fp8_offset(
            tl.minimum(index, NUM_ELEMENTS - 1), METADATA
        )
        raw = tl.load(x_ptr.to(tl.pointer_type(tl.uint16)) + offset).to(
            tl.uint32
        )
        value = (raw << 16).to(tl.float32, bitcast=True)
        value = tl.where(index < NUM_ELEMENTS, value, 0.0)
    else:
        value = _per_tensor_quant_mla_fp8_load_value(
            x_ptr, index, METADATA, NUM_ELEMENTS % BLOCK_SIZE == 0
        )
    scale_bits = _per_tensor_quant_mla_fp8_epsilon_scale_bits(
        _per_tensor_quant_mla_fp8_max_nan(tl.abs(value)), EPS
    )
    code = _per_tensor_quant_mla_fp8_quant_code(
        value,
        scale_bits,
        x_ptr.dtype.element_ty == tl.bfloat16
        and EPS >= 1e-30
        and (EPS < float("inf")),
        EPS < 1e-30 or EPS != EPS,
    )
    tl.store(out_ptr + index, code, index < NUM_ELEMENTS)
    tl.store(out_scale_ptr.to(tl.pointer_type(tl.uint32)), scale_bits)


@triton.jit
def _per_tensor_quant_mla_fp8_partial_kernel(
    x_ptr,
    partial_ptr,
    METADATA: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    NUM_ELEMENTS: tl.constexpr = METADATA[0]
    index = _per_tensor_quant_mla_fp8_linear_index(
        tl.program_id(0), BLOCK_SIZE, NUM_ELEMENTS
    )
    if x_ptr.dtype.element_ty == tl.bfloat16:
        if NUM_ELEMENTS % BLOCK_SIZE == 0:
            offsets = _per_tensor_quant_mla_fp8_offset(index, METADATA)
        else:
            offsets = _per_tensor_quant_mla_fp8_offset(
                tl.minimum(index, NUM_ELEMENTS - 1), METADATA
            )
        raw = tl.load(x_ptr.to(tl.pointer_type(tl.uint16)) + offsets).to(
            tl.uint32
        )
        maximum_bits = tl.max(raw & 32767, 0) << 16
    else:
        value = _per_tensor_quant_mla_fp8_load_value(
            x_ptr, index, METADATA, NUM_ELEMENTS % BLOCK_SIZE == 0
        )
        maximum_bits = tl.max(tl.abs(value).to(tl.uint32, bitcast=True), 0)
    tl.store(
        partial_ptr.to(tl.pointer_type(tl.uint32)) + tl.program_id(0),
        maximum_bits,
    )


@triton.jit
def _per_tensor_quant_mla_fp8_scale_kernel(
    partial_ptr,
    out_scale_ptr,
    NUM_PARTIALS: tl.constexpr,
    BLOCK_PARTIAL: tl.constexpr,
    EPS: tl.constexpr,
):
    partial_offsets = tl.arange(0, BLOCK_PARTIAL)
    partial_max = tl.load(
        partial_ptr + partial_offsets, partial_offsets < NUM_PARTIALS, other=0
    )
    scale_bits = _per_tensor_quant_mla_fp8_epsilon_scale_bits(
        _per_tensor_quant_mla_fp8_max_nan(partial_max), EPS
    )
    tl.store(out_scale_ptr.to(tl.pointer_type(tl.uint32)), scale_bits)


@triton.jit
def _per_tensor_quant_mla_fp8_quant_kernel(
    x_ptr,
    partial_ptr,
    out_ptr,
    out_scale_ptr,
    METADATA: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    NUM_PARTIALS: tl.constexpr,
    BLOCK_PARTIAL: tl.constexpr,
    REDUCE: tl.constexpr,
):
    NUM_ELEMENTS: tl.constexpr = METADATA[0]
    if REDUCE:
        partial_offsets = tl.arange(0, BLOCK_PARTIAL)
        partial_max = tl.load(
            partial_ptr + partial_offsets,
            partial_offsets < NUM_PARTIALS,
            other=0,
        )
        scale_bits = _per_tensor_quant_mla_fp8_epsilon_scale_bits(
            _per_tensor_quant_mla_fp8_max_nan(partial_max), EPS
        )
        if tl.program_id(0) == 0:
            tl.store(out_scale_ptr.to(tl.pointer_type(tl.uint32)), scale_bits)
    else:
        scale_bits = tl.load(out_scale_ptr.to(tl.pointer_type(tl.uint32)))
    index = _per_tensor_quant_mla_fp8_linear_index(
        tl.program_id(0), BLOCK_SIZE, NUM_ELEMENTS
    )
    value = _per_tensor_quant_mla_fp8_load_value(
        x_ptr, index, METADATA, NUM_ELEMENTS % BLOCK_SIZE == 0
    )
    code = _per_tensor_quant_mla_fp8_quant_code(
        value,
        scale_bits,
        x_ptr.dtype.element_ty == tl.bfloat16
        and EPS >= 1e-30
        and (EPS < float("inf")),
        EPS < 1e-30 or EPS != EPS,
    )
    tl.store(out_ptr + index, code, index < NUM_ELEMENTS)


def per_tensor_quant_mla_fp8(x, eps=1e-12):
    (num_heads, num_tokens, head_dim) = x.shape
    num_elements = num_heads * num_tokens * head_dim
    out = torch.empty(
        (num_heads, num_tokens, head_dim), dtype=torch.uint8, device=x.device
    )
    scale = torch.empty((1,), dtype=torch.float32, device=x.device)
    metadata = (num_elements, num_tokens, head_dim, *x.stride())
    if num_elements <= 2048 or (
        x.dtype == torch.bfloat16
        and x.is_contiguous()
        and (
            num_elements <= 4096
            or (num_elements <= 8192 and num_elements & num_elements - 1 == 0)
        )
    ):
        _per_tensor_quant_mla_fp8_single_kernel[(1,)](
            x,
            out,
            scale,
            metadata,
            eps,
            1 << (num_elements - 1).bit_length(),
            num_warps=4,
            enable_fp_fusion=False,
        )
    else:
        quant_block = (
            16384
            if num_elements >= 2097152
            and num_elements % 16384 == 0
            and (x.dtype == torch.bfloat16)
            and x.is_contiguous()
            else (
                4096
                if num_elements >= 65536
                and num_elements % 4096 == 0
                and x.is_contiguous()
                else 1024
            )
        )
        partial_block = (
            16384
            if num_elements >= 65536
            and num_elements % 16384 == 0
            and (x.dtype == torch.bfloat16)
            and x.is_contiguous()
            else 4096
        )
        blocks = (num_elements + partial_block - 1) // partial_block
        partial = torch.empty((blocks,), dtype=torch.float32, device=x.device)
        block_partial = 1 << (blocks - 1).bit_length()
        _per_tensor_quant_mla_fp8_partial_kernel[(blocks,)](
            x,
            partial,
            metadata,
            partial_block,
            num_warps=4,
        )
        _per_tensor_quant_mla_fp8_scale_kernel[(1,)](
            partial,
            scale,
            blocks,
            block_partial,
            eps,
            num_warps=4,
            enable_fp_fusion=False,
        )
        if quant_block == 16384 and 1e-30 <= eps < float("inf"):
            _per_tensor_quant_mla_fp8_quant_kernel[
                ((num_elements + quant_block - 1) // quant_block,)
            ](
                x,
                partial,
                out,
                scale,
                metadata,
                eps,
                quant_block,
                blocks,
                block_partial,
                False,
                num_warps=4,
                enable_fp_fusion=False,
                isCloseMemoryAsync=False,
            )
        else:
            _per_tensor_quant_mla_fp8_quant_kernel[
                ((num_elements + quant_block - 1) // quant_block,)
            ](
                x,
                partial,
                out,
                scale,
                metadata,
                eps,
                quant_block,
                blocks,
                block_partial,
                False,
                num_warps=4,
                enable_fp_fusion=False,
            )
    return (out.view(torch.float8_e4m3fn), scale)


__all__ = ["per_tensor_quant_mla_fp8"]
