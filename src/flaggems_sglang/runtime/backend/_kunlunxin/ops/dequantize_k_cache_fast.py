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

"""dequantize_k_cache_fast implementation for kunlunxin."""

import torch
import triton
import triton.language as tl


@triton.jit
def _dequantize_k_cache_fast_decode_e4m3(raw_byte):
    raw_bits = raw_byte.to(tl.uint16)
    magnitude = raw_bits & 127
    bits = magnitude << 7 | (raw_bits & 128) << 8
    value = bits.to(tl.float16, bitcast=True).to(tl.float32) * 256.0
    return tl.where(magnitude == 127, float("nan"), value)


@triton.jit
def _dequantize_k_cache_fast_rows_kernel(
    cache_ptr,
    out_ptr,
    NUM_ROWS: tl.constexpr,
    STRIDE_CACHE: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
):
    program_id = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, 512 * BLOCK_ROWS)
    row = program_id * BLOCK_ROWS + cols // 512
    col = cols % 512
    rope_cols = tl.arange(0, 64 * BLOCK_ROWS)
    rope_row = program_id * BLOCK_ROWS + rope_cols // 64
    if NUM_ROWS % BLOCK_ROWS:
        input_row = row
        rope_input_row = rope_row
        valid = program_id * (512 * BLOCK_ROWS) + cols < NUM_ROWS * 512
        rope_valid = program_id * (64 * BLOCK_ROWS) + rope_cols < NUM_ROWS * 64
    else:
        input_row = row
        rope_input_row = rope_row
        valid = tl.full((512 * BLOCK_ROWS,), True, tl.int1)
        rope_valid = tl.full((64 * BLOCK_ROWS,), True, tl.int1)
    base = cache_ptr + input_row * STRIDE_CACHE
    value = _dequantize_k_cache_fast_decode_e4m3(tl.load(base + col, valid))
    scale = tl.load(
        (base + 512 + col // 128 * 4).to(tl.pointer_type(tl.float32)), valid
    )
    tl.store(out_ptr + row * 576 + col, value * scale, valid)
    rope = tl.load(
        (cache_ptr + rope_input_row * STRIDE_CACHE + 528).to(
            tl.pointer_type(tl.uint16)
        )
        + rope_cols % 64,
        rope_valid,
    )
    tl.store(
        (out_ptr + rope_row * 576 + 512 + rope_cols % 64).to(
            tl.pointer_type(tl.uint16)
        ),
        rope,
        rope_valid,
    )


def dequantize_k_cache_fast(quant_k_cache, group_size=128):
    num_rows = quant_k_cache.shape[0]
    out = torch.empty(
        (num_rows, 576), dtype=torch.bfloat16, device=quant_k_cache.device
    )
    if num_rows:
        _dequantize_k_cache_fast_rows_kernel[(triton.cdiv(num_rows, 8),)](
            quant_k_cache.view(torch.uint8),
            out,
            num_rows,
            quant_k_cache.stride(0),
            8,
            num_warps=4,
        )
    return out


__all__ = ["dequantize_k_cache_fast"]
