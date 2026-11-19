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
def _fused_qk_complex_rope_strided_kernel(
    q_ptr,
    k_ptr,
    freqs_ptr,
    q_out_ptr,
    k_out_ptr,
    NUM_PAIRS: tl.constexpr,
    SHAPE: tl.constexpr,
    Q_STRIDES: tl.constexpr,
    K_STRIDES: tl.constexpr,
    FREQ_STRIDES: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    USE_INT64: tl.constexpr,
):
    program_id = tl.program_id(0)
    if USE_INT64:
        program_id = program_id.to(tl.int64)
    for block_index in range(
        program_id, triton.cdiv(NUM_PAIRS, BLOCK_SIZE), tl.num_programs(0)
    ):
        pair = block_index * (BLOCK_SIZE // 2) + tl.arange(0, BLOCK_SIZE // 2)
        dim_offsets = pair % (SHAPE[-1] // 2)
        head_index = pair // (SHAPE[-1] // 2) % SHAPE[-2]
        token_index = pair // (SHAPE[-1] // 2 * SHAPE[-2])
        q_out_offsets = (
            head_index * Q_STRIDES[-2] + 2 * dim_offsets * Q_STRIDES[-1]
        )
        k_out_offsets = (
            head_index * K_STRIDES[-2] + 2 * dim_offsets * K_STRIDES[-1]
        )
        freq_offset = dim_offsets * FREQ_STRIDES[-2]
        for axis in tl.static_range(len(SHAPE) - 3, -1, -1):
            coord = token_index % SHAPE[axis]
            token_index = token_index // SHAPE[axis]
            q_out_offsets += coord * Q_STRIDES[axis]
            k_out_offsets += coord * K_STRIDES[axis]
            freq_offset += coord * FREQ_STRIDES[axis]
        mask = pair < NUM_PAIRS // 2
        q_real = tl.load(q_ptr + q_out_offsets, mask, 0).to(tl.float32)
        q_imag = tl.load(q_ptr + q_out_offsets + Q_STRIDES[-1], mask, 0).to(
            tl.float32
        )
        k_real = tl.load(k_ptr + k_out_offsets, mask, 0).to(tl.float32)
        k_imag = tl.load(k_ptr + k_out_offsets + K_STRIDES[-1], mask, 0).to(
            tl.float32
        )
        freq_real = tl.load(freqs_ptr + freq_offset, mask, 0)
        freq_imag = tl.load(
            freqs_ptr + freq_offset + FREQ_STRIDES[-1], mask, 0
        )
        q_out_real = q_real * freq_real + (-1.0 * q_imag) * freq_imag
        q_out_imag = q_imag * freq_real + (1.0 * q_real) * freq_imag
        k_out_real = k_real * freq_real + (-1.0 * k_imag) * freq_imag
        k_out_imag = k_imag * freq_real + (1.0 * k_real) * freq_imag
        qout = tl.reshape(tl.join(q_out_real, q_out_imag), (BLOCK_SIZE,))
        kout = tl.reshape(tl.join(k_out_real, k_out_imag), (BLOCK_SIZE,))
        x = block_index * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        tl.store(q_out_ptr + x, qout, x < NUM_PAIRS)
        tl.store(k_out_ptr + x, kout, x < NUM_PAIRS)


def _fused_qk_complex_rope_compute_strided(
    q,
    k,
    freqs_cis,
):
    q_out = torch.empty(q.shape, dtype=q.dtype, device=q.device)
    k_out = torch.empty(k.shape, dtype=k.dtype, device=k.device)
    num_elements = q.numel()
    if num_elements:
        freqs_real = torch.view_as_real(freqs_cis)
        use_int64 = (
            max(
                num_elements + 1024,
                *(
                    sum(
                        (
                            (rotation_sign - 1) * stride
                            for rotation_sign, stride in zip(
                                x.shape, x.stride()
                            )
                        )
                    )
                    for x in (q, k, freqs_real)
                ),
            )
            > 2147483647
        )
        _fused_qk_complex_rope_strided_kernel[
            min((num_elements + 1024 - 1) // 1024, 32768),
        ](
            q,
            k,
            freqs_real,
            q_out,
            k_out,
            num_elements,
            tuple(q.shape),
            q.stride(),
            k.stride(),
            freqs_real.stride(),
            1024,
            use_int64,
            num_warps=4,
            enable_fp_fusion=False,
        )
    return (q_out, k_out)


@triton.jit
def _fused_qk_complex_rope_to_bf16_bits(value):
    return (
        value.to(tl.bfloat16).to(tl.int16, bitcast=True).to(tl.int32) & 65535
    )


@triton.jit
def _fused_qk_complex_rope_packed_tokens_kernel(
    q_ptr: tl.pointer_type(tl.uint32),
    k_ptr: tl.pointer_type(tl.uint32),
    freqs_ptr: tl.pointer_type(tl.float32),
    q_out_ptr: tl.pointer_type(tl.uint32),
    k_out_ptr: tl.pointer_type(tl.uint32),
    METADATA: tl.constexpr,
):
    NUM_TOKENS: tl.constexpr = METADATA[0]
    NUM_HEADS: tl.constexpr = METADATA[1]
    HEAD_DIM: tl.constexpr = METADATA[2]
    BLOCK_TOKENS: tl.constexpr = METADATA[3]
    NUM_PROGRAMS: tl.constexpr = METADATA[4]
    USE_INT64: tl.constexpr = METADATA[5]
    index = tl.arange(0, NUM_HEADS * HEAD_DIM // 2)
    dim_offsets = tl.arange(0, HEAD_DIM)
    program_id = tl.program_id(0)
    if USE_INT64:
        program_id = program_id.to(tl.int64)
    for block_index in range(
        program_id, tl.cdiv(NUM_TOKENS, BLOCK_TOKENS), NUM_PROGRAMS
    ):
        token = block_index * BLOCK_TOKENS + tl.arange(0, BLOCK_TOKENS)
        position = (
            token[:, None] * (NUM_HEADS * HEAD_DIM // 2) + index[None, :]
        )
        freq_ptr = freqs_ptr + token[:, None] * HEAD_DIM + dim_offsets[None, :]
        if NUM_TOKENS % BLOCK_TOKENS == 0:
            q_bits = tl.load(q_ptr + position)
            k_bits = tl.load(k_ptr + position)
            freq_values = tl.load(freq_ptr)
        else:
            q_bits = tl.load(q_ptr + position, token[:, None] < NUM_TOKENS, 0)
            k_bits = tl.load(k_ptr + position, token[:, None] < NUM_TOKENS, 0)
            freq_values = tl.load(freq_ptr, token[:, None] < NUM_TOKENS, 0)
        q_real = (q_bits << 16).to(tl.float32, bitcast=True)
        q_imag = (q_bits & 4294901760).to(tl.float32, bitcast=True)
        k_real = (k_bits << 16).to(tl.float32, bitcast=True)
        k_imag = (k_bits & 4294901760).to(tl.float32, bitcast=True)
        freq_real, freq_imag = tl.split(
            tl.reshape(freq_values, (BLOCK_TOKENS, HEAD_DIM // 2, 2))
        )
        freq_real = tl.reshape(
            tl.broadcast_to(
                freq_real[:, None, :], (BLOCK_TOKENS, NUM_HEADS, HEAD_DIM // 2)
            ),
            (BLOCK_TOKENS, NUM_HEADS * HEAD_DIM // 2),
        )
        freq_imag = tl.reshape(
            tl.broadcast_to(
                freq_imag[:, None, :], (BLOCK_TOKENS, NUM_HEADS, HEAD_DIM // 2)
            ),
            (BLOCK_TOKENS, NUM_HEADS * HEAD_DIM // 2),
        )
        q_out_real = q_real * freq_real - q_imag * freq_imag
        q_out_imag = q_imag * freq_real + q_real * freq_imag
        k_out_real = k_real * freq_real - k_imag * freq_imag
        k_out_imag = k_imag * freq_real + k_real * freq_imag
        q_out_offsets = (
            _fused_qk_complex_rope_to_bf16_bits(q_out_real)
            | q_out_imag.to(tl.bfloat16)
            .to(tl.int16, bitcast=True)
            .to(tl.int32)
            << 16
        )
        k_out_offsets = (
            _fused_qk_complex_rope_to_bf16_bits(k_out_real)
            | k_out_imag.to(tl.bfloat16)
            .to(tl.int16, bitcast=True)
            .to(tl.int32)
            << 16
        )
        if NUM_TOKENS % BLOCK_TOKENS == 0:
            tl.store(q_out_ptr + position, q_out_offsets)
            tl.store(k_out_ptr + position, k_out_offsets)
        else:
            tl.store(
                q_out_ptr + position,
                q_out_offsets,
                token[:, None] < NUM_TOKENS,
            )
            tl.store(
                k_out_ptr + position,
                k_out_offsets,
                token[:, None] < NUM_TOKENS,
            )


@triton.jit
def _fused_qk_complex_rope_unpacked_tokens_kernel(
    q_ptr,
    k_ptr,
    freqs_ptr: tl.pointer_type(tl.float32),
    q_out_ptr,
    k_out_ptr,
    METADATA: tl.constexpr,
):
    NUM_TOKENS: tl.constexpr = METADATA[0]
    NUM_HEADS: tl.constexpr = METADATA[1]
    HEAD_DIM: tl.constexpr = METADATA[2]
    BLOCK_TOKENS: tl.constexpr = METADATA[3]
    NUM_PROGRAMS: tl.constexpr = METADATA[4]
    USE_INT64: tl.constexpr = METADATA[5]
    index = tl.arange(0, NUM_HEADS * HEAD_DIM)
    dim_offsets = tl.arange(0, HEAD_DIM)
    program_id = tl.program_id(0)
    if USE_INT64:
        program_id = program_id.to(tl.int64)
    for block_index in range(
        program_id, tl.cdiv(NUM_TOKENS, BLOCK_TOKENS), NUM_PROGRAMS
    ):
        token = block_index * BLOCK_TOKENS + tl.arange(0, BLOCK_TOKENS)
        position = token[:, None] * NUM_HEADS * HEAD_DIM + index[None, :]
        freq_ptr = freqs_ptr + token[:, None] * HEAD_DIM + dim_offsets[None, :]
        if NUM_TOKENS % BLOCK_TOKENS == 0:
            q_values = tl.load(q_ptr + position).to(tl.float32)
            k_values = tl.load(k_ptr + position).to(tl.float32)
            freq_values = tl.load(freq_ptr)
        else:
            q_values = tl.load(
                q_ptr + position, token[:, None] < NUM_TOKENS, 0
            ).to(tl.float32)
            k_values = tl.load(
                k_ptr + position, token[:, None] < NUM_TOKENS, 0
            ).to(tl.float32)
            freq_values = tl.load(freq_ptr, token[:, None] < NUM_TOKENS, 0)
        q_real, q_imag = tl.split(
            tl.reshape(q_values, (BLOCK_TOKENS, NUM_HEADS * HEAD_DIM // 2, 2))
        )
        k_real, k_imag = tl.split(
            tl.reshape(k_values, (BLOCK_TOKENS, NUM_HEADS * HEAD_DIM // 2, 2))
        )
        freq_real, freq_imag = tl.split(
            tl.reshape(freq_values, (BLOCK_TOKENS, HEAD_DIM // 2, 2))
        )
        freq_real = tl.reshape(
            tl.broadcast_to(
                freq_real[:, None, :], (BLOCK_TOKENS, NUM_HEADS, HEAD_DIM // 2)
            ),
            (BLOCK_TOKENS, NUM_HEADS * HEAD_DIM // 2),
        )
        freq_imag = tl.reshape(
            tl.broadcast_to(
                freq_imag[:, None, :], (BLOCK_TOKENS, NUM_HEADS, HEAD_DIM // 2)
            ),
            (BLOCK_TOKENS, NUM_HEADS * HEAD_DIM // 2),
        )
        q_out_real = q_real * freq_real - q_imag * freq_imag
        q_out_imag = q_imag * freq_real + q_real * freq_imag
        k_out_real = k_real * freq_real - k_imag * freq_imag
        k_out_imag = k_imag * freq_real + k_real * freq_imag
        q_out_offsets = tl.reshape(
            tl.join(q_out_real, q_out_imag),
            (BLOCK_TOKENS, NUM_HEADS * HEAD_DIM),
        )
        k_out_offsets = tl.reshape(
            tl.join(k_out_real, k_out_imag),
            (BLOCK_TOKENS, NUM_HEADS * HEAD_DIM),
        )
        if NUM_TOKENS % BLOCK_TOKENS == 0:
            tl.store(q_out_ptr + position, q_out_offsets)
            tl.store(k_out_ptr + position, k_out_offsets)
        else:
            tl.store(
                q_out_ptr + position,
                q_out_offsets,
                token[:, None] < NUM_TOKENS,
            )
            tl.store(
                k_out_ptr + position,
                k_out_offsets,
                token[:, None] < NUM_TOKENS,
            )


@triton.jit
def _fused_qk_complex_rope_tokens_unpacked_bf16_kernel(
    q_ptr: tl.pointer_type(tl.bfloat16),
    k_ptr: tl.pointer_type(tl.bfloat16),
    freqs_ptr: tl.pointer_type(tl.float32),
    q_out_ptr: tl.pointer_type(tl.bfloat16),
    k_out_ptr: tl.pointer_type(tl.bfloat16),
    METADATA: tl.constexpr,
):
    _fused_qk_complex_rope_unpacked_tokens_kernel(
        q_ptr, k_ptr, freqs_ptr, q_out_ptr, k_out_ptr, METADATA
    )


@triton.jit
def _fused_qk_complex_rope_stream_tokens(
    q_ptr: tl.pointer_type(tl.uint32),
    freqs_ptr: tl.pointer_type(tl.float32),
    q_out_ptr: tl.pointer_type(tl.uint32),
    METADATA: tl.constexpr,
    group_program_id,
    GROUPS: tl.constexpr,
):
    NUM_TOKENS: tl.constexpr = METADATA[0]
    NUM_HEADS: tl.constexpr = METADATA[1]
    HEAD_DIM: tl.constexpr = METADATA[2]
    BLOCK_TOKENS: tl.constexpr = METADATA[3]
    USE_INT64: tl.constexpr = METADATA[5]
    index = tl.arange(0, NUM_HEADS * HEAD_DIM // 2)
    dim_offsets = tl.arange(0, HEAD_DIM)
    program_id = group_program_id
    if USE_INT64:
        program_id = program_id.to(tl.int64)
    NUM_BLOCKS: tl.constexpr = tl.cdiv(NUM_TOKENS, BLOCK_TOKENS)
    start = program_id * NUM_BLOCKS // GROUPS
    end = (program_id + 1) * NUM_BLOCKS // GROUPS
    for block_index in range(start, end):
        token = block_index * BLOCK_TOKENS + tl.arange(0, BLOCK_TOKENS)
        position = (
            token[:, None] * (NUM_HEADS * HEAD_DIM // 2) + index[None, :]
        )
        freq_ptr = freqs_ptr + token[:, None] * HEAD_DIM + dim_offsets[None, :]
        if NUM_TOKENS % BLOCK_TOKENS == 0:
            freq_values = tl.load(freq_ptr)
        else:
            freq_values = tl.load(freq_ptr, token[:, None] < NUM_TOKENS, 0)
        freq_real, freq_imag = tl.split(
            tl.reshape(freq_values, (BLOCK_TOKENS, HEAD_DIM // 2, 2))
        )
        freq_real = tl.reshape(
            tl.broadcast_to(
                freq_real[:, None, :], (BLOCK_TOKENS, NUM_HEADS, HEAD_DIM // 2)
            ),
            (BLOCK_TOKENS, NUM_HEADS * HEAD_DIM // 2),
        )
        freq_imag = tl.reshape(
            tl.broadcast_to(
                freq_imag[:, None, :], (BLOCK_TOKENS, NUM_HEADS, HEAD_DIM // 2)
            ),
            (BLOCK_TOKENS, NUM_HEADS * HEAD_DIM // 2),
        )
        if NUM_TOKENS % BLOCK_TOKENS == 0:
            q_bits = tl.load(q_ptr + position)
        else:
            q_bits = tl.load(q_ptr + position, token[:, None] < NUM_TOKENS, 0)
        q_real = (q_bits << 16).to(tl.float32, bitcast=True)
        q_imag = (q_bits & 4294901760).to(tl.float32, bitcast=True)
        q_out_real = q_real * freq_real - q_imag * freq_imag
        q_out_imag = q_imag * freq_real + q_real * freq_imag
        q_out_offsets = (
            _fused_qk_complex_rope_to_bf16_bits(q_out_real)
            | q_out_imag.to(tl.bfloat16)
            .to(tl.int16, bitcast=True)
            .to(tl.int32)
            << 16
        )
        if NUM_TOKENS % BLOCK_TOKENS == 0:
            tl.store(q_out_ptr + position, q_out_offsets)
        else:
            tl.store(
                q_out_ptr + position,
                q_out_offsets,
                token[:, None] < NUM_TOKENS,
            )


@triton.jit
def _fused_qk_complex_rope_sequential_tokens_kernel(
    q_ptr: tl.pointer_type(tl.uint32),
    k_ptr: tl.pointer_type(tl.uint32),
    freqs_ptr: tl.pointer_type(tl.float32),
    q_out_ptr: tl.pointer_type(tl.uint32),
    k_out_ptr: tl.pointer_type(tl.uint32),
    METADATA: tl.constexpr,
):
    NUM_PROGRAMS: tl.constexpr = METADATA[4]
    GROUPS: tl.constexpr = NUM_PROGRAMS // 2
    program_id = tl.program_id(0)
    if program_id < GROUPS:
        _fused_qk_complex_rope_stream_tokens(
            q_ptr, freqs_ptr, q_out_ptr, METADATA, program_id, GROUPS
        )
    else:
        _fused_qk_complex_rope_stream_tokens(
            k_ptr, freqs_ptr, k_out_ptr, METADATA, program_id - GROUPS, GROUPS
        )


@triton.jit
def _fused_qk_complex_rope_packed_tokens_inplace_kernel(
    q_ptr: tl.pointer_type(tl.uint32),
    k_ptr: tl.pointer_type(tl.uint32),
    freqs_ptr: tl.pointer_type(tl.float32),
    METADATA: tl.constexpr,
):
    _fused_qk_complex_rope_packed_tokens_kernel(
        q_ptr, k_ptr, freqs_ptr, q_ptr, k_ptr, METADATA
    )


def fused_qk_complex_rope(
    q,
    k,
    freqs_cis,
):
    num_elements = q.numel()
    if not num_elements:
        return (
            torch.empty(q.shape, dtype=q.dtype, device=q.device),
            torch.empty(k.shape, dtype=k.dtype, device=k.device),
        )
    num_heads, head_dim = q.shape[-2:]
    if not (
        q.is_contiguous()
        and k.is_contiguous()
        and freqs_cis.is_contiguous()
        and (num_heads & num_heads - 1 == 0)
        and (head_dim & head_dim - 1 == 0)
        and (num_heads * head_dim <= 2048)
    ):
        return _fused_qk_complex_rope_compute_strided(q, k, freqs_cis)
    num_tokens = num_elements // (num_heads * head_dim)
    block_tokens = min(8, max(1, 4096 // (num_heads * head_dim)))
    if num_tokens < block_tokens:
        block_tokens = 1 << num_tokens.bit_length() - 1
    packed = (
        q.dtype == torch.bfloat16
        and (num_elements >= 2097152 or block_tokens * head_dim > 512)
        and (q.storage_offset() % 2 == k.storage_offset() % 2 == 0)
    )
    if not packed and block_tokens * head_dim > 512:
        return _fused_qk_complex_rope_compute_strided(q, k, freqs_cis)
    use_seq = False
    if packed and num_elements >= 33554432:
        alt = min(4, 8192 // (num_heads * head_dim), 4096 // head_dim)
        if alt > block_tokens and num_tokens % alt == 0:
            block_tokens = alt
            use_seq = True
    num_programs = min((num_tokens + block_tokens - 1) // block_tokens, 32768)
    if use_seq:
        num_programs = min(num_programs, 40)
    inplace = q.is_contiguous() and k.is_contiguous()
    if inplace:
        q_strides = q.untyped_storage().data_ptr()
        k_strides = k.untyped_storage().data_ptr()
        freq_strides = freqs_cis.untyped_storage().data_ptr()
        inplace = (
            q_strides != k_strides
            and q_strides != freq_strides
            and k_strides != freq_strides
        )
    if inplace:
        q_out, k_out = q, k
    else:
        q_out = torch.empty_like(q)
        k_out = torch.empty_like(k)
    freqs_real = freqs_cis.data_ptr()
    if packed and inplace and not use_seq:
        q_address = q.data_ptr()
        k_address = k.data_ptr()
        kernel = _fused_qk_complex_rope_packed_tokens_inplace_kernel
        kernel.run(
            q_address,
            k_address,
            freqs_real,
            (
                num_tokens,
                num_heads,
                head_dim,
                block_tokens,
                num_programs,
                num_elements + 4096 > 2147483647,
            ),
            grid=(num_programs,),
            warmup=False,
            num_warps=4,
            enable_fp_fusion=False,
        )
        return (q_out, k_out)
    if packed:
        q_address = q.data_ptr()
        k_address = k.data_ptr()
        q_out_address = q_out.data_ptr()
        k_out_address = k_out.data_ptr()
        kernel = (
            _fused_qk_complex_rope_sequential_tokens_kernel
            if use_seq
            else _fused_qk_complex_rope_packed_tokens_kernel
        )
    else:
        if q.dtype == torch.bfloat16 and k.dtype == torch.bfloat16:
            q_address, k_address, q_out_address, k_out_address = (
                q.data_ptr(),
                k.data_ptr(),
                q_out.data_ptr(),
                k_out.data_ptr(),
            )
            kernel = _fused_qk_complex_rope_tokens_unpacked_bf16_kernel
        else:
            q_address, k_address, q_out_address, k_out_address = (
                q,
                k,
                q_out,
                k_out,
            )
            kernel = _fused_qk_complex_rope_unpacked_tokens_kernel
    kernel.run(
        q_address,
        k_address,
        freqs_real,
        q_out_address,
        k_out_address,
        (
            num_tokens,
            num_heads,
            head_dim,
            block_tokens,
            num_programs,
            num_elements + 4096 > 2147483647,
        ),
        grid=(num_programs,),
        warmup=False,
        num_warps=4,
        enable_fp_fusion=False,
    )
    return (q_out, k_out)


__all__ = ["fused_qk_complex_rope"]
