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
def _fused_qk_complex_rope_rows_kernel(
    q_ptr,
    k_ptr,
    freqs_ptr,
    q_out_ptr,
    k_out_ptr,
    METADATA: tl.constexpr,
):
    SHAPE: tl.constexpr = METADATA[0]
    Q_STRIDES: tl.constexpr = METADATA[1]
    K_STRIDES: tl.constexpr = METADATA[2]
    FREQ_STRIDES: tl.constexpr = METADATA[3]
    NUM_ROWS: tl.constexpr = METADATA[4]
    BLOCK_ROWS: tl.constexpr = METADATA[5]
    BLOCK_DIM: tl.constexpr = METADATA[6]
    USE_INT64: tl.constexpr = METADATA[7]
    program_id = tl.program_id(0)
    if USE_INT64:
        program_id = program_id.to(tl.int64)
    row = program_id * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    head_index = row % SHAPE[-2]
    token_index = row // SHAPE[-2]
    q_base = head_index * Q_STRIDES[-2]
    k_base = head_index * K_STRIDES[-2]
    freq_base = tl.full((BLOCK_ROWS,), 0, tl.int32)
    for axis in tl.static_range(len(SHAPE) - 3, -1, -1):
        coord = token_index % SHAPE[axis]
        token_index = token_index // SHAPE[axis]
        q_base += coord * Q_STRIDES[axis]
        k_base += coord * K_STRIDES[axis]
        freq_base += coord * FREQ_STRIDES[axis]
    dim_offsets = tl.arange(0, BLOCK_DIM)
    if USE_INT64:
        dim_offsets = dim_offsets.to(tl.int64)
    mask = (row[:, None] < NUM_ROWS) & (dim_offsets[None, :] < SHAPE[-1])
    q_values = tl.load(
        q_ptr + q_base[:, None] + dim_offsets[None, :] * Q_STRIDES[-1], mask, 0
    ).to(tl.float32)
    k_values = tl.load(
        k_ptr + k_base[:, None] + dim_offsets[None, :] * K_STRIDES[-1], mask, 0
    ).to(tl.float32)
    q_real, q_imag = tl.split(
        tl.reshape(q_values, (BLOCK_ROWS, BLOCK_DIM // 2, 2))
    )
    k_real, k_imag = tl.split(
        tl.reshape(k_values, (BLOCK_ROWS, BLOCK_DIM // 2, 2))
    )
    if SHAPE[-2] % BLOCK_ROWS == 0:
        freq_token = program_id * BLOCK_ROWS // SHAPE[-2]
        freq_base_offset = 0
        for axis in tl.static_range(len(SHAPE) - 3, -1, -1):
            coord = freq_token % SHAPE[axis]
            freq_token = freq_token // SHAPE[axis]
            freq_base_offset += coord * FREQ_STRIDES[axis]
        freq_values = tl.load(
            freqs_ptr
            + freq_base_offset
            + dim_offsets // 2 * FREQ_STRIDES[-2]
            + dim_offsets % 2 * FREQ_STRIDES[-1],
            dim_offsets < SHAPE[-1],
            0,
        )
        freq_real, freq_imag = tl.split(
            tl.reshape(freq_values, (BLOCK_DIM // 2, 2))
        )
        q_out_real = q_real * freq_real[None, :] - q_imag * freq_imag[None, :]
        q_out_imag = q_imag * freq_real[None, :] + q_real * freq_imag[None, :]
        k_out_real = k_real * freq_real[None, :] - k_imag * freq_imag[None, :]
        k_out_imag = k_imag * freq_real[None, :] + k_real * freq_imag[None, :]
    else:
        freq_values = tl.load(
            freqs_ptr
            + freq_base[:, None]
            + dim_offsets[None, :] // 2 * FREQ_STRIDES[-2]
            + dim_offsets[None, :] % 2 * FREQ_STRIDES[-1],
            mask,
            0,
        )
        freq_real, freq_imag = tl.split(
            tl.reshape(freq_values, (BLOCK_ROWS, BLOCK_DIM // 2, 2))
        )
        q_out_real = q_real * freq_real - q_imag * freq_imag
        q_out_imag = q_imag * freq_real + q_real * freq_imag
        k_out_real = k_real * freq_real - k_imag * freq_imag
        k_out_imag = k_imag * freq_real + k_real * freq_imag
    q_out_values = tl.reshape(
        tl.join(q_out_real, q_out_imag), (BLOCK_ROWS, BLOCK_DIM)
    )
    k_out_values = tl.reshape(
        tl.join(k_out_real, k_out_imag), (BLOCK_ROWS, BLOCK_DIM)
    )
    tl.store(
        q_out_ptr + row[:, None] * SHAPE[-1] + dim_offsets[None, :],
        q_out_values,
        mask,
    )
    tl.store(
        k_out_ptr + row[:, None] * SHAPE[-1] + dim_offsets[None, :],
        k_out_values,
        mask,
    )


@triton.jit
def _fused_qk_complex_rope_flat_kernel(
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
        x = block_index * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        dim_offsets = x % (SHAPE[-1] // 2)
        head_index = x // (SHAPE[-1] // 2) % SHAPE[-2]
        token_index = x // (SHAPE[-1] // 2 * SHAPE[-2])
        q_offsets = (
            head_index * Q_STRIDES[-2] + 2 * dim_offsets * Q_STRIDES[-1]
        )
        k_offsets = (
            head_index * K_STRIDES[-2] + 2 * dim_offsets * K_STRIDES[-1]
        )
        freq_offset = dim_offsets * FREQ_STRIDES[-2]
        for axis in tl.static_range(len(SHAPE) - 3, -1, -1):
            coord = token_index % SHAPE[axis]
            token_index = token_index // SHAPE[axis]
            q_offsets += coord * Q_STRIDES[axis]
            k_offsets += coord * K_STRIDES[axis]
            freq_offset += coord * FREQ_STRIDES[axis]
        q_real = tl.load(q_ptr + q_offsets, x < NUM_PAIRS, 0).to(tl.float32)
        q_imag = tl.load(
            q_ptr + q_offsets + Q_STRIDES[-1], x < NUM_PAIRS, 0
        ).to(tl.float32)
        k_real = tl.load(k_ptr + k_offsets, x < NUM_PAIRS, 0).to(tl.float32)
        k_imag = tl.load(
            k_ptr + k_offsets + K_STRIDES[-1], x < NUM_PAIRS, 0
        ).to(tl.float32)
        freq_real = tl.load(freqs_ptr + freq_offset, x < NUM_PAIRS, 0)
        freq_imag = tl.load(
            freqs_ptr + freq_offset + FREQ_STRIDES[-1], x < NUM_PAIRS, 0
        )
        q_out_real = q_real * freq_real - q_imag * freq_imag
        q_out_imag = q_imag * freq_real + q_real * freq_imag
        k_out_real = k_real * freq_real - k_imag * freq_imag
        k_out_imag = k_imag * freq_real + k_real * freq_imag
        tl.store(q_out_ptr + 2 * x, q_out_real, x < NUM_PAIRS)
        tl.store(q_out_ptr + 2 * x + 1, q_out_imag, x < NUM_PAIRS)
        tl.store(k_out_ptr + 2 * x, k_out_real, x < NUM_PAIRS)
        tl.store(k_out_ptr + 2 * x + 1, k_out_imag, x < NUM_PAIRS)


@triton.jit
def _fused_qk_complex_rope_dense_rows_kernel(
    q_ptr,
    k_ptr,
    freqs_ptr: tl.pointer_type(tl.float32),
    q_out_ptr,
    k_out_ptr,
    METADATA: tl.constexpr,
):
    SHAPE: tl.constexpr = METADATA[0]
    Q_STRIDES: tl.constexpr = (SHAPE[1] * SHAPE[2], SHAPE[2], 1)
    K_STRIDES: tl.constexpr = Q_STRIDES
    FREQ_STRIDES: tl.constexpr = (SHAPE[2], 2, 1)
    NUM_ROWS: tl.constexpr = SHAPE[0] * SHAPE[1]
    BLOCK_ROWS: tl.constexpr = 8
    BLOCK_DIM: tl.constexpr = METADATA[1]
    USE_INT64: tl.constexpr = False
    program_id = tl.program_id(0)
    tl.assume(program_id >= 0)
    tl.assume(program_id < (NUM_ROWS + BLOCK_ROWS - 1) // BLOCK_ROWS)
    if USE_INT64:
        program_id = program_id.to(tl.int64)
    row = program_id * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    q_base = row * SHAPE[-1]
    k_base = q_base
    freq_base = row // SHAPE[-2] * SHAPE[-1]
    dim_offsets = tl.arange(0, BLOCK_DIM)
    if USE_INT64:
        dim_offsets = dim_offsets.to(tl.int64)
    mask = (row[:, None] < NUM_ROWS) & (dim_offsets[None, :] < SHAPE[-1])
    q_values = tl.load(
        q_ptr + q_base[:, None] + dim_offsets[None, :] * Q_STRIDES[-1], mask, 0
    ).to(tl.float32)
    q_real, q_imag = tl.split(
        tl.reshape(q_values, (BLOCK_ROWS, BLOCK_DIM // 2, 2))
    )
    if SHAPE[-2] % BLOCK_ROWS == 0:
        freq_base_offset = program_id * BLOCK_ROWS // SHAPE[-2] * SHAPE[-1]
        freq_values = tl.load(
            freqs_ptr
            + freq_base_offset
            + dim_offsets // 2 * FREQ_STRIDES[-2]
            + dim_offsets % 2 * FREQ_STRIDES[-1],
            dim_offsets < SHAPE[-1],
            0,
        )
        freq_real, freq_imag = tl.split(
            tl.reshape(freq_values, (BLOCK_DIM // 2, 2))
        )
        q_out_real = q_real * freq_real[None, :] - q_imag * freq_imag[None, :]
        q_out_imag = q_imag * freq_real[None, :] + q_real * freq_imag[None, :]
    else:
        freq_values = tl.load(
            freqs_ptr
            + freq_base[:, None]
            + dim_offsets[None, :] // 2 * FREQ_STRIDES[-2]
            + dim_offsets[None, :] % 2 * FREQ_STRIDES[-1],
            mask,
            0,
        )
        freq_real, freq_imag = tl.split(
            tl.reshape(freq_values, (BLOCK_ROWS, BLOCK_DIM // 2, 2))
        )
        q_out_real = q_real * freq_real - q_imag * freq_imag
        q_out_imag = q_imag * freq_real + q_real * freq_imag
    q_out_values = tl.reshape(
        tl.join(q_out_real, q_out_imag), (BLOCK_ROWS, BLOCK_DIM)
    )
    tl.store(
        q_out_ptr + row[:, None] * SHAPE[-1] + dim_offsets[None, :],
        q_out_values,
        mask,
    )
    k_values = tl.load(
        k_ptr + k_base[:, None] + dim_offsets[None, :] * K_STRIDES[-1], mask, 0
    ).to(tl.float32)
    k_real, k_imag = tl.split(
        tl.reshape(k_values, (BLOCK_ROWS, BLOCK_DIM // 2, 2))
    )
    if SHAPE[-2] % BLOCK_ROWS == 0:
        k_out_real = k_real * freq_real[None, :] - k_imag * freq_imag[None, :]
        k_out_imag = k_imag * freq_real[None, :] + k_real * freq_imag[None, :]
    else:
        k_out_real = k_real * freq_real - k_imag * freq_imag
        k_out_imag = k_imag * freq_real + k_real * freq_imag
    k_out_values = tl.reshape(
        tl.join(k_out_real, k_out_imag), (BLOCK_ROWS, BLOCK_DIM)
    )
    tl.store(
        k_out_ptr + row[:, None] * SHAPE[-1] + dim_offsets[None, :],
        k_out_values,
        mask,
    )


@triton.jit
def _fused_qk_complex_rope_dense_bf16_rows_kernel(
    q_ptr: tl.pointer_type(tl.bfloat16),
    k_ptr: tl.pointer_type(tl.bfloat16),
    freqs_ptr: tl.pointer_type(tl.float32),
    q_out_ptr: tl.pointer_type(tl.bfloat16),
    k_out_ptr: tl.pointer_type(tl.bfloat16),
    METADATA: tl.constexpr,
):
    _fused_qk_complex_rope_dense_rows_kernel(
        q_ptr, k_ptr, freqs_ptr, q_out_ptr, k_out_ptr, METADATA
    )


@triton.jit
def _fused_qk_complex_rope_dense_bf16_inplace_kernel(
    q_ptr: tl.pointer_type(tl.bfloat16),
    k_ptr: tl.pointer_type(tl.bfloat16),
    freqs_ptr: tl.pointer_type(tl.float32),
    METADATA: tl.constexpr,
):
    _fused_qk_complex_rope_dense_bf16_rows_kernel(
        q_ptr, k_ptr, freqs_ptr, q_ptr, k_ptr, METADATA
    )


def fused_qk_complex_rope(
    q,
    k,
    freqs_cis,
):
    inplace = q.is_contiguous() and k.is_contiguous()
    if inplace:
        q_storage_address = q.untyped_storage().data_ptr()
        k_storage_address = k.untyped_storage().data_ptr()
        freq_storage_address = freqs_cis.untyped_storage().data_ptr()
        inplace = (
            q_storage_address != k_storage_address
            and q_storage_address != freq_storage_address
            and (k_storage_address != freq_storage_address)
        )
    if inplace:
        q_out, k_out = (q, k)
    else:
        q_out = torch.empty(q.shape, dtype=q.dtype, device=q.device)
        k_out = torch.empty(k.shape, dtype=k.dtype, device=k.device)
    num_elements = q.numel()
    if num_elements:
        head_dim = q.shape[-1]
        num_rows = num_elements // head_dim
        block_rows = 8
        block_dim = 1 << (head_dim - 1).bit_length()
        use_int64 = num_elements + block_rows * head_dim > 2147483647
        dense = (
            not use_int64
            and q.is_contiguous()
            and k.is_contiguous()
            and freqs_cis.is_contiguous()
        )
        freqs_real = (
            freqs_cis.data_ptr()
            if dense and q.ndim == 3 and (block_rows * block_dim <= 1024)
            else torch.view_as_real(freqs_cis)
        )
        if not use_int64 and (not dense):
            use_int64 = (
                max(
                    (
                        sum(
                            (
                                (rotation_sign - 1) * stride
                                for rotation_sign, stride in zip(
                                    x.shape, x.stride()
                                )
                            )
                        )
                        for x in (q, k, freqs_real)
                    )
                )
                > 2147483647
            )
        if block_rows * block_dim <= 1024:
            if dense and q.ndim == 3:
                if q.dtype == k.dtype == torch.bfloat16:
                    q_address, k_address = (q.data_ptr(), k.data_ptr())
                    q_out_address = q_address if inplace else q_out.data_ptr()
                    k_out_address = k_address if inplace else k_out.data_ptr()
                    if inplace:
                        _fused_qk_complex_rope_dense_bf16_inplace_kernel.run(
                            q_address,
                            k_address,
                            freqs_real,
                            (tuple(q.shape), block_dim),
                            grid=((num_rows + block_rows - 1) // block_rows,),
                            warmup=False,
                            num_warps=4,
                            enable_fp_fusion=False,
                        )
                    else:
                        _fused_qk_complex_rope_dense_bf16_rows_kernel.run(
                            q_address,
                            k_address,
                            freqs_real,
                            q_out_address,
                            k_out_address,
                            (tuple(q.shape), block_dim),
                            grid=((num_rows + block_rows - 1) // block_rows,),
                            warmup=False,
                            num_warps=4,
                            enable_fp_fusion=False,
                        )
                else:
                    _fused_qk_complex_rope_dense_rows_kernel.run(
                        q,
                        k,
                        freqs_real,
                        q_out,
                        k_out,
                        (tuple(q.shape), block_dim),
                        grid=((num_rows + block_rows - 1) // block_rows,),
                        warmup=False,
                        num_warps=4,
                        enable_fp_fusion=False,
                    )
            else:
                _fused_qk_complex_rope_rows_kernel.run(
                    q,
                    k,
                    freqs_real,
                    q_out,
                    k_out,
                    (
                        tuple(q.shape),
                        q.stride(),
                        k.stride(),
                        freqs_real.stride(),
                        num_rows,
                        block_rows,
                        block_dim,
                        use_int64,
                    ),
                    grid=((num_rows + block_rows - 1) // block_rows,),
                    warmup=False,
                    num_warps=4,
                    enable_fp_fusion=False,
                )
        else:
            pairs = num_elements // 2
            _fused_qk_complex_rope_flat_kernel.run(
                q,
                k,
                freqs_real,
                q_out,
                k_out,
                pairs,
                tuple(q.shape),
                q.stride(),
                k.stride(),
                freqs_real.stride(),
                512,
                use_int64,
                grid=(min((pairs + 512 - 1) // 512, 32768),),
                warmup=False,
                num_warps=4,
                enable_fp_fusion=False,
            )
    return (q_out, k_out)


__all__ = ["fused_qk_complex_rope"]
