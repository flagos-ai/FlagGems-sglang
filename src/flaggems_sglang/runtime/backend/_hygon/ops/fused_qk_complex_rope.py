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
def _fused_qk_complex_rope_metadata_rows_kernel(
    q_ptr,
    k_ptr,
    freqs_ptr,
    q_out_ptr,
    k_out_ptr,
    METADATA: tl.constexpr,
    IS_HIP: tl.constexpr,
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
        q_out_real = tl.fma(
            q_real, freq_real[None, :], -(q_imag * freq_imag[None, :])
        )
        q_out_imag = (
            tl.fma(q_imag, freq_real[None, :], q_real * freq_imag[None, :])
            if IS_HIP
            else tl.fma(
                q_real, freq_imag[None, :], q_imag * freq_real[None, :]
            )
        )
        k_out_real = tl.fma(
            k_real, freq_real[None, :], -(k_imag * freq_imag[None, :])
        )
        k_out_imag = (
            tl.fma(k_imag, freq_real[None, :], k_real * freq_imag[None, :])
            if IS_HIP
            else tl.fma(
                k_real, freq_imag[None, :], k_imag * freq_real[None, :]
            )
        )
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
        q_out_real = tl.fma(q_real, freq_real, -(q_imag * freq_imag))
        q_out_imag = (
            tl.fma(q_imag, freq_real, q_real * freq_imag)
            if IS_HIP
            else tl.fma(q_real, freq_imag, q_imag * freq_real)
        )
        k_out_real = tl.fma(k_real, freq_real, -(k_imag * freq_imag))
        k_out_imag = (
            tl.fma(k_imag, freq_real, k_real * freq_imag)
            if IS_HIP
            else tl.fma(k_real, freq_imag, k_imag * freq_real)
        )
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
def _fused_qk_complex_rope_metadata_flat_kernel(
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
    IS_HIP: tl.constexpr,
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
        q_out_real = tl.fma(q_real, freq_real, -(q_imag * freq_imag))
        q_out_imag = (
            tl.fma(q_imag, freq_real, q_real * freq_imag)
            if IS_HIP
            else tl.fma(q_real, freq_imag, q_imag * freq_real)
        )
        k_out_real = tl.fma(k_real, freq_real, -(k_imag * freq_imag))
        k_out_imag = (
            tl.fma(k_imag, freq_real, k_real * freq_imag)
            if IS_HIP
            else tl.fma(k_real, freq_imag, k_imag * freq_real)
        )
        tl.store(q_out_ptr + 2 * x, q_out_real, x < NUM_PAIRS)
        tl.store(q_out_ptr + 2 * x + 1, q_out_imag, x < NUM_PAIRS)
        tl.store(k_out_ptr + 2 * x, k_out_real, x < NUM_PAIRS)
        tl.store(k_out_ptr + 2 * x + 1, k_out_imag, x < NUM_PAIRS)


def _fused_qk_complex_rope_compute_metadata(
    q,
    k,
    freqs_cis,
):
    q_out = torch.empty(q.shape, dtype=q.dtype, device=q.device)
    k_out = torch.empty(k.shape, dtype=k.dtype, device=k.device)
    num_elements = q.numel()
    if num_elements:
        head_dim = q.shape[-1]
        num_rows = num_elements // head_dim
        block_rows = 8
        block_dim = 1 << (head_dim - 1).bit_length()
        freqs_real = torch.view_as_real(freqs_cis)
        use_int64 = num_elements + block_rows * head_dim > 2147483647
        if not use_int64 and (
            not (
                q.is_contiguous()
                and k.is_contiguous()
                and freqs_cis.is_contiguous()
            )
        ):
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
            _fused_qk_complex_rope_metadata_rows_kernel.run(
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
                torch.version.hip is not None,
                grid=((num_rows + block_rows - 1) // block_rows,),
                warmup=False,
                num_warps=4,
                enable_fp_fusion=False,
            )
        else:
            pairs = num_elements // 2
            _fused_qk_complex_rope_metadata_flat_kernel.run(
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
                torch.version.hip is not None,
                grid=(min((pairs + 512 - 1) // 512, 32768),),
                warmup=False,
                num_warps=4,
                enable_fp_fusion=False,
            )
    return (q_out, k_out)


@triton.jit
def _fused_qk_complex_rope_sequential_rows_kernel(
    q_ptr,
    k_ptr,
    freqs_ptr,
    q_out_ptr,
    k_out_ptr,
    SHAPE: tl.constexpr,
    Q_STRIDES: tl.constexpr,
    K_STRIDES: tl.constexpr,
    FREQ_STRIDES: tl.constexpr,
    NUM_ROWS: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_DIM: tl.constexpr,
    USE_INT64: tl.constexpr,
    IS_HIP: tl.constexpr,
):
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
        freq_real = freq_real[None, :]
        freq_imag = freq_imag[None, :]
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
    q_values = tl.load(
        q_ptr + q_base[:, None] + dim_offsets[None, :] * Q_STRIDES[-1], mask, 0
    ).to(tl.float32)
    q_real, q_imag = tl.split(
        tl.reshape(q_values, (BLOCK_ROWS, BLOCK_DIM // 2, 2))
    )
    q_out_real = tl.fma(q_real, freq_real, -(q_imag * freq_imag))
    q_out_imag = (
        tl.fma(q_imag, freq_real, q_real * freq_imag)
        if IS_HIP
        else tl.fma(q_real, freq_imag, q_imag * freq_real)
    )
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
    k_out_real = tl.fma(k_real, freq_real, -(k_imag * freq_imag))
    k_out_imag = (
        tl.fma(k_imag, freq_real, k_real * freq_imag)
        if IS_HIP
        else tl.fma(k_real, freq_imag, k_imag * freq_real)
    )
    k_out_values = tl.reshape(
        tl.join(k_out_real, k_out_imag), (BLOCK_ROWS, BLOCK_DIM)
    )
    tl.store(
        k_out_ptr + row[:, None] * SHAPE[-1] + dim_offsets[None, :],
        k_out_values,
        mask,
    )


@triton.jit
def _fused_qk_complex_rope_sequential_flat_kernel(
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
    IS_HIP: tl.constexpr,
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
        q_out_real = tl.fma(q_real, freq_real, -(q_imag * freq_imag))
        q_out_imag = (
            tl.fma(q_imag, freq_real, q_real * freq_imag)
            if IS_HIP
            else tl.fma(q_real, freq_imag, q_imag * freq_real)
        )
        k_out_real = tl.fma(k_real, freq_real, -(k_imag * freq_imag))
        k_out_imag = (
            tl.fma(k_imag, freq_real, k_real * freq_imag)
            if IS_HIP
            else tl.fma(k_real, freq_imag, k_imag * freq_real)
        )
        tl.store(q_out_ptr + 2 * x, q_out_real, x < NUM_PAIRS)
        tl.store(q_out_ptr + 2 * x + 1, q_out_imag, x < NUM_PAIRS)
        tl.store(k_out_ptr + 2 * x, k_out_real, x < NUM_PAIRS)
        tl.store(k_out_ptr + 2 * x + 1, k_out_imag, x < NUM_PAIRS)


def _fused_qk_complex_rope_compute_sequential(
    q,
    k,
    freqs_cis,
):
    q_out = torch.empty(q.shape, dtype=q.dtype, device=q.device)
    k_out = torch.empty(k.shape, dtype=k.dtype, device=k.device)
    num_elements = q.numel()
    if num_elements:
        head_dim = q.shape[-1]
        num_rows = num_elements // head_dim
        block_rows = 8
        block_dim = 1 << (head_dim - 1).bit_length()
        freqs_real = torch.view_as_real(freqs_cis)
        use_int64 = num_elements + block_rows * head_dim > 2147483647
        if not use_int64 and (
            not (
                q.is_contiguous()
                and k.is_contiguous()
                and freqs_cis.is_contiguous()
            )
        ):
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
            _fused_qk_complex_rope_sequential_rows_kernel.run(
                q,
                k,
                freqs_real,
                q_out,
                k_out,
                tuple(q.shape),
                q.stride(),
                k.stride(),
                freqs_real.stride(),
                num_rows,
                block_rows,
                block_dim,
                use_int64,
                torch.version.hip is not None,
                grid=((num_rows + block_rows - 1) // block_rows,),
                warmup=False,
                num_warps=4,
                enable_fp_fusion=False,
            )
        else:
            pairs = num_elements // 2
            _fused_qk_complex_rope_sequential_flat_kernel.run(
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
                torch.version.hip is not None,
                grid=(min((pairs + 512 - 1) // 512, 32768),),
                warmup=False,
                num_warps=4,
                enable_fp_fusion=False,
            )
    return (q_out, k_out)


def _fused_qk_complex_rope_compute_strided(
    q,
    k,
    freqs_cis,
):
    if (
        q.dtype == torch.bfloat16
        and q.ndim >= 3
        and (q.shape[-2:] == (16, 128))
        and (q.numel() >= 33554432)
        and q.is_contiguous()
        and k.is_contiguous()
        and freqs_cis.is_contiguous()
    ):
        return _fused_qk_complex_rope_compute_sequential(q, k, freqs_cis)
    return _fused_qk_complex_rope_compute_metadata(q, k, freqs_cis)


@triton.jit
def _fused_qk_complex_rope_decode_packed_pair(bits, IS_BF16: tl.constexpr):
    if IS_BF16:
        real = (bits << 16).to(tl.float32, bitcast=True)
        imag = (bits & 4294901760).to(tl.float32, bitcast=True)
    else:
        real = (
            (bits & 65535)
            .to(tl.uint16)
            .to(tl.float16, bitcast=True)
            .to(tl.float32)
        )
        imag = (
            (bits >> 16)
            .to(tl.uint16)
            .to(tl.float16, bitcast=True)
            .to(tl.float32)
        )
    return (real, imag)


@triton.jit
def _fused_qk_complex_rope_encode_packed_pair(
    real,
    imag,
    IS_BF16: tl.constexpr,
):
    if IS_BF16:
        real_bits = (
            real.to(tl.bfloat16).to(tl.uint16, bitcast=True).to(tl.uint32)
        )
        imag_bits = (
            imag.to(tl.bfloat16).to(tl.uint16, bitcast=True).to(tl.uint32)
        )
    else:
        real_bits = (
            real.to(tl.float16).to(tl.uint16, bitcast=True).to(tl.uint32)
        )
        imag_bits = (
            imag.to(tl.float16).to(tl.uint16, bitcast=True).to(tl.uint32)
        )
    return real_bits | imag_bits << 16


@triton.jit
def _fused_qk_complex_rope_packed_pairs_kernel(
    q_ptr: tl.pointer_type(tl.uint32),
    k_ptr: tl.pointer_type(tl.uint32),
    freqs_ptr,
    q_out_ptr: tl.pointer_type(tl.uint32),
    k_out_ptr: tl.pointer_type(tl.uint32),
    METADATA: tl.constexpr,
):
    NUM_ROWS: tl.constexpr = METADATA[0]
    NUM_HEADS: tl.constexpr = METADATA[1]
    HEAD_DIM: tl.constexpr = METADATA[2]
    BLOCK_PAIRS: tl.constexpr = METADATA[3]
    IS_BF16: tl.constexpr = METADATA[4]
    IS_HIP: tl.constexpr = METADATA[5]
    STORE_CACHE_MODIFIER: tl.constexpr = ".cs" if METADATA[6] else ""
    q_ptr = q_ptr.to(tl.pointer_type(tl.uint32))
    k_ptr = k_ptr.to(tl.pointer_type(tl.uint32))
    q_out_ptr = q_out_ptr.to(tl.pointer_type(tl.uint32))
    k_out_ptr = k_out_ptr.to(tl.pointer_type(tl.uint32))
    program_id = tl.program_id(0)
    row = program_id * 8 + tl.arange(0, 8)
    q_base = row * (HEAD_DIM // 2)
    k_base = q_base
    dim_offsets = tl.arange(0, BLOCK_PAIRS)
    mask = (row[:, None] < NUM_ROWS) & (dim_offsets[None, :] < HEAD_DIM // 2)
    freq_base_offset = program_id * 8 // NUM_HEADS * HEAD_DIM
    freq_real = tl.load(
        freqs_ptr + freq_base_offset + 2 * dim_offsets,
        dim_offsets < HEAD_DIM // 2,
        0,
    )[None, :]
    freq_imag = tl.load(
        freqs_ptr + freq_base_offset + 2 * dim_offsets + 1,
        dim_offsets < HEAD_DIM // 2,
        0,
    )[None, :]
    q = tl.load(q_ptr + q_base[:, None] + dim_offsets[None, :], mask, 0)
    q_real, q_imag = _fused_qk_complex_rope_decode_packed_pair(q, IS_BF16)
    q_out_real = tl.fma(q_real, freq_real, -(q_imag * freq_imag))
    q_out_imag = (
        tl.fma(q_imag, freq_real, q_real * freq_imag)
        if IS_HIP
        else tl.fma(q_real, freq_imag, q_imag * freq_real)
    )
    tl.store(
        q_out_ptr + row[:, None] * (HEAD_DIM // 2) + dim_offsets[None, :],
        _fused_qk_complex_rope_encode_packed_pair(
            q_out_real, q_out_imag, IS_BF16
        ),
        mask,
        cache_modifier=STORE_CACHE_MODIFIER,
    )
    k = tl.load(k_ptr + k_base[:, None] + dim_offsets[None, :], mask, 0)
    k_real, k_imag = _fused_qk_complex_rope_decode_packed_pair(k, IS_BF16)
    k_out_real = tl.fma(k_real, freq_real, -(k_imag * freq_imag))
    k_out_imag = (
        tl.fma(k_imag, freq_real, k_real * freq_imag)
        if IS_HIP
        else tl.fma(k_real, freq_imag, k_imag * freq_real)
    )
    tl.store(
        k_out_ptr + row[:, None] * (HEAD_DIM // 2) + dim_offsets[None, :],
        _fused_qk_complex_rope_encode_packed_pair(
            k_out_real, k_out_imag, IS_BF16
        ),
        mask,
        cache_modifier=STORE_CACHE_MODIFIER,
    )


def fused_qk_complex_rope(
    q,
    k,
    freqs_cis,
):
    q_shape = q.shape
    q_dtype = q.dtype
    num_elements = q.numel()
    eligible = (
        num_elements
        and q_dtype in (torch.bfloat16, torch.float16)
        and (q_shape[-1] % 2 == 0)
        and (q_shape[-1] <= 128)
        and (q_shape[-2] % 8 == 0)
        and q.is_contiguous()
        and k.is_contiguous()
        and freqs_cis.is_contiguous()
        and (q.data_ptr() % 4 == k.data_ptr() % 4 == 0)
        and (num_elements + 1024 <= 2147483647)
    )
    if not eligible:
        return _fused_qk_complex_rope_compute_strided(q, k, freqs_cis)
    q_out = torch.empty_like(q)
    k_out = torch.empty_like(k)
    freqs_real = freqs_cis.view(torch.float32)
    num_rows = num_elements // q_shape[-1]
    _fused_qk_complex_rope_packed_pairs_kernel.run(
        q.data_ptr(),
        k.data_ptr(),
        freqs_real,
        q_out.data_ptr(),
        k_out.data_ptr(),
        (
            num_rows,
            q_shape[-2],
            q_shape[-1],
            1 << (q_shape[-1] // 2 - 1).bit_length(),
            q_dtype == torch.bfloat16,
            torch.version.hip is not None,
            num_elements >= 33554432,
        ),
        grid=((num_rows + 8 - 1) // 8,),
        warmup=False,
        enable_fp_fusion=False,
    )
    return (q_out, k_out)


__all__ = ["fused_qk_complex_rope"]
