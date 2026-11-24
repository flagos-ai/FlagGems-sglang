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
    SHAPE: tl.constexpr,
    Q_STRIDES: tl.constexpr,
    K_STRIDES: tl.constexpr,
    FREQ_STRIDES: tl.constexpr,
    NUM_ROWS: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_DIM: tl.constexpr,
):
    row = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
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
        freq_token = tl.program_id(0) * BLOCK_ROWS // SHAPE[-2]
        freq_base_offset = 0
        for axis in tl.static_range(len(SHAPE) - 3, -1, -1):
            coord = freq_token % SHAPE[axis]
            freq_token = freq_token // SHAPE[axis]
            freq_base_offset += coord * FREQ_STRIDES[axis]
        freq_values = tl.load(
            freqs_ptr
            + freq_base_offset
            + (dim_offsets // 2) * FREQ_STRIDES[-2]
            + (dim_offsets % 2) * FREQ_STRIDES[-1],
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
            + (dim_offsets[None, :] // 2) * FREQ_STRIDES[-2]
            + (dim_offsets[None, :] % 2) * FREQ_STRIDES[-1],
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


def fused_qk_complex_rope(
    q,
    k,
    freqs_cis,
):
    q_out = torch.empty(q.shape, dtype=q.dtype, device=q.device)
    k_out = torch.empty(k.shape, dtype=k.dtype, device=k.device)
    num_rows = q.numel() // q.shape[-1]
    if num_rows:
        freqs_real = torch.view_as_real(freqs_cis)
        block_rows = 8
        _fused_qk_complex_rope_rows_kernel.run(
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
            triton.next_power_of_2(q.shape[-1]),
            grid=(triton.cdiv(num_rows, block_rows),),
            warmup=False,
            num_warps=4,
            enable_fp_fusion=False,
        )
    return q_out, k_out


__all__ = ["fused_qk_complex_rope"]
