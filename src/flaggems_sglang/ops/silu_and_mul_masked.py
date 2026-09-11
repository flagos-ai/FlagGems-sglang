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

from __future__ import annotations

import torch
import triton
import triton.language as tl


def _row_configs():
    cfgs = []

    for bt in (1, 2, 4, 8, 16, 32):
        for w in (4, 8, 16):
            for s in (2, 3, 4):
                cfgs.append(
                    triton.Config({"BLOCK_T": bt}, num_warps=w, num_stages=s)
                )

    for bt in (8, 16, 32):
        cfgs.append(triton.Config({"BLOCK_T": bt}, num_warps=2, num_stages=3))
    return cfgs


@triton.autotune(configs=_row_configs(), key=["T", "H"])
@triton.jit
def _silu_and_mul_masked_row_kernel(
    input_ptr,
    out_ptr,
    masked_m_ptr,
    T,
    stride_e_in,
    stride_t_in,
    stride_h_in,
    stride_e_out,
    stride_t_out,
    stride_h_out,
    H: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    half: tl.constexpr = H // 2
    e = tl.program_id(1)
    t_block = tl.program_id(0)

    n = tl.load(masked_m_ptr + e)

    t_idx = t_block * BLOCK_T + tl.arange(0, BLOCK_T)
    row_valid = (t_idx < T) & (t_idx < n)

    in_row_ptr = input_ptr + e * stride_e_in + t_idx * stride_t_in
    cols = tl.arange(0, half)
    gate = tl.load(
        in_row_ptr[:, None] + cols[None, :] * stride_h_in,
        mask=row_valid[:, None],
        other=0.0,
    ).to(tl.float32)
    up = tl.load(
        in_row_ptr[:, None] + (cols[None, :] + half) * stride_h_in,
        mask=row_valid[:, None],
        other=0.0,
    ).to(tl.float32)

    val = gate * tl.sigmoid(gate) * up

    out_row_ptr = out_ptr + e * stride_e_out + t_idx * stride_t_out
    out_cols = tl.arange(0, half)
    tl.store(
        out_row_ptr[:, None] + out_cols[None, :] * stride_h_out,
        val.to(out_ptr.dtype.element_ty),
        mask=row_valid[:, None],
    )


def _col_configs():
    cfgs = []

    for b in (1024, 2048):
        for r in (1, 2, 4, 8, 16):
            for w in (4, 8, 16):
                for s in (3, 4, 5):
                    cfgs.append(
                        triton.Config(
                            {"BLOCK": b, "ROWS": r, "FUSED": 0},
                            num_warps=w,
                            num_stages=s,
                        )
                    )

    for r in (1, 2):
        for s in (3, 4):
            cfgs.append(
                triton.Config(
                    {"BLOCK": 2048, "ROWS": r, "FUSED": 0},
                    num_warps=2,
                    num_stages=s,
                )
            )

    for r in (1, 2, 4):
        for w in (4, 8, 16):
            for s in (3, 4, 5):
                cfgs.append(
                    triton.Config(
                        {"BLOCK": 2048, "ROWS": r, "FUSED": 1},
                        num_warps=w,
                        num_stages=s,
                    )
                )
    for r in (1, 2):
        cfgs.append(
            triton.Config(
                {"BLOCK": 2048, "ROWS": r, "FUSED": 1},
                num_warps=2,
                num_stages=3,
            )
        )
    return cfgs


@triton.autotune(configs=_col_configs(), key=["E", "T", "half"])
@triton.jit
def _silu_and_mul_masked_col_kernel(
    input_ptr,
    out_ptr,
    masked_m_ptr,
    E,
    T,
    stride_e_in,
    stride_t_in,
    stride_h_in,
    stride_e_out,
    stride_t_out,
    stride_h_out,
    BLOCK: tl.constexpr,
    half: tl.constexpr,
    ROWS: tl.constexpr,
    FUSED: tl.constexpr,
):
    col_block = tl.program_id(0)
    row_block = tl.program_id(1)

    # Flat (e, t) indices for the ROWS rows this program owns.
    rows = row_block * ROWS + tl.arange(0, ROWS)
    rows_in_range = rows < E * T
    e = tl.where(rows_in_range, rows // T, 0)
    t = rows % T

    n = tl.load(masked_m_ptr + e)
    row_valid = (t < n) & (t < T) & rows_in_range

    cols = col_block * BLOCK + tl.arange(0, BLOCK)
    col_mask = cols < half

    in_ptr = input_ptr + e[:, None] * stride_e_in + t[:, None] * stride_t_in

    if FUSED:

        h_cols = col_block * BLOCK + tl.arange(0, 2 * BLOCK)
        full = tl.load(
            in_ptr + h_cols[None, :] * stride_h_in,
            mask=row_valid[:, None],
            other=0.0,
        ).to(tl.float32)
        full = tl.reshape(full, (ROWS, 2, BLOCK))
        full = tl.permute(full, (0, 2, 1))
        gate, up = tl.split(full)
    else:
        gate = tl.load(
            in_ptr + cols[None, :] * stride_h_in,
            mask=col_mask[None, :] & row_valid[:, None],
            other=0.0,
        ).to(tl.float32)
        up = tl.load(
            in_ptr + (cols[None, :] + half) * stride_h_in,
            mask=col_mask[None, :] & row_valid[:, None],
            other=0.0,
        ).to(tl.float32)

    val = gate * tl.sigmoid(gate) * up

    out_ptr2 = out_ptr + e[:, None] * stride_e_out + t[:, None] * stride_t_out
    tl.store(
        out_ptr2 + cols[None, :] * stride_h_out,
        val.to(out_ptr.dtype.element_ty),
        mask=col_mask[None, :] & row_valid[:, None],
    )


_ROW_KERNEL_MAX_ROWS = 1024


def silu_and_mul_masked(input, masked_m):
    E, T, H = input.shape
    half = H // 2

    out = torch.empty(E, T, half, dtype=input.dtype, device=input.device)

    if not input.is_contiguous():
        input = input.contiguous()

    s_ei, s_ti, s_hi = input.stride(0), input.stride(1), input.stride(2)
    s_eo, s_to, s_ho = out.stride(0), out.stride(1), out.stride(2)

    if E * T <= _ROW_KERNEL_MAX_ROWS:
        grid = lambda meta: (triton.cdiv(T, meta["BLOCK_T"]), E)
        _silu_and_mul_masked_row_kernel[grid](
            input,
            out,
            masked_m,
            T,
            s_ei,
            s_ti,
            s_hi,
            s_eo,
            s_to,
            s_ho,
            H,
        )
    else:
        grid = lambda meta: (
            triton.cdiv(half, meta["BLOCK"]),
            triton.cdiv(E * T, meta["ROWS"]),
        )
        _silu_and_mul_masked_col_kernel[grid](
            input,
            out,
            masked_m,
            E,
            T,
            s_ei,
            s_ti,
            s_hi,
            s_eo,
            s_to,
            s_ho,
            half=half,
        )
    return out


__all__ = ["silu_and_mul_masked"]
