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
def _concat_mla_absorb_q_row_kernel(
    a_ptr,
    b_ptr,
    out_ptr,
    num_rows,
    a_last,
    b_last,
    A_BLOCK: tl.constexpr,
    B_BLOCK: tl.constexpr,
    ROWS: tl.constexpr,
    FULL_A: tl.constexpr,
    FULL_B: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = pid * ROWS + tl.arange(0, ROWS)
    out_row_stride = a_last + b_last
    a_cols = tl.arange(0, A_BLOCK)
    b_cols = tl.arange(0, B_BLOCK)
    if FULL_A:
        a_vals = tl.load(a_ptr + rows[:, None] * a_last + a_cols[None, :])
        tl.store(
            out_ptr + rows[:, None] * out_row_stride + a_cols[None, :], a_vals
        )
    else:
        a_mask = (rows < num_rows)[:, None] & (a_cols < a_last)[None, :]
        a_vals = tl.load(
            a_ptr + rows[:, None] * a_last + a_cols[None, :],
            mask=a_mask,
            other=0.0,
        )
        tl.store(
            out_ptr + rows[:, None] * out_row_stride + a_cols[None, :],
            a_vals,
            mask=a_mask,
        )
    if FULL_B:
        b_vals = tl.load(b_ptr + rows[:, None] * b_last + b_cols[None, :])
        tl.store(
            out_ptr
            + rows[:, None] * out_row_stride
            + (a_last + b_cols[None, :]),
            b_vals,
        )
    else:
        b_mask = (rows < num_rows)[:, None] & (b_cols < b_last)[None, :]
        b_vals = tl.load(
            b_ptr + rows[:, None] * b_last + b_cols[None, :],
            mask=b_mask,
            other=0.0,
        )
        tl.store(
            out_ptr
            + rows[:, None] * out_row_stride
            + (a_last + b_cols[None, :]),
            b_vals,
            mask=b_mask,
        )


@triton.jit
def _concat_mla_absorb_q_fused_masked_kernel(
    a_ptr,
    b_ptr,
    out_ptr,
    num_rows,
    a_last,
    b_last,
    A_BLOCK: tl.constexpr,
    B_BLOCK: tl.constexpr,
    ROWS: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = pid * ROWS + tl.arange(0, ROWS)
    out_row_stride = a_last + b_last
    a_cols = tl.arange(0, A_BLOCK)
    b_cols = tl.arange(0, B_BLOCK)
    a_mask = (rows < num_rows)[:, None] & (a_cols < a_last)[None, :]
    a_vals = tl.load(
        a_ptr + rows[:, None] * a_last + a_cols[None, :],
        mask=a_mask,
        other=0.0,
    )
    tl.store(
        out_ptr + rows[:, None] * out_row_stride + a_cols[None, :],
        a_vals,
        mask=a_mask,
    )
    b_mask = (rows < num_rows)[:, None] & (b_cols < b_last)[None, :]
    b_vals = tl.load(
        b_ptr + rows[:, None] * b_last + b_cols[None, :],
        mask=b_mask,
        other=0.0,
    )
    tl.store(
        out_ptr + rows[:, None] * out_row_stride + (a_last + b_cols[None, :]),
        b_vals,
        mask=b_mask,
    )


_ROW_KERNEL_MAX_SMALL = 1024
_ROW_KERNEL_MIN_LARGE = 16384


def _is_pow2(n):
    return n > 0 and (n & (n - 1)) == 0


def concat_mla_absorb_q(a, b):
    if not a.is_contiguous():
        a = a.contiguous()
    if not b.is_contiguous():
        b = b.contiguous()

    a_last = a.shape[-1]
    b_last = b.shape[-1]
    leading = a.shape[:-1]

    out = torch.empty(
        leading + (a_last + b_last,), dtype=a.dtype, device=a.device
    )
    num_rows = 1
    for s in leading:
        num_rows *= s
    if num_rows == 0 or a_last == 0 and b_last == 0:
        return out

    a2 = a.reshape(num_rows, a_last)
    b2 = b.reshape(num_rows, b_last)
    o2 = out.reshape(num_rows, a_last + b_last)

    a_block = max(triton.next_power_of_2(a_last), 1)
    b_block = max(triton.next_power_of_2(b_last), 1)
    full_cols = (
        a_block == a_last and b_block == b_last and a_last > 0 and b_last > 0
    )

    a_half = a_last // 2
    b_half = b_last // 2
    if (
        a.element_size() == 2
        and full_cols
        and a_last % 2 == 0
        and b_last % 2 == 0
        and _is_pow2(a_half)
        and _is_pow2(b_half)
        and num_rows % 4 == 0
    ):
        a32 = a2.view(torch.int32)
        b32 = b2.view(torch.int32)
        o32 = o2.view(torch.int32)
        if num_rows <= _ROW_KERNEL_MAX_SMALL:
            rows, nw = 1, 4
        elif num_rows < _ROW_KERNEL_MIN_LARGE:
            rows, nw = 2, 8
        else:
            rows, nw = 4, 16
        _concat_mla_absorb_q_row_kernel[(num_rows // rows,)](
            a32,
            b32,
            o32,
            num_rows,
            a_half,
            b_half,
            A_BLOCK=a_half,
            B_BLOCK=b_half,
            ROWS=rows,
            FULL_A=True,
            FULL_B=True,
            num_warps=nw,
            num_stages=1,
        )
        return out

    if full_cols and num_rows % 4 == 0:
        _concat_mla_absorb_q_row_kernel[(num_rows // 4,)](
            a2,
            b2,
            o2,
            num_rows,
            a_last,
            b_last,
            A_BLOCK=a_block,
            B_BLOCK=b_block,
            ROWS=4,
            FULL_A=True,
            FULL_B=True,
            num_warps=16,
            num_stages=1,
        )
        return out

    _concat_mla_absorb_q_fused_masked_kernel[(triton.cdiv(num_rows, 2),)](
        a2,
        b2,
        o2,
        num_rows,
        a_last,
        b_last,
        A_BLOCK=a_block,
        B_BLOCK=b_block,
        ROWS=2,
        num_warps=8,
        num_stages=1,
    )
    return out


__all__ = ["concat_mla_absorb_q"]
