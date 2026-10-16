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

_I64 = tl.int64


@triton.jit
def _concat_mla_absorb_q_flat_kernel(
    a_ptr,
    b_ptr,
    out_ptr,
    n_out,
    A_LAST: tl.constexpr,
    B_LAST: tl.constexpr,
    BLOCK: tl.constexpr,
):
    OUT_LAST: tl.constexpr = A_LAST + B_LAST
    pid = tl.program_id(0)
    idx = pid * BLOCK + tl.arange(0, BLOCK)
    m = idx < n_out
    row = idx // OUT_LAST
    col = idx % OUT_LAST
    in_a = col < A_LAST
    a_v = tl.load(a_ptr + row * A_LAST + col, mask=m & in_a, other=0.0)
    b_v = tl.load(
        b_ptr + row * B_LAST + (col - A_LAST), mask=m & (~in_a), other=0.0
    )
    tl.store(out_ptr + idx, tl.where(in_a, a_v, b_v), mask=m)


@triton.jit
def _concat_mla_absorb_q_wide_kernel(
    a_ptr,
    b_ptr,
    out_ptr,
    n_out,
    A_LAST: tl.constexpr,
    B_LAST: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pa = a_ptr.to(tl.pointer_type(_I64))
    pb = b_ptr.to(tl.pointer_type(_I64))
    po = out_ptr.to(tl.pointer_type(_I64))
    OUT_LAST: tl.constexpr = A_LAST + B_LAST
    pid = tl.program_id(0)
    idx = pid * BLOCK + tl.arange(0, BLOCK)
    m = idx < n_out
    row = idx // OUT_LAST
    col = idx % OUT_LAST
    in_a = col < A_LAST
    a_v = tl.load(pa + row * A_LAST + col, mask=m & in_a, other=0)
    b_v = tl.load(
        pb + row * B_LAST + (col - A_LAST), mask=m & (~in_a), other=0
    )
    tl.store(po + idx, tl.where(in_a, a_v, b_v), mask=m)


@triton.jit
def _concat_mla_absorb_q_dense_kernel(
    a_ptr,
    b_ptr,
    out_ptr,
    num_rows,
    R: tl.constexpr,
    AL: tl.constexpr,
    BL: tl.constexpr,
):
    pa = a_ptr.to(tl.pointer_type(_I64))
    pb = b_ptr.to(tl.pointer_type(_I64))
    po = out_ptr.to(tl.pointer_type(_I64))
    pid = tl.program_id(0)
    rows = pid * R + tl.arange(0, R)
    rm = rows < num_rows
    ca = tl.arange(0, AL)
    va = tl.load(
        pa + rows[:, None] * AL + ca[None, :], mask=rm[:, None], other=0
    )
    tl.store(
        po + rows[:, None] * (AL + BL) + ca[None, :], va, mask=rm[:, None]
    )
    cb = tl.arange(0, BL)
    vb = tl.load(
        pb + rows[:, None] * BL + cb[None, :], mask=rm[:, None], other=0
    )
    tl.store(
        po + rows[:, None] * (AL + BL) + AL + cb[None, :], vb, mask=rm[:, None]
    )


@triton.jit
def _concat_mla_absorb_q_strided_kernel(
    a_ptr,
    b_ptr,
    out_ptr,
    num_rows,
    sa0,
    sa1,
    sb0,
    sb1,
    H: tl.constexpr,
    A_LAST: tl.constexpr,
    B_LAST: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = pid * BLOCK_R + tl.arange(0, BLOCK_R)
    rmask = rows < num_rows
    tok = rows // H
    head = rows % H

    a_cols = tl.arange(0, triton.next_power_of_2(A_LAST))
    b_cols = tl.arange(0, triton.next_power_of_2(B_LAST))
    a_mask = a_cols < A_LAST
    b_mask = b_cols < B_LAST

    a_base = a_ptr + tok[:, None] * sa0 + head[:, None] * sa1
    b_base = b_ptr + tok[:, None] * sb0 + head[:, None] * sb1
    out_base = out_ptr + rows[:, None] * (A_LAST + B_LAST)
    m = rmask[:, None]

    a_v = tl.load(
        a_base + a_cols[None, :], mask=m & a_mask[None, :], other=0.0
    )
    tl.store(out_base + a_cols[None, :], a_v, mask=m & a_mask[None, :])
    b_v = tl.load(
        b_base + b_cols[None, :], mask=m & b_mask[None, :], other=0.0
    )
    tl.store(
        out_base + A_LAST + b_cols[None, :], b_v, mask=m & b_mask[None, :]
    )


def concat_mla_absorb_q(a, b):
    dim0, dim1, a_last = a.shape
    b_last = b.shape[-1]
    num_rows = dim0 * dim1
    out = torch.empty(
        dim0, dim1, a_last + b_last, dtype=a.dtype, device=a.device
    )

    if a.is_contiguous() and b.is_contiguous():
        esz = a.element_size()
        if (
            esz in (2, 4)
            and a_last % (8 // esz) == 0
            and b_last % (8 // esz) == 0
        ):
            epl = 8 // esz
            n_out = out.numel() // epl
            if n_out >= (1 << 23):
                grid = ((num_rows + 1) // 2,)
                _concat_mla_absorb_q_dense_kernel[grid](
                    a,
                    b,
                    out,
                    num_rows,
                    R=2,
                    AL=a_last // epl,
                    BL=b_last // epl,
                    num_warps=2,
                    num_stages=2,
                )
            else:
                block = 1024
                num_warps = 8 if n_out >= (1 << 22) else 4
                grid = ((n_out + block - 1) // block,)
                _concat_mla_absorb_q_wide_kernel[grid](
                    a,
                    b,
                    out,
                    n_out,
                    A_LAST=a_last // epl,
                    B_LAST=b_last // epl,
                    BLOCK=block,
                    num_warps=num_warps,
                )
        else:
            n_out = out.numel()
            if n_out >= (1 << 28):
                block, num_warps = 2048, 8
            elif n_out >= (1 << 19):
                block, num_warps = 4096, 8
            else:
                block, num_warps = 1024, 4
            grid = ((n_out + block - 1) // block,)
            _concat_mla_absorb_q_flat_kernel[grid](
                a,
                b,
                out,
                n_out,
                A_LAST=a_last,
                B_LAST=b_last,
                BLOCK=block,
                num_warps=num_warps,
            )
    else:
        block_r, num_warps = (8, 4) if num_rows >= 65536 else (8, 1)
        grid = ((num_rows + block_r - 1) // block_r,)
        _concat_mla_absorb_q_strided_kernel[grid](
            a,
            b,
            out,
            num_rows,
            a.stride(0),
            a.stride(1),
            b.stride(0),
            b.stride(1),
            H=dim1,
            A_LAST=a_last,
            B_LAST=b_last,
            BLOCK_R=block_r,
            num_warps=num_warps,
        )
    return out


__all__ = ["concat_mla_absorb_q"]
