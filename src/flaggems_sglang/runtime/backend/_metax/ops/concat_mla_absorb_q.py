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


@triton.jit(do_not_specialize=["num_rows"])
def _concat_mla_absorb_q_kernel(
    a_ptr,
    b_ptr,
    out_ptr,
    num_rows,
    A_LAST: tl.constexpr,
    B_LAST: tl.constexpr,
    BLOCK_A: tl.constexpr,
    BLOCK_B: tl.constexpr,
    ROWS: tl.constexpr,
    ROW_EXACT: tl.constexpr,
):
    A_EXACT: tl.constexpr = BLOCK_A == A_LAST
    B_EXACT: tl.constexpr = BLOCK_B == B_LAST

    pid = tl.program_id(0)
    rows = pid * ROWS + tl.arange(0, ROWS)

    OUT_ROW: tl.constexpr = A_LAST + B_LAST

    offs_a = tl.arange(0, BLOCK_A)
    offs_b = tl.arange(0, BLOCK_B)

    a_ptrs = a_ptr + rows[:, None] * A_LAST + offs_a[None, :]
    b_ptrs = b_ptr + rows[:, None] * B_LAST + offs_b[None, :]
    out_a = out_ptr + rows[:, None] * OUT_ROW + offs_a[None, :]
    out_b = out_ptr + rows[:, None] * OUT_ROW + A_LAST + offs_b[None, :]

    if ROW_EXACT:
        if A_EXACT and B_EXACT:
            va = tl.load(a_ptrs, eviction_policy="evict_first")
            vb = tl.load(b_ptrs, eviction_policy="evict_first")
            tl.store(out_a, va)
            tl.store(out_b, vb)
        else:
            mask_a = offs_a[None, :] < A_LAST
            mask_b = offs_b[None, :] < B_LAST
            tl.store(out_a, tl.load(a_ptrs, mask=mask_a, other=0), mask=mask_a)
            tl.store(out_b, tl.load(b_ptrs, mask=mask_b, other=0), mask=mask_b)
    else:
        row_mask = rows < num_rows
        if A_EXACT and B_EXACT:
            mask_a = row_mask[:, None]
            mask_b = row_mask[:, None]
        else:
            mask_a = row_mask[:, None] & (offs_a[None, :] < A_LAST)
            mask_b = row_mask[:, None] & (offs_b[None, :] < B_LAST)
        tl.store(out_a, tl.load(a_ptrs, mask=mask_a, other=0), mask=mask_a)
        tl.store(out_b, tl.load(b_ptrs, mask=mask_b, other=0), mask=mask_b)


def _next_pow2(x):
    n = 1
    while n < x:
        n *= 2
    return n


def _launch_config(num_rows):
    if num_rows <= 256:
        return 1, 2
    if num_rows <= 4096:
        return 2, 2
    return 8, 8


def concat_mla_absorb_q(a, b):
    if not a.is_contiguous():
        a = a.contiguous()
    if not b.is_contiguous():
        b = b.contiguous()

    a_last = a.shape[-1]
    b_last = b.shape[-1]

    out_shape = a.shape[:-1] + (a_last + b_last,)
    out = torch.empty(out_shape, dtype=a.dtype, device=a.device)

    num_rows = a.numel() // a_last if a_last else b.numel() // b_last
    if num_rows == 0:
        return out

    rows, num_warps = _launch_config(num_rows)
    grid = (triton.cdiv(num_rows, rows),)

    _concat_mla_absorb_q_kernel[grid](
        a,
        b,
        out,
        num_rows,
        A_LAST=a_last,
        B_LAST=b_last,
        BLOCK_A=_next_pow2(a_last),
        BLOCK_B=_next_pow2(b_last),
        ROWS=rows,
        ROW_EXACT=(num_rows % rows == 0),
        num_warps=num_warps,
        num_stages=1,
    )
    return out


__all__ = ["concat_mla_absorb_q"]
