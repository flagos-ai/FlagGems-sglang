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

# Element count for the generic fallback launches (element-wide addressing).
_GEN_BLOCK = 8192

_G_SMALL = 64
_G_MED = 128
_G_LARGE = 256
_G_XLARGE = 512
_ROWS_MED = 2048
_ROWS_LARGE = 32768


@triton.jit
def _concat_rows_packed(
    a_ptr,
    b_ptr,
    o_ptr,
    A64: tl.constexpr,
    B64: tl.constexpr,
    ROW64: tl.constexpr,
    G: tl.constexpr,
):
    """Copy packed input rows into their output column ranges."""
    pid = tl.program_id(0)

    offs_a = pid * (G * A64) + tl.arange(0, G * A64)
    tl.store(
        o_ptr + (offs_a // A64) * ROW64 + (offs_a % A64),
        tl.load(a_ptr + offs_a),
    )

    offs_b = pid * (G * B64) + tl.arange(0, G * B64)
    tl.store(
        o_ptr + (offs_b // B64) * ROW64 + A64 + (offs_b % B64),
        tl.load(b_ptr + offs_b),
    )


@triton.jit
def _concat_flat_a(
    a_ptr,
    o_ptr,
    n_a,
    A_LAST: tl.constexpr,
    ROW: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Generic fallback, a-half: flat 1-D a range -> column 0 of each row."""
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n_a
    k = tl.where(m, offs, 0)
    tl.store(
        o_ptr + (k // A_LAST) * ROW + (k % A_LAST),
        tl.load(a_ptr + k, mask=m, other=0),
        mask=m,
    )


@triton.jit
def _concat_flat_b(
    b_ptr,
    o_ptr,
    n_b,
    A_LAST: tl.constexpr,
    B_LAST: tl.constexpr,
    ROW: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Generic fallback, b-half: flat 1-D b range -> columns [A_LAST, ROW)."""
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n_b
    k = tl.where(m, offs, 0)
    tl.store(
        o_ptr + (k // B_LAST) * ROW + A_LAST + (k % B_LAST),
        tl.load(b_ptr + k, mask=m, other=0),
        mask=m,
    )


def _target_rows(rows):
    """Select rows per program from the total row count."""
    if rows <= _ROWS_MED:
        return _G_SMALL if rows <= _ROWS_MED // 8 else _G_MED
    return _G_XLARGE if rows > _ROWS_LARGE else _G_LARGE


def concat_mla_absorb_q(a, b):
    a_last = a.shape[-1]
    b_last = b.shape[-1]
    row = a_last + b_last
    out = torch.empty(a.shape[:-1] + (row,), dtype=a.dtype, device=a.device)
    n_a = a.numel()
    if n_a == 0 or b.numel() == 0:
        return out

    pack = 8 // a.element_size()
    rows = n_a // a_last
    if (
        a.is_contiguous()
        and b.is_contiguous()
        and a.shape[:-1] == b.shape[:-1]
        and a_last % pack == 0
        and b_last % pack == 0
        and rows * (a_last + b_last) // pack < 0x7FFFFFFF
    ):
        a64c = a_last // pack
        b64c = b_last // pack
        o64c = a64c + b64c
        g = rows & -rows
        tgt = _target_rows(rows)
        if g > tgt:
            g = tgt
        _concat_rows_packed[(rows // g,)](
            a.view(torch.int64),
            b.view(torch.int64),
            out.view(torch.int64),
            A64=a64c,
            B64=b64c,
            ROW64=o64c,
            G=g,
            num_warps=8,
        )
        return out

    ao = out.reshape(-1, row)
    af = a.reshape(-1)
    bf = b.reshape(-1)
    _concat_flat_a[(triton.cdiv(n_a, _GEN_BLOCK),)](
        af,
        ao,
        n_a,
        A_LAST=a_last,
        ROW=row,
        BLOCK=_GEN_BLOCK,
        num_warps=4,
    )
    _concat_flat_b[(triton.cdiv(b.numel(), _GEN_BLOCK),)](
        bf,
        ao,
        b.numel(),
        A_LAST=a_last,
        B_LAST=b_last,
        ROW=row,
        BLOCK=_GEN_BLOCK,
        num_warps=4,
    )
    return out


__all__ = ["concat_mla_absorb_q"]
