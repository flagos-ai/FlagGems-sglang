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

import triton
import triton.language as tl

_PLAN_SMALL = (6, 8192, 256, 2)
_NPROG_TINY = 4
_NPROG_MID = 8
_NPROG_BIG = 20
_NPROG_HUGE = 16
_OUT_BLOCK = 8192
_LSE_BLOCK = 256


@triton.jit
def _fixup_zero_kv_small(
    out_ptr,
    lse_ptr,
    kv_lens_ptr,
    cum_ptr,
    batch,
    ROW_SIZE: tl.constexpr,
    LSE_ROW: tl.constexpr,
    OUT_BLOCK: tl.constexpr,
    LSE_BLOCK: tl.constexpr,
    NPROG: tl.constexpr,
):
    pid = tl.program_id(0)
    ooff = tl.arange(0, OUT_BLOCK).to(tl.int64)
    loff = tl.arange(0, LSE_BLOCK).to(tl.int64)
    ozeros = tl.zeros((OUT_BLOCK,), dtype=out_ptr.dtype.element_ty)
    linf = tl.full((LSE_BLOCK,), float("-inf"), dtype=lse_ptr.dtype.element_ty)

    for b in range(pid, batch, NPROG):
        kv = tl.load(kv_lens_ptr + b)
        if kv == 0:
            beg = tl.load(cum_ptr + b).to(tl.int64)
            end = tl.load(cum_ptr + b + 1).to(tl.int64)

            p = beg * ROW_SIZE
            e = end * ROW_SIZE
            while p < e:
                tl.store(out_ptr + p + ooff, ozeros, mask=p + ooff < e)
                p += OUT_BLOCK

            q = beg * LSE_ROW
            qe = end * LSE_ROW
            while q < qe:
                tl.store(lse_ptr + q + loff, linf, mask=q + loff < qe)
                q += LSE_BLOCK


@triton.jit
def _fixup_zero_kv_window(
    out_ptr,
    lse_ptr,
    kv_lens_ptr,
    cum_ptr,
    batch,
    b0,
    ROW_SIZE: tl.constexpr,
    LSE_ROW: tl.constexpr,
    OUT_BLOCK: tl.constexpr,
    LSE_BLOCK: tl.constexpr,
    KB: tl.constexpr,
):
    pid = tl.program_id(0)
    bstart = pid * b0

    idx = tl.arange(0, KB)
    bidx = bstart + idx
    inb = bidx < batch
    kvec = tl.load(kv_lens_ptr + bidx, mask=inb, other=1)
    zmask = (kvec == 0) & inb
    cvec = tl.load(cum_ptr + bidx, mask=inb, other=0)
    cvec2 = tl.load(cum_ptr + bidx + 1, mask=inb, other=0)
    nz = tl.sum(zmask.to(tl.int32), 0)
    zpref = tl.cumsum(zmask.to(tl.int32), 0)

    ooff = tl.arange(0, OUT_BLOCK).to(tl.int64)
    loff = tl.arange(0, LSE_BLOCK).to(tl.int64)
    ozeros = tl.zeros((OUT_BLOCK,), dtype=out_ptr.dtype.element_ty)
    linf = tl.full((LSE_BLOCK,), float("-inf"), dtype=lse_ptr.dtype.element_ty)

    for r in range(nz):
        sel = zmask & (zpref == r + 1)
        beg = tl.sum(tl.where(sel, cvec, 0)).to(tl.int64)
        end = tl.sum(tl.where(sel, cvec2, 0)).to(tl.int64)

        p = beg * ROW_SIZE
        e = end * ROW_SIZE
        while p <= e - OUT_BLOCK:
            tl.store(out_ptr + p + ooff, ozeros)
            p += OUT_BLOCK
        if p < e:
            tl.store(out_ptr + p + ooff, ozeros, mask=p + ooff < e)

        q = beg * LSE_ROW
        qe = end * LSE_ROW
        while q <= qe - LSE_BLOCK:
            tl.store(lse_ptr + q + loff, linf)
            q += LSE_BLOCK
        if q < qe:
            tl.store(lse_ptr + q + loff, linf, mask=q + loff < qe)


def _next_pow2(x):
    n = 1
    while n < x:
        n *= 2
    return n


def fixup_zero_kv(out, lse, kv_lens, cum_seq_lens, max_seq_len):
    batch_size = kv_lens.shape[0]
    if batch_size == 0 or out.shape[0] == 0:
        return out, lse

    row_size = out.shape[1] * out.shape[2]
    lse_row = lse.shape[1]

    if batch_size <= 2:
        nprog, out_block, lse_block, num_warps = _PLAN_SMALL
        _fixup_zero_kv_small[(min(nprog, batch_size),)](
            out,
            lse,
            kv_lens,
            cum_seq_lens,
            batch_size,
            ROW_SIZE=row_size,
            LSE_ROW=lse_row,
            OUT_BLOCK=out_block,
            LSE_BLOCK=lse_block,
            NPROG=nprog,
            num_warps=num_warps,
        )
        return out, lse

    if batch_size <= 32:
        nprog = _NPROG_TINY
    elif batch_size <= 128:
        nprog = _NPROG_MID
    elif batch_size <= 1024:
        nprog = _NPROG_BIG
    else:
        nprog = _NPROG_HUGE

    b0 = (batch_size + nprog - 1) // nprog
    _fixup_zero_kv_window[(nprog,)](
        out,
        lse,
        kv_lens,
        cum_seq_lens,
        batch_size,
        b0,
        ROW_SIZE=row_size,
        LSE_ROW=lse_row,
        OUT_BLOCK=_OUT_BLOCK,
        LSE_BLOCK=_LSE_BLOCK,
        KB=_next_pow2(b0),
        num_warps=1,
    )
    return out, lse


__all__ = ["fixup_zero_kv"]
