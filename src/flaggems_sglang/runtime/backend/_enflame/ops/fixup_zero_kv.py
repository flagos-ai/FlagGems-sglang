# Copyright 2026, The FlagOS Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import triton
import triton.language as tl

_MAX_PROGRAMS = 48
_HOIST_MAX_BATCH = 128
_I32_MAX_NUMEL = 2**31


@triton.jit
def _fixup_zero_kv_pair_kernel(
    out_ptr,
    lse_ptr,
    kv_lens_ptr,
    cum_ptr,
    batch,
    ROW_SIZE: tl.constexpr,
    LSE_ROW: tl.constexpr,
    OUT_BLOCK: tl.constexpr,
    LSE_BLOCK: tl.constexpr,
    IDX64: tl.constexpr,
):
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    if IDX64:
        ooff = tl.arange(0, OUT_BLOCK).to(tl.int64)
        loff = tl.arange(0, LSE_BLOCK).to(tl.int64)
    else:
        ooff = tl.arange(0, OUT_BLOCK)
        loff = tl.arange(0, LSE_BLOCK)
    ozeros = tl.zeros((OUT_BLOCK,), dtype=out_ptr.dtype.element_ty)
    linf = tl.full((LSE_BLOCK,), float("-inf"), dtype=lse_ptr.dtype.element_ty)

    for b in range(pid, batch, 2 * nprog):
        b2 = b + nprog
        m2 = b2 < batch
        kv0 = tl.load(kv_lens_ptr + b)
        kv1 = tl.load(kv_lens_ptr + b2, mask=m2, other=1)
        if IDX64:
            beg0 = tl.load(cum_ptr + b).to(tl.int64)
            end0 = tl.load(cum_ptr + b + 1).to(tl.int64)
            beg1 = tl.load(cum_ptr + b2, mask=m2, other=0).to(tl.int64)
            end1 = tl.load(cum_ptr + b2 + 1, mask=m2, other=0).to(tl.int64)
        else:
            beg0 = tl.load(cum_ptr + b)
            end0 = tl.load(cum_ptr + b + 1)
            beg1 = tl.load(cum_ptr + b2, mask=m2, other=0)
            end1 = tl.load(cum_ptr + b2 + 1, mask=m2, other=0)
        if kv0 == 0:
            p = beg0 * ROW_SIZE
            e = end0 * ROW_SIZE
            while p + OUT_BLOCK <= e:
                tl.store(out_ptr + p + ooff, ozeros)
                p += OUT_BLOCK
            if p < e:
                tl.store(out_ptr + p + ooff, ozeros, mask=p + ooff < e)
            q = beg0 * LSE_ROW
            qe = end0 * LSE_ROW
            while q + LSE_BLOCK <= qe:
                tl.store(lse_ptr + q + loff, linf)
                q += LSE_BLOCK
            if q < qe:
                tl.store(lse_ptr + q + loff, linf, mask=q + loff < qe)
        if kv1 == 0:
            p = beg1 * ROW_SIZE
            e = end1 * ROW_SIZE
            while p + OUT_BLOCK <= e:
                tl.store(out_ptr + p + ooff, ozeros)
                p += OUT_BLOCK
            if p < e:
                tl.store(out_ptr + p + ooff, ozeros, mask=p + ooff < e)
            q = beg1 * LSE_ROW
            qe = end1 * LSE_ROW
            while q + LSE_BLOCK <= qe:
                tl.store(lse_ptr + q + loff, linf)
                q += LSE_BLOCK
            if q < qe:
                tl.store(lse_ptr + q + loff, linf, mask=q + loff < qe)


@triton.jit
def _fixup_zero_kv_pipe_kernel(
    out_ptr,
    lse_ptr,
    kv_lens_ptr,
    cum_ptr,
    batch,
    ROW_SIZE: tl.constexpr,
    LSE_ROW: tl.constexpr,
    OUT_BLOCK: tl.constexpr,
    LSE_BLOCK: tl.constexpr,
    IDX64: tl.constexpr,
):
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    if IDX64:
        ooff = tl.arange(0, OUT_BLOCK).to(tl.int64)
        loff = tl.arange(0, LSE_BLOCK).to(tl.int64)
    else:
        ooff = tl.arange(0, OUT_BLOCK)
        loff = tl.arange(0, LSE_BLOCK)
    ozeros = tl.zeros((OUT_BLOCK,), dtype=out_ptr.dtype.element_ty)
    linf = tl.full((LSE_BLOCK,), float("-inf"), dtype=lse_ptr.dtype.element_ty)
    stride = 2 * nprog
    b = pid
    kA0 = tl.load(kv_lens_ptr + b, mask=b < batch, other=1)
    kA1 = tl.load(kv_lens_ptr + b + nprog, mask=b + nprog < batch, other=1)
    j = b + stride
    kB0 = tl.load(kv_lens_ptr + j, mask=j < batch, other=1)
    kB1 = tl.load(kv_lens_ptr + j + nprog, mask=j + nprog < batch, other=1)
    while b < batch:
        b2 = b + nprog
        m2 = b2 < batch
        j = b + 2 * stride
        jm = j < batch
        jm2 = j + nprog < batch
        kN0 = tl.load(kv_lens_ptr + j, mask=jm, other=1)
        kN1 = tl.load(kv_lens_ptr + j + nprog, mask=jm2, other=1)
        if kA0 == 0:
            if IDX64:
                beg0 = tl.load(cum_ptr + b).to(tl.int64)
                end0 = tl.load(cum_ptr + b + 1).to(tl.int64)
            else:
                beg0 = tl.load(cum_ptr + b)
                end0 = tl.load(cum_ptr + b + 1)
            p = beg0 * ROW_SIZE
            e = end0 * ROW_SIZE
            while p + OUT_BLOCK <= e:
                tl.store(out_ptr + p + ooff, ozeros)
                p += OUT_BLOCK
            if p < e:
                tl.store(out_ptr + p + ooff, ozeros, mask=p + ooff < e)
            q = beg0 * LSE_ROW
            qe = end0 * LSE_ROW
            while q + LSE_BLOCK <= qe:
                tl.store(lse_ptr + q + loff, linf)
                q += LSE_BLOCK
            if q < qe:
                tl.store(lse_ptr + q + loff, linf, mask=q + loff < qe)
        if kA1 == 0:
            if IDX64:
                beg1 = tl.load(cum_ptr + b2, mask=m2, other=0).to(tl.int64)
                end1 = tl.load(cum_ptr + b2 + 1, mask=m2, other=0).to(tl.int64)
            else:
                beg1 = tl.load(cum_ptr + b2, mask=m2, other=0)
                end1 = tl.load(cum_ptr + b2 + 1, mask=m2, other=0)
            p = beg1 * ROW_SIZE
            e = end1 * ROW_SIZE
            while p + OUT_BLOCK <= e:
                tl.store(out_ptr + p + ooff, ozeros)
                p += OUT_BLOCK
            if p < e:
                tl.store(out_ptr + p + ooff, ozeros, mask=p + ooff < e)
            q = beg1 * LSE_ROW
            qe = end1 * LSE_ROW
            while q + LSE_BLOCK <= qe:
                tl.store(lse_ptr + q + loff, linf)
                q += LSE_BLOCK
            if q < qe:
                tl.store(lse_ptr + q + loff, linf, mask=q + loff < qe)
        kA0 = kB0
        kA1 = kB1
        kB0 = kN0
        kB1 = kN1
        b += stride


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
    max_q = max(1, int(max_seq_len))

    out_block = _next_pow2(max(2, row_size))
    lse_block = min(_next_pow2(max(2, max_q * lse_row)), 2048)
    nprog = min(_MAX_PROGRAMS, batch_size)
    idx64 = out.numel() >= _I32_MAX_NUMEL or lse.numel() >= _I32_MAX_NUMEL

    if batch_size <= _HOIST_MAX_BATCH:
        _fixup_zero_kv_pair_kernel[(nprog,)](
            out,
            lse,
            kv_lens,
            cum_seq_lens,
            batch_size,
            ROW_SIZE=row_size,
            LSE_ROW=lse_row,
            OUT_BLOCK=out_block,
            LSE_BLOCK=lse_block,
            IDX64=idx64,
            num_warps=2,
        )
    else:
        _fixup_zero_kv_pipe_kernel[(nprog,)](
            out,
            lse,
            kv_lens,
            cum_seq_lens,
            batch_size,
            ROW_SIZE=row_size,
            LSE_ROW=lse_row,
            OUT_BLOCK=out_block,
            LSE_BLOCK=lse_block,
            IDX64=idx64,
            num_warps=1,
        )
    return out, lse


__all__ = ["fixup_zero_kv"]
