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
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

import torch
import triton
import triton.language as tl


@triton.jit
def _kv_indices_kernel(
    pool_ptr,
    lens_ptr,
    indptr_ptr,
    start_ptr,
    table_ptr,
    out_ptr,
    BS: tl.constexpr,
    ROW: tl.constexpr,
    TS: tl.constexpr,
    PS: tl.constexpr,
    LS: tl.constexpr,
    IS: tl.constexpr,
    SS: tl.constexpr,
    HAS_START: tl.constexpr,
    BLOCK: tl.constexpr,
    TB: tl.constexpr,
    NP: tl.constexpr,
):
    cols_b = tl.arange(0, BLOCK)
    cols_t = tl.arange(0, TB)
    for j in range(tl.program_id(0), BS, NP):
        n = tl.load(lens_ptr + j * LS)
        beg = tl.load(indptr_ptr + j * IS)
        base = tl.load(pool_ptr + j * PS) * ROW
        if HAS_START:
            base += tl.load(start_ptr + j * SS)
        last = tl.maximum(n - 1, 0)
        nb = tl.cdiv(n, BLOCK)
        full_end = (nb - 1) * BLOCK
        for c in range(0, full_end, BLOCK):
            offs = c + cols_b
            tl.store(
                out_ptr + beg + offs, tl.load(table_ptr + (base + offs) * TS)
            )
        tail_start = tl.maximum(full_end, 0)
        for c in range(tail_start, n, TB):
            offs = c + cols_t
            safe = tl.minimum(tl.maximum(offs, 0), last)
            tl.store(
                out_ptr + beg + offs,
                tl.load(table_ptr + (base + safe) * TS),
                mask=offs < n,
            )


def _int64_word_stride():
    return (
        2 if os.environ.get("TORCH_GCU_ENABLE_INT64_AND_UINT64") == "1" else 1
    )


def _index(t):
    if t.dtype == torch.int64:
        return t.contiguous().view(torch.int32), _int64_word_stride()
    return t, t.stride(0)


def create_flashinfer_kv_indices(
    req_to_token,
    req_pool_indices,
    page_kernel_lens,
    kv_indptr,
    kv_start_idx,
    kv_indices,
):
    pool, ps = _index(req_pool_indices)
    lens, ls = _index(page_kernel_lens)
    indptr, is_ = _index(kv_indptr)
    table = req_to_token.contiguous()
    row = table.stride(0)
    if table.dtype == torch.int64:
        table = table.view(torch.int32)
        ts = _int64_word_stride()
    else:
        ts = 1
    bs = req_pool_indices.shape[0]
    if bs == 0:
        return kv_indices
    if kv_start_idx is None:
        start, ss, has_start = indptr, 1, False
    else:
        start, ss = _index(kv_start_idx)
        has_start = True
    np_ = bs if bs <= 48 else 48
    if bs <= 2:
        block, tb, warps = 2048, 256, 1
    elif bs <= 16:
        block, tb, warps = 2048, 2048, 2
    else:
        block, tb, warps = 1024, 1024, 1
    _kv_indices_kernel[(np_,)](
        pool,
        lens,
        indptr,
        start,
        table,
        kv_indices,
        bs,
        row,
        ts,
        ps,
        ls,
        is_,
        ss,
        has_start,
        block,
        tb,
        np_,
        num_warps=warps,
    )
    return kv_indices


__all__ = ["create_flashinfer_kv_indices"]
