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
    ROW: tl.constexpr,
    PS: tl.constexpr,
    LS: tl.constexpr,
    IS: tl.constexpr,
    SS: tl.constexpr,
    HAS_START: tl.constexpr,
    BLOCK: tl.constexpr,
    CH: tl.constexpr,
):
    j = tl.program_id(0)
    k = tl.program_id(1)
    cols = tl.arange(0, BLOCK)
    n = tl.load(lens_ptr + j * LS)
    beg = tl.load(indptr_ptr + j * IS)
    base = tl.load(pool_ptr + j * PS) * ROW
    if HAS_START:
        base += tl.load(start_ptr + j * SS)
    for c in range(k * BLOCK, n, CH * BLOCK):
        offs = c + cols
        m = offs < n
        tl.store(
            out_ptr + beg + offs,
            tl.load(table_ptr + base + offs, mask=m, other=0),
            mask=m,
        )


def _index(t):
    if t.dtype == torch.int64:
        return t.contiguous().view(torch.int32), 2
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
    bs = req_pool_indices.shape[0]
    n = kv_indices.shape[0]
    row = req_to_token.stride(0)
    span = row if row < n else n
    ch = -(-span // 512)
    avg = -(-(-(-n // bs)) // 512)
    if avg < 1:
        avg = 1
    if avg < ch:
        ch = avg
    if kv_start_idx is None:
        start, ss, has_start = indptr, 1, False
    else:
        start, ss = _index(kv_start_idx)
        has_start = True
    _kv_indices_kernel[(bs, ch)](
        pool,
        lens,
        indptr,
        start,
        req_to_token,
        kv_indices,
        row,
        ps,
        ls,
        is_,
        ss,
        has_start,
        512,
        ch,
        num_warps=8,
    )
    return kv_indices


__all__ = ["create_flashinfer_kv_indices"]
