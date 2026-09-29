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
    ROW: tl.constexpr,
    PW: tl.constexpr,
    LW: tl.constexpr,
    IW: tl.constexpr,
    SW: tl.constexpr,
    HAS_START: tl.constexpr,
    BLOCK: tl.constexpr,
    BS: tl.constexpr,
    NP: tl.constexpr,
    pool_ptr,
    lens_ptr,
    indptr_ptr,
    start_ptr,
    table_ptr,
    out_ptr,
):
    cols = tl.arange(0, BLOCK)
    for j in range(tl.program_id(0), BS, NP):
        n = tl.load(lens_ptr + j * LW)
        beg = tl.load(indptr_ptr + j * IW)
        base = tl.load(pool_ptr + j * PW) * ROW
        if HAS_START:
            base += tl.load(start_ptr + j * SW)
        for c in range(0, n, BLOCK):
            offs = c + cols
            m = offs < n
            tl.store(
                out_ptr + beg + offs,
                tl.load(table_ptr + base + offs, mask=m, other=0),
                mask=m,
            )


def _width(dtype):
    if dtype == torch.int64:
        return 2
    if dtype == torch.int32:
        return 1
    raise TypeError("index tensors must be int32 or int64")


def create_flashinfer_kv_indices(
    req_to_token,
    req_pool_indices,
    page_kernel_lens,
    kv_indptr,
    kv_start_idx,
    kv_indices,
):
    sdt = None if kv_start_idx is None else kv_start_idx.dtype
    bs = req_pool_indices.shape[0]
    np_ = bs // 2
    if np_ < 12:
        np_ = 12
    if np_ > 40:
        np_ = 40
    if np_ > bs:
        np_ = bs
    tail = (
        req_to_token.stride(0),
        _width(req_pool_indices.dtype),
        _width(page_kernel_lens.dtype),
        _width(kv_indptr.dtype),
        1 if sdt is None else _width(sdt),
        sdt is not None,
        4096,
        bs,
        np_,
    )
    pool = req_pool_indices.view(torch.int32)
    lens = page_kernel_lens.view(torch.int32)
    indptr = kv_indptr.view(torch.int32)
    start = indptr if kv_start_idx is None else kv_start_idx.view(torch.int32)
    _kv_indices_kernel[(np_,)](
        *tail, pool, lens, indptr, start, req_to_token, kv_indices, num_warps=1
    )
    return kv_indices


__all__ = ["create_flashinfer_kv_indices"]
