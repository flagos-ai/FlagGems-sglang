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

import triton
import triton.language as tl


@triton.jit
def _fill_page_table_kernel(
    req_to_token_ptr,
    req_pool_indices_ptr,
    cache_seqlens_ptr,
    page_table_ptr,
    max_pages,
    req_to_token_stride0: tl.constexpr,
    req_to_token_stride1: tl.constexpr,
    page_table_stride0: tl.constexpr,
    BLOCK_P: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    PAGE_SHIFT: tl.constexpr,
):
    req_id = tl.program_id(0)

    seq_len = tl.load(cache_seqlens_ptr + req_id)
    n_pages = tl.cdiv(seq_len, PAGE_SIZE)
    n_pages = tl.minimum(n_pages, max_pages)

    pool_i = tl.load(req_pool_indices_ptr + req_id).to(tl.int64)
    row_off = pool_i * req_to_token_stride0
    out_row = req_id.to(tl.int64) * page_table_stride0
    tok_stride = PAGE_SIZE * req_to_token_stride1

    n_iter = tl.cdiv(n_pages, BLOCK_P)
    for i in range(n_iter):
        offs = i * BLOCK_P + tl.arange(0, BLOCK_P)
        mask = offs < n_pages
        slots = tl.load(
            req_to_token_ptr + row_off + offs.to(tl.int64) * tok_stride,
            mask=mask,
            other=0,
        )
        tl.store(
            page_table_ptr + out_row + offs.to(tl.int64),
            (slots >> PAGE_SHIFT).to(tl.int32),
            mask=mask,
        )


def build_trtllm_mha_page_table(
    req_to_token, req_pool_indices, cache_seqlens, page_table, page_size
):
    bs = req_pool_indices.shape[0]
    max_pages = page_table.shape[1]
    if bs == 0 or max_pages == 0:
        return page_table

    page_size = int(page_size)
    BLOCK_P = 256

    _fill_page_table_kernel[(bs,)](
        req_to_token,
        req_pool_indices,
        cache_seqlens,
        page_table,
        max_pages,
        req_to_token.stride(0),
        req_to_token.stride(1),
        page_table.stride(0),
        BLOCK_P=BLOCK_P,
        PAGE_SIZE=page_size,
        PAGE_SHIFT=page_size.bit_length() - 1,
        num_warps=4,
    )
    return page_table


__all__ = ["build_trtllm_mha_page_table"]
