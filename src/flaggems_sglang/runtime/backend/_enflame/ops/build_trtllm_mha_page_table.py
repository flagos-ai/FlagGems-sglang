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
    tile = tl.program_id(1)

    seq_len = tl.load(cache_seqlens_ptr + req_id)
    n_pages = (seq_len + PAGE_SIZE - 1) // PAGE_SIZE
    n_pages = tl.where(n_pages < max_pages, n_pages, max_pages)

    offs = tile * BLOCK_P + tl.arange(0, BLOCK_P)
    mask = offs < n_pages

    pool_i = tl.load(req_pool_indices_ptr + req_id)
    tok_offs = pool_i * req_to_token_stride0 + offs * (
        PAGE_SIZE * req_to_token_stride1
    )
    slots = tl.load(req_to_token_ptr + tok_offs, mask=mask, other=0)
    tl.store(
        page_table_ptr + req_id * page_table_stride0 + offs,
        (slots >> PAGE_SHIFT).to(tl.int32),
        mask=mask,
    )


def _next_pow2(n):
    p = 1
    while p < n:
        p *= 2
    return p


def build_trtllm_mha_page_table(
    req_to_token, req_pool_indices, cache_seqlens, page_table, page_size
):
    bs = req_pool_indices.shape[0]
    max_pages = page_table.shape[1]
    if bs == 0 or max_pages == 0:
        return page_table

    page_size = int(page_size)
    BLOCK_P = _next_pow2(max_pages)
    if BLOCK_P < 64:
        BLOCK_P = 64
    if BLOCK_P > 1024:
        BLOCK_P = 1024
    num_tiles = triton.cdiv(max_pages, BLOCK_P)

    _fill_page_table_kernel[(bs, num_tiles)](
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
    )
    return page_table


__all__ = ["build_trtllm_mha_page_table"]
