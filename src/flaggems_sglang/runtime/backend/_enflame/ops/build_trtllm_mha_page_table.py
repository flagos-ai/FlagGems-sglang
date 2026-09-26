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

__all__ = ["build_trtllm_mha_page_table"]


@triton.jit(
    do_not_specialize_on_alignment=[
        "table_ptr",
        "tok_ptr",
        "pool_ptr",
        "seq_ptr",
    ]
)
def _page_rows_kernel(
    table_ptr,
    tok_ptr,
    pool_ptr,
    seq_ptr,
    PAGE_SIZE: tl.constexpr,
    SHIFT: tl.constexpr,
    TOK_STRIDE: tl.constexpr,
    BLOCK_P: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_P)
    seq = tl.load(seq_ptr + row)
    pool = tl.load(pool_ptr + row)
    n_pages = (seq + PAGE_SIZE - 1) // PAGE_SIZE
    fresh = offs < n_pages
    tok = tl.load(
        tok_ptr + pool * TOK_STRIDE + offs * PAGE_SIZE, mask=fresh, other=0
    )
    if SHIFT >= 0:
        page = tok >> SHIFT
    else:
        page = tok // PAGE_SIZE
    tl.store(
        table_ptr + row * BLOCK_P + offs,
        page.to(table_ptr.dtype.element_ty),
        mask=fresh,
    )


@triton.jit
def _page_blocks_kernel(
    table_ptr,
    tok_ptr,
    pool_ptr,
    seq_ptr,
    PAGE_SIZE: tl.constexpr,
    SHIFT: tl.constexpr,
    MAX_PAGES: tl.constexpr,
    TOK_STRIDE: tl.constexpr,
    TAB_STRIDE: tl.constexpr,
    BLOCK_P: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.program_id(1) * BLOCK_P + tl.arange(0, BLOCK_P)
    seq = tl.load(seq_ptr + row)
    pool = tl.load(pool_ptr + row)
    n_pages = (seq + PAGE_SIZE - 1) // PAGE_SIZE
    fresh = offs < tl.minimum(n_pages, MAX_PAGES)
    tok = tl.load(
        tok_ptr
        + pool * TOK_STRIDE
        + tl.minimum(offs, MAX_PAGES - 1) * PAGE_SIZE,
        mask=fresh,
        other=0,
    )
    if SHIFT >= 0:
        page = tok >> SHIFT
    else:
        page = tok // PAGE_SIZE
    tl.store(
        table_ptr + row * TAB_STRIDE + offs,
        page.to(table_ptr.dtype.element_ty),
        mask=fresh,
    )


def build_trtllm_mha_page_table(
    req_to_token, req_pool_indices, cache_seqlens, page_table, page_size
):
    if req_to_token.dtype != torch.int32:
        req_to_token = req_to_token.to(torch.int32)
    if req_pool_indices.dtype != torch.int32:
        req_pool_indices = req_pool_indices.to(torch.int32)
    if cache_seqlens.dtype != torch.int32:
        cache_seqlens = cache_seqlens.to(torch.int32)
    bs, max_pages = page_table.shape
    shift = page_size.bit_length() - 1
    if (1 << shift) != page_size:
        shift = -1
    stride = req_to_token.stride(0)
    if (
        shift >= 0
        and max_pages & (max_pages - 1) == 0
        and max_pages <= 1024
        and page_table.is_contiguous()
    ):
        tail = (page_size, shift, stride, max_pages)
        _page_rows_kernel[(bs,)](
            page_table,
            req_to_token,
            req_pool_indices,
            cache_seqlens,
            *tail,
            num_warps=2,
            num_stages=1,
        )
        return page_table
    block = 1 << (max_pages - 1).bit_length()
    if block > 1024:
        block = 1024
    elif block < 16:
        block = 16
    _page_blocks_kernel[(bs, (max_pages + block - 1) // block)](
        page_table,
        req_to_token,
        req_pool_indices,
        cache_seqlens,
        page_size,
        shift,
        max_pages,
        stride,
        page_table.stride(0),
        block,
    )
    return page_table
