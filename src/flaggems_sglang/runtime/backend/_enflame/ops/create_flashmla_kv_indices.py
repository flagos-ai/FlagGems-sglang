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
def _flashmla_block_table_row(
    req_to_token_ptr,
    req_pool_indices_ptr,
    page_kernel_lens_ptr,
    kv_start_idx_ptr,
    kv_indices_ptr,
    i,
    REQ_TO_TOKEN_STRIDE: tl.constexpr,
    KV_INDICES_STRIDE: tl.constexpr,
    MAX_PAGES: tl.constexpr,
    page_size: tl.constexpr,
    BLOCK_P: tl.constexpr,
    HAS_START: tl.constexpr,
):
    pool = tl.load(req_pool_indices_ptr + i).to(tl.int32)
    n = tl.load(page_kernel_lens_ptr + i).to(tl.int32)
    if HAS_START:
        start = tl.load(kv_start_idx_ptr + i).to(tl.int32)
    else:
        start = 0
    num_pages = (n + page_size - 1) // page_size
    row = req_to_token_ptr + pool * REQ_TO_TOKEN_STRIDE + start
    out_row = kv_indices_ptr + i * KV_INDICES_STRIDE

    for seg in tl.range(0, tl.cdiv(MAX_PAGES, BLOCK_P)):
        p = seg * BLOCK_P + tl.arange(0, BLOCK_P)
        m = p < num_pages
        slots = tl.load(row + p * page_size, mask=m, other=0).to(tl.int32)
        tl.store(out_row + p, slots // page_size, mask=m)


@triton.jit
def _flashmla_block_table_kernel(
    req_to_token_ptr,
    req_pool_indices_ptr,
    page_kernel_lens_ptr,
    kv_start_idx_ptr,
    kv_indices_ptr,
    REQ_TO_TOKEN_STRIDE: tl.constexpr,
    KV_INDICES_STRIDE: tl.constexpr,
    MAX_PAGES: tl.constexpr,
    page_size: tl.constexpr,
    BLOCK_P: tl.constexpr,
    HAS_START: tl.constexpr,
):
    i = tl.program_id(0)
    _flashmla_block_table_row(
        req_to_token_ptr,
        req_pool_indices_ptr,
        page_kernel_lens_ptr,
        kv_start_idx_ptr,
        kv_indices_ptr,
        i,
        REQ_TO_TOKEN_STRIDE=REQ_TO_TOKEN_STRIDE,
        KV_INDICES_STRIDE=KV_INDICES_STRIDE,
        MAX_PAGES=MAX_PAGES,
        page_size=page_size,
        BLOCK_P=BLOCK_P,
        HAS_START=HAS_START,
    )


@triton.jit
def _flashmla_block_table_kernel_pair(
    req_to_token_ptr,
    req_pool_indices_ptr,
    page_kernel_lens_ptr,
    kv_start_idx_ptr,
    kv_indices_ptr,
    bs,
    REQ_TO_TOKEN_STRIDE: tl.constexpr,
    KV_INDICES_STRIDE: tl.constexpr,
    MAX_PAGES: tl.constexpr,
    page_size: tl.constexpr,
    BLOCK_P: tl.constexpr,
    HAS_START: tl.constexpr,
):
    base = tl.program_id(0) * 2
    for j in tl.static_range(2):
        i = base + j
        if i < bs:
            _flashmla_block_table_row(
                req_to_token_ptr,
                req_pool_indices_ptr,
                page_kernel_lens_ptr,
                kv_start_idx_ptr,
                kv_indices_ptr,
                i,
                REQ_TO_TOKEN_STRIDE=REQ_TO_TOKEN_STRIDE,
                KV_INDICES_STRIDE=KV_INDICES_STRIDE,
                MAX_PAGES=MAX_PAGES,
                page_size=page_size,
                BLOCK_P=BLOCK_P,
                HAS_START=HAS_START,
            )


def create_flashmla_kv_indices(
    req_to_token,
    req_pool_indices,
    page_kernel_lens,
    kv_start_idx,
    kv_indices,
    page_size,
):
    bs = req_pool_indices.shape[0]
    max_pages = kv_indices.shape[1]
    if bs == 0 or max_pages == 0:
        return kv_indices

    has_start = kv_start_idx is not None
    start_arg = kv_start_idx if has_start else req_pool_indices

    RT_STRIDE = req_to_token.stride(0)
    OUT_STRIDE = kv_indices.stride(0)

    BLOCK_P = 256
    if bs >= 64:
        _flashmla_block_table_kernel_pair[(triton.cdiv(bs, 2),)](
            req_to_token,
            req_pool_indices,
            page_kernel_lens,
            start_arg,
            kv_indices,
            bs,
            REQ_TO_TOKEN_STRIDE=RT_STRIDE,
            KV_INDICES_STRIDE=OUT_STRIDE,
            MAX_PAGES=max_pages,
            page_size=page_size,
            BLOCK_P=BLOCK_P,
            HAS_START=has_start,
            num_warps=1,
        )
    else:
        num_warps = 2 if bs <= 8 else 1
        _flashmla_block_table_kernel[(bs,)](
            req_to_token,
            req_pool_indices,
            page_kernel_lens,
            start_arg,
            kv_indices,
            REQ_TO_TOKEN_STRIDE=RT_STRIDE,
            KV_INDICES_STRIDE=OUT_STRIDE,
            MAX_PAGES=max_pages,
            page_size=page_size,
            BLOCK_P=BLOCK_P,
            HAS_START=has_start,
            num_warps=num_warps,
        )
    return kv_indices


__all__ = ["create_flashmla_kv_indices"]
