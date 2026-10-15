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
def _flashmla_kv_indices_kernel(
    req_to_token_ptr,
    req_pool_indices_ptr,
    page_kernel_lens_ptr,
    kv_start_idx_ptr,
    kv_indices_ptr,
    rt_row_stride,
    out_row_stride,
    HAS_KV_START: tl.constexpr,
    page_size: tl.constexpr,
    BLOCK_P: tl.constexpr,
):
    i = tl.program_id(0)
    j = tl.program_id(1)

    pool = tl.load(req_pool_indices_ptr + i)
    seq_len = tl.load(page_kernel_lens_ptr + i)
    if HAS_KV_START:
        start = tl.load(kv_start_idx_ptr + i)
    else:
        start = 0

    num_pages = (seq_len + page_size - 1) // page_size

    pages = j * BLOCK_P + tl.arange(0, BLOCK_P)
    mask = pages < num_pages
    token_slots = tl.load(
        req_to_token_ptr + pool * rt_row_stride + start + pages * page_size,
        mask=mask,
        other=0,
    )
    tl.store(
        kv_indices_ptr + i * out_row_stride + pages,
        token_slots // page_size,
        mask=mask,
    )


def _block_p(max_pages):
    return max(64, triton.next_power_of_2(triton.cdiv(max_pages, 4)))


def create_flashmla_kv_indices(
    req_to_token,
    req_pool_indices,
    page_kernel_lens,
    kv_start_idx,
    kv_indices,
    page_size,
):
    bs = req_pool_indices.shape[0]
    if bs == 0:
        return kv_indices
    max_pages = kv_indices.shape[1]

    block_p = _block_p(max_pages)
    grid = (bs, triton.cdiv(max_pages, block_p))
    _flashmla_kv_indices_kernel[grid](
        req_to_token,
        req_pool_indices,
        page_kernel_lens,
        kv_start_idx,
        kv_indices,
        req_to_token.stride(0),
        kv_indices.stride(0),
        kv_start_idx is not None,
        page_size,
        block_p,
        num_warps=1,
    )
    return kv_indices


__all__ = ["create_flashmla_kv_indices"]
