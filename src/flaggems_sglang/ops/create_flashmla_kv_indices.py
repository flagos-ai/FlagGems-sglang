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
def _kv_indices_kernel(
    req_to_token_ptr,
    req_pool_indices_ptr,
    page_kernel_lens_ptr,
    kv_start_idx_ptr,
    out_ptr,
    max_context,
    out_stride,
    max_pages: tl.constexpr,
    page_size: tl.constexpr,
    HAS_START: tl.constexpr,
    WIDE_ADDR: tl.constexpr,
    BLOCK_P: tl.constexpr,
):
    pid = tl.program_id(0)
    NUM_PID_P: tl.constexpr = (max_pages + BLOCK_P - 1) // BLOCK_P
    pid_b = pid // NUM_PID_P
    pid_p = pid % NUM_PID_P

    pool = tl.load(req_pool_indices_ptr + pid_b)
    n = tl.load(page_kernel_lens_ptr + pid_b)
    if HAS_START:
        start = tl.load(kv_start_idx_ptr + pid_b)
    else:
        start = 0

    num_pages = (n + page_size - 1) // page_size
    page_offs = pid_p * BLOCK_P + tl.arange(0, BLOCK_P)
    valid = page_offs < num_pages
    if WIDE_ADDR:
        token_base = pool.to(tl.int64) * max_context + start
        slots = tl.load(
            req_to_token_ptr + token_base + page_offs.to(tl.int64) * page_size,
            mask=valid,
            other=0,
        )
    else:
        token_base = pool * max_context
        slots = tl.load(
            req_to_token_ptr + token_base + start + page_offs * page_size,
            mask=valid,
            other=0,
        )
    pages = slots // page_size
    tl.store(out_ptr + pid_b * out_stride + page_offs, pages, mask=valid)


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
    has_start = kv_start_idx is not None
    if not has_start:
        kv_start_idx = req_pool_indices

    max_addr = (
        (req_to_token.shape[0] - 1) * req_to_token.stride(0)
        + (req_to_token.shape[1] - 1) * req_to_token.stride(1)
        + page_size
    )
    wide_addr = max_addr > 0x7FFFFFFF

    BLOCK_P, num_warps, num_stages = 32, 1, 1
    grid = (bs * triton.cdiv(max_pages, BLOCK_P),)
    _kv_indices_kernel[grid](
        req_to_token,
        req_pool_indices,
        page_kernel_lens,
        kv_start_idx,
        kv_indices,
        req_to_token.stride(0),
        kv_indices.stride(0),
        max_pages=max_pages,
        page_size=page_size,
        HAS_START=has_start,
        WIDE_ADDR=wide_addr,
        BLOCK_P=BLOCK_P,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return kv_indices


__all__ = ["create_flashmla_kv_indices"]
