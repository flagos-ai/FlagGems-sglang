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

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_P": 32}, num_warps=1),
        triton.Config({"BLOCK_P": 64}, num_warps=1),
        triton.Config({"BLOCK_P": 64}, num_warps=2),
        triton.Config({"BLOCK_P": 128}, num_warps=2),
        triton.Config({"BLOCK_P": 128}, num_warps=4),
        triton.Config({"BLOCK_P": 256}, num_warps=4),
        triton.Config({"BLOCK_P": 256}, num_warps=8),
    ],
    key=[
        "bs",
        "max_pages",
        "page_size",
        "HAS_START",
        "USE_SHIFT",
        "WIDE_INDEX",
    ],
)
@triton.jit
def _create_flashmla_kv_indices_kernel(
    req_to_token_ptr,
    req_pool_indices_ptr,
    page_kernel_lens_ptr,
    kv_start_idx_ptr,
    kv_indices_ptr,
    out_ptr,
    bs,
    max_pages,
    page_size,
    rtt_stride,
    BLOCK_P: tl.constexpr,
    HAS_START: tl.constexpr,
    USE_SHIFT: tl.constexpr,
    LOG2_PAGE_SIZE: tl.constexpr,
    WIDE_INDEX: tl.constexpr,
):
    row = tl.program_id(0)
    pb = tl.program_id(1)

    pool = tl.load(req_pool_indices_ptr + row)
    n = tl.load(page_kernel_lens_ptr + row)
    if HAS_START:
        start = tl.load(kv_start_idx_ptr + row)
    else:
        start = 0

    num_pages = (n + page_size - 1) // page_size

    offs = pb * BLOCK_P + tl.arange(0, BLOCK_P)
    blk_end = pb * BLOCK_P + BLOCK_P

    if WIDE_INDEX:
        base = pool.to(tl.int64) * rtt_stride + start
        out_base = row.to(tl.int64) * max_pages
    else:
        base = pool * rtt_stride + start
        out_base = row * max_pages

    if (blk_end <= num_pages) & (blk_end <= max_pages):
        slots = tl.load(req_to_token_ptr + base + offs * page_size)
        if USE_SHIFT:
            res = slots >> LOG2_PAGE_SIZE
        else:
            res = slots // page_size
        tl.store(out_ptr + out_base + offs, res)
    else:
        in_range = offs < max_pages
        valid = in_range & (offs < num_pages)
        slots = tl.load(
            req_to_token_ptr + base + offs * page_size, mask=valid, other=0
        )
        if USE_SHIFT:
            new_vals = slots >> LOG2_PAGE_SIZE
        else:
            new_vals = slots // page_size
        old_vals = tl.load(
            kv_indices_ptr + out_base + offs, mask=in_range, other=0
        )
        res = tl.where(valid, new_vals, old_vals)
        tl.store(out_ptr + out_base + offs, res, mask=in_range)


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
    out = torch.empty_like(kv_indices, memory_format=torch.contiguous_format)
    if bs == 0 or max_pages == 0:
        out.copy_(kv_indices)
        return out

    has_start = kv_start_idx is not None
    start_tensor = kv_start_idx if has_start else page_kernel_lens

    page_size_i = int(page_size)
    use_shift = page_size_i > 0 and (page_size_i & (page_size_i - 1)) == 0
    log2_page_size = page_size_i.bit_length() - 1 if use_shift else 0

    wide_index = req_to_token.numel() >= (1 << 31)

    def grid(meta):
        return (bs, triton.cdiv(max_pages, meta["BLOCK_P"]))

    _create_flashmla_kv_indices_kernel[grid](
        req_to_token,
        req_pool_indices,
        page_kernel_lens,
        start_tensor,
        kv_indices,
        out,
        bs,
        max_pages,
        page_size_i,
        req_to_token.stride(0),
        HAS_START=has_start,
        USE_SHIFT=use_shift,
        LOG2_PAGE_SIZE=log2_page_size,
        WIDE_INDEX=wide_index,
    )
    return out


__all__ = ["create_flashmla_kv_indices"]
