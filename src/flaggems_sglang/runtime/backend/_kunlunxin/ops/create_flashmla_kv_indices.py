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

# Pages gathered per program.  Wide enough that a whole 16K-token context
# (256 pages) fits in a single program, which measured fastest: this operator
# is bound by per-program overhead rather than by the gather itself.
_BLOCK_P = 512
# Cap on launch warps.  The gather exposes little independent work per page, so
# wide launches only add overhead.
_MAX_NUM_WARPS = 4


def _pick_num_warps(block_p):
    return min(_MAX_NUM_WARPS, max(1, block_p // 128))


@triton.jit
def _flashmla_kv_indices_kernel(
    req_to_token_ptr,
    req_pool_indices_ptr,
    page_kernel_lens_ptr,
    kv_start_idx_ptr,
    kv_indices_ptr,
    max_context,
    width,
    HAS_KV_START: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    BLOCK_P: tl.constexpr,
):
    row = tl.program_id(0)
    blk = tl.program_id(1)

    n = tl.load(page_kernel_lens_ptr + row)
    num_pages = (n + (PAGE_SIZE - 1)) // PAGE_SIZE

    # Page indices for this block.  Only the pages below ``num_pages`` are
    # valid; both the gather and the store are masked by that, which also
    # bounds the offset inside the pool row.
    offs = blk * BLOCK_P + tl.arange(0, BLOCK_P)
    mask = offs < num_pages

    pool = tl.load(req_pool_indices_ptr + row)
    if HAS_KV_START:
        start = tl.load(kv_start_idx_ptr + row)
    else:
        start = 0

    # Boundary token slot of each page: stride == PAGE_SIZE elements.
    src = (
        req_to_token_ptr
        + pool.to(tl.int64) * max_context
        + start
        + offs * PAGE_SIZE
    )
    slot = tl.load(src, mask=mask, other=0)
    tl.store(kv_indices_ptr + row * width + offs, slot // PAGE_SIZE, mask=mask)


def create_flashmla_kv_indices(
    req_to_token,
    req_pool_indices,
    page_kernel_lens,
    kv_start_idx,
    kv_indices,
    page_size,
):
    num_rows = req_pool_indices.shape[0]
    width = kv_indices.shape[1]
    if num_rows == 0 or width == 0:
        return kv_indices

    # Fixed wide block, even for tables narrower than it: the surplus lanes are
    # masked off, and the smaller program count that buys back outweighs the
    # masked lanes (measured on narrow tables too).
    block_p = _BLOCK_P
    num_warps = _pick_num_warps(block_p)

    _flashmla_kv_indices_kernel[(num_rows, triton.cdiv(width, block_p))](
        req_to_token,
        req_pool_indices,
        page_kernel_lens,
        kv_start_idx if kv_start_idx is not None else req_pool_indices,
        kv_indices,
        req_to_token.shape[1],
        width,
        HAS_KV_START=kv_start_idx is not None,
        PAGE_SIZE=page_size,
        BLOCK_P=block_p,
        num_warps=num_warps,
    )
    return kv_indices


__all__ = ["create_flashmla_kv_indices"]
