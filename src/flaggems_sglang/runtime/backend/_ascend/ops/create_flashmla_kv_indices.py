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
    req_to_token_ptr,  # [max_batch, max_context] token slots
    req_pool_indices_ptr,  # [bs]
    page_kernel_lens_ptr,  # [bs]
    kv_indices_ptr,  # [bs, max_pages], written in place
    page_size: tl.constexpr,
    max_pages: tl.constexpr,
    stride_req_to_token_b: tl.constexpr,
    stride_out_b: tl.constexpr,
    BLOCK_P: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_p = tl.program_id(1)

    lens = tl.load(page_kernel_lens_ptr + pid_b)
    num_pages = (lens + page_size - 1) // page_size
    num_pages = tl.minimum(num_pages, max_pages)

    if pid_p * BLOCK_P < num_pages:
        pool = tl.load(req_pool_indices_ptr + pid_b)
        offs_p = pid_p * BLOCK_P + tl.arange(0, BLOCK_P)
        in_bounds = offs_p < num_pages
        # Gather the token slot at each page boundary of this request.
        slots = tl.load(
            req_to_token_ptr
            + pool * stride_req_to_token_b
            + offs_p * page_size,
            mask=in_bounds,
            other=0,
        )
        tl.store(
            kv_indices_ptr + pid_b * stride_out_b + offs_p,
            slots // page_size,
            mask=in_bounds,
        )


@triton.jit
def _kv_indices_kernel_start(
    req_to_token_ptr,
    req_pool_indices_ptr,
    page_kernel_lens_ptr,
    kv_start_idx_ptr,  # [bs] per-request start offset
    kv_indices_ptr,
    page_size: tl.constexpr,
    max_pages: tl.constexpr,
    stride_req_to_token_b: tl.constexpr,
    stride_out_b: tl.constexpr,
    BLOCK_P: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_p = tl.program_id(1)

    lens = tl.load(page_kernel_lens_ptr + pid_b)
    start = tl.load(kv_start_idx_ptr + pid_b)
    num_pages = (lens + page_size - 1) // page_size
    num_pages = tl.minimum(num_pages, max_pages)

    if pid_p * BLOCK_P < num_pages:
        pool = tl.load(req_pool_indices_ptr + pid_b)
        offs_p = pid_p * BLOCK_P + tl.arange(0, BLOCK_P)
        in_bounds = offs_p < num_pages
        slots = tl.load(
            req_to_token_ptr
            + pool * stride_req_to_token_b
            + start
            + offs_p * page_size,
            mask=in_bounds,
            other=0,
        )
        tl.store(
            kv_indices_ptr + pid_b * stride_out_b + offs_p,
            slots // page_size,
            mask=in_bounds,
        )


# Stream captured once at import, exactly like ops/add3.py; benchmarks run on
# the default stream, so re-querying per launch would only add host overhead.


def _pick(bs):
    """Launch config per batch size (device-sweep measured).

    The gather is latency-bound, so fewer-but-wider tiles win: one program
    covers a large page range and the device hides per-lane gather latency by
    overlapping programs across the batch dimension.  Splitting the page range
    finer (small BLOCK_P) multiplies program count without adding useful
    parallelism and measurably hurts on this device.  Tiny batches keep one
    narrow tile because they are launch-bound, not throughput-bound.
    """
    if bs <= 4:
        return 64, 1
    if bs <= 16:
        return 128, 4
    return 256, 4


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

    BLOCK_P, W = _pick(bs)
    stride_b = req_to_token.stride(0)
    stride_out = kv_indices.stride(0)
    has_start = kv_start_idx is not None

    grid1 = (max_pages + BLOCK_P - 1) // BLOCK_P
    # Standard kernel[grid](...) launch: Triton's own JIT cache handles
    # compilation reuse for every (page_size, shape, config) key.
    if has_start:
        _kv_indices_kernel_start[(bs, grid1)](
            req_to_token,
            req_pool_indices,
            page_kernel_lens,
            kv_start_idx,
            kv_indices,
            page_size,
            max_pages,
            stride_b,
            stride_out,
            BLOCK_P=BLOCK_P,
            num_warps=W,
        )
    else:
        _kv_indices_kernel[(bs, grid1)](
            req_to_token,
            req_pool_indices,
            page_kernel_lens,
            kv_indices,
            page_size,
            max_pages,
            stride_b,
            stride_out,
            BLOCK_P=BLOCK_P,
            num_warps=W,
        )
    return kv_indices


__all__ = ["create_flashmla_kv_indices"]
