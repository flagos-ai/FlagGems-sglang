# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""One native contiguous cache window per request, with exact integer records.

The CANN selector receives one request-row index, not one index per token.
Triton's metadata-only reinterpret wrapper supplies the FP32 pointer ABI;
there is no device pointer cast, host Tensor.view or payload float arithmetic.
"""

import torch
import triton
import triton.language as tl
import triton.language.extra.cann.extension as al


@triton.jit
def _assign_native_request_window_kernel(
    req_pool_indices,
    cache_words,
    seq_lens,
    output_words,
    POOL_ROWS: tl.constexpr,
    POOL_ROW_WORDS: tl.constexpr,
    COPY_WORDS: tl.constexpr,
    WORDS_PER_ITEM: tl.constexpr,
    REQUEST_STRIDE: tl.constexpr,
    LENGTH_STRIDE: tl.constexpr,
):
    request = tl.program_id(0)
    pool = tl.load(req_pool_indices + request * REQUEST_STRIDE).to(tl.int32)
    start = tl.load(seq_lens + request * LENGTH_STRIDE).to(tl.int32)
    # A length-one native index vector selects one complete contiguous window.
    selected_row = tl.full((1,), pool, tl.int32)
    records = al.index_select_simd(
        src=cache_words,
        dim=0,
        index=selected_row,
        src_shape=(POOL_ROWS, POOL_ROW_WORDS),
        src_offset=(-1, start * WORDS_PER_ITEM),
        read_shape=(-1, COPY_WORDS),
    )
    words = tl.arange(0, COPY_WORDS)
    tl.store(
        output_words + request * COPY_WORDS + words,
        records.reshape(COPY_WORDS),
    )


@triton.jit
def _assign_draft_cache_locs_official_fixed128_kernel(
    req_pool_indices,
    req_to_token,
    seq_lens,
    output,
    POOL_LEN: tl.constexpr,
    COPY_LEN: tl.constexpr,
):
    block_size: tl.constexpr = 128
    request = tl.program_id(0)
    output_ptr = output + request * COPY_LEN
    start = tl.load(seq_lens + request)
    pool = tl.load(req_pool_indices + request)
    source_ptr = req_to_token + pool * POOL_LEN + start

    loop_count: tl.constexpr = tl.cdiv(COPY_LEN, block_size)
    for loop_index in range(loop_count):
        local = loop_index * block_size + tl.arange(0, block_size)
        mask = local < COPY_LEN
        values = tl.load(source_ptr + local, mask=mask)
        tl.store(output_ptr + local, values, mask=mask)


@triton.jit
def _assign_strided_window_kernel(
    req_pool_indices,
    req_to_token,
    seq_lens,
    output,
    COPY_LEN: tl.constexpr,
    POOL_STRIDE: tl.constexpr,
    COL_STRIDE: tl.constexpr,
    REQUEST_STRIDE: tl.constexpr,
    LENGTH_STRIDE: tl.constexpr,
    OUTPUT_STRIDE: tl.constexpr,
):
    request = tl.program_id(0).to(tl.int64)
    pool = tl.load(req_pool_indices + request * REQUEST_STRIDE).to(tl.int64)
    start = tl.load(seq_lens + request * LENGTH_STRIDE).to(tl.int64)
    for chunk in range(tl.cdiv(COPY_LEN, 128)):
        local = (chunk * 128 + tl.arange(0, 128)).to(tl.int64)
        keep = local < COPY_LEN
        values = tl.load(
            req_to_token + pool * POOL_STRIDE + (start + local) * COL_STRIDE,
            keep,
        )
        tl.store(
            output + (request * COPY_LEN + local) * OUTPUT_STRIDE, values, keep
        )


def assign_draft_cache_locs_contiguous(
    req_pool_indices,
    req_to_token,
    seq_lens,
    out_cache_loc,
    topk,
    speculative_num_steps,
):
    requests = req_pool_indices.numel()
    copy_len = topk * speculative_num_steps
    if requests == 0 or copy_len == 0:
        return out_cache_loc
    words = req_to_token.element_size() // 4
    native = (
        req_to_token.dtype in (torch.int32, torch.int64)
        and req_to_token.dtype == out_cache_loc.dtype
        and req_pool_indices.dtype in (torch.int32, torch.int64)
        and seq_lens.dtype in (torch.int32, torch.int64)
        and req_to_token.stride(1) == 1
        and out_cache_loc.stride(0) == 1
        and req_to_token.stride(0) >= req_to_token.shape[1]
        and 0
        < req_to_token.shape[0] * req_to_token.stride(0) * words
        <= 2147483647
        and 1 <= copy_len <= 1024
        and copy_len & (copy_len - 1) == 0
        and copy_len * req_to_token.element_size() >= 32
    )
    if native:
        _assign_native_request_window_kernel[(requests,)](
            req_pool_indices,
            triton.reinterpret(req_to_token, tl.float32),
            seq_lens,
            triton.reinterpret(out_cache_loc, tl.float32),
            POOL_ROWS=req_to_token.shape[0],
            POOL_ROW_WORDS=req_to_token.stride(0) * words,
            COPY_WORDS=copy_len * words,
            WORDS_PER_ITEM=words,
            REQUEST_STRIDE=req_pool_indices.stride(0),
            LENGTH_STRIDE=seq_lens.stride(0),
            num_warps=1,
            num_stages=1,
        )
    elif (
        req_to_token.stride(1) == 1
        and out_cache_loc.stride(0) == 1
        and req_pool_indices.stride(0) == seq_lens.stride(0) == 1
    ):
        _assign_draft_cache_locs_official_fixed128_kernel[(requests,)](
            req_pool_indices,
            req_to_token,
            seq_lens,
            out_cache_loc,
            POOL_LEN=req_to_token.stride(0),
            COPY_LEN=copy_len,
            num_warps=1,
            num_stages=1,
        )
    else:
        _assign_strided_window_kernel[(requests,)](
            req_pool_indices,
            req_to_token,
            seq_lens,
            out_cache_loc,
            COPY_LEN=copy_len,
            POOL_STRIDE=req_to_token.stride(0),
            COL_STRIDE=req_to_token.stride(1),
            REQUEST_STRIDE=req_pool_indices.stride(0),
            LENGTH_STRIDE=seq_lens.stride(0),
            OUTPUT_STRIDE=out_cache_loc.stride(0),
            num_warps=1,
            num_stages=1,
        )
    return out_cache_loc


solution = assign_draft_cache_locs_contiguous


__all__ = ["assign_draft_cache_locs_contiguous"]
