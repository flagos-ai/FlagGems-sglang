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

import triton
import triton.language as tl


@triton.jit
def _assign_draft_cache_locs_request_kernel(
    req_pool_indices,
    req_to_token,
    seq_lens,
    output,
    req_stride,
    ITEMS_PER_REQUEST: tl.constexpr,
    BLOCK: tl.constexpr,
):
    request = tl.program_id(0)
    local = tl.arange(0, BLOCK)
    mask = local < ITEMS_PER_REQUEST
    pool = tl.load(req_pool_indices + request)
    start = tl.load(seq_lens + request)
    values = tl.load(
        req_to_token + pool * req_stride + start + local,
        mask=mask,
    )
    tl.store(output + request * ITEMS_PER_REQUEST + local, values, mask=mask)


@triton.jit
def _assign_draft_cache_locs_grouped4_kernel(
    req_pool_indices,
    req_to_token,
    seq_lens,
    output,
    req_stride,
    n_req,
    ITEMS_PER_REQUEST: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    cmask = cols < ITEMS_PER_REQUEST
    base = pid * 4
    p0 = tl.load(req_pool_indices + tl.minimum(base + 0, n_req - 1))
    p1 = tl.load(req_pool_indices + tl.minimum(base + 1, n_req - 1))
    p2 = tl.load(req_pool_indices + tl.minimum(base + 2, n_req - 1))
    p3 = tl.load(req_pool_indices + tl.minimum(base + 3, n_req - 1))
    s0 = tl.load(seq_lens + tl.minimum(base + 0, n_req - 1))
    s1 = tl.load(seq_lens + tl.minimum(base + 1, n_req - 1))
    s2 = tl.load(seq_lens + tl.minimum(base + 2, n_req - 1))
    s3 = tl.load(seq_lens + tl.minimum(base + 3, n_req - 1))
    v0 = tl.load(req_to_token + p0 * req_stride + s0 + cols, mask=cmask)
    v1 = tl.load(req_to_token + p1 * req_stride + s1 + cols, mask=cmask)
    v2 = tl.load(req_to_token + p2 * req_stride + s2 + cols, mask=cmask)
    v3 = tl.load(req_to_token + p3 * req_stride + s3 + cols, mask=cmask)
    tl.store(output + (base + 0) * ITEMS_PER_REQUEST + cols, v0, mask=cmask & (base + 0 < n_req))
    tl.store(output + (base + 1) * ITEMS_PER_REQUEST + cols, v1, mask=cmask & (base + 1 < n_req))
    tl.store(output + (base + 2) * ITEMS_PER_REQUEST + cols, v2, mask=cmask & (base + 2 < n_req))
    tl.store(output + (base + 3) * ITEMS_PER_REQUEST + cols, v3, mask=cmask & (base + 3 < n_req))


@triton.jit
def _assign_draft_cache_locs_grouped8_kernel(
    req_pool_indices,
    req_to_token,
    seq_lens,
    output,
    req_stride,
    n_req,
    ITEMS_PER_REQUEST: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    cmask = cols < ITEMS_PER_REQUEST
    base = pid * 8
    p0 = tl.load(req_pool_indices + tl.minimum(base + 0, n_req - 1))
    p1 = tl.load(req_pool_indices + tl.minimum(base + 1, n_req - 1))
    p2 = tl.load(req_pool_indices + tl.minimum(base + 2, n_req - 1))
    p3 = tl.load(req_pool_indices + tl.minimum(base + 3, n_req - 1))
    p4 = tl.load(req_pool_indices + tl.minimum(base + 4, n_req - 1))
    p5 = tl.load(req_pool_indices + tl.minimum(base + 5, n_req - 1))
    p6 = tl.load(req_pool_indices + tl.minimum(base + 6, n_req - 1))
    p7 = tl.load(req_pool_indices + tl.minimum(base + 7, n_req - 1))
    s0 = tl.load(seq_lens + tl.minimum(base + 0, n_req - 1))
    s1 = tl.load(seq_lens + tl.minimum(base + 1, n_req - 1))
    s2 = tl.load(seq_lens + tl.minimum(base + 2, n_req - 1))
    s3 = tl.load(seq_lens + tl.minimum(base + 3, n_req - 1))
    s4 = tl.load(seq_lens + tl.minimum(base + 4, n_req - 1))
    s5 = tl.load(seq_lens + tl.minimum(base + 5, n_req - 1))
    s6 = tl.load(seq_lens + tl.minimum(base + 6, n_req - 1))
    s7 = tl.load(seq_lens + tl.minimum(base + 7, n_req - 1))
    v0 = tl.load(req_to_token + p0 * req_stride + s0 + cols, mask=cmask)
    v1 = tl.load(req_to_token + p1 * req_stride + s1 + cols, mask=cmask)
    v2 = tl.load(req_to_token + p2 * req_stride + s2 + cols, mask=cmask)
    v3 = tl.load(req_to_token + p3 * req_stride + s3 + cols, mask=cmask)
    v4 = tl.load(req_to_token + p4 * req_stride + s4 + cols, mask=cmask)
    v5 = tl.load(req_to_token + p5 * req_stride + s5 + cols, mask=cmask)
    v6 = tl.load(req_to_token + p6 * req_stride + s6 + cols, mask=cmask)
    v7 = tl.load(req_to_token + p7 * req_stride + s7 + cols, mask=cmask)
    tl.store(output + (base + 0) * ITEMS_PER_REQUEST + cols, v0, mask=cmask & (base + 0 < n_req))
    tl.store(output + (base + 1) * ITEMS_PER_REQUEST + cols, v1, mask=cmask & (base + 1 < n_req))
    tl.store(output + (base + 2) * ITEMS_PER_REQUEST + cols, v2, mask=cmask & (base + 2 < n_req))
    tl.store(output + (base + 3) * ITEMS_PER_REQUEST + cols, v3, mask=cmask & (base + 3 < n_req))
    tl.store(output + (base + 4) * ITEMS_PER_REQUEST + cols, v4, mask=cmask & (base + 4 < n_req))
    tl.store(output + (base + 5) * ITEMS_PER_REQUEST + cols, v5, mask=cmask & (base + 5 < n_req))
    tl.store(output + (base + 6) * ITEMS_PER_REQUEST + cols, v6, mask=cmask & (base + 6 < n_req))
    tl.store(output + (base + 7) * ITEMS_PER_REQUEST + cols, v7, mask=cmask & (base + 7 < n_req))


_GROUPED_MIN_REQUESTS = 64
_GROUPED_MAX_BLOCK = 1024
_GROUPED8_MIN_REQUESTS = 256


def assign_draft_cache_locs_contiguous(
    req_pool_indices,
    req_to_token,
    seq_lens,
    out_cache_loc,
    topk,
    speculative_num_steps,
):
    requests = req_pool_indices.numel()
    if requests == 0:
        return out_cache_loc
    items_per_request = topk * speculative_num_steps
    block = triton.next_power_of_2(items_per_request)
    if requests >= _GROUPED_MIN_REQUESTS and block <= _GROUPED_MAX_BLOCK:
        if requests >= _GROUPED8_MIN_REQUESTS:
            rows = 8
            grid = ((requests + rows - 1) // rows,)
            _assign_draft_cache_locs_grouped8_kernel[grid](
                req_pool_indices,
                req_to_token,
                seq_lens,
                out_cache_loc,
                req_to_token.stride(0),
                requests,
                ITEMS_PER_REQUEST=items_per_request,
                BLOCK=block,
                num_warps=4,
                num_stages=1,
            )
        else:
            rows = 4
            grid = ((requests + rows - 1) // rows,)
            _assign_draft_cache_locs_grouped4_kernel[grid](
                req_pool_indices,
                req_to_token,
                seq_lens,
                out_cache_loc,
                req_to_token.stride(0),
                requests,
                ITEMS_PER_REQUEST=items_per_request,
                BLOCK=block,
                num_warps=4,
                num_stages=1,
            )
        return out_cache_loc
    _assign_draft_cache_locs_request_kernel[(requests,)](
        req_pool_indices,
        req_to_token,
        seq_lens,
        out_cache_loc,
        req_to_token.stride(0),
        ITEMS_PER_REQUEST=items_per_request,
        BLOCK=block,
        num_warps=1 if block <= 256 else 4,
        num_stages=1,
    )
    return out_cache_loc


__all__ = ["assign_draft_cache_locs_contiguous"]
