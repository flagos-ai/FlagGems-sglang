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
def _assign_draft_cache_locs_grouped_kernel(
    req_pool_indices,
    req_to_token,
    seq_lens,
    output,
    req_stride,
    ITEMS_PER_REQUEST: tl.constexpr,
    BLOCK_I: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    group = tl.program_id(0)
    request_local = tl.arange(0, BLOCK_R)[:, None]
    item = tl.arange(0, BLOCK_I)[None, :]
    request = group * BLOCK_R + request_local
    item_mask = item < ITEMS_PER_REQUEST

    pool = tl.load(req_pool_indices + request)
    start = tl.load(seq_lens + request)
    values = tl.load(
        req_to_token + pool * req_stride + start + item,
        mask=item_mask,
    )
    tl.store(
        output + request * ITEMS_PER_REQUEST + item,
        values,
        mask=item_mask,
    )


def _largest_power_of_two_divisor_at_most(value, limit):
    group = 1
    while group * 2 <= limit and value % (group * 2) == 0:
        group *= 2
    return group


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
    block_i = triton.next_power_of_2(items_per_request)
    # Keep each program near 256 lanes while grouping exact request divisors.
    # Exact division removes the request-tail mask and its runtime comparison.
    group_limit = max(1, min(16, 256 // block_i))
    block_r = _largest_power_of_two_divisor_at_most(requests, group_limit)
    _assign_draft_cache_locs_grouped_kernel[(requests // block_r,)](
        req_pool_indices,
        req_to_token,
        seq_lens,
        out_cache_loc,
        req_to_token.stride(0),
        ITEMS_PER_REQUEST=items_per_request,
        BLOCK_I=block_i,
        BLOCK_R=block_r,
        num_warps=1 if block_i * block_r <= 64 else 4,
        num_stages=1,
    )
    return out_cache_loc


__all__ = ["assign_draft_cache_locs_contiguous"]
