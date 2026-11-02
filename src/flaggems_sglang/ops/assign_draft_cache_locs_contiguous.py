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
