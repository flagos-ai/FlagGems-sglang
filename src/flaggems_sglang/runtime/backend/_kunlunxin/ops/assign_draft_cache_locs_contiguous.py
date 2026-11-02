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

"""Task110 Kunlun probe using the native XPU libentry/program-id ABI.

Keep the already validated fixed-128 request owner and change only the backend
entry contract.  Other Task110 Kunlun candidates used a normal Triton JIT
entry and ``tl.program_id``; successful Kunlun routes in neighboring tasks use
FlagGems ``libentry`` plus ``triton_lang_extension.program_id`` instead.
"""

import triton
import triton.language as tl

from flaggems_sglang.utils import triton_lang_extension as ext
from flaggems_sglang.utils.libentry import libentry


@libentry()
@triton.jit
def _assign_draft_cache_locs_xpu_entry_kernel(
    req_pool_indices,
    req_to_token,
    seq_lens,
    output,
    POOL_LEN: tl.constexpr,
    TOPK: tl.constexpr,
    SPECULATIVE_NUM_STEPS: tl.constexpr,
):
    block_size: tl.constexpr = 128
    copy_len: tl.constexpr = TOPK * SPECULATIVE_NUM_STEPS
    request = ext.program_id(axis=0).to(tl.int64)

    pool = tl.load(req_pool_indices + request)
    start = tl.load(seq_lens + request)
    source = req_to_token + pool * POOL_LEN + start
    destination = output + request * copy_len

    loop_count: tl.constexpr = tl.cdiv(copy_len, block_size)
    for loop_index in range(loop_count):
        item = loop_index * block_size + tl.arange(0, block_size)
        mask = item < copy_len
        values = tl.load(source + item, mask=mask)
        tl.store(destination + item, values, mask=mask)


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

    _assign_draft_cache_locs_xpu_entry_kernel[(requests, 1, 1)](
        req_pool_indices,
        req_to_token,
        seq_lens,
        out_cache_loc,
        POOL_LEN=req_to_token.stride(0),
        TOPK=topk,
        SPECULATIVE_NUM_STEPS=speculative_num_steps,
        num_warps=1,
        num_stages=1,
        buffer_size_limit=2048,
    )
    return out_cache_loc


__all__ = ["assign_draft_cache_locs_contiguous"]
