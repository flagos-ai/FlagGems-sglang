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

"""Stream-compaction kernel for speculative decoding output cache."""

import triton
import triton.language as tl


@triton.jit
def _fill_accept_out_cache_loc_kernel(
    accept_index_ptr,
    out_cache_loc_ptr,
    out_ptr,
    size,
    BLOCK: tl.constexpr,
):
    dst = tl.zeros((), dtype=tl.int32)
    for s in range(0, size, BLOCK):
        offs = s + tl.arange(0, BLOCK)
        # Clamp the load address instead of masking the load: OOB lanes read
        # the last element and are discarded by ``keep`` below.
        lc = tl.minimum(offs, size - 1)
        idx = tl.load(accept_index_ptr + lc)
        valid = offs < size
        keep = (idx > -1) & valid
        keep_i = keep.to(tl.int32)
        # Intra-block exclusive prefix count of accepted entries.
        prefix = tl.cumsum(keep_i, axis=0) - keep_i
        # Gather surviving entries; rejected/OOB lanes are clamped to slot 0 so
        # the address is always in-range even though the value is discarded.
        safe_idx = tl.maximum(idx, 0)
        slot = tl.load(out_cache_loc_ptr + safe_idx)
        # Rejected/OOB lanes park on the last output slot, which is undefined
        # for them (or holds the same value they would write anyway).
        pos = tl.where(keep, dst + prefix, size - 1)
        tl.store(out_ptr + pos, slot)
        dst += tl.sum(keep_i, axis=0)


def _pick_block(size):
    if size <= 64:
        return 64, 1
    if size <= 256:
        return 256, 2
    return 1024, 4


def fill_accept_out_cache_loc(
    accept_index, out_cache_loc, accept_out_cache_loc, size
):
    if size <= 0:
        return accept_out_cache_loc
    block, num_warps = _pick_block(size)
    _fill_accept_out_cache_loc_kernel[(1,)](
        accept_index,
        out_cache_loc,
        accept_out_cache_loc,
        size,
        BLOCK=block,
        num_warps=num_warps,
        num_stages=1,
    )
    return accept_out_cache_loc


__all__ = ["fill_accept_out_cache_loc"]
