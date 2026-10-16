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

"""fill_accept_out_cache_loc -- compact accepted draft KV slots (pure."""

import triton
import triton.language as tl

_MAX_BLOCK = 4096


@triton.jit
def _fill_accept_out_cache_loc_kernel(
    accept_index_ptr,  # [size] int64, -1 = rejected
    out_cache_loc_ptr,  # [>= size] int64, source slot ids
    out_ptr,  # [size] int64, destination (only [0, n) is written)
    size,
    BLOCK: tl.constexpr,
):
    running = tl.zeros([1], dtype=tl.int32)
    for start in range(0, size, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        mask = offs < size
        idx = tl.load(accept_index_ptr + offs, mask=mask, other=-1)
        keep = idx > -1
        keep_i = keep.to(tl.int32)
        # Exclusive prefix-sum position of each surviving lane, offset by the
        # count accumulated by previous chunks.
        pos = running + (tl.cumsum(keep_i, axis=0) - keep_i)
        vals = tl.load(out_cache_loc_ptr + idx, mask=keep, other=0)
        tl.store(out_ptr + pos, vals, mask=keep)
        running += tl.sum(keep_i, axis=0)


def fill_accept_out_cache_loc(
    accept_index, out_cache_loc, accept_out_cache_loc, size
):
    out = accept_out_cache_loc
    block = triton.next_power_of_2(max(min(int(size), _MAX_BLOCK), 1))
    num_warps = 1 if block <= 256 else 8
    _fill_accept_out_cache_loc_kernel[(1,)](
        accept_index,
        out_cache_loc,
        out,
        size,
        BLOCK=block,
        num_warps=num_warps,
    )
    return out


__all__ = ["fill_accept_out_cache_loc"]
