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
def _compact_chunk_kernel(
    accept_index_ptr,
    out_cache_loc_ptr,
    out_ptr,
    size,
    CHUNK: tl.constexpr,
    PREFIX_BLOCK: tl.constexpr,
    EVEN: tl.constexpr,
):
    pid = tl.program_id(0)

    offs = pid * CHUNK + tl.arange(0, CHUNK)
    if EVEN:
        ai = tl.load(accept_index_ptr + offs)
    else:
        ai = tl.load(accept_index_ptr + offs, mask=offs < size, other=-1)
    keep = ai > -1
    ki = keep.to(tl.int32)
    local = tl.cumsum(ki, axis=0) - ki

    hi_ptr = accept_index_ptr.to(tl.pointer_type(tl.int32))
    poffs = tl.arange(0, PREFIX_BLOCK)
    pmask = poffs < pid * CHUNK
    phi = tl.load(hi_ptr + 2 * poffs + 1, mask=pmask, other=-1)
    prefix = tl.sum((phi >= 0).to(tl.int32), axis=0)

    dst = local + prefix

    src = tl.where(keep, ai, 0)
    vals = tl.load(out_cache_loc_ptr + src)

    if EVEN:
        tl.store(out_ptr + dst, vals, mask=keep)
    else:
        tl.store(out_ptr + dst, vals, mask=keep & (offs < size))


@triton.jit
def _fill_accept_out_cache_loc_kernel(
    accept_index_ptr,
    out_cache_loc_ptr,
    out_ptr,
    size,
    BLOCK: tl.constexpr,
    EVEN: tl.constexpr,
):
    offs = tl.arange(0, BLOCK)
    if EVEN:
        ai = tl.load(accept_index_ptr + offs)
    else:
        ai = tl.load(accept_index_ptr + offs, mask=offs < size, other=-1)
    keep = ai > -1
    ki = keep.to(tl.int32)

    dst = tl.cumsum(ki, axis=0) - ki

    src = tl.where(keep, ai, 0)
    vals = tl.load(out_cache_loc_ptr + src)

    if EVEN:
        tl.store(out_ptr + dst, vals, mask=keep)
    else:
        tl.store(out_ptr + dst, vals, mask=keep & (offs < size))


def _pow2_at_least(x):
    return 1 << (x - 1).bit_length()


def fill_accept_out_cache_loc(
    accept_index, out_cache_loc, accept_out_cache_loc, size
):
    size = int(size)
    if size <= 0:
        return accept_out_cache_loc

    if size <= 512:
        block = _pow2_at_least(size)
        if block <= 128:
            num_warps = 1
        else:
            num_warps = 8
        _fill_accept_out_cache_loc_kernel[(1,)](
            accept_index,
            out_cache_loc,
            accept_out_cache_loc,
            size,
            BLOCK=block,
            EVEN=block == size,
            num_warps=num_warps,
        )
        return accept_out_cache_loc

    chunk = 1024
    grid = (size + chunk - 1) // chunk
    prefix_block = _pow2_at_least((grid - 1) * chunk)
    _compact_chunk_kernel[(grid,)](
        accept_index,
        out_cache_loc,
        accept_out_cache_loc,
        size,
        CHUNK=chunk,
        PREFIX_BLOCK=prefix_block,
        EVEN=size % chunk == 0,
        num_warps=16,
    )
    return accept_out_cache_loc


__all__ = ["fill_accept_out_cache_loc"]
