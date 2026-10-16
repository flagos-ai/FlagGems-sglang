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

_ROWS = 1024
_COLS = 2
_BLOCK = _ROWS * _COLS


@triton.jit
def _one_shot_kernel(
    accept_index_ptr,  # [size] int64, -1 = rejected
    out_cache_loc_ptr,  # [>= max(accept_index)+1] int64, source slots
    out_ptr,  # [>= size] int64 output
    size,
    BLOCK: tl.constexpr,
):
    offs = tl.arange(0, BLOCK)
    inb = offs < size
    # Rejected lanes read ``other=-1`` and are dropped by ``keep``.
    ai = tl.load(accept_index_ptr + offs, mask=inb, other=-1)
    keep = (ai > -1) & inb
    inc = keep.to(tl.int32)
    # Exclusive prefix sum: every accepted lane's destination slot.
    dst = tl.cumsum(inc, axis=0) - inc
    src = tl.load(out_cache_loc_ptr + ai, mask=keep, other=0)
    tl.store(out_ptr + dst, src, mask=keep)


@triton.jit
def _tiled_kernel(
    accept_index_ptr,
    out_cache_loc_ptr,
    out_ptr,
    size,
    ROWS: tl.constexpr,
    COLS: tl.constexpr,
):
    pid = tl.program_id(0)
    start = pid * ROWS * COLS

    cnt = 0
    for s in tl.range(0, start, ROWS * COLS):
        offs = s + tl.arange(0, ROWS * COLS)
        ai = tl.load(accept_index_ptr + offs)
        cnt += tl.sum((ai > -1).to(tl.int32), axis=0)

    row = tl.arange(0, ROWS)[:, None]
    col = tl.arange(0, COLS)[None, :]
    offs2 = start + row * COLS + col
    inb2 = offs2 < size
    ai2 = tl.load(accept_index_ptr + offs2, mask=inb2, other=-1)
    keep2 = (ai2 > -1) & inb2
    inc2 = keep2.to(tl.int32)
    rowtot = tl.sum(inc2, axis=1)
    rowpre = tl.cumsum(rowtot, axis=0) - rowtot
    dst2 = cnt + rowpre[:, None] + tl.cumsum(inc2, axis=1) - inc2
    src2 = tl.load(out_cache_loc_ptr + ai2, mask=keep2, other=0)
    tl.store(out_ptr + dst2, src2, mask=keep2)


def fill_accept_out_cache_loc(
    accept_index, out_cache_loc, accept_out_cache_loc, size
):
    if size > 0:
        if size <= 512:
            _one_shot_kernel[(1,)](
                accept_index,
                out_cache_loc,
                accept_out_cache_loc,
                size,
                BLOCK=512,
                num_warps=1,
            )
        elif size <= _BLOCK:
            _one_shot_kernel[(1,)](
                accept_index,
                out_cache_loc,
                accept_out_cache_loc,
                size,
                BLOCK=_BLOCK,
                num_warps=4,
            )
        else:
            grid = (triton.cdiv(size, _BLOCK),)
            _tiled_kernel[grid](
                accept_index,
                out_cache_loc,
                accept_out_cache_loc,
                size,
                ROWS=_ROWS,
                COLS=_COLS,
                num_warps=8,
            )
    return accept_out_cache_loc


__all__ = ["fill_accept_out_cache_loc"]
