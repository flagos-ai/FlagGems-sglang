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

import torch
import triton
import triton.language as tl

_SMALL_BS = 32


@triton.jit(do_not_specialize=[0, 1, 2, 3, 4, 5, 6, 7])
def _pack_kernel(
    out_cache_loc_ptr,  # [bs * NVT] flat verify output cache locations
    num_correct_ptr,  # [bs] counts of accepted draft tokens
    to_free_ptr,  # [bs] counts of slots to free
    tgt_ptr,  # [bs * NVT] upper-bound output for kept slots
    free_out_ptr,  # [bs * NVT] upper-bound output for to-free slots
    totals_ptr,  # [2] int32 out: [sum(keep), sum(free)]
    bs,
    NVT,  # num_verify_tokens (row length)
    BLOCK_P: tl.constexpr,  # >= bs, power of two (counts scan width)
    BLOCK_T: tl.constexpr,  # >= NVT, power of two (row copy width)
):
    b = tl.program_id(0)
    keep = tl.load(num_correct_ptr + b).to(tl.int32) + 1
    f = tl.load(to_free_ptr + b).to(tl.int32)

    # Exclusive prefix sums for this request via masked reductions over the
    # whole (tiny, L2-resident) counts vector.
    offs = tl.arange(0, BLOCK_P)
    pmask = offs < bs
    keep_all = tl.load(num_correct_ptr + offs, mask=pmask, other=0).to(
        tl.int32
    )
    keep_all = tl.where(pmask, keep_all + 1, 0)
    free_all = tl.load(to_free_ptr + offs, mask=pmask, other=0).to(tl.int32)
    tgt_start = tl.sum(tl.where(offs < b, keep_all, 0), axis=0).to(tl.int64)
    free_start = tl.sum(tl.where(offs < b, free_all, 0), axis=0).to(tl.int64)

    if b == 0:
        tl.store(totals_ptr + 0, tl.sum(keep_all, axis=0).to(tl.int32))
        tl.store(totals_ptr + 1, tl.sum(free_all, axis=0).to(tl.int32))

    j = tl.arange(0, BLOCK_T)
    row = out_cache_loc_ptr + b.to(tl.int64) * NVT

    vals = tl.load(row + j, mask=j < NVT, other=0)

    # Head of the row (the accepted bonus token + accepted drafts) -> tgt.
    m_keep = j < keep
    tl.store(tgt_ptr + tgt_start + j, vals, mask=m_keep)

    # Tail of the row (rejected drafts) -> to_free_slots.
    m_free = (j >= NVT - f) & (j < NVT)
    tl.store(free_out_ptr + free_start + (j - (NVT - f)), vals, mask=m_free)


@triton.jit(do_not_specialize=[0, 1, 2, 3, 4, 5, 6, 7])
def _pack_kernel_small(
    out_cache_loc_ptr,  # [bs * NVT] flat verify output cache locations
    num_correct_ptr,  # [bs] counts of accepted draft tokens
    to_free_ptr,  # [bs] counts of slots to free
    tgt_ptr,  # [bs * NVT] upper-bound output for kept slots
    free_out_ptr,  # [bs * NVT] upper-bound output for to-free slots
    totals_ptr,  # [2] int32 out: [sum(keep), sum(free)]
    bs,
    NVT,  # num_verify_tokens (row length)
    BLOCK_T: tl.constexpr,  # >= NVT, power of two (row copy width)
):
    tgt_start = 0
    free_start = 0
    for b in range(0, bs):
        keep = tl.load(num_correct_ptr + b).to(tl.int32) + 1
        f = tl.load(to_free_ptr + b).to(tl.int32)

        j = tl.arange(0, BLOCK_T)
        row = out_cache_loc_ptr + b.to(tl.int64) * NVT
        vals = tl.load(row + j, mask=j < NVT, other=0)

        m_keep = j < keep
        tl.store(tgt_ptr + tgt_start + j, vals, mask=m_keep)
        tgt_start += keep

        m_free = (j >= NVT - f) & (j < NVT)
        tl.store(
            free_out_ptr + free_start + (j - (NVT - f)), vals, mask=m_free
        )
        free_start += f

    tl.store(totals_ptr + 0, tgt_start.to(tl.int32))
    tl.store(totals_ptr + 1, free_start.to(tl.int32))


def get_target_cache_loc(
    num_correct_drafts, to_free_num_slots, out_cache_loc, num_verify_tokens
):
    bs = num_correct_drafts.shape[0]
    dev = torch.device(num_correct_drafts.device)
    if bs == 0:
        empty = torch.zeros(2, dtype=out_cache_loc.dtype, device=dev)
        return empty[:1], empty[1:]

    out_cache_loc = out_cache_loc.contiguous()
    num_correct_drafts = num_correct_drafts.contiguous()
    to_free_num_slots = to_free_num_slots.contiguous()

    nvt = int(num_verify_tokens)
    row_max = bs * nvt
    buf = torch.empty(2 * row_max, dtype=out_cache_loc.dtype, device=dev)
    tgt_buf = buf[:row_max]
    free_buf = buf[row_max:]
    totals = torch.empty(2, dtype=torch.int32, device=dev)

    block_t = max(triton.next_power_of_2(nvt), 2)
    if bs <= _SMALL_BS:
        # Small batch: one persistent program walks the requests serially.
        _pack_kernel_small[(1,)](
            out_cache_loc,
            num_correct_drafts,
            to_free_num_slots,
            tgt_buf,
            free_buf,
            totals,
            bs,
            nvt,
            BLOCK_T=block_t,
        )
    else:
        block_p = max(triton.next_power_of_2(bs), 16)
        _pack_kernel[(bs,)](
            out_cache_loc,
            num_correct_drafts,
            to_free_num_slots,
            tgt_buf,
            free_buf,
            totals,
            bs,
            nvt,
            BLOCK_P=block_p,
            BLOCK_T=block_t,
            num_warps=4 if block_p >= 256 else 1,
        )

    n_tgt, n_free = totals.tolist()
    tgt = (
        tgt_buf[:n_tgt]
        if n_tgt > 0
        else torch.zeros(1, dtype=out_cache_loc.dtype, device=dev)
    )
    if n_free > 0:
        free = free_buf[:n_free]
    else:
        free = torch.zeros(1, dtype=out_cache_loc.dtype, device=dev)
    return tgt, free


__all__ = ["get_target_cache_loc"]
