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

_MAX_SCALAR_IDS = 16

# n <= _MAX_SCALAR_IDS path: flat copy, in-row programs fold bits in registers.
_SCALAR_BLOCK = 256
_SCALAR_WARPS = 4

# n > _MAX_SCALAR_IDS path: copy programs + per-chunk atomic row programs.
_COPY_BLOCK = 2048
_ROW_PROGS = 4
_TOK_BLOCK = 512
_TENSOR_WARPS = 8


@triton.jit
def _set_token_filter_flat(
    src_ptr,
    out_ptr,
    row_start,
    words,
    fill,
    total,
    n,
    t0,
    t1,
    t2,
    t3,
    t4,
    t5,
    t6,
    t7,
    t8,
    t9,
    t10,
    t11,
    t12,
    t13,
    t14,
    t15,
    RESET: tl.constexpr,
    IS_ALLOWED: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """One program per contiguous block of the flattened mask."""
    pid = tl.program_id(0)
    start = pid * BLOCK
    offs = start + tl.arange(0, BLOCK)
    bv = offs < total
    if start + BLOCK <= row_start or start >= row_start + words:
        # Entirely outside the target row: stream the block verbatim.
        tl.store(out_ptr + offs, tl.load(src_ptr + offs, mask=bv), mask=bv)
    else:
        # Overlaps the target row: only the in-row lanes change, the rest of
        # the block is still copied through unchanged.
        inrow = (offs >= row_start) & (offs < row_start + words)
        v = tl.load(src_ptr + offs, mask=bv, other=0)
        if RESET:
            v = tl.where(inrow, tl.full((BLOCK,), fill, tl.int32), v)
        for i in tl.static_range(16):
            if i == 0:
                tid = t0
            elif i == 1:
                tid = t1
            elif i == 2:
                tid = t2
            elif i == 3:
                tid = t3
            elif i == 4:
                tid = t4
            elif i == 5:
                tid = t5
            elif i == 6:
                tid = t6
            elif i == 7:
                tid = t7
            elif i == 8:
                tid = t8
            elif i == 9:
                tid = t9
            elif i == 10:
                tid = t10
            elif i == 11:
                tid = t11
            elif i == 12:
                tid = t12
            elif i == 13:
                tid = t13
            elif i == 14:
                tid = t14
            else:
                tid = t15
            # Slots past ``n`` are inert: ``i < n`` is False, so ``hit`` is
            # all-False and the padding value never lands.
            hit = (i < n) & (offs == row_start + tid // 32)
            bit = 1 << (tid % 32)
            if IS_ALLOWED:
                v = v | tl.where(hit, bit, 0)
            else:
                v = v & ~tl.where(hit, bit, 0)
        tl.store(out_ptr + offs, v, mask=bv)


@triton.jit
def _set_token_filter_row(
    src_ptr,
    out_ptr,
    ids_ptr,
    total,
    row_start,
    words,
    copy_progs,
    words_per_chunk,
    fill,
    n,
    RESET: tl.constexpr,
    IS_ALLOWED: tl.constexpr,
    BLOCK: tl.constexpr,
    TOK: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid < copy_progs:
        # Copy programs: clone every word outside the target row.
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        m = (offs < total) & ~(
            (offs >= row_start) & (offs < row_start + words)
        )
        tl.store(out_ptr + offs, tl.load(src_ptr + offs, mask=m), mask=m)
    else:
        rp = pid - copy_progs
        w_lo = rp * words_per_chunk
        w_hi = tl.minimum(w_lo + words_per_chunk, words)
        # Pass 1: materialise the chunk (reset fill or source copy).
        for w0 in range(w_lo, w_hi, BLOCK):
            woffs = w0 + tl.arange(0, BLOCK)
            wm = woffs < w_hi
            if RESET:
                v = tl.full((BLOCK,), fill, tl.int32)
            else:
                v = tl.load(src_ptr + row_start + woffs, mask=wm, other=0)
            tl.store(out_ptr + row_start + woffs, v, mask=wm)
        # Order this program's pass-1 stores before its own atomics.
        tl.debug_barrier()
        # Coalesced gather: lane i reads ids element t0 + i.
        for t0 in range(0, n, TOK):
            toffs = t0 + tl.arange(0, TOK)
            tm = toffs < n
            tids = tl.load(ids_ptr + toffs, mask=tm, other=0)
            word = tids // 32
            bit = 1 << (tids % 32)
            in_chunk = tm & (word >= w_lo) & (word < w_hi)
            ptr = out_ptr + row_start + word
            if IS_ALLOWED:
                tl.atomic_or(ptr, bit, mask=in_chunk)
            else:
                tl.atomic_and(ptr, ~bit, mask=in_chunk)


def set_token_filter(
    vocab_mask, token_ids, batch_idx, is_allowed, reset_vocab_mask
):
    words = int(vocab_mask.shape[1])
    total = vocab_mask.numel()

    out = torch.empty_like(vocab_mask)

    row_start = int(batch_idx) * words
    fill = 0 if is_allowed else -1
    n = len(token_ids)
    reset = bool(reset_vocab_mask)
    allowed = bool(is_allowed)

    if n <= _MAX_SCALAR_IDS:
        ids = list(token_ids) + [0] * (_MAX_SCALAR_IDS - n)
        grid = (triton.cdiv(total, _SCALAR_BLOCK),)
        _set_token_filter_flat[grid](
            vocab_mask,
            out,
            row_start,
            words,
            fill,
            total,
            n,
            *ids,
            RESET=reset,
            IS_ALLOWED=allowed,
            BLOCK=_SCALAR_BLOCK,
            num_warps=_SCALAR_WARPS,
        )
        return out

    ids_dev = torch.tensor(token_ids, dtype=torch.int32).pin_memory()
    copy_progs = triton.cdiv(total, _COPY_BLOCK)
    grid = (copy_progs + _ROW_PROGS,)
    _set_token_filter_row[grid](
        vocab_mask,
        out,
        ids_dev,
        total,
        row_start,
        words,
        copy_progs,
        triton.cdiv(words, _ROW_PROGS),
        fill,
        n,
        RESET=reset,
        IS_ALLOWED=allowed,
        BLOCK=_COPY_BLOCK,
        TOK=_TOK_BLOCK,
        num_warps=_TENSOR_WARPS,
    )
    return out


__all__ = ["set_token_filter"]
