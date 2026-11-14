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

import array

import torch
import triton
import triton.language as tl


@triton.jit
def _set_token_filter_kernel(
    mask_ptr,
    ids_ptr,
    p_ptr,
    n_words,
    n_tokens,
    batch_idx,
    RESET: tl.constexpr,
    RESET_VAL: tl.constexpr,
    IS_ALLOWED: tl.constexpr,
    HAS_TOKENS: tl.constexpr,
    BLOCK: tl.constexpr,
    TID_BLOCK: tl.constexpr,
):
    # Zero-atomic presence design (2026-11-13): each program exclusively owns
    # the word range [pid*BLOCK, (pid+1)*BLOCK), so cross-program RMW conflicts
    # cannot exist and the old atomic_or/and only served duplicate token ids
    # inside one vector op. Instead scatter presence=1 into a per-program
    # scratch region (duplicate tids rewrite the same value, benign) and fold
    # the [BLOCK, 32] presence tile back into per-word bits with a plain
    # sum (0/1 presence shifted by distinct bit positions cannot collide).
    # The barriers are required: the scatter addresses (p_ptr + tid) alias
    # the later read (p_ptr + slot_base + slots) but the compiler cannot see
    # that; without the barrier the load is hoisted above the scatter
    # (measured silent wrong results on the flat-read variant).
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < n_words
    base = batch_idx * n_words

    if HAS_TOKENS:
        w_lo = pid * BLOCK
        w_hi = w_lo + BLOCK
        slot_base = w_lo * 32
        slots = tl.arange(0, BLOCK * 32)
        smask = (slot_base + slots) < n_words * 32
        tl.store(p_ptr + slot_base + slots, tl.zeros([BLOCK * 32], tl.int32), mask=smask)
        tl.debug_barrier()
        for i in range(0, n_tokens, TID_BLOCK):
            toffs = i + tl.arange(0, TID_BLOCK)
            tmask = toffs < n_tokens
            tid = tl.load(ids_ptr + toffs, mask=tmask, other=0)
            word = tid >> 5
            in_tile = tmask & (word >= w_lo) & (word < w_hi) & (word < n_words)
            ones = tl.full([TID_BLOCK], 1, tl.int32)
            tl.store(p_ptr + tid, ones, mask=in_tile)
        tl.debug_barrier()
        # Flat load + reshape: the strided 2-D load lowering is pathologically
        # slow here (measured 3.7-6.6x slower than flat+reshape).
        pf = tl.load(p_ptr + slot_base + slots, mask=smask, other=0)
        p2 = tl.reshape(pf, (BLOCK, 32))
        shifts = tl.arange(0, 32)
        wb = tl.sum(p2 << shifts[None, :], axis=1)
        if IS_ALLOWED:
            if RESET:
                val = tl.full([BLOCK], RESET_VAL, tl.int32) | wb
                tl.store(mask_ptr + base + offs, val, mask=m)
            else:
                orig = tl.load(mask_ptr + base + offs, mask=m, other=0)
                tl.store(mask_ptr + base + offs, orig | wb, mask=m)
        else:
            if RESET:
                val = tl.full([BLOCK], RESET_VAL, tl.int32) & ~wb
                tl.store(mask_ptr + base + offs, val, mask=m)
            else:
                orig = tl.load(mask_ptr + base + offs, mask=m, other=0)
                tl.store(mask_ptr + base + offs, orig & ~wb, mask=m)
    else:
        if RESET:
            tl.store(
                mask_ptr + base + offs,
                tl.full([BLOCK], RESET_VAL, tl.int32),
                mask=m,
            )


@triton.jit(
    do_not_specialize=[
        "t0",
        "t1",
        "t2",
        "t3",
        "t4",
        "t5",
        "t6",
        "t7",
        "t8",
        "t9",
        "t10",
        "t11",
        "t12",
        "t13",
        "t14",
        "t15",
    ]
)
def _set_token_filter_small_kernel(
    mask_ptr,
    n_words,
    batch_idx,
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
    RESET_VAL: tl.constexpr,
    IS_ALLOWED: tl.constexpr,
    BLOCK: tl.constexpr,
    NTOK: tl.constexpr,
):
    # Small-token-count arm: the ids arrive as scalar kernel arguments, so
    # the launch needs neither an H2D copy nor any pointer read (measured
    # 4.1 vs 7.3 us at n=16). do_not_specialize keeps varying token values
    # from recompiling the kernel.
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < n_words
    base = batch_idx * n_words

    if RESET:
        tl.store(
            mask_ptr + base + offs,
            tl.full([BLOCK], RESET_VAL, tl.int32),
            mask=m,
        )

    r = tl.arange(0, 16)
    tid = tl.zeros([16], dtype=tl.int32)
    if NTOK > 0:
        tid = tl.where(r == 0, t0, tid)
    if NTOK > 1:
        tid = tl.where(r == 1, t1, tid)
    if NTOK > 2:
        tid = tl.where(r == 2, t2, tid)
    if NTOK > 3:
        tid = tl.where(r == 3, t3, tid)
    if NTOK > 4:
        tid = tl.where(r == 4, t4, tid)
    if NTOK > 5:
        tid = tl.where(r == 5, t5, tid)
    if NTOK > 6:
        tid = tl.where(r == 6, t6, tid)
    if NTOK > 7:
        tid = tl.where(r == 7, t7, tid)
    if NTOK > 8:
        tid = tl.where(r == 8, t8, tid)
    if NTOK > 9:
        tid = tl.where(r == 9, t9, tid)
    if NTOK > 10:
        tid = tl.where(r == 10, t10, tid)
    if NTOK > 11:
        tid = tl.where(r == 11, t11, tid)
    if NTOK > 12:
        tid = tl.where(r == 12, t12, tid)
    if NTOK > 13:
        tid = tl.where(r == 13, t13, tid)
    if NTOK > 14:
        tid = tl.where(r == 14, t14, tid)
    if NTOK > 15:
        tid = tl.where(r == 15, t15, tid)

    tmask = r < NTOK
    word = tid >> 5
    bit = tl.where(tmask, (1 << (tid & 31)).to(tl.int32), 0)
    w_lo = pid * BLOCK
    w_hi = w_lo + BLOCK
    in_tile = tmask & (word >= w_lo) & (word < w_hi)
    ptrs = mask_ptr + base + word
    if IS_ALLOWED:
        tl.atomic_or(ptrs, bit, mask=in_tile)
    else:
        tl.atomic_and(ptrs, ~bit, mask=in_tile)


def _stage_ids(token_ids, device):
    # Host -> pinned -> device staging. Zero-copy reads of pinned memory
    # serialize per program on GCU (measured +0.7~2 us per program for every
    # extra program scanning the ids), while device-resident ids keep the
    # scan flat (~3 us) at any grid. The async H2D copy is a memcpy, not a
    # kernel launch, so the single-kernel contract is preserved.
    packed = array.array("i", token_ids)
    cpu = torch.frombuffer(memoryview(packed), dtype=torch.int32)
    pinned = torch.empty(len(token_ids), dtype=torch.int32, pin_memory=True)
    pinned.copy_(cpu)
    device_ids = torch.empty(len(token_ids), dtype=torch.int32, device=device)
    device_ids.copy_(pinned, non_blocking=True)
    return device_ids


def _pick_block(n_words):
    # Grid sizing for the presence kernel: each program zeroes/reads a
    # BLOCK*32-slot scratch region and scans every token. Measured optimum at
    # n_words=1024 is BLOCK=128 (grid 8, single wave); the flat-read lowering
    # is pathological below BLOCK=128 (70-101us at 32/64) while the
    # per-program scratch traffic dominates above 512, so clamp there.
    block = triton.next_power_of_2(max(1, (n_words + 7) // 8))
    return max(128, min(512, block))


def set_token_filter(
    vocab_mask, token_ids, batch_idx, is_allowed, reset_vocab_mask
):
    if not vocab_mask.is_contiguous():
        vocab_mask = vocab_mask.contiguous()
    batch, n_words = vocab_mask.shape
    if n_words == 0:
        return vocab_mask

    if batch_idx < 0:
        batch_idx += batch

    n_tokens = len(token_ids)
    reset = bool(reset_vocab_mask)
    reset_val = 0 if is_allowed else -1
    allowed = bool(is_allowed)

    block = _pick_block(n_words)
    grid = (triton.cdiv(n_words, block),)

    if 0 < n_tokens <= 16:
        toks = list(token_ids) + [0] * (16 - n_tokens)
        _set_token_filter_small_kernel[grid](
            vocab_mask,
            n_words,
            batch_idx,
            *toks,
            RESET=reset,
            RESET_VAL=reset_val,
            IS_ALLOWED=allowed,
            BLOCK=block,
            NTOK=n_tokens,
            num_warps=4,
        )
        return vocab_mask

    ids_dev = (
        _stage_ids(token_ids, vocab_mask.device) if n_tokens > 0 else None
    )
    # Presence scratch, one int32 slot per vocab id; uninitialized is fine
    # because each program zeroes its own region before scattering. Unused
    # when there are no tokens (dummy pointer, never dereferenced).
    presence = (
        torch.empty(n_words * 32, dtype=torch.int32, device=vocab_mask.device)
        if n_tokens > 0
        else vocab_mask
    )

    tid_block = 1024
    if n_tokens <= 16:
        tid_block = 16
    elif n_tokens <= 32:
        tid_block = 32
    elif n_tokens <= 64:
        tid_block = 64
    elif n_tokens <= 256:
        tid_block = 256

    _set_token_filter_kernel[grid](
        vocab_mask,
        ids_dev,
        presence,
        n_words,
        n_tokens,
        batch_idx,
        RESET=reset,
        RESET_VAL=reset_val,
        IS_ALLOWED=allowed,
        HAS_TOKENS=n_tokens > 0,
        BLOCK=block,
        TID_BLOCK=tid_block,
        num_warps=1,
    )
    return vocab_mask


__all__ = ["set_token_filter"]
