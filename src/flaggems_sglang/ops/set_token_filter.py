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

    if HAS_TOKENS:
        w_lo = pid * BLOCK
        w_hi = w_lo + BLOCK
        for i in range(0, n_tokens, TID_BLOCK):
            toffs = i + tl.arange(0, TID_BLOCK)
            tmask = toffs < n_tokens
            tid = tl.load(ids_ptr + toffs, mask=tmask, other=0)
            word = tid >> 5
            bit = tl.where(tmask, (1 << (tid & 31)).to(tl.int32), 0)
            in_tile = tmask & (word >= w_lo) & (word < w_hi)
            ptrs = mask_ptr + base + word
            if IS_ALLOWED:
                tl.atomic_or(ptrs, bit, mask=in_tile)
            else:
                tl.atomic_and(ptrs, ~bit, mask=in_tile)


def _stage_ids(token_ids):
    packed = array.array("i", token_ids)
    cpu = torch.frombuffer(memoryview(packed), dtype=torch.int32)
    pinned = torch.empty(len(token_ids), dtype=torch.int32, pin_memory=True)
    pinned.copy_(cpu)
    return pinned


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
    ids_pinned = _stage_ids(token_ids) if n_tokens > 0 else None

    tid_block = 1024
    if n_tokens <= 16:
        tid_block = 16
    elif n_tokens <= 32:
        tid_block = 32
    elif n_tokens <= 64:
        tid_block = 64
    elif n_tokens <= 256:
        tid_block = 256

    _set_token_filter_kernel[(triton.cdiv(n_words, 2048),)](
        vocab_mask,
        ids_pinned,
        n_words,
        n_tokens,
        batch_idx,
        RESET=bool(reset_vocab_mask),
        RESET_VAL=0 if is_allowed else -1,
        IS_ALLOWED=bool(is_allowed),
        HAS_TOKENS=n_tokens > 0,
        BLOCK=2048,
        TID_BLOCK=tid_block,
        num_warps=4,
    )
    return vocab_mask


__all__ = ["set_token_filter"]
