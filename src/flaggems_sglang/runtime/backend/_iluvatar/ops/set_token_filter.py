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


@triton.autotune(
    configs=[
        triton.Config({"BLOCK": 1024, "TB": 256}, num_warps=4),
        triton.Config({"BLOCK": 2048, "TB": 512}, num_warps=4),
        triton.Config({"BLOCK": 4096, "TB": 1024}, num_warps=8),
        triton.Config({"BLOCK": 8192, "TB": 1024}, num_warps=16),
        triton.Config({"BLOCK": 8192, "TB": 1024}, num_warps=32),
        triton.Config({"BLOCK": 16384, "TB": 1024}, num_warps=32),
    ],
    key=["n_words", "n_tokens"],
)
@triton.jit
def _set_token_filter_kernel(
    row_ptr,
    tokens_ptr,
    n_words,
    n_tokens,
    fill_value,
    BLOCK: tl.constexpr,
    TB: tl.constexpr,
    RESET: tl.constexpr,
    ALLOW: tl.constexpr,
):
    pid = tl.program_id(0)

    toffs = tl.arange(0, TB)
    tmask = toffs < n_tokens
    tids = tl.load(tokens_ptr + toffs, mask=tmask, other=0)

    offs = pid * BLOCK + tl.arange(0, BLOCK)
    wmask = offs < n_words
    if RESET:
        tl.store(row_ptr + offs, fill_value, mask=wmask)
    else:
        vals = tl.load(row_ptr + offs, mask=wmask, other=0)
        tl.store(row_ptr + offs, vals, mask=wmask)

    tl.debug_barrier()

    words = tids // 32
    bits = 1 << (tids % 32)
    hit = tmask & (words >= pid * BLOCK) & (words < pid * BLOCK + BLOCK)
    if ALLOW:
        tl.atomic_or(row_ptr + words, bits, mask=hit)
    else:
        tl.atomic_and(row_ptr + words, ~bits, mask=hit)

    if n_tokens > TB:
        for t0 in range(TB, n_tokens, TB):
            toffs2 = t0 + tl.arange(0, TB)
            tmask2 = toffs2 < n_tokens
            tids2 = tl.load(tokens_ptr + toffs2, mask=tmask2, other=0)
            words2 = tids2 // 32
            bits2 = 1 << (tids2 % 32)
            hit2 = (
                tmask2
                & (words2 >= pid * BLOCK)
                & (words2 < pid * BLOCK + BLOCK)
            )
            if ALLOW:
                tl.atomic_or(row_ptr + words2, bits2, mask=hit2)
            else:
                tl.atomic_and(row_ptr + words2, ~bits2, mask=hit2)


def set_token_filter(
    vocab_mask, token_ids, batch_idx, is_allowed, reset_vocab_mask
):
    row = vocab_mask[batch_idx]
    n_words = row.numel()
    n_tokens = len(token_ids)

    if n_tokens > 0:
        tokens = torch.tensor(token_ids, dtype=torch.int32, pin_memory=True)
    else:
        tokens = row

    if reset_vocab_mask:
        fill = 0 if is_allowed else -1
    else:
        fill = 0

    def grid(meta):
        return (triton.cdiv(n_words, meta["BLOCK"]),)

    _set_token_filter_kernel[grid](
        row,
        tokens,
        n_words,
        n_tokens,
        fill,
        RESET=bool(reset_vocab_mask),
        ALLOW=bool(is_allowed),
    )
    return vocab_mask


__all__ = ["set_token_filter"]
