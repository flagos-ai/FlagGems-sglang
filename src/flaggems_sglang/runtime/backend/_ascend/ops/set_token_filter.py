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


@triton.jit(do_not_specialize=["row_off", "n_words"])
def _stf_reset_kernel(
    out_ptr, row_off, n_words, FILL: tl.constexpr, BLOCK: tl.constexpr
):
    """把 out 的 batch_idx 行 (从 row_off 起, 共 n_words 个 int32) 重置。

    FILL=True -> 全 0 (allow 场景); FILL=False -> 全 -1 (deny 场景)。
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < n_words
    v = tl.zeros([BLOCK], tl.int32) if FILL else tl.full([BLOCK], -1, tl.int32)
    tl.store(out_ptr + row_off + offs, v, mask=m)


@triton.jit(do_not_specialize=["row_off", "n_tok"])
def _stf_apply_bits_kernel(
    tok_ptr,
    out_ptr,
    row_off,
    n_tok,
    IS_ALLOW: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """每个 program 处理 BLOCK 个 token, 原子更新对应 word 的 bit。"""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < n_tok
    t = tl.load(tok_ptr + offs, mask=m, other=0)
    word = (t >> 5).to(tl.int32)
    bv = (1 << (t & 31)).to(tl.int32)
    row_ptr = out_ptr + row_off
    if IS_ALLOW:
        tl.atomic_or(row_ptr + word, bv, mask=m)
    else:
        tl.atomic_and(row_ptr + word, (~bv).to(tl.int32), mask=m)


_RESET_BLOCK = 2048
_BITS_BLOCK = 256


def set_token_filter(
    vocab_mask, token_ids, batch_idx, is_allowed, reset_vocab_mask
):
    out = vocab_mask.clone()
    n_words = out.shape[1]
    row_off = int(batch_idx) * n_words
    if reset_vocab_mask:
        _stf_reset_kernel[(triton.cdiv(n_words, _RESET_BLOCK),)](
            out,
            row_off,
            n_words,
            FILL=bool(is_allowed),
            BLOCK=_RESET_BLOCK,
            num_warps=8,
        )
    n_tok = len(token_ids)
    if n_tok:
        tok = torch.tensor(
            token_ids, dtype=torch.int32, device=vocab_mask.device
        )
        _stf_apply_bits_kernel[(triton.cdiv(n_tok, _BITS_BLOCK),)](
            tok,
            out,
            row_off,
            n_tok,
            IS_ALLOW=bool(is_allowed),
            BLOCK=_BITS_BLOCK,
            num_warps=4,
        )
    return out


__all__ = ["set_token_filter"]
