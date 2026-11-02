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


@triton.jit
def _copy_tgt_kernel(
    rows_ptr,
    tgt_ptr,
    prefix_ptr,
    num_correct_ptr,
    num_verify_tokens,
    BLOCK: tl.constexpr,
):
    b = tl.program_id(0)
    n = tl.load(num_correct_ptr + b) + 1
    start = tl.load(prefix_ptr + b).to(tl.int64)
    offs = tl.arange(0, BLOCK)
    m = offs < n
    col = tl.minimum(offs, num_verify_tokens - 1)
    v = tl.load(
        rows_ptr + b.to(tl.int64) * num_verify_tokens + col, mask=m, other=0
    )
    tl.store(tgt_ptr + start + offs, v, mask=m)


@triton.jit
def _copy_free_kernel(
    rows_ptr,
    free_ptr,
    prefix_ptr,
    to_free_ptr,
    num_verify_tokens,
    BLOCK: tl.constexpr,
):
    # Per-row gather of the trailing to_free cache locs.
    b = tl.program_id(0)
    f = tl.load(to_free_ptr + b)
    start = tl.load(prefix_ptr + b).to(tl.int64)
    offs = tl.arange(0, BLOCK)
    m = offs < f
    col = tl.maximum(num_verify_tokens - f + tl.minimum(offs, f - 1), 0)
    v = tl.load(
        rows_ptr + b.to(tl.int64) * num_verify_tokens + col, mask=m, other=0
    )
    tl.store(free_ptr + start + offs, v, mask=m)


def get_target_cache_loc(
    num_correct_drafts, to_free_num_slots, out_cache_loc, num_verify_tokens
):
    bs = num_correct_drafts.shape[0]
    rows = out_cache_loc.reshape(bs, num_verify_tokens)
    n_tgt = int(num_correct_drafts.sum().item()) + bs
    n_free = int(to_free_num_slots.sum().item())
    tgt = torch.zeros(
        max(n_tgt, 1), dtype=out_cache_loc.dtype, device=rows.device
    )
    free = torch.zeros(
        max(n_free, 1), dtype=out_cache_loc.dtype, device=rows.device
    )

    # host-side exclusive prefix sums (bs <= 1024, cheap)
    ncd = num_correct_drafts.cpu().tolist()
    tf = to_free_num_slots.cpu().tolist()
    tgt_prefix = torch.tensor(
        [sum(ncd[:i]) + i for i in range(bs)],
        dtype=torch.int64,
        device=rows.device,
    )
    free_prefix = torch.tensor(
        [sum(tf[:i]) for i in range(bs)], dtype=torch.int64, device=rows.device
    )

    BLOCK = 128
    _copy_tgt_kernel[(bs,)](
        rows,
        tgt,
        tgt_prefix,
        num_correct_drafts,
        num_verify_tokens,
        BLOCK=BLOCK,
    )
    _copy_free_kernel[(bs,)](
        rows,
        free,
        free_prefix,
        to_free_num_slots,
        num_verify_tokens,
        BLOCK=BLOCK,
    )
    return tgt, free


__all__ = ["get_target_cache_loc"]
