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

"""compute_src2dst (routing-permutation inverse scatter) -- Kunlun XPU specialization.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _compute_src2dst_kernel(
    reorder_ptr,  # [num_toks] int64 (argsort output), permutation of [0, n)
    out_ptr,  # [num_toks] int32 output
    numel,
    BLOCK: tl.constexpr,
    NEED_MASK: tl.constexpr,  # 0 -> numel % BLOCK == 0, skip all predicates
):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    if NEED_MASK:
        mask = offs < numel
        # Coalesced load of the permutation; scattered lanes masked off idle.
        ids = tl.load(reorder_ptr + offs, mask=mask, other=0)
        # src2dst[reorder_ids[d]] = d — one scattered int32 store per element.
        tl.store(out_ptr + ids, offs.to(tl.int32), mask=mask)
    else:
        ids = tl.load(reorder_ptr + offs)
        tl.store(out_ptr + ids, offs.to(tl.int32))


def compute_src2dst(reorder_ids, num_toks):
    out = torch.empty(num_toks, dtype=torch.int32, device=reorder_ids.device)
    if num_toks == 0:
        return out

    if num_toks <= 512:
        block, warps = 64, 4
    elif num_toks <= 8192:
        block, warps = 512, 4
    else:
        block, warps = 1024, 4

    need_mask = num_toks % block != 0
    grid = (triton.cdiv(num_toks, block),)
    _compute_src2dst_kernel[grid](
        reorder_ids,
        out,
        num_toks,
        BLOCK=block,
        NEED_MASK=need_mask,
        num_warps=warps,
        num_stages=1,
    )
    return out


__all__ = ["compute_src2dst"]
