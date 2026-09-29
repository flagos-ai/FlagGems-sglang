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

"""compute_src2dst (routing-permutation inverse scatter) -- Hygon DCU specialization.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _src2dst_kernel(reorder_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    idx = tl.load(reorder_ptr + offs, mask=mask, other=0)
    tl.store(out_ptr + idx, offs.to(tl.int32), mask=mask)


def compute_src2dst(reorder_ids, num_toks):
    src2dst = torch.empty(
        num_toks, dtype=torch.int32, device=reorder_ids.device
    )
    if num_toks > 262144:
        block, warps = 2048, 2
    elif num_toks > 8192:
        block, warps = 256, 4
    else:
        block, warps = 64, 1
    grid = (triton.cdiv(num_toks, block),)
    _src2dst_kernel[grid](
        reorder_ids,
        src2dst,
        num_toks,
        BLOCK=block,
        num_warps=warps,
        num_stages=1,
    )
    return src2dst


__all__ = ["compute_src2dst"]
