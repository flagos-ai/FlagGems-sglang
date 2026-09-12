# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import triton
import triton.language as tl


@triton.jit
def _apply_token_bitmask_kernel(
    logits_ptr,
    bitmask_ptr,
    out_ptr,
    V,
    NW,
    BLOCK: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_v = tl.program_id(1)

    offs_v = pid_v * BLOCK + tl.arange(0, BLOCK)
    m_v = offs_v < V

    word = offs_v // 32
    bit = offs_v % 32
    w = tl.load(bitmask_ptr + pid_b * NW + word, mask=m_v, other=0)
    keep = ((w >> bit) & 1) != 0

    x = tl.load(logits_ptr + pid_b * V + offs_v, mask=m_v, other=0.0)
    neg_inf = tl.full(x.shape, float("-inf"), x.dtype)
    y = tl.where(keep, x, neg_inf)
    tl.store(out_ptr + pid_b * V + offs_v, y, mask=m_v)


def apply_token_bitmask(logits, bitmask):
    B, V = logits.shape
    logits = logits.contiguous()
    bitmask = bitmask.contiguous()
    NW = bitmask.shape[1]

    out = torch.empty((B, V), dtype=logits.dtype, device=logits.device)

    BLOCK = 2048
    _apply_token_bitmask_kernel[(B, triton.cdiv(V, BLOCK))](
        logits,
        bitmask,
        out,
        V,
        NW,
        BLOCK=BLOCK,
        num_warps=8,
    )
    return out


reference = apply_token_bitmask

__all__ = ["apply_token_bitmask"]
