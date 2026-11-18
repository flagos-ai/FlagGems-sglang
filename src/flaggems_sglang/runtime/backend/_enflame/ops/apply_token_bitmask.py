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
    SPR,
    S,
    EXACT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    npg = tl.num_programs(0)
    WORDS: tl.constexpr = BLOCK // 32
    bits = tl.arange(0, 32)
    inv_bits = 31 - bits
    woffs = tl.arange(0, WORDS)
    eoffs = tl.arange(0, BLOCK)
    for s in tl.range(pid, S, npg):
        b = s // SPR
        j = s - b * SPR
        base = b * V
        if EXACT:
            words = tl.load(bitmask_ptr + b * NW + j * WORDS + woffs)
            keep = tl.reshape(
                (words[:, None] << inv_bits[None, :]) < 0, (BLOCK,)
            )
            vals = tl.load(logits_ptr + base + j * BLOCK + eoffs)
            res = tl.where(keep, vals, float("-inf"))
            tl.store(out_ptr + base + j * BLOCK + eoffs, res)
        else:
            m_w = (j * WORDS + woffs) < NW
            words = tl.load(
                bitmask_ptr + b * NW + j * WORDS + woffs, mask=m_w, other=0
            )
            keep = tl.reshape(
                (words[:, None] << inv_bits[None, :]) < 0, (BLOCK,)
            )
            m_v = (j * BLOCK + eoffs) < V
            vals = tl.load(
                logits_ptr + base + j * BLOCK + eoffs, mask=m_v, other=0.0
            )
            res = tl.where(keep, vals, float("-inf"))
            tl.store(out_ptr + base + j * BLOCK + eoffs, res, mask=m_v)


def apply_token_bitmask(logits, bitmask):
    B, V = logits.shape
    logits = logits.contiguous()
    bitmask = bitmask.contiguous()
    out = torch.empty_like(logits)
    if B == 0 or V == 0:
        return out

    NW = bitmask.shape[1]
    for BLOCK in (32768, 16384, 4096):
        if B * triton.cdiv(V, BLOCK) >= 48:
            break
    SPR = triton.cdiv(V, BLOCK)
    S = B * SPR
    _apply_token_bitmask_kernel[(min(S, 48),)](
        logits,
        bitmask,
        out,
        V,
        NW,
        SPR,
        S,
        EXACT=(V % BLOCK == 0),
        BLOCK=BLOCK,
        num_warps=1,
    )
    return out


__all__ = ["apply_token_bitmask"]
