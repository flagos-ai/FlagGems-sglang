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
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import triton
import triton.language as tl


@triton.jit
def _sum(
    X,
    Out,
    TOPK: tl.constexpr,
    HIDDEN: tl.constexpr,
    NC: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    token = pid // NC
    offsets = (pid % NC) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < HIDDEN
    base = X + token * TOPK * HIDDEN + offsets
    acc = tl.zeros((BLOCK,), tl.float32)
    for expert in tl.range(0, TOPK):
        acc += tl.load(
            base + expert * HIDDEN, mask=mask, other=0.0, cache_modifier=".cg"
        ).to(tl.float32)
    tl.store(
        Out + token * HIDDEN + offsets, acc.to(Out.dtype.element_ty), mask=mask
    )


def moe_topk_sum(x, out):
    tokens, topk, hidden = x.shape
    if tokens == 0 or hidden == 0:
        return out
    block = min(triton.next_power_of_2(hidden), 1024)
    chunks = (hidden + block - 1) // block
    _sum[(tokens * chunks,)](x, out, topk, hidden, chunks, block, num_warps=8)
    return out


__all__ = ["moe_topk_sum"]
