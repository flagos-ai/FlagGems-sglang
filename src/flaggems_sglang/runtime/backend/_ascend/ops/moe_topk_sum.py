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
    BM: tl.constexpr,
    BD: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = (pid // NC) * BM + tl.arange(0, BM)
    cols = (pid % NC) * BD + tl.arange(0, BD)
    mask = cols[None, :] < HIDDEN
    base = X + rows[:, None] * (TOPK * HIDDEN) + cols[None, :]
    acc = tl.zeros((BM, BD), tl.float32)
    for expert in tl.range(0, TOPK):
        acc += tl.load(base + expert * HIDDEN, mask=mask, other=0.0).to(
            tl.float32
        )
    tl.store(
        Out + rows[:, None] * HIDDEN + cols[None, :],
        acc.to(Out.dtype.element_ty),
        mask=mask,
    )


def moe_topk_sum(x, out):
    tokens, topk, hidden = x.shape
    if tokens == 0 or hidden == 0:
        return out
    bd = min(triton.next_power_of_2(hidden), 1024)
    bm = 8192 // bd
    while bm > 1 and tokens % bm:
        bm = bm // 2
    chunks = (hidden + bd - 1) // bd
    _sum[((tokens // bm) * chunks,)](
        x, out, topk, hidden, chunks, bm, bd, num_warps=4
    )
    return out


__all__ = ["moe_topk_sum"]
