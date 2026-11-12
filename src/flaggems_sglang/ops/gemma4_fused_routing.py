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

"""Triton gemma4_fused_routing.

Per token: select the topk experts by descending logit (ties prefer the
smaller expert id), take a softmax over the selected logits, scale each
weight by ``per_expert_scale`` of its expert, and return fp32 weights
plus int32 expert ids in selection order.

One CTA per row. The whole expert axis lives in registers; top-k is K
argmax passes with ``tie_break_left`` (exactly "descending order, prefer
smaller id on ties"), and the softmax denominator is accumulated inside
the same loop, so the row is read exactly once: 1R plus two tiny K-wide
stores. No sorting, no scratch, fully parallel across tokens.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _gemma4_fused_routing_kernel(
    G,
    S,
    W,
    I,
    SG: tl.constexpr,
    SW: tl.constexpr,
    SI: tl.constexpr,
    E: tl.constexpr,
    K: tl.constexpr,
    KP: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    m = cols < E
    x = tl.load(G + row.to(tl.int64) * SG + cols, mask=m, other=float("-inf"))
    x = x.to(tl.float32)
    scale = tl.load(S + cols, mask=m, other=0.0).to(tl.float32)

    kvec = tl.arange(0, KP)
    evec = tl.zeros((KP,), tl.float32)
    svec = tl.zeros((KP,), tl.float32)
    acc = tl.zeros((), tl.float32)
    base = tl.zeros((), tl.float32)
    work = x

    for k in tl.static_range(K):
        mx, idx = tl.max(
            work,
            axis=0,
            return_indices=True,
            return_indices_tie_break_left=True,
        )
        if k == 0:
            base = mx
        e = tl.exp(mx - base)
        acc += e
        evec = tl.where(kvec == k, e, evec)
        gw = tl.sum(tl.where(cols == idx, scale, 0.0), 0)
        svec = tl.where(kvec == k, gw, svec)
        tl.store(I + row.to(tl.int64) * SI + k, idx.to(tl.int32))
        work = tl.where(cols == idx, float("-inf"), work)

    out = evec / acc * svec
    tl.store(W + row.to(tl.int64) * SW + kvec, out, mask=kvec < K)


def gemma4_fused_routing(gating_output, per_expert_scale, topk):
    t, e = gating_output.shape
    k = int(topk)
    weights = torch.empty(
        (t, k), dtype=torch.float32, device=gating_output.device
    )
    ids = torch.empty(
        (t, k), dtype=torch.int32, device=gating_output.device
    )
    if t == 0 or k == 0:
        return weights, ids
    block = max(16, triton.next_power_of_2(e))
    kp = max(1, triton.next_power_of_2(k))
    warps = 1 if block <= 256 else 2 if block <= 1024 else 4
    _gemma4_fused_routing_kernel[(t,)](
        gating_output,
        per_expert_scale,
        weights,
        ids,
        gating_output.stride(0),
        weights.stride(0),
        ids.stride(0),
        e,
        k,
        kp,
        block,
        num_warps=warps,
    )
    return weights, ids


__all__ = ["gemma4_fused_routing"]
