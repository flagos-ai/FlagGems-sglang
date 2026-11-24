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

from math import gcd

import torch
import triton
import triton.language as tl


@triton.jit
def _take1(vals, best_v, best_i, k, cols, slots, E: tl.constexpr):
    top = tl.max(vals, 0)
    pos = tl.min(tl.where(vals == top, cols, E - 1), 0)
    slot = slots == k
    best_v = tl.where(slot, top, best_v)
    best_i = tl.where(slot, pos, best_i)
    vals = tl.where(cols == pos, -3.0e38, vals)
    return vals, best_v, best_i


@triton.jit
def _routing_kernel_1d(
    logits_ptr,
    scale_ptr,
    weight_ptr,
    index_ptr,
    S_ROW: tl.constexpr,
    S_COL: tl.constexpr,
    E: tl.constexpr,
    BLOCK_E: tl.constexpr,
    TOPK: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_E)
    slots = tl.arange(0, BLOCK_K)
    ptrs = logits_ptr + row * S_ROW + cols * S_COL
    if BLOCK_E == E:
        vals = tl.load(ptrs).to(tl.float32)
    else:
        vals = tl.load(ptrs, mask=cols < E, other=-3.0e38).to(tl.float32)

    best_v = tl.full((BLOCK_K,), -3.0e38, tl.float32)
    best_i = tl.zeros((BLOCK_K,), tl.int32)
    for k in tl.static_range(TOPK):
        vals, best_v, best_i = _take1(vals, best_v, best_i, k, cols, slots, E)

    expo = tl.exp(best_v - tl.max(best_v, 0))
    if BLOCK_K != TOPK:
        expo = tl.where(slots < TOPK, expo, 0.0)
    weight = expo / tl.sum(expo, 0)
    weight = weight * tl.load(scale_ptr + best_i).to(tl.float32)

    out = row * TOPK + slots
    if BLOCK_K == TOPK:
        tl.store(weight_ptr + out, weight)
        tl.store(index_ptr + out, best_i)
    else:
        keep = slots < TOPK
        tl.store(weight_ptr + out, weight, mask=keep)
        tl.store(index_ptr + out, best_i, mask=keep)


@triton.jit
def _take(
    vals, best_v, best_i, k, cols, slots, E: tl.constexpr, JOINT: tl.constexpr
):
    if JOINT:
        top, idx = tl.max(
            vals, 1, return_indices=True, return_indices_tie_break_left=True
        )
        pos = idx.to(tl.int32)
    else:
        top = tl.max(vals, 1)
        pos = tl.min(tl.where(vals == top[:, None], cols[None, :], E - 1), 1)
    slot = slots[None, :] == k
    best_v = tl.where(slot, top[:, None], best_v)
    best_i = tl.where(slot, pos[:, None], best_i)
    vals = tl.where(cols[None, :] == pos[:, None], -3.0e38, vals)
    return vals, best_v, best_i


@triton.jit
def _routing_kernel(
    logits_ptr,
    scale_ptr,
    weight_ptr,
    index_ptr,
    S_ROW: tl.constexpr,
    S_COL: tl.constexpr,
    E: tl.constexpr,
    BLOCK_E: tl.constexpr,
    TOPK: tl.constexpr,
    BLOCK_K: tl.constexpr,
    ROWS: tl.constexpr,
    UNROLL: tl.constexpr,
    JOINT: tl.constexpr,
):
    rows = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    cols = tl.arange(0, BLOCK_E)
    slots = tl.arange(0, BLOCK_K)
    ptrs = logits_ptr + rows[:, None] * S_ROW + cols[None, :] * S_COL
    if BLOCK_E == E:
        vals = tl.load(ptrs).to(tl.float32)
    else:
        vals = tl.load(ptrs, mask=(cols < E)[None, :], other=-3.0e38).to(
            tl.float32
        )

    best_v = tl.full((ROWS, BLOCK_K), -3.0e38, tl.float32)
    best_i = tl.zeros((ROWS, BLOCK_K), tl.int32)
    if UNROLL:
        for k in tl.static_range(TOPK):
            vals, best_v, best_i = _take(
                vals, best_v, best_i, k, cols, slots, E, JOINT
            )
    else:
        for k in tl.range(0, TOPK):
            vals, best_v, best_i = _take(
                vals, best_v, best_i, k, cols, slots, E, JOINT
            )

    expo = tl.exp(best_v - tl.max(best_v, 1)[:, None])
    if BLOCK_K != TOPK:
        expo = tl.where((slots < TOPK)[None, :], expo, 0.0)
    weight = expo / tl.sum(expo, 1)[:, None]
    weight = weight * tl.load(scale_ptr + best_i).to(tl.float32)

    out = rows[:, None] * TOPK + slots[None, :]
    if BLOCK_K == TOPK:
        tl.store(weight_ptr + out, weight)
        tl.store(index_ptr + out, best_i)
    else:
        keep = (slots < TOPK)[None, :]
        tl.store(weight_ptr + out, weight, mask=keep)
        tl.store(index_ptr + out, best_i, mask=keep)


def gemma4_fused_routing(gating_output, per_expert_scale, topk):
    t, e = gating_output.shape
    weights = gating_output.new_empty(t, topk, dtype=torch.float32)
    ids = gating_output.new_empty(t, topk, dtype=torch.int32)
    if t == 0 or topk == 0:
        return weights, ids
    block_e = 1 << (e - 1).bit_length()
    rows = min(gcd(t, 64), 8192 // block_e)
    if t <= 8:
        _routing_kernel_1d[(t,)](
            gating_output,
            per_expert_scale,
            weights,
            ids,
            gating_output.stride(0),
            gating_output.stride(1),
            e,
            block_e,
            topk,
            1 << (topk - 1).bit_length(),
            num_warps=1,
        )
        return weights, ids
    _routing_kernel[(t // rows,)](
        gating_output,
        per_expert_scale,
        weights,
        ids,
        gating_output.stride(0),
        gating_output.stride(1),
        e,
        block_e,
        topk,
        1 << (topk - 1).bit_length(),
        rows,
        t * e >= 200000,
        topk >= 8 and block_e <= 128,
        num_warps=1,
    )
    return weights, ids


__all__ = ["gemma4_fused_routing"]
