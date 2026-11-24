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

import torch
import triton
import triton.language as tl


@triton.jit
def _take1(packed, best_u, best_i, k, cols, slots, BLOCK_E: tl.constexpr):
    top = tl.max(packed, 0)
    pos = tl.min(tl.where(packed == top, cols, BLOCK_E - 1), 0)
    slot = slots == k
    best_u = tl.where(slot, top, best_u)
    best_i = tl.where(slot, pos, best_i)
    packed = tl.where(cols == pos, -2147483648, packed)
    return (packed, best_u, best_i)


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
    UNROLL: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_E)
    slots = tl.arange(0, BLOCK_K)
    ptrs = logits_ptr + row * S_ROW + cols * S_COL
    if BLOCK_E == E:
        vals = tl.load(ptrs).to(tl.float32)
    else:
        vals = tl.load(ptrs, mask=cols < E, other=-3e38).to(tl.float32)
    bits = vals.to(tl.int32, bitcast=True)
    packed = tl.where(bits >= 0, bits, bits ^ 2147483647)
    best_u = tl.zeros((BLOCK_K,), tl.int32)
    best_i = tl.zeros((BLOCK_K,), tl.int32)
    if UNROLL:
        for k in tl.static_range(TOPK):
            packed, best_u, best_i = _take1(
                packed, best_u, best_i, k, cols, slots, BLOCK_E
            )
    else:
        for k in tl.range(0, TOPK):
            packed, best_u, best_i = _take1(
                packed, best_u, best_i, k, cols, slots, BLOCK_E
            )
    bits = tl.where(best_u >= 0, best_u, best_u ^ 2147483647)
    best_v = bits.to(tl.float32, bitcast=True)
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


def gemma4_fused_routing(gating_output, per_expert_scale, topk):
    t, e = gating_output.shape
    if t >= 1024 and topk == 8 and (e in (64, 128, 256)):
        return _fast_gemma4_fused_routing(
            gating_output, per_expert_scale, topk
        )
    weights = gating_output.new_empty(t, topk, dtype=torch.float32)
    ids = gating_output.new_empty(t, topk, dtype=torch.int32)
    if t == 0 or topk == 0:
        return (weights, ids)
    block_e = 1 << (e - 1).bit_length()
    block_k = 1 << (topk - 1).bit_length()
    _routing_kernel_1d[t,](
        gating_output,
        per_expert_scale,
        weights,
        ids,
        gating_output.stride(0),
        gating_output.stride(1),
        e,
        block_e,
        topk,
        block_k,
        t * e >= 200000,
        num_warps=1,
        num_stages=1,
    )
    return (weights, ids)


@triton.jit
def _fast__kernel(
    X,
    S,
    W,
    I,
    SR: tl.constexpr,
    SC: tl.constexpr,
    E: tl.constexpr,
    B: tl.constexpr,
    K: tl.constexpr,
    BK: tl.constexpr,
    LOG_B: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, B)
    slots = tl.arange(0, BK)
    vals = tl.load(X + row * SR + cols * SC, cols < E, other=-float("inf")).to(
        tl.float32
    )
    bits = vals.to(tl.int32, bitcast=True)
    keys = tl.where(bits >= 0, bits, bits ^ 2147483647)
    keys = tl.where(vals == 0.0, 0, keys)
    peak = tl.max(keys, 0)
    limit: tl.constexpr = 4294967295 // B
    delta = peak.to(tl.uint32) - keys.to(tl.uint32)
    include = (delta <= limit) & (cols < E)
    packed = (limit - delta) * B | (B - 1 - cols).to(tl.uint32)
    packed = tl.where(include, packed, 0).to(tl.uint32)
    picked_ids = tl.zeros((BK,), tl.int32)
    picked_keys = tl.zeros((BK,), tl.int32)
    for k in tl.static_range(K):
        top = tl.max(packed, 0)
        pos = (B - 1 - (top & B - 1)).to(tl.int32)
        key = (peak.to(tl.uint32) - (limit - (top >> LOG_B))).to(tl.int32)
        picked_ids = tl.where(slots == k, pos, picked_ids)
        picked_keys = tl.where(slots == k, key, picked_keys)
        packed = tl.where(cols == pos, 0, packed)
    if top == 0:
        active = keys
        for k in tl.static_range(K):
            best_key = tl.max(active, 0)
            pos = tl.min(
                tl.where((active == best_key) & (cols < E), cols, B), 0
            )
            picked_ids = tl.where(slots == k, pos, picked_ids)
            picked_keys = tl.where(slots == k, best_key, picked_keys)
            active = tl.where(cols == pos, -2147483648, active)
    picked_bits = tl.where(
        picked_keys >= 0, picked_keys, picked_keys ^ 2147483647
    )
    best = picked_bits.to(tl.float32, bitcast=True)
    peak_bits = tl.where(peak >= 0, peak, peak ^ 2147483647)
    best_peak = peak_bits.to(tl.float32, bitcast=True)
    expo = tl.where(slots < K, tl.exp(best - best_peak), 0.0)
    weights = expo / tl.sum(expo, 0)
    weights *= tl.load(S + picked_ids, slots < K, other=0).to(tl.float32)
    out = row * K + slots
    tl.store(W + out, weights, slots < K)
    tl.store(I + out, picked_ids, slots < K)


def _fast_gemma4_fused_routing(gating_output, per_expert_scale, topk):
    t, e = gating_output.shape
    weights = gating_output.new_empty(t, topk, dtype=torch.float32)
    ids = gating_output.new_empty(t, topk, dtype=torch.int32)
    if t == 0 or topk == 0:
        return (weights, ids)
    _fast__kernel[t,](
        gating_output,
        per_expert_scale,
        weights,
        ids,
        gating_output.stride(0),
        gating_output.stride(1),
        e,
        triton.next_power_of_2(e),
        topk,
        triton.next_power_of_2(topk),
        (e - 1).bit_length(),
        num_warps=1,
        num_stages=1,
    )
    return (weights, ids)


__all__ = ["gemma4_fused_routing"]
