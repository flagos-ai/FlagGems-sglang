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
def _top8(a0, a1, a2, a3, a4, a5, a6, a7, b0, b1, b2, b3, b4, b5, b6, b7):
    h0 = tl.maximum(a0, b0)
    l0 = tl.minimum(a0, b0)
    h1 = tl.maximum(a4, b4)
    h2 = tl.maximum(h1, l0)
    l2 = tl.minimum(h1, l0)
    h3 = tl.maximum(a2, b2)
    l3 = tl.minimum(a2, b2)
    h4 = tl.maximum(a6, b6)
    h5 = tl.maximum(h4, l3)
    h6 = tl.maximum(h3, h2)
    l6 = tl.minimum(h3, h2)
    h7 = tl.maximum(h5, l2)
    l7 = tl.minimum(h5, l2)
    h9 = tl.maximum(a1, b1)
    l9 = tl.minimum(a1, b1)
    h10 = tl.maximum(a5, b5)
    h11 = tl.maximum(h10, l9)
    l11 = tl.minimum(h10, l9)
    h12 = tl.maximum(a3, b3)
    l12 = tl.minimum(a3, b3)
    h13 = tl.maximum(a7, b7)
    h14 = tl.maximum(h13, l12)
    h15 = tl.maximum(h12, h11)
    l15 = tl.minimum(h12, h11)
    h16 = tl.maximum(h14, l11)
    h18 = tl.maximum(h9, h6)
    l18 = tl.minimum(h9, h6)
    h19 = tl.maximum(h15, l6)
    l19 = tl.minimum(h15, l6)
    h20 = tl.maximum(l15, h7)
    l20 = tl.minimum(l15, h7)
    h21 = tl.maximum(h16, l7)
    return (h0, h18, l18, h19, l19, h20, l20, h21)


@triton.jit
def _top4(a0, a1, a2, a3, b0, b1, b2, b3):
    h0 = tl.maximum(a0, b0)
    l0 = tl.minimum(a0, b0)
    h1 = tl.maximum(a2, b2)
    h2 = tl.maximum(h1, l0)
    l2 = tl.minimum(h1, l0)
    h3 = tl.maximum(a1, b1)
    l3 = tl.minimum(a1, b1)
    h4 = tl.maximum(a3, b3)
    h5 = tl.maximum(h4, l3)
    h6 = tl.maximum(h3, h2)
    l6 = tl.minimum(h3, h2)
    h7 = tl.maximum(h5, l2)
    return (h0, h6, l6, h7)


@triton.jit
def _kernel(
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
    picked = tl.zeros((BK,), tl.uint32)
    if K == 8:
        even, odd = tl.split(tl.reshape(packed, (B // 2, 2)))
        p0, p2 = tl.split(tl.reshape(even, (B // 4, 2)))
        p1, p3 = tl.split(tl.reshape(odd, (B // 4, 2)))
        a = tl.maximum(p0, p1)
        b = tl.minimum(p0, p1)
        c = tl.maximum(p2, p3)
        d = tl.minimum(p2, p3)
        u = tl.maximum(b, d)
        v = tl.minimum(a, c)
        local_first = tl.maximum(a, c)
        local_second = tl.maximum(u, v)
        local_third = tl.minimum(u, v)
        local_fourth = tl.minimum(b, d)
        z = tl.zeros((B // 4,), tl.uint32)
        best0, best1, best2, best3, best4, best5, best6, best7 = tl.reduce(
            (local_first, local_second, local_third, local_fourth, z, z, z, z),
            0,
            _top8,
        )
        picked = tl.where(slots == 0, best0, picked)
        picked = tl.where(slots == 1, best1, picked)
        picked = tl.where(slots == 2, best2, picked)
        picked = tl.where(slots == 3, best3, picked)
        picked = tl.where(slots == 4, best4, picked)
        picked = tl.where(slots == 5, best5, picked)
        picked = tl.where(slots == 6, best6, picked)
        picked = tl.where(slots == 7, best7, picked)
        top = best7
    elif K >= 4:
        even, odd = tl.split(tl.reshape(packed, (B // 2, 2)))
        p0, p2 = tl.split(tl.reshape(even, (B // 4, 2)))
        p1, p3 = tl.split(tl.reshape(odd, (B // 4, 2)))
        a = tl.maximum(p0, p1)
        b = tl.minimum(p0, p1)
        c = tl.maximum(p2, p3)
        d = tl.minimum(p2, p3)
        u = tl.maximum(b, d)
        v = tl.minimum(a, c)
        local_first = tl.maximum(a, c)
        local_second = tl.maximum(u, v)
        local_third = tl.minimum(u, v)
        local_fourth = tl.minimum(b, d)
        for four_step in tl.static_range(K // 4):
            first, second, third, fourth = tl.reduce(
                (local_first, local_second, local_third, local_fourth),
                0,
                _top4,
            )
            picked = tl.where(slots == four_step * 4, first, picked)
            picked = tl.where(slots == four_step * 4 + 1, second, picked)
            picked = tl.where(slots == four_step * 4 + 2, third, picked)
            picked = tl.where(slots == four_step * 4 + 3, fourth, picked)
            packed = tl.where(packed >= fourth, 0, packed)
            removed = (
                (local_first >= fourth).to(tl.int32)
                + (local_second >= fourth).to(tl.int32)
                + (local_third >= fourth).to(tl.int32)
                + (local_fourth >= fourth).to(tl.int32)
            )
            next_first = tl.where(
                removed == 0,
                local_first,
                tl.where(
                    removed == 1,
                    local_second,
                    tl.where(
                        removed == 2,
                        local_third,
                        tl.where(removed == 3, local_fourth, 0),
                    ),
                ),
            )
            next_second = tl.where(
                removed == 0,
                local_second,
                tl.where(
                    removed == 1,
                    local_third,
                    tl.where(removed == 2, local_fourth, 0),
                ),
            )
            next_third = tl.where(
                removed == 0,
                local_third,
                tl.where(removed == 1, local_fourth, 0),
            )
            next_fourth = tl.where(removed == 0, local_fourth, 0)
            local_first, local_second, local_third, local_fourth = (
                next_first,
                next_second,
                next_third,
                next_fourth,
            )
            top = fourth
    for tail_step in tl.static_range(K % 4):
        top = tl.max(packed, 0)
        picked = tl.where(slots == K // 4 * 4 + tail_step, top, picked)
        packed = tl.where(packed == top, 0, packed)
    picked_ids = (B - 1 - (picked & B - 1)).to(tl.int32)
    picked_keys = (peak.to(tl.uint32) - (limit - (picked >> LOG_B))).to(
        tl.int32
    )
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


def gemma4_fused_routing(gating_output, per_expert_scale, topk):
    t, e = gating_output.shape
    weights = gating_output.new_empty(t, topk, dtype=torch.float32)
    ids = gating_output.new_empty(t, topk, dtype=torch.int32)
    if t == 0 or topk == 0:
        return (weights, ids)
    _kernel[t,](
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
