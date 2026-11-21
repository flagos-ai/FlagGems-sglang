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


@triton.jit
def _direct4(
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
    cols4 = tl.arange(0, B // 4) * 4
    slots = tl.arange(0, BK)
    limit: tl.constexpr = 4294967295 // B
    cols0 = cols4 + 0
    vals0 = tl.load(
        X + row * SR + cols0 * SC, cols0 < E, other=-float("inf")
    ).to(tl.float32)
    bits0 = vals0.to(tl.int32, bitcast=True)
    keys0 = tl.where(bits0 >= 0, bits0, bits0 ^ 2147483647)
    keys0 = tl.where(vals0 == 0.0, 0, keys0)
    cols1 = cols4 + 1
    vals1 = tl.load(
        X + row * SR + cols1 * SC, cols1 < E, other=-float("inf")
    ).to(tl.float32)
    bits1 = vals1.to(tl.int32, bitcast=True)
    keys1 = tl.where(bits1 >= 0, bits1, bits1 ^ 2147483647)
    keys1 = tl.where(vals1 == 0.0, 0, keys1)
    cols2 = cols4 + 2
    vals2 = tl.load(
        X + row * SR + cols2 * SC, cols2 < E, other=-float("inf")
    ).to(tl.float32)
    bits2 = vals2.to(tl.int32, bitcast=True)
    keys2 = tl.where(bits2 >= 0, bits2, bits2 ^ 2147483647)
    keys2 = tl.where(vals2 == 0.0, 0, keys2)
    cols3 = cols4 + 3
    vals3 = tl.load(
        X + row * SR + cols3 * SC, cols3 < E, other=-float("inf")
    ).to(tl.float32)
    bits3 = vals3.to(tl.int32, bitcast=True)
    keys3 = tl.where(bits3 >= 0, bits3, bits3 ^ 2147483647)
    keys3 = tl.where(vals3 == 0.0, 0, keys3)
    local_peak = tl.maximum(tl.maximum(keys0, keys1), tl.maximum(keys2, keys3))
    peak = tl.max(local_peak, 0)
    delta0 = peak.to(tl.uint32) - keys0.to(tl.uint32)
    p0 = (limit - delta0) * B | (B - 1 - cols0).to(tl.uint32)
    p0 = tl.where((delta0 <= limit) & (cols0 < E), p0, 0).to(tl.uint32)
    delta1 = peak.to(tl.uint32) - keys1.to(tl.uint32)
    p1 = (limit - delta1) * B | (B - 1 - cols1).to(tl.uint32)
    p1 = tl.where((delta1 <= limit) & (cols1 < E), p1, 0).to(tl.uint32)
    delta2 = peak.to(tl.uint32) - keys2.to(tl.uint32)
    p2 = (limit - delta2) * B | (B - 1 - cols2).to(tl.uint32)
    p2 = tl.where((delta2 <= limit) & (cols2 < E), p2, 0).to(tl.uint32)
    delta3 = peak.to(tl.uint32) - keys3.to(tl.uint32)
    p3 = (limit - delta3) * B | (B - 1 - cols3).to(tl.uint32)
    p3 = tl.where((delta3 <= limit) & (cols3 < E), p3, 0).to(tl.uint32)
    picked = tl.zeros((BK,), tl.uint32)
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
    picked_ids = (B - 1 - (picked & B - 1)).to(tl.int32)
    picked_keys = (peak.to(tl.uint32) - (limit - (picked >> LOG_B))).to(
        tl.int32
    )
    if top == 0:
        cols = tl.arange(0, B)
        vals = tl.load(
            X + row * SR + cols * SC, cols < E, other=-float("inf")
        ).to(tl.float32)
        bits = vals.to(tl.int32, bitcast=True)
        keys = tl.where(bits >= 0, bits, bits ^ 2147483647)
        keys = tl.where(vals == 0.0, 0, keys)
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


@triton.jit
def _direct4_rows(
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
    T: tl.constexpr,
    R: tl.constexpr,
):
    row = tl.program_id(0) * R + tl.arange(0, R)
    cols4 = tl.arange(0, B // 4) * 4
    slots = tl.arange(0, BK)
    limit: tl.constexpr = 4294967295 // B
    cols0 = (cols4 + 0)[None, :]
    vals0 = tl.load(
        X + row[:, None] * SR + cols0 * SC,
        (cols0 < E) & (row[:, None] < T),
        other=-float("inf"),
    ).to(tl.float32)
    bits0 = vals0.to(tl.int32, bitcast=True)
    keys0 = tl.where(bits0 >= 0, bits0, bits0 ^ 2147483647)
    keys0 = tl.where(vals0 == 0.0, 0, keys0)
    cols1 = (cols4 + 1)[None, :]
    vals1 = tl.load(
        X + row[:, None] * SR + cols1 * SC,
        (cols1 < E) & (row[:, None] < T),
        other=-float("inf"),
    ).to(tl.float32)
    bits1 = vals1.to(tl.int32, bitcast=True)
    keys1 = tl.where(bits1 >= 0, bits1, bits1 ^ 2147483647)
    keys1 = tl.where(vals1 == 0.0, 0, keys1)
    cols2 = (cols4 + 2)[None, :]
    vals2 = tl.load(
        X + row[:, None] * SR + cols2 * SC,
        (cols2 < E) & (row[:, None] < T),
        other=-float("inf"),
    ).to(tl.float32)
    bits2 = vals2.to(tl.int32, bitcast=True)
    keys2 = tl.where(bits2 >= 0, bits2, bits2 ^ 2147483647)
    keys2 = tl.where(vals2 == 0.0, 0, keys2)
    cols3 = (cols4 + 3)[None, :]
    vals3 = tl.load(
        X + row[:, None] * SR + cols3 * SC,
        (cols3 < E) & (row[:, None] < T),
        other=-float("inf"),
    ).to(tl.float32)
    bits3 = vals3.to(tl.int32, bitcast=True)
    keys3 = tl.where(bits3 >= 0, bits3, bits3 ^ 2147483647)
    keys3 = tl.where(vals3 == 0.0, 0, keys3)
    local_peak = tl.maximum(tl.maximum(keys0, keys1), tl.maximum(keys2, keys3))
    peak = tl.max(local_peak, 1)
    delta0 = peak[:, None].to(tl.uint32) - keys0.to(tl.uint32)
    p0 = (limit - delta0) * B | (B - 1 - cols0).to(tl.uint32)
    p0 = tl.where(
        (delta0 <= limit) & ((cols0 < E) & (row[:, None] < T)), p0, 0
    ).to(tl.uint32)
    delta1 = peak[:, None].to(tl.uint32) - keys1.to(tl.uint32)
    p1 = (limit - delta1) * B | (B - 1 - cols1).to(tl.uint32)
    p1 = tl.where(
        (delta1 <= limit) & ((cols1 < E) & (row[:, None] < T)), p1, 0
    ).to(tl.uint32)
    delta2 = peak[:, None].to(tl.uint32) - keys2.to(tl.uint32)
    p2 = (limit - delta2) * B | (B - 1 - cols2).to(tl.uint32)
    p2 = tl.where(
        (delta2 <= limit) & ((cols2 < E) & (row[:, None] < T)), p2, 0
    ).to(tl.uint32)
    delta3 = peak[:, None].to(tl.uint32) - keys3.to(tl.uint32)
    p3 = (limit - delta3) * B | (B - 1 - cols3).to(tl.uint32)
    p3 = tl.where(
        (delta3 <= limit) & ((cols3 < E) & (row[:, None] < T)), p3, 0
    ).to(tl.uint32)
    picked = tl.zeros((R, BK), tl.uint32)
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
    z = tl.zeros((R, B // 4), tl.uint32)
    best0, best1, best2, best3, best4, best5, best6, best7 = tl.reduce(
        (local_first, local_second, local_third, local_fourth, z, z, z, z),
        1,
        _top8,
    )
    picked = tl.where(slots[None, :] == 0, best0[:, None], picked)
    picked = tl.where(slots[None, :] == 1, best1[:, None], picked)
    picked = tl.where(slots[None, :] == 2, best2[:, None], picked)
    picked = tl.where(slots[None, :] == 3, best3[:, None], picked)
    picked = tl.where(slots[None, :] == 4, best4[:, None], picked)
    picked = tl.where(slots[None, :] == 5, best5[:, None], picked)
    picked = tl.where(slots[None, :] == 6, best6[:, None], picked)
    picked = tl.where(slots[None, :] == 7, best7[:, None], picked)
    top = best7
    picked_ids = (B - 1 - (picked & B - 1)).to(tl.int32)
    picked_keys = (
        peak[:, None].to(tl.uint32) - (limit - (picked >> LOG_B))
    ).to(tl.int32)
    if tl.sum((top == 0).to(tl.int32), 0) > 0:
        cols = tl.arange(0, B)[None, :]
        vals = tl.load(
            X + row[:, None] * SR + cols * SC,
            (cols < E) & (row[:, None] < T),
            other=-float("inf"),
        ).to(tl.float32)
        bits = vals.to(tl.int32, bitcast=True)
        keys = tl.where(bits >= 0, bits, bits ^ 2147483647)
        keys = tl.where(vals == 0.0, 0, keys)
        active = keys
        for k in tl.static_range(K):
            best_key = tl.max(active, 1)
            pos = tl.min(
                tl.where(
                    (active == best_key[:, None])
                    & ((cols < E) & (row[:, None] < T)),
                    cols,
                    B,
                ),
                1,
            )
            picked_ids = tl.where(
                slots[None, :] == k, pos[:, None], picked_ids
            )
            picked_keys = tl.where(
                slots[None, :] == k, best_key[:, None], picked_keys
            )
            active = tl.where(cols == pos[:, None], -2147483648, active)
    picked_bits = tl.where(
        picked_keys >= 0, picked_keys, picked_keys ^ 2147483647
    )
    best = picked_bits.to(tl.float32, bitcast=True)
    peak_bits = tl.where(peak >= 0, peak, peak ^ 2147483647)
    best_peak = peak_bits.to(tl.float32, bitcast=True)
    expo = tl.where(
        (slots[None, :] < K) & (row[:, None] < T),
        tl.exp(best - best_peak[:, None]),
        0.0,
    )
    weights = expo / tl.sum(expo, 1)[:, None]
    weights *= tl.load(
        S + picked_ids, (slots[None, :] < K) & (row[:, None] < T), other=0
    ).to(tl.float32)
    out = row[:, None] * K + slots[None, :]
    tl.store(W + out, weights, (slots[None, :] < K) & (row[:, None] < T))
    tl.store(I + out, picked_ids, (slots[None, :] < K) & (row[:, None] < T))


@triton.jit
def _direct8_rows(
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
    T: tl.constexpr,
    R: tl.constexpr,
):
    row = tl.program_id(0) * R + tl.arange(0, R)
    cols8 = tl.arange(0, B // 8) * 8
    slots = tl.arange(0, BK)
    limit: tl.constexpr = 4294967295 // B
    cols0 = (cols8 + 0)[None, :]
    vals0 = tl.load(
        X + row[:, None] * SR + cols0 * SC,
        (cols0 < E) & (row[:, None] < T),
        other=-float("inf"),
    ).to(tl.float32)
    bits0 = vals0.to(tl.int32, bitcast=True)
    keys0 = tl.where(bits0 >= 0, bits0, bits0 ^ 2147483647)
    keys0 = tl.where(vals0 == 0.0, 0, keys0)
    cols1 = (cols8 + 1)[None, :]
    vals1 = tl.load(
        X + row[:, None] * SR + cols1 * SC,
        (cols1 < E) & (row[:, None] < T),
        other=-float("inf"),
    ).to(tl.float32)
    bits1 = vals1.to(tl.int32, bitcast=True)
    keys1 = tl.where(bits1 >= 0, bits1, bits1 ^ 2147483647)
    keys1 = tl.where(vals1 == 0.0, 0, keys1)
    cols2 = (cols8 + 2)[None, :]
    vals2 = tl.load(
        X + row[:, None] * SR + cols2 * SC,
        (cols2 < E) & (row[:, None] < T),
        other=-float("inf"),
    ).to(tl.float32)
    bits2 = vals2.to(tl.int32, bitcast=True)
    keys2 = tl.where(bits2 >= 0, bits2, bits2 ^ 2147483647)
    keys2 = tl.where(vals2 == 0.0, 0, keys2)
    cols3 = (cols8 + 3)[None, :]
    vals3 = tl.load(
        X + row[:, None] * SR + cols3 * SC,
        (cols3 < E) & (row[:, None] < T),
        other=-float("inf"),
    ).to(tl.float32)
    bits3 = vals3.to(tl.int32, bitcast=True)
    keys3 = tl.where(bits3 >= 0, bits3, bits3 ^ 2147483647)
    keys3 = tl.where(vals3 == 0.0, 0, keys3)
    cols4 = (cols8 + 4)[None, :]
    vals4 = tl.load(
        X + row[:, None] * SR + cols4 * SC,
        (cols4 < E) & (row[:, None] < T),
        other=-float("inf"),
    ).to(tl.float32)
    bits4 = vals4.to(tl.int32, bitcast=True)
    keys4 = tl.where(bits4 >= 0, bits4, bits4 ^ 2147483647)
    keys4 = tl.where(vals4 == 0.0, 0, keys4)
    cols5 = (cols8 + 5)[None, :]
    vals5 = tl.load(
        X + row[:, None] * SR + cols5 * SC,
        (cols5 < E) & (row[:, None] < T),
        other=-float("inf"),
    ).to(tl.float32)
    bits5 = vals5.to(tl.int32, bitcast=True)
    keys5 = tl.where(bits5 >= 0, bits5, bits5 ^ 2147483647)
    keys5 = tl.where(vals5 == 0.0, 0, keys5)
    cols6 = (cols8 + 6)[None, :]
    vals6 = tl.load(
        X + row[:, None] * SR + cols6 * SC,
        (cols6 < E) & (row[:, None] < T),
        other=-float("inf"),
    ).to(tl.float32)
    bits6 = vals6.to(tl.int32, bitcast=True)
    keys6 = tl.where(bits6 >= 0, bits6, bits6 ^ 2147483647)
    keys6 = tl.where(vals6 == 0.0, 0, keys6)
    cols7 = (cols8 + 7)[None, :]
    vals7 = tl.load(
        X + row[:, None] * SR + cols7 * SC,
        (cols7 < E) & (row[:, None] < T),
        other=-float("inf"),
    ).to(tl.float32)
    bits7 = vals7.to(tl.int32, bitcast=True)
    keys7 = tl.where(bits7 >= 0, bits7, bits7 ^ 2147483647)
    keys7 = tl.where(vals7 == 0.0, 0, keys7)
    local_peak = tl.maximum(
        tl.maximum(tl.maximum(keys0, keys1), tl.maximum(keys2, keys3)),
        tl.maximum(tl.maximum(keys4, keys5), tl.maximum(keys6, keys7)),
    )
    peak = tl.max(local_peak, 1)
    delta0 = peak[:, None].to(tl.uint32) - keys0.to(tl.uint32)
    p0 = (limit - delta0) * B | (B - 1 - cols0).to(tl.uint32)
    p0 = tl.where(
        (delta0 <= limit) & (cols0 < E) & (row[:, None] < T), p0, 0
    ).to(tl.uint32)
    delta1 = peak[:, None].to(tl.uint32) - keys1.to(tl.uint32)
    p1 = (limit - delta1) * B | (B - 1 - cols1).to(tl.uint32)
    p1 = tl.where(
        (delta1 <= limit) & (cols1 < E) & (row[:, None] < T), p1, 0
    ).to(tl.uint32)
    delta2 = peak[:, None].to(tl.uint32) - keys2.to(tl.uint32)
    p2 = (limit - delta2) * B | (B - 1 - cols2).to(tl.uint32)
    p2 = tl.where(
        (delta2 <= limit) & (cols2 < E) & (row[:, None] < T), p2, 0
    ).to(tl.uint32)
    delta3 = peak[:, None].to(tl.uint32) - keys3.to(tl.uint32)
    p3 = (limit - delta3) * B | (B - 1 - cols3).to(tl.uint32)
    p3 = tl.where(
        (delta3 <= limit) & (cols3 < E) & (row[:, None] < T), p3, 0
    ).to(tl.uint32)
    delta4 = peak[:, None].to(tl.uint32) - keys4.to(tl.uint32)
    p4 = (limit - delta4) * B | (B - 1 - cols4).to(tl.uint32)
    p4 = tl.where(
        (delta4 <= limit) & (cols4 < E) & (row[:, None] < T), p4, 0
    ).to(tl.uint32)
    delta5 = peak[:, None].to(tl.uint32) - keys5.to(tl.uint32)
    p5 = (limit - delta5) * B | (B - 1 - cols5).to(tl.uint32)
    p5 = tl.where(
        (delta5 <= limit) & (cols5 < E) & (row[:, None] < T), p5, 0
    ).to(tl.uint32)
    delta6 = peak[:, None].to(tl.uint32) - keys6.to(tl.uint32)
    p6 = (limit - delta6) * B | (B - 1 - cols6).to(tl.uint32)
    p6 = tl.where(
        (delta6 <= limit) & (cols6 < E) & (row[:, None] < T), p6, 0
    ).to(tl.uint32)
    delta7 = peak[:, None].to(tl.uint32) - keys7.to(tl.uint32)
    p7 = (limit - delta7) * B | (B - 1 - cols7).to(tl.uint32)
    p7 = tl.where(
        (delta7 <= limit) & (cols7 < E) & (row[:, None] < T), p7, 0
    ).to(tl.uint32)
    picked = tl.zeros((R, BK), tl.uint32)
    h0 = tl.maximum(p0, p1)
    l0 = tl.minimum(p0, p1)
    h1 = tl.maximum(p2, p3)
    l1 = tl.minimum(p2, p3)
    h2 = tl.maximum(h0, h1)
    l2 = tl.minimum(h0, h1)
    h3 = tl.maximum(l0, l1)
    l3 = tl.minimum(l0, l1)
    h4 = tl.maximum(h3, l2)
    l4 = tl.minimum(h3, l2)
    h5 = tl.maximum(p4, p5)
    l5 = tl.minimum(p4, p5)
    h6 = tl.maximum(p6, p7)
    l6 = tl.minimum(p6, p7)
    h7 = tl.maximum(h5, h6)
    l7 = tl.minimum(h5, h6)
    h8 = tl.maximum(l5, l6)
    l8 = tl.minimum(l5, l6)
    h9 = tl.maximum(h8, l7)
    l9 = tl.minimum(h8, l7)
    h10 = tl.maximum(h2, h7)
    l10 = tl.minimum(h2, h7)
    h11 = tl.maximum(l4, l9)
    l11 = tl.minimum(l4, l9)
    h12 = tl.maximum(h11, l10)
    l12 = tl.minimum(h11, l10)
    h13 = tl.maximum(h4, h9)
    l13 = tl.minimum(h4, h9)
    h14 = tl.maximum(l3, l8)
    l14 = tl.minimum(l3, l8)
    h15 = tl.maximum(h14, l13)
    l15 = tl.minimum(h14, l13)
    h16 = tl.maximum(h13, h12)
    l16 = tl.minimum(h13, h12)
    h17 = tl.maximum(h15, l12)
    l17 = tl.minimum(h15, l12)
    h18 = tl.maximum(l15, l11)
    l18 = tl.minimum(l15, l11)
    best0, best1, best2, best3, best4, best5, best6, best7 = tl.reduce(
        (h10, h16, l16, h17, l17, h18, l18, l14), 1, _top8
    )
    picked = tl.where(slots[None, :] == 0, best0[:, None], picked)
    picked = tl.where(slots[None, :] == 1, best1[:, None], picked)
    picked = tl.where(slots[None, :] == 2, best2[:, None], picked)
    picked = tl.where(slots[None, :] == 3, best3[:, None], picked)
    picked = tl.where(slots[None, :] == 4, best4[:, None], picked)
    picked = tl.where(slots[None, :] == 5, best5[:, None], picked)
    picked = tl.where(slots[None, :] == 6, best6[:, None], picked)
    picked = tl.where(slots[None, :] == 7, best7[:, None], picked)
    top = best7
    picked_ids = (B - 1 - (picked & B - 1)).to(tl.int32)
    picked_keys = (
        peak[:, None].to(tl.uint32) - (limit - (picked >> LOG_B))
    ).to(tl.int32)
    if tl.sum((top == 0).to(tl.int32), 0) > 0:
        cols = tl.arange(0, B)[None, :]
        vals = tl.load(
            X + row[:, None] * SR + cols * SC,
            (cols < E) & (row[:, None] < T),
            other=-float("inf"),
        ).to(tl.float32)
        bits = vals.to(tl.int32, bitcast=True)
        keys = tl.where(bits >= 0, bits, bits ^ 2147483647)
        keys = tl.where(vals == 0.0, 0, keys)
        active = keys
        for k in tl.static_range(K):
            best_key = tl.max(active, 1)
            pos = tl.min(
                tl.where(
                    (active == best_key[:, None])
                    & ((cols < E) & (row[:, None] < T)),
                    cols,
                    B,
                ),
                1,
            )
            picked_ids = tl.where(
                slots[None, :] == k, pos[:, None], picked_ids
            )
            picked_keys = tl.where(
                slots[None, :] == k, best_key[:, None], picked_keys
            )
            active = tl.where(cols == pos[:, None], -2147483648, active)
    picked_bits = tl.where(
        picked_keys >= 0, picked_keys, picked_keys ^ 2147483647
    )
    best = picked_bits.to(tl.float32, bitcast=True)
    peak_bits = tl.where(peak >= 0, peak, peak ^ 2147483647)
    best_peak = peak_bits.to(tl.float32, bitcast=True)
    expo = tl.where(
        (slots[None, :] < K) & (row[:, None] < T),
        tl.exp(best - best_peak[:, None]),
        0.0,
    )
    weights = expo / tl.sum(expo, 1)[:, None]
    weights *= tl.load(
        S + picked_ids, (slots[None, :] < K) & (row[:, None] < T), other=0
    ).to(tl.float32)
    out = row[:, None] * K + slots[None, :]
    tl.store(W + out, weights, (slots[None, :] < K) & (row[:, None] < T))
    tl.store(I + out, picked_ids, (slots[None, :] < K) & (row[:, None] < T))


def gemma4_fused_routing(gating_output, per_expert_scale, topk):
    t, e = gating_output.shape
    weights = gating_output.new_empty(t, topk, dtype=torch.float32)
    ids = gating_output.new_empty(t, topk, dtype=torch.int32)
    if t == 0 or topk == 0:
        return (weights, ids)
    r = max(1, 256 // triton.next_power_of_2(e))
    r8 = max(1, 512 // triton.next_power_of_2(e))
    if topk == 8 and e == 128 and (t >= 512):
        _direct8_rows[triton.cdiv(t, r8),](
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
            t,
            r8,
            num_warps=1,
            num_stages=1,
        )
    elif topk == 8 and t >= 512 and (r > 1):
        _direct4_rows[triton.cdiv(t, r),](
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
            t,
            r,
            num_warps=1,
            num_stages=1,
        )
    elif topk == 8 and e >= 8:
        _direct4[t,](
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
    else:
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
