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
def _sum8(a, b):
    return (a + b).to(tl.int8)


@triton.jit
def _sum16(a, b):
    return (a + b).to(tl.int16)


@triton.jit
def _stream_select(
    P,
    K,
    O,
    V: tl.constexpr,
    KS: tl.constexpr,
    KT: tl.constexpr,
    SINGLE: tl.constexpr,
    KV: tl.constexpr,
    B: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    a = tl.arange(0, B)
    if KT:
        ki = 0 if SINGLE else row * KS
        k = tl.load(K + ki).to(tl.int32)
    else:
        k = tl.full((), KV, tl.int32)
    lo = tl.full((), 2147483647, tl.int32)
    hi = tl.full((), 0, tl.int32)
    for start in range(triton.cdiv(V, B)):
        c = start * B + a
        x = tl.load(P + row * V + c, c < V, 0)
        bits = x.to(tl.int32, bitcast=True) & 2147483647
        lo = tl.minimum(lo, tl.min(tl.where(c < V, bits, 2147483647), 0))
        hi = tl.maximum(hi, tl.max(bits, 0))
    if k == 1:
        lo = hi
    if k == V:
        hi = lo
    nlo = tl.full((), V, tl.int32)
    nhi = tl.full((), 0, tl.int32)
    step = tl.full((), 0, tl.int32)
    while lo < hi:
        lf = lo.to(tl.float32, bitcast=True)
        hf = hi.to(tl.float32, bitcast=True)
        fraction = (nlo - k).to(tl.float32) / tl.maximum(nlo - nhi, 1)
        guess = (lf + (hf - lf) * fraction).to(tl.int32, bitcast=True)
        mid = tl.minimum(hi, tl.maximum(lo + 1, guess))
        mid = tl.where(step % 4 == 3, lo + (hi - lo + 1) // 2, mid)
        counts = tl.full((B,), 0, tl.int32)
        lower = tl.full((), 0, tl.int32)
        upper = tl.full((), 2147483647, tl.int32)
        for start in range(triton.cdiv(V, B)):
            c = start * B + a
            x = tl.load(P + row * V + c, c < V, 0)
            bits = x.to(tl.int32, bitcast=True) & 2147483647
            above = (c < V) & (bits >= mid)
            counts += above.to(tl.int32)
            lower = tl.maximum(
                lower, tl.max(tl.where((c < V) & (bits < mid), bits, 0), 0)
            )
            upper = tl.minimum(
                upper, tl.min(tl.where(above, bits, 2147483647), 0)
            )
        count = tl.sum(counts, 0)
        lo = tl.where(count >= k, upper, lo)
        hi = tl.where(count <= k, tl.where(count == k, upper, lower), hi)
        nlo = tl.where(count >= k, count, nlo)
        nhi = tl.where(count < k, count, nhi)
        step += 1
    pivot = lo.to(tl.float32, bitcast=True)
    sums = tl.full((B,), 0.0, tl.float32)
    for start in range(triton.cdiv(V, B)):
        c = start * B + a
        x = tl.load(P + row * V + c, c < V, 0)
        sums += tl.where(x >= pivot, x, 0.0)
    denominator = tl.sum(sums, 0)
    return (pivot, denominator)


@triton.jit
def _stream_select_wide(
    P,
    K,
    O,
    V: tl.constexpr,
    KS: tl.constexpr,
    KT: tl.constexpr,
    SINGLE: tl.constexpr,
    KV: tl.constexpr,
    B: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    a = tl.arange(0, B)
    if KT:
        ki = 0 if SINGLE else row * KS
        k = tl.load(K + ki).to(tl.int64)
    else:
        k = tl.full((), KV, tl.int64)
    lo = tl.full((), 2147483647, tl.int32)
    hi = tl.full((), 0, tl.int32)
    for start in range(triton.cdiv(V, B)):
        c = start.to(tl.int64) * B + a
        x = tl.load(P + row * V + c, c < V, 0)
        bits = x.to(tl.int32, bitcast=True) & 2147483647
        lo = tl.minimum(lo, tl.min(tl.where(c < V, bits, 2147483647), 0))
        hi = tl.maximum(hi, tl.max(bits, 0))
    if k == 1:
        lo = hi
    if k == V:
        hi = lo
    nlo = tl.full((), V, tl.int64)
    nhi = tl.full((), 0, tl.int64)
    step = tl.full((), 0, tl.int32)
    while lo < hi:
        lf = lo.to(tl.float32, bitcast=True)
        hf = hi.to(tl.float32, bitcast=True)
        fraction = (nlo - k).to(tl.float32) / tl.maximum(nlo - nhi, 1)
        guess = (lf + (hf - lf) * fraction).to(tl.int32, bitcast=True)
        mid = tl.minimum(hi, tl.maximum(lo + 1, guess))
        mid = tl.where(step % 4 == 3, lo + (hi - lo + 1) // 2, mid)
        counts = tl.full((B,), 0, tl.int64)
        lower = tl.full((), 0, tl.int32)
        upper = tl.full((), 2147483647, tl.int32)
        for start in range(triton.cdiv(V, B)):
            c = start.to(tl.int64) * B + a
            x = tl.load(P + row * V + c, c < V, 0)
            bits = x.to(tl.int32, bitcast=True) & 2147483647
            above = (c < V) & (bits >= mid)
            counts += above.to(tl.int32)
            lower = tl.maximum(
                lower, tl.max(tl.where((c < V) & (bits < mid), bits, 0), 0)
            )
            upper = tl.minimum(
                upper, tl.min(tl.where(above, bits, 2147483647), 0)
            )
        count = tl.sum(counts, 0)
        lo = tl.where(count >= k, upper, lo)
        hi = tl.where(count <= k, tl.where(count == k, upper, lower), hi)
        nlo = tl.where(count >= k, count, nlo)
        nhi = tl.where(count < k, count, nhi)
        step += 1
    pivot = lo.to(tl.float32, bitcast=True)
    sums = tl.full((B,), 0.0, tl.float32)
    correction = tl.full((B,), 0.0, tl.float32)
    for start in range(triton.cdiv(V, B)):
        c = start.to(tl.int64) * B + a
        x = tl.load(P + row * V + c, c < V, 0)
        value = tl.where(x >= pivot, x, 0.0) - correction
        updated = sums + value
        correction = updated - sums - value
        sums = updated
    denominator = tl.sum(sums, 0)
    return (pivot, denominator)


@triton.jit
def _stats(ac, al, au, bc, bl, bu):
    return (ac + bc, tl.maximum(al, bl), tl.minimum(au, bu))


@triton.jit
def _search(x, valid, k, W: tl.constexpr):
    if W <= 512:
        ordered = tl.sort(tl.where(valid, x, float("-inf")), descending=True)
        pivot = tl.gather(ordered, tl.full((1,), 0, tl.int32) + k - 1, 0).sum(
            0
        )
        kept = tl.where(valid & (x >= pivot), x, 0.0)
        return (pivot, tl.sum(kept, 0))
    else:
        bits = x.to(tl.int32, bitcast=True) & 2147483647
        nlo, hi, lo = tl.reduce(
            (
                valid.to(tl.int32),
                tl.where(valid, bits, 0),
                tl.where(valid, bits, 2147483647),
            ),
            0,
            _stats,
        )
        nhi = tl.full((), 0, tl.int32)
        step = tl.full((), 0, tl.int32)
        while lo < hi:
            lf = lo.to(tl.float32, bitcast=True)
            hf = hi.to(tl.float32, bitcast=True)
            fraction = (nlo - k).to(tl.float32) / tl.maximum(nlo - nhi, 1)
            guess = (lf + (hf - lf) * fraction).to(tl.int32, bitcast=True)
            mid = tl.minimum(hi, tl.maximum(lo + 1, guess))
            mid = tl.where(step % 4 == 3, lo + (hi - lo + 1) // 2, mid)
            above = valid & (bits >= mid)
            count, lower, upper = tl.reduce(
                (
                    above.to(tl.int32),
                    tl.where(valid & (bits < mid), bits, 0),
                    tl.where(above, bits, 2147483647),
                ),
                0,
                _stats,
            )
            lo = tl.where(count >= k, upper, lo)
            hi = tl.where(count <= k, tl.where(count == k, upper, lower), hi)
            nlo = tl.where(count >= k, count, nlo)
            nhi = tl.where(count < k, count, nhi)
            step += 1
        kept = tl.where(valid & (bits >= lo), x, 0.0)
        return (lo.to(tl.float32, bitcast=True), tl.sum(kept, 0))


@triton.jit
def _small(P, K, Out, V: tl.constexpr, KS: tl.constexpr, KV: tl.constexpr):
    KT: tl.constexpr = KV == 0
    SINGLE: tl.constexpr = KS == 0
    W: tl.constexpr = triton.next_power_of_2(V)
    row = tl.program_id(0).to(tl.int64)
    c = tl.arange(0, W)
    x = tl.load(P + row * V + c, c < V, 0.0)
    if KT:
        k = tl.load(K + (0 if SINGLE else row * KS)).to(tl.int32)
    else:
        k = tl.full((), KV, tl.int32)
    if not KT and KV == 1:
        pivot = tl.max(x, 0)
        total = tl.sum(tl.where(x >= pivot, x, 0.0), 0)
    elif not KT and KV == V:
        pivot = tl.full((), 0.0, tl.float32)
        total = tl.sum(x, 0)
    elif V <= 1024:
        ordered = tl.sort(tl.where(c < V, x, float("-inf")), descending=True)
        pivot = tl.sum(tl.where(c == k - 1, ordered, 0.0), 0)
        total = tl.sum(tl.where(x >= pivot, x, 0.0), 0)
    else:
        pivot, total = _search(x, c < V, k, W)
    tl.store(Out + row * V + c, tl.where(x >= pivot, x / total, 0.0), c < V)


@triton.jit
def _filter(P, K, S, Out, V: tl.constexpr, KS: tl.constexpr, KV: tl.constexpr):
    KT: tl.constexpr = KV == 0
    SINGLE: tl.constexpr = KS == 0
    W: tl.constexpr = 2048
    T: tl.constexpr = triton.cdiv(V, W)
    CAP: tl.constexpr = 32
    C = S.to(tl.pointer_type(tl.int32))
    S = S + tl.num_programs(1).to(tl.int64) * T
    row = tl.program_id(1).to(tl.int64)
    tile = tl.program_id(0)
    a = tl.arange(0, W)
    if KT:
        k = tl.load(K + (0 if SINGLE else row * KS)).to(tl.int32)
    else:
        k = tl.full((), KV, tl.int32)
    c = tile * W + a
    x = tl.load(P + row * V + tl.minimum(c, V - 1))
    maximum = tl.max(x.to(tl.int32, bitcast=True) & 2147483647, 0).to(
        tl.float32, bitcast=True
    )
    cutoff = maximum * tl.maximum(
        0.0, 1.0 - (1.125 if V > 65536 else 1.5) * k / V
    )
    active = (c < V) & (x >= cutoff)
    lanes188 = tl.arange(0, 32)
    matrix188 = active.to(tl.uint32).reshape((W // 32, 32))
    masks188 = tl.sum(matrix188 << lanes188[None, :], 1).to(tl.uint32)
    totals188 = _popcount188(masks188)
    count = tl.sum(totals188, 0)
    tl.store(C + row * T * 2 + tile, count)
    tl.store(
        C.to(tl.pointer_type(tl.float32)) + row * T * 2 + T + tile, cutoff
    )
    if count <= CAP:
        ends188 = tl.cumsum(totals188, 0)
        q188 = tl.arange(0, CAP)
        first188 = tl.full((CAP,), 0, tl.int32)
        last188 = tl.full((CAP,), W // 32 - 1, tl.int32)
        for iteration188 in tl.static_range(6):
            middle188 = (first188 + last188) // 2
            before188 = tl.gather(ends188, middle188, 0) <= q188
            first188 = tl.where(before188, middle188 + 1, first188)
            last188 = tl.where(before188, last188, middle188)
        group188 = tl.minimum(first188, W // 32 - 1)
        prior188 = tl.where(
            group188 > 0, tl.gather(ends188, tl.maximum(0, group188 - 1), 0), 0
        )
        rank188 = q188 - prior188
        mask188 = tl.gather(masks188, group188, 0)
        lane188 = tl.full((CAP,), 0, tl.int32)
        for shift188 in tl.static_range(4, -1, -1):
            width188 = 1 << shift188
            lowmask188 = (1 << width188) - 1
            low188 = _popcount188(mask188 & lowmask188)
            upper188 = rank188 >= low188
            lane188 += tl.where(upper188, width188, 0)
            rank188 -= tl.where(upper188, low188, 0)
            mask188 = tl.where(
                upper188, mask188 >> width188, mask188 & lowmask188
            )
        local188 = group188 * 32 + lane188
        position188 = tile * W + local188
        if V > 65536:
            value217 = tl.load(P + row * V + position188, q188 < count, 0.0)
            packed217 = position188.to(tl.int64) << 32 | value217.to(
                tl.int32, bitcast=True
            ).to(tl.uint32).to(tl.int64)
            tl.store(
                S + (row * T + tile) * CAP + q188, packed217, q188 < count
            )
        else:
            tl.store(
                S.to(tl.pointer_type(tl.int16))
                + (row * T + tile) * CAP
                + q188,
                (position188 % W).to(tl.int16),
                q188 < count,
            )
    tl.store(Out + row * V + c, tl.full((W,), 0.0, tl.float32), c < V)


@triton.jit
def _compactscatter(
    P,
    K,
    S,
    C,
    D,
    row,
    part,
    k,
    total_count,
    maximum_cutoff,
    V: tl.constexpr,
    KS: tl.constexpr,
    KV: tl.constexpr,
    SIZE: tl.constexpr,
):
    T: tl.constexpr = triton.cdiv(V, 2048)
    CAP: tl.constexpr = 32
    PARTS: tl.constexpr = 1 if V > 8192 else 8
    a = tl.arange(0, SIZE)
    t = tl.arange(0, triton.next_power_of_2(T))
    counts = tl.load(C + row * T * 2 + t, t < T, 0)
    ends = tl.cumsum(counts, 0)
    starts = ends - counts
    lo = tl.full((SIZE,), 0, tl.int32)
    hi = tl.full((SIZE,), T - 1, tl.int32)
    for step in tl.static_range(9):
        if 1 << step < T:
            mid = (lo + hi) // 2
            end = tl.gather(ends, mid, 0)
            above = a >= end
            lo = tl.where(above, mid + 1, lo)
            hi = tl.where(above, hi, mid)
    group = tl.minimum(lo, T - 1)
    offset = a - tl.gather(starts, group, 0)
    index = row * T * CAP + group * CAP + offset
    valid = a < total_count
    positions = (
        tl.load(S.to(tl.pointer_type(tl.int16)) + index, valid, 0).to(tl.int32)
        + group * 2048
    )
    candidates = tl.load(P + row * V + positions, valid, 0.0)
    pivot, total = _search(candidates, valid, k, SIZE)
    if pivot >= maximum_cutoff:
        selected = valid & (a % PARTS == part) & (candidates >= pivot)
        tl.store(D + row * V + positions, candidates / total, selected)
    else:
        pivot, total = _stream_select(
            P, K, D, V, KS, KV == 0, KS == 0, KV, 2048
        )
        for tile in range(part, triton.cdiv(V, 4096), PARTS):
            c = tile * 4096 + tl.arange(0, 4096)
            x = tl.load(P + row * V + c, c < V, 0.0)
            tl.store(
                D + row * V + c, tl.where(x >= pivot, x / total, 0.0), c < V
            )


@triton.jit
def _selectscatter(
    P, K, S, D, V: tl.constexpr, KS: tl.constexpr, KV: tl.constexpr
):
    KT: tl.constexpr = KV == 0
    SINGLE: tl.constexpr = KS == 0
    T: tl.constexpr = triton.cdiv(V, 2048)
    CAP: tl.constexpr = 32
    W: tl.constexpr = triton.next_power_of_2(T * CAP)
    PARTS: tl.constexpr = 1 if V > 8192 else 8
    C = S.to(tl.pointer_type(tl.int32))
    S = S + tl.num_programs(0).to(tl.int64) * T
    row = tl.program_id(0).to(tl.int64)
    part = tl.program_id(1)
    if KT:
        k = tl.load(K + (0 if SINGLE else row * KS)).to(tl.int32)
    else:
        k = tl.full((), KV, tl.int32)
    a = tl.arange(0, W)
    counts = tl.load(C + row * T * 2 + a // CAP, a < T * CAP, 0)
    ta = tl.arange(0, triton.next_power_of_2(T))
    cutoffs = tl.load(
        C.to(tl.pointer_type(tl.float32)) + row * T * 2 + T + ta, ta < T, 0.0
    )
    maximum_cutoff = tl.max(cutoffs, 0)
    counts353 = tl.load(C + row * T * 2 + ta, ta < T, 0)
    encoded353 = tl.sum(
        counts353.to(tl.uint32) + ((counts353 > CAP).to(tl.uint32) << 21), 0
    )
    total_count = (encoded353 & 2097151).to(tl.int32)
    overflow = encoded353 >> 21 != 0
    if (total_count >= k) & ~overflow:
        SIZE: tl.constexpr = 256 if V <= 65536 else 1024
        if W > SIZE and V <= 65536:
            if total_count <= SIZE:
                _compactscatter(
                    P,
                    K,
                    S,
                    C,
                    D,
                    row,
                    part,
                    k,
                    total_count,
                    maximum_cutoff,
                    V,
                    KS,
                    KV,
                    SIZE,
                )
            else:
                valid = (a < T * CAP) & (a % CAP < counts)
                if V > 8192:
                    if V > 65536:
                        packed217 = tl.load(S + row * T * CAP + a, valid, 0)
                        positions = (packed217 >> 32).to(tl.int32)
                        candidates = (
                            (packed217 & 4294967295)
                            .to(tl.int32)
                            .to(tl.float32, bitcast=True)
                        )
                    else:
                        positions = (
                            tl.load(
                                S.to(tl.pointer_type(tl.int16))
                                + row * T * CAP
                                + a,
                                valid,
                                0,
                            ).to(tl.int32)
                            + a // CAP * 2048
                        )
                        candidates = tl.load(
                            P + row * V + positions, valid, 0.0
                        )
                else:
                    packed = tl.load(S + row * T * CAP + a, valid, 0)
                    candidates = (
                        (packed & 4294967295)
                        .to(tl.int32)
                        .to(tl.float32, bitcast=True)
                    )
                    positions = (packed >> 32).to(tl.int32)
                pivot, total = _search(candidates, valid, k, W)
                selected = valid & (a % PARTS == part) & (candidates >= pivot)
                if pivot >= maximum_cutoff:
                    tl.store(
                        D + row * V + positions, candidates / total, selected
                    )
                else:
                    pivot, total = _stream_select(
                        P, K, D, V, KS, KT, SINGLE, KV, 2048
                    )
                    for tile in range(part, triton.cdiv(V, 4096), PARTS):
                        c = tile * 4096 + tl.arange(0, 4096)
                        x = tl.load(P + row * V + c, c < V, 0.0)
                        tl.store(
                            D + row * V + c,
                            tl.where(x >= pivot, x / total, 0.0),
                            c < V,
                        )
        else:
            valid = (a < T * CAP) & (a % CAP < counts)
            if V > 8192:
                if V > 65536:
                    packed217 = tl.load(S + row * T * CAP + a, valid, 0)
                    positions = (packed217 >> 32).to(tl.int32)
                    candidates = (
                        (packed217 & 4294967295)
                        .to(tl.int32)
                        .to(tl.float32, bitcast=True)
                    )
                else:
                    positions = (
                        tl.load(
                            S.to(tl.pointer_type(tl.int16))
                            + row * T * CAP
                            + a,
                            valid,
                            0,
                        ).to(tl.int32)
                        + a // CAP * 2048
                    )
                    candidates = tl.load(P + row * V + positions, valid, 0.0)
            else:
                packed = tl.load(S + row * T * CAP + a, valid, 0)
                candidates = (
                    (packed & 4294967295)
                    .to(tl.int32)
                    .to(tl.float32, bitcast=True)
                )
                positions = (packed >> 32).to(tl.int32)
            pivot, total = _search(candidates, valid, k, W)
            selected = valid & (a % PARTS == part) & (candidates >= pivot)
            if pivot >= maximum_cutoff:
                tl.store(D + row * V + positions, candidates / total, selected)
            else:
                pivot, total = _stream_select(
                    P, K, D, V, KS, KT, SINGLE, KV, 2048
                )
                for tile in range(part, triton.cdiv(V, 4096), PARTS):
                    c = tile * 4096 + tl.arange(0, 4096)
                    x = tl.load(P + row * V + c, c < V, 0.0)
                    tl.store(
                        D + row * V + c,
                        tl.where(x >= pivot, x / total, 0.0),
                        c < V,
                    )
    else:
        pivot, total = _stream_select(P, K, D, V, KS, KT, SINGLE, KV, 2048)
        for tile in range(part, triton.cdiv(V, 4096), PARTS):
            c = tile * 4096 + tl.arange(0, 4096)
            x = tl.load(P + row * V + c, c < V, 0.0)
            tl.store(
                D + row * V + c, tl.where(x >= pivot, x / total, 0.0), c < V
            )


@triton.jit
def _write(P, D, Out, V: tl.constexpr, W: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    tile = tl.program_id(1).to(tl.int64)
    c = tile * W + tl.arange(0, W)
    pivot = tl.load(D + row * 2)
    total = tl.load(D + row * 2 + 1)
    x = tl.load(P + row * V + c, c < V, 0.0)
    tl.store(Out + row * V + c, tl.where(x >= pivot, x / total, 0.0), c < V)


@triton.jit
def _huge(P, K, Out, V: tl.constexpr, KS: tl.constexpr, KV: tl.constexpr):
    KT: tl.constexpr = KV == 0
    SINGLE: tl.constexpr = KS == 0
    row = tl.program_id(0).to(tl.int64)
    pivot, total = _stream_select_wide(P, K, Out, V, KS, KT, SINGLE, KV, 2048)
    for start in range(triton.cdiv(V, 4096)):
        c = start.to(tl.int64) * 4096 + tl.arange(0, 4096)
        x = tl.load(P + row * V + c, c < V, 0.0)
        tl.store(
            Out + row * V + c, tl.where(x >= pivot, x / total, 0.0), c < V
        )


@triton.jit
def _copy_complete(P, Out, N: tl.constexpr, W: tl.constexpr):
    c = tl.program_id(0).to(tl.int64) * W + tl.arange(0, W)
    x = tl.load(P + tl.minimum(c, N - 1))
    tl.store(Out + c, x, c < N)


MAX_BATCH = 65535


def top_k_renorm(probs, top_k):
    n, v = probs.shape
    out = torch.empty_like(probs)
    if n > MAX_BATCH:
        for start in range(0, n, MAX_BATCH):
            end = min(start + MAX_BATCH, n)
            current_k = (
                top_k[start:end]
                if isinstance(top_k, torch.Tensor) and top_k.numel() > 1
                else top_k
            )
            current = top_k_renorm(probs[start:end], current_k)
            _copy_complete[triton.cdiv(current.numel(), 4096),](
                current,
                out[start:end],
                current.numel(),
                4096,
                num_warps=4,
                num_stages=1,
            )
        return out
    kt = isinstance(top_k, torch.Tensor)
    keys = top_k if kt else probs
    single = keys.numel() == 1 if kt else True
    ks = keys.stride(0) if kt and (not single) else 0
    kv = 0 if kt else int(top_k)
    if v > 1048576:
        _huge[n,](
            probs,
            keys,
            out,
            v,
            ks,
            kv,
            num_warps=4,
            num_stages=1,
            enable_fp_fusion=False,
        )
        return out
    if v <= 8192:
        _small[n,](
            probs,
            keys,
            out,
            v,
            ks,
            kv,
            num_warps=1 if v <= 1024 else 4,
            num_stages=1,
            enable_fp_fusion=False,
        )
    else:
        parts = 1
        tile = 2048
        cap = 32
        tiles = triton.cdiv(v, tile)
        scratch = torch.empty(
            (n, tiles * (cap + 1 if v > 65536 else cap // 4 + 1)),
            dtype=torch.int64,
            device=probs.device,
        )
        _filter[tiles, n](
            probs,
            keys,
            scratch,
            out,
            v,
            ks,
            kv,
            num_warps=1 if v > 65536 else 4,
            num_stages=1,
        )
        _selectscatter[n, parts](
            probs,
            keys,
            scratch,
            out,
            v,
            ks,
            kv,
            num_warps=8 if v > 65536 else 4,
            num_stages=1,
            enable_fp_fusion=False,
        )
    return out


@triton.jit
def _popcount188(value):
    value = value.to(tl.uint32)
    value = value - (value >> 1 & 1431655765)
    value = (value & 858993459) + (value >> 2 & 858993459)
    value = value + (value >> 4) & 252645135
    return (value * 16843009 >> 24).to(tl.int32)


__all__ = ["top_k_renorm"]
