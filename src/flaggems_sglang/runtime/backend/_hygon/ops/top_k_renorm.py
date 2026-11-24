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
def _search_countlarge(x, valid, k, W: tl.constexpr):
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
        for trial in tl.static_range(2):
            lf = lo.to(tl.float32, bitcast=True)
            hf = hi.to(tl.float32, bitcast=True)
            fraction = (nlo - k).to(tl.float32) / tl.maximum(nlo - nhi, 1)
            guess = (lf + (hf - lf) * fraction).to(tl.int32, bitcast=True)
            mid = tl.minimum(hi, tl.maximum(lo + 1, guess))
            count = tl.sum((valid & (bits >= mid)).to(tl.int32), 0)
            lo = tl.where(count >= k, mid, lo)
            hi = tl.where(count <= k, tl.where(count == k, mid, mid - 1), hi)
            nlo = tl.where(count >= k, count, nlo)
            nhi = tl.where(count < k, count, nhi)
            step += 1
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
def _small(
    P,
    K,
    O,
    V: tl.constexpr,
    KS: tl.constexpr,
    KT: tl.constexpr,
    SINGLE: tl.constexpr,
    KV: tl.constexpr,
    W: tl.constexpr,
):
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
        pivot, total = _search_smallpure(x, c < V, k, W)
    tl.store(O + row * V + c, tl.where(x >= pivot, x / total, 0.0), c < V)


@triton.jit
def _filter(
    P,
    K,
    S,
    C,
    O,
    V: tl.constexpr,
    T: tl.constexpr,
    CAP: tl.constexpr,
    KS: tl.constexpr,
    KT: tl.constexpr,
    SINGLE: tl.constexpr,
    KV: tl.constexpr,
    W: tl.constexpr,
):
    S = S + tl.num_programs(0).to(tl.int64) * T
    C = C.to(tl.pointer_type(tl.int32))
    row = tl.program_id(0).to(tl.int64)
    tile = tl.program_id(1).to(tl.int64)
    a = tl.arange(0, W)
    if KT:
        k = tl.load(K + (0 if SINGLE else row * KS)).to(tl.int32)
    else:
        k = tl.full((), KV, tl.int32)
    sample = tl.load(P + row * V + a, a < tl.minimum(V, 1024), 0.0)
    maximum = tl.max(sample, 0)
    cutoff = (
        maximum * tl.maximum(0.0, 1.0 - 1.125 * k / V)
        if V > 65536
        else maximum * tl.maximum(0.0, 1.0 - 1.5 * k / V)
    )
    c = tile * W + a
    if V > 65536:
        lanes239 = tl.arange(0, 32)
        groups239 = tl.arange(0, W // 32)
        positions259 = (
            tile * W + lanes239[:, None] * (W // 32) + groups239[None, :]
        )
        values259 = tl.load(
            P + row * V + tl.max_contiguous(positions259, (1, 4)),
            positions259 < V,
            0.0,
        )
        matrix239 = ((positions259 < V) & (values259 >= cutoff)).to(tl.uint32)
        masks239 = tl.sum(matrix239 << lanes239[:, None], 0).to(tl.uint32)
        totals239 = _popcount188(masks239)
        count239 = tl.sum(totals239, 0)
        if count239 <= CAP:
            maximum240 = tl.max(totals239, 0)
            ends239 = tl.cumsum(totals239, 0)
            tl.store(C + row * T + tile, count239)
            starts239 = ends239 - totals239
            remaining239 = masks239
            for slot239 in tl.static_range(1):
                low239 = (
                    remaining239 & tl.full((), 0, tl.uint32) - remaining239
                )
                lane239 = (
                    low239.to(tl.float32).to(tl.uint32, bitcast=True) >> 23
                ).to(tl.int32) - 127
                lane239 = tl.maximum(0, tl.minimum(31, lane239))
                positions239 = tile * W + lane239 * (W // 32) + groups239
                valid239 = slot239 < totals239
                values239 = tl.load(P + row * V + positions239, valid239, 0.0)
                tl.store(
                    S + (row * T + tile) * CAP + starts239 + slot239,
                    positions239.to(tl.int64) << 32
                    | values239.to(tl.int32, bitcast=True)
                    .to(tl.uint32)
                    .to(tl.int64),
                    valid239,
                )
                remaining239 = remaining239 & remaining239 - 1
            for slot239 in range(1, maximum240):
                low239 = (
                    remaining239 & tl.full((), 0, tl.uint32) - remaining239
                )
                lane239 = (
                    low239.to(tl.float32).to(tl.uint32, bitcast=True) >> 23
                ).to(tl.int32) - 127
                lane239 = tl.maximum(0, tl.minimum(31, lane239))
                positions239 = tile * W + lane239 * (W // 32) + groups239
                valid239 = slot239 < totals239
                values239 = tl.load(P + row * V + positions239, valid239, 0.0)
                tl.store(
                    S + (row * T + tile) * CAP + starts239 + slot239,
                    positions239.to(tl.int64) << 32
                    | values239.to(tl.int32, bitcast=True)
                    .to(tl.uint32)
                    .to(tl.int64),
                    valid239,
                )
                remaining239 = remaining239 & remaining239 - 1
        else:
            tl.store(C + row * T + tile, count239)
    else:
        x = tl.load(P + row * V + c, c < V, 0.0)
        active = (c < V) & (x >= cutoff)
        matrix = active.to(tl.int32).reshape((W // 64, 64))
        within = tl.cumsum(matrix, 1)
        totals = tl.sum(matrix, 1)
        offsets = tl.cumsum(totals, 0) - totals
        ranks = (within + offsets[:, None]).reshape((W,))
        count = tl.sum(totals, 0)
        tl.store(C + row * T + tile, count)
        packed = c.to(tl.int64) << 32 | x.to(tl.int32, bitcast=True).to(
            tl.uint32
        ).to(tl.int64)
        tl.store(
            S + (row * T + tile) * CAP + ranks - 1,
            packed,
            active & (ranks <= CAP),
        )
    if V <= 65536:
        tl.store(O + row * V + c, tl.full((W,), 0.0, tl.float32), c < V)
    if V > 65536:
        if tile == 0:
            tl.store(
                C.to(tl.pointer_type(tl.float32))
                + tl.num_programs(0).to(tl.int64) * T
                + row,
                cutoff,
            )


@triton.jit
def _selectscatter(
    P,
    K,
    S,
    C,
    D,
    V: tl.constexpr,
    T: tl.constexpr,
    CAP: tl.constexpr,
    KS: tl.constexpr,
    KT: tl.constexpr,
    SINGLE: tl.constexpr,
    KV: tl.constexpr,
    W: tl.constexpr,
    FULL: tl.constexpr,
    PARTS: tl.constexpr,
):
    S = S + tl.num_programs(0).to(tl.int64) * T
    C = C.to(tl.pointer_type(tl.int32))
    row = tl.program_id(0).to(tl.int64)
    part = tl.program_id(1)
    if KT:
        k = tl.load(K + (0 if SINGLE else row * KS)).to(tl.int32)
    else:
        k = tl.full((), KV, tl.int32)
    if V > 65536:
        pair280 = tl.arange(0, W // (CAP * 2))
        counta280 = tl.load(C + row * T + pair280 * 2, pair280 * 2 < T, 0)
        countb280 = tl.load(
            C + row * T + pair280 * 2 + 1, pair280 * 2 + 1 < T, 0
        )
        pairtotal280 = counta280 + countb280
        total_count = tl.sum(pairtotal280, 0).to(tl.uint32)
        narrow280 = (
            (tl.max(pairtotal280, 0) <= CAP)
            & (total_count >= k)
            & (total_count <= 1023)
        )
    else:
        narrow280 = tl.full((), False, tl.int1)
    if narrow280:
        a280_narrow280 = tl.arange(0, W // 2)
        first280_narrow280 = tl.load(
            C + row * T + a280_narrow280 // CAP * 2,
            a280_narrow280 // CAP * 2 < T,
            0,
        )
        second280_narrow280 = tl.load(
            C + row * T + a280_narrow280 // CAP * 2 + 1,
            a280_narrow280 // CAP * 2 + 1 < T,
            0,
        )
        offset280_narrow280 = a280_narrow280 % CAP
        a_narrow280 = a280_narrow280 // CAP * (2 * CAP) + tl.where(
            offset280_narrow280 < first280_narrow280,
            offset280_narrow280,
            CAP + offset280_narrow280 - first280_narrow280,
        )
        valid_narrow280 = (
            offset280_narrow280 < first280_narrow280 + second280_narrow280
        )
        packed_narrow280 = tl.load(
            S + row * T * CAP + a_narrow280, valid_narrow280, 0
        )
        candidates_narrow280 = (
            (packed_narrow280 & 4294967295)
            .to(tl.int32)
            .to(tl.float32, bitcast=True)
        )
        positions_narrow280 = (packed_narrow280 >> 32).to(tl.int32)
        if V > 65536:
            pivot_narrow280, total_narrow280 = _search_seeded(
                candidates_narrow280,
                valid_narrow280,
                k,
                W // 2,
                total_count,
                tl.load(
                    C.to(tl.pointer_type(tl.float32))
                    + tl.num_programs(0).to(tl.int64) * T
                    + row
                ).to(tl.int32, bitcast=True)
                & 2147483647,
            )
        else:
            pivot_narrow280, total_narrow280 = _search(
                candidates_narrow280, valid_narrow280, k, W // 2
            )
        selected_narrow280 = (
            valid_narrow280
            & (positions_narrow280 // 4096 % PARTS == part)
            & (candidates_narrow280 >= pivot_narrow280)
            if V > 65536
            else valid_narrow280
            & (a_narrow280 % PARTS == part)
            & (candidates_narrow280 >= pivot_narrow280)
        )
        if V > 65536:
            for tile275_narrow280 in range(part, triton.cdiv(V, 4096), PARTS):
                c275_narrow280 = tile275_narrow280 * 4096 + tl.arange(0, 4096)
                tl.store(
                    D + row * V + c275_narrow280,
                    tl.full((4096,), 0.0, tl.float32),
                    c275_narrow280 < V,
                )
            tl.debug_barrier()
        tl.store(
            D + row * V + positions_narrow280,
            candidates_narrow280 / total_narrow280,
            selected_narrow280,
        )
    else:
        a = tl.arange(0, W)
        meta349 = tl.arange(0, triton.next_power_of_2(T))
        counts349 = tl.load(C + row * T + meta349, meta349 < T, 0)
        counts = tl.gather(counts349, a // CAP, 0)
        encoded349 = tl.sum(
            counts349.to(tl.uint32) + ((counts349 > CAP).to(tl.uint32) << 21),
            0,
        )
        total_count = encoded349 & 2097151
        overflow = encoded349 >> 21 != 0
        if (total_count >= k) & ~overflow:
            valid = (a < T * CAP) & (a % CAP < counts)
            packed = tl.load(S + row * T * CAP + a, valid, 0)
            candidates = (
                (packed & 4294967295).to(tl.int32).to(tl.float32, bitcast=True)
            )
            positions = (packed >> 32).to(tl.int32)
            if V > 65536:
                pivot, total = _search_seeded(
                    candidates,
                    valid,
                    k,
                    W,
                    total_count,
                    tl.load(
                        C.to(tl.pointer_type(tl.float32))
                        + tl.num_programs(0).to(tl.int64) * T
                        + row
                    ).to(tl.int32, bitcast=True)
                    & 2147483647,
                )
            else:
                pivot, total = _search(candidates, valid, k, W)
            selected = (
                valid
                & (positions // 4096 % PARTS == part)
                & (candidates >= pivot)
                if V > 65536
                else valid & (a % PARTS == part) & (candidates >= pivot)
            )
            if V > 65536:
                for tile275 in range(part, triton.cdiv(V, 4096), PARTS):
                    c275 = tile275 * 4096 + tl.arange(0, 4096)
                    tl.store(
                        D + row * V + c275,
                        tl.full((4096,), 0.0, tl.float32),
                        c275 < V,
                    )
                tl.debug_barrier()
            tl.store(D + row * V + positions, candidates / total, selected)
        else:
            pivot, total = _stream_select(P, K, D, V, KS, KT, SINGLE, KV, 2048)
            for tile in range(part, triton.cdiv(V, 4096), PARTS):
                c = tile * 4096 + tl.arange(0, 4096)
                x = tl.load(P + row * V + c, c < V, 0.0)
                tl.store(
                    D + row * V + c,
                    tl.where(x >= pivot, x / total, 0.0),
                    c < V,
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
def _huge(
    P,
    K,
    O,
    V: tl.constexpr,
    KS: tl.constexpr,
    KT: tl.constexpr,
    SINGLE: tl.constexpr,
    KV: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    pivot, total = _stream_select_wide(P, K, O, V, KS, KT, SINGLE, KV, 2048)
    for start in range(triton.cdiv(V, 4096)):
        c = start.to(tl.int64) * 4096 + tl.arange(0, 4096)
        x = tl.load(P + row * V + c, c < V, 0.0)
        tl.store(O + row * V + c, tl.where(x >= pivot, x / total, 0.0), c < V)


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
    keys = top_k if kt else None
    single = keys.numel() == 1 if kt else True
    ks = keys.stride(0) if kt else 0
    kv = 0 if kt else int(top_k)
    if v > 1048576:
        _huge[n,](
            probs,
            keys,
            out,
            v,
            ks,
            kt,
            single,
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
            kt,
            single,
            kv,
            triton.next_power_of_2(v),
            num_warps=1 if v <= 1024 else 4,
            num_stages=1,
            enable_fp_fusion=False,
        )
    else:
        parts = 4 if v > 65536 else 8
        tile = 2048
        cap = 32
        tiles = triton.cdiv(v, tile)
        scratch = torch.empty(
            (n, tiles * (cap + 1)), dtype=torch.int64, device=probs.device
        )
        counts = scratch
        _filter[n, tiles](
            probs,
            keys,
            scratch,
            counts,
            out,
            v,
            tiles,
            cap,
            ks,
            kt,
            single,
            kv,
            tile,
            num_warps=1 if v > 65536 else 2,
            num_stages=1,
        )
        _selectscatter[n, parts](
            probs,
            keys,
            scratch,
            counts,
            out,
            v,
            tiles,
            cap,
            ks,
            kt,
            single,
            kv,
            triton.next_power_of_2(tiles * cap),
            triton.next_power_of_2(v),
            parts,
            num_warps=4,
            num_stages=1,
            enable_fp_fusion=False,
        )
    return out


@triton.jit
def _search_smallpure(x, valid, k, W: tl.constexpr):
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
            count = tl.sum((valid & (bits >= mid)).to(tl.int32), 0)
            lo = tl.where(count >= k, mid, lo)
            hi = tl.where(count <= k, tl.where(count == k, mid, mid - 1), hi)
            nlo = tl.where(count >= k, count, nlo)
            nhi = tl.where(count < k, count, nhi)
            step += 1
        kept = tl.where(valid & (bits >= lo), x, 0.0)
        return (lo.to(tl.float32, bitcast=True), tl.sum(kept, 0))


@triton.jit
def _search_seeded(x, valid, k, W: tl.constexpr, nlo, lo):
    if W <= 16384:
        bits312 = tl.where(
            valid, x.to(tl.int32, bitcast=True) & 2147483647, -1
        )
        hi312 = tl.max(bits312, 0)
        lo312 = lo
        nlo312 = nlo
        nhi312 = tl.full((), 0, tl.int32)
        step312 = tl.full((), 0, tl.int32)
        if nlo <= 1023:
            while lo312 < hi312:
                lf312 = lo312.to(tl.float32, bitcast=True)
                hf312 = hi312.to(tl.float32, bitcast=True)
                fraction312 = (nlo312 - k).to(tl.float32) / tl.maximum(
                    nlo312 - nhi312, 1
                )
                guess312 = (lf312 + (hf312 - lf312) * fraction312).to(
                    tl.int32, bitcast=True
                )
                guess312 = tl.where(
                    step312 % 4 == 3,
                    lo312 + (hi312 - lo312 + 1) // 2,
                    guess312,
                )
                gap391 = tl.maximum(1, (hi312 - lo312) // 32)
                a391 = tl.minimum(
                    hi312, tl.maximum(lo312 + 1, guess312 - gap391)
                )
                b391 = tl.minimum(hi312, tl.maximum(a391, guess312))
                c391 = tl.minimum(hi312, tl.maximum(b391, guess312 + gap391))
                encoded391 = (
                    (bits312 >= a391).to(tl.uint32)
                    | (bits312 >= b391).to(tl.uint32) << 10
                    | (bits312 >= c391).to(tl.uint32) << 20
                )
                sum391 = tl.sum(encoded391, 0)
                na391 = (sum391 & 1023).to(tl.int32)
                nb391 = (sum391 >> 10 & 1023).to(tl.int32)
                nc391 = (sum391 >> 20).to(tl.int32)
                lo312 = tl.where(
                    nc391 >= k,
                    c391,
                    tl.where(
                        nb391 >= k, b391, tl.where(na391 >= k, a391, lo312)
                    ),
                )
                nlo312 = tl.where(
                    nc391 >= k,
                    nc391,
                    tl.where(
                        nb391 >= k, nb391, tl.where(na391 >= k, na391, nlo312)
                    ),
                )
                hi312 = tl.where(
                    na391 <= k,
                    tl.where(na391 == k, a391, a391 - 1),
                    tl.where(
                        nb391 <= k,
                        tl.where(nb391 == k, b391, b391 - 1),
                        tl.where(
                            nc391 <= k,
                            tl.where(nc391 == k, c391, c391 - 1),
                            hi312,
                        ),
                    ),
                )
                nhi312 = tl.where(
                    na391 < k,
                    na391,
                    tl.where(
                        nb391 < k, nb391, tl.where(nc391 < k, nc391, nhi312)
                    ),
                )
                step312 += 1
        else:
            while lo312 < hi312:
                lf312 = lo312.to(tl.float32, bitcast=True)
                hf312 = hi312.to(tl.float32, bitcast=True)
                fraction312 = (nlo312 - k).to(tl.float32) / tl.maximum(
                    nlo312 - nhi312, 1
                )
                guess312 = (lf312 + (hf312 - lf312) * fraction312).to(
                    tl.int32, bitcast=True
                )
                guess312 = tl.where(
                    step312 % 4 == 3,
                    lo312 + (hi312 - lo312 + 1) // 2,
                    guess312,
                )
                gap312 = tl.maximum(1, (hi312 - lo312) // 16)
                a312 = tl.minimum(
                    hi312, tl.maximum(lo312 + 1, guess312 - gap312)
                )
                b312 = tl.minimum(hi312, tl.maximum(a312, guess312 + gap312))
                pair312 = (bits312 >= a312).to(tl.uint32) + (
                    (bits312 >= b312).to(tl.uint32) << 16
                )
                packed312 = tl.sum(pair312, 0)
                na312 = (packed312 & 65535).to(tl.int32)
                nb312 = (packed312 >> 16).to(tl.int32)
                lo312 = tl.where(
                    nb312 >= k, b312, tl.where(na312 >= k, a312, lo312)
                )
                nlo312 = tl.where(
                    nb312 >= k, nb312, tl.where(na312 >= k, na312, nlo312)
                )
                hi312 = tl.where(
                    na312 <= k,
                    tl.where(na312 == k, a312, a312 - 1),
                    tl.where(
                        nb312 <= k, tl.where(nb312 == k, b312, b312 - 1), hi312
                    ),
                )
                nhi312 = tl.where(
                    na312 < k, na312, tl.where(nb312 < k, nb312, nhi312)
                )
                step312 += 1
        kept312 = tl.where(bits312 >= lo312, x, 0.0)
        return (lo312.to(tl.float32, bitcast=True), tl.sum(kept312, 0))
    else:
        return _search_seeded312_old(x, valid, k, W, nlo, lo)


@triton.jit
def _popcount188(value):
    value = value.to(tl.uint32)
    value = value - (value >> 1 & 1431655765)
    value = (value & 858993459) + (value >> 2 & 858993459)
    value = value + (value >> 4) & 252645135
    return (value * 16843009 >> 24).to(tl.int32)


@triton.jit
def _search_seeded312_old(x, valid, k, W: tl.constexpr, nlo, lo):
    bits = x.to(tl.int32, bitcast=True) & 2147483647
    hi = tl.max(tl.where(valid, bits, 0), 0)
    nhi = tl.full((), 0, tl.int32)
    step = tl.full((), 0, tl.int32)
    for trial in tl.static_range(2):
        lf = lo.to(tl.float32, bitcast=True)
        hf = hi.to(tl.float32, bitcast=True)
        fraction = (nlo - k).to(tl.float32) / tl.maximum(nlo - nhi, 1)
        guess = (lf + (hf - lf) * fraction).to(tl.int32, bitcast=True)
        mid = tl.minimum(hi, tl.maximum(lo + 1, guess))
        count = tl.sum((valid & (bits >= mid)).to(tl.int32), 0)
        lo = tl.where(count >= k, mid, lo)
        hi = tl.where(count <= k, tl.where(count == k, mid, mid - 1), hi)
        nlo = tl.where(count >= k, count, nlo)
        nhi = tl.where(count < k, count, nhi)
        step += 1
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


__all__ = ["top_k_renorm"]
