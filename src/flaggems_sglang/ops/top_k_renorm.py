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
def _search_largecount(x, valid, k, W: tl.constexpr):
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
    cutoff = maximum * tl.maximum(0.0, 1.0 - 1.5 * k / V)
    c = tile * W + a
    x = tl.load(P + row * V + c, c < V, 0.0, cache_modifier=".cg")
    active = (c < V) & (x >= cutoff)
    matrix = active.to(tl.int32).reshape((W // 32, 32))
    within = tl.cumsum(matrix, 1)
    totals = tl.sum(matrix, 1)
    offsets = tl.cumsum(totals, 0) - totals
    ranks = (within + offsets[:, None]).reshape((W,))
    count = tl.sum(totals, 0)
    tl.store(C + row * T + tile, count)
    tl.store(
        S + (row * T + tile) * CAP + ranks - 1,
        c.to(tl.int64) << 32
        | x.to(tl.int32, bitcast=True).to(tl.uint32).to(tl.int64),
        active & (ranks <= CAP),
    )


@triton.jit
def _selectwrite(
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
    OW: tl.constexpr,
):
    S = S + tl.num_programs(0).to(tl.int64) * T
    C = C.to(tl.pointer_type(tl.int32))
    row = tl.program_id(0).to(tl.int64)
    if KT:
        k = tl.load(K + (0 if SINGLE else row * KS)).to(tl.int32)
    else:
        k = tl.full((), KV, tl.int32)
    a = tl.arange(0, W)
    counts = tl.load(C + row * T + a // CAP, a < T * CAP, 0)
    total_count = tl.sum(
        tl.where((a < T * CAP) & (a % CAP == 0), counts, 0), 0
    )
    overflow = tl.max(counts, 0) > CAP
    tile = tl.program_id(1).to(tl.int64)
    if (total_count >= k) & ~overflow:
        valid = (a < T * CAP) & (a % CAP < counts)
        candidates = tl.load(
            S.to(tl.pointer_type(tl.float32)) + (row * T * CAP + a) * 2,
            valid,
            0.0,
        )
        if V > 65536:
            pivot, total = _search_largecount(candidates, valid, k, W)
        elif W <= 512 and KV > 1:
            pivot, total = _search_mediumcount(candidates, valid, k, W)
        else:
            pivot, total = _search(candidates, valid, k, W)
        positions = tl.load(
            S.to(tl.pointer_type(tl.int32)) + (row * T * CAP + a) * 2 + 1,
            valid,
            0,
        )
        c = tile * OW + tl.arange(0, OW)
        tl.store(
            D + row * V + c,
            tl.full((OW,), 0.0, tl.float32),
            c < V,
            cache_modifier=".cs",
        )
        tl.debug_barrier()
        selected = (
            valid
            & (candidates >= pivot)
            & (positions >= tile * OW)
            & (positions < (tile + 1) * OW)
        )
        tl.store(D + row * V + positions, candidates / total, selected)
    else:
        pivot, total = _stream_select(P, K, D, V, KS, KT, SINGLE, KV, 2048)
        for start in range(triton.cdiv(OW, 4096)):
            c = tile * OW + start * 4096 + tl.arange(0, 4096)
            x = tl.load(P + row * V + c, c < V, 0.0, cache_modifier=".cg")
            tl.store(
                D + row * V + c,
                tl.where(x >= pivot, x / total, 0.0),
                c < V,
                cache_modifier=".cs",
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
    keys = top_k if kt else probs
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
            max(2, triton.next_power_of_2(v)),
            num_warps=1 if v <= 1024 else 4,
            num_stages=1,
            enable_fp_fusion=False,
        )
    else:
        output_width = min(16384, max(4096, triton.next_power_of_2(v // 8)))
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
            v,
            tiles,
            cap,
            ks,
            kt,
            single,
            kv,
            tile,
            num_warps=4,
            num_stages=1,
        )
        _selectwrite[n, triton.cdiv(v, output_width)](
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
            output_width,
            num_warps=4,
            num_stages=1,
            enable_fp_fusion=False,
        )
    return out


@triton.jit
def _search_smallpure(x, valid, k, W: tl.constexpr):
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
def _search_mediumcount(x, valid, k, W: tl.constexpr):
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


__all__ = ["top_k_renorm"]
