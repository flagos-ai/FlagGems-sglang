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
def _add_pairs(ah, al, bh, bl):
    s = ah + bh
    v = s - ah
    e = (ah - (s - v)) + (bh - v)
    t = (al + bl) + e
    h = s + t
    low_part = t - (h - s)
    return h, low_part


@triton.jit
def _top_p_select(
    P,
    T,
    O,
    V: tl.constexpr,
    S0: tl.constexpr,
    S1: tl.constexpr,
    TS: tl.constexpr,
    IS_TENSOR: tl.constexpr,
    SINGLE: tl.constexpr,
    PV: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK)
    if IS_TENSOR:
        ti = 0 if SINGLE else row * TS
        p = tl.load(T + ti).to(tl.float32)
    else:
        p = tl.full((), PV, tl.float32)
    target = 1.0 - p
    low = tl.full((), float("inf"), tl.float32)
    high = tl.full((), 0.0, tl.float32)
    for start in range(triton.cdiv(V, BLOCK)):
        cols = start * BLOCK + offsets
        valid = cols < V
        x = tl.load(P + row * S0 + cols * S1, valid, other=0).to(tl.float32)
        low = tl.minimum(low, tl.min(tl.where(valid, x, float("inf")), 0))
        high = tl.maximum(high, tl.max(tl.where(valid, x, 0.0), 0))
    lo = low.to(tl.int32, bitcast=True)
    hi = high.to(tl.int32, bitcast=True)
    if target > 0:
        while lo < hi:
            mid = lo + (hi - lo) // 2
            pivot = mid.to(tl.float32, bitcast=True)
            mass_hi = tl.full((BLOCK,), 0.0, tl.float32)
            mass_lo = tl.full((BLOCK,), 0.0, tl.float32)
            lower = tl.full((), 0.0, tl.float32)
            upper = tl.full((), float("inf"), tl.float32)
            for start in range(triton.cdiv(V, BLOCK)):
                cols = start * BLOCK + offsets
                valid = cols < V
                x = tl.load(P + row * S0 + cols * S1, valid, other=0).to(
                    tl.float32
                )
                below = valid & (x <= pivot)
                selected = tl.where(below, x, 0.0)
                mass_hi, mass_lo = _add_pairs(
                    mass_hi,
                    mass_lo,
                    selected,
                    tl.full((BLOCK,), 0.0, tl.float32),
                )
                lower = tl.maximum(lower, tl.max(tl.where(below, x, 0.0), 0))
                upper = tl.minimum(
                    upper,
                    tl.min(tl.where(valid & (x > pivot), x, float("inf")), 0),
                )
            sum_hi, sum_lo = tl.reduce((mass_hi, mass_lo), 0, _add_pairs)
            mass = sum_hi + sum_lo
            enough = mass >= target
            hi = tl.where(enough, lower.to(tl.int32, bitcast=True), hi)
            lo = tl.where(enough, lo, upper.to(tl.int32, bitcast=True))
    pivot = lo.to(tl.float32, bitcast=True)
    total = tl.full((BLOCK,), 0.0, tl.float32)
    for start in range(triton.cdiv(V, BLOCK)):
        cols = start * BLOCK + offsets
        valid = cols < V
        x = tl.load(P + row * S0 + cols * S1, valid, other=0).to(tl.float32)
        total += tl.where(valid & (x >= pivot), x, 0.0)
    denom = tl.sum(total, 0)
    for start in range(triton.cdiv(V, BLOCK)):
        cols = start * BLOCK + offsets
        valid = cols < V
        x = tl.load(P + row * S0 + cols * S1, valid, other=0).to(tl.float32)
        y = tl.where(x >= pivot, x / denom, 0.0)
        tl.store(O + row * V + cols, y, valid)


def _single_row(probs, top_p):
    batch, vocab = probs.shape
    out = torch.empty((batch, vocab), dtype=torch.float32, device=probs.device)
    is_tensor = isinstance(top_p, torch.Tensor)
    if is_tensor:
        values = top_p.to(device=probs.device).reshape(-1)
        single = values.numel() == 1
        stride = values.stride(0)
        scalar = 0.0
    else:
        values = probs
        single = True
        stride = 0
        scalar = float(top_p)
    if batch > 0 and vocab > 0:
        block = min(4096, triton.next_power_of_2(vocab))
        _top_p_select[(batch,)](
            probs,
            values,
            out,
            vocab,
            probs.stride(0),
            probs.stride(1),
            stride,
            is_tensor,
            single,
            scalar,
            block,
            num_warps=4,
            num_stages=1,
            enable_fp_fusion=False,
        )
    return out


@triton.jit
def _radix_pack(
    P,
    A,
    V: tl.constexpr,
    PAD: tl.constexpr,
    S0: tl.constexpr,
    S1: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    source = tl.minimum(cols, V - 1)
    x = tl.load(P + row * S0 + source.to(tl.int64) * S1).to(tl.float32)
    x = tl.where(cols < V, x, float("inf"))
    tl.store(A + row * PAD + cols, x)


@triton.jit
def _radix_hist(
    A,
    H,
    SHIFT,
    PAD: tl.constexpr,
    TILES: tl.constexpr,
    BINS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    tile = tl.program_id(1)
    digit = tl.program_id(2)
    cols = tile * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(A + row * PAD + cols)
    bits = x.to(tl.uint32, bitcast=True) & 2147483647
    key = (bits >> SHIFT) & (BINS - 1)
    count = tl.sum((key == digit).to(tl.int32), 0)
    tl.store(H + (row * BINS + digit) * TILES + tile, count)


@triton.jit
def _radix_prefix(
    H, E, TILES: tl.constexpr, BINS: tl.constexpr, WIDTH: tl.constexpr
):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, WIDTH)
    source = tl.minimum(cols, BINS * TILES - 1)
    value = tl.load(H + row * BINS * TILES + source)
    value = tl.where(cols < BINS * TILES, value, 0)
    exclusive = tl.cumsum(value, 0) - value
    tl.store(E + row * WIDTH + cols, exclusive)


@triton.jit
def _radix_scatter(
    A,
    E,
    O,
    SHIFT,
    PAD: tl.constexpr,
    TILES: tl.constexpr,
    BINS: tl.constexpr,
    WIDTH: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    tile = tl.program_id(1)
    lane = tl.arange(0, BLOCK)
    cols = tile * BLOCK + lane
    x = tl.load(A + row * PAD + cols)
    bits = x.to(tl.uint32, bitcast=True) & 2147483647
    key = (bits >> SHIFT) & (BINS - 1)
    if BINS == 2:
        zero = (key == 0).to(tl.int32)
        zeros = tl.cumsum(zero, 0)
        base_zero = tl.load(E + row * WIDTH + tile)
        base_one = tl.load(E + row * WIDTH + TILES + tile)
        position = tl.where(
            key == 0, base_zero + zeros - 1, base_one + lane - zeros
        )
    else:
        position = tl.full((BLOCK,), 0, tl.int32)
        for digit in range(BINS):
            matches = (key == digit).to(tl.int32)
            local = tl.cumsum(matches, 0) - 1
            base = tl.load(E + row * WIDTH + digit * TILES + tile)
            position = tl.where(key == digit, base + local, position)
    tl.store(O + row * PAD + position.to(tl.int64), x)


def _radix_sort_nonnegative(probs, digit_bits=4):
    batch, vocab = probs.shape
    block = 1024
    padded = triton.cdiv(vocab, block) * block
    first = torch.empty(
        (batch, padded), dtype=torch.float32, device=probs.device
    )
    if batch == 0 or vocab == 0:
        return first
    second = torch.empty_like(first)
    tiles = padded // block
    bins = 1 << digit_bits
    width = triton.next_power_of_2(bins * tiles)
    histogram = torch.empty(
        (batch, bins * tiles), dtype=torch.int32, device=probs.device
    )
    prefix = torch.empty(
        (batch, width), dtype=torch.int32, device=probs.device
    )
    _radix_pack[(batch, tiles)](
        probs,
        first,
        vocab,
        padded,
        *probs.stride(),
        block,
        num_warps=4,
        num_stages=1,
    )
    for shift in range(0, 31, digit_bits):
        _radix_hist[(batch, tiles, bins)](
            first,
            histogram,
            shift,
            padded,
            tiles,
            bins,
            block,
            num_warps=4,
            num_stages=1,
        )
        _radix_prefix[(batch,)](
            histogram, prefix, tiles, bins, width, num_warps=4, num_stages=1
        )
        _radix_scatter[(batch, tiles)](
            first,
            prefix,
            second,
            shift,
            padded,
            tiles,
            bins,
            width,
            block,
            num_warps=4,
            num_stages=1,
        )
        first, second = second, first
    return first


def radix_sort_nonnegative(probs):
    return _radix_sort_nonnegative(probs)[:, : probs.shape[1]]


@triton.jit
def _sklansky_pivot(
    SORTED,
    TOP,
    PIVOTS,
    V: tl.constexpr,
    PAD: tl.constexpr,
    TENSOR: tl.constexpr,
    SINGLE: tl.constexpr,
    TS: tl.constexpr,
    PV: tl.constexpr,
    BLOCK: tl.constexpr,
    LEVELS: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    lane = tl.arange(0, BLOCK)
    if TENSOR:
        offset = 0 if SINGLE else row * TS
        p = tl.load(TOP + offset).to(tl.float32)
    else:
        p = tl.full((), PV, tl.float32)
    target = 1.0 - p
    carry = tl.full((), 0.0, tl.float32)
    start = 0
    found = V
    while (start < V) & (found == V):
        column = start + lane
        value = tl.load(SORTED + row * PAD + column, column < V, other=0.0)
        value = tl.where(lane == 0, value + carry, value)
        for level in tl.static_range(0, LEVELS):
            step = 1 << level
            source = (lane // (2 * step)) * (2 * step) + step - 1
            left = tl.gather(value, source, 0)
            value = tl.where((lane & step) != 0, value + left, value)
        found = tl.min(
            tl.where((column < V) & (value >= target), column, V), 0
        )
        carry = tl.sum(tl.where(lane == BLOCK - 1, value, 0.0), 0)
        start += BLOCK
    index = tl.minimum(found, V - 1)
    pivot = tl.load(SORTED + row * PAD + index)
    tl.store(PIVOTS + row, pivot)


@triton.jit
def _normalize_with_pivot(
    P,
    PIVOTS,
    OUT,
    V: tl.constexpr,
    S0: tl.constexpr,
    S1: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    lane = tl.arange(0, BLOCK)
    pivot = tl.load(PIVOTS + row)
    total = tl.full((BLOCK,), 0.0, tl.float32)
    for start in range(triton.cdiv(V, BLOCK)):
        column = start * BLOCK + lane
        x = tl.load(
            P + row * S0 + column.to(tl.int64) * S1, column < V, other=0.0
        ).to(tl.float32)
        total += tl.where(x >= pivot, x, 0.0)
    denom = tl.sum(total, 0)
    for start in range(triton.cdiv(V, BLOCK)):
        column = start * BLOCK + lane
        x = tl.load(
            P + row * S0 + column.to(tl.int64) * S1, column < V, other=0.0
        ).to(tl.float32)
        y = tl.where(x >= pivot, x / denom, 0.0)
        tl.store(OUT + row * V + column, y, column < V)


@triton.jit
def _radix_hist_fused(
    A,
    H,
    SHIFT,
    PAD: tl.constexpr,
    TILES: tl.constexpr,
    BINS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    tile = tl.program_id(1)
    cols = tile * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(A + row * PAD + cols)
    bits = x.to(tl.uint32, bitcast=True) & 2147483647
    key = (bits >> SHIFT) & (BINS - 1)
    for pair in range(BINS // 2):
        digit = pair * 2
        encoded = tl.where(
            key == digit, 1, tl.where(key == digit + 1, 65536, 0)
        ).to(tl.int32)
        count = tl.sum(encoded, 0)
        tl.store(H + (row * BINS + digit) * TILES + tile, count & 65535)
        tl.store(H + (row * BINS + digit + 1) * TILES + tile, count >> 16)


@triton.jit
def _radix_scatter_pairs(
    A,
    E,
    O,
    SHIFT,
    PAD: tl.constexpr,
    TILES: tl.constexpr,
    BINS: tl.constexpr,
    WIDTH: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    tile = tl.program_id(1)
    lane = tl.arange(0, BLOCK)
    cols = tile * BLOCK + lane
    x = tl.load(A + row * PAD + cols)
    bits = x.to(tl.uint32, bitcast=True) & 2147483647
    key = (bits >> SHIFT) & (BINS - 1)
    if BINS == 2:
        zero = (key == 0).to(tl.int32)
        zeros = tl.cumsum(zero, 0)
        base_zero = tl.load(E + row * WIDTH + tile)
        base_one = tl.load(E + row * WIDTH + TILES + tile)
        position = tl.where(
            key == 0, base_zero + zeros - 1, base_one + lane - zeros
        )
    else:
        position = tl.full((BLOCK,), 0, tl.int32)
        odd = (key & 1) != 0
        for pair in range(BINS // 2):
            digit = pair * 2
            encoded = tl.where(
                key == digit, 1, tl.where(key == digit + 1, 65536, 0)
            ).to(tl.int32)
            counts = tl.cumsum(encoded, 0)
            local = tl.where(odd, counts >> 16, counts & 65535) - 1
            base_even = tl.load(E + row * WIDTH + digit * TILES + tile)
            base_odd = tl.load(E + row * WIDTH + (digit + 1) * TILES + tile)
            base = tl.where(odd, base_odd, base_even)
            position = tl.where((key >> 1) == pair, base + local, position)
    tl.store(O + row * PAD + position.to(tl.int64), x)


def _radix_sort_fused(probs, digit_bits=4):
    batch, vocab = probs.shape
    block = 1024
    padded = triton.cdiv(vocab, block) * block
    first = torch.empty(
        (batch, padded), dtype=torch.float32, device=probs.device
    )
    if batch == 0 or vocab == 0:
        return first
    second = torch.empty_like(first)
    tiles = padded // block
    bins = 1 << digit_bits
    width = triton.next_power_of_2(bins * tiles)
    histogram = torch.empty(
        (batch, bins * tiles), dtype=torch.int32, device=probs.device
    )
    prefix = torch.empty(
        (batch, width), dtype=torch.int32, device=probs.device
    )
    _radix_pack[(batch, tiles)](
        probs,
        first,
        vocab,
        padded,
        *probs.stride(),
        block,
        num_warps=4,
        num_stages=1,
    )
    for shift in range(0, 31, digit_bits):
        _radix_hist_fused[(batch, tiles)](
            first,
            histogram,
            shift,
            padded,
            tiles,
            bins,
            block,
            num_warps=4,
            num_stages=1,
        )
        _radix_prefix[(batch,)](
            histogram, prefix, tiles, bins, width, num_warps=4, num_stages=1
        )
        _radix_scatter_pairs[(batch, tiles)](
            first,
            prefix,
            second,
            shift,
            padded,
            tiles,
            bins,
            width,
            block,
            num_warps=4,
            num_stages=1,
        )
        first, second = second, first
    return first


@triton.jit
def _initial(
    P,
    S,
    V: tl.constexpr,
    NP: tl.constexpr,
    S0: tl.constexpr,
    S1: tl.constexpr,
    B: tl.constexpr,
):
    row = tl.program_id(0)
    r = tl.program_id(1) * B + tl.arange(0, B)
    x = tl.load(P + row * S0 + r * S1, r < V, other=float("inf")).to(
        tl.float32
    )
    x = tl.sort(x, descending=False)
    tl.store(S + row * NP + r, x)


@triton.jit
def _merge(
    S,
    Out,
    NP: tl.constexpr,
    M: tl.constexpr,
    LOG: tl.constexpr,
    B: tl.constexpr,
):
    row = tl.program_id(0)
    r = tl.program_id(1) * B + tl.arange(0, B)
    x = tl.load(S + row * NP + r)
    group = r // (2 * M) * (2 * M)
    own = r % M
    left = r % (2 * M) < M
    other = group + tl.where(left, M, 0)
    low = tl.full((B,), 0, tl.int32)
    high = tl.full((B,), M, tl.int32)
    for step in tl.static_range(LOG):
        mid = (low + high) // 2
        y = tl.load(S + row * NP + other + tl.minimum(mid, M - 1))
        right = (mid < M) & tl.where(left, y < x, y <= x)
        low = tl.where(right, mid + 1, low)
        high = tl.where(right, high, mid)
    tl.store(Out + row * NP + group + own + low, x)


def _merge_sort_values(probs):
    batch, vocab = probs.shape
    padded = max(1024, triton.next_power_of_2(vocab))
    first = torch.empty(
        (batch, padded), dtype=torch.float32, device=probs.device
    )
    if batch == 0 or vocab == 0:
        return first
    second = torch.empty_like(first)
    _initial[(batch, padded // 1024)](
        probs,
        first,
        vocab,
        padded,
        probs.stride(0),
        probs.stride(1),
        1024,
        num_warps=4,
    )
    width = 1024
    while width < padded:
        _merge[(batch, padded // 512)](
            first, second, padded, width, width.bit_length(), 512, num_warps=4
        )
        first, second = second, first
        width *= 2
    return first


def top_p_renorm(probs, top_p):
    batch, vocab = probs.shape
    if batch == 1:
        return _single_row(probs, top_p)
    out = torch.empty((batch, vocab), dtype=torch.float32, device=probs.device)
    if batch == 0 or vocab == 0:
        return out
    ordered = _merge_sort_values(probs)
    pivots = torch.empty((batch,), dtype=torch.float32, device=probs.device)
    is_tensor = isinstance(top_p, torch.Tensor)
    if is_tensor:
        values = top_p.to(device=probs.device).reshape(-1)
        single = values.numel() == 1
        stride = values.stride(0)
        scalar = 0.0
    else:
        values = probs
        single = True
        stride = 0
        scalar = float(top_p)
    log_threads = max(
        4,
        min(9, (9 + (vocab - 1).bit_length() - (batch - 1).bit_length()) // 2),
    )
    levels = log_threads + 1
    block = 1 << levels
    _sklansky_pivot[(batch,)](
        ordered,
        values,
        pivots,
        vocab,
        ordered.shape[1],
        is_tensor,
        single,
        stride,
        scalar,
        block,
        levels,
        num_warps=4,
        num_stages=1,
        enable_fp_fusion=False,
    )
    norm_block = min(4096, triton.next_power_of_2(vocab))
    _normalize_with_pivot[(batch,)](
        probs,
        pivots,
        out,
        vocab,
        probs.stride(0),
        probs.stride(1),
        norm_block,
        num_warps=4,
        num_stages=1,
        enable_fp_fusion=False,
    )
    return out


__all__ = ["top_p_renorm"]
