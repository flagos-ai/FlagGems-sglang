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


def _radix_sort_nonnegative(probs, digit_bits=4):
    batch, vocab = probs.shape
    block = max(1024, min(8192, 4096 * batch, triton.next_power_of_2(vocab)))
    padded = triton.cdiv(vocab, block) * block
    first = torch.empty(
        (batch, padded), dtype=torch.float32, device=probs.device
    )
    if batch == 0 or vocab == 0:
        return first
    second = torch.empty(
        (batch, padded), dtype=torch.float32, device=probs.device
    )
    tiles = padded // block
    bins = 1 << digit_bits
    width = max(128, triton.next_power_of_2(bins * tiles))
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
        isCloseUnrollControl=True,
    )
    for shift in range(0, 31, digit_bits):
        pass_bins = 1 << min(digit_bits, 31 - shift)
        pass_width = max(128, triton.next_power_of_2(pass_bins * tiles))
        _radix_hist[(batch, tiles, pass_bins)](
            first,
            histogram,
            shift,
            padded,
            tiles,
            pass_bins,
            block,
            num_warps=4,
            num_stages=1,
            isCloseUnrollControl=True,
        )
        _radix_prefix[(batch,)](
            histogram,
            prefix,
            tiles,
            pass_bins,
            pass_width,
            num_warps=4,
            num_stages=1,
            isCloseUnrollControl=True,
        )
        _radix_scatter[(batch, tiles)](
            first,
            prefix,
            second,
            shift,
            padded,
            tiles,
            pass_bins,
            pass_width,
            block,
            num_warps=4,
            num_stages=1,
            isCloseUnrollControl=True,
        )
        first, second = second, first
    return first


@triton.jit
def _serial_pivot(
    Sorted,
    T,
    Pivot,
    V: tl.constexpr,
    PAD: tl.constexpr,
    TS: tl.constexpr,
    IS_TENSOR: tl.constexpr,
    SINGLE: tl.constexpr,
    PV: tl.constexpr,
):
    row = tl.program_id(0)
    if IS_TENSOR:
        ti = 0 if SINGLE else row * TS
        p = tl.load(T + ti).to(tl.float32)
    else:
        p = tl.full((), PV, tl.float32)
    target = 1.0 - p
    pivot = tl.load(Sorted + row * PAD).to(tl.float32)
    accumulator = pivot
    index = 1
    while (index < V) & (accumulator < target):
        pivot = tl.load(Sorted + row * PAD + index).to(tl.float32)
        accumulator = accumulator + pivot
        index += 1
    tl.store(Pivot + row, pivot)


def _select_sorted_pivot(sorted_probs, top_p, vocab):
    batch, padded = sorted_probs.shape
    pivots = torch.empty(
        (batch,), dtype=torch.float32, device=sorted_probs.device
    )
    is_tensor = isinstance(top_p, torch.Tensor)
    if is_tensor:
        values = top_p.to(device=sorted_probs.device).reshape(-1)
        single = values.numel() == 1
        stride = values.stride(0)
        scalar = 0.0
    else:
        values = sorted_probs
        single = True
        stride = 0
        scalar = float(top_p)
    if batch > 0 and vocab > 0:
        _serial_pivot[(batch,)](
            sorted_probs,
            values,
            pivots,
            vocab,
            padded,
            stride,
            is_tensor,
            single,
            scalar,
            num_warps=4,
            num_stages=1,
            enable_fp_fusion=False,
            groups_per_cluster=1,
        )
    return pivots


@triton.jit
def _mask_normalize(
    P,
    Pivot,
    O,
    V: tl.constexpr,
    S0: tl.constexpr,
    S1: tl.constexpr,
    BLOCK: tl.constexpr,
    SCANS: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK)
    pivot = tl.load(Pivot + row).to(tl.float32)
    total = tl.full((BLOCK,), 0.0, tl.float32)
    for start in range(SCANS):
        cols = start * BLOCK + offsets
        valid = cols < V
        x = tl.load(P + row * S0 + tl.minimum(cols, V - 1) * S1).to(tl.float32)
        x = tl.where(valid, x, 0.0)
        total += tl.where(x >= pivot, x, 0.0)
    denominator = tl.sum(total, 0)
    for start in range(triton.cdiv(V, BLOCK)):
        cols = start * BLOCK + offsets
        valid = cols < V
        x = tl.load(P + row * S0 + tl.minimum(cols, V - 1) * S1).to(tl.float32)
        x = tl.where(valid, x, 0.0)
        y = tl.where(x >= pivot, x / denominator, 0.0)
        tl.store(O + row * V + cols, y, valid)


@triton.jit
def _transpose_sorted(
    Sorted,
    Transposed,
    B: tl.constexpr,
    V: tl.constexpr,
    PAD: tl.constexpr,
    RP: tl.constexpr,
    BLOCK: tl.constexpr,
):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row = index % RP
    col = index // RP
    source_row = tl.minimum(row, B - 1)
    source_col = tl.minimum(col, V - 1)
    value = tl.load(Sorted + source_row * PAD + source_col).to(tl.float32)
    value = tl.where((row < B) & (col < V), value, 0.0)
    tl.store(Transposed + index, value, index < V * RP)


@triton.jit
def _vector_pivot(
    Transposed,
    T,
    Pivot,
    B: tl.constexpr,
    V: tl.constexpr,
    RP: tl.constexpr,
    TS: tl.constexpr,
    IS_TENSOR: tl.constexpr,
    SINGLE: tl.constexpr,
    PV: tl.constexpr,
    ROWS: tl.constexpr,
):
    rows = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    valid = rows < B
    if IS_TENSOR:
        if SINGLE:
            positions = tl.full((ROWS,), 0, tl.int32)
        else:
            positions = tl.minimum(rows, B - 1) * TS
        p = tl.load(T + positions).to(tl.float32)
    else:
        p = tl.full((ROWS,), PV, tl.float32)
    target = 1.0 - p
    accumulator = tl.full((ROWS,), 0.0, tl.float32)
    pivot = tl.load(Transposed + rows).to(tl.float32)
    for index in range(V):
        value = tl.load(Transposed + index * RP + rows).to(tl.float32)
        active = valid & (accumulator < target)
        accumulator = tl.where(active, accumulator + value, accumulator)
        pivot = tl.where(active, value, pivot)
    tl.store(Pivot + rows, pivot)


def _select_vector_pivot(sorted_probs, top_p, vocab):
    batch, padded = sorted_probs.shape
    row_block = 128
    rows_pad = triton.cdiv(batch, row_block) * row_block
    transposed = torch.empty(
        (vocab, rows_pad), dtype=torch.float32, device=sorted_probs.device
    )
    pivots = torch.empty(
        (rows_pad,), dtype=torch.float32, device=sorted_probs.device
    )
    is_tensor = isinstance(top_p, torch.Tensor)
    if is_tensor:
        values = top_p.to(device=sorted_probs.device).reshape(-1)
        single = values.numel() == 1
        stride = values.stride(0)
        scalar = 0.0
    else:
        values = sorted_probs
        single = True
        stride = 0
        scalar = float(top_p)
    if batch > 0 and vocab > 0:
        _transpose_sorted[(triton.cdiv(vocab * rows_pad, 4096),)](
            sorted_probs,
            transposed,
            batch,
            vocab,
            padded,
            rows_pad,
            4096,
            num_warps=4,
            num_stages=1,
            enable_fp_fusion=False,
            isCloseUnrollControl=True,
        )
        _vector_pivot[(rows_pad // row_block,)](
            transposed,
            values,
            pivots,
            batch,
            vocab,
            rows_pad,
            stride,
            is_tensor,
            single,
            scalar,
            row_block,
            num_warps=4,
            num_stages=1,
            enable_fp_fusion=False,
            groups_per_cluster=1,
            isCloseUnrollControl=True,
        )
    return pivots[:batch]


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


def _radix_sort_fused(probs, digit_bits=4):
    batch, vocab = probs.shape
    block = max(1024, min(8192, 4096 * batch, triton.next_power_of_2(vocab)))
    padded = triton.cdiv(vocab, block) * block
    first = torch.empty(
        (batch, padded), dtype=torch.float32, device=probs.device
    )
    if batch == 0 or vocab == 0:
        return first
    second = torch.empty(
        (batch, padded), dtype=torch.float32, device=probs.device
    )
    tiles = padded // block
    bins = 1 << digit_bits
    width = max(128, triton.next_power_of_2(bins * tiles))
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
        isCloseUnrollControl=True,
    )
    for shift in range(0, 31, digit_bits):
        pass_bins = 1 << min(digit_bits, 31 - shift)
        pass_width = max(128, triton.next_power_of_2(pass_bins * tiles))
        _radix_hist_fused[(batch, tiles)](
            first,
            histogram,
            shift,
            padded,
            tiles,
            pass_bins,
            block,
            num_warps=4,
            num_stages=1,
            isCloseUnrollControl=True,
        )
        _radix_prefix[(batch,)](
            histogram,
            prefix,
            tiles,
            pass_bins,
            pass_width,
            num_warps=4,
            num_stages=1,
            isCloseUnrollControl=True,
        )
        _radix_scatter[(batch, tiles)](
            first,
            prefix,
            second,
            shift,
            padded,
            tiles,
            pass_bins,
            pass_width,
            block,
            num_warps=4,
            num_stages=1,
            isCloseUnrollControl=True,
        )
        first, second = second, first
    return first


def top_p_renorm(probs, top_p):
    batch, vocab = probs.shape
    out = torch.empty((batch, vocab), dtype=torch.float32, device=probs.device)
    if batch == 0 or vocab == 0:
        return out
    sorted_probs = (
        _radix_sort_fused(probs)
        if batch >= 64
        else _radix_sort_nonnegative(probs)
    )
    if batch >= 64:
        pivots = _select_vector_pivot(sorted_probs, top_p, vocab)
    else:
        pivots = _select_sorted_pivot(sorted_probs, top_p, vocab)
    _mask_normalize[(batch,)](
        probs,
        pivots,
        out,
        vocab,
        probs.stride(0),
        probs.stride(1),
        4096,
        max(2, triton.cdiv(vocab, 4096)),
        num_warps=4,
        num_stages=1,
        enable_fp_fusion=False,
        isCloseUnrollControl=True,
    )
    return out


__all__ = ["top_p_renorm"]
