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


import triton
import triton.language as tl

TOPK = 2048
CANDIDATES = 8192
TILE = 2048


_TOPK = tl.constexpr(TOPK)
_CANDIDATES = tl.constexpr(CANDIDATES)
_TILE = tl.constexpr(TILE)


@triton.jit
def _ordered(score):
    # Canonicalize signed zeros before translating FP32 to ordered bits.
    bits = tl.where(score == 0.0, 0.0, score).to(tl.uint32, bitcast=True)
    return tl.where((bits >> 31) != 0, ~bits, bits ^ 0x80000000)


@triton.jit
def _pack(key, token):
    return (key.to(tl.uint64) << 32) | (0xFFFFFFFF - token.to(tl.uint32))


@triton.jit
def _id(pair):
    return (0xFFFFFFFF - pair.to(tl.uint32)).to(tl.int32)


@triton.jit
def _visible_length(
    lengths,
    row_to_request,
    row,
    lens_stride: tl.constexpr,
    request_stride: tl.constexpr,
    context: tl.constexpr,
    num_requests: tl.constexpr,
):
    length = tl.load(lengths + row * lens_stride)
    request = tl.load(row_to_request + row * request_stride)
    valid = (
        (length >= 0)
        & (length <= context)
        & (request >= 0)
        & (request < num_requests)
    )
    return tl.where(valid, length, 0)


@triton.jit
def _store_ids(
    ids,
    columns,
    valid,
    dst,
    row,
    row_to_request,
    output_table,
    offsets,
    out_stride: tl.constexpr,
    map_row_stride: tl.constexpr,
    map_col_stride: tl.constexpr,
    request_stride: tl.constexpr,
    offset_stride: tl.constexpr,
    output_mode: tl.constexpr,
    store_mask,
):
    if output_mode == "paged":
        request = tl.load(row_to_request + row * request_stride)
        page = tl.load(
            output_table
            + request.to(tl.int64) * map_row_stride
            + (ids // 64) * map_col_stride,
            valid & store_mask,
            -1,
        )
        valid &= (page >= 0) & (page <= 33554431)
        ids = page * 64 + ids % 64
    elif output_mode == "ragged":
        offset = tl.load(offsets + row * offset_stride)
        ids += offset
    tl.store(
        dst + row.to(tl.int64) * out_stride + columns,
        tl.where(valid, ids, -1),
        store_mask,
    )


@triton.jit
def _direct_or_small(
    scores,
    lengths,
    dst,
    row_to_request,
    output_table,
    offsets,
    pitch: tl.constexpr,
    lens_stride: tl.constexpr,
    out_stride: tl.constexpr,
    map_row_stride: tl.constexpr,
    map_col_stride: tl.constexpr,
    request_stride: tl.constexpr,
    offset_stride: tl.constexpr,
    output_mode: tl.constexpr,
    context: tl.constexpr,
    num_requests: tl.constexpr,
    block: tl.constexpr,
):
    row = tl.program_id(0)
    length = _visible_length(
        lengths,
        row_to_request,
        row,
        lens_stride,
        request_stride,
        context,
        num_requests,
    )
    columns = tl.arange(0, _TOPK)
    ids = columns
    if context > _TOPK:
        if length > _TOPK:
            tokens = tl.arange(0, block)
            values = tl.load(
                scores + row.to(tl.int64) * pitch + tokens,
                tokens < length,
                -float("inf"),
            )
            pairs = tl.where(
                tokens < length,
                _pack(_ordered(values), tokens),
                0,
            )
            sorted_pairs = tl.sort(pairs, descending=True)
            ids = _id(tl.gather(sorted_pairs, columns, axis=0))
    _store_ids(
        ids,
        columns,
        columns < length,
        dst,
        row,
        row_to_request,
        output_table,
        offsets,
        out_stride,
        map_row_stride,
        map_col_stride,
        request_stride,
        offset_stride,
        output_mode,
        columns < _TOPK,
    )


@triton.jit
def _predict(
    scores,
    lengths,
    row_to_request,
    meta,
    pitch: tl.constexpr,
    lens_stride: tl.constexpr,
    request_stride: tl.constexpr,
    context: tl.constexpr,
    num_requests: tl.constexpr,
):
    row = tl.program_id(0)
    length = _visible_length(
        lengths,
        row_to_request,
        row,
        lens_stride,
        request_stride,
        context,
        num_requests,
    )
    if length > _TOPK:
        sample = tl.arange(0, _TILE)
        # One position from each contiguous stratum. The actual counts still
        # decide whether the lower bound was sufficient.
        pos = ((2 * sample + 1).to(tl.int64) * length // (2 * _TILE)).to(
            tl.int32
        )
        keys = _ordered(tl.load(scores + row.to(tl.int64) * pitch + pos))
        sorted_keys = tl.sort(keys, descending=True)
        target = tl.maximum(_TOPK * _TILE // length, 1)
        pad = (4.0 * tl.sqrt(target.to(tl.float32))).to(tl.int32) + 2
        rank = tl.minimum(target + pad, _TILE - 1)
        cut = tl.max(tl.where(sample == rank, sorted_keys, 0), axis=0)
        tl.store(meta + row * 4, cut.to(tl.int32, bitcast=True))
        tl.store(meta + row * 4 + 1, 0)
        tl.store(meta + row * 4 + 2, -1)
        tl.store(meta + row * 4 + 3, 0)


@triton.jit
def _compact(
    scores,
    lengths,
    row_to_request,
    meta,
    candidates,
    pitch: tl.constexpr,
    lens_stride: tl.constexpr,
    request_stride: tl.constexpr,
    context: tl.constexpr,
    num_requests: tl.constexpr,
):
    row = tl.program_id(1)
    start = tl.program_id(0) * _TILE
    length = _visible_length(
        lengths,
        row_to_request,
        row,
        lens_stride,
        request_stride,
        context,
        num_requests,
    )
    if (length > _TOPK) & (start < length):
        tokens = start + tl.arange(0, _TILE)
        valid = tokens < length
        key = _ordered(
            tl.load(
                scores + row.to(tl.int64) * pitch + tokens,
                valid,
                -float("inf"),
            )
        )
        cut = tl.load(meta + row * 4).to(tl.uint32, bitcast=True)
        keep = valid & (key >= cut)
        count = tl.sum(keep.to(tl.int32), 0)
        if count > 0:
            base = tl.atomic_add(meta + row * 4 + 1, count, sem="relaxed")
            prefix = tl.cumsum(keep.to(tl.int32), axis=0)
            pos = base + prefix - 1
            tl.store(
                candidates + row.to(tl.int64) * _CANDIDATES + pos,
                _pack(key, tokens).to(tl.int64, bitcast=True),
                keep & (pos < _CANDIDATES),
            )
            # The ordered integer min/max also certifies an overfull band
            # made entirely of exact ties.
            min_key = tl.min(tl.where(keep, key, 0xFFFFFFFF), 0)
            max_key = tl.max(tl.where(keep, key, 0), 0)
            meta_keys = meta.to(tl.pointer_type(tl.uint32))
            tl.atomic_min(meta_keys + row * 4 + 2, min_key, sem="relaxed")
            tl.atomic_max(meta_keys + row * 4 + 3, max_key, sem="relaxed")


@triton.jit
def _repair(
    scores,
    dst,
    row,
    row_to_request,
    output_table,
    offsets,
    length,
    out_stride: tl.constexpr,
    map_row_stride: tl.constexpr,
    map_col_stride: tl.constexpr,
    request_stride: tl.constexpr,
    offset_stride: tl.constexpr,
    output_mode: tl.constexpr,
):
    """Find the exact FP32 cutoff by walking its four bytes."""
    bins = tl.arange(0, 256)
    lane = tl.arange(0, _TILE)
    prefix = tl.full((), 0, tl.uint32)
    rank = _TOPK
    for shift in tl.static_range(24, -1, -8):
        hist = tl.full((256,), 0, tl.int32)
        for offset in range(0, length, _TILE):
            pos = offset + lane
            key = _ordered(tl.load(scores + pos, pos < length, -float("inf")))
            active = pos < length
            if shift < 24:
                active &= (key >> (shift + 8)) == prefix
            b = ((key >> shift) & 255).to(tl.int32)
            hist += tl.histogram(tl.where(active, b, 256), 256)
        suffix = tl.cumsum(hist, axis=0, reverse=True)
        bucket = tl.max(tl.where(suffix >= rank, bins, 0), 0)
        rank -= tl.sum(tl.where(bins > bucket, hist, 0), 0)
        prefix = (prefix << 8) | bucket.to(tl.uint32)

    n_greater = 0
    n_equal = 0
    for offset in range(0, length, _TILE):
        pos = offset + lane
        key = _ordered(tl.load(scores + pos, pos < length, -float("inf")))
        gt = (pos < length) & (key > prefix)
        eq = (pos < length) & (key == prefix)
        gt_pos = n_greater + tl.cumsum(gt.to(tl.int32), axis=0) - 1
        eq_rank = n_equal + tl.cumsum(eq.to(tl.int32), axis=0) - 1
        eq_pos = _TOPK - rank + eq_rank
        keep = gt | (eq & (eq_rank < rank))
        _store_ids(
            pos,
            tl.where(gt, gt_pos, eq_pos),
            keep,
            dst,
            row,
            row_to_request,
            output_table,
            offsets,
            out_stride,
            map_row_stride,
            map_col_stride,
            request_stride,
            offset_stride,
            output_mode,
            keep,
        )
        n_greater += tl.sum(gt.to(tl.int32), 0)
        n_equal += tl.sum(eq.to(tl.int32), 0)


@triton.jit
def _finish(
    scores,
    lengths,
    meta,
    candidates,
    dst,
    row_to_request,
    output_table,
    offsets,
    pitch: tl.constexpr,
    lens_stride: tl.constexpr,
    out_stride: tl.constexpr,
    map_row_stride: tl.constexpr,
    map_col_stride: tl.constexpr,
    request_stride: tl.constexpr,
    offset_stride: tl.constexpr,
    output_mode: tl.constexpr,
    context: tl.constexpr,
    num_requests: tl.constexpr,
):
    row = tl.program_id(0)
    length = _visible_length(
        lengths,
        row_to_request,
        row,
        lens_stride,
        request_stride,
        context,
        num_requests,
    )
    columns = tl.arange(0, _TOPK)
    if length <= _TOPK:
        _store_ids(
            columns,
            columns,
            columns < length,
            dst,
            row,
            row_to_request,
            output_table,
            offsets,
            out_stride,
            map_row_stride,
            map_col_stride,
            request_stride,
            offset_stride,
            output_mode,
            columns < _TOPK,
        )
    else:
        count = tl.load(meta + row * 4 + 1)
        minimum = tl.load(meta + row * 4 + 2)
        maximum = tl.load(meta + row * 4 + 3)
        base = candidates + row.to(tl.int64) * _CANDIDATES
        if (count >= _TOPK) & (minimum == maximum):
            keys = tl.load(base + columns).to(tl.uint64, bitcast=True)
            _store_ids(
                _id(keys),
                columns,
                columns < _TOPK,
                dst,
                row,
                row_to_request,
                output_table,
                offsets,
                out_stride,
                map_row_stride,
                map_col_stride,
                request_stride,
                offset_stride,
                output_mode,
                columns < _TOPK,
            )
        elif (count >= _TOPK) & (count <= _CANDIDATES):
            slots = tl.arange(0, _CANDIDATES)
            keys = tl.load(base + slots, slots < count, 0).to(
                tl.uint64, bitcast=True
            )
            selected = tl.gather(
                tl.sort(keys, descending=True), columns, axis=0
            )
            _store_ids(
                _id(selected),
                columns,
                columns < _TOPK,
                dst,
                row,
                row_to_request,
                output_table,
                offsets,
                out_stride,
                map_row_stride,
                map_col_stride,
                request_stride,
                offset_stride,
                output_mode,
                columns < _TOPK,
            )
        else:
            _repair(
                scores + row.to(tl.int64) * pitch,
                dst,
                row,
                row_to_request,
                output_table,
                offsets,
                length,
                out_stride,
                map_row_stride,
                map_col_stride,
                request_stride,
                offset_stride,
                output_mode,
            )


def select(
    scores,
    lengths,
    row_to_request,
    output_table,
    offsets,
    dst,
    context,
    pitch,
    mode,
    num_requests,
    candidates=None,
    meta=None,
):
    rows = dst.shape[0]
    table_strides = (0, 0) if output_table is None else output_table.stride()
    offset_stride = 0 if offsets is None else offsets.stride(0)
    request_stride = row_to_request.stride(0)
    common = (
        pitch,
        lengths.stride(0),
        dst.stride(0),
        *table_strides,
        request_stride,
        offset_stride,
        mode,
        context,
        num_requests,
    )
    if context <= CANDIDATES:
        _direct_or_small[(rows,)](
            scores,
            lengths,
            dst,
            row_to_request,
            output_table,
            offsets,
            *common,
            max(TOPK, triton.next_power_of_2(context)),
            num_warps=8 if context > TOPK else 4,
        )
        return
    scan = (pitch, lengths.stride(0), request_stride, context, num_requests)
    _predict[(rows,)](scores, lengths, row_to_request, meta, *scan)
    _compact[(triton.cdiv(context, TILE), rows)](
        scores, lengths, row_to_request, meta, candidates, *scan
    )
    _finish[(rows,)](
        scores,
        lengths,
        meta,
        candidates,
        dst,
        row_to_request,
        output_table,
        offsets,
        *common,
        num_warps=8,
    )
