# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
# Adapted from SGLang qsa/graph_metadata.py at 1b6e7d09a751bc53dec30e839e944e8662ca0bc5.
"""Build non-speculative QSA decode metadata in caller-owned buffers."""

import triton
import triton.language as tl

__all__ = ["qsa_build_decode_metadata"]


@triton.jit
def _qsa_build_decode_metadata(
    lengths,
    requests,
    table,
    sequence,
    row_requests,
    prefix,
    compressed,
    writes,
    pages,
    positions,
    states,
    ring,
    LENGTH_STRIDE: tl.constexpr,
    REQUEST_STRIDE: tl.constexpr,
    TABLE_ROW: tl.constexpr,
    TABLE_COL: tl.constexpr,
    TABLE_WIDTH: tl.constexpr,
    MAX_PAGES: tl.constexpr,
    RATIO: tl.constexpr,
    FULL_PAGE: tl.constexpr,
    REAL_ROWS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    real = row < REAL_ROWS
    length = tl.load(lengths + row * LENGTH_STRIDE, real, other=1).to(tl.int32)
    request = tl.load(requests + row * REQUEST_STRIDE, real, other=0).to(
        tl.int64
    )
    current = tl.maximum(length - 1, 0)
    if tl.program_id(1) == 0:
        tl.store(sequence + row, length)
        tl.store(row_requests + row, request)
        tl.store(prefix + row, current)
        tl.store(compressed + row, length // RATIO)
        last = tl.load(table + request * TABLE_ROW + current * TABLE_COL)
        tl.store(
            writes + row,
            tl.where((length > 0) & (length % RATIO == 0), last // RATIO, 0),
        )
        tl.store(positions + row, current)
        tl.store(states + row, request * RATIO + current % RATIO)
        for member in tl.static_range(RATIO):
            position = tl.maximum(current - (RATIO - 1 - member), 0)
            tl.store(
                ring + row * RATIO + member, request * RATIO + position % RATIO
            )
    page = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    valid = page < tl.minimum(MAX_PAGES, TABLE_WIDTH // FULL_PAGE)
    slot = tl.load(
        table + request * TABLE_ROW + page * FULL_PAGE * TABLE_COL,
        valid,
        other=0,
    )
    tl.store(
        pages + row * MAX_PAGES + page, tl.maximum(slot // FULL_PAGE, 0), valid
    )


def qsa_build_decode_metadata(
    seq_lens,
    req_pool_indices,
    req_to_token,
    sequence_lengths,
    row_req_pool_indices,
    prefix_lengths,
    compressed_lengths,
    write_locs,
    page_table,
    logical_positions,
    state_slots,
    ring_locs,
    compress_ratio,
    full_page_size,
    num_padding=0,
):
    """Mutate the nine contiguous output buffers; return None.

    Inputs are integer tensors. The request table may have arbitrary strides.
    The output row count is the captured batch capacity. Padded tail rows use
    sequence length 1 and reserved request slot 0. Page columns outside the
    request-table width are left unchanged. Inputs must not alias outputs.
    No allocation, host readback, pool object or speculative state is required.
    """
    rows = sequence_lengths.numel()
    max_pages = page_table.shape[1]
    _qsa_build_decode_metadata[(rows, triton.cdiv(max_pages, 128))](
        seq_lens,
        req_pool_indices,
        req_to_token,
        sequence_lengths,
        row_req_pool_indices,
        prefix_lengths,
        compressed_lengths,
        write_locs,
        page_table,
        logical_positions,
        state_slots,
        ring_locs,
        seq_lens.stride(0),
        req_pool_indices.stride(0),
        req_to_token.stride(0),
        req_to_token.stride(1),
        req_to_token.shape[1],
        max_pages,
        compress_ratio,
        full_page_size,
        rows - num_padding,
        128,
        num_warps=1,
    )
