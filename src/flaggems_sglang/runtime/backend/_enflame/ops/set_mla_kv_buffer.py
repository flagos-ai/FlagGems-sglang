# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

import torch
import triton
import triton.language as tl

__all__ = ["set_mla_kv_buffer"]

# GCU adaptations vs the generic copy (keep in sync with the launch sites):
# 1. Zero-copy int32 word view of int64 ``loc`` (no ATen cast): torch_gcu
#    without TORCH_GCU_ENABLE_INT64_AND_UINT64 keeps device int64 values as
#    compact int32 words (element j at word j), while with the gate on they
#    sit in the low word of each 8-byte slot (word 2*j). Either way a Triton
#    load at the declared int64 stride reads neighbor-joined garbage (the
#    baseline abort), so slots are read through ``loc.view(torch.int32)``
#    with the matching word stride. Slot ids stay far below 2**31, so the
#    low word is value-exact (two's complement keeps -1 as -1). This drops
#    the per-call ATen int64->int32 cast kernel that the env gate turns
#    into a real 8-byte read (measured +1.3~2.8us per bench call, i.e.
#    +32% at n=1; same scheme as concat_and_cache_mla /
#    indexer_k_quant_and_cache).
# 2. The generic (ROWS, BLOCK) rank-2 masked tiles failed to compile at
#    the 11-08 toolchain state (``memref_ext.slice_pad_start ... 8 vs 7``,
#    see compiler_issues/sglang_set_mla_kv_buffer.md; no longer
#    reproducing under the int64 env as of 2026-11-12, but the env-off
#    generic still wild-writes via un-narrowed int64 slots), so every GCU
#    path is
#    strictly 1D. The fast path mirrors get_mla_kv_buffer: statically
#    unrolled three-segment batches (loc scalars -> row loads -> stores,
#    min-clamp tails, no per-lane div/mod) with an n-based dispatch
#    (u1 <= 48 / u4 <= 128 / u8 <= 384 / persistent 8-row grid-stride
#    grid=48; scattered stores prefer few long-lived programs: 256 tiled
#    blocks 40.4us vs persistent 23.8us at n=4096). The persistent arm
#    runs num_warps=2 (2026-11-12 sweep on GCU400: n=4096 24.97 -> 17.53us,
#    -30% vs nw4; u1/u4/u8 arms are nw-flat and stay at nw4; grid 96
#    regresses to 29.4us). Measured vs the flat form (bench core):
#    4.04/6.29/7.00/9.50/41.53 -> ~3.8/4.0/4.3/6.6/22.9 us at
#    n=1/8/64/512/4096.
# 3. Fallback for non-row-major inputs keeps the earlier flat kernels
#    (single flat index space, ``row = flat // W`` native rem, word-packed
#    uint64/uint32 when 8B/4B alignment holds). Note the packing only
#    helps the flat per-lane form; in the batched form bf16 lanes beat
#    uint64 packing (probe: 23.8 vs 31.2us at n=4096).


@triton.jit
def _fused_small_packed(
    src,
    dst,
    loc,
    nope,
    rope,
    rows,
    n,
    SRC_S0: tl.constexpr,
    NOPE_S0: tl.constexpr,
    ROPE_S0: tl.constexpr,
    NOPE_WORDS: tl.constexpr,
    ROPE_WORDS: tl.constexpr,
    LOC_S0: tl.constexpr,
    ROW_BLOCK: tl.constexpr,
    LOC_BLOCK: tl.constexpr,
    WORD_BLOCK: tl.constexpr,
):
    row = tl.program_id(0) * ROW_BLOCK + tl.arange(0, ROW_BLOCK)
    col = tl.arange(0, WORD_BLOCK)
    row_mask = row < rows
    word_mask = col < (NOPE_WORDS + ROPE_WORDS)
    loc_index = tl.arange(0, LOC_BLOCK)
    loc_mask = loc_index < n
    loc_values = tl.load(loc + loc_index * LOC_S0, mask=loc_mask, other=-1)
    matches = loc_values[None, :] == row[:, None].to(tl.int64)
    matches = matches & row_mask[:, None] & loc_mask[None, :]
    source_row = tl.max(tl.where(matches, loc_index[None, :], -1), axis=1)
    hit = source_row >= 0
    src_mask = row_mask[:, None] & word_mask[None, :]
    old_value = tl.load(
        src + row[:, None] * SRC_S0 + col[None, :], mask=src_mask
    )
    nope_mask = hit[:, None] & (col[None, :] < NOPE_WORDS)
    rope_mask = (
        hit[:, None] & (col[None, :] >= NOPE_WORDS) & word_mask[None, :]
    )
    nope_value = tl.load(
        nope + source_row[:, None] * NOPE_S0 + col[None, :], mask=nope_mask
    )
    rope_col = col[None, :] - NOPE_WORDS
    rope_value = tl.load(
        rope + source_row[:, None] * ROPE_S0 + rope_col, mask=rope_mask
    )
    update = tl.where(col[None, :] < NOPE_WORDS, nope_value, rope_value)
    value = tl.where(hit[:, None], update, old_value)
    tl.store(dst + row[:, None] * SRC_S0 + col[None, :], value, mask=src_mask)


@triton.jit
def _copy_words(src, dst, words, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < words
    value = tl.load(src + offsets, mask=mask)
    tl.store(dst + offsets, value, mask=mask)


@triton.jit
def _fill_flags(flags, rows, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < rows
    tl.store(flags + offsets, 0, mask=mask)


@triton.jit
def _copy_strided(
    src,
    dst,
    rows,
    width,
    SRC_S0: tl.constexpr,
    SRC_S1: tl.constexpr,
    DST_S0: tl.constexpr,
    DST_S1: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = (row < rows) & (col < width)
    value = tl.load(src + row * SRC_S0 + col * SRC_S1, mask=mask)
    tl.store(dst + row * DST_S0 + col * DST_S1, value, mask=mask)


@triton.jit
def _scatter_words(
    dst,
    loc,
    nope,
    rope,
    total,
    NOPE_WORDS: tl.constexpr,
    ROPE_WORDS: tl.constexpr,
    DST_S0: tl.constexpr,
    LOC_S0: tl.constexpr,
    NOPE_S0: tl.constexpr,
    ROPE_S0: tl.constexpr,
    WORD_BLOCK: tl.constexpr,
):
    flat = tl.program_id(0) * WORD_BLOCK + tl.arange(0, WORD_BLOCK)
    flat_mask = flat < total
    row = flat // (NOPE_WORDS + ROPE_WORDS)
    word = flat % (NOPE_WORDS + ROPE_WORDS)
    target = tl.load(loc + row * LOC_S0, mask=flat_mask, other=0)
    nope_mask = flat_mask & (word < NOPE_WORDS)
    rope_mask = flat_mask & (word >= NOPE_WORDS)
    nope_value = tl.load(nope + row * NOPE_S0 + word, mask=nope_mask)
    rope_value = tl.load(
        rope + row * ROPE_S0 + word - NOPE_WORDS, mask=rope_mask
    )
    tl.store(dst + target * DST_S0 + word, nope_value, mask=nope_mask)
    tl.store(dst + target * DST_S0 + word, rope_value, mask=rope_mask)


@triton.jit
def _scatter_words_mark(
    dst,
    loc,
    nope,
    rope,
    flags,
    rows,
    NOPE_WORDS: tl.constexpr,
    ROPE_WORDS: tl.constexpr,
    DST_S0: tl.constexpr,
    LOC_S0: tl.constexpr,
    NOPE_S0: tl.constexpr,
    ROPE_S0: tl.constexpr,
    ROW_BLOCK: tl.constexpr,
    NOPE_BLOCK: tl.constexpr,
    ROPE_BLOCK: tl.constexpr,
):
    row = tl.program_id(0) * ROW_BLOCK + tl.arange(0, ROW_BLOCK)
    row_mask = row < rows
    target = tl.load(loc + row * LOC_S0, mask=row_mask, other=0).to(tl.int32)
    tl.store(flags + target, 1, mask=row_mask)
    nope_col = tl.arange(0, NOPE_BLOCK)
    rope_col = tl.arange(0, ROPE_BLOCK)
    nope_mask = row_mask[:, None] & (nope_col[None, :] < NOPE_WORDS)
    rope_mask = row_mask[:, None] & (rope_col[None, :] < ROPE_WORDS)
    nope_value = tl.load(
        nope + row[:, None] * NOPE_S0 + nope_col[None, :], mask=nope_mask
    )
    rope_value = tl.load(
        rope + row[:, None] * ROPE_S0 + rope_col[None, :], mask=rope_mask
    )
    tl.store(
        dst + target[:, None] * DST_S0 + nope_col[None, :],
        nope_value,
        mask=nope_mask,
    )
    tl.store(
        dst + target[:, None] * DST_S0 + NOPE_WORDS + rope_col[None, :],
        rope_value,
        mask=rope_mask,
    )


@triton.jit
def _copy_unmarked_rows(
    src,
    dst,
    flags,
    rows,
    WORDS_PER_ROW: tl.constexpr,
    SRC_S0: tl.constexpr,
    DST_S0: tl.constexpr,
    ROW_BLOCK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0) * ROW_BLOCK + tl.arange(0, ROW_BLOCK)
    col = tl.arange(0, BLOCK)
    row_mask = row < rows
    active = tl.load(flags + row, mask=row_mask, other=1) == 0
    mask = row_mask[:, None] & active[:, None] & (col[None, :] < WORDS_PER_ROW)
    value = tl.load(src + row[:, None] * SRC_S0 + col[None, :], mask=mask)
    tl.store(dst + row[:, None] * DST_S0 + col[None, :], value, mask=mask)


@triton.jit
def _scatter_values(
    dst,
    loc,
    nope,
    rope,
    total,
    NOPE: tl.constexpr,
    ROPE: tl.constexpr,
    DST_S0: tl.constexpr,
    DST_S1: tl.constexpr,
    LOC_S0: tl.constexpr,
    NOPE_S0: tl.constexpr,
    NOPE_S1: tl.constexpr,
    ROPE_S0: tl.constexpr,
    ROPE_S1: tl.constexpr,
    BLOCK: tl.constexpr,
):
    flat = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    flat_mask = flat < total
    row = flat // (NOPE + ROPE)
    col = flat % (NOPE + ROPE)
    target = tl.load(loc + row * LOC_S0, mask=flat_mask, other=0)
    nope_mask = flat_mask & (col < NOPE)
    rope_mask = flat_mask & (col >= NOPE)
    nope_value = tl.load(
        nope + row * NOPE_S0 + col * NOPE_S1, mask=nope_mask
    )
    rope_value = tl.load(
        rope + row * ROPE_S0 + (col - NOPE) * ROPE_S1, mask=rope_mask
    )
    tl.store(dst + target * DST_S0 + col * DST_S1, nope_value, mask=nope_mask)
    tl.store(dst + target * DST_S0 + col * DST_S1, rope_value, mask=rope_mask)


# GCU launch form for the batched scatter arms (mirrors get_mla_kv_buffer):
# one program per SP on the 48-SP part; the per-program fixed cost
# dominates small blocks, so the persistent form lets each SP amortize it
# across many 8-row blocks.
_PERSISTENT_GRID = 48
# The batch kernels stage the scattered rows in on-chip memory at roughly
# 25 B per lane; 8 rows of BLOCK_N + BLOCK_R lanes must stay far below the
# ~1 MB per-program budget (the read-side twin measured the u64 x 576-lane
# form failing compilation with "dsmSize ... Insufficient memory").
_FAST_PATH_MAX_LANES = 16384
# Empirical L600 dispatch lines (probe, do_bench 400/200ms median):
# single-row programs win while they fit one 48-program wave, 4-row batches
# win up to 128 rows, 8-row batches up to 384 (= 48 programs), beyond that
# the persistent 8-row grid-stride form wins (scattered stores prefer few
# long-lived programs over many short-lived ones: 256 blocks tiled 40.4µs
# vs persistent 23.8µs at n=4096).
_ROWS_U1_MAX = 48
_ROWS_U4_MAX = 128
_ROWS_U8_MAX = 384


@triton.jit
def _row_target(loc_ptr, row, n, stride_loc):
    # Tail-safe row pick: a clamped row re-reads source row n-1 and
    # re-writes that row's own target slot with the same value (both from
    # the same program, sequentially), so partial batches need no masks or
    # branches. loc_ptr is the zero-copy int32 word view of the slot ids
    # (stride_loc 2 under TORCH_GCU_ENABLE_INT64_AND_UINT64, else 1; see
    # the wrapper note above).
    c = tl.minimum(row, n - 1)
    t = tl.load(loc_ptr + c * stride_loc).to(tl.int64)
    return c, t


@triton.jit
def _load_nope_row(nope_ptr, c, stride_n, d_n, NOPE_D: tl.constexpr, BLOCK_N: tl.constexpr):
    if BLOCK_N == NOPE_D:
        return tl.load(nope_ptr + c * stride_n + d_n)
    return tl.load(nope_ptr + c * stride_n + d_n, mask=d_n < NOPE_D, other=0)


@triton.jit
def _load_rope_row(rope_ptr, c, stride_r, d_r, ROPE_D: tl.constexpr, BLOCK_R: tl.constexpr):
    if BLOCK_R == ROPE_D:
        return tl.load(rope_ptr + c * stride_r + d_r)
    return tl.load(rope_ptr + c * stride_r + d_r, mask=d_r < ROPE_D, other=0)


@triton.jit
def _store_nope_row(dst_ptr, t, v, stride_d, d_n, NOPE_D: tl.constexpr, BLOCK_N: tl.constexpr):
    if BLOCK_N == NOPE_D:
        tl.store(dst_ptr + t * stride_d + d_n, v.to(dst_ptr.dtype.element_ty))
    else:
        tl.store(
            dst_ptr + t * stride_d + d_n,
            v.to(dst_ptr.dtype.element_ty),
            mask=d_n < NOPE_D,
        )


@triton.jit
def _store_rope_row(
    dst_ptr, t, v, stride_d, d_r, NOPE_D: tl.constexpr, ROPE_D: tl.constexpr, BLOCK_R: tl.constexpr
):
    if BLOCK_R == ROPE_D:
        tl.store(dst_ptr + t * stride_d + NOPE_D + d_r, v.to(dst_ptr.dtype.element_ty))
    else:
        tl.store(
            dst_ptr + t * stride_d + NOPE_D + d_r,
            v.to(dst_ptr.dtype.element_ty),
            mask=d_r < ROPE_D,
        )


@triton.jit
def _scatter_rows_u1(
    dst,
    loc,
    nope,
    rope,
    n,
    stride_d,
    stride_loc,
    stride_n,
    stride_r,
    NOPE_D: tl.constexpr,
    ROPE_D: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    base = tl.program_id(0) * 1
    d_n = tl.max_contiguous(tl.multiple_of(tl.arange(0, BLOCK_N), BLOCK_N), BLOCK_N)
    d_r = tl.max_contiguous(tl.multiple_of(tl.arange(0, BLOCK_R), BLOCK_R), BLOCK_R)
    c0, t0 = _row_target(loc, base + 0, n, stride_loc)
    v0n = _load_nope_row(nope, c0, stride_n, d_n, NOPE_D, BLOCK_N)
    v0r = _load_rope_row(rope, c0, stride_r, d_r, ROPE_D, BLOCK_R)
    _store_nope_row(dst, t0, v0n, stride_d, d_n, NOPE_D, BLOCK_N)
    _store_rope_row(dst, t0, v0r, stride_d, d_r, NOPE_D, ROPE_D, BLOCK_R)


@triton.jit
def _scatter_rows_u4(
    dst,
    loc,
    nope,
    rope,
    n,
    stride_d,
    stride_loc,
    stride_n,
    stride_r,
    NOPE_D: tl.constexpr,
    ROPE_D: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    base = tl.program_id(0) * 4
    d_n = tl.max_contiguous(tl.multiple_of(tl.arange(0, BLOCK_N), BLOCK_N), BLOCK_N)
    d_r = tl.max_contiguous(tl.multiple_of(tl.arange(0, BLOCK_R), BLOCK_R), BLOCK_R)
    c0, t0 = _row_target(loc, base + 0, n, stride_loc)
    c1, t1 = _row_target(loc, base + 1, n, stride_loc)
    c2, t2 = _row_target(loc, base + 2, n, stride_loc)
    c3, t3 = _row_target(loc, base + 3, n, stride_loc)
    v0n = _load_nope_row(nope, c0, stride_n, d_n, NOPE_D, BLOCK_N)
    v1n = _load_nope_row(nope, c1, stride_n, d_n, NOPE_D, BLOCK_N)
    v2n = _load_nope_row(nope, c2, stride_n, d_n, NOPE_D, BLOCK_N)
    v3n = _load_nope_row(nope, c3, stride_n, d_n, NOPE_D, BLOCK_N)
    v0r = _load_rope_row(rope, c0, stride_r, d_r, ROPE_D, BLOCK_R)
    v1r = _load_rope_row(rope, c1, stride_r, d_r, ROPE_D, BLOCK_R)
    v2r = _load_rope_row(rope, c2, stride_r, d_r, ROPE_D, BLOCK_R)
    v3r = _load_rope_row(rope, c3, stride_r, d_r, ROPE_D, BLOCK_R)
    _store_nope_row(dst, t0, v0n, stride_d, d_n, NOPE_D, BLOCK_N)
    _store_nope_row(dst, t1, v1n, stride_d, d_n, NOPE_D, BLOCK_N)
    _store_nope_row(dst, t2, v2n, stride_d, d_n, NOPE_D, BLOCK_N)
    _store_nope_row(dst, t3, v3n, stride_d, d_n, NOPE_D, BLOCK_N)
    _store_rope_row(dst, t0, v0r, stride_d, d_r, NOPE_D, ROPE_D, BLOCK_R)
    _store_rope_row(dst, t1, v1r, stride_d, d_r, NOPE_D, ROPE_D, BLOCK_R)
    _store_rope_row(dst, t2, v2r, stride_d, d_r, NOPE_D, ROPE_D, BLOCK_R)
    _store_rope_row(dst, t3, v3r, stride_d, d_r, NOPE_D, ROPE_D, BLOCK_R)


@triton.jit
def _scatter_rows_u8(
    dst,
    loc,
    nope,
    rope,
    n,
    stride_d,
    stride_loc,
    stride_n,
    stride_r,
    NOPE_D: tl.constexpr,
    ROPE_D: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    base = tl.program_id(0) * 8
    d_n = tl.max_contiguous(tl.multiple_of(tl.arange(0, BLOCK_N), BLOCK_N), BLOCK_N)
    d_r = tl.max_contiguous(tl.multiple_of(tl.arange(0, BLOCK_R), BLOCK_R), BLOCK_R)
    c0, t0 = _row_target(loc, base + 0, n, stride_loc)
    c1, t1 = _row_target(loc, base + 1, n, stride_loc)
    c2, t2 = _row_target(loc, base + 2, n, stride_loc)
    c3, t3 = _row_target(loc, base + 3, n, stride_loc)
    c4, t4 = _row_target(loc, base + 4, n, stride_loc)
    c5, t5 = _row_target(loc, base + 5, n, stride_loc)
    c6, t6 = _row_target(loc, base + 6, n, stride_loc)
    c7, t7 = _row_target(loc, base + 7, n, stride_loc)
    v0n = _load_nope_row(nope, c0, stride_n, d_n, NOPE_D, BLOCK_N)
    v1n = _load_nope_row(nope, c1, stride_n, d_n, NOPE_D, BLOCK_N)
    v2n = _load_nope_row(nope, c2, stride_n, d_n, NOPE_D, BLOCK_N)
    v3n = _load_nope_row(nope, c3, stride_n, d_n, NOPE_D, BLOCK_N)
    v4n = _load_nope_row(nope, c4, stride_n, d_n, NOPE_D, BLOCK_N)
    v5n = _load_nope_row(nope, c5, stride_n, d_n, NOPE_D, BLOCK_N)
    v6n = _load_nope_row(nope, c6, stride_n, d_n, NOPE_D, BLOCK_N)
    v7n = _load_nope_row(nope, c7, stride_n, d_n, NOPE_D, BLOCK_N)
    v0r = _load_rope_row(rope, c0, stride_r, d_r, ROPE_D, BLOCK_R)
    v1r = _load_rope_row(rope, c1, stride_r, d_r, ROPE_D, BLOCK_R)
    v2r = _load_rope_row(rope, c2, stride_r, d_r, ROPE_D, BLOCK_R)
    v3r = _load_rope_row(rope, c3, stride_r, d_r, ROPE_D, BLOCK_R)
    v4r = _load_rope_row(rope, c4, stride_r, d_r, ROPE_D, BLOCK_R)
    v5r = _load_rope_row(rope, c5, stride_r, d_r, ROPE_D, BLOCK_R)
    v6r = _load_rope_row(rope, c6, stride_r, d_r, ROPE_D, BLOCK_R)
    v7r = _load_rope_row(rope, c7, stride_r, d_r, ROPE_D, BLOCK_R)
    _store_nope_row(dst, t0, v0n, stride_d, d_n, NOPE_D, BLOCK_N)
    _store_nope_row(dst, t1, v1n, stride_d, d_n, NOPE_D, BLOCK_N)
    _store_nope_row(dst, t2, v2n, stride_d, d_n, NOPE_D, BLOCK_N)
    _store_nope_row(dst, t3, v3n, stride_d, d_n, NOPE_D, BLOCK_N)
    _store_nope_row(dst, t4, v4n, stride_d, d_n, NOPE_D, BLOCK_N)
    _store_nope_row(dst, t5, v5n, stride_d, d_n, NOPE_D, BLOCK_N)
    _store_nope_row(dst, t6, v6n, stride_d, d_n, NOPE_D, BLOCK_N)
    _store_nope_row(dst, t7, v7n, stride_d, d_n, NOPE_D, BLOCK_N)
    _store_rope_row(dst, t0, v0r, stride_d, d_r, NOPE_D, ROPE_D, BLOCK_R)
    _store_rope_row(dst, t1, v1r, stride_d, d_r, NOPE_D, ROPE_D, BLOCK_R)
    _store_rope_row(dst, t2, v2r, stride_d, d_r, NOPE_D, ROPE_D, BLOCK_R)
    _store_rope_row(dst, t3, v3r, stride_d, d_r, NOPE_D, ROPE_D, BLOCK_R)
    _store_rope_row(dst, t4, v4r, stride_d, d_r, NOPE_D, ROPE_D, BLOCK_R)
    _store_rope_row(dst, t5, v5r, stride_d, d_r, NOPE_D, ROPE_D, BLOCK_R)
    _store_rope_row(dst, t6, v6r, stride_d, d_r, NOPE_D, ROPE_D, BLOCK_R)
    _store_rope_row(dst, t7, v7r, stride_d, d_r, NOPE_D, ROPE_D, BLOCK_R)


@triton.jit
def _scatter_rows_persistent(
    dst,
    loc,
    nope,
    rope,
    n_blocks,
    n,
    stride_d,
    stride_loc,
    stride_n,
    stride_r,
    NOPE_D: tl.constexpr,
    ROPE_D: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    # Persistent grid-stride form: each program walks 8-row blocks
    # pid, pid + P, pid + 2P, ... (P = tl.num_programs(0) = 48).
    d_n = tl.max_contiguous(tl.multiple_of(tl.arange(0, BLOCK_N), BLOCK_N), BLOCK_N)
    d_r = tl.max_contiguous(tl.multiple_of(tl.arange(0, BLOCK_R), BLOCK_R), BLOCK_R)
    for blk in range(tl.program_id(0), n_blocks, tl.num_programs(0)):
        base = blk * 8
        c0, t0 = _row_target(loc, base + 0, n, stride_loc)
        c1, t1 = _row_target(loc, base + 1, n, stride_loc)
        c2, t2 = _row_target(loc, base + 2, n, stride_loc)
        c3, t3 = _row_target(loc, base + 3, n, stride_loc)
        c4, t4 = _row_target(loc, base + 4, n, stride_loc)
        c5, t5 = _row_target(loc, base + 5, n, stride_loc)
        c6, t6 = _row_target(loc, base + 6, n, stride_loc)
        c7, t7 = _row_target(loc, base + 7, n, stride_loc)
        v0n = _load_nope_row(nope, c0, stride_n, d_n, NOPE_D, BLOCK_N)
        v1n = _load_nope_row(nope, c1, stride_n, d_n, NOPE_D, BLOCK_N)
        v2n = _load_nope_row(nope, c2, stride_n, d_n, NOPE_D, BLOCK_N)
        v3n = _load_nope_row(nope, c3, stride_n, d_n, NOPE_D, BLOCK_N)
        v4n = _load_nope_row(nope, c4, stride_n, d_n, NOPE_D, BLOCK_N)
        v5n = _load_nope_row(nope, c5, stride_n, d_n, NOPE_D, BLOCK_N)
        v6n = _load_nope_row(nope, c6, stride_n, d_n, NOPE_D, BLOCK_N)
        v7n = _load_nope_row(nope, c7, stride_n, d_n, NOPE_D, BLOCK_N)
        v0r = _load_rope_row(rope, c0, stride_r, d_r, ROPE_D, BLOCK_R)
        v1r = _load_rope_row(rope, c1, stride_r, d_r, ROPE_D, BLOCK_R)
        v2r = _load_rope_row(rope, c2, stride_r, d_r, ROPE_D, BLOCK_R)
        v3r = _load_rope_row(rope, c3, stride_r, d_r, ROPE_D, BLOCK_R)
        v4r = _load_rope_row(rope, c4, stride_r, d_r, ROPE_D, BLOCK_R)
        v5r = _load_rope_row(rope, c5, stride_r, d_r, ROPE_D, BLOCK_R)
        v6r = _load_rope_row(rope, c6, stride_r, d_r, ROPE_D, BLOCK_R)
        v7r = _load_rope_row(rope, c7, stride_r, d_r, ROPE_D, BLOCK_R)
        _store_nope_row(dst, t0, v0n, stride_d, d_n, NOPE_D, BLOCK_N)
        _store_nope_row(dst, t1, v1n, stride_d, d_n, NOPE_D, BLOCK_N)
        _store_nope_row(dst, t2, v2n, stride_d, d_n, NOPE_D, BLOCK_N)
        _store_nope_row(dst, t3, v3n, stride_d, d_n, NOPE_D, BLOCK_N)
        _store_nope_row(dst, t4, v4n, stride_d, d_n, NOPE_D, BLOCK_N)
        _store_nope_row(dst, t5, v5n, stride_d, d_n, NOPE_D, BLOCK_N)
        _store_nope_row(dst, t6, v6n, stride_d, d_n, NOPE_D, BLOCK_N)
        _store_nope_row(dst, t7, v7n, stride_d, d_n, NOPE_D, BLOCK_N)
        _store_rope_row(dst, t0, v0r, stride_d, d_r, NOPE_D, ROPE_D, BLOCK_R)
        _store_rope_row(dst, t1, v1r, stride_d, d_r, NOPE_D, ROPE_D, BLOCK_R)
        _store_rope_row(dst, t2, v2r, stride_d, d_r, NOPE_D, ROPE_D, BLOCK_R)
        _store_rope_row(dst, t3, v3r, stride_d, d_r, NOPE_D, ROPE_D, BLOCK_R)
        _store_rope_row(dst, t4, v4r, stride_d, d_r, NOPE_D, ROPE_D, BLOCK_R)
        _store_rope_row(dst, t5, v5r, stride_d, d_r, NOPE_D, ROPE_D, BLOCK_R)
        _store_rope_row(dst, t6, v6r, stride_d, d_r, NOPE_D, ROPE_D, BLOCK_R)
        _store_rope_row(dst, t7, v7r, stride_d, d_r, NOPE_D, ROPE_D, BLOCK_R)


def set_mla_kv_buffer(kv_buffer, loc, cache_k_nope, cache_k_rope):
    out = kv_buffer
    rows = kv_buffer.shape[0]  # noqa: F841
    width = kv_buffer.shape[1]
    count = kv_buffer.numel()
    packed_copy = (
        kv_buffer.element_size() == 2
        and width % 2 == 0
        and kv_buffer.stride(1) == 1
        and kv_buffer.stride(0) == width
        and kv_buffer.storage_offset() % 2 == 0
        and out.stride(1) == 1
        and out.stride(0) == width
        and out.storage_offset() % 2 == 0
        and count % 2 == 0
    )
    n = loc.numel()
    loc_word_stride = loc.stride(0)
    if loc.dtype == torch.int64:
        # Zero-copy int32 word view (see note 1 above): value-exact for
        # slot ids in [-2**31, 2**31) and avoids the per-call ATen cast
        # kernel that the int64 env gate turns into a real 8-byte read.
        loc = loc.contiguous().view(torch.int32)
        loc_word_stride = (
            2 if os.environ.get("TORCH_GCU_ENABLE_INT64_AND_UINT64") == "1" else 1
        )
    nvidia_like = getattr(torch.version, "hip", None) is None  # noqa: F841
    flags = None  # noqa: F841
    nope_width = cache_k_nope.shape[-1]
    rope_width = cache_k_rope.shape[-1]
    packed_scatter = (
        packed_copy
        and cache_k_nope.dtype == kv_buffer.dtype
        and cache_k_rope.dtype == kv_buffer.dtype
        and nope_width % 2 == 0
        and rope_width % 2 == 0
        and cache_k_nope.stride(1) == 1
        and cache_k_rope.stride(1) == 1
        and cache_k_nope.stride(0) == nope_width
        and cache_k_rope.stride(0) == rope_width
        and cache_k_nope.storage_offset() % 2 == 0
        and cache_k_rope.storage_offset() % 2 == 0
    )
    if n:
        block_n = triton.next_power_of_2(max(nope_width, 1))
        block_r = triton.next_power_of_2(max(rope_width, 1))
        # Batched three-segment arm (mirrors get_mla_kv_buffer): loc
        # scalars -> row loads -> stores, statically unrolled, min-clamp
        # tails. Fastest for row-major tensors of any element dtype.
        if (
            kv_buffer.stride(1) == 1
            and cache_k_nope.stride(1) == 1
            and cache_k_rope.stride(1) == 1
            and 8 * (block_n + block_r) <= _FAST_PATH_MAX_LANES
        ):
            tail = (
                n,
                out.stride(0),
                loc_word_stride,
                cache_k_nope.stride(0),
                cache_k_rope.stride(0),
            )
            shapes = dict(
                NOPE_D=nope_width, ROPE_D=rope_width, BLOCK_N=block_n, BLOCK_R=block_r
            )
            if n <= _ROWS_U1_MAX:
                _scatter_rows_u1[(n,)](
                    out, loc, cache_k_nope, cache_k_rope, *tail, num_warps=4, **shapes
                )
            elif n <= _ROWS_U4_MAX:
                _scatter_rows_u4[(triton.cdiv(n, 4),)](
                    out, loc, cache_k_nope, cache_k_rope, *tail, num_warps=4, **shapes
                )
            elif n <= _ROWS_U8_MAX:
                _scatter_rows_u8[(triton.cdiv(n, 8),)](
                    out, loc, cache_k_nope, cache_k_rope, *tail, num_warps=4, **shapes
                )
            else:
                _scatter_rows_persistent[(_PERSISTENT_GRID,)](
                    out,
                    loc,
                    cache_k_nope,
                    cache_k_rope,
                    triton.cdiv(n, 8),
                    *tail,
                    num_warps=2,
                    **shapes,
                )
            return out
        if packed_scatter:
            if (
                width % 4 == 0
                and kv_buffer.storage_offset() % 4 == 0
                and nope_width % 4 == 0
                and rope_width % 4 == 0
                and cache_k_nope.storage_offset() % 4 == 0
                and cache_k_rope.storage_offset() % 4 == 0
            ):
                # 8-byte packing halves the per-lane scatter traffic.
                word_dtype = tl.uint64
                pack = 4
                word_block_cap = 2048
            else:
                word_dtype = tl.uint32
                pack = 2
                word_block_cap = 4096
            nope_words = nope_width // pack
            rope_words = rope_width // pack
            words_per_row = nope_words + rope_words
            total_words = n * words_per_row
            word_block = min(
                triton.next_power_of_2(max(total_words, 1)), word_block_cap
            )
            _scatter_words[(triton.cdiv(total_words, word_block),)](
                triton.reinterpret(out, word_dtype),
                loc,
                triton.reinterpret(cache_k_nope, word_dtype),
                triton.reinterpret(cache_k_rope, word_dtype),
                total_words,
                nope_words,
                rope_words,
                out.stride(0) // pack,
                loc_word_stride,
                cache_k_nope.stride(0) // pack,
                cache_k_rope.stride(0) // pack,
                word_block,
                num_warps=4,
            )
        else:
            row_width = nope_width + rope_width
            total_elems = n * row_width
            col_block = min(triton.next_power_of_2(max(total_elems, 1)), 4096)
            _scatter_values[(triton.cdiv(total_elems, col_block),)](
                out,
                loc,
                cache_k_nope,
                cache_k_rope,
                total_elems,
                nope_width,
                rope_width,
                out.stride(0),
                out.stride(1),
                loc_word_stride,
                cache_k_nope.stride(0),
                cache_k_nope.stride(1),
                cache_k_rope.stride(0),
                cache_k_rope.stride(1),
                col_block,
                num_warps=4,
            )
    return out
