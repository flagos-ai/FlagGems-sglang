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
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language
# governing permissions and limitations under the License.

"""chunk_local_cumsum_vector — vector-mode local cumulative sum within chunks.

Splits the input `g` of shape [B, T, H, S] into chunks of `chunk_size` along the
time axis (T must be divisible by chunk_size), then does a cumulative sum along
the time-within-chunk axis for every (head, vector) column. Supports a reversed
cumsum and an optional scalar scale.

Semantics (matches the PyTorch reference exactly):

    B, T, H, S = g.shape
    BT = chunk_size;  NT = T // BT
    g_c = g.float().view(B, NT, BT, H, S)
    if reverse: g_c = g_c.flip(2)
    out = g_c.cumsum(dim=2)
    if scale is not None: out = out * scale
    if reverse: out = out.flip(2)
    return out.reshape(B, T, H, S)

The output is always float32 (the reference never casts back to the input
dtype). This kernel is written in portable Triton only and never calls any
pre-compiled / vendor-specific cached operator.

v6 design (measured on the GCU backend; four per-tile-size launch tiers):
  * TINY tiles (2 * BLOCK_BT * BLOCK_HS <= TINY_MAX_ELEMS, even chunk count):
    the common case (NCOL == 1, pow2 BT/HS, no scale / reverse / masks) is
    handled by the fused-3D kernel: NCH consecutive chunks are loaded as one
    contiguous [NCH*BT, HS] tile, reshaped to [NCH, BT, HS] and scanned per
    chunk with a single tl.cumsum(axis=1).  Scanning each chunk with its own
    rows is exactly the reference semantics (the cumsum resets at every chunk
    boundary), so -- unlike the fused-2 kernel it replaces -- there is no
    cross-chunk leak and no masked-reduce carry correction; the whole kernel
    is load + reshape + scan + store.  The larger fused tile amortizes the
    per-tile dispatch/scan setup that dominates tiny shapes: on
    b1-t1024-h8-s16-c16 the kernel-only time drops from ~22.4us (fused-2) to
    ~16.4us, at the measured pure-tile-copy floor (~16.1us).  NCH is the
    largest of {8, 4, 2} dividing nchunks with NCH*BT*HS <= TINY_3D_MAX_ELEMS
    so the tile stays off local memory (GCU spills above ~256KB fp32).  The
    masked / scale / reverse tiny cases keep the fused-2 kernel (single
    [2*BT, HS] scan + one masked-reduce carry correction).

v7: the tiny shapes are CPU/dispatch bound on the GCU backend -- the wrapper
  measures ~26us while the fused-3D kernel itself is only ~15us (the rest is
  the torch allocation, the Triton launch dispatch, and the wrapper Python).
  v7 therefore attacks the wrapper: the common tiny case is now selected by a
  pure-arithmetic fast path at the top of reference() (pow2 BT/HS checks,
  tile-size and even-chunk tests, inline NCH selection -- no pow2 helper
  calls, no NCOL/tile computation), and the output is allocated with
  torch.empty_like (a few tenths of a us cheaper than torch.empty on this
  backend).  Both changes are exact-preserving: the fast-path conditions
  reproduce the general path's dispatch for these shapes, so the fused-3D
  kernel is launched identically.  The GPU-bound medium/large tiers are
  untouched (they are already at the pure-copy bandwidth floor and their
  wrapper cost is hidden under the kernel).

  * MEDIUM tiles (BLOCK_BT * BLOCK_HS <= MEDIUM_MAX_ELEMS): the NCHUNK-loop
    kernel with NCHUNK=2, 1 warp and 3 software-pipeline stages on the chunk
    loop (overlaps the next tile load with the current scan/store; ~-4.5% on
    b2-t2048-h16-s32-c32).  A minimal-signature variant of this kernel was
    measured too: no gain (35.7us vs 36.0us), so it is not used.
  * LARGE tiles (else): the NCHUNK-loop kernel with 2 warps, no pipelining,
    and NCHUNK=2 for the widest tiles (BLOCK_BT*BLOCK_HS at MAX_TILE_ELEMS,
    e.g. b2-t8192-h64-s64-c128, where two sequential huge tiles per program
    beats four; ~-2% there) or NCHUNK=4 otherwise.
  * All tiers: consecutive chunks are contiguous in memory (c == b*NT + nt),
    so the per-chunk base offset is c * (BT*HS) — no div/mod by NT in the hot
    loop.  reverse mode is folded in by loading/storing rows in reversed
    order inside each chunk (cumsum over the reversed sequence equals the
    reference flip-cumsum-flip), and scale is applied after the scan.
  * Masked (non-power-of-two BT / H*S) shapes keep working: the tile is padded
    to pow2 and rows/columns are masked on load/store.

The kernel remains memory-bandwidth-bound on the large production shapes
(pure bf16->fp32 tile copy and the full cumsum run within ~1-2% of each
other), so scan restructuring there would buy nothing.
"""

import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# Tunables.
# ---------------------------------------------------------------------------
# Tiny tiles (2 * BLOCK_BT * BLOCK_HS at or below this) use the fused-2 kernel
# (two consecutive chunks as one [2*BT, HS] tile, single scan with a carry
# correction).  The cap stays well below the point where fusing spills to
# local memory on GCU (a ~16K-element fused tile is 4x slower).
TINY_MAX_ELEMS = 16384
# Tiny tiles whose launch matches the common case (NCOL == 1 and no
# scale / reverse / masks) use a minimal-signature fused-2 kernel; measured
# ~1.5us faster on b1-t1024-h8-s16-c16 than the full-signature variant.
NUM_WARPS_FUSED = 1
# Fused-3D tiny kernel: N consecutive chunks as one [N*BT, HS] tile scanned
# per-chunk (reshape + tl.cumsum(axis=1), no carry).  Cap on the fused tile
# (N * BT * HS) in elements -- above ~256KB fp32 the GCU backend spills to
# local memory / trips the L1 allocator.
TINY_3D_MAX_ELEMS = 24576
NUM_WARPS_3D = 1
# Tiles above TINY_MAX_ELEMS and at or below MEDIUM_MAX_ELEMS use the medium
# launch config (NCHUNK=2, 1 warp, pipelined chunk loop) instead of the
# wide-tile config.
MEDIUM_MAX_ELEMS = 32768
# Medium tiles: chunks per program, warps, and chunk-loop pipeline stages.
NCHUNK_MEDIUM = 2
NUM_WARPS_MEDIUM = 1
NUM_STAGES_MEDIUM = 3
# Large tiles: chunks per program and warps.
NCHUNK_DEFAULT = 4
NUM_WARPS = 2
# Widest tiles (BLOCK_BT*BLOCK_HS == MAX_TILE_ELEMS) prefer fewer sequential
# huge tiles per program (2 instead of 4) -- measured on b2-t8192-h64-s64-c128.
NCHUNK_WIDEST = 2
# Upper bound on tile columns (h*s); wider tiles don't help and raise
# register pressure.
BLOCK_HS_MAX = 1024
# Upper bound on tile elements (BLOCK_BT * BLOCK_HS); avoids local-memory
# spilling on large (chunk_size, h*s) combinations.
MAX_TILE_ELEMS = 131072


def _next_pow2(n):
    p = 1
    while p < n:
        p *= 2
    return p


def _largest_pow2_le(n):
    p = 1
    while p * 2 <= n:
        p *= 2
    return p


def _pick_nchunk(nchunks, cap):
    """Largest divisor of nchunks that is <= cap (no OOB, no guard)."""
    for nc in (cap, cap - 1, cap // 2, 2, 1):
        if nc >= 1 and nchunks % nc == 0:
            return nc
    return 1


def _pick_3d_n(nchunks, BT, HS):
    """Largest N in {8, 4, 2} for the fused-3D tiny kernel.

    N consecutive chunks are fused into one [N*BT, HS] tile that is scanned
    per-chunk via a reshape + tl.cumsum(axis=1) -- no cross-chunk leak, so no
    carry correction.  N is capped so the fused tile stays off local memory
    (measured on the GCU backend: 256KB+ tiles spill / hit the L1 allocator).
    Returns 1 when no N >= 2 divides nchunks within the cap (caller then keeps
    the fused-2 path).
    """
    for nc in (8, 4, 2):
        if nchunks % nc == 0 and nc * BT * HS <= TINY_3D_MAX_ELEMS:
            return nc
    return 1


@triton.jit
def _chunk_local_cumsum_vector_kernel(
    g_ptr,
    out_ptr,
    scale,
    BT: tl.constexpr,
    HS: tl.constexpr,
    NCOL: tl.constexpr,
    NCHUNK: tl.constexpr,
    BLOCK_BT: tl.constexpr,
    BLOCK_HS: tl.constexpr,
    HAS_SCALE: tl.constexpr,
    IS_REVERSE: tl.constexpr,
    HAS_ROW_MASK: tl.constexpr,
    HAS_COL_MASK: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    pid = tl.program_id(0)
    # pid -> (column-block, chunk-group): column blocks innermost so the
    # chunk-group decode stays cheap and consecutive programs touch adjacent
    # memory.
    cb = pid % NCOL
    rem = pid // NCOL
    chunk0 = rem * NCHUNK

    rows = tl.arange(0, BLOCK_BT)
    cols = tl.arange(0, BLOCK_HS)

    # Within-chunk time index of each tile row; reversed when IS_REVERSE
    # (cumsum over reversed rows == reference flip-cumsum-flip).  For masked
    # rows the offset is negative but never dereferenced (load/store masked).
    if IS_REVERSE:
        bt = BT - 1 - rows
    else:
        bt = rows

    # Chunks are contiguous in memory (c == b*NT + nt), so the base offset of
    # chunk c is simply c * (BT*HS) -- no div/mod by NT needed in the loop;
    # consecutive chunk tiles are BT*HS elements apart.
    off_cols = cb * BLOCK_HS + tl.max_contiguous(
        tl.multiple_of(cols, BLOCK_HS), BLOCK_HS
    )
    row_off = bt[:, None] * HS + off_cols[None, :]
    base = (chunk0 * BT) * HS

    if HAS_ROW_MASK and HAS_COL_MASK:
        ncols = HS - cb * BLOCK_HS
        m = (rows[:, None] < BT) & (cols[None, :] < ncols)
        for k in tl.range(0, NCHUNK, num_stages=NUM_STAGES):
            off = base + row_off
            x = tl.load(g_ptr + off, mask=m, other=0.0).to(tl.float32)
            y = tl.cumsum(x, axis=0)
            if HAS_SCALE:
                y = y * scale
            tl.store(out_ptr + off, y, mask=m)
            base += BT * HS
    elif HAS_ROW_MASK:
        m = rows[:, None] < BT
        for k in tl.range(0, NCHUNK, num_stages=NUM_STAGES):
            off = base + row_off
            x = tl.load(g_ptr + off, mask=m, other=0.0).to(tl.float32)
            y = tl.cumsum(x, axis=0)
            if HAS_SCALE:
                y = y * scale
            tl.store(out_ptr + off, y, mask=m)
            base += BT * HS
    elif HAS_COL_MASK:
        ncols = HS - cb * BLOCK_HS
        m = cols[None, :] < ncols
        for k in tl.range(0, NCHUNK, num_stages=NUM_STAGES):
            off = base + row_off
            x = tl.load(g_ptr + off, mask=m, other=0.0).to(tl.float32)
            y = tl.cumsum(x, axis=0)
            if HAS_SCALE:
                y = y * scale
            tl.store(out_ptr + off, y, mask=m)
            base += BT * HS
    else:
        for k in tl.range(0, NCHUNK, num_stages=NUM_STAGES):
            off = base + row_off
            x = tl.load(g_ptr + off).to(tl.float32)
            y = tl.cumsum(x, axis=0)
            if HAS_SCALE:
                y = y * scale
            tl.store(out_ptr + off, y)
            base += BT * HS


@triton.jit
def _chunk_local_cumsum_fused2_kernel(
    g_ptr,
    out_ptr,
    scale,
    BT: tl.constexpr,
    HS: tl.constexpr,
    NCOL: tl.constexpr,
    BLOCK_BT: tl.constexpr,
    BLOCK_HS: tl.constexpr,
    HAS_SCALE: tl.constexpr,
    IS_REVERSE: tl.constexpr,
    HAS_ROW_MASK: tl.constexpr,
    HAS_COL_MASK: tl.constexpr,
):
    # Tiny-tile path: two consecutive chunks (rows 0..BT-1, rows BT..2*BT-1)
    # are loaded as ONE contiguous [2*BT, HS] tile (they are adjacent in
    # memory).  A single cumsum over the fused tile is exact per column
    # except that the second chunk inherits the first chunk's prefix; the
    # leak is removed by subtracting s[BT-1] (the first chunk's total) from
    # the second chunk's rows.
    #   Grid: axis 0 = chunk pair.  When NCOL > 1 there is a second axis of
    # column blocks (axis 1 = column block, no div/mod decode); when NCOL ==
    # 1 that axis is dropped and cb folds to 0 at trace time (a 1D grid
    # launches ~0.5us faster per program on the GCU backend than a 2D grid
    # whose second axis has extent 1).
    c0 = tl.program_id(0) * 2
    if NCOL == 1:
        cb = 0
    else:
        cb = tl.program_id(1)

    rows = tl.arange(0, BLOCK_BT)
    cols = tl.max_contiguous(
        tl.multiple_of(tl.arange(0, BLOCK_HS), BLOCK_HS), BLOCK_HS
    )

    if IS_REVERSE:
        # reversed within each chunk; chunk order preserved
        r = (rows // BT) * BT + (BT - 1 - (rows % BT))
    else:
        r = rows

    # Column-block offset is folded into the scalar base so the column
    # address stays a pure annotated arange (keeps vectorized loads on the
    # backend that annotate tt.make_range).
    base = (c0 * BT) * HS + cb * BLOCK_HS
    off = base + r[:, None] * HS + cols[None, :]

    if HAS_ROW_MASK or HAS_COL_MASK:
        m = rows[:, None] < 2 * BT
        if HAS_COL_MASK:
            m = m & (cols[None, :] < HS - cb * BLOCK_HS)
        x = tl.load(g_ptr + off, mask=m, other=0.0).to(tl.float32)
        s = tl.cumsum(x, axis=0)
        c1 = tl.sum(tl.where(rows[:, None] == BT - 1, s, 0.0), axis=0)
        y = tl.where(
            (rows[:, None] >= BT) & (rows[:, None] < 2 * BT),
            s - c1[None, :],
            s,
        )
        if HAS_SCALE:
            y = y * scale
        tl.store(out_ptr + off, y, mask=m)
    else:
        x = tl.load(g_ptr + off).to(tl.float32)
        s = tl.cumsum(x, axis=0)
        c1 = tl.sum(tl.where(rows[:, None] == BT - 1, s, 0.0), axis=0)
        y = tl.where(
            (rows[:, None] >= BT) & (rows[:, None] < 2 * BT),
            s - c1[None, :],
            s,
        )
        if HAS_SCALE:
            y = y * scale
        tl.store(out_ptr + off, y)


@triton.jit
def _chunk_local_cumsum_fused2_fast_kernel(
    g_ptr,
    out_ptr,
    BT: tl.constexpr,
    HS: tl.constexpr,
    BLOCK_BT: tl.constexpr,
    BLOCK_HS: tl.constexpr,
):
    # Minimal-signature fast path of the fused-2 kernel, used only when the
    # launch matches the common case: NCOL == 1 (single column block covering
    # all of HS) and no scale / reverse / row / column masks.  With the
    # cb decode, the flag plumbing and the scale runtime argument all gone,
    # the dispatch is ~1.5us cheaper on tiny shapes (measured on
    # b1-t1024-h8-s16-c16: 23.7us vs 25.1us wrapper-level in the same
    # process).  The body is byte-for-byte what the full kernel traces to
    # when every flag is False.
    c0 = tl.program_id(0) * 2
    rows = tl.arange(0, BLOCK_BT)
    cols = tl.max_contiguous(
        tl.multiple_of(tl.arange(0, BLOCK_HS), BLOCK_HS), BLOCK_HS
    )
    off = (c0 * BT) * HS + rows[:, None] * HS + cols[None, :]
    x = tl.load(g_ptr + off).to(tl.float32)
    s = tl.cumsum(x, axis=0)
    c1 = tl.sum(tl.where(rows[:, None] == BT - 1, s, 0.0), axis=0)
    y = tl.where(
        (rows[:, None] >= BT) & (rows[:, None] < 2 * BT), s - c1[None, :], s
    )
    tl.store(out_ptr + off, y)


@triton.jit
def _chunk_local_cumsum_3d_fast_kernel(
    g_ptr,
    out_ptr,
    BT: tl.constexpr,
    HS: tl.constexpr,
    NCH: tl.constexpr,
):
    # v6 fast path for the common tiny-tile case (NCOL == 1, pow2 BT/HS, no
    # scale / reverse / masks): NCH consecutive chunks are loaded as one
    # contiguous [NCH*BT, HS] tile and the scan runs per-chunk through a
    # reshape to [NCH, BT, HS] + tl.cumsum(axis=1).  Scanning chunk k with
    # rows [k*BT, (k+1)*BT) of the tile is exactly the reference semantics
    # (the cumsum resets at each chunk boundary), so unlike the fused-2
    # kernel there is no cross-chunk leak and no carry correction at all --
    # the whole computation is one load + one reshape + one scan + one store.
    # On b1-t1024-h8-s16-c16 this runs at the pure-tile-copy floor (~16.4us
    # vs ~19.6us for the fused-2 kernel's 32-row tile; the larger fused tile
    # amortizes the per-tile dispatch/scan setup that dominates tiny shapes).
    # NCH is picked by _pick_3d_n so the fused tile stays off local memory.
    c0 = tl.program_id(0) * NCH
    rows = tl.arange(0, NCH * BT)
    cols = tl.max_contiguous(tl.multiple_of(tl.arange(0, HS), HS), HS)
    off = (c0 * BT) * HS + rows[:, None] * HS + cols[None, :]
    x = tl.load(g_ptr + off).to(tl.float32)
    x3 = tl.reshape(x, (NCH, BT, HS))
    s = tl.cumsum(x3, axis=1)
    tl.store(out_ptr + off, tl.reshape(s, (NCH * BT, HS)))


def chunk_local_cumsum_vector(g, chunk_size, reverse=False, scale=None):
    """Vector-mode local cumulative sum within chunks (Triton implementation).

    Args:
        g:           [B, T, H, S] input (float32 / bfloat16 / float16).
        chunk_size:  within-chunk time width; T must be a multiple of it.
        reverse:     if True, cumsum runs from the end of each chunk.
        scale:       optional scalar multiplier applied after the cumsum.

    Returns:
        out: [B, T, H, S] float32.
    """
    B, T, H, S = g.shape
    BT = chunk_size
    HS = H * S
    nchunks = B * (T // BT)
    # v7: allocate via empty_like (output is always the input shape in fp32) --
    # ~1us cheaper than torch.empty(shape...) on the GCU backend's patched
    # allocator, and it matters because the tiny shapes are CPU/dispatch bound.
    out = torch.empty_like(g, dtype=torch.float32)

    # v7 fast path: the common tiny case (NCOL == 1, pow2 BT/HS, no scale /
    # reverse / row or column masks) is recognized with pure arithmetic, so the
    # CPU-bound tiny shapes pay the least possible Python per call (no pow2
    # helper calls, no NCOL/tile computation).  On the GCU backend these shapes
    # measure ~26us wrapper vs ~15us GPU kernel, so every Python op counts.
    # The checks are exactly equivalent to the general path's decision here:
    #   * (BT & (BT - 1)) == 0            -> BLOCK_BT == BT    (no row mask)
    #   * HS pow2 and HS <= BLOCK_HS_MAX  -> BLOCK_HS == HS    (no col mask,
    #                                                           NCOL == 1)
    #   * 2*BT*HS <= TINY_MAX_ELEMS and nchunks even -> tiny fused tier
    # NCH is then the largest of {8, 4, 2} dividing nchunks within the fused-3D
    # tile cap (same selection as _pick_3d_n); nchunks even + tiny tier always
    # yields NCH >= 2, so the fused-3D kernel always applies here.
    if scale is None and not reverse:
        if (
            (HS & (HS - 1)) == 0
            and HS <= BLOCK_HS_MAX
            and (BT & (BT - 1)) == 0
            and 2 * BT * HS <= TINY_MAX_ELEMS
            and (nchunks & 1) == 0
        ):
            if nchunks % 8 == 0 and 8 * BT * HS <= TINY_3D_MAX_ELEMS:
                nch3 = 8
            elif nchunks % 4 == 0 and 4 * BT * HS <= TINY_3D_MAX_ELEMS:
                nch3 = 4
            else:
                nch3 = 2
            grid = (nchunks // nch3,)
            _chunk_local_cumsum_3d_fast_kernel[grid](
                g,
                out,
                BT=BT,
                HS=HS,
                NCH=nch3,
                num_warps=NUM_WARPS_3D,
            )
            return out

    BLOCK_BT = _next_pow2(BT)
    BLOCK_HS = min(_largest_pow2_le(HS), BLOCK_HS_MAX)
    # Keep the tile from spilling into local memory.
    while BLOCK_BT * BLOCK_HS > MAX_TILE_ELEMS and BLOCK_HS > 16:
        BLOCK_HS //= 2
    NCOL = (HS + BLOCK_HS - 1) // BLOCK_HS
    tile_elems = BLOCK_BT * BLOCK_HS

    scale_f = float(scale) if scale is not None else 1.0
    has_scale = scale is not None
    is_reverse = bool(reverse)

    if 2 * tile_elems <= TINY_MAX_ELEMS and nchunks % 2 == 0:
        # Tiny tiles.  The common case (NCOL == 1, pow2 BT/HS, no scale /
        # reverse / masks) takes the fused-3D kernel: N consecutive chunks
        # fused into one [N*BT, HS] tile scanned per-chunk (no carry), which
        # runs at the pure-copy floor.  Everything else falls back to the
        # fused-2 kernel (single [2*BT, HS] scan + carry correction).
        has_row_mask = BT != BLOCK_BT
        has_col_mask = HS % BLOCK_HS != 0
        if (
            NCOL == 1
            and not has_scale
            and not is_reverse
            and not has_row_mask
            and not has_col_mask
        ):
            NCH3 = _pick_3d_n(nchunks, BT, HS)
            if NCH3 >= 2:
                grid = (nchunks // NCH3,)
                _chunk_local_cumsum_3d_fast_kernel[grid](
                    g,
                    out,
                    BT=BT,
                    HS=HS,
                    NCH=NCH3,
                    num_warps=NUM_WARPS_3D,
                )
                return out
            # fall through to fused-2 when no N >= 2 fits
        FBT = 2 * BLOCK_BT
        if (
            NCOL == 1
            and not has_scale
            and not is_reverse
            and not has_row_mask
            and not has_col_mask
        ):
            grid = (nchunks // 2,)
            _chunk_local_cumsum_fused2_fast_kernel[grid](
                g,
                out,
                BT=BT,
                HS=HS,
                BLOCK_BT=FBT,
                BLOCK_HS=BLOCK_HS,
                num_warps=NUM_WARPS_FUSED,
            )
        else:
            grid = ((nchunks // 2),) if NCOL == 1 else ((nchunks // 2), NCOL)
            _chunk_local_cumsum_fused2_kernel[grid](
                g,
                out,
                scale_f,
                BT=BT,
                HS=HS,
                NCOL=NCOL,
                BLOCK_BT=FBT,
                BLOCK_HS=BLOCK_HS,
                HAS_SCALE=has_scale,
                IS_REVERSE=is_reverse,
                HAS_ROW_MASK=has_row_mask,
                HAS_COL_MASK=has_col_mask,
                num_warps=NUM_WARPS_FUSED,
            )
    else:
        # Medium / large tiles: NCHUNK-loop kernel.
        if tile_elems <= MEDIUM_MAX_ELEMS:
            nc, num_warps, num_stages = (
                NCHUNK_MEDIUM,
                NUM_WARPS_MEDIUM,
                NUM_STAGES_MEDIUM,
            )
        elif tile_elems >= MAX_TILE_ELEMS:
            nc, num_warps, num_stages = NCHUNK_WIDEST, NUM_WARPS, 1
        else:
            nc, num_warps, num_stages = NCHUNK_DEFAULT, NUM_WARPS, 1
        NCHUNK = _pick_nchunk(nchunks, nc)
        grid = ((nchunks + NCHUNK - 1) // NCHUNK * NCOL,)
        _chunk_local_cumsum_vector_kernel[grid](
            g,
            out,
            scale_f,
            BT=BT,
            HS=HS,
            NCOL=NCOL,
            NCHUNK=NCHUNK,
            BLOCK_BT=BLOCK_BT,
            BLOCK_HS=BLOCK_HS,
            HAS_SCALE=has_scale,
            IS_REVERSE=is_reverse,
            HAS_ROW_MASK=(BT != BLOCK_BT),
            HAS_COL_MASK=(HS % BLOCK_HS != 0),
            NUM_STAGES=num_stages,
            num_warps=num_warps,
        )
    return out


__all__ = ["chunk_local_cumsum_vector"]
