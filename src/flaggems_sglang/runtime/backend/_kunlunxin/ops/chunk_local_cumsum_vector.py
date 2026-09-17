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

"""chunk_local_cumsum_vector -- per-chunk cumulative sum of a vector
per-token, per-head gate.

FLA-family building block: turns raw log-decay gate values ``g`` into
within-chunk cumulative decays for gated linear attention.

Semantics (matches the PyTorch reference exactly):

    B, T, H, S = g.shape
    g_c = g.float().view(B, T // chunk_size, chunk_size, H, S)
    if reverse: g_c = g_c.flip(2)
    out = g_c.cumsum(dim=2)
    if scale is not None: out = out * scale
    if reverse: out = out.flip(2)
    return out.reshape(B, T, H, S)

Scope:
    - head_first=False (g is [B, T, H, S], contiguous)
    - fixed-size batching (no cu_seqlens)
    - T is always an exact multiple of chunk_size (chunk_size is a power of 2)
    - output dtype is always float32 (accumulation done in float32)

Kunlun P800 XPU backend adaptation:
    On this backend 2D load/store / tl.cumsum(axis=0) / reshape / trans return
    wrong data or crash, and every memory instruction issued by a program is
    executed serially with a fixed ~25 ns cost, so total wall time is
    essentially ``#memory_instructions * ~25 ns``: every wide load or store
    costs the same ~25 ns no matter how many of its lanes are in bounds, and
    instructions from different programs do not overlap.  The two levers are
    therefore (a) make every instruction as wide as the backend reliably
    compiles and (b) keep every lane of every instruction in bounds.

    v1-v5 scanned one (chunk, 2048-lane hs-block) per program, so shapes with
    H*S < 2048 paid the full per-instruction cost for a mostly-masked,
    mostly-wasted tile (e.g. H*S=64 -> 64 of 2048 lanes used, 32x waste).

    v6 refinement: the unit of parallel work is a (chunk, hs-segment) whose
    lane width W = min(H*S, BLOCK).  When H*S < BLOCK, G = BLOCK // W
    consecutive chunks are packed into one program (each chunk's within-chunk
    scan is independent, so all G scans advance in lockstep in the same BLOCK
    lanes), turning every masked-out lane into a real one.

    v7 refinement: the per-instruction cost is ~25 ns for 2048-wide
    instructions and grows only slightly (to ~29-30 ns) for 8192-wide ones,
    so *total time is dominated by the instruction count*, and the instruction
    count is 2*N / BLOCK for a single-pass scan.  Widening the fast path from
    BLOCK=2048 to BLOCK=8192 (4x fewer load/store instructions) is therefore
    a ~3.4x speedup on the benchmark's large shapes, which moves the kernel
    from ~0.33x of the PyTorch reference to ~1.05-1.16x on those shapes.

    v8 refinement: the per-instruction cost model breaks down at the very
    small program counts that small shapes produce with BLOCK=8192.  When the
    widest valid tile leaves P <= 4 programs, the per-program overhead (the
    scan is a serial chain, and with so few programs nothing overlaps) makes
    8192 *slower* than 4096, and 4096 with P in [4, 8] is the measured optimum
    (e.g. (1, 8192, 8, 32, 64): 41.9 us at 4096 vs 44.6 us at 8192).  For
    P >= 8 the widest tile still wins (the large benchmark shapes are within
    ~7% of a pure copy's memory floor, so there is no headroom there).  The
    host therefore downgrades 8192 -> 4096 exactly when the 8192 config would
    leave P <= 4 programs and 4096 is also mask-free-valid.

    v9 refinement: adding a 16384-lane tile to the top of the candidate list
    is a *conditional* win.  At 16384 wide the per-instruction cost roughly
    doubles (~29 ns -> ~57 ns on the (4, 16384, 128, 64, 128) shape), so it
    only pays when the halved instruction count (2*N / 16384 vs 2*N / 8192)
    outweighs that: measured A/B on every benchmark shape shows 16384 wins by
    ~3.4% only when it still leaves P >= 256 programs (the b4-t16384 profile,
    HS=8192 -> 512 chunks packed 2-per-program -> 256 programs), and loses on
    every shape with P < 256 (P=128: 1970 vs 1893 us; P=32: 538 vs 500 us;
    P=4 on the small shapes: far worse).  The host therefore accepts a
    16384 tile only when P >= 256 and otherwise falls back to the v8 logic
    unchanged.

    Two constraints discovered while widening:

    1. The XPU backend's ``unroll_control`` compiler pass fails (spurious
       "out of resource: uni_sram") on scan kernels whose tile is >= 4096
       lanes, even though the same kernel compiles fine at 2048 and a plain
       8192-wide memcpy compiles fine.  The pass can be skipped per-launch by
       passing the standard backend option ``isCloseUnrollControl=True``
       (a documented XPUOptions field; unknown options are ignored on other
       backends, so the kernel stays portable).  With the flag set, BLOCK
       4096/8192/16384 scans compile, run, and hit the same ~25-60 ns/
       instruction model as the 2048 baseline.  Measured A/B: the flag has
       zero effect on the 2048-wide path (bit-identical times), so it is only
       passed on the wide (>= 4096) launches.

    2. A given BLOCK is only usable when G = BLOCK // W divides the total
       number of scan units U (the kernel is deliberately mask-free: a masked
       store with a grouped address vector returns wrong data on this
       backend).  The host therefore picks the widest candidate BLOCK from
       (16384, 8192, 4096, 2048) that satisfies the divisibility; shapes that
       match none (never hit by the benchmark, but some test shapes) keep the
       v5 masked kernel at BLOCK=2048.

    Addressing form is load-bearing on this backend: a per-lane load/store
    address that is *already a vector* when the loop-varying scalar is added
    (``g + base0 + t*HS``, base0 a vector) makes the backend scalarize every
    memory instruction (~125x slower).  The kernel therefore keeps every
    loop-varying term in a scalar ``scalar_base`` and folds the entire
    per-lane grouping into one loop-invariant vector ``vec``
    (``(offs//W)*(BT*HS) + offs%W`` for the packed case, plain ``offs``
    otherwise) that is added *last*: ``g + scalar_base + t*HS + vec``
    compiles back to fast wide vector loads/stores.
"""

import torch
import triton
import triton.language as tl

# Largest single load/store vector width that the scan kernel compiles to at
# model speed on this backend.  The scan's accumulator and address vectors cap
# the *kernel* at 2048 lanes by default; skipping the broken unroll_control
# pass (see module docstring) lets BLOCK 4096/8192/16384 compile, at which
# point the instruction count -- the true cost driver -- drops 2-8x.  16384 is
# only a net win at P >= 256 programs (v9 rule in reference()).
_BLOCK_HS_CANDIDATES = (16384, 8192, 4096, 2048)

# Per-launch backend option that makes >=4096-lane scan tiles compile.  It is
# an XPUOptions field in triton 3.0.0's XPU backend and is silently ignored by
# backends that do not define it, so passing it keeps the kernel portable.
_XPU_WIDE_BLOCK_OPTS = {"isCloseUnrollControl": True}


@triton.jit
def _chunk_local_cumsum_vector_kernel(
    g_ptr,
    out_ptr,
    scale,
    HS: tl.constexpr,  # H * S, contiguous inner extent per (b, t)
    BT: tl.constexpr,  # within-chunk time width
    W: tl.constexpr,  # lane width of one unit: min(HS, BLOCK)
    G: tl.constexpr,  # units per program: BLOCK // W  (>= 1)
    NH: tl.constexpr,  # hs-segments per chunk: HS // W     (>= 1)
    REVERSE: tl.constexpr,
    HAS_SCALE: tl.constexpr,
):
    """Per-chunk cumulative sum over the within-chunk time dim (BT).

    Grid: (U / G,), U = B * NT * NH.  Every lane of every load/store is in
    bounds by construction (the host only dispatches to this kernel when
    W divides BLOCK and G divides U, i.e. no masking is ever required).
    A "unit" is one independent scan: a (chunk, hs-segment) pair, i.e. a set
    of W lanes that share a within-chunk time axis.  Each program owns G
    consecutive units and scans all of them in lockstep: at every within-chunk
    step it issues one wide load/store that touches all G units' slices.

    NOTE: this kernel is deliberately mask-free.  On this backend a *masked*
    store with a non-contiguous (grouped) address vector returns wrong data
    (lane values are shifted), while mask-free grouped stores/loads and masked
    contiguous ones are all correct.  Shapes that would need a mask (U % G or
    HS % W nonzero) are routed by the host to the v5 masked kernel instead.
    """
    pid = tl.program_id(0)

    BLOCK: tl.constexpr = W * G
    offs = tl.arange(0, BLOCK)

    if G == 1:
        # Single unit per program: the chunk/segment base is scalar and the
        # per-lane part is the plain contiguous arange.
        if NH == 1:
            c = pid
            seg = 0
        else:
            c = pid // NH
            seg = pid % NH
        scalar_base = c * (BT * HS) + seg * W
        vec = offs
    else:
        # G > 1 (implies NH == 1, i.e. H*S < BLOCK): pack G consecutive
        # chunks into one program.  The chunk index pid*G + offs//W is
        # per-lane, so the whole chunk/slice base stays a loop-invariant
        # vector; only scalar terms (t*HS) are ever added to it inside the
        # scan loop (adding the loop scalar to an already-vector address makes
        # this backend scalarize every load/store, see module docstring).
        vec = (offs // W) * (BT * HS) + (offs % W)
        scalar_base = pid * G * (BT * HS)

    # g is [B, T, H, S] contiguous: element (b, t, h, s) is at
    # (b*T+t)*(H*S) + h*S + s.  In chunk layout c = b*NT + n covers
    # t in [c*BT, (c+1)*BT).  scalar_base + vec = this unit's (chunk,
    # hs-segment) position at within-chunk time t=0, lane j; per-step offsets
    # (t*HS) are scalars added to scalar_base only.

    acc = tl.zeros((BLOCK,), dtype=tl.float32)

    if BT % 4 == 0:
        # Unroll the sequential scan by 4 and issue the group's four loads
        # before the accumulate/store pairs.  The accumulator is a loop-carried
        # dependency, but the load for step t+k is independent of the loads of
        # steps t..t+k-1, so issuing the whole group up front overlaps the
        # group's memory latency with the ALU/store work of the earlier steps.
        if REVERSE:
            # Within-chunk times descend (suffix cumsum), then flipped back:
            # step t of the unrolled loop handles times BT-1-t ... BT-4-t.
            for t in range(0, BT, 4):
                o0 = (BT - 1 - t) * HS
                o1 = o0 - HS
                o2 = o1 - HS
                o3 = o2 - HS
                x0 = tl.load(g_ptr + scalar_base + o0 + vec).to(tl.float32)
                x1 = tl.load(g_ptr + scalar_base + o1 + vec).to(tl.float32)
                x2 = tl.load(g_ptr + scalar_base + o2 + vec).to(tl.float32)
                x3 = tl.load(g_ptr + scalar_base + o3 + vec).to(tl.float32)
                acc += x0
                y0 = acc * scale if HAS_SCALE else acc
                tl.store(out_ptr + scalar_base + o0 + vec, y0)
                acc += x1
                y1 = acc * scale if HAS_SCALE else acc
                tl.store(out_ptr + scalar_base + o1 + vec, y1)
                acc += x2
                y2 = acc * scale if HAS_SCALE else acc
                tl.store(out_ptr + scalar_base + o2 + vec, y2)
                acc += x3
                y3 = acc * scale if HAS_SCALE else acc
                tl.store(out_ptr + scalar_base + o3 + vec, y3)
        else:
            for t in range(0, BT, 4):
                o0 = t * HS
                o1 = o0 + HS
                o2 = o1 + HS
                o3 = o2 + HS
                x0 = tl.load(g_ptr + scalar_base + o0 + vec).to(tl.float32)
                x1 = tl.load(g_ptr + scalar_base + o1 + vec).to(tl.float32)
                x2 = tl.load(g_ptr + scalar_base + o2 + vec).to(tl.float32)
                x3 = tl.load(g_ptr + scalar_base + o3 + vec).to(tl.float32)
                acc += x0
                y0 = acc * scale if HAS_SCALE else acc
                tl.store(out_ptr + scalar_base + o0 + vec, y0)
                acc += x1
                y1 = acc * scale if HAS_SCALE else acc
                tl.store(out_ptr + scalar_base + o1 + vec, y1)
                acc += x2
                y2 = acc * scale if HAS_SCALE else acc
                tl.store(out_ptr + scalar_base + o2 + vec, y2)
                acc += x3
                y3 = acc * scale if HAS_SCALE else acc
                tl.store(out_ptr + scalar_base + o3 + vec, y3)
    else:
        # BT not a multiple of 4: never hit in the workload (chunk_size is a
        # power of 2), kept as a safety fallback.  Same serial scan as v4.
        for t in range(0, BT):
            tt = BT - 1 - t if REVERSE else t
            x = tl.load(g_ptr + scalar_base + tt * HS + vec).to(tl.float32)
            acc += x
            y = acc * scale if HAS_SCALE else acc
            tl.store(out_ptr + scalar_base + tt * HS + vec, y)


@triton.jit
def _chunk_local_cumsum_vector_masked_kernel(
    g_ptr,
    out_ptr,
    scale,
    HS: tl.constexpr,  # H * S, contiguous inner extent per (b, t)
    BT: tl.constexpr,  # within-chunk time width
    BLOCK_HS: tl.constexpr,  # contiguous load/store width (spans heads x vector dim)
    HAS_MASK: tl.constexpr,  # True iff HS is not an exact multiple of BLOCK_HS
    REVERSE: tl.constexpr,
    HAS_SCALE: tl.constexpr,
):
    """General fallback: v5 layout, one (chunk, BLOCK_HS hs-block) per program.

    Used only when H*S has no power-of-two relationship with BLOCK_HS (never
    hit by the benchmark shapes).  Per-lane tail masking keeps arbitrary H, S
    correct.
    """
    pid = tl.program_id(0)

    n_hs_blocks = (HS + BLOCK_HS - 1) // BLOCK_HS

    pid_bn = pid // n_hs_blocks  # linearized (B*NT) index = (b, n)
    pid_hs = pid % n_hs_blocks  # contiguous hs-block index

    offs = pid_hs * BLOCK_HS + tl.arange(0, BLOCK_HS)  # [BLOCK_HS]

    base = pid_bn * (BT * HS)

    acc = tl.zeros((BLOCK_HS,), dtype=tl.float32)

    if HAS_MASK:
        m = offs < HS

    if BT % 4 == 0:
        if REVERSE:
            for t in range(0, BT, 4):
                o0 = (BT - 1 - t) * HS
                o1 = o0 - HS
                o2 = o1 - HS
                o3 = o2 - HS
                if HAS_MASK:
                    x0 = tl.load(
                        g_ptr + base + o0 + offs, mask=m, other=0.0
                    ).to(tl.float32)
                    x1 = tl.load(
                        g_ptr + base + o1 + offs, mask=m, other=0.0
                    ).to(tl.float32)
                    x2 = tl.load(
                        g_ptr + base + o2 + offs, mask=m, other=0.0
                    ).to(tl.float32)
                    x3 = tl.load(
                        g_ptr + base + o3 + offs, mask=m, other=0.0
                    ).to(tl.float32)
                else:
                    x0 = tl.load(g_ptr + base + o0 + offs).to(tl.float32)
                    x1 = tl.load(g_ptr + base + o1 + offs).to(tl.float32)
                    x2 = tl.load(g_ptr + base + o2 + offs).to(tl.float32)
                    x3 = tl.load(g_ptr + base + o3 + offs).to(tl.float32)
                acc += x0
                y0 = acc * scale if HAS_SCALE else acc
                if HAS_MASK:
                    tl.store(out_ptr + base + o0 + offs, y0, mask=m)
                else:
                    tl.store(out_ptr + base + o0 + offs, y0)
                acc += x1
                y1 = acc * scale if HAS_SCALE else acc
                if HAS_MASK:
                    tl.store(out_ptr + base + o1 + offs, y1, mask=m)
                else:
                    tl.store(out_ptr + base + o1 + offs, y1)
                acc += x2
                y2 = acc * scale if HAS_SCALE else acc
                if HAS_MASK:
                    tl.store(out_ptr + base + o2 + offs, y2, mask=m)
                else:
                    tl.store(out_ptr + base + o2 + offs, y2)
                acc += x3
                y3 = acc * scale if HAS_SCALE else acc
                if HAS_MASK:
                    tl.store(out_ptr + base + o3 + offs, y3, mask=m)
                else:
                    tl.store(out_ptr + base + o3 + offs, y3)
        else:
            for t in range(0, BT, 4):
                o0 = t * HS
                o1 = o0 + HS
                o2 = o1 + HS
                o3 = o2 + HS
                if HAS_MASK:
                    x0 = tl.load(
                        g_ptr + base + o0 + offs, mask=m, other=0.0
                    ).to(tl.float32)
                    x1 = tl.load(
                        g_ptr + base + o1 + offs, mask=m, other=0.0
                    ).to(tl.float32)
                    x2 = tl.load(
                        g_ptr + base + o2 + offs, mask=m, other=0.0
                    ).to(tl.float32)
                    x3 = tl.load(
                        g_ptr + base + o3 + offs, mask=m, other=0.0
                    ).to(tl.float32)
                else:
                    x0 = tl.load(g_ptr + base + o0 + offs).to(tl.float32)
                    x1 = tl.load(g_ptr + base + o1 + offs).to(tl.float32)
                    x2 = tl.load(g_ptr + base + o2 + offs).to(tl.float32)
                    x3 = tl.load(g_ptr + base + o3 + offs).to(tl.float32)
                acc += x0
                y0 = acc * scale if HAS_SCALE else acc
                if HAS_MASK:
                    tl.store(out_ptr + base + o0 + offs, y0, mask=m)
                else:
                    tl.store(out_ptr + base + o0 + offs, y0)
                acc += x1
                y1 = acc * scale if HAS_SCALE else acc
                if HAS_MASK:
                    tl.store(out_ptr + base + o1 + offs, y1, mask=m)
                else:
                    tl.store(out_ptr + base + o1 + offs, y1)
                acc += x2
                y2 = acc * scale if HAS_SCALE else acc
                if HAS_MASK:
                    tl.store(out_ptr + base + o2 + offs, y2, mask=m)
                else:
                    tl.store(out_ptr + base + o2 + offs, y2)
                acc += x3
                y3 = acc * scale if HAS_SCALE else acc
                if HAS_MASK:
                    tl.store(out_ptr + base + o3 + offs, y3, mask=m)
                else:
                    tl.store(out_ptr + base + o3 + offs, y3)
    else:
        for t in range(0, BT):
            tt = BT - 1 - t if REVERSE else t

            if HAS_MASK:
                x = tl.load(
                    g_ptr + base + tt * HS + offs,
                    mask=m,
                    other=0.0,
                ).to(tl.float32)
            else:
                x = tl.load(g_ptr + base + tt * HS + offs).to(tl.float32)

            acc += x
            y = acc * scale if HAS_SCALE else acc

            if HAS_MASK:
                tl.store(out_ptr + base + tt * HS + offs, y, mask=m)
            else:
                tl.store(out_ptr + base + tt * HS + offs, y)


def _launch_masked(g, out, scale, HS, BT, reverse, has_scale):
    """v5 masked-kernel fallback (BLOCK_HS=2048).  Kept unchanged from v6."""
    n_hs_blocks = (HS + _BLOCK_HS_CANDIDATES[-1] - 1) // _BLOCK_HS_CANDIDATES[
        -1
    ]
    grid = (g.shape[0] * (g.shape[1] // BT) * n_hs_blocks,)
    _chunk_local_cumsum_vector_masked_kernel[grid](
        g,
        out,
        1.0 if scale is None else scale,
        HS=HS,
        BT=BT,
        BLOCK_HS=_BLOCK_HS_CANDIDATES[-1],
        HAS_MASK=(HS % _BLOCK_HS_CANDIDATES[-1]) != 0,
        REVERSE=reverse,
        HAS_SCALE=has_scale,
        num_warps=1,  # ignored on Kunlun (warp_size = 1)
    )


def chunk_local_cumsum_vector(g, chunk_size, reverse=False, scale=None):
    """Per-chunk (within-chunk) cumulative sum of a vector per-token, per-head gate.

    Args:
        g: [B, T, H, S] input gate tensor (float32 / bfloat16 / float16).
        chunk_size: within-chunk width (power of 2), T must be a multiple of it.
        reverse: if True, cumsum runs from the chunk end toward its start.
        scale: optional scalar multiplier applied to the output (or None).

    Returns:
        [B, T, H, S] float32 tensor.
    """
    B, T, H, S = g.shape
    BT = chunk_size
    NT = T // BT
    HS = H * S

    out = torch.empty_like(g, dtype=torch.float32)

    has_scale = scale is not None

    # Collect every mask-free-valid tile, widest first.  Total time on this
    # backend is #memory_instructions * ~25-60 ns, so BLOCK (the lanes per
    # instruction) directly trades instruction count against time: 16384 >
    # 8192 > 4096 > 2048.  A candidate BLOCK is usable iff either
    #   - HS < BLOCK: pack G = BLOCK // HS chunks per program and G divides
    #     U = B * NT (mask-free fast path), or
    #   - HS >= BLOCK (and BLOCK divides HS): W = BLOCK, one hs-segment per
    #     program, G = 1 (always divides U).
    # Shapes satisfying neither (e.g. HS not a power of two, or U too small to
    # fill the tile) take the v5 masked kernel at BLOCK=2048.
    valid = []
    for BLOCK in _BLOCK_HS_CANDIDATES:
        if HS <= BLOCK and BLOCK % HS == 0:
            w = HS
            g_units = BLOCK // HS
            nh = 1
            if (B * NT) % g_units == 0:
                valid.append((BLOCK, w, g_units, nh))
        elif HS % BLOCK == 0:
            w = BLOCK
            g_units = 1
            nh = HS // BLOCK
            valid.append((BLOCK, w, g_units, nh))

    if valid:
        chosen = valid[0]  # widest tile by default
        # v9 rule (see module docstring): the 16384 tile halves the
        # instruction count but roughly doubles the per-instruction cost, so
        # it only wins at P >= 256 programs; below that the 8192 tile (and the
        # v8 downgrade below) is the measured optimum.  16384-valid implies
        # 8192-valid, so a rejected 16384 always falls back to valid[1].
        if chosen[0] == 16384:
            U = B * NT * chosen[3]
            if U // chosen[2] < 256 and len(valid) > 1:
                chosen = valid[1]
        # v8 rule (see module docstring): at P <= 4 programs the 8192 tile's
        # per-program overhead exceeds its width advantage, and 4096 with its
        # P in [4, 8] is the measured optimum.  Only downgrade to 4096 (never
        # to 2048: 2048 at P ~ 16 is far worse than either).
        if chosen is not None and chosen[0] == 8192:
            U = B * NT * chosen[3]
            if U // chosen[2] <= 4:
                for v in valid:
                    if v[0] == 4096:
                        chosen = v
                        break
    else:
        chosen = None

    if chosen is None:
        _launch_masked(g, out, scale, HS, BT, reverse, has_scale)
        return out

    BLOCK, w, g_units, nh = chosen
    U = B * NT * nh
    grid = (U // g_units,)

    launch_opts = {}
    if BLOCK >= 4096:
        # The XPU backend's unroll_control pass spursiously fails on scan
        # kernels with >= 4096-lane tiles (see module docstring); skipping it
        # per-launch is a documented backend option that other backends ignore.
        launch_opts.update(_XPU_WIDE_BLOCK_OPTS)

    _chunk_local_cumsum_vector_kernel[grid](
        g,
        out,
        1.0 if scale is None else scale,
        HS=HS,
        BT=BT,
        W=w,
        G=g_units,
        NH=nh,
        REVERSE=reverse,
        HAS_SCALE=has_scale,
        num_warps=1,  # ignored on Kunlun (warp_size = 1)
        **launch_opts,
    )
    return out


__all__ = ["chunk_local_cumsum_vector"]
