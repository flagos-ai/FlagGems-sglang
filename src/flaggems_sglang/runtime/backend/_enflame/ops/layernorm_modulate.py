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

# Enflame vendor dispatch (GCU400, framework bench kernel-mode do_bench
# median w1000/r100, benchmark/test_layernorm_modulate.py --level core):
# * The upstream flat grid launches one CTA per row; at the Sana shapes
#   (d=2240, bf16) the per-row serial chain (load -> mean tree -> center ->
#   var tree -> store) times 8192 rows against only 48 resident CTAs
#   (2/SPC x 24 SP) and dominated the wall: 644.5us at (2,4096,2240),
#   ~230GB/s effective demand vs the ~1.5TB/s bf16 dual-stream floor.
# * The rewrite keeps ONE persistent program per wave slot (grid =
#   min(rows, 48)) iterating rows grid-stride. Measured core latencies
#   (baseline -> this, two-run stable to +/-0.3%):
#     (1,77)    9.5 -> 5.8us (1.65x)   (1,1024)  79.9 -> 30.7us (2.61x)
#     (1,4096) 313.5 -> 109.5us (2.86x) (2,77)   15.3 -> 8.0us (1.91x)
#     (2,1024) 158.4 -> 57.2us (2.77x) (2,4096) 645.5 -> 214.7us (3.01x)
#   geomean 2.41x; every shape now beats the fp32 torch composite too
#   (upstream was 0.45-0.90x on rows >= 1024).
# * How (each step measured on (2,4096,2240)):
#   - one-pass E[x^2]-E[x]^2 statistics: the two reduction trees issue in
#     parallel off the same loads instead of the serial mean->center->var
#     chain (group_norm_silu precedent, max(var,0) clamped);
#   - the row split into 512-wide chunks with independent tl.sum trees
#     (nested constexpr NC in {1,2,4,8}): the ablation ladder (load+store
#     floor 98.7us / +a/h 106.1 / full 279) put the trees at ~62% of the
#     wall; halving the per-tree serial depth took 279 -> 215us;
#   - X chunks grouped BEFORE the per-batch scale/shift loads (DRAM loads
#     first, L2 loads after): the interleaved form measured 290us vs 215us
#     for x-first (~35%) -- the scheduler only hoists the DRAM loads above
#     the trees when they are contiguous in trace order;
#   - a mask-free main span (largest pow2 <= d) plus one narrow masked
#     tail (next_pow2(d-M)): no predicate on the hot chunks;
#   - per-row 1D loads/stores only, num_warps=1 (nw2/nw4 pay the
#     cross-warp reduction tax, 1.4-1.6x slower here).
# * Dead ends measured and rejected: 2D [U,B] row tiles (affine AND
#   make_block_ptr, masked AND unmasked) fail first-compile with the
#   memref_ext.slice_pad_start operandSegmentSizes 7vs8 regression -- see
#   compiler_issues/sglang_2d_tile_fresh_compile_slice_pad_start.md; R-row
#   static_range unrolls gain no cross-row ILP (+10% slower); manual
#   next-row prefetch with a carried [2048] fp32 vector hits the DSM
#   round-trip pathology (+62%); num_stages pipelining has no effect.
# * Rows wider than 4096 (M > 4096) and non-contiguous inputs fall back
#   to _current, which is byte-identical to the generic implementation.
# * Known limit shared with the upstream kernels: row*D offsets stay in
#   int32 (rows > ~959k at d=2240 would overflow; such bf16 tensors
#   exceed 4GB anyway).
#
# Round-2 (E-1, 2026-11-12) — _ln_mod_rows4 four-row phase-batched arm:
# * The round-1 shim blocker on 2D tiles is gone (shared _gcuir_compat
#   downgrade retry, 2026-11-09): a 2D affine [U,B] tile now fresh-compiles
#   through the production shim chain. It is NOT adopted: probed
#   U{4,8}x nw{1,2} two-pass-L2-reload and bf16-hold forms measured
#   246.7-519.8us at (2,4096,2240) vs the 1D kernel's 213.7us -- the
#   row-segment DMA tax dominates (qk_head_layernorm P01a verdict).
# * Instead the qk_head_layernorm "multi-load" pattern (2026-11-12) is
#   transplanted onto the strictly-1D form: each grid-stride group takes
#   FOUR rows, issues all their x-chunk loads first, then all per-batch
#   a/h (L2) loads, and only then runs the per-row
#   (stats -> modulate -> store) chains. Row r+1's DMA fills under row
#   r's trees instead of behind its stores; the 1-row loop pays memory
#   and tree time additively.
# * Measured (kernel-mode do_bench median w1000/r100, six core shapes,
#   anchor = the round-1 kernel):
#     (1,77)    5.79 -> 5.67us (-2.1%)   (1,1024)  30.46 -> 20.70 (-32.0%)
#     (1,4096) 108.74 -> 66.80us (-38.6%) (2,77)    7.94 -> 6.40 (-19.4%)
#     (2,1024)  56.54 -> 35.46us (-37.3%) (2,4096) 213.74 -> 127.53 (-40.3%)
#   127.53us is 1.29x the 98.7us load+store floor; geomean -29.9%.
# * Round-2 dead ends measured and rejected: M=8 rows per group
#   (192.6us at (2,4096) -- live-tile pressure past the DSM budget,
#   same mechanism as the round-1 "+62% carried [2048]" veto); CW=1024
#   (144.3us, even) and CW=2048 (168.3us, 2048-lane vectors pay the
#   serialization); phase-batch without the a/h hoist (144.6us); grid 24
#   (296.5us at (2,4096) -- one wave wants all 48 DMA streams).
# * Dispatch: the 4-row arm covers m <= 2048 (NC <= 4) with rows >= 64
#   and non-fp32 dtypes (SHIFT stays on the round-1 kernel); m > 2048
#   (NC = 8, d = 4096 class) and fp32 inputs keep _ln_mod_rows.

import torch
import triton
import triton.language as tl

_DISPATCH_FLOOR = 48  # GCU400: one wave = 48 CTAs (2/SPC); extras cost ~50ns


@triton.jit
def _norm(
    X,
    A,
    H,
    Y,
    S: tl.constexpr,
    D: tl.constexpr,
    X0: tl.constexpr,
    X1: tl.constexpr,
    X2: tl.constexpr,
    A0: tl.constexpr,
    A1: tl.constexpr,
    H0: tl.constexpr,
    H1: tl.constexpr,
    EPS: tl.constexpr,
    C: tl.constexpr,
    SHIFT: tl.constexpr,
):
    row = tl.program_id(0)
    b = row // S
    s = row % S
    col = tl.arange(0, C)
    x = tl.load(X + b * X0 + s * X1 + col * X2, col < D, 0).to(tl.float32)
    if SHIFT:
        anchor = tl.load(X + b * X0 + s * X1).to(tl.float32)
        x = tl.where(col < D, x - anchor, 0.0)
    mean = tl.sum(x, 0) / D
    z = tl.where(col < D, x - mean, 0.0)
    var = tl.sum(z * z, 0) / D
    a = tl.load(A + b * A0 + col * A1, col < D, 0).to(tl.float32)
    h = tl.load(H + b * H0 + col * H1, col < D, 0).to(tl.float32)
    y = z * tl.rsqrt(var + EPS)
    y = y * (1.0 + a) + h
    tl.store(Y + row * D + col, y, col < D)


def _current(x, scale, shift, eps):
    b, s, d = x.shape
    y = torch.empty((b, s, d), device=x.device, dtype=x.dtype)
    c = triton.next_power_of_2(d)
    _norm[b * s,](
        x,
        scale,
        shift,
        y,
        s,
        d,
        *x.stride(),
        *scale.stride(),
        *shift.stride(),
        eps,
        c,
        x.dtype == torch.float32,
        num_warps=4,
        enable_fp_fusion=False,
    )
    return y


@triton.jit
def _ln_mod_rows(
    X,
    A,
    H,
    Y,
    S,
    ROWS,
    eps,
    D: tl.constexpr,
    CW: tl.constexpr,
    NC: tl.constexpr,
    TB: tl.constexpr,
    HAS_TAIL: tl.constexpr,
    SHIFT: tl.constexpr,
):
    pid = tl.program_id(0)
    npg = tl.num_programs(0)
    c = tl.arange(0, CW)
    if HAS_TAIL:
        ct = NC * CW + tl.arange(0, TB)
    for r in tl.range(pid, ROWS, npg):
        b = r // S
        x0 = tl.load(X + r * D + c).to(tl.float32)
        if NC >= 2:
            o1 = CW + c
            x1 = tl.load(X + r * D + o1).to(tl.float32)
            if NC >= 4:
                o2 = 2 * CW + c
                x2 = tl.load(X + r * D + o2).to(tl.float32)
                o3 = 3 * CW + c
                x3 = tl.load(X + r * D + o3).to(tl.float32)
                if NC >= 8:
                    o4 = 4 * CW + c
                    x4 = tl.load(X + r * D + o4).to(tl.float32)
                    o5 = 5 * CW + c
                    x5 = tl.load(X + r * D + o5).to(tl.float32)
                    o6 = 6 * CW + c
                    x6 = tl.load(X + r * D + o6).to(tl.float32)
                    o7 = 7 * CW + c
                    x7 = tl.load(X + r * D + o7).to(tl.float32)
        if HAS_TAIL:
            xt = tl.load(X + r * D + ct, ct < D, 0).to(tl.float32)
        a0 = tl.load(A + b * D + c).to(tl.float32)
        h0 = tl.load(H + b * D + c).to(tl.float32)
        if NC >= 2:
            a1 = tl.load(A + b * D + o1).to(tl.float32)
            h1 = tl.load(H + b * D + o1).to(tl.float32)
            if NC >= 4:
                a2 = tl.load(A + b * D + o2).to(tl.float32)
                h2 = tl.load(H + b * D + o2).to(tl.float32)
                a3 = tl.load(A + b * D + o3).to(tl.float32)
                h3 = tl.load(H + b * D + o3).to(tl.float32)
                if NC >= 8:
                    a4 = tl.load(A + b * D + o4).to(tl.float32)
                    h4 = tl.load(H + b * D + o4).to(tl.float32)
                    a5 = tl.load(A + b * D + o5).to(tl.float32)
                    h5 = tl.load(H + b * D + o5).to(tl.float32)
                    a6 = tl.load(A + b * D + o6).to(tl.float32)
                    h6 = tl.load(H + b * D + o6).to(tl.float32)
                    a7 = tl.load(A + b * D + o7).to(tl.float32)
                    h7 = tl.load(H + b * D + o7).to(tl.float32)
        if HAS_TAIL:
            at = tl.load(A + b * D + ct, ct < D, 0).to(tl.float32)
            ht = tl.load(H + b * D + ct, ct < D, 0).to(tl.float32)
        if SHIFT:
            anchor = tl.load(X + r * D).to(tl.float32)
            x0 = x0 - anchor
            if NC >= 2:
                x1 = x1 - anchor
                if NC >= 4:
                    x2 = x2 - anchor
                    x3 = x3 - anchor
                    if NC >= 8:
                        x4 = x4 - anchor
                        x5 = x5 - anchor
                        x6 = x6 - anchor
                        x7 = x7 - anchor
            if HAS_TAIL:
                xt = tl.where(ct < D, xt - anchor, 0.0)
        sx = tl.sum(x0, 0)
        sxx = tl.sum(x0 * x0, 0)
        if NC >= 2:
            sx += tl.sum(x1, 0)
            sxx += tl.sum(x1 * x1, 0)
            if NC >= 4:
                sx += tl.sum(x2, 0)
                sxx += tl.sum(x2 * x2, 0)
                sx += tl.sum(x3, 0)
                sxx += tl.sum(x3 * x3, 0)
                if NC >= 8:
                    sx += tl.sum(x4, 0)
                    sxx += tl.sum(x4 * x4, 0)
                    sx += tl.sum(x5, 0)
                    sxx += tl.sum(x5 * x5, 0)
                    sx += tl.sum(x6, 0)
                    sxx += tl.sum(x6 * x6, 0)
                    sx += tl.sum(x7, 0)
                    sxx += tl.sum(x7 * x7, 0)
        if HAS_TAIL:
            sx += tl.sum(xt, 0)
            sxx += tl.sum(xt * xt, 0)
        mean = sx / D
        var = tl.maximum(sxx / D - mean * mean, 0.0)
        rstd = tl.rsqrt(var + eps)
        y0 = (x0 - mean) * rstd
        y0 = y0 * (1.0 + a0) + h0
        tl.store(Y + r * D + c, y0)
        if NC >= 2:
            y1 = (x1 - mean) * rstd
            y1 = y1 * (1.0 + a1) + h1
            tl.store(Y + r * D + o1, y1)
            if NC >= 4:
                y2 = (x2 - mean) * rstd
                y2 = y2 * (1.0 + a2) + h2
                tl.store(Y + r * D + o2, y2)
                y3 = (x3 - mean) * rstd
                y3 = y3 * (1.0 + a3) + h3
                tl.store(Y + r * D + o3, y3)
                if NC >= 8:
                    y4 = (x4 - mean) * rstd
                    y4 = y4 * (1.0 + a4) + h4
                    tl.store(Y + r * D + o4, y4)
                    y5 = (x5 - mean) * rstd
                    y5 = y5 * (1.0 + a5) + h5
                    tl.store(Y + r * D + o5, y5)
                    y6 = (x6 - mean) * rstd
                    y6 = y6 * (1.0 + a6) + h6
                    tl.store(Y + r * D + o6, y6)
                    y7 = (x7 - mean) * rstd
                    y7 = y7 * (1.0 + a7) + h7
                    tl.store(Y + r * D + o7, y7)
        if HAS_TAIL:
            yt = (xt - mean) * rstd
            yt = yt * (1.0 + at) + ht
            tl.store(Y + r * D + ct, yt, ct < D)


@triton.jit
def _l4_loads(
    P,
    r,
    D: tl.constexpr,
    CW: tl.constexpr,
    NC: tl.constexpr,
    c,
    ct,
    HAS_TAIL: tl.constexpr,
):
    x0 = tl.load(P + r * D + c).to(tl.float32)
    x1 = tl.zeros_like(x0)
    x2 = tl.zeros_like(x0)
    x3 = tl.zeros_like(x0)
    if NC >= 2:
        x1 = tl.load(P + r * D + CW + c).to(tl.float32)
    if NC >= 4:
        x2 = tl.load(P + r * D + 2 * CW + c).to(tl.float32)
        x3 = tl.load(P + r * D + 3 * CW + c).to(tl.float32)
    xt = tl.zeros_like(x0)
    if HAS_TAIL:
        xt = tl.load(P + r * D + ct, ct < D, 0).to(tl.float32)
    return x0, x1, x2, x3, xt


@triton.jit
def _l4_ah(
    PA,
    PH,
    b,
    D: tl.constexpr,
    CW: tl.constexpr,
    NC: tl.constexpr,
    c,
    ct,
    HAS_TAIL: tl.constexpr,
):
    a0 = tl.load(PA + b * D + c).to(tl.float32)
    h0 = tl.load(PH + b * D + c).to(tl.float32)
    a1 = tl.zeros_like(a0)
    h1 = tl.zeros_like(h0)
    a2 = tl.zeros_like(a0)
    h2 = tl.zeros_like(h0)
    a3 = tl.zeros_like(a0)
    h3 = tl.zeros_like(h0)
    if NC >= 2:
        a1 = tl.load(PA + b * D + CW + c).to(tl.float32)
        h1 = tl.load(PH + b * D + CW + c).to(tl.float32)
    if NC >= 4:
        a2 = tl.load(PA + b * D + 2 * CW + c).to(tl.float32)
        h2 = tl.load(PH + b * D + 2 * CW + c).to(tl.float32)
        a3 = tl.load(PA + b * D + 3 * CW + c).to(tl.float32)
        h3 = tl.load(PH + b * D + 3 * CW + c).to(tl.float32)
    at = tl.zeros_like(a0)
    ht = tl.zeros_like(h0)
    if HAS_TAIL:
        at = tl.load(PA + b * D + ct, ct < D, 0).to(tl.float32)
        ht = tl.load(PH + b * D + ct, ct < D, 0).to(tl.float32)
    return a0, a1, a2, a3, at, h0, h1, h2, h3, ht


@triton.jit
def _l4_stat(
    x0,
    x1,
    x2,
    x3,
    xt,
    NC: tl.constexpr,
    HAS_TAIL: tl.constexpr,
    D: tl.constexpr,
    eps,
):
    # Same accumulation order as _ln_mod_rows (chunk-serial, left-assoc).
    sx = tl.sum(x0, 0)
    sxx = tl.sum(x0 * x0, 0)
    if NC >= 2:
        sx += tl.sum(x1, 0)
        sxx += tl.sum(x1 * x1, 0)
    if NC >= 4:
        sx += tl.sum(x2, 0)
        sxx += tl.sum(x2 * x2, 0)
        sx += tl.sum(x3, 0)
        sxx += tl.sum(x3 * x3, 0)
    if HAS_TAIL:
        sx += tl.sum(xt, 0)
        sxx += tl.sum(xt * xt, 0)
    mean = sx / D
    var = tl.maximum(sxx / D - mean * mean, 0.0)
    return mean, tl.rsqrt(var + eps)


@triton.jit
def _l4_out(
    x0,
    x1,
    x2,
    x3,
    xt,
    a0,
    a1,
    a2,
    a3,
    at,
    h0,
    h1,
    h2,
    h3,
    ht,
    Y,
    r,
    D: tl.constexpr,
    CW: tl.constexpr,
    NC: tl.constexpr,
    c,
    ct,
    HAS_TAIL: tl.constexpr,
    mean,
    rstd,
):
    y0 = (x0 - mean) * rstd
    tl.store(Y + r * D + c, y0 * (1.0 + a0) + h0)
    if NC >= 2:
        y1 = (x1 - mean) * rstd
        tl.store(Y + r * D + CW + c, y1 * (1.0 + a1) + h1)
    if NC >= 4:
        y2 = (x2 - mean) * rstd
        tl.store(Y + r * D + 2 * CW + c, y2 * (1.0 + a2) + h2)
        y3 = (x3 - mean) * rstd
        tl.store(Y + r * D + 3 * CW + c, y3 * (1.0 + a3) + h3)
    if HAS_TAIL:
        yt = (xt - mean) * rstd
        tl.store(Y + r * D + ct, yt * (1.0 + at) + ht, ct < D)


@triton.jit
def _ln_mod_rows4(
    X,
    A,
    H,
    Y,
    S,
    ROWS,
    eps,
    D: tl.constexpr,
    CW: tl.constexpr,
    NC: tl.constexpr,
    TB: tl.constexpr,
    HAS_TAIL: tl.constexpr,
):
    # Four-row phase-batched groups on the same persistent grid-stride
    # scheme as _ln_mod_rows (round-2, qk_head_layernorm multi-load
    # precedent): per group, ALL x-chunk loads of the four rows are
    # issued first, then the four per-batch a/h (L2) loads, and only
    # then the per-row (stats -> modulate -> store) chains interleave.
    # Row r+1's DMA therefore fills under row r's reduction trees
    # instead of behind its stores.
    pid = tl.program_id(0)
    npg = tl.num_programs(0)
    c = tl.arange(0, CW)
    ct = NC * CW + tl.arange(0, TB)
    n_groups = tl.cdiv(ROWS, 4)
    for g in tl.range(pid, n_groups, npg):
        # Tail-anchored group (idempotent rewrite of the last <=3 rows).
        r0 = tl.minimum(g * 4, ROWS - 4)
        xa0, xa1, xa2, xa3, xat = _l4_loads(X, r0, D, CW, NC, c, ct, HAS_TAIL)
        xb0, xb1, xb2, xb3, xbt = _l4_loads(
            X, r0 + 1, D, CW, NC, c, ct, HAS_TAIL
        )
        xc0, xc1, xc2, xc3, xct = _l4_loads(
            X, r0 + 2, D, CW, NC, c, ct, HAS_TAIL
        )
        xd0, xd1, xd2, xd3, xdt = _l4_loads(
            X, r0 + 3, D, CW, NC, c, ct, HAS_TAIL
        )
        aa0, aa1, aa2, aa3, aat, ha0, ha1, ha2, ha3, hat = _l4_ah(
            A, H, r0 // S, D, CW, NC, c, ct, HAS_TAIL
        )
        ab0, ab1, ab2, ab3, abt, hb0, hb1, hb2, hb3, hbt = _l4_ah(
            A, H, (r0 + 1) // S, D, CW, NC, c, ct, HAS_TAIL
        )
        ac0, ac1, ac2, ac3, act, hc0, hc1, hc2, hc3, hct = _l4_ah(
            A, H, (r0 + 2) // S, D, CW, NC, c, ct, HAS_TAIL
        )
        ad0, ad1, ad2, ad3, adt, hd0, hd1, hd2, hd3, hdt = _l4_ah(
            A, H, (r0 + 3) // S, D, CW, NC, c, ct, HAS_TAIL
        )
        mean_a, rstd_a = _l4_stat(
            xa0, xa1, xa2, xa3, xat, NC, HAS_TAIL, D, eps
        )
        _l4_out(
            xa0,
            xa1,
            xa2,
            xa3,
            xat,
            aa0,
            aa1,
            aa2,
            aa3,
            aat,
            ha0,
            ha1,
            ha2,
            ha3,
            hat,
            Y,
            r0,
            D,
            CW,
            NC,
            c,
            ct,
            HAS_TAIL,
            mean_a,
            rstd_a,
        )
        mean_b, rstd_b = _l4_stat(
            xb0, xb1, xb2, xb3, xbt, NC, HAS_TAIL, D, eps
        )
        _l4_out(
            xb0,
            xb1,
            xb2,
            xb3,
            xbt,
            ab0,
            ab1,
            ab2,
            ab3,
            abt,
            hb0,
            hb1,
            hb2,
            hb3,
            hbt,
            Y,
            r0 + 1,
            D,
            CW,
            NC,
            c,
            ct,
            HAS_TAIL,
            mean_b,
            rstd_b,
        )
        mean_c, rstd_c = _l4_stat(
            xc0, xc1, xc2, xc3, xct, NC, HAS_TAIL, D, eps
        )
        _l4_out(
            xc0,
            xc1,
            xc2,
            xc3,
            xct,
            ac0,
            ac1,
            ac2,
            ac3,
            act,
            hc0,
            hc1,
            hc2,
            hc3,
            hct,
            Y,
            r0 + 2,
            D,
            CW,
            NC,
            c,
            ct,
            HAS_TAIL,
            mean_c,
            rstd_c,
        )
        mean_d, rstd_d = _l4_stat(
            xd0, xd1, xd2, xd3, xdt, NC, HAS_TAIL, D, eps
        )
        _l4_out(
            xd0,
            xd1,
            xd2,
            xd3,
            xdt,
            ad0,
            ad1,
            ad2,
            ad3,
            adt,
            hd0,
            hd1,
            hd2,
            hd3,
            hdt,
            Y,
            r0 + 3,
            D,
            CW,
            NC,
            c,
            ct,
            HAS_TAIL,
            mean_d,
            rstd_d,
        )


def layernorm_modulate(x, scale, shift, eps):
    if not (
        x.is_contiguous() and scale.is_contiguous() and shift.is_contiguous()
    ):
        return _current(x, scale, shift, eps)
    b, s, d = x.shape
    rows = b * s
    y = torch.empty_like(x)
    if rows == 0 or d == 0:
        return y
    m = 1 << max(d.bit_length() - 1, 0)
    if m > 4096:
        return _current(x, scale, shift, eps)
    t = d - m
    tb = triton.next_power_of_2(t) if t else 1
    cw = m if m < 512 else 512
    nc = m // cw
    if m <= 2048 and rows >= 64 and x.dtype is not torch.float32:
        # Four-row phase-batched arm (round-2): hides the per-row DMA
        # latency behind other rows' reduction trees.
        _ln_mod_rows4[(min(-(-rows // 4), _DISPATCH_FLOOR),)](
            x,
            scale,
            shift,
            y,
            s,
            rows,
            float(eps),
            d,
            cw,
            nc,
            tb,
            t > 0,
            num_warps=1,
            enable_fp_fusion=False,
        )
        return y
    _ln_mod_rows[(min(rows, _DISPATCH_FLOOR),)](
        x,
        scale,
        shift,
        y,
        s,
        rows,
        float(eps),
        d,
        cw,
        nc,
        tb,
        t > 0,
        x.dtype == torch.float32,
        num_warps=1,
        enable_fp_fusion=False,
    )
    return y


__all__ = ["layernorm_modulate"]
