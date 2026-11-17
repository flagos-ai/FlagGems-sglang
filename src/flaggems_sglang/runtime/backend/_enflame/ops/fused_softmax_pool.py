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

"""GCU rewrite of fused_softmax_pool (FlagGems-sglang enflame backend).

The generic implementation dispatches to kernels tuned for CUDA/ROCm:
``_vector`` loads 3D tiles (broken on the 2026-11-07 toolchain: rank>=2
tile loads fail to compile with ``memref_ext.slice_(pad_)start`` 8 vs 7 /
7 vs 6 -- see compiler_issues/sglang_fused_softmax_pool.md), the ``.cg``
cache hints are ignored by the GCU backend, and its num_warps choices
(2/4 for 128-lane ops) plus natural ``grid=(bs,)`` launches leave the
CTA dispatch floor (~54ns/CTA x 1024 CTAs = 55us at bs=1024) dominating.

The rewrite keeps the proven-safe 1D access form (one flat [D]-wide load
per (row, k) half) and targets the two measured walls:

- Persistent grid-stride over rows, capped at 48 CTAs: removes the CTA
  dispatch floor for large bs (55us -> ~3us of dispatch).
- num_warps=1 per the 128-lane gather-kernel precedent: the generic
  2/4-warp choices are 1.1-9x slower here (k=64 nw4 1044us vs nw1 117us).
- Row pairing (U=4 for k=4, U=2 for k=16): batches U rows' loads into
  one loop iteration so more 256B loads are in flight; the useful depth
  follows an in-flight budget of roughly U*K <= 32 loads (k=64 pairing
  collapses to 621us -- see knowledge/perf_playbook.md).

Non-128 head dims and k values outside (4, 16, 64) keep the generic
fallback kernels verbatim (``_small``/``_large``), which compile and
pass accuracy unchanged.

Numbers (GCU400/L600, official bench core, do_bench median kernel mode,
warmup 1000ms / rep 100ms): bs=1024 k=4 55.1->10.6us, k=16 62.7->22.3us,
k=64 190.4->118.2us; bs=1 k=16/k=64 (3D-tile compile failure) 5.8->4.1 /
17.7->7.0us.

E-1 second round (2026-11-12, after the P01a rank>=2-tile toolchain fix
of 2026-11-10): k=64 now uses ``_pool_rows_tile`` -- one ``[K, D]`` tile
load per half, both issued before any consumer. The old k=64 path paid
the 1D load-issue wall (192 x 256B loads/row at ~28ns each per CTA);
probe ladder: 1D floor 65us + compute 52us = 117us vs tile floor 39us
+ compute 11us. Where-guard removal (single tile spans full k, so the
all -inf row already lands on the reference NaN path) and upfront load
issue (v no longer waits on the max of s) each contributed; num_stages,
block_ptr, nw>1 warp splits, D/K tile splits, fused [2D] loads with
register extraction (permute/split 17x, where+max 2x tax) and U-row
tile pairing were all probed and rejected. Numbers (same protocol):
bs=1024 k=64 118.7->50.6us (-57%), bs=4096 427.3->177.7 (linear), bs=64
12.9->8.6, bs=1 7.0->5.8. k=4/16 keep their 1D U-paired kernels: tile
forms lose there (a [16, 128] tile load carries a ~700ns fixed cost
that the 1D stream hides behind its 32-load in-flight budget).
"""

import torch
import triton
import triton.language as tl

# Persistent grid cap: CTA dispatch costs ~52-54ns each; 48 programs keep
# the machine full without paying the natural grid=(bs,) dispatch floor.
_GRID_CAP = 48
# Row-pairing thresholds: pairing duplicates clamped tail rows, so tiny
# batches stay on the unpaired kernel (launch floor dominates there).
_PAIR_MIN_ROWS = 16


@triton.jit
def _pool_rows(X, Y, N, K: tl.constexpr, D: tl.constexpr):
    # One output row per loop iteration, two passes over k (max, then
    # weighted sum). Strictly 1D accesses; 1D loads are immune to the
    # rank>=2 slice_(pad_)start lowering bug on this toolchain.
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    offs = tl.arange(0, D)
    for b in tl.range(pid, N, nprog):
        base = X + b * K * 2 * D
        m = tl.full((D,), float("-inf"), tl.float32)
        for k in tl.static_range(K):
            s = tl.load(base + k * 2 * D + D + offs).to(tl.float32)
            m = tl.maximum(m, s)
        acc = tl.zeros((D,), tl.float32)
        den = tl.zeros((D,), tl.float32)
        for k in tl.static_range(K):
            s = tl.load(base + k * 2 * D + D + offs).to(tl.float32)
            v = tl.load(base + k * 2 * D + offs).to(tl.float32)
            w = tl.exp(s - m)
            acc += v * w
            den += w
        tl.store(Y + b * D + offs, acc / den)


@triton.jit
def _pool_rows_u2(X, Y, N, K: tl.constexpr, D: tl.constexpr):
    # Two adjacent rows per iteration: both rows' k-loads issue together,
    # doubling the in-flight load depth per loop iteration.
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    offs = tl.arange(0, D)
    for b0 in tl.range(pid * 2, N, 2 * nprog):
        b1 = tl.minimum(b0 + 1, N - 1)
        base0 = X + b0 * K * 2 * D
        base1 = X + b1 * K * 2 * D
        m0 = tl.full((D,), float("-inf"), tl.float32)
        m1 = tl.full((D,), float("-inf"), tl.float32)
        for k in tl.static_range(K):
            s0 = tl.load(base0 + k * 2 * D + D + offs).to(tl.float32)
            s1 = tl.load(base1 + k * 2 * D + D + offs).to(tl.float32)
            m0 = tl.maximum(m0, s0)
            m1 = tl.maximum(m1, s1)
        a0 = tl.zeros((D,), tl.float32)
        d0 = tl.zeros((D,), tl.float32)
        a1 = tl.zeros((D,), tl.float32)
        d1 = tl.zeros((D,), tl.float32)
        for k in tl.static_range(K):
            s0 = tl.load(base0 + k * 2 * D + D + offs).to(tl.float32)
            v0 = tl.load(base0 + k * 2 * D + offs).to(tl.float32)
            s1 = tl.load(base1 + k * 2 * D + D + offs).to(tl.float32)
            v1 = tl.load(base1 + k * 2 * D + offs).to(tl.float32)
            w0 = tl.exp(s0 - m0)
            w1 = tl.exp(s1 - m1)
            a0 += v0 * w0
            d0 += w0
            a1 += v1 * w1
            d1 += w1
        tl.store(Y + b0 * D + offs, a0 / d0)
        if b1 != b0:
            tl.store(Y + b1 * D + offs, a1 / d1)


@triton.jit
def _pool_rows_u4(X, Y, N, K: tl.constexpr, D: tl.constexpr):
    # Four adjacent rows per iteration (k=4 shape: keeps the in-flight
    # load budget near U*K=16 x 256B loads). Tail rows clamp to N-1 and
    # recompute idempotently; duplicate stores write identical values.
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    offs = tl.arange(0, D)
    for b0 in tl.range(pid * 4, N, 4 * nprog):
        b1 = tl.minimum(b0 + 1, N - 1)
        b2 = tl.minimum(b0 + 2, N - 1)
        b3 = tl.minimum(b0 + 3, N - 1)
        base0 = X + b0 * K * 2 * D
        base1 = X + b1 * K * 2 * D
        base2 = X + b2 * K * 2 * D
        base3 = X + b3 * K * 2 * D
        m0 = tl.full((D,), float("-inf"), tl.float32)
        m1 = tl.full((D,), float("-inf"), tl.float32)
        m2 = tl.full((D,), float("-inf"), tl.float32)
        m3 = tl.full((D,), float("-inf"), tl.float32)
        for k in tl.static_range(K):
            s0 = tl.load(base0 + k * 2 * D + D + offs).to(tl.float32)
            s1 = tl.load(base1 + k * 2 * D + D + offs).to(tl.float32)
            s2 = tl.load(base2 + k * 2 * D + D + offs).to(tl.float32)
            s3 = tl.load(base3 + k * 2 * D + D + offs).to(tl.float32)
            m0 = tl.maximum(m0, s0)
            m1 = tl.maximum(m1, s1)
            m2 = tl.maximum(m2, s2)
            m3 = tl.maximum(m3, s3)
        a0 = tl.zeros((D,), tl.float32)
        d0 = tl.zeros((D,), tl.float32)
        a1 = tl.zeros((D,), tl.float32)
        d1 = tl.zeros((D,), tl.float32)
        a2 = tl.zeros((D,), tl.float32)
        d2 = tl.zeros((D,), tl.float32)
        a3 = tl.zeros((D,), tl.float32)
        d3 = tl.zeros((D,), tl.float32)
        for k in tl.static_range(K):
            s0 = tl.load(base0 + k * 2 * D + D + offs).to(tl.float32)
            v0 = tl.load(base0 + k * 2 * D + offs).to(tl.float32)
            s1 = tl.load(base1 + k * 2 * D + D + offs).to(tl.float32)
            v1 = tl.load(base1 + k * 2 * D + offs).to(tl.float32)
            s2 = tl.load(base2 + k * 2 * D + D + offs).to(tl.float32)
            v2 = tl.load(base2 + k * 2 * D + offs).to(tl.float32)
            s3 = tl.load(base3 + k * 2 * D + D + offs).to(tl.float32)
            v3 = tl.load(base3 + k * 2 * D + offs).to(tl.float32)
            w0 = tl.exp(s0 - m0)
            w1 = tl.exp(s1 - m1)
            w2 = tl.exp(s2 - m2)
            w3 = tl.exp(s3 - m3)
            a0 += v0 * w0
            d0 += w0
            a1 += v1 * w1
            d1 += w1
            a2 += v2 * w2
            d2 += w2
            a3 += v3 * w3
            d3 += w3
        tl.store(Y + b0 * D + offs, a0 / d0)
        if b1 != b0:
            tl.store(Y + b1 * D + offs, a1 / d1)
        if b2 != b0:
            tl.store(Y + b2 * D + offs, a2 / d2)
        if b3 != b0:
            tl.store(Y + b3 * D + offs, a3 / d3)


@triton.jit
def _pool_rows_tile(X, Y, N, K: tl.constexpr, D: tl.constexpr):
    # k=64 arm (E-1 second round): one [K, D] tile per half, both loads
    # issued before any consumer so their latencies overlap, replacing the
    # 3K-per-row 1D load stream whose issue rate dominated the old form.
    # Single tile spans the full k range: 2-pass math in one load pass, no
    # online rescale, and no -inf guards (an all--inf row yields
    # -inf - -inf = NaN, matching the reference softmax NaN).
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    cols = tl.arange(0, D)
    krows = tl.arange(0, K)
    for b in tl.range(pid, N, nprog):
        base = X + b * K * 2 * D
        s = tl.load(base + krows[:, None] * (2 * D) + D + cols[None, :]).to(
            tl.float32
        )
        v = tl.load(base + krows[:, None] * (2 * D) + cols[None, :]).to(
            tl.float32
        )
        m = tl.max(s, axis=0)
        w = tl.exp(s - m[None, :])
        acc = tl.sum(v * w, axis=0)
        den = tl.sum(w, axis=0)
        tl.store(Y + b * D + cols, acc / den)


# ---------------------------------------------------------------------------
# Generic fallback (verbatim from the generic implementation; serves every
# shape outside the specialized 128/4|16|64 dispatch above).
# ---------------------------------------------------------------------------


@triton.jit(do_not_specialize=["JOBS"])
def _small(
    X,
    Y,
    K: tl.constexpr,
    D: tl.constexpr,
    BLOCK: tl.constexpr,
    TILES: tl.constexpr,
    JOBS,
):
    offs = tl.arange(0, BLOCK)
    stride_k = D * 2
    for job in tl.range(tl.program_id(0), JOBS, tl.num_programs(0)):
        b = job // TILES
        c = (job - b * TILES) * BLOCK + offs
        mask = c < D
        base = X + b * K * stride_k
        m = tl.full((BLOCK,), float("-inf"), tl.float32)
        for k in tl.static_range(K):
            s = tl.load(
                base + k * stride_k + D + c, mask=mask, other=float("-inf")
            ).to(tl.float32)
            m = tl.maximum(m, s)
        m = tl.where(m == float("-inf"), 0.0, m)
        acc = tl.zeros((BLOCK,), tl.float32)
        den = tl.zeros((BLOCK,), tl.float32)
        for k in tl.static_range(K):
            s = tl.load(
                base + k * stride_k + D + c, mask=mask, other=float("-inf")
            ).to(tl.float32)
            v = tl.load(base + k * stride_k + c, mask=mask, other=0.0).to(
                tl.float32
            )
            w = tl.exp(s - m)
            acc += v * w
            den += w
        tl.store(Y + b * D + c, acc / den, mask=mask)


@triton.jit(do_not_specialize=["K", "JOBS"])
def _large(
    X, Y, K, D: tl.constexpr, BLOCK: tl.constexpr, TILES: tl.constexpr, JOBS
):
    offs = tl.arange(0, BLOCK)
    stride_k = D * 2
    for job in tl.range(tl.program_id(0), JOBS, tl.num_programs(0)):
        b = job // TILES
        c = (job - b * TILES) * BLOCK + offs
        mask = c < D
        base = X + b * K * stride_k
        m = tl.full((BLOCK,), float("-inf"), tl.float32)
        acc = tl.zeros((BLOCK,), tl.float32)
        den = tl.zeros((BLOCK,), tl.float32)
        for k in range(K):
            s = tl.load(
                base + k * stride_k + D + c, mask=mask, other=float("-inf")
            ).to(tl.float32)
            v = tl.load(base + k * stride_k + c, mask=mask, other=0.0).to(
                tl.float32
            )
            m2 = tl.maximum(m, s)
            alpha = tl.where(m == float("-inf"), 0.0, tl.exp(m - m2))
            w = tl.where(s == float("-inf"), 0.0, tl.exp(s - m2))
            acc = acc * alpha + v * w
            den = den * alpha + w
            m = m2
        tl.store(Y + b * D + c, acc / den, mask=mask)


def _block(k, d):
    if d <= 1:
        return 1
    if k <= 16 and d >= 512:
        return 512
    if k <= 16 and d >= 256:
        return 256
    if d >= 128:
        return 128
    return triton.next_power_of_2(d)


@triton.jit
def _zero(Y, N: tl.constexpr):
    i = tl.program_id(0) * 256 + tl.arange(0, 256)
    tl.store(Y + i, 0.0, i < N)


def fused_softmax_pool(kv_score, head_dim):
    bs, k, _ = kv_score.shape
    out = kv_score.new_empty((bs, head_dim))
    if k == 0 and bs > 0 and (head_dim > 0):
        _zero[triton.cdiv(bs * head_dim, 256),](out, bs * head_dim)
        return out
    if bs == 0 or head_dim == 0 or k == 0:
        return out
    if head_dim == 128 and k in (4, 16, 64):
        if k == 64:
            _pool_rows_tile[min(_GRID_CAP, bs),](
                kv_score, out, bs, K=64, D=128, num_warps=1
            )
            return out
        if k == 4 and bs >= _PAIR_MIN_ROWS:
            _pool_rows_u4[min(_GRID_CAP, triton.cdiv(bs, 4)),](
                kv_score, out, bs, K=4, D=128, num_warps=1
            )
            return out
        if k == 16 and bs >= _PAIR_MIN_ROWS:
            _pool_rows_u2[min(_GRID_CAP, triton.cdiv(bs, 2)),](
                kv_score, out, bs, K=16, D=128, num_warps=1
            )
            return out
        _pool_rows[min(_GRID_CAP, bs),](
            kv_score, out, bs, K=k, D=128, num_warps=1
        )
        return out
    block = _block(k, head_dim)
    tiles = triton.cdiv(head_dim, block)
    jobs = bs * tiles
    warps = 4 if block >= 128 else 2 if block >= 64 else 1
    if (
        head_dim == 128
        and (not torch.version.hip)
        and (bs >= 1024)
        and (k >= 16)
    ):
        warps = 2
    kernel = _small if k <= 64 else _large
    kernel[min(jobs, 4096),](
        kv_score,
        out,
        k,
        head_dim,
        block,
        tiles,
        jobs,
        num_warps=warps,
        enable_fp_fusion=False,
    )
    return out


__all__ = ["fused_softmax_pool"]
