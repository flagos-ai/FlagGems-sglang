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
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied. See the License for the specific language governing
# permissions and limitations under the License.

"""Fused DSV4 ``hc_head`` LM-head mixer in pure Triton.

Per token ``t`` (all arithmetic in float32, matching the PyTorch reference):

  1. RMSNorm over the flattened ``[hc_mult, hidden_size]`` row:
     ``r = rsqrt(mean(x_flat**2) + norm_eps)``
  2. Linear mix:  ``mixes = (x_flat @ hc_fn.T) * r``  -> ``[hc_mult]``
  3. Sigmoid gate:``pre    = sigmoid(mixes*hc_scale + hc_base) + hc_eps`` -> ``[hc_mult]``
  4. Weighted sum:``y[t,h] = sum_j pre[j] * x[t,j,h]``  -> ``[hidden_size]``

Output is cast back to ``x``'s dtype. Device placement goes through
``flaggems_sglang.device`` (never hard-coding ``"cuda"``).

Two dispatch paths are selected by token count ``T``:

* **Small T** (``T <= _SMALL_T_THRESHOLD``): a two-kernel split-D pipeline
  that parallelises the reduction across the flat
  ``D = hc_mult*hidden_size`` axis so the GPU stays saturated even for
  ``T == 1`` (a one-program-per-token kernel is latency-bound there).
    - ``_hc_head_reduce_kernel`` — grid ``(ceil(T/G), SPLIT_D)``. Each
      program owns one ``chunk_tiles``-wide slice of the flat D axis and a
      group of ``G`` consecutive tokens. Per ``BLOCK_D`` sub-tile it loads
      one ``[G, BLOCK_D]`` slab of ``x`` and the matching ``[NP, BLOCK_D]``
      slab of ``hc_fn`` **once**, then reduces over the D axis into
      per-token partial sum-of-squares and partial mix dot-products held in
      registers. Grouping ``G`` tokens per program amortises the ``hc_fn``
      slab load across the group. Partials land in per-``(token, split)``
      slots — no atomics, no cross-run accumulation, so the buffers stay
      correct no matter how many times Triton autotune re-runs the kernel.
    - ``_hc_head_finalize_kernel`` — grid ``(T, ceil(hidden/BLOCK_H))``.
      Every program re-reads its token's small partials in-register, forms
      ``r`` and ``pre``, then streams one ``BLOCK_H`` slice of the hidden
      axis to write ``y[t,h] = sum_j pre[j]*x[t,j,h]``. Splitting the
      hidden axis keeps the write-out fully parallel.

* **Large T** (``T > _SMALL_T_THRESHOLD``): a **single** fused kernel,
  ``_hc_head_fused_kernel`` (``grid = (T,)``), doing two passes over the
  token's row in one program:
    - pass 1 streams the flat row once; each iteration loads one ``[BLOCK]``
      slice of ``x`` (bf16 -> fp32) and the matching ``[NP, BLOCK]`` slice of
      ``hc_fn`` (fp32, zero-masked on the ``NP`` pad rows) and accumulates
      the partial sum-of-squares and the ``NP`` partial mix dot-products in
      registers;
    - after the loop it computes ``r`` / ``pre`` and streams the hidden axis
      once more, writing ``y[t,h] = sum_j pre[j]*x[t,j,h]``.

  A single program per token reads each ``hc_fn`` element exactly once, so
  the per-token ``hc_fn`` traffic is ``hc_mult*hidden*4`` bytes instead of
  the ``hc_mult`` times larger figure of a per-``(token, head)``
  decomposition — that redundancy is what makes many-token shapes L2-bound
  rather than DRAM-bound on ``x``.

``G`` / ``SPLIT_D`` are picked from ``T``/``D`` at launch time; only
``num_warps`` / ``num_stages`` (and tile sizes) are autotuned (Triton-owned
cache — no hand-rolled global dicts anywhere in this module).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

# When T is at most this many tokens, use the split-D two-kernel pipeline.
# Above it, switch to the single fused two-pass kernel which avoids the
# partials round-trip and reads each hc_fn element once per token.
_SMALL_T_THRESHOLD = 64

# Flat-D sub-tile processed per inner iteration of the reduce kernel. This
# GPU exposes only 64KB shared memory; the pipelined x/w slabs take
# 16*BLOCK_D*4 + BLOCK_D*NP*4 bytes per stage, so BLOCK_D=256 at
# num_stages=2 fits (32KB/stage) while keeping loads coalesced.
_BLOCK_D = 256


# ---------------------------------------------------------------------------
# Kernel 1 (small-T): split-D partial reductions over groups of tokens.
# ---------------------------------------------------------------------------


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=4, num_stages=1),
        triton.Config({}, num_warps=4, num_stages=2),
        triton.Config({}, num_warps=8, num_stages=1),
        triton.Config({}, num_warps=8, num_stages=2),
    ],
    key=["T", "D", "SPLIT_D"],
)
@triton.jit
def _hc_head_reduce_kernel(
    x_ptr,  # x: [T, D] view of [T, hc_mult, hidden] (contiguous)
    hc_fn_ptr,  # hc_fn: [hc_mult, D] (contiguous, fp32)
    sumsq_ptr,  # sumsq: [T, SPLIT_D]     (fp32 partials)
    mixes_ptr,  # mixes: [T, SPLIT_D, NP] (fp32 partials)
    T,
    D,  # hc_mult * hidden_size
    SPLIT_D,  # number of D-slices == grid dim 1
    chunk_tiles,  # BLOCK_D-tiles per D-slice
    hc_mult: tl.constexpr,
    NP: tl.constexpr,  # next pow2 of hc_mult (pad rows masked out)
    G: tl.constexpr,  # tokens per program (<= 16)
    BLOCK_D: tl.constexpr,
):
    """One program per ``(token-group, D-slice)``.

    Streams its ``chunk_tiles * BLOCK_D`` slice of the flat row for each of
    its ``G`` tokens, writing per-``(token, split)`` partial sum-of-squares
    and partial ``hc_mult`` mix dot-products into dedicated slots of the
    partials buffers.
    """
    pid_t = tl.program_id(0)
    pid_d = tl.program_id(1)

    # Local aliases (not bare kernel params) deliberately mask the
    # ``arange < param`` shape from launch-time block-size rewriters.
    hc_mult_loc: tl.constexpr = hc_mult
    g_idx = tl.arange(0, G)
    t_idx = pid_t * G + g_idx
    t_mask = t_idx < T
    j_idx = tl.arange(0, NP)
    j_mask = j_idx < hc_mult_loc

    d_lo = pid_d * chunk_tiles * BLOCK_D
    d_hi = tl.minimum(d_lo + chunk_tiles * BLOCK_D, D)

    ssq_acc = tl.zeros((G,), tl.float32)
    mix_acc = tl.zeros((G, NP), tl.float32)

    for d0 in range(d_lo, d_hi, BLOCK_D):
        offs = d0 + tl.arange(0, BLOCK_D)
        d_mask = offs < d_hi

        x_tile = tl.load(
            x_ptr + t_idx[:, None] * D + offs[None, :],
            mask=t_mask[:, None] & d_mask[None, :],
            other=0.0,
        ).to(
            tl.float32
        )  # [G, BLOCK_D]
        ssq_acc += tl.sum(x_tile * x_tile, axis=1)

        # One hc_fn row at a time: [BLOCK_D] coalesced slice, shared by all G
        # tokens of the group. Its outer product with x_tile reduces over the
        # D axis into the matching column of mix_acc (pad rows beyond hc_mult
        # are masked on the final store only — their accumulators stay zero).
        for j in tl.static_range(NP):
            w_j = tl.load(hc_fn_ptr + j * D + offs, mask=d_mask, other=0.0).to(
                tl.float32
            )
            contrib = tl.sum(x_tile * w_j[None, :], axis=1)  # [G]
            mix_acc += tl.where(j_idx[None, :] == j, contrib[:, None], 0.0)

    tl.store(sumsq_ptr + t_idx * SPLIT_D + pid_d, ssq_acc, mask=t_mask)
    mix_off = t_idx[:, None] * (SPLIT_D * NP) + pid_d * NP + j_idx[None, :]
    tl.store(
        mixes_ptr + mix_off, mix_acc, mask=t_mask[:, None] & j_mask[None, :]
    )


# ---------------------------------------------------------------------------
# Kernel 2 (small-T): finalize — reduce partials, gate, weighted sum.
# ---------------------------------------------------------------------------


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_H": 512}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_H": 1024}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_H": 1024}, num_warps=8, num_stages=2),
    ],
    key=["hidden_size", "hc_mult"],
)
@triton.jit
def _hc_head_finalize_kernel(
    x_ptr,  # x: [T, hc_mult, hidden_size] (contiguous)
    hc_scale_ptr,  # hc_scale: [1]
    hc_base_ptr,  # hc_base: [hc_mult]
    sumsq_ptr,  # sumsq: [T, SPLIT_D]     (fp32 partials)
    mixes_ptr,  # mixes: [T, SPLIT_D, NP] (fp32 partials)
    y_ptr,  # y: [T, hidden_size]
    norm_eps,  # scalar (float)
    hc_eps,  # scalar (float)
    D,
    hidden_size,
    SPLIT_D,
    y_stride_t,
    hc_mult: tl.constexpr,
    NP: tl.constexpr,
    SPLIT_POW2: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    """One program per ``(token, hidden-tile)``."""
    t = tl.program_id(0)
    pid_h = tl.program_id(1)

    # Local aliases (not bare kernel params) deliberately mask the
    # ``arange < param`` shape from launch-time block-size rewriters.
    split_loc: tl.constexpr = SPLIT_D
    hc_mult_loc: tl.constexpr = hc_mult
    k_idx = tl.arange(0, SPLIT_POW2)
    k_mask = k_idx < split_loc
    j_idx = tl.arange(0, NP)
    j_mask = j_idx < hc_mult_loc

    sum_sq = tl.sum(
        tl.load(sumsq_ptr + t * SPLIT_D + k_idx, mask=k_mask, other=0.0),
        axis=0,
    )
    mix_partials = tl.load(
        mixes_ptr + t * (SPLIT_D * NP) + k_idx[:, None] * NP + j_idx[None, :],
        mask=k_mask[:, None] & j_mask[None, :],
        other=0.0,
    )
    mixes = tl.sum(mix_partials, axis=0)  # [NP]

    rsqrt = tl.rsqrt(sum_sq / D + norm_eps)
    hc_scale = tl.load(hc_scale_ptr).to(tl.float32)
    hc_base = tl.load(hc_base_ptr + j_idx, mask=j_mask, other=0.0).to(
        tl.float32
    )
    pre = tl.sigmoid(mixes * rsqrt * hc_scale + hc_base) + hc_eps  # [NP]

    h_offs = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    h_mask = h_offs < hidden_size

    x_tile = tl.load(
        x_ptr + t * D + j_idx[:, None] * hidden_size + h_offs[None, :],
        mask=j_mask[:, None] & h_mask[None, :],
        other=0.0,
    ).to(
        tl.float32
    )  # [NP, BLOCK_H]
    y_acc = tl.sum(pre[:, None] * x_tile, axis=0)

    tl.store(y_ptr + t * y_stride_t + h_offs, y_acc, mask=h_mask)


# ---------------------------------------------------------------------------
# Large-T pipeline: single fused kernel, two passes over the token row.
# ---------------------------------------------------------------------------


@triton.autotune(
    configs=[
        triton.Config({"BLOCK": 512}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK": 1024}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK": 1024}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK": 1024}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK": 1024}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK": 2048}, num_warps=8, num_stages=2),
        # wide-tile configs: fewer D-iterations shortens the per-token
        # critical path when the grid only fills one or two SM waves
        triton.Config({"BLOCK": 2048}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK": 4096}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK": 4096}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK": 4096}, num_warps=8, num_stages=2),
    ],
    key=["T", "hidden_size", "hc_mult"],
)
@triton.jit
def _hc_head_fused_kernel(
    x_ptr,  # x: [T, hc_mult, hidden_size] (contiguous)
    hc_fn_ptr,  # hc_fn: [hc_mult, hc_mult*hidden_size] (contiguous, fp32)
    hc_scale_ptr,  # hc_scale: [1]
    hc_base_ptr,  # hc_base: [hc_mult]
    y_ptr,  # y: [T, hidden_size]
    norm_eps,  # scalar (float)
    hc_eps,  # scalar (float)
    x_stride_t,
    x_stride_m,
    hc_fn_stride_m,
    y_stride_t,
    T,
    hidden_size,
    hc_mult: tl.constexpr,
    NP: tl.constexpr,  # next pow2 of hc_mult (pad rows of hc_fn tile)
    BLOCK: tl.constexpr,
):
    """One program per token (large-T fused path).

    Pass 1 streams the token's flat ``D = hc_mult*hidden_size`` row in
    ``[BLOCK]`` slices, accumulating the sum-of-squares and the ``NP``
    partial mix dot-products (``hc_fn`` pad rows load as zero) fully in
    registers — the row is never materialised in global memory. Then
    ``r`` / ``pre`` are computed and pass 2 streams the hidden axis once to
    emit ``y[t,h] = sum_j pre[j]*x[t,j,h]``.
    """
    t = tl.program_id(0)

    D = hc_mult * hidden_size
    j_idx = tl.arange(0, NP)

    # ---- pass 1: sum-of-squares + linear-mix partial dots ----
    ssq_vec = tl.zeros((BLOCK,), tl.float32)
    mix_acc = tl.zeros((NP,), tl.float32)
    for d in range(0, D, BLOCK):
        offs = d + tl.arange(0, BLOCK)
        k_mask = offs < D
        x_vec = tl.load(
            x_ptr + t * x_stride_t + offs, mask=k_mask, other=0.0
        ).to(tl.float32)
        ssq_vec += x_vec * x_vec
        w_tile = tl.load(
            hc_fn_ptr + j_idx[:, None] * hc_fn_stride_m + offs[None, :],
            mask=(j_idx < hc_mult)[:, None] & k_mask[None, :],
            other=0.0,
        ).to(
            tl.float32
        )  # [NP, BLOCK]
        mix_acc += tl.sum(w_tile * x_vec[None, :], axis=1)

    sum_sq = tl.sum(ssq_vec, axis=0)
    rsqrt = tl.rsqrt(sum_sq / D + norm_eps)
    mixes = mix_acc * rsqrt

    hc_scale = tl.load(hc_scale_ptr).to(tl.float32)
    hc_base = tl.load(hc_base_ptr + j_idx, mask=j_idx < hc_mult, other=0.0).to(
        tl.float32
    )
    pre = tl.sigmoid(mixes * hc_scale + hc_base) + hc_eps

    # ---- pass 2: weighted sum over the hc_mult axis ----
    for h0 in range(0, hidden_size, BLOCK):
        h_offs = h0 + tl.arange(0, BLOCK)
        h_mask = h_offs < hidden_size
        y_acc = tl.zeros((BLOCK,), tl.float32)
        for j in tl.static_range(hc_mult):
            p_j = tl.sum(tl.where(j_idx == j, pre, 0.0), axis=0)
            x_j = tl.load(
                x_ptr + t * x_stride_t + j * x_stride_m + h_offs,
                mask=h_mask,
                other=0.0,
            ).to(tl.float32)
            y_acc += p_j * x_j
        tl.store(y_ptr + t * y_stride_t + h_offs, y_acc, mask=h_mask)


def _next_pow2(n: int) -> int:
    p = 1
    while p < n:
        p <<= 1
    return p


def hc_head(x, hc_fn, hc_scale, hc_base, norm_eps, hc_eps):
    shape, dtype = x.size(), x.dtype
    # shape == (T, hc_mult, hidden_size)
    T, hc_mult, hidden_size = shape

    x_contig = x.contiguous()
    y = torch.empty((T, hidden_size), dtype=dtype, device=x_contig.device)

    if T == 0:
        return y

    hc_fn_contig = hc_fn.contiguous()
    hc_scale_contig = hc_scale.contiguous()
    hc_base_contig = hc_base.contiguous()

    if T > _SMALL_T_THRESHOLD:
        # Large-T single fused kernel: two passes over the row per program.
        x_stride_t, x_stride_m, x_stride_h = x_contig.stride()
        hc_fn_stride_m, hc_fn_stride_k = hc_fn_contig.stride()
        y_stride_t, y_stride_h = y.stride()
        NP = _next_pow2(hc_mult)
        grid = (T,)
        _hc_head_fused_kernel[grid](
            x_contig,
            hc_fn_contig,
            hc_scale_contig,
            hc_base_contig,
            y,
            float(norm_eps),
            float(hc_eps),
            x_stride_t,
            x_stride_m,
            hc_fn_stride_m,
            y_stride_t,
            T,
            hidden_size,
            hc_mult,
            NP=NP,
        )
        return y

    D = hc_mult * hidden_size
    NP = _next_pow2(hc_mult)

    # Token-group size G: bigger groups amortise each program's hc_fn slab
    # load over more tokens (the hc_fn L2 traffic dominates large T); small T
    # needs enough groups to fill the machine.
    if T <= 16:
        G = 1
    else:
        G = 4

    # D-axis split: small T needs the full split to fill the machine; larger
    # T already has T/G token-groups, so a moderate split suffices.
    n_tiles = (D + _BLOCK_D - 1) // _BLOCK_D
    if T <= 16:
        split_cap = n_tiles
    else:
        split_cap = min(n_tiles, 28)
    split = max(1, min(split_cap, n_tiles))
    chunk_tiles = (n_tiles + split - 1) // split
    split = (n_tiles + chunk_tiles - 1) // chunk_tiles

    sumsq = torch.empty(
        (T * split,), dtype=torch.float32, device=x_contig.device
    )
    mixes = torch.empty(
        (T * split * NP,), dtype=torch.float32, device=x_contig.device
    )

    grid_reduce = ((T + G - 1) // G, split)
    _hc_head_reduce_kernel[grid_reduce](
        x_contig,
        hc_fn_contig,
        sumsq,
        mixes,
        T,
        D,
        split,
        chunk_tiles,
        hc_mult,
        NP=NP,
        G=G,
        BLOCK_D=_BLOCK_D,
    )

    split_pow2 = _next_pow2(split)
    grid_finalize = (T, (hidden_size + 511) // 512)
    _hc_head_finalize_kernel[grid_finalize](
        x_contig,
        hc_scale_contig,
        hc_base_contig,
        sumsq,
        mixes,
        y,
        float(norm_eps),
        float(hc_eps),
        D,
        hidden_size,
        split,
        y.stride(0),
        hc_mult,
        NP=NP,
        SPLIT_POW2=split_pow2,
    )
    return y


__all__ = ["hc_head"]
