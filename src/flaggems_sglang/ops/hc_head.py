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

"""Fused ``hc_head`` mixer with per-``hc_mult`` specializations in pure Triton.

Per token ``t`` (all arithmetic in float32, matching the PyTorch reference):

  1. RMSNorm over the flattened ``[hc_mult, hidden_size]`` row:
     ``r = rsqrt(mean(x_flat**2) + norm_eps)``
  2. Linear mix:  ``mixes = (x_flat @ hc_fn.T) * r``  -> ``[hc_mult]``
  3. Sigmoid gate:``pre    = sigmoid(mixes*hc_scale + hc_base) + hc_eps`` -> ``[hc_mult]``
  4. Weighted sum:``y[t,h] = sum_j pre[j] * x[t,j,h]``  -> ``[hidden_size]``

Output is cast back to ``x``'s dtype.

Two dispatch paths are selected by ``hc_mult``:

* **hc_mult == 2 / hc_mult == 4** (the shapes the platform actually runs):
  a **single** fused kernel per specialisation, ``_hc_head_fused_hc2_kernel``
  / ``_hc_head_fused_hc4_kernel`` (``grid = (T,)``), doing two passes over
  the token's row in one program:
    - pass 1 streams the flat ``D = hc_mult*hidden_size`` row once; each
      iteration loads one ``[BLOCK]`` slice per ``x`` row (bf16 -> fp32) and
      the matching ``[BLOCK]`` slice of each ``hc_fn`` row, accumulating the
      partial sum-of-squares and the unrolled per-head mix dot-products in
      registers;
    - after the loop it computes ``r`` / ``pre`` and streams the hidden axis
      once more, writing ``y[t,h] = sum_j pre[j]*x[t,j,h]``.

  Unrolling the ``hc_mult`` mix accumulators keeps every intermediate in
  registers — no partials round-trip, no atomics, and each ``hc_fn`` element
  is read exactly once per token.

* **Any other ``hc_mult``**: a two-kernel generic pipeline.
    - ``_hc_head_pre_kernel`` — grid ``(T, hc_mult)``. Each program reduces
      one token's flat row against one ``hc_fn`` row and writes the sigmoid
      gate ``pre[t, j]``.
    - ``_hc_head_mix_kernel`` — grid ``(T, ceil(hidden/BLOCK))``. Every
      program streams one ``BLOCK`` slice of the hidden axis and accumulates
      ``y[t, h] = sum_j pre[j]*x[t, j, h]``.

``BLOCK`` sizes are picked from ``hidden_size`` / the flat ``D`` at launch
time (capped at ``_MAX_BLOCK``); ``num_warps`` / ``num_stages`` are handed
to Triton at launch — no autotune, no hand-rolled global dicts anywhere in
this module.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

# Largest tile processed per inner iteration. Keeps register pressure and
# shared-memory use predictable across hidden sizes while staying coalesced.
_MAX_BLOCK = 1024

# Above this hidden size the per-token two-pass kernel needs more warps to
# hide the latency of the second streaming pass.
_WIDE_HIDDEN_THRESHOLD = 2048


# ---------------------------------------------------------------------------
# Fused path (hc_mult == 2): single kernel, two passes over the token row.
# ---------------------------------------------------------------------------


@triton.jit
def _hc_head_fused_hc2_kernel(
    x_ptr,  # x: [T, 2, hidden_size] (contiguous)
    hc_fn_ptr,  # hc_fn: [2, 2*hidden_size] (contiguous, fp32)
    hc_scale_ptr,  # hc_scale: [1]
    hc_base_ptr,  # hc_base: [2]
    y_ptr,  # y: [T, hidden_size]
    norm_eps,  # scalar (float)
    hc_eps,  # scalar (float)
    x_stride_t,
    x_stride_m,
    hc_fn_stride_m,
    y_stride_t,
    hidden_size: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """One program per token (hc_mult == 2 fused path).

    Pass 1 streams the token's flat ``D = 2*hidden_size`` row in ``[BLOCK]``
    slices, accumulating the sum-of-squares and the two unrolled mix
    dot-products fully in registers. Then ``r`` / ``pre`` are computed and
    pass 2 streams the hidden axis once to emit
    ``y[t,h] = pre0*x[t,0,h] + pre1*x[t,1,h]``.
    """
    t = tl.program_id(0)

    x_base = t * x_stride_t

    # ---- pass 1: sum-of-squares + linear-mix partial dots ----
    ssq_acc = tl.zeros((BLOCK,), tl.float32)
    mix_acc0 = tl.zeros((BLOCK,), tl.float32)
    mix_acc1 = tl.zeros((BLOCK,), tl.float32)
    for h0 in range(0, hidden_size, BLOCK):
        h_offs = h0 + tl.arange(0, BLOCK)
        h_mask = h_offs < hidden_size
        x0 = tl.load(
            x_ptr + x_base + 0 * x_stride_m + h_offs, mask=h_mask, other=0.0
        ).to(tl.float32)
        x1 = tl.load(
            x_ptr + x_base + 1 * x_stride_m + h_offs, mask=h_mask, other=0.0
        ).to(tl.float32)
        ssq_acc += x0 * x0 + x1 * x1
        w00 = tl.load(
            hc_fn_ptr + 0 * hc_fn_stride_m + 0 * hidden_size + h_offs,
            mask=h_mask,
            other=0.0,
        ).to(tl.float32)
        w01 = tl.load(
            hc_fn_ptr + 0 * hc_fn_stride_m + 1 * hidden_size + h_offs,
            mask=h_mask,
            other=0.0,
        ).to(tl.float32)
        mix_acc0 += x0 * w00 + x1 * w01
        w10 = tl.load(
            hc_fn_ptr + 1 * hc_fn_stride_m + 0 * hidden_size + h_offs,
            mask=h_mask,
            other=0.0,
        ).to(tl.float32)
        w11 = tl.load(
            hc_fn_ptr + 1 * hc_fn_stride_m + 1 * hidden_size + h_offs,
            mask=h_mask,
            other=0.0,
        ).to(tl.float32)
        mix_acc1 += x0 * w10 + x1 * w11

    sum_sq = tl.sum(ssq_acc, axis=0)
    rsqrt = tl.rsqrt(sum_sq / (2 * hidden_size) + norm_eps)

    hc_scale = tl.load(hc_scale_ptr).to(tl.float32)
    pre0 = (
        tl.sigmoid(
            tl.sum(mix_acc0, axis=0) * rsqrt * hc_scale
            + tl.load(hc_base_ptr + 0).to(tl.float32)
        )
        + hc_eps
    )
    pre1 = (
        tl.sigmoid(
            tl.sum(mix_acc1, axis=0) * rsqrt * hc_scale
            + tl.load(hc_base_ptr + 1).to(tl.float32)
        )
        + hc_eps
    )

    # ---- pass 2: weighted sum over the hc_mult axis ----
    for h0 in range(0, hidden_size, BLOCK):
        h_offs = h0 + tl.arange(0, BLOCK)
        h_mask = h_offs < hidden_size
        x0 = tl.load(
            x_ptr + x_base + 0 * x_stride_m + h_offs, mask=h_mask, other=0.0
        ).to(tl.float32)
        x1 = tl.load(
            x_ptr + x_base + 1 * x_stride_m + h_offs, mask=h_mask, other=0.0
        ).to(tl.float32)
        y_acc = pre0 * x0 + pre1 * x1
        tl.store(
            y_ptr + t * y_stride_t + h_offs,
            y_acc.to(y_ptr.dtype.element_ty),
            mask=h_mask,
        )


# ---------------------------------------------------------------------------
# Fused path (hc_mult == 4): single kernel, two passes over the token row.
# ---------------------------------------------------------------------------


@triton.jit
def _hc_head_fused_hc4_kernel(
    x_ptr,  # x: [T, 4, hidden_size] (contiguous)
    hc_fn_ptr,  # hc_fn: [4, 4*hidden_size] (contiguous, fp32)
    hc_scale_ptr,  # hc_scale: [1]
    hc_base_ptr,  # hc_base: [4]
    y_ptr,  # y: [T, hidden_size]
    norm_eps,  # scalar (float)
    hc_eps,  # scalar (float)
    x_stride_t,
    x_stride_m,
    hc_fn_stride_m,
    y_stride_t,
    hidden_size: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """One program per token (hc_mult == 4 fused path).

    Same two-pass structure as ``_hc_head_fused_hc2_kernel`` with the four
    mix accumulators unrolled.
    """
    t = tl.program_id(0)

    x_base = t * x_stride_t

    # ---- pass 1: sum-of-squares + linear-mix partial dots ----
    ssq_acc = tl.zeros((BLOCK,), tl.float32)
    mix_acc0 = tl.zeros((BLOCK,), tl.float32)
    mix_acc1 = tl.zeros((BLOCK,), tl.float32)
    mix_acc2 = tl.zeros((BLOCK,), tl.float32)
    mix_acc3 = tl.zeros((BLOCK,), tl.float32)
    for h0 in range(0, hidden_size, BLOCK):
        h_offs = h0 + tl.arange(0, BLOCK)
        h_mask = h_offs < hidden_size
        x0 = tl.load(
            x_ptr + x_base + 0 * x_stride_m + h_offs, mask=h_mask, other=0.0
        ).to(tl.float32)
        x1 = tl.load(
            x_ptr + x_base + 1 * x_stride_m + h_offs, mask=h_mask, other=0.0
        ).to(tl.float32)
        x2 = tl.load(
            x_ptr + x_base + 2 * x_stride_m + h_offs, mask=h_mask, other=0.0
        ).to(tl.float32)
        x3 = tl.load(
            x_ptr + x_base + 3 * x_stride_m + h_offs, mask=h_mask, other=0.0
        ).to(tl.float32)
        ssq_acc += x0 * x0 + x1 * x1 + x2 * x2 + x3 * x3
        w00 = tl.load(
            hc_fn_ptr + 0 * hc_fn_stride_m + 0 * hidden_size + h_offs,
            mask=h_mask,
            other=0.0,
        ).to(tl.float32)
        w01 = tl.load(
            hc_fn_ptr + 0 * hc_fn_stride_m + 1 * hidden_size + h_offs,
            mask=h_mask,
            other=0.0,
        ).to(tl.float32)
        w02 = tl.load(
            hc_fn_ptr + 0 * hc_fn_stride_m + 2 * hidden_size + h_offs,
            mask=h_mask,
            other=0.0,
        ).to(tl.float32)
        w03 = tl.load(
            hc_fn_ptr + 0 * hc_fn_stride_m + 3 * hidden_size + h_offs,
            mask=h_mask,
            other=0.0,
        ).to(tl.float32)
        mix_acc0 += x0 * w00 + x1 * w01 + x2 * w02 + x3 * w03
        w10 = tl.load(
            hc_fn_ptr + 1 * hc_fn_stride_m + 0 * hidden_size + h_offs,
            mask=h_mask,
            other=0.0,
        ).to(tl.float32)
        w11 = tl.load(
            hc_fn_ptr + 1 * hc_fn_stride_m + 1 * hidden_size + h_offs,
            mask=h_mask,
            other=0.0,
        ).to(tl.float32)
        w12 = tl.load(
            hc_fn_ptr + 1 * hc_fn_stride_m + 2 * hidden_size + h_offs,
            mask=h_mask,
            other=0.0,
        ).to(tl.float32)
        w13 = tl.load(
            hc_fn_ptr + 1 * hc_fn_stride_m + 3 * hidden_size + h_offs,
            mask=h_mask,
            other=0.0,
        ).to(tl.float32)
        mix_acc1 += x0 * w10 + x1 * w11 + x2 * w12 + x3 * w13
        w20 = tl.load(
            hc_fn_ptr + 2 * hc_fn_stride_m + 0 * hidden_size + h_offs,
            mask=h_mask,
            other=0.0,
        ).to(tl.float32)
        w21 = tl.load(
            hc_fn_ptr + 2 * hc_fn_stride_m + 1 * hidden_size + h_offs,
            mask=h_mask,
            other=0.0,
        ).to(tl.float32)
        w22 = tl.load(
            hc_fn_ptr + 2 * hc_fn_stride_m + 2 * hidden_size + h_offs,
            mask=h_mask,
            other=0.0,
        ).to(tl.float32)
        w23 = tl.load(
            hc_fn_ptr + 2 * hc_fn_stride_m + 3 * hidden_size + h_offs,
            mask=h_mask,
            other=0.0,
        ).to(tl.float32)
        mix_acc2 += x0 * w20 + x1 * w21 + x2 * w22 + x3 * w23
        w30 = tl.load(
            hc_fn_ptr + 3 * hc_fn_stride_m + 0 * hidden_size + h_offs,
            mask=h_mask,
            other=0.0,
        ).to(tl.float32)
        w31 = tl.load(
            hc_fn_ptr + 3 * hc_fn_stride_m + 1 * hidden_size + h_offs,
            mask=h_mask,
            other=0.0,
        ).to(tl.float32)
        w32 = tl.load(
            hc_fn_ptr + 3 * hc_fn_stride_m + 2 * hidden_size + h_offs,
            mask=h_mask,
            other=0.0,
        ).to(tl.float32)
        w33 = tl.load(
            hc_fn_ptr + 3 * hc_fn_stride_m + 3 * hidden_size + h_offs,
            mask=h_mask,
            other=0.0,
        ).to(tl.float32)
        mix_acc3 += x0 * w30 + x1 * w31 + x2 * w32 + x3 * w33

    sum_sq = tl.sum(ssq_acc, axis=0)
    rsqrt = tl.rsqrt(sum_sq / (4 * hidden_size) + norm_eps)

    hc_scale = tl.load(hc_scale_ptr).to(tl.float32)
    pre0 = (
        tl.sigmoid(
            tl.sum(mix_acc0, axis=0) * rsqrt * hc_scale
            + tl.load(hc_base_ptr + 0).to(tl.float32)
        )
        + hc_eps
    )
    pre1 = (
        tl.sigmoid(
            tl.sum(mix_acc1, axis=0) * rsqrt * hc_scale
            + tl.load(hc_base_ptr + 1).to(tl.float32)
        )
        + hc_eps
    )
    pre2 = (
        tl.sigmoid(
            tl.sum(mix_acc2, axis=0) * rsqrt * hc_scale
            + tl.load(hc_base_ptr + 2).to(tl.float32)
        )
        + hc_eps
    )
    pre3 = (
        tl.sigmoid(
            tl.sum(mix_acc3, axis=0) * rsqrt * hc_scale
            + tl.load(hc_base_ptr + 3).to(tl.float32)
        )
        + hc_eps
    )

    # ---- pass 2: weighted sum over the hc_mult axis ----
    for h0 in range(0, hidden_size, BLOCK):
        h_offs = h0 + tl.arange(0, BLOCK)
        h_mask = h_offs < hidden_size
        x0 = tl.load(
            x_ptr + x_base + 0 * x_stride_m + h_offs, mask=h_mask, other=0.0
        ).to(tl.float32)
        x1 = tl.load(
            x_ptr + x_base + 1 * x_stride_m + h_offs, mask=h_mask, other=0.0
        ).to(tl.float32)
        x2 = tl.load(
            x_ptr + x_base + 2 * x_stride_m + h_offs, mask=h_mask, other=0.0
        ).to(tl.float32)
        x3 = tl.load(
            x_ptr + x_base + 3 * x_stride_m + h_offs, mask=h_mask, other=0.0
        ).to(tl.float32)
        y_acc = pre0 * x0 + pre1 * x1 + pre2 * x2 + pre3 * x3
        tl.store(
            y_ptr + t * y_stride_t + h_offs,
            y_acc.to(y_ptr.dtype.element_ty),
            mask=h_mask,
        )


# ---------------------------------------------------------------------------
# Generic path: two-kernel pipeline — gate pre-computation, then mix.
# ---------------------------------------------------------------------------


@triton.jit
def _hc_head_pre_kernel(
    x_ptr,  # x: [T, D] view of [T, hc_mult, hidden_size] (contiguous)
    hc_fn_ptr,  # hc_fn: [hc_mult, D] (contiguous, fp32)
    hc_scale_ptr,  # hc_scale: [1]
    hc_base_ptr,  # hc_base: [hc_mult]
    pre_ptr,  # pre: [T, hc_mult] (fp32)
    norm_eps,  # scalar (float)
    hc_eps,  # scalar (float)
    D: tl.constexpr,  # hc_mult * hidden_size
    hc_mult: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """One program per ``(token, hc_fn row)``."""
    t = tl.program_id(0)
    j = tl.program_id(1)
    x_base = t * D
    fn_base = j * D
    ssq_acc = tl.zeros((BLOCK,), tl.float32)
    dot_acc = tl.zeros((BLOCK,), tl.float32)
    for d0 in range(0, D, BLOCK):
        offs = d0 + tl.arange(0, BLOCK)
        d_mask = offs < D
        x_vec = tl.load(x_ptr + x_base + offs, mask=d_mask, other=0.0).to(
            tl.float32
        )
        w_vec = tl.load(hc_fn_ptr + fn_base + offs, mask=d_mask, other=0.0).to(
            tl.float32
        )
        ssq_acc += x_vec * x_vec
        dot_acc += x_vec * w_vec
    inv_rms = tl.rsqrt(tl.sum(ssq_acc, axis=0) / D + norm_eps)
    hc_scale = tl.load(hc_scale_ptr).to(tl.float32)
    hc_base = tl.load(hc_base_ptr + j).to(tl.float32)
    pre = (
        tl.sigmoid(tl.sum(dot_acc, axis=0) * inv_rms * hc_scale + hc_base)
        + hc_eps
    )
    tl.store(pre_ptr + t * hc_mult + j, pre)


@triton.jit
def _hc_head_mix_kernel(
    x_ptr,  # x: [T, hc_mult, hidden_size] (contiguous)
    pre_ptr,  # pre: [T, hc_mult] (fp32)
    y_ptr,  # y: [T, hidden_size]
    y_stride_t,
    hidden_size: tl.constexpr,
    hc_mult: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """One program per ``(token, hidden-tile)``."""
    t = tl.program_id(0)
    h_offs = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    h_mask = h_offs < hidden_size
    y_acc = tl.zeros((BLOCK,), tl.float32)
    for j in range(hc_mult):
        p_j = tl.load(pre_ptr + t * hc_mult + j)
        x_vec = tl.load(
            x_ptr + (t * hc_mult + j) * hidden_size + h_offs,
            mask=h_mask,
            other=0.0,
        ).to(tl.float32)
        y_acc += p_j * x_vec
    tl.store(
        y_ptr + t * y_stride_t + h_offs,
        y_acc.to(y_ptr.dtype.element_ty),
        mask=h_mask,
    )


def _next_pow2(n: int, cap: int) -> int:
    p = 1
    while p < n and p < cap:
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

    num_warps = 8 if hidden_size >= _WIDE_HIDDEN_THRESHOLD else 4

    if hc_mult == 2:
        # Fused single kernel: two passes over the row per program.
        x_stride_t, x_stride_m, x_stride_h = x_contig.stride()
        hc_fn_stride_m, hc_fn_stride_k = hc_fn_contig.stride()
        _hc_head_fused_hc2_kernel[(T,)](
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
            y.stride(0),
            hidden_size=hidden_size,
            BLOCK=_next_pow2(hidden_size, _MAX_BLOCK),
            num_warps=num_warps,
            num_stages=2,
        )
        return y

    if hc_mult == 4:
        # Fused single kernel: two passes over the row per program.
        x_stride_t, x_stride_m, x_stride_h = x_contig.stride()
        hc_fn_stride_m, hc_fn_stride_k = hc_fn_contig.stride()
        _hc_head_fused_hc4_kernel[(T,)](
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
            y.stride(0),
            hidden_size=hidden_size,
            BLOCK=_next_pow2(hidden_size, _MAX_BLOCK),
            num_warps=num_warps,
            num_stages=2,
        )
        return y

    # Generic two-kernel pipeline for hc_mult values without a fused
    # specialisation.
    D = hc_mult * hidden_size
    block = _next_pow2(hidden_size, _MAX_BLOCK)
    block_d = _next_pow2(D, _MAX_BLOCK)
    pre = torch.empty(
        (T, hc_mult), dtype=torch.float32, device=x_contig.device
    )
    _hc_head_pre_kernel[(T, hc_mult)](
        x_contig,
        hc_fn_contig,
        hc_scale_contig,
        hc_base_contig,
        pre,
        float(norm_eps),
        float(hc_eps),
        D=D,
        hc_mult=hc_mult,
        BLOCK=block_d,
        num_warps=num_warps,
        num_stages=2,
    )
    _hc_head_mix_kernel[(T, (hidden_size + block - 1) // block)](
        x_contig,
        pre,
        y,
        y.stride(0),
        hidden_size=hidden_size,
        hc_mult=hc_mult,
        BLOCK=block,
        num_warps=num_warps,
        num_stages=2,
    )
    return y


__all__ = ["hc_head"]
