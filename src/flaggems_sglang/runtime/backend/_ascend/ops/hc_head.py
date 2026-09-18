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

"""Triton kernel for activation_norm/hc_head.

Fused DSV4 "hc_head" LM-head mixer: RMSNorm + linear mix + sigmoid gating +
weighted sum, folding the ``hc_mult`` axis of the multi-head codebook LM head
into a single ``hidden_size`` output vector per token. Matches
``sglang.kernels.ops.layernorm.mhc_head.fused_hc_head``.

Signature: ``hc_head(x, hc_fn, hc_scale, hc_base, norm_eps, hc_eps)``.

Math (all in float32, ``x`` upcast from its native dtype)::

    x_flat = flatten(x, dim=1)            # [T, hc_mult * hidden]
    r      = rsqrt(mean(x_flat**2) + norm_eps)        # [T, 1]
    mixes  = (x_flat @ hc_fn.T) * r        # [T, hc_mult]
    pre    = sigmoid(mixes * hc_scale + hc_base) + hc_eps   # [T, hc_mult]
    y      = sum_j pre[:,j] * x[:,j,:]    # [T, hidden]  (cast back to x.dtype)

Implementation
--------------
Two dispatch paths, selected by ``T``:

  * **Small / mid T (``T < BLOCK_M_THRESHOLD``)**: a single fused row kernel
    (``_hc_head_fused_row_kernel``), one program per token row, that does the
    whole op in one launch — RMSNorm + linear mix (elementwise multiply +
    ``tl.sum``) + sigmoid + weighted sum. Launch-/latency-bound regime.

  * **Large T (``T >= BLOCK_M_THRESHOLD``)**: two kernels.
      - **cast kernel** (``_hc_head_cast_fn_kernel``, bf16 ``x`` only): casts
        the fp32 ``hc_fn`` into an error-compensated bf16 pair
        ``hi = bf16(fn)``, ``lo = bf16(fn - hi)`` once per call (~30 us for
        the 458 KB matrix). The mix kernel then runs two native bf16 MMAs
        (``dot(x, hi) + dot(x, lo)``) whose sum reconstructs the fp32 product
        to ~fp32 accuracy — a single bf16 cast alone leaves ~9e-6 of output
        elements just outside the bf16 tolerance, while the compensated pair
        passes with margin. The pair still costs ~1.6x less than the hf32
        schedule (900 vs 1460 us at t8192) and halves the ``hc_fn`` tile
        traffic.
      - **mix kernel** (``_hc_head_mix_block_kernel``): matmul-style, each
        program handles ``BLOCK_M`` tokens. Loads the ``hc_fn`` hi/lo pair
        once per program and reuses it across the token block via ``tl.dot``
        of ``[BLOCK_M, BLOCK_K]`` x ``[BLOCK_K, HC_MULT_PAD]`` (``hc_mult``
        padded to the MMA minimum of 16). The RMSNorm ``sum_sq`` is
        accumulated in the same loop (shares the x tile load). When ``K``
        divides ``BLOCK_K`` evenly the per-iteration ``k`` mask is compiled
        out (``EVEN_K``). For bf16 ``x`` the operands are native bf16
        (``DOT_BF16``); fp32 sources use ``input_precision="ieee"`` and fp16
        sources ``"hf32"``. Writes the per-token ``[hc_mult]`` gates to a
        small fp32 scratch ``[T, hc_mult]``.
      - **weighted-sum kernel** (``_hc_head_sum_block_kernel``): a *block*
        of ``BLOCK_M`` token rows per program (``BLOCK_M=4`` keeps the
        ``[BLOCK_M, BLOCK_H]`` tile inside the 192 KB UB budget while giving
        the memory system 4 rows' worth of contiguous streaming loads per
        program — one-row programs were measurably slower). Reads the
        precomputed ``pre`` gates once as a ``[BLOCK_M, HC_MULT]`` vector and
        does ``y[m, h] = sum_j pre[m, j] * x[m, j, h]`` over the hidden axis
        tiled by ``BLOCK_H``. Reaches ~1.0 TB/s effective bandwidth (the
        measured device copy ceiling is ~1.24 TB/s).

Notes from on-device tuning (Ascend 910B4, portable Triton):

  * ``hf32`` is the only competitive fp32 MMA input precision here — ``bf16``
    / ``fp16`` ``tl.dot`` with *fp32 sources* run ~9x slower on this backend.
    But with **pre-cast bf16 sources** (fp32 ``hc_fn`` -> bf16 scratch, then a
    native bf16 MMA) the mix kernel runs ~1.6x faster than hf32; the extra
    rounding is within bf16 tolerance. fp16 pre-cast is both slower (~1.3x)
    and unnecessary.
  * ``BLOCK_M > 16`` with ``BLOCK_K >= 1024`` overflows the 192 KB UB
    ("ub overflow" from the vendor compiler); the usable tile envelope is
    narrow. With bf16 operands the sweep converged to ``BLOCK_M=32,
    BLOCK_K=512`` (900us at t8192) ahead of ``16x1024`` (916us) and
    ``64x256`` (915us).
  * split-K and elementwise (non-``tl.dot``) mix variants were both slower.
  * the weighted-sum kernel is bandwidth-bound; ``BLOCK_M=4`` per program
    beats one-row programs by ~15% (571us vs 687us at t8192).

Launch config is a pure function of the shapes (no ``@triton.autotune``
wrapper): its per-call dispatch hook would only add latency on the small-T
launch-bound shapes.

Device is taken from ``flaggems_sglang.device`` (never hardcoded). Pure
portable Triton: no vendor-private ops, no fallbacks, no module-level mutable
state.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

import flaggems_sglang  # noqa: F401  (resolves device/vendor at import)


def _hc_mult_pad(hc_mult: int) -> int:
    """Pad ``hc_mult`` up to the ``tl.dot`` MMA output minimum (16)."""
    return max(16, triton.next_power_of_2(hc_mult))


# ---------------------------------------------------------------------------
# Cast kernel — fp32 hc_fn -> bf16/fp16 scratch (one launch, ~20 us).
# The mix kernel then runs native low-precision MMA: on this backend a bf16
# x bf16 tl.dot is ~1.6x faster than the hf32 schedule, and the extra rounding
# on hc_fn stays well inside the bf16 output tolerance.
#
# Grid is 2D (row, column-tile): flat 1D offsets into a 2D output tensor
# silently drop writes past the first row on this backend, so index each
# hc_fn row explicitly through its stride.
# ---------------------------------------------------------------------------


@triton.jit
def _hc_head_cast_fn_kernel(
    fn_ptr,
    hi_ptr,
    lo_ptr,
    K,
    stride_fr,  # row stride of fn/hi/lo ([hc_mult, K] row-major)
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    off = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = off < K
    v = tl.load(fn_ptr + row * stride_fr + off, mask=mask)
    hi = v.to(tl.bfloat16)
    # error-compensated pair: fn ~= hi + lo to ~fp32 accuracy, so the mix
    # kernel's two bf16 MMAs reconstruct the fp32 product.
    lo = (v - hi.to(tl.float32)).to(tl.bfloat16)
    tl.store(hi_ptr + row * stride_fr + off, hi, mask=mask)
    tl.store(lo_ptr + row * stride_fr + off, lo, mask=mask)


# ---------------------------------------------------------------------------
# Mix kernel — matmul-style (large T): BLOCK_M tokens per program.
# ---------------------------------------------------------------------------


@triton.jit
def _hc_head_mix_block_kernel(
    x_ptr,
    hi_ptr,  # bf16 hc_fn hi part (DOT_BF16) / fp32 hc_fn (else)
    lo_ptr,  # bf16 hc_fn lo part (DOT_BF16), unused otherwise
    hc_scale_ptr,
    hc_base_ptr,
    pre_ptr,  # [T, HC_MULT] fp32 scratch
    T,
    HC_MULT: tl.constexpr,
    HIDDEN: tl.constexpr,
    HC_MULT_PAD: tl.constexpr,
    DOT_BF16: tl.constexpr,  # error-compensated bf16 dot pair (pre-cast fn)
    USE_IEEE: tl.constexpr,  # exact-ieee fp32 dot (fp32 x); else hf32
    EVEN_K: tl.constexpr,
    norm_eps,
    hc_eps,
    stride_xt,
    stride_fj,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    m_base = pid * BLOCK_M
    m_offs = m_base + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    m_mask = m_offs < T  # [BLOCK_M]
    m_rows = m_offs.to(tl.int64)  # [BLOCK_M] actual token rows

    K = HC_MULT * HIDDEN  # reduction length of the flattened row

    # Phase 1: matmul-style reduction over K.
    #   mixes[m, :] = (x_flat[m] @ hc_fn.T) * rsqrt(m)
    sum_sq = tl.zeros([BLOCK_M], dtype=tl.float32)
    mixes = tl.zeros([BLOCK_M, HC_MULT_PAD], dtype=tl.float32)

    j_pad = tl.arange(0, HC_MULT_PAD)  # [HC_MULT_PAD]
    j_mask = j_pad < HC_MULT  # which padded cols are real

    for k0 in range(0, K, BLOCK_K):
        k_offs = k0 + tl.arange(0, BLOCK_K)
        if EVEN_K:
            # k_mask compiles out when K % BLOCK_K == 0.
            xk = tl.load(
                x_ptr + m_rows[:, None] * stride_xt + k_offs[None, :],
                mask=m_mask[:, None],
                other=0.0,
            )
            fn_hi = tl.load(
                hi_ptr + j_pad[:, None] * stride_fj + k_offs[None, :],
                mask=j_mask[:, None],
                other=0.0,
            )
            if DOT_BF16:
                fn_lo = tl.load(
                    lo_ptr + j_pad[:, None] * stride_fj + k_offs[None, :],
                    mask=j_mask[:, None],
                    other=0.0,
                )
        else:
            k_mask = k_offs < K
            xk = tl.load(
                x_ptr + m_rows[:, None] * stride_xt + k_offs[None, :],
                mask=m_mask[:, None] & k_mask[None, :],
                other=0.0,
            )
            fn_hi = tl.load(
                hi_ptr + j_pad[:, None] * stride_fj + k_offs[None, :],
                mask=j_mask[:, None] & k_mask[None, :],
                other=0.0,
            )
            if DOT_BF16:
                fn_lo = tl.load(
                    lo_ptr + j_pad[:, None] * stride_fj + k_offs[None, :],
                    mask=j_mask[:, None] & k_mask[None, :],
                    other=0.0,
                )
        xb = xk.to(tl.float32)
        sum_sq += tl.sum(xb * xb, axis=1)  # [BLOCK_M]

        # w = hc_fn.T -> [BLOCK_K, HC_MULT_PAD]
        w = tl.trans(fn_hi)

        if DOT_BF16:
            xb16 = xk.to(tl.bfloat16)
            mixes = tl.dot(xb16, tl.trans(fn_lo), acc=mixes)
            mixes = tl.dot(xb16, w, acc=mixes)
        elif USE_IEEE:
            mixes += tl.dot(xb, w.to(tl.float32), input_precision="ieee")
        else:
            mixes += tl.dot(xb, w.to(tl.float32), input_precision="hf32")

    rsqrt = 1.0 / tl.sqrt(sum_sq / K + norm_eps)  # [BLOCK_M]
    mixes = mixes * rsqrt[:, None]  # [BLOCK_M, HC_MULT_PAD]

    scale = tl.load(hc_scale_ptr + tl.arange(0, 1))  # [1]
    base = tl.load(hc_base_ptr + j_pad, mask=j_mask, other=0.0).to(tl.float32)
    pre = tl.sigmoid(mixes * scale + base) + hc_eps  # [BLOCK_M, HC_MULT_PAD]

    # Write the real hc_mult columns to the scratch buffer (padded cols masked
    # out). pre_ptr layout: [T, HC_MULT] (row-major); store over the padded
    # arange, masked to real columns and valid rows.
    tl.store(
        pre_ptr + m_rows[:, None] * HC_MULT + j_pad[None, :],
        pre,
        mask=m_mask[:, None] & j_mask[None, :],
    )


# ---------------------------------------------------------------------------
# Weighted-sum kernel — a block of token rows per program:
#   y[m, h] = sum_j pre[m, j] * x[m, j, h]
# ---------------------------------------------------------------------------


@triton.jit
def _hc_head_sum_block_kernel(
    x_ptr,
    pre_ptr,  # [T, HC_MULT] fp32
    y_ptr,
    T,
    HC_MULT: tl.constexpr,
    HIDDEN: tl.constexpr,
    stride_xt,
    BLOCK_M: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    pid = tl.program_id(0)
    m_offs = pid * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    m_mask = m_offs < T
    m_rows = m_offs.to(tl.int64)

    j_idx = tl.arange(0, HC_MULT)  # [HC_MULT]
    # pre[m, j] gates for the whole row block, loaded once.
    pre = tl.load(
        pre_ptr + m_rows[:, None] * HC_MULT + j_idx[None, :],
        mask=m_mask[:, None],
        other=0.0,
    )  # [BLOCK_M, HC_MULT] fp32

    for h0 in range(0, HIDDEN, BLOCK_H):
        h_offs = h0 + tl.arange(0, BLOCK_H)
        h_mask = h_offs < HIDDEN

        acc = tl.zeros([BLOCK_M, BLOCK_H], dtype=tl.float32)
        for j in tl.static_range(HC_MULT):
            # pre[:, j] -> [BLOCK_M] (select column j from the loaded vector)
            pj = tl.sum(tl.where(j_idx[None, :] == j, pre, 0.0), axis=1)
            # x[m, j, h0:h0+BLOCK_H] -> [BLOCK_M, BLOCK_H]
            x_block = tl.load(
                x_ptr
                + m_rows[:, None] * stride_xt
                + j * HIDDEN
                + h_offs[None, :],
                mask=m_mask[:, None] & h_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            acc += pj[:, None] * x_block

        tl.store(
            y_ptr + m_rows[:, None] * HIDDEN + h_offs[None, :],
            acc,
            mask=m_mask[:, None] & h_mask[None, :],
        )


# ---------------------------------------------------------------------------
# Fused single-kernel — per token row (small / mid T).
# One program per token row; mixes RMSNorm + linear mix + sigmoid + weighted
# sum in a single launch (no two-launch overhead). Used when T is small/mid,
# where the per-call launch cost of the two-kernel split dominates.
# ---------------------------------------------------------------------------


@triton.jit
def _hc_head_fused_row_kernel(
    x_ptr,
    hc_fn_ptr,
    hc_scale_ptr,
    hc_base_ptr,
    y_ptr,
    T,
    HC_MULT: tl.constexpr,
    HIDDEN: tl.constexpr,
    norm_eps,
    hc_eps,
    stride_xt,
    stride_fj,
    BLOCK_K: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    row = tl.program_id(0)
    if row >= T:
        return

    K = HC_MULT * HIDDEN  # reduction length of the flattened row
    x_row = x_ptr + row * stride_xt  # start of x[t].flatten() (contiguous)

    # Phase 1: one pass over the flat row -> rsqrt + mixes (-> pre)
    sum_sq = tl.zeros((), dtype=tl.float32)
    mixes = tl.zeros([HC_MULT], dtype=tl.float32)

    j_idx = tl.arange(0, HC_MULT)  # [HC_MULT]

    for k0 in range(0, K, BLOCK_K):
        k_offs = k0 + tl.arange(0, BLOCK_K)
        k_mask = k_offs < K

        xk = tl.load(x_row + k_offs, mask=k_mask, other=0.0).to(tl.float32)
        sum_sq += tl.sum(xk * xk)

        fn = tl.load(
            hc_fn_ptr + j_idx[:, None] * stride_fj + k_offs[None, :],
            mask=k_mask[None, :],
            other=0.0,
        ).to(
            tl.float32
        )  # [HC_MULT, BLOCK_K]
        mixes += tl.sum(fn * xk[None, :], axis=1)  # [HC_MULT]

    rsqrt = 1.0 / tl.sqrt(sum_sq / K + norm_eps)
    mixes = mixes * rsqrt  # [HC_MULT]

    scale = tl.load(hc_scale_ptr + tl.arange(0, 1))  # [1]
    base = tl.load(hc_base_ptr + j_idx).to(tl.float32)  # [HC_MULT]
    pre = tl.sigmoid(mixes * scale + base) + hc_eps  # [HC_MULT]

    # Phase 2: weighted sum over the hidden axis -> y[t, :]
    for h0 in range(0, HIDDEN, BLOCK_H):
        h_offs = h0 + tl.arange(0, BLOCK_H)
        h_mask = h_offs < HIDDEN

        x_block = tl.load(
            x_row + j_idx[:, None] * HIDDEN + h_offs[None, :],
            mask=h_mask[None, :],
            other=0.0,
        ).to(tl.float32)

        y_block = tl.sum(pre[:, None] * x_block, axis=0)  # [BLOCK_H]
        tl.store(y_ptr + row * HIDDEN + h_offs, y_block, mask=h_mask)


def hc_head(x, hc_fn, hc_scale, hc_base, norm_eps, hc_eps):
    # Shapes / dtype.
    T, hc_mult, hidden = x.shape
    dtype = x.dtype
    K = hc_mult * hidden
    assert hc_fn.shape == (
        hc_mult,
        K,
    ), f"hc_fn expected shape ({hc_mult}, {K}), got {tuple(hc_fn.shape)}"

    y = torch.empty((T, hidden), dtype=dtype, device=x.device)

    hc_mult_pad = _hc_mult_pad(hc_mult)
    # float32 source needs exact-ieee matmul; bf16/fp16 source already lost
    # mantissa bits, so hf32's faster MMA schedule is within tolerance.
    use_ieee = dtype == torch.float32

    # Threshold: below it the per-call launch cost of the two-kernel split
    # (mix + weighted-sum) dominates, so a single fused row kernel wins; above
    # it the bandwidth saving from the tl.dot block mix (amortising the hc_fn
    # load across BLOCK_M rows) dominates and the two launches are cheap.
    BLOCK_M_THRESHOLD = 512

    if T < BLOCK_M_THRESHOLD:
        # Fused single kernel, one program per token row. Launch config as a
        # function of T: small T (launch-bound) wants more warps; mid T
        # (still mostly launch/latency-bound at the row granularity) a wider
        # reduction tile and fewer warps.
        if T <= 1:
            f_BLOCK_K = 4096 if K >= 4096 else (1024 if K >= 1024 else 512)
            f_BLOCK_H = 1024 if hidden >= 1024 else 256
            f_warps = 8
            f_stages = 2
        else:
            # Single tuned config: W=8/S=2 measured best-or-equal at every
            # small/mid T (the earlier W=4/S=4 mid-T variant was within noise).
            f_BLOCK_K = 4096 if K >= 8192 else (1024 if K >= 4096 else 512)
            f_BLOCK_H = 1024 if hidden >= 1024 else 256
            f_warps = 8
            f_stages = 2
        grid = (T,)
        _hc_head_fused_row_kernel[grid](
            x,
            hc_fn,
            hc_scale,
            hc_base,
            y,
            T,
            HC_MULT=hc_mult,
            HIDDEN=hidden,
            norm_eps=norm_eps,
            hc_eps=hc_eps,
            stride_xt=x.stride(0),
            stride_fj=hc_fn.stride(0),
            BLOCK_K=f_BLOCK_K,
            BLOCK_H=f_BLOCK_H,
            num_warps=f_warps,
            num_stages=f_stages,
        )
    else:
        # Two-kernel path: matmul-style block mix + block weighted sum.
        # Small fp32 scratch for the per-token gates [T, hc_mult].
        pre = torch.empty((T, hc_mult), dtype=torch.float32, device=x.device)

        # For bf16 x, pre-cast the fp32 hc_fn (only ~hc_mult*K*4 B) into an
        # error-compensated bf16 pair (hi + lo): two native bf16 MMAs
        # reconstruct the fp32 product at ~1.6x lower cost than the hf32
        # schedule on this backend. A single bf16 cast is faster still but
        # leaves a sliver of elements outside the bf16 tolerance.
        # fp16/fp32 x keep the fp32 dot (hf32 / ieee respectively).
        dot_bf16 = dtype == torch.bfloat16
        if dot_bf16:
            fn_hi = torch.empty(
                hc_fn.shape, dtype=torch.bfloat16, device=x.device
            )
            fn_lo = torch.empty(
                hc_fn.shape, dtype=torch.bfloat16, device=x.device
            )
            _hc_head_cast_fn_kernel[(hc_mult, (K + 4095) // 4096)](
                hc_fn,
                fn_hi,
                fn_lo,
                K,
                stride_fr=hc_fn.stride(0),
                BLOCK=4096,
                num_warps=8,
            )
        else:
            fn_hi = hc_fn
            fn_lo = hc_fn

        # bf16 operands shift the sweet spot: BM=32/BK=512 edges out 16x1024
        # (900 vs 916 us at t8192); fp32 operands stay at the 16-row hf32/
        # ieee envelope (BM=32/BK>=1024 overflows UB). Mid T (t2048) prefers
        # the wider tile BM=64/BK=256 (232 vs 240 us) — more token rows per
        # hc_fn load while the MMA pipe is still not saturated.
        if dot_bf16:
            if T <= 4096:
                BLOCK_M = 64
                b_BLOCK_K = 256
            else:
                BLOCK_M = 32
                b_BLOCK_K = 512 if K >= 512 else 256
        else:
            BLOCK_M = 16
            b_BLOCK_K = 1024 if K >= 1024 else 512
        even_k = (K % b_BLOCK_K) == 0
        b_BLOCK_H = 1024 if hidden >= 1024 else 256
        # Weighted sum: 4 rows per program keeps the [BLOCK_M, BLOCK_H] fp32
        # accumulator inside UB while streaming 4 contiguous rows per program.
        s_BLOCK_M = 4

        num_m_tiles = (T + BLOCK_M - 1) // BLOCK_M
        grid_mix = (num_m_tiles,)
        _hc_head_mix_block_kernel[grid_mix](
            x,
            fn_hi,
            fn_lo,
            hc_scale,
            hc_base,
            pre,
            T,
            HC_MULT=hc_mult,
            HIDDEN=hidden,
            HC_MULT_PAD=hc_mult_pad,
            DOT_BF16=dot_bf16,
            USE_IEEE=use_ieee,
            EVEN_K=even_k,
            norm_eps=norm_eps,
            hc_eps=hc_eps,
            stride_xt=x.stride(0),
            stride_fj=fn_hi.stride(0),
            BLOCK_M=BLOCK_M,
            BLOCK_K=b_BLOCK_K,
            num_warps=8,
            num_stages=2,
        )

        grid_sum = ((T + s_BLOCK_M - 1) // s_BLOCK_M,)
        _hc_head_sum_block_kernel[grid_sum](
            x,
            pre,
            y,
            T,
            HC_MULT=hc_mult,
            HIDDEN=hidden,
            stride_xt=x.stride(0),
            BLOCK_M=s_BLOCK_M,
            BLOCK_H=b_BLOCK_H,
            num_warps=4,
            num_stages=2,
        )
    return y


__all__ = ["hc_head"]
