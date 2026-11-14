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

"""Triton implementation of the fused MoE router (tensorcore variant).

2026-11-13 round-3 large-tier RSI (E-1 residual 0.78x/0.92x)
--------------------------------------------------------------------
Stage attribution (do_bench median, E=256/H=4096/bf16) showed the two-kernel
split's costs at the failing tiers are NOT GEMM-bound: at m=4096 the post
kernel cost 104.3us of the 172.8us total (GEMM 71.2us is already at the
~121 TFLOPS MMA ceiling), and at m=512 the post was 22.0us of 53.0us. The
ablation ladder pinned the whole post cost on the two
``tl.max(..., return_indices=True)`` index-tracked reductions: a kernel with
plain load/max/exp-sum stores at 18.5us == the load-store floor, while adding
index tracking + the penalised second max costs +86us (m=4096).

* ``_post_kernel`` is rewritten in the same 2-D tile structure but with the
  index-free selection algebra (sibling op ``fused_moe_router_cudacore``
  prior art): plain max for the row max, ``tl.min(tl.where(x == max, offs,
  E))`` for the leftmost tied index, a single ``where`` penalise, plain max
  again, same min-where for the runner-up. Weights keep the exact closed
  forms, so outputs are bit-identical to the previous form (verified over 22
  shape/tie/edge cases including real GEMM logits). An ``M_ALIGNED`` fast
  path drops the row mask when ``M % ROW_TILE == 0`` and E needs no pad.
  Measured post: m=512 21.9->8.7us, m=1024 36.4->10.2us, m=2048 57.4->12.9,
  m=4096 104.3->19.4us (5.4x). Config: ROW_TILE = next_pow2(max(M/32, 8))
  capped at 128, num_warps 2 (<32 rows) / 4 (>=32 rows) — the probe sweep's
  per-M optima; the 1-D per-row arm keeps M < 64 (it still wins there).
  A flat 1-D load + ``reshape`` post variant does not compile on this
  backend (probe6, P01a-family reshape wall) — direction closed.
* ``_KLOOP_CFGS`` gains two mid-M occupancy rows (probe5/6 GEMM-only
  medians): 256<=m<=512 uses (rt64, be64, bk1024, nw4, st4) = 23.8us at
  m=512 (was 32.7 — 8 programs on 48 SPCs were the limit) and 512<m<=1024
  uses (rt128, be64, bk1024, nw4, st4) = 31.3us at m=1024 (was 55.7).
  1024<m<=2048 moves to (rt256, be128, bk1024, nw8, st4) = 52.0us at m=2048
  (was 58.6); m>2048 keeps (rt256, be128, bk512, nw8, st4) — m=4096 is at
  the MMA ceiling and every re-tune (bk1024/nw4/st3/rt512) lost or tied.

2026-11-12 dot re-migration (P01a lifted)
-----------------------------------------
The 2026-11-08 rewrite below was forced by the 2026-11-07 20:22 toolchain
skew that broke every ``tl.dot`` / 2-D tile form (archive:
``compiler_issues/sglang_fused_moe_router_tensorcore.md``). P01a was
declared lifted on 2026-11-10 (external gcu-compiler-opt re-aligned its
operandSegmentSizes; ``fused_moe_gemm`` / ``bmm_chunk`` re-ran green on
11-11), so the MMA window is open again — same re-migration the sibling
op ``fused_moe_router_cudacore`` landed on 2026-11-12 (2.33-14.39x on the
E=256/H=4096 bench shapes, GCU400).

This module keeps the two-kernel contract and adds two MMA arms for the
GEMM stage (bf16 operands, fp32 accumulate):
* ``_gemm_kloop_kernel`` (M >= 256): large-tile 3-operand ``tl.dot(a, b,
  acc)`` over a BLOCK_K-tiled H reduction, ``[ROW_TILE, BLOCK_E]`` logits
  slices (256x128 / 512x128), configs re-tuned on 2026-11-12 from the
  11-07 originals (whose tiles now exceed the P01a-lifted compiler's DSM
  accounting). GEMM-only 48.8us @ m=512 / 92.6us @ m=4096 vs ~465us for
  the PAD_M arm.
* ``_gemm_dot_kernel`` (2 <= M < 256, or K-loop fallback): full-K single
  ``tl.dot`` per tile, ``PAD_M=32`` row padding so the MMA units are fed
  whole 32-row batches, ``num_warps=1``: x rows as a ``[PAD_M, BLOCK_K]``
  block-ptr tile, router weight as its transposed ``[BLOCK_K, BLOCK_N]``
  tile, 2-D affine masked store into the fp32 logits. Launch configs
  follow the sibling op's three-round probe calibration: ``BLOCK_N=64``
  only when ``M>=64 and E>=256 and E%64==0`` (whole tiles, no masked
  waste), otherwise ``BLOCK_N=32``; ``BLOCK_K=next_pow2(H)`` (single dot
  over the whole reduction, the proven fused_moe_gemm form — smaller M
  keeps this arm: m=8/64 measure 11.4/16.0us, faster than both the
  dot-free form and the 11-07 build).

Domain guards keep every unverified shape on the previous arms: bf16 x
and w only, ``2 <= M`` (M==1 keeps the GEMV arm — MMA row fill 1/PAD_M
loses ~16% on single-row GEMV; ``M >= 256`` routes to the K-loop arm,
``E >= 128`` there), ``H <= 4096`` (DSM: the PAD_M arm's a+b tiles peak at
(32+64)*4096*2B = 768KB, the K-loop arm's at (256+128)*1024*2B = 786KB,
both < the 917504B OOR limit), ``64 <= E <= 8160`` (grid.y limits), and
one-shot try/except flags per arm that permanently downgrade to the next
arm if a kernel ever fails to build (K-loop -> PAD_M dot -> dot-free).
The dot-free kernels below stay as the correctness fallback (fp32/fp16
inputs, small E, compile failure).

2026-11-08 toolchain-constrained rewrite (dot-free vector-FMA forms)
--------------------------------------------------------------------
The 2026-11-07 20:22 enflame toolchain (triton 3.6.0+flagtree 0.7.0 /
torch_gcu 2.11.0+4.1.20261015) breaks every high-throughput GEMM form this
op previously relied on (all details + probe evidence archived in
``compiler_issues/sglang_fused_moe_router_tensorcore.md``):

* ``tl.dot`` — *every* form fails to compile fresh (``gcu.matmul``
  operandSegmentSizes 8 vs 5; probe matrix in the archive covers 2-arg /
  3-arg-acc / ieee / fp16 / bf16 / out_dtype / nw1-8 / small+large tiles).
* Ascending or unmasked 2-D tile loads — compile error
  (``memref_ext.slice_(pad_)start`` 7 vs 6 / 8 vs 7).
* The only compilable 2-D tile (descending rows + masked + num_warps=1)
  is correct **only** for axis=0 reductions; axis=1 reductions silently
  return wrong values, and a column-broadcast multiply (``v[:, None]``)
  feeding an axis=0 reduction silently corrupts as well.
* Any pid-derived *vector* offset (``e_offs = pid * BE + arange``) used in
  a gather load or vector store silently drops the tile base (every
  non-first tile writes tile 0's slot and leaves its own columns
  unwritten), and two pid-derived axes in one grid (e-tile on one axis,
  rows on the other) corrupt too — regardless of constexpr/runtime
  multipliers and of which grid axis carries which role.

The forms that *are* verified correct (and on which this rewrite is
built — every item has a probe in the job tmp dir):

* flat 1-D grid with runtime ``div``/``mod`` tile decomposition, every
  address a **scalar base + arange** (probe28 g29: M1..M4096 all-pass);
* runtime scalar arithmetic chains (``pid * C``, ``(e0 + e) * K``) into
  1-D loads (probe21 s1..s10 all-pass once the reference comparison is
  done right);
* per-expert 1-D ``tl.sum`` tree reductions with a ``where``-assembled
  [BE] accumulator (probe28);
* the per-row post kernel (1-D [E] load, closed-form top-2 weights,
  ``tl.max(..., return_indices=True)`` — post1d probe, bit-exact ids).

So the GEMM is now a pure vector-FMA inner-product kernel: each program
owns BE experts' outputs for RT rows and walks the full H reduction with
1-D loads; the tree-reduction count (M*E) is the physical throughput
limit (~0.1-0.5 TFLOPS vs the tensor core's ~121 TFLOPS the 11-07 dot
build delivered). Performance vs the 11-07 baseline therefore regresses
structurally for larger M — that regression is a toolchain artifact,
documented in the archive file, not a tuning shortfall: every faster
*correct* form was probed and rejected by the compiler backend.

Two-kernel split (unchanged contract)
-------------------------------------
1. ``_gemm_rows_kernel`` — the dot-free GEMM (or ``_gemv_row_kernel`` for
   M == 1) producing the final ``[M, E]`` fp32 logits (softcap + bias
   fused in registers before the store). bf16/fp16 operands multiply in
   fp32 (products exact, matching the bf16 tolerance); fp32 inputs use
   the same fp32 FMA path (IEEE semantics, 1e-4 tolerance met).
2. ``_post_kernel`` — per-row softmax over all E experts + top-2
   selection + weight gather, all in registers over one [E] row.
   ``tl.max(..., return_indices=True)`` tie-breaks to the lowest index,
   matching ``torch.topk``; the top-1 weight is the closed form
   ``1/sm_denom`` (a max returns its input verbatim, so
   ``logits[idx0] == sm_max`` exactly) and the top-2 weight is
   ``exp(val1 - sm_max)/sm_denom`` with ``val1 = logits[idx1]`` recovered
   exactly from the penalised-tile max. Indices are bit-identical to the
   reference.

Portability / safety
--------------------
* Device follows ``flaggems_sglang.device``; no vendor names.
* No module-level mutable state: launch configs are a pure function of
  ``(M, E, H, dtype)`` with a grid-limit ladder (flat grid <= 65535,
  RT doubles when a bigger M would overflow).
* No atomics, no torch fallback, no vendor intrinsic.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

import flaggems_sglang


@triton.jit
def _gemm_dot_kernel(
    x_ptr,  # [M, H] bf16
    w_ptr,  # [E, H] bf16
    bias_ptr,  # [E] (fp32) or dummy
    logits_ptr,  # [M, E] fp32 (output, post-softcap + post-bias)
    M: tl.constexpr,
    H: tl.constexpr,
    E: tl.constexpr,
    SXB: tl.constexpr,  # x row stride
    SXH: tl.constexpr,  # x col stride
    SWE: tl.constexpr,  # w row (expert) stride
    SWH: tl.constexpr,  # w col stride
    SLB: tl.constexpr,  # logits row stride
    SLE: tl.constexpr,  # logits col stride
    CAP: tl.constexpr,  # moe_softcapping (unused when SOFTCAP is False)
    SOFTCAP: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_N: tl.constexpr,  # expert-tile width (32/64)
    BLOCK_K: tl.constexpr,  # K tile = next_pow2(H): full-K single dot
    PAD_M: tl.constexpr,  # row padding (32: whole MMA row batches)
):
    # MMA arm (fused_moe_gemm / fused_moe_router_cudacore proven form):
    # full-K single tl.dot per (m-tile, n-tile) program — runtime-loop
    # dots with >= 2 iterations miscompile on this backend, so the K
    # dimension is never looped. bf16 operands, fp32 accumulate.
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    m0 = pid_m * PAD_M
    p_a = tl.make_block_ptr(
        x_ptr + m0 * SXB,
        (M - m0, H),
        (SXB, SXH),
        (0, 0),
        (PAD_M, BLOCK_K),
        (1, 0),
    )
    a = tl.load(p_a, boundary_check=(0, 1))
    n0 = pid_n * BLOCK_N
    p_b = tl.make_block_ptr(
        w_ptr,
        (H, E),
        (SWH, SWE),
        (0, n0),
        (BLOCK_K, BLOCK_N),
        (0, 1),
    )
    b = tl.load(p_b, boundary_check=(0, 1))
    acc = tl.dot(a, b, out_dtype=tl.float32)

    # ---- optional logit softcap: tanh(logits / cap) * cap (in registers) ----
    if SOFTCAP:
        acc = CAP * libdevice.tanh(acc / CAP)

    # ---- optional expert correction bias (in registers) ----
    if HAS_BIAS:
        offs_nb = n0 + tl.arange(0, BLOCK_N)
        bias = tl.load(
            bias_ptr + offs_nb, mask=offs_nb < E, other=0.0
        ).to(tl.float32)
        acc += bias[None, :]

    # ---- 2-D affine masked store of this [PAD_M, BLOCK_N] logits tile ----
    offs_m = m0 + tl.arange(0, PAD_M)
    offs_n = n0 + tl.arange(0, BLOCK_N)
    tl.store(
        logits_ptr + offs_m[:, None] * SLB + offs_n[None, :] * SLE,
        acc,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < E),
    )


@triton.jit
def _gemm_kloop_kernel(
    x_ptr,  # [M, H] bf16
    w_ptr,  # [E, H] bf16
    bias_ptr,  # [E] (fp32) or dummy
    logits_ptr,  # [M, E] fp32 (output, post-softcap + post-bias)
    M,
    H,
    E,
    moe_softcapping,
    BLOCK_E: tl.constexpr,  # E-tile width per program
    BLOCK_K: tl.constexpr,  # K (H) reduction tile
    ROW_TILE: tl.constexpr,  # rows per program
    SOFTCAP: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    E_TILE_FULL: tl.constexpr,  # E % BLOCK_E == 0 (no expert mask)
    H_ALIGNED: tl.constexpr,  # H % BLOCK_K == 0 (no K-tail mask)
):
    # K-loop MMA arm (2026-11-12 dot re-migration, large-M path): each
    # program owns a [ROW_TILE, BLOCK_E] logits slice and walks the full H
    # reduction with the 3-operand ``tl.dot(a, b, acc)`` form so the running
    # accumulator feeds straight into the MMA (one fewer full-tile fp32 add
    # per K-iteration). Large ROW_TILE amortises the w-tile re-reads that
    # the PAD_M=32 arm pays per 32-row batch; the smaller BLOCK_K keeps the
    # operand tiles inside the DSM budget ((RT+BE)*BK*2B <= 786432B).
    # Verified numerically correct on the P01a-lifted toolchain (the 11-07
    # build's multi-iteration runtime-loop dot miscompile does not
    # reproduce; probe 2026-11-12, max|dlogit| 1.5e-4 @ bf16 tolerance).
    pid_m = tl.program_id(0)
    pid_e = tl.program_id(1)
    row_start = pid_m * ROW_TILE
    row_offs = row_start + tl.arange(0, ROW_TILE)
    row_mask = row_offs < M
    e_start = pid_e * BLOCK_E
    e_offs = e_start + tl.arange(0, BLOCK_E)
    e_mask = e_offs < E

    acc = tl.zeros((ROW_TILE, BLOCK_E), dtype=tl.float32)
    for k0 in range(0, H, BLOCK_K):
        k_offs = k0 + tl.arange(0, BLOCK_K)
        x_ptrs = x_ptr + row_offs[:, None] * H + k_offs[None, :]
        w_ptrs = w_ptr + e_offs[None, :] * H + k_offs[:, None]
        if H_ALIGNED:
            x_block = tl.load(x_ptrs, mask=row_mask[:, None], other=0.0)
            if E_TILE_FULL:
                w_block = tl.load(w_ptrs)
            else:
                w_block = tl.load(w_ptrs, mask=e_mask[None, :], other=0.0)
        else:
            k_mask = k_offs < H
            x_block = tl.load(
                x_ptrs, mask=row_mask[:, None] & k_mask[None, :], other=0.0
            )
            w_block = tl.load(
                w_ptrs, mask=k_mask[:, None] & e_mask[None, :], other=0.0
            )
        acc = tl.dot(x_block, w_block, acc, out_dtype=tl.float32)

    # ---- optional logit softcap: tanh(logits / cap) * cap (in registers) ----
    if SOFTCAP:
        acc = moe_softcapping * libdevice.tanh(acc / moe_softcapping)

    # ---- optional expert correction bias (in registers) ----
    if HAS_BIAS:
        bias = tl.load(bias_ptr + e_offs, mask=e_mask, other=0.0).to(tl.float32)
        acc = acc + bias[None, :]

    # ---- store this program's [ROW_TILE, BLOCK_E] slice of the logits ----
    if E_TILE_FULL:
        tl.store(
            logits_ptr + row_offs[:, None] * E + e_offs[None, :],
            acc,
            mask=row_mask[:, None],
        )
    else:
        tl.store(
            logits_ptr + row_offs[:, None] * E + e_offs[None, :],
            acc,
            mask=row_mask[:, None] & e_mask[None, :],
        )


@triton.jit
def _gemm_rows_kernel(
    x_ptr,  # [M, H] (fp32 / bf16 / fp16)
    w_ptr,  # [E, H] (fp32 / bf16 / fp16)
    bias_ptr,  # [E] (fp32) or dummy
    logits_ptr,  # [M, E] fp32  (output, post-softcap + post-bias)
    M,
    H,
    E,
    moe_softcapping,
    BE_RT,  # runtime expert-tile width (== BE value; runtime keeps the
    # pid-derived scalar chain off the broken constexpr-fold path)
    NRT_RT,  # runtime row-tile count (grid = num_e_tiles * NRT_RT)
    BE: tl.constexpr,  # experts per program (pow2, >= 4)
    BK: tl.constexpr,  # K (H) reduction tile
    RT: tl.constexpr,  # rows per program
    SOFTCAP: tl.constexpr,  # moe_softcapping != 0.0
    HAS_BIAS: tl.constexpr,
):
    # grid = (num_e_tiles * NRT_RT,) flat. Tile decomposition via runtime
    # div/mod keeps every address a scalar base + arange vector — the only
    # address form the current toolchain lowers correctly at every tile
    # offset (probe28 g29 all-pass).
    pid = tl.program_id(0)
    e_tile = pid // NRT_RT
    row_tile = pid % NRT_RT
    e0 = e_tile * BE_RT
    row0 = row_tile * RT
    k_offs = tl.arange(0, BK)

    for r in range(RT):
        row = row0 + r
        rm = row < M
        acc = tl.zeros((BE,), dtype=tl.float32)
        for k0 in range(0, H, BK):
            kk = k0 + k_offs
            km = kk < H
            xv = tl.load(x_ptr + row * H + kk, mask=rm & km, other=0.0)
            for e in range(BE):  # static unroll: scalar w-row loads
                wv = tl.load(
                    w_ptr + (e0 + e) * H + kk, mask=(e0 + e < E) & km, other=0.0
                )
                s = tl.sum(wv.to(tl.float32) * xv.to(tl.float32))
                acc += tl.where(tl.arange(0, BE) == e, s, 0.0)

    # ---- optional logit softcap: tanh(logits / cap) * cap (in registers) ----
        if SOFTCAP:
            acc = moe_softcapping * libdevice.tanh(acc / moe_softcapping)

        # ---- optional expert correction bias (in registers) ----
        if HAS_BIAS:
            bias = tl.load(
                bias_ptr + e0 + tl.arange(0, BE),
                mask=(e0 + tl.arange(0, BE)) < E,
                other=0.0,
            ).to(tl.float32)
            acc = acc + bias

        # ---- store this row's [BE] slice: scalar base + arange ----
        tl.store(
            logits_ptr + row * E + e0 + tl.arange(0, BE),
            acc,
            mask=rm & ((e0 + tl.arange(0, BE)) < E),
        )


@triton.jit
def _gemv_row_kernel(
    x_ptr,  # [1, H]
    w_ptr,  # [E, H]
    bias_ptr,  # [E] (fp32) or dummy
    logits_ptr,  # [1, E] fp32
    H,
    E,
    moe_softcapping,
    BE_RT,  # runtime expert-tile width
    BE: tl.constexpr,
    BK: tl.constexpr,
    SOFTCAP: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    # M == 1 specialised path: grid = (num_e_tiles,); the single row is a
    # constant base (no row arithmetic at all).
    pid = tl.program_id(0)
    e0 = pid * BE_RT
    k_offs = tl.arange(0, BK)
    acc = tl.zeros((BE,), dtype=tl.float32)
    for k0 in range(0, H, BK):
        kk = k0 + k_offs
        km = kk < H
        xv = tl.load(x_ptr + kk, mask=km, other=0.0)
        for e in range(BE):
            wv = tl.load(
                w_ptr + (e0 + e) * H + kk, mask=(e0 + e < E) & km, other=0.0
            )
            s = tl.sum(wv.to(tl.float32) * xv.to(tl.float32))
            acc += tl.where(tl.arange(0, BE) == e, s, 0.0)

    if SOFTCAP:
        acc = moe_softcapping * libdevice.tanh(acc / moe_softcapping)
    if HAS_BIAS:
        bias = tl.load(
            bias_ptr + e0 + tl.arange(0, BE),
            mask=(e0 + tl.arange(0, BE)) < E,
            other=0.0,
        ).to(tl.float32)
        acc = acc + bias
    tl.store(
        logits_ptr + e0 + tl.arange(0, BE),
        acc,
        mask=(e0 + tl.arange(0, BE)) < E,
    )


@triton.jit
def _post_rows_kernel(
    logits_ptr,  # [M, E] fp32 (post-softcap + post-bias)
    weights_ptr,  # [M, topk] fp32
    indices_ptr,  # [M, topk] int32
    M,
    E,
    BLOCK_E: tl.constexpr,  # next_pow2(E)
    TOPK: tl.constexpr,
    BLOCK_TOPK: tl.constexpr,  # next_pow2(topk)
    E_IS_FULL: tl.constexpr,  # BLOCK_E == E (no expert padding)
):
    # One program per token row (1-D [E] load) — the small-M post form:
    # at M <= 64 the per-row programs already fill the fabric and the
    # 2-D tile form's fewer, fatter programs lose (measured 13.8us vs
    # ~5us at m=64). Three tree reductions (max-with-index, exp-sum,
    # penalised max-with-index); closed-form weights are bit-identical to
    # gather(probs, topk_ids) because a max reduction returns one of its
    # inputs verbatim.
    row = tl.program_id(0)
    if row >= M:
        return
    offs = tl.arange(0, BLOCK_E)
    if E_IS_FULL:
        logits = tl.load(logits_ptr + row * E + offs)
        sm_max, idx0 = tl.max(logits, axis=0, return_indices=True)
        e_exp = tl.exp(logits - sm_max)
        sm_denom = tl.sum(e_exp, axis=0)
    else:
        em = offs < E
        logits = tl.load(logits_ptr + row * E + offs, mask=em, other=float("-inf"))
        sm_max, idx0 = tl.max(logits, axis=0, return_indices=True)
        e_exp = tl.exp(logits - sm_max) * em
        sm_denom = tl.sum(e_exp, axis=0)

    _PEN = 1.0e38
    if TOPK >= 2:
        cur = logits - (offs == idx0).to(tl.float32) * _PEN
        val1, idx1 = tl.max(cur, axis=0, return_indices=True)
        w0 = 1.0 / sm_denom
        w1 = tl.exp(val1 - sm_max) / sm_denom
        k_offs = tl.arange(0, BLOCK_TOPK)
        sel_idx = tl.where(k_offs == 0, idx0, idx1)
        sel_w = tl.where(k_offs == 0, w0, w1)
        out_mask = k_offs < TOPK
        tl.store(weights_ptr + row * TOPK + k_offs, sel_w, mask=out_mask)
        tl.store(indices_ptr + row * TOPK + k_offs, sel_idx.to(tl.int32), mask=out_mask)
    else:
        w0 = 1.0 / sm_denom
        tl.store(weights_ptr + row * TOPK, w0)
        tl.store(indices_ptr + row * TOPK, idx0.to(tl.int32))


@triton.jit
def _post_kernel(
    logits_ptr,  # [M, E] fp32 (post-softcap + post-bias)
    weights_ptr,  # [M, topk] fp32
    indices_ptr,  # [M, topk] int32
    M,
    E,
    BLOCK_E: tl.constexpr,  # next_pow2(E)
    ROW_TILE: tl.constexpr,  # rows per program (pow2)
    TOPK: tl.constexpr,
    BLOCK_TOPK: tl.constexpr,  # next_pow2(topk)
    E_IS_FULL: tl.constexpr,  # BLOCK_E == E (no expert padding)
    M_ALIGNED: tl.constexpr,  # M % ROW_TILE == 0 (no row mask needed)
):
    # 2-D tile post stage (2026-11-13 round-3 rewrite): each program owns a
    # [ROW_TILE, BLOCK_E] logits tile and does the row softmax + top-2 +
    # closed-form weight extraction entirely in registers. The previous
    # form's two ``tl.max(..., return_indices=True)`` reductions cost 5.4x
    # the whole stage (ablation: a plain load/max/exp-sum kernel runs at the
    # load-store floor, adding index tracking + the penalised second max
    # costs +86us @ m=4096); this form tracks indices with the index-free
    # selection algebra (``tl.min(tl.where(x == max, offs, E))`` — leftmost
    # tied index, the same ``torch.topk`` tie-break the tracked max
    # provided) and is bit-identical on weights and indices (22-case verify
    # battery incl. all-equal rows, quantised ties, real GEMM logits). The
    # top-1 weight stays the closed form ``1/sm_denom`` (a max returns its
    # input verbatim, so ``logits[idx0] == sm_max`` exactly) and the top-2
    # weight is ``exp(val1 - sm_max)/sm_denom`` with ``val1 = logits[idx1]``
    # recovered exactly from the penalised-tile max.
    pid = tl.program_id(0)
    row_start = pid * ROW_TILE
    row_offs = row_start + tl.arange(0, ROW_TILE)
    row_mask = row_offs < M
    e_offs = tl.arange(0, BLOCK_E)
    e_mask = e_offs < E

    # ---- load the [ROW_TILE, E] logits tile (padded slots -> -inf) ----
    if M_ALIGNED and E_IS_FULL:
        logits = tl.load(logits_ptr + row_offs[:, None] * E + e_offs[None, :])
    elif E_IS_FULL:
        logits = tl.load(
            logits_ptr + row_offs[:, None] * E + e_offs[None, :],
            mask=row_mask[:, None],
            other=float("-inf"),
        )
    else:
        logits = tl.load(
            logits_ptr + row_offs[:, None] * E + e_offs[None, :],
            mask=row_mask[:, None] & e_mask[None, :],
            other=float("-inf"),
        )

    # ---- row max + softmax denominator (plain reductions) ----
    if E_IS_FULL:
        sm_max = tl.max(logits, axis=1)
        e_exp = tl.exp(logits - sm_max[:, None])
        sm_denom = tl.sum(e_exp, axis=1)
    else:
        masked = tl.where(e_mask[None, :], logits, float("-inf"))
        sm_max = tl.max(masked, axis=1)
        e_exp = tl.exp(logits - sm_max[:, None]) * e_mask[None, :]
        sm_denom = tl.sum(e_exp, axis=1)

    # ---- top-1: leftmost index among the max-tied elements ----
    if E_IS_FULL:
        hit0 = logits == sm_max[:, None]
    else:
        hit0 = (logits == sm_max[:, None]) & e_mask[None, :]
    idx0 = tl.min(tl.where(hit0, e_offs[None, :], BLOCK_E), axis=1)

    # ---- top-2: penalise the winner, plain max, leftmost tied index ----
    rem = tl.where(e_offs[None, :] == idx0[:, None], float("-inf"), logits)
    if not E_IS_FULL:
        rem = tl.where(e_mask[None, :], rem, float("-inf"))
    val1 = tl.max(rem, axis=1)
    hit1 = rem == val1[:, None]
    idx1 = tl.min(tl.where(hit1, e_offs[None, :], BLOCK_E), axis=1)

    w0 = 1.0 / sm_denom
    w1 = tl.exp(val1 - sm_max) / sm_denom

    # ---- assemble and store the [ROW_TILE, topk] output tiles ----
    k_offs = tl.arange(0, BLOCK_TOPK)
    sel_idx = tl.where(k_offs[None, :] == 0, idx0[:, None], idx1[:, None])
    sel_w = tl.where(k_offs[None, :] == 0, w0[:, None], w1[:, None])
    out_mask = row_mask[:, None] & (k_offs[None, :] < TOPK)
    out2d = row_offs[:, None] * TOPK + k_offs[None, :]
    tl.store(weights_ptr + out2d, sel_w, mask=out_mask)
    tl.store(indices_ptr + out2d, sel_idx.to(tl.int32), mask=out_mask)


def _next_pow2(x: int) -> int:
    p = 1
    while p < x:
        p <<= 1
    return p


# ---------------------------------------------------------------------------
# Launch configs (probe28 sweep, 2026-11-08, bench shapes H=4096/E=256/bf16,
# triton.testing.do_bench median): the dot-free tree form's per-M optima.
#   M=1    gemv  BE8/BK4096            13.8us  (11-07 dot build: 8.3us)
#   M=8    BE16/BK4096/RT1             64.2us  (11-07: 11.4us)
#   M=64   BE8 /BK4096/RT1            328.0us  (11-07: 14.1us)
#   M=512  BE16/BK4096/RT4           2157.6us  (11-07: 49.0us)
#   M=4096 BE16/BK4096/RT16         17345.8us  (11-07: 180.3us)
# The regression is the toolchain's tl.dot blackout (vector-tree FMA tops
# out ~2 orders below MMA throughput); see the module docstring + archive.
# ---------------------------------------------------------------------------

_GEMM_CFGS = (
    # (BE, BK, RT) per M-bucket
    (16, 4096, 1),  # m <= 8
    (8, 4096, 1),  # m <= 64
    (16, 4096, 4),  # m <= 512
    (16, 4096, 16),  # m > 512
)

_GEMV_CFG = (8, 4096)  # (BE, BK) for M == 1


def _m_bucket(m: int) -> int:
    if m <= 8:
        return 0
    if m <= 64:
        return 1
    if m <= 512:
        return 2
    return 3


# ---------------------------------------------------------------------------
# MMA arm dispatch (2026-11-12 dot re-migration; config calibration from the
# fused_moe_router_cudacore three-round probes: PAD_M=32, full-K single dot,
# num_warps=1; BLOCK_N=64 only for M>=64 & E>=256 & E%64==0, else 32).
# Domain guards keep unverified shapes on the dot-free arms; a one-shot
# try/except permanently downgrades to the dot-free arm if the dot kernel
# ever fails to build, so the re-migration is risk-free for correctness.
# ---------------------------------------------------------------------------

_DOT_PAD_M = 32
_dot_arm_broken = False


def _launch_gemm_dot(
    x, router_weight, bias_arg, logits, m, h, e, cap, softcap, has_bias
):
    """MMA GEMM arm (bf16 operands, fp32 accumulate). Returns False when the
    shape is outside the arm's domain or the kernel fails to build, in which
    case the caller keeps the dot-free kernels as fallback."""
    global _dot_arm_broken
    if (
        _dot_arm_broken
        or m < 2
        or h > 4096
        or e < 64
        or e > 8160  # grid.y = ceil(e / BLOCK_N) <= 255 at BLOCK_N=32
        or (m + _DOT_PAD_M - 1) // _DOT_PAD_M > 65535
        or x.dtype is not torch.bfloat16
        or router_weight.dtype is not torch.bfloat16
    ):
        return False
    if m >= 64 and e >= 256 and e % 64 == 0:
        block_n = 64
    else:
        block_n = 32
    sx = x.stride()
    sw = router_weight.stride()
    sl = logits.stride()
    try:
        _gemm_dot_kernel[
            (
                (m + _DOT_PAD_M - 1) // _DOT_PAD_M,
                (e + block_n - 1) // block_n,
            )
        ](
            x,
            router_weight,
            bias_arg,
            logits,
            m,
            h,
            e,
            sx[0],
            sx[1],
            sw[0],
            sw[1],
            sl[0],
            sl[1],
            (cap if cap != 0.0 else 1.0),
            softcap,
            has_bias,
            block_n,
            _next_pow2(h),
            _DOT_PAD_M,
            num_warps=1,
            num_stages=1,
        )
        return True
    except Exception:
        _dot_arm_broken = True
        return False


# ---------------------------------------------------------------------------
# K-loop MMA arm (large M >= 256): 3-operand dot over a BLOCK_K-tiled H
# reduction with a large ROW_TILE. Config table from the 2026-11-12 probe
# (interleaved do_bench medians, E=256/H=4096/bf16, GEMM-only):
#   m=512:  rt256/be128/bk1024/nw8/st4 = 48.8us  (PAD_M arm: ~465us)
#   m=4096: rt512/be128/bk512/nw4/st3  = 92.6us (PAD_M arm: ~465us)
# DSM budget (RT+BE)*BLOCK_K*2B: 786432B / 655360B, both under the 917504B
# OOR limit (the 11-07 originals rt256/be256/bk1024 and rt1024/be256/bk512
# now exceed it on the P01a-lifted compiler and were re-tuned down).
# Falls back to the PAD_M dot arm on any build failure, which in turn falls
# back to the dot-free kernels.
# ---------------------------------------------------------------------------

_KLOOP_CFGS = (
    # (m_threshold, row_tile, block_e, block_k, num_warps, num_stages)
    # 2026-11-12 interleaved sweeps, GEMM-only medians (E=256/H=4096/bf16):
    #   m=512:  rt128/be128/bk1024/nw8/st4 = 33.1us (rt256: 48.6)
    #   m=4096: rt256/be128/bk512/nw8/st4  = 70.5us (rt512/nw4: 92.2)
    # 2026-11-13 round-3 occupancy resweep (probe5/6, same protocol):
    #   m=512:  rt64/be64/bk1024/nw4/st4   = 23.8us (rt128/be128: 32.7 —
    #           8 programs on 48 SPCs was the limit; 32 programs now)
    #   m=1024: rt128/be64/bk1024/nw4/st4  = 31.3us (rt256/be128: 55.7)
    #   m=2048: rt256/be128/bk1024/nw8/st4 = 52.0us (bk512: 58.6)
    #   m=4096: unchanged — 71.4us is at the ~121 TFLOPS MMA ceiling and
    #           every re-tune (bk1024/nw4/st3/rt512) lost or tied.
    # DSM budgets (RT+BE)*BK*2B: 262144 / 393216 / 786432 / 393216B, all
    # under the 917504B limit.
    (512, 64, 64, 1024, 4, 4),  # 256 <= m <= 512
    (1024, 128, 64, 1024, 4, 4),  # 512 < m <= 1024
    (2048, 256, 128, 1024, 8, 4),  # 1024 < m <= 2048
    (None, 256, 128, 512, 8, 4),  # m > 2048
)
_kloop_arm_broken = False


def _launch_gemm_kloop(
    x, router_weight, bias_arg, logits, m, h, e, cap, softcap, has_bias
):
    """K-loop MMA arm (large M, bf16). Returns False when the shape is out
    of the arm's domain or the kernel fails to build; the caller then keeps
    the PAD_M dot arm / dot-free kernels."""
    global _kloop_arm_broken
    if (
        _kloop_arm_broken
        or m < 256
        or h > 4096
        or e < 128
        or e > 8160  # grid.y = ceil(e / BLOCK_E) <= 128 at BLOCK_E=64
        or x.dtype is not torch.bfloat16
        or router_weight.dtype is not torch.bfloat16
    ):
        return False
    for m_thr, rt, be, bk, nw, ns in _KLOOP_CFGS:
        if m_thr is None or m <= m_thr:
            break
    if (m + rt - 1) // rt > 65535:  # grid.x limit for the selected ROW_TILE
        return False
    try:
        _gemm_kloop_kernel[
            ((m + rt - 1) // rt, (e + be - 1) // be)
        ](
            x,
            router_weight,
            bias_arg,
            logits,
            m,
            h,
            e,
            cap,
            be,
            bk,
            rt,
            softcap,
            has_bias,
            (e % be == 0),
            (h % bk == 0),
            num_warps=nw,
            num_stages=ns,
        )
        return True
    except Exception:
        _kloop_arm_broken = True
        return False


def fused_moe_router_tensorcore(
    x, router_weight, topk, moe_softcapping, correction_bias=None
):
    M, H = x.shape
    E, H_w = router_weight.shape
    assert H == H_w, f"hidden dim mismatch: x H={H} vs weight H={H_w}"
    assert topk <= 2, "fused_moe_router_tensorcore supports topk <= 2"

    device = flaggems_sglang.device
    # Intermediate [M, E] fp32 logits — written by the GEMM kernel
    # (post-softcap + post-bias), read by the post kernel. Every output
    # element is stored exactly once (no atomics), so torch.empty suffices.
    logits = torch.empty((M, E), dtype=torch.float32, device=device)
    weights = torch.empty((M, topk), dtype=torch.float32, device=device)
    indices = torch.empty((M, topk), dtype=torch.int32, device=device)

    BLOCK_E_POST = _next_pow2(E)
    if BLOCK_E_POST < 4:
        BLOCK_E_POST = 4
    BLOCK_TOPK = _next_pow2(topk) if topk >= 1 else 1

    softcap = bool(moe_softcapping != 0.0)
    has_bias = correction_bias is not None
    bias_ptr = correction_bias if has_bias else x  # dummy ptr, unused

    if M == 1:
        # ---- M==1 specialised GEMV (constant row base, e-tile grid) ----
        be, bk = _GEMV_CFG
        be = min(be, _next_pow2(E))
        if be < 4:
            be = 4
        bk = min(bk, _next_pow2(H))
        num_e_tiles = (E + be - 1) // be
        _gemv_row_kernel[(num_e_tiles,)](
            x,
            router_weight,
            bias_ptr,
            logits,
            H,
            E,
            float(moe_softcapping),
            be,
            BE=be,
            BK=bk,
            SOFTCAP=softcap,
            HAS_BIAS=has_bias,
            num_warps=4,
            num_stages=1,
        )
    elif _launch_gemm_kloop(
        x,
        router_weight,
        bias_ptr,
        logits,
        M,
        H,
        E,
        float(moe_softcapping),
        softcap,
        has_bias,
    ):
        # ---- K-loop MMA arm: large-M 3-operand dot tiles (fp32 logits) ----
        pass
    elif _launch_gemm_dot(
        x,
        router_weight,
        bias_ptr,
        logits,
        M,
        H,
        E,
        float(moe_softcapping),
        softcap,
        has_bias,
    ):
        # ---- MMA arm: bf16 full-K single-dot tiles (fp32 logits) ----
        pass
    else:
        # ---- general dot-free GEMM: flat grid, div/mod tile split ----
        be, bk, rt = _GEMM_CFGS[_m_bucket(M)]
        be = min(be, _next_pow2(E))
        if be < 4:
            be = 4
        bk = min(bk, _next_pow2(H))
        num_e_tiles = (E + be - 1) // be
        # Flat grid must respect the 65535 grid.x limit: grow RT until the
        # row-tile count fits (each doubling halves the row-tile count).
        while (M + rt - 1) // rt * num_e_tiles > 65535:
            rt *= 2
        num_row_tiles = (M + rt - 1) // rt
        grid = (num_e_tiles * num_row_tiles,)
        _gemm_rows_kernel[grid](
            x,
            router_weight,
            bias_ptr,
            logits,
            M,
            H,
            E,
            float(moe_softcapping),
            be,
            num_row_tiles,
            BE=be,
            BK=bk,
            RT=rt,
            SOFTCAP=softcap,
            HAS_BIAS=has_bias,
            num_warps=4,
            num_stages=1,
        )

    # ---- stage 2: softmax + top-2 + gather ----
    # Small M (< 64): the per-row 1-D form — the fabric is already full of
    # single-row programs and the 2-D tile form's fatter programs lose
    # (probe5: m=8/37 1-D 4.8/5.5us vs 2-D ~7us; m=64 flips to the 2-D form).
    # Larger M: the 2-D tile form ([ROW_TILE, BLOCK_E] per program). The
    # 2026-11-13 round-3 sweep's per-M optima collapse onto one rule:
    # ROW_TILE = next_pow2(max(M / 32, 8)) capped at 128 (== 32 programs once
    # M >= 1024) with num_warps 2 below 32 rows and 4 from 32 rows up —
    # measured m=64/128/256/512/1024/2048/4096 = 7.4/7.6/8.0/8.7/10.2/12.9/
    # 19.4us, each within noise of its per-M optimum. Its DSM accounting
    # materialises up to five live ROW_TILE*BLOCK_E tiles (logits / exp /
    # penalised / two int index tiles): 5*ROW_TILE*BLOCK_E*4B <= ~880KB
    # under the 917504B limit, so ROW_TILE halves from 8 whenever a big E
    # inflates BLOCK_E (e.g. E=8160 -> BLOCK_E=8192 caps ROW_TILE at 4).
    # M_ALIGNED drops the row mask whenever M % ROW_TILE == 0 and E needs
    # no padding. num_stages=1: no K axis to pipeline.
    if M < 64:
        _post_rows_kernel[(M,)](
            logits,
            weights,
            indices,
            M,
            E,
            BLOCK_E=BLOCK_E_POST,
            TOPK=topk,
            BLOCK_TOPK=BLOCK_TOPK,
            E_IS_FULL=(BLOCK_E_POST == E),
            num_warps=1,
            num_stages=1,
        )
    else:
        rt_post = min(128, _next_pow2(max(M // 32, 8)))
        while rt_post > 8 and 5 * rt_post * BLOCK_E_POST * 4 > 880000:
            rt_post //= 2
        nw_post = 4 if rt_post >= 32 else 2
        _post_kernel[((M + rt_post - 1) // rt_post,)](
            logits,
            weights,
            indices,
            M,
            E,
            BLOCK_E=BLOCK_E_POST,
            ROW_TILE=rt_post,
            TOPK=topk,
            BLOCK_TOPK=BLOCK_TOPK,
            E_IS_FULL=(BLOCK_E_POST == E),
            M_ALIGNED=(M % rt_post == 0 and BLOCK_E_POST == E),
            num_warps=nw_post,
            num_stages=1,
        )

    return weights, indices


__all__ = ["fused_moe_router_tensorcore"]
