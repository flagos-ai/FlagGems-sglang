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

"""Triton kernel for moe/fused_moe_router_tensorcore.

Fused MoE router (Tensor Core path): computes routing logits via a tiled
``tl.dot`` GEMM (``logits = x.float() @ router_weight.float().T``), optional
logit soft-capping (``tanh``) and expert correction bias, a global softmax over
all experts, then top-k (k <= 2) expert selection with the softmax weights
gathered (not re-normalised). Mathematically identical to the
``fused_moe_router_cudacore`` reference; the difference is the GEMM uses
``tl.dot`` (Tensor Core) and ``topk`` is restricted to <= 2.

Signature matches ``reference(x, router_weight, topk, moe_softcapping,
correction_bias=None)``.

Implementation strategy
----------------------
The router is a small matmul (``[B, H] @ [H, E]``) feeding an elementwise
softcap + bias, a row-wise softmax and a top-2 selection. The reference runs
five separate kernels. Two regimes have very different cost structures on
this device:

  * Launch-bound (small B, the common bench shape B = 1..512): the actual
    compute is a few microseconds but each kernel launch is ~30us, so the
    reference spends ~157us mostly in launch overhead. Here the win is
    *fusion*: a single Triton kernel does the GEMM, softcap, bias, softmax,
    top-k and gather in one launch, so the per-shape time collapses to one
    launch's worth of work.

  * Compute-bound (large B, the B = 4096 bench shape): the GEMM dominates
    (~470us in the reference) and the launch overhead is negligible. A fully
    fused single kernel can't keep the GEMM efficient here because the
    *full-expert-axis* tile the softmax / top-k needs (``BLOCK_N = next_pow2(E)
    = 256``) forces a small row tile (``BLOCK_M = 16/32``) to fit the fused
    fp32 working set in the Ascend unified buffer, and the small-``BLOCK_M``
    GEMM is ~3-4x slower than the compute-bound sweet spot (``BLOCK_M = 64,
    BLOCK_N = 128`` with N-splitting). So fusion *loses* on the compute-bound
    shape.

The launcher therefore picks the regime from ``M``:

  * ``M <= _FUSED_M_LIMIT`` (default 512) -> the **single fused kernel**
    (``_fused_router_kernel``): each program owns ``BLOCK_M`` rows × the whole
    expert axis (``BLOCK_N = next_pow2(E)``), so the softmax, top-k and weight
    gather are all in-register, no second pass.
  * ``M > _FUSED_M_LIMIT`` -> the **two-kernel path**: an N-split GEMM kernel
    (``_logits_gemm_kernel``) using the compute-bound sweet-spot tile writes
    the ``[M, E]`` fp32 logits; a second row-tiled kernel
    (``_softmax_topk_kernel``) reads those logits once and does the softcap +
    bias + softmax + top-2 + gather in registers. The intermediate logits are
    one ``[M, E]`` fp32 buffer (4MB for the bench) — written once, read once —
    which is cheaper on this shape than the inefficient fused GEMM would be.

Precision: the reference upcasts to float32 and runs a strict ieee fp32 matmul.
For float32 input we must match that exactly (``input_precision="ieee"``); for
the lower-precision dtypes (bfloat16/float16) the tolerance is wide
(1.5e-2 / 1e-2), so we use ``input_precision="hf32"`` — the Ascend MMA's
faster fp32 schedule that keeps ~10 mantissa bits. The source data is bf16/f16
(<=8 mantissa bits), so hf32 loses nothing the input dtype hadn't already
discarded and stays well within tolerance, while running the GEMM at
substantially higher throughput than a strict ieee fp32 dot. (Keeping the
inputs in native bf16 for the dot trips a hardware MTE fault on Ascend, so we
upcast to float32 and rely on the precision flag — same approach as
sgemm_lora_a.)

Top-k (k <= 2) is an iterative row-wise argmax over the logits tile: each
pick gathers its softmax weight via a masked sum (``sum(probs * (col==idx))``)
and masks that expert to ``-inf`` for the next pick. The test harness seeds
the RNG so the random logits have no ties and this matches ``torch.topk``
exactly (descending value order: pick 0 = largest, pick 1 = 2nd largest).

Soft-cap uses the numerically-stable sigmoid form of tanh
(``cap * (2*sigmoid(2z/cap) - 1)``) rather than ``tl.tanh``, which is not
available on all backends (e.g. Ascend).

This is pure portable Triton: device via ``flaggems_sglang.device`` (never
hardcoded), no vendor-private ops, no fallbacks, no module-level mutable
state. Launch configs are chosen by a pure function of the input shapes (no
autotune wrapper — its per-call dispatch hook would only add latency on these
small, regular shapes).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

# M threshold between the launch-bound fused regime and the compute-bound
# two-kernel regime. Below this a single fused launch wins (the GEMM is tiny,
# launch overhead dominates); above it the GEMM is compute-bound and the
# fused kernel's small-row-tile GEMM is slower than an N-split GEMM + a
# second softmax/topk pass. Pure launch constant; not mutated.
_FUSED_M_LIMIT = 512

# Ascend unified-buffer budget in bytes (~192KB = 1572864 bits). Used by the
# shape-keyed config picker to keep tiles inside the UB. Pure constant.
_UB_BYTES = 1572864 // 8


def _gemm_config(M, N, H):
    """Pick (BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages) for the
    compute-bound GEMM kernel. Pure function of the shapes.

    Measured sweep on this device (Ascend NPU) for the bench GEMM
    (``[M, 4096] @ [4096, 256]``, bf16->fp32, N-split into [BLOCK_N] tiles)
    shows the winning tile depends on M:

      * Large M (M = 4096, compute-bound): an N-split tile ``[256, 128]`` with
        ``BLOCK_K = 128`` is fastest (~199us). Two N-tiles (grid_n=2) double the
        program count to 32 and a 256-row M-tile amortises launch + MMA fixed
        cost better than the full-N ``[128, 256]`` tile (~206us) — the wider
        M-tile matters more here than collapsing grid_n, and ``[256, 256]``
        overflows the UB so the N axis must split at this M-tile size. (The
        earlier ``[128, 256, 128]`` full-N tile is the next best; the softmax
        kernel reads the full axis either way, so splitting N on the GEMM is
        free.)
      * Mid M (64..512): a single full-N tile ``[64, 256]`` with ``BLOCK_K =
        128`` wins for M=512 (~87us); the ``[64, 128, 256]`` N-split tile is
        slightly faster at M=64 (~82us) where grid_n=2 gives more parallelism.
      * Small M (1..8): launch-bound — the full-N ``[64, 256]`` tile
        (``BLOCK_K = 128``) covers the whole row in one program (~79-94us).

    Rather than carry an M-branch on the BN (full-N vs N-split), we pick the
    per-M sweet spot directly. ``BLOCK_K = 128`` is the common winner: the UB
    cannot fit the ``[BM, 256, 256]`` dot (compile failure) and 256-wide N is
    already consuming the K budget; ``BLOCK_K = 64`` is ~30% slower. For the
    large-M compute-bound regime the N axis splits (BN=128) so the grid keeps
    enough programs to fill the device while the wider M-tile keeps the MMA
    busy; for the smaller-M regimes the N-axis fits in one tile (E <= 256).
    """
    # Large compute-bound M: N-split [256, 128] tile is the measured sweet spot
    # (~199us vs ~206us for full-N [128, 256]). grid_n=2 keeps enough programs
    # live, and the 256-row M-tile amortises the K-loop's fixed cost. The
    # softmax kernel reads the full expert axis regardless of how the GEMM
    # tiles N, so splitting N here is free.
    if M > 512:
        block_n = min(triton.next_power_of_2(max(N, 16)), 128)
        block_m = 256
        block_k = min(triton.next_power_of_2(max(H, 16)), 128)
        if H >= 256 and block_k >= 64:
            return block_m, block_n, block_k, 8, 3
        return block_m, block_n, block_k, 4, 2

    # Smaller M (launch-bound / mid): default to a full-N tile (BN =
    # next_pow2(E)); this is what the bench E=256 case wants. Cap BN so a huge
    # E still tiles.
    block_n = triton.next_power_of_2(max(N, 16))
    if block_n > 256:
        block_n = (
            128  # very large E -> fall back to the classic 128-wide N-split
        )

    # BLOCK_K: 128 is the cap that compiles on the [BM, 256] dot (256 fails).
    # For the (rare) small-E / large-K case keep 256 only when BN<=128.
    block_k = triton.next_power_of_2(max(H, 16))
    if block_k > 256:
        block_k = 256
    if block_n >= 256:
        block_k = min(block_k, 128)

    # BLOCK_M: per-M sweet spot. The full-N tile amortises launch cost better
    # with a wider M-tile, but a tile much bigger than M just wastes rows.
    if M <= 16:
        block_m = 64
    elif M <= 64:
        block_m = 32
    elif M <= 512:
        block_m = 64
    else:
        block_m = 128

    # When E is small the full-N tile is tiny; shrink BLOCK_M so a tile isn't
    # absurdly bigger than the actual rows, and keep the dot in a supported
    # Ascend MMA shape.
    if block_n <= 16:
        block_n = 16
        block_m = min(block_m, 64)
        block_k = min(block_k, 256)
    elif block_n <= 32:
        block_m = min(block_m, 64)
        block_k = min(block_k, 256)

    # Pipelining: 3 stages overlaps the K-loop's loads with compute on the
    # wide-K tiles. Keep num_warps=8 (matches the [BM, BN, 128] dot size).
    if H >= 256 and block_k >= 64:
        num_warps = 8
        num_stages = 3
    else:
        num_warps = 4
        num_stages = 2
    return block_m, block_n, block_k, num_warps, num_stages


def _fused_config(M, E, H):
    """Pick (BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages, kloop_stages)
    for the single fused kernel (launch-bound regime). Pure function of the
    shapes; written as straight-line branches (no closures / loops) so the
    Python-side launcher overhead stays low — on this device the kernel launch
    is async, so CPU work between back-to-back launches serialises them and
    shows up as GPU idle time in the benchmark timer.

    The fused kernel needs the *whole* expert axis resident for the in-register
    softmax / top-k, so ``BLOCK_N = next_pow2(E)``. The fused fp32 working set
    (accumulator + softmax intermediates + topk work tile) is several
    ``[BLOCK_M, BLOCK_N]`` fp32 tiles, so ``BLOCK_M`` is picked against that
    budget, not just the dot. For the bench ``E = 256`` (``BLOCK_N = 256``) the
    fused tile's BLOCK_K is UB-capped at 64 once BLOCK_N covers the full axis,
    forcing many (4096/64 = 64) K-loop iterations; the row tile then becomes
    the lever: a *smaller* ``BLOCK_M`` (8) wins for small M because it gives
    the GEMM a better per-program row granularity and leaves more UB headroom
    for the in-register softmax/topk working set — measured BM=8 is ~105us vs
    BM=16's ~110us for M<=64 (the extra padded rows just cost work). For M=512
    the wider BM=32 amortises the 64-iteration K-loop across real rows (BM=16
    there is ~287us — far too many tiles).
    """
    block_n = triton.next_power_of_2(max(E, 16))

    if block_n >= 128:
        # Full-axis (BN=256) bench case: K-loop is BK=64-capped (UB), so pick
        # BLOCK_M by the actual row count, not by an eager "wider is better".
        # Measured on this device: BM=8 wins for M<=64 (~5us faster than BM=16
        # — the smaller tile leaves more unified-buffer headroom for the
        # softmax/topk working set and gives a leaner GEMM per program; the
        # padded rows are cheap). BM=16 then wins at M=128; BM=32 amortises
        # the 64-iteration K-loop across real rows for the larger M.
        block_m = 8 if M <= 64 else 32
    else:
        block_m = 64

    # BLOCK_K starts at the smallest power of two >= H, capped at 512; for the
    # small-E tiles we cap lower (256) where the N axis is tiny.
    block_k = triton.next_power_of_2(max(H, 16))
    if block_k > 512:
        block_k = 512
    if block_n <= 32 and block_k > 256:
        block_k = 256

    # The fused softmax/topk buffers live alongside the dot tile in the UB, so
    # BLOCK_K and num_stages must fit the budget. For the common bench shape
    # (E=256, H=4096) this caps BLOCK_K at 64 / num_stages at 2; for the smaller
    # correctness shapes (E<=32) there is headroom for BK=128/256 and 3 stages.
    # The fit math is inlined (no closure) for low launcher overhead.
    acc_bytes = block_m * block_n * 4

    def _ok(k, s):
        return acc_bytes + s * (block_m * k + block_n * k) * 4 <= _UB_BYTES

    if _ok(block_k, 3):
        num_stages = 3
    elif _ok(block_k, 2):
        num_stages = 2
    else:
        num_stages = 2
        while block_k > 32 and not _ok(block_k, num_stages):
            block_k //= 2
        if not _ok(block_k, num_stages):
            num_stages = 1
            while block_k > 32 and not _ok(block_k, num_stages):
                block_k //= 2
            if not _ok(block_k, num_stages):
                block_k = 32

    num_warps = 8 if H >= 256 else 4

    # K-loop (software-pipelining) stages, independent of the kernel-wide
    # num_stages. ``tl.range(num_stages=...)`` overlaps K-tile loads with the
    # dot compute; the wide-N (BN=256) bench K-loop runs 4096/64 = 64 iters so
    # deeper pipelining helps (measured M=8: LNS=4 ~111us vs LNS=2 ~121us;
    # M=1 flat ~109us). The conservative UB-fit ``num_stages`` above is about
    # the kernel-level staging; the K-loop's own staging is allowed higher
    # here (verified to compile/run for the bench tiles). Keep it modest for
    # the wide-K / small-N correctness tiles where the dot tile is big.
    kloop_stages = 4
    if block_n <= 32:
        kloop_stages = min(num_stages, 3)
    return block_m, block_n, block_k, num_warps, num_stages, kloop_stages


def _softmax_topk_config(M, E):
    """Pick (BLOCK_M, BLOCK_N, num_warps) for the softmax/topk pass of the
    two-kernel path. The full expert axis is one tile (``BLOCK_N =
    next_pow2(E)``); BLOCK_M picks the row tile. Pure function of shapes.
    """
    block_n = triton.next_power_of_2(max(E, 16))
    # The softmax/topk kernel keeps several [BLOCK_M, BLOCK_N] fp32 tiles live
    # at once (acc, exp, probs, topk work tile, masks). For the full-axis
    # BLOCK_N = 256 bench that is too big at BLOCK_M = 64 (UB overflow on
    # Ascend); BLOCK_M = 32 fits at mid M. For large compute-bound M (the
    # m4096 bench) BLOCK_M = 16 is faster — each program owns fewer rows so
    # the grid carries more programs (4096/16 = 256 vs 128) and the row-wise
    # softmax/topk parallelism matters more than the per-program work
    # amortisation (measured BM=16 ~93us vs BM=32 ~94us). Small M pads (one
    # program per tile).
    block_m = 32
    if M <= 16:
        block_m = 16
    elif M <= 64 and block_n >= 256:
        block_m = 32
    elif M > 512:
        block_m = 16
    num_warps = 8
    return block_m, block_n, num_warps


@triton.jit
def _fused_router_kernel(
    x_ptr,
    w_ptr,
    bias_ptr,
    out_w_ptr,
    out_id_ptr,
    x_stride_m,  # stride along row dim of x [H] (contiguous -> H)
    w_stride_n,  # stride along expert dim of w [E, H] (contiguous -> H)
    M,  # B (tokens)
    N,  # E (experts)
    K,  # H (hidden)
    SOFTCAP,  # runtime float; 0.0 when disabled
    TOPK: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    USE_SOFTCAP: tl.constexpr,
    USE_IEEE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    KLOOP_STAGES: tl.constexpr,
):
    pid = tl.program_id(0)
    rm = pid * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    rn = tl.arange(0, BLOCK_N)  # [BLOCK_N]
    rk = tl.arange(0, BLOCK_K)  # [BLOCK_K]
    row_mask = rm < M  # [BLOCK_M]
    col_mask = rn < N  # [BLOCK_N]

    NEG_INF = -float("inf")

    # ---- GEMM: logits[m, n] = sum_k x[m, k] * w[n, k] ---------------------
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for kk in tl.range(0, K, BLOCK_K, num_stages=KLOOP_STAGES):
        k_off = kk + rk
        k_mask = k_off < K
        x = tl.load(
            x_ptr + rm[:, None] * x_stride_m + k_off[None, :],
            mask=row_mask[:, None] & k_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        w = tl.load(
            w_ptr + rn[:, None] * w_stride_n + k_off[None, :],
            mask=col_mask[:, None] & k_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        if USE_IEEE:
            acc += tl.dot(x, tl.trans(w), input_precision="ieee")
        else:
            acc += tl.dot(x, tl.trans(w), input_precision="hf32")

    # ---- Soft-cap (tanh) -------------------------------------------------
    if USE_SOFTCAP:
        # tanh(z) = 2*sigmoid(2z) - 1; stable form (tl.tanh not on all backends).
        acc = SOFTCAP * (
            2.0 * (1.0 / (1.0 + tl.exp(-2.0 * acc / SOFTCAP))) - 1.0
        )

    # ---- Correction bias ------------------------------------------------
    if HAS_BIAS:
        b = tl.load(bias_ptr + rn, mask=col_mask, other=0.0).to(tl.float32)
        acc = acc + b[None, :]

    # Force padding experts / rows inert: never selected, zero weight.
    valid = row_mask[:, None] & col_mask[None, :]
    acc = tl.where(valid, acc, NEG_INF)

    # ---- Softmax over the full expert axis -------------------------------
    mx = tl.max(acc, axis=1)  # [BLOCK_M]
    mx = tl.where(row_mask, mx, 0.0)  # guard invalid rows
    e = tl.exp(acc - mx[:, None])
    e = tl.where(valid, e, 0.0)
    sm = tl.sum(e, axis=1)  # [BLOCK_M]
    sm = tl.where(row_mask, sm, 1.0)
    probs = e / sm[:, None]  # [BLOCK_M, BLOCK_N] float32

    # ---- Top-k (k <= 2) selection + weight gather -----------------------
    work = acc
    for pick in range(TOPK):
        idx = tl.argmax(work, axis=1)  # [BLOCK_M] int64
        pick_mask = rn[None, :] == idx[:, None]  # [BLOCK_M, BLOCK_N]
        w_pick = tl.sum(tl.where(pick_mask, probs, 0.0), axis=1)  # [BLOCK_M]
        tl.store(out_w_ptr + rm * TOPK + pick, w_pick, mask=row_mask)
        tl.store(
            out_id_ptr + rm * TOPK + pick, idx.to(tl.int32), mask=row_mask
        )
        work = tl.where(pick_mask, NEG_INF, work)


@triton.jit
def _logits_gemm_kernel(
    x_ptr,
    w_ptr,
    logits_ptr,
    x_stride_m,  # K
    w_stride_n,  # K
    l_stride_m,  # E (contiguous)
    M,
    N,
    K,
    USE_IEEE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """N-split GEMM: ``logits[m, n] = x[m, :] @ w[n, :].T`` (float32 accumulate).

    Each program owns one [BLOCK_M, BLOCK_N] output tile; the N-axis is split
    across the grid's second dim. This is the compute-bound sweet-spot tile
    (BLOCK_M=64, BLOCK_N=128) — much faster than the full-axis tile the fused
    kernel is forced to use.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
    row_mask = rm < M
    col_mask = rn < N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for kk in tl.range(0, K, BLOCK_K, num_stages=2):
        k_off = kk + rk
        k_mask = k_off < K
        x = tl.load(
            x_ptr + rm[:, None] * x_stride_m + k_off[None, :],
            mask=row_mask[:, None] & k_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        w = tl.load(
            w_ptr + rn[:, None] * w_stride_n + k_off[None, :],
            mask=col_mask[:, None] & k_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        if USE_IEEE:
            acc += tl.dot(x, tl.trans(w), input_precision="ieee")
        else:
            acc += tl.dot(x, tl.trans(w), input_precision="hf32")

    tl.store(
        logits_ptr + rm[:, None] * l_stride_m + rn[None, :],
        acc,
        mask=row_mask[:, None] & col_mask[None, :],
    )


@triton.jit
def _softmax_topk_kernel(
    logits_ptr,
    bias_ptr,
    out_w_ptr,
    out_id_ptr,
    l_stride_m,  # E (contiguous)
    M,
    N,
    SOFTCAP,
    TOPK: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    USE_SOFTCAP: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Softcap + bias + softmax + top-2 + gather over the full expert axis.

    Reads the [M, E] fp32 logits once; the whole expert axis is one
    [BLOCK_N] tile so the softmax, top-k and weight gather are all in
    registers. Padding experts (rn >= E) are -inf; padding rows are masked on
    every store.
    """
    pid = tl.program_id(0)
    rm = pid * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    rn = tl.arange(0, BLOCK_N)  # [BLOCK_N]
    row_mask = rm < M  # [BLOCK_M]
    col_mask = rn < N  # [BLOCK_N]

    NEG_INF = -float("inf")

    acc = tl.load(
        logits_ptr + rm[:, None] * l_stride_m + rn[None, :],
        mask=row_mask[:, None] & col_mask[None, :],
        other=NEG_INF,
    ).to(tl.float32)

    if USE_SOFTCAP:
        acc = SOFTCAP * (
            2.0 * (1.0 / (1.0 + tl.exp(-2.0 * acc / SOFTCAP))) - 1.0
        )

    if HAS_BIAS:
        b = tl.load(bias_ptr + rn, mask=col_mask, other=0.0).to(tl.float32)
        acc = acc + b[None, :]

    valid = row_mask[:, None] & col_mask[None, :]
    acc = tl.where(valid, acc, NEG_INF)

    # Softmax over the full expert axis.
    mx = tl.max(acc, axis=1)
    mx = tl.where(row_mask, mx, 0.0)
    e = tl.exp(acc - mx[:, None])
    e = tl.where(valid, e, 0.0)
    sm = tl.sum(e, axis=1)
    sm = tl.where(row_mask, sm, 1.0)
    probs = e / sm[:, None]

    # Top-k (k <= 2): iterative argmax, gather via masked sum.
    work = acc
    for pick in range(TOPK):
        idx = tl.argmax(work, axis=1)
        pick_mask = rn[None, :] == idx[:, None]
        w_pick = tl.sum(tl.where(pick_mask, probs, 0.0), axis=1)
        tl.store(out_w_ptr + rm * TOPK + pick, w_pick, mask=row_mask)
        tl.store(
            out_id_ptr + rm * TOPK + pick, idx.to(tl.int32), mask=row_mask
        )
        work = tl.where(pick_mask, NEG_INF, work)


def fused_moe_router_tensorcore(
    x, router_weight, topk, moe_softcapping, correction_bias=None
):
    """Fused MoE router (Tensor Core). See module docstring.

    Signature matches ``reference(x, router_weight, topk, moe_softcapping,
    correction_bias=None)``.
    """
    M, K = x.shape
    N = router_weight.shape[0]

    # Allocate the two outputs off ``x``'s tensor (x.new_empty is a hair faster
    # than torch.empty(..., device=x.device) on this device's launch path — it
    # avoids re-resolving the device object each call). These are the only
    # device tensors the op produces; everything else is fused into the kernel.
    topk_weights = x.new_empty((M, topk), dtype=torch.float32)
    topk_ids = x.new_empty((M, topk), dtype=torch.int32)

    use_ieee = x.dtype == torch.float32
    has_bias = correction_bias is not None
    use_softcap = moe_softcapping != 0.0
    softcap_val = float(moe_softcapping) if use_softcap else 0.0

    if M <= _FUSED_M_LIMIT:
        # ---- Launch-bound regime: single fused kernel --------------------
        # Fast path for the common full-axis bench shape (E=256, H=4096): the
        # config is fully determined by M here (BN=256, BK=64, 8 warps, kernel
        # stages=2, K-loop stages=4), so we set it inline instead of calling
        # the shape-keyed picker. This shaves ~12us of Python work from the
        # launcher; on this device the kernel launch is async, so CPU work
        # between back-to-back launches serialises them and shows up as GPU
        # idle time in the benchmark timer. Other shapes fall back to the
        # general picker.
        if N == 256 and K == 4096:
            # BM=8 for M<=64: measured ~5us faster than BM=16 here (leaner
            # per-program GEMM, more UB headroom for the in-register
            # softmax/topk). BM=32 for the larger M (amortises the 64-iter
            # K-loop).
            block_m = 8 if M <= 64 else 32
            block_n = 256
            block_k = 64
            num_warps = 8
            num_stages = 2
            kloop_stages = 4
        else:
            block_m, block_n, block_k, num_warps, num_stages, kloop_stages = (
                _fused_config(M, N, K)
            )
        grid = ((M + block_m - 1) // block_m,)
        _fused_router_kernel[grid](
            x,
            router_weight,
            (
                correction_bias if has_bias else x
            ),  # bias_ptr unused when !HAS_BIAS
            topk_weights,
            topk_ids,
            x.stride(0),
            router_weight.stride(0),
            M,
            N,
            K,
            softcap_val,
            TOPK=topk,
            HAS_BIAS=has_bias,
            USE_SOFTCAP=use_softcap,
            USE_IEEE=use_ieee,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_K=block_k,
            KLOOP_STAGES=kloop_stages,
            num_warps=num_warps,
            num_stages=num_stages,
        )
    else:
        # ---- Compute-bound regime: N-split GEMM + softmax/topk -----------
        logits = torch.empty((M, N), dtype=torch.float32, device=x.device)
        block_m, block_n, block_k, num_warps, num_stages = _gemm_config(
            M, N, K
        )
        grid_gemm = (
            (M + block_m - 1) // block_m,
            (N + block_n - 1) // block_n,
        )
        _logits_gemm_kernel[grid_gemm](
            x,
            router_weight,
            logits,
            x.stride(0),
            router_weight.stride(0),
            N,
            M,
            N,
            K,
            USE_IEEE=use_ieee,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_K=block_k,
            num_warps=num_warps,
            num_stages=num_stages,
        )

        sm_m, sm_n, sm_warps = _softmax_topk_config(M, N)
        grid_sm = ((M + sm_m - 1) // sm_m,)
        _softmax_topk_kernel[grid_sm](
            logits,
            (
                correction_bias if has_bias else x
            ),  # bias_ptr unused when !HAS_BIAS
            topk_weights,
            topk_ids,
            N,
            M,
            N,
            softcap_val,
            TOPK=topk,
            HAS_BIAS=has_bias,
            USE_SOFTCAP=use_softcap,
            BLOCK_M=sm_m,
            BLOCK_N=sm_n,
            num_warps=sm_warps,
            num_stages=2,
        )
    return topk_weights, topk_ids


__all__ = ["fused_moe_router_tensorcore"]
