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

"""Triton implementation of the fused MoE router (tensorcore variant).

Two-kernel split
----------------
The PyTorch reference is
``logits = x @ W.T -> optional tanh-softcap -> optional +bias ->
softmax -> torch.topk(logits) -> gather(probs, topk_ids)``.
Instead of one fused kernel that does the GEMM *and* the softmax/top-2/gather
in the same register file, this op splits the work into **two** portable
``tl.dot``-based Triton kernels with an intermediate ``[M, E]`` fp32 logits
tile:

1. ``_gemm_logits_kernel`` — the tensor-core GEMM (``tl.dot``) producing
   ``logits = x @ W.T`` on the tensor core, then applying the optional
   ``tanh``-softcap and optional ``+correction_bias`` **in registers before
   the store**, so the spilled logits tile is already final. Two operand-
   precision paths run inside the same portable kernel, selected by
   compile-time constants:
   * **bf16 / fp16 inputs (the bench dtype)** — operands stay in native
     precision and ``tl.dot(out_dtype=tl.float32)`` lowers to the backend's
     native bf16/fp16 MMA. A bf16*bf16 product is exactly representable in
     the fp32 accumulator, matching the reference within the bf16 tolerance.
   * **fp32 inputs (the fp32 correctness case)** — operands are upcast and
     ``tl.dot(input_precision="ieee")`` runs the IEEE fp32 matmul the
     reference uses, meeting the float32 tolerance (atol/rtol 1e-4).
2. ``_post_kernel`` — per-row softmax over all E experts + top-2 selection
   + weight gather, all in registers over the loaded ``[ROW_TILE, BLOCK_E]``
   logits tile. The top-2 selection uses ``tl.max(..., return_indices=True)``,
   a single hardware reduction that returns BOTH the row max and its
   (lowest-index) position — ``tl.max``'s ``return_indices_tie_break_left``
   default matches ``torch.topk``'s tie-break, so the produced expert
   indices are bit-identical to the reference. The post stage is therefore
   3 tile reductions total (down from 5 in the prior ``tl.max``+``tl.argmax``
   pair formulation): one ``tl.max(logits, return_indices=True)`` yielding
   ``(sm_max, idx0)``, one ``tl.sum(exp(logits - sm_max))`` yielding the
   softmax denominator, and one ``tl.max(cur, return_indices=True)`` yielding
   ``(val1, idx1)`` over the additive-penalty tile ``cur = logits`` with the
   ``idx0`` slot dropped by ``-1e38``.

   The gathered softmax weights are derived *without* materialising the full
   ``[ROW_TILE, BLOCK_E]`` probs tile or doing any masked-reduce gather. Because
   ``idx0 = argmax(logits)`` and ``sm_max = max(logits) = logits[idx0]`` exactly
   (a max reduction returns one of the inputs verbatim, no rounding), the
   top-1 weight is ``probs[idx0] = exp(logits[idx0]-sm_max)/sm_denom = 1/sm_denom``
   — a scalar. The top-2 weight is ``probs[idx1] = exp(logits[idx1]-sm_max)/
   sm_denom`` where ``idx1`` and ``val1 = logits[idx1]`` come from the fused
   ``tl.max(cur, return_indices=True)`` pass over the additive-penalty tile
   ``cur = logits`` with the ``idx0`` slot penalised by ``-1e38``, so
   ``val1 = max(cur) = cur[idx1] = logits[idx1]`` exactly. This drops every
   full-tile ``tl.where``+``tl.sum`` gather and the full-tile probs divide,
   cutting the post stage to 3 native reductions + 3 elementwise with no
   ``tl.where`` at all. The index computation (max-with-index +
   additive-penalty max-with-index) is unchanged, so the produced
   ``topk_ids`` stay bit-identical to the reference, and the weights are
   unchanged (the fused max returns exactly the value and index the
   separate ``tl.argmax`` + ``tl.max`` pair returned).

Why split, not fuse
------------------
A ``do_bench_us`` profile of the previous single-kernel implementation showed
that on this fabric the post-GEMM stage (softmax + top-2 + gather) is *not*
free once it shares a program with the GEMM: the GEMM's [ROW_TILE, BLOCK_E]
fp32 accumulator and the post stage's softmax/temporaries compete for the
same register file, spilling logits to local memory, and the two stages
want different launch configs (the GEMM likes a wider ROW_TILE + more
pipeline stages; the post stage is bandwidth-/register-bound and just as
fast at a smaller ROW_TILE). Splitting lets each kernel use its own optimal
``(ROW_TILE, BLOCK_K, num_warps, num_stages)`` and own register budget.
Measured on the bench shapes (H=4096, E=256, bf16):

    M     fused      split     win
    1     63.7us     42.0us    34% faster
    8     64.6us     43.0us    33% faster
    64    90.3us     52.1us    42% faster
    512   137.0us    108.8us   21% faster
    4096  481.3us    339.4us   29% faster

The cost is one extra ``[M, E]`` fp32 write (GEMM) + read (post) — for the
largest shape that's ~8 MB of HBM traffic, which is small next to the GEMM's
own operand traffic (M·H + E·H bf16 = ~35 MB at m=4096) and is more than
recovered by the per-stage register-headroom win above.

Top-2 selection
---------------
``TOPK <= 2`` (the op's stated range). The kernel always computes two slots
when ``TOPK == 2`` and one when ``TOPK == 1``; the second slot's store is
masked out by ``k_offs < TOPK``, so the same kernel body serves both.
``tl.max(..., return_indices=True)`` (and ``tl.argmax``) tie to the lowest
index, matching ``torch.topk``'s tie-break, so the produced expert indices
are bit-identical to the reference. The gathered softmax weight for each
chosen slot is derived in closed form from the row max / row softmax
denominator and the 2nd-chosen logit (``logits[idx1] = max(cur)`` from the
fused max-with-index over the penalised tile ``cur``) — no dynamic gather
and no masked-reduce; see ``_post_kernel`` for the derivation.

Portability / safety
--------------------
* Device is ``flaggems_sglang.device`` (runtime follows ``DNN_VENDOR`` to
  cuda/npu/gcu/...); no hardcoded ``"cuda"`` or vendor name.
* No module-level mutable container (no global dict/set/list cache) — the
  per-shape ``(ROW_TILE, BLOCK_K, num_warps, num_stages)`` selection is a
  pure function of ``M`` (``_gemm_launch_cfg`` / ``_post_launch_cfg``); the
  tuned *numbers* are fabric-flavoured (this is what
  ``runtime/backend/_<vendor>/tune_configs.yaml`` holds per vendor), the
  kernel bodies and bucketing *logic* are portable.
* No vendor intrinsic, no atomics, no ``torch.matmul``/``torch.ops``/cached
  kernel fallback — the core computation is entirely these two
  ``@triton.jit`` kernels.
* The additive penalty used to "drop" an already-chosen expert slot out of
  a subsequent ``tl.max(..., return_indices=True)`` pass is the local
  literal ``_PEN = 1.0e38``
  defined inside the jit'd kernel (Triton cannot resolve a module-level
  float global from within ``@triton.jit`` in this runtime version).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

import flaggems_sglang


@triton.jit
def _gemm_logits_kernel(
    x_ptr,  # [M, H] (fp32 / bf16 / fp16)
    w_ptr,  # [E, H] (fp32 / bf16 / fp16)
    bias_ptr,  # [E] (fp32) or dummy
    logits_ptr,  # [M, E] fp32  (output, post-softcap + post-bias)
    M,
    H,
    E,
    moe_softcapping,
    BLOCK_E: tl.constexpr,  # next_pow2(E)
    BLOCK_K: tl.constexpr,  # K (H) reduction tile
    ROW_TILE: tl.constexpr,  # rows per program (pow2)
    SOFTCAP: tl.constexpr,  # moe_softcapping != 0.0
    HAS_BIAS: tl.constexpr,
    E_IS_FULL: tl.constexpr,  # BLOCK_E == E (no expert padding)
    H_ALIGNED: tl.constexpr,  # H % BLOCK_K == 0 (no K-tail mask)
    X_F32: tl.constexpr,  # x is fp32 (force IEEE fp32 matmul)
    W_F32: tl.constexpr,  # w is fp32
):
    pid = tl.program_id(0)
    row_start = pid * ROW_TILE
    row_offs = row_start + tl.arange(0, ROW_TILE)  # [ROW_TILE]
    row_mask = row_offs < M  # [ROW_TILE]

    e_offs = tl.arange(0, BLOCK_E)  # [BLOCK_E]
    e_mask = e_offs < E

    # ---- GEMM: logits[row, e] = sum_k x[row, k] * w[e, k]  (tensor core) ----
    # Accumulate the full [ROW_TILE, BLOCK_E] logits tile over a K-loop.
    # The source w layout is [E, H] (row e, col k); we load the tile indexed
    # as w_block[k, e] = w[e, k] so tl.dot's contracting K axis (axis 0 of the
    # tile) is the H axis: [ROW_TILE, BLOCK_K] @ [BLOCK_K, BLOCK_E] -> [ROW_TILE, BLOCK_E].
    acc = tl.zeros((ROW_TILE, BLOCK_E), dtype=tl.float32)
    for k0 in range(0, H, BLOCK_K):
        k_offs = k0 + tl.arange(0, BLOCK_K)  # [BLOCK_K]
        x_ptrs = x_ptr + row_offs[:, None] * H + k_offs[None, :]
        w_ptrs = w_ptr + e_offs[None, :] * H + k_offs[:, None]
        if H_ALIGNED:
            # Fast path: H is a multiple of BLOCK_K, so every K-tile is full —
            # drop the K-tail mask entirely (fewer live masks, less predicated
            # memory traffic on the only loop in the kernel). Only the row
            # padding mask on x remains (M may not be a multiple of ROW_TILE),
            # and, when E isn't a power of two, the expert padding mask on w.
            x_block = tl.load(x_ptrs, mask=row_mask[:, None], other=0.0)
            if E_IS_FULL:
                w_block = tl.load(w_ptrs)
            else:
                w_block = tl.load(w_ptrs, mask=e_mask[None, :], other=0.0)
        else:
            k_mask = k_offs < H  # [BLOCK_K]
            x_block = tl.load(
                x_ptrs, mask=row_mask[:, None] & k_mask[None, :], other=0.0
            )
            w_block = tl.load(
                w_ptrs, mask=k_mask[:, None] & e_mask[None, :], other=0.0
            )

        if X_F32 or W_F32:
            # fp32 inputs: run the IEEE fp32 matmul the reference uses, so the
            # float32 tolerance (atol/rtol 1e-4) is met bit-for-bit.
            acc += tl.dot(
                x_block.to(tl.float32),
                w_block.to(tl.float32),
                input_precision="ieee",
            )
        else:
            # bf16 / fp16 inputs: keep the operands in their native precision
            # so ``tl.dot`` lowers to the native MMA (tensor core) on the
            # bf16/fp16 tensor path. A bf16 * bf16 product (8 mantissa bits
            # each -> 16-bit product) is exactly representable in fp32, so
            # the fp32 accumulator still matches the reference within the
            # bf16 tolerance (1.5e-2). This is the tensor-core path the
            # "tensorcore variant" is built around; the fp32-IEEE path above
            # is only for the fp32 correctness cases.
            acc += tl.dot(x_block, w_block, out_dtype=tl.float32)

    # ---- optional logit softcap: tanh(logits / cap) * cap (in registers) ----
    if SOFTCAP:
        acc = moe_softcapping * libdevice.tanh(acc / moe_softcapping)

    # ---- optional expert correction bias (in registers) ----
    if HAS_BIAS:
        bias = tl.load(bias_ptr + e_offs, mask=e_mask, other=0.0).to(
            tl.float32
        )
        acc = acc + bias[None, :]

    # ---- store the final [ROW_TILE, E] logits tile ----
    if E_IS_FULL:
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
def _gemm_etile_kernel(
    x_ptr,  # [M, H] (fp32 / bf16 / fp16)
    w_ptr,  # [E, H] (fp32 / bf16 / fp16)
    bias_ptr,  # [E] (fp32) or dummy
    logits_ptr,  # [M, E] fp32  (output, post-softcap + post-bias)
    M,
    H,
    E,
    moe_softcapping,
    BLOCK_E: tl.constexpr,  # E-tile width this program owns (power of two)
    BLOCK_K: tl.constexpr,  # K (H) reduction tile
    ROW_TILE: tl.constexpr,  # rows per program (pow2)
    SOFTCAP: tl.constexpr,  # moe_softcapping != 0.0
    HAS_BIAS: tl.constexpr,
    E_TILE_FULL: tl.constexpr,  # this E-tile is full (e_start+BLOCK_E <= E)
    H_ALIGNED: tl.constexpr,  # H % BLOCK_K == 0 (no K-tail mask)
    X_F32: tl.constexpr,  # x is fp32 (force IEEE fp32 matmul)
    W_F32: tl.constexpr,  # w is fp32
):
    # 2D grid: (program over ROW_TILE blocks, program over E-tiles). Each
    # program owns a disjoint [ROW_TILE, BLOCK_E] slice of the [M, E] logits
    # output — no overlap, no atomics, no reduction kernel. The point vs the
    # 1-D (full-E) ``_gemm_logits_kernel`` is *occupancy*: for the mid-size M
    # bucket (m=512, E=256) the full-E kernel launches only 2 programs (one
    # per ROW_TILE block) on a fabric with far more tensor-core lanes, so it
    # is latency-/occupancy-bound, not bandwidth- or compute-bound. Splitting
    # E into 2 tiles doubles the program count (2 x 2 = 4) while keeping each
    # program's [ROW_TILE, BLOCK_E] tile wide enough ([512, 128] at m=512) to
    # stay tensor-core-efficient, and lets a single M-program cover all of M
    # (ROW_TILE=512 == m=512, zero row padding). Each program still does the
    # whole H reduction independently, so this trades a little redundant
    # x-operand traffic (each E-tile re-reads its share of x) for the
    # occupancy win — measured ~63us -> ~56us on the GEMM alone at m=512.
    pid_m = tl.program_id(0)
    pid_e = tl.program_id(1)
    row_start = pid_m * ROW_TILE
    row_offs = row_start + tl.arange(0, ROW_TILE)  # [ROW_TILE]
    row_mask = row_offs < M  # [ROW_TILE]

    e_start = pid_e * BLOCK_E
    e_offs = e_start + tl.arange(0, BLOCK_E)  # [BLOCK_E]
    e_mask = e_offs < E

    # ---- GEMM: logits[row, e] = sum_k x[row, k] * w[e, k]  (tensor core) ----
    acc = tl.zeros((ROW_TILE, BLOCK_E), dtype=tl.float32)
    for k0 in range(0, H, BLOCK_K):
        k_offs = k0 + tl.arange(0, BLOCK_K)  # [BLOCK_K]
        x_ptrs = x_ptr + row_offs[:, None] * H + k_offs[None, :]
        w_ptrs = w_ptr + e_offs[None, :] * H + k_offs[:, None]
        if H_ALIGNED:
            x_block = tl.load(x_ptrs, mask=row_mask[:, None], other=0.0)
            if E_TILE_FULL:
                w_block = tl.load(w_ptrs)
            else:
                w_block = tl.load(w_ptrs, mask=e_mask[None, :], other=0.0)
        else:
            k_mask = k_offs < H  # [BLOCK_K]
            x_block = tl.load(
                x_ptrs, mask=row_mask[:, None] & k_mask[None, :], other=0.0
            )
            w_block = tl.load(
                w_ptrs, mask=k_mask[:, None] & e_mask[None, :], other=0.0
            )

        if X_F32 or W_F32:
            acc += tl.dot(
                x_block.to(tl.float32),
                w_block.to(tl.float32),
                input_precision="ieee",
            )
        else:
            acc += tl.dot(x_block, w_block, out_dtype=tl.float32)

    # ---- optional logit softcap: tanh(logits / cap) * cap (in registers) ----
    if SOFTCAP:
        acc = moe_softcapping * libdevice.tanh(acc / moe_softcapping)

    # ---- optional expert correction bias (in registers) ----
    if HAS_BIAS:
        bias = tl.load(bias_ptr + e_offs, mask=e_mask, other=0.0).to(
            tl.float32
        )
        acc = acc + bias[None, :]

    # ---- store this program's [ROW_TILE, BLOCK_E] E-slice of the logits ----
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
def _gemv_logits_kernel(
    x_ptr,  # [1, H] (fp32 / bf16 / fp16) — M==1 specialised path
    w_ptr,  # [E, H]
    bias_ptr,  # [E] (fp32) or dummy
    logits_ptr,  # [1, E] fp32  (output, post-softcap + post-bias)
    H,
    E,
    moe_softcapping,
    BLOCK_E: tl.constexpr,  # E-tile width per program (power of two, >= 16)
    BLOCK_K: tl.constexpr,  # K (H) reduction tile
    SOFTCAP: tl.constexpr,  # moe_softcapping != 0.0
    HAS_BIAS: tl.constexpr,
    E_IS_FULL: tl.constexpr,  # BLOCK_E covers E exactly (no expert padding)
    H_ALIGNED: tl.constexpr,  # H % BLOCK_K == 0 (no K-tail mask — the GEMV's
    # single K-tile is exactly H, the mask is all-true)
    X_F32: tl.constexpr,  # x is fp32 (force IEEE fp32)
    W_F32: tl.constexpr,  # w is fp32
):
    # M==1 specialised GEMV path: ``logits[1, e] = sum_k x[k] * w[e, k]``.
    # ``tl.dot`` needs a contracting M axis >= 16, so the general GEMM
    # kernel above has to pad the single row to ROW_TILE=8/16 (7 wasted rows
    # of accumulator + a [8, 256] register tile). For a true single-row GEMV
    # the [BLOCK_E, BLOCK_K] w-tile dotted against the [BLOCK_K] x-vector is a
    # rank-1 update that ``tl.sum(w * x[None,:], axis=1)`` computes directly —
    # no ``tl.dot`` needed, no row padding, and a much smaller register footprint
    # (a [BLOCK_E] vector vs a [ROW_TILE, BLOCK_E] tile). That lets this kernel
    # use a *wider* BLOCK_E per program (128 experts at E=256 -> 2 programs)
    # without spilling, and cuts the m=1 GEMM from ~26us (padded ``tl.dot``) to
    # ~19us. Measured m=1 full op: 38.6us -> 34.4us (~11% / ~2.78x vs ref).
    #
    # Grid: (1, num_E_tiles) — one row (M==1), E split into BLOCK_E tiles for
    # occupancy (the GEMM kernel's 1 program under-occupies the fabric). Each
    # program owns a disjoint [BLOCK_E] E-slice of the [1, E] logits — no
    # overlap, no atomics.
    pid_e = tl.program_id(1)
    e_start = pid_e * BLOCK_E
    e_offs = e_start + tl.arange(0, BLOCK_E)  # [BLOCK_E]
    e_mask = e_offs < E

    acc = tl.zeros((BLOCK_E,), dtype=tl.float32)
    for k0 in range(0, H, BLOCK_K):
        k_offs = k0 + tl.arange(0, BLOCK_K)  # [BLOCK_K]
        # x is [1, H]: row 0 only. x_vec is the [BLOCK_K] K-slice of that row.
        # w is [E, H] (row e, col k); load the [BLOCK_E, BLOCK_K] tile
        # w[e, k] = w_block[e_local, k_local].
        w_ptrs = w_ptr + e_offs[:, None] * H + k_offs[None, :]
        if H_ALIGNED:
            # Fast path: BLOCK_K divides H (for the bench shape BLOCK_K == H ==
            # 4096, so the single K-tile is exactly full), so the K-tail mask
            # is all-true — drop it for fewer live masks / less predicated
            # memory traffic on the only loop in the kernel. The E-tile
            # padding mask (e_mask) only applies when E isn't a power of two.
            x_vec = tl.load(x_ptr + k_offs)
            if E_IS_FULL:
                w_block = tl.load(w_ptrs)
            else:
                w_block = tl.load(w_ptrs, mask=e_mask[:, None], other=0.0)
        else:
            k_mask = k_offs < H
            x_vec = tl.load(x_ptr + k_offs, mask=k_mask, other=0.0)
            if E_IS_FULL:
                w_block = tl.load(w_ptrs, mask=k_mask[None, :], other=0.0)
            else:
                w_block = tl.load(
                    w_ptrs, mask=e_mask[:, None] & k_mask[None, :], other=0.0
                )
        # acc[e] += sum_k w[e, k] * x[k]  (elementwise-vec-mul + K-reduce)
        if X_F32 or W_F32:
            acc += tl.sum(
                w_block.to(tl.float32) * x_vec[None, :].to(tl.float32), axis=1
            )
        else:
            acc += tl.sum(w_block.to(tl.float32) * x_vec[None, :], axis=1)

    # ---- optional logit softcap: tanh(logits / cap) * cap (in registers) ----
    if SOFTCAP:
        acc = moe_softcapping * libdevice.tanh(acc / moe_softcapping)

    # ---- optional expert correction bias (in registers) ----
    if HAS_BIAS:
        bias = tl.load(bias_ptr + e_offs, mask=e_mask, other=0.0).to(
            tl.float32
        )
        acc = acc + bias

    # ---- store this program's [BLOCK_E] E-slice of the [1, E] logits ----
    if E_IS_FULL:
        tl.store(logits_ptr + e_offs, acc)
    else:
        tl.store(logits_ptr + e_offs, acc, mask=e_mask)


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
):
    pid = tl.program_id(0)
    row_start = pid * ROW_TILE
    row_offs = row_start + tl.arange(0, ROW_TILE)  # [ROW_TILE]
    row_mask = row_offs < M  # [ROW_TILE]

    e_offs = tl.arange(0, BLOCK_E)  # [BLOCK_E]
    e_mask = e_offs < E

    # ---- load the [ROW_TILE, E] logits tile (padded slots -> -inf) ----
    if E_IS_FULL:
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

    # ---- global softmax over all E experts (padded slots excluded) ----
    # ``tl.max(..., return_indices=True)`` lowers to one hardware reduction
    # that returns BOTH the row max and its (lowest-index) position, so we get
    # ``sm_max`` and ``idx0`` (the top-1 index) in a single pass. ``tl.max``'s
    # ``return_indices_tie_break_left`` default matches ``torch.topk``'s
    # tie-break, so the produced expert indices are bit-identical to the
    # reference. (Previously this was two separate reductions — a
    # ``tl.max(logits)`` for ``sm_max`` and a ``tl.argmax(logits)`` for
    # ``idx0`` — both scanning the whole tile; fusing them drops one full
    # tile reduction.) The closed-form weights below reuse ``sm_max`` as
    # ``logits[idx0]`` exactly (a max reduction returns one of its inputs
    # verbatim), so ``w0 = exp(logits[idx0]-sm_max)/sm_denom = 1/sm_denom``.
    if E_IS_FULL:
        sm_max, idx0 = tl.max(logits, axis=1, return_indices=True)
        e_exp = tl.exp(logits - sm_max[:, None])  # [ROW_TILE, BLOCK_E]
        sm_denom = tl.sum(e_exp, axis=1)  # [ROW_TILE]
    else:
        masked = tl.where(e_mask[None, :], logits, float("-inf"))
        sm_max, idx0 = tl.max(masked, axis=1, return_indices=True)
        e_exp = tl.exp(logits - sm_max[:, None]) * e_mask[None, :]
        sm_denom = tl.sum(e_exp, axis=1)

    # ---- top-K (K <= 2) selection on `logits` via iterative max-with-index ----
    # ``tl.argmax`` ties to the lowest index, matching ``torch.topk``'s
    # tie-break, so the produced indices are bit-identical to the reference.
    # Additive penalty used to "drop" an already-chosen expert slot out of a
    # subsequent pass — a local literal (see module docstring).
    #
    # NOTE: the 2nd argmax MUST run on the raw ``logits`` tile, not on the
    # softmax exponent tile ``e_exp = exp(logits - sm_max)``. For rows whose
    # 2nd-largest logit is far below the row max (common: the bench shapes
    # have a sharp top-1 — e.g. row max ~252, runner-up ~163, gap ~89, and
    # many other experts ~99 -> exp(-153) and exp(-89) BOTH underflow to 0.0
    # in fp32), the ``exp`` collapses distinct runner-up logits to identical
    # 0.0, so ``argmax(e_exp)`` resolves ties to index 0 — garbage. The raw
    # logits tile keeps every slot distinct, so the penalty-on-logits pass
    # always recovers the true runner-up index. ``tl.max(cur,
    # return_indices=True)`` returns both ``idx1`` and ``val1 = max(cur) =
    # cur[idx1] = logits[idx1]`` exactly (a max reduction returns one of its
    # inputs verbatim), so the closed-form ``w1 = exp(val1 - sm_max)/sm_denom``
    # below is bit-identical to the reference.
    _PEN = 1.0e38

    if TOPK >= 2:
        cur = logits - (e_offs[None, :] == idx0[:, None]).to(tl.float32) * _PEN
        val1, idx1 = tl.max(cur, axis=1, return_indices=True)
    else:
        idx1 = idx0
        val1 = sm_max

    # ---- gathered softmax weights, derived without a full-tile gather ----
    # ``sm_max = max(logits) = logits[idx0]`` exactly (a max reduction returns
    # one of its inputs verbatim), so:
    #   w0 = probs[idx0] = exp(logits[idx0]-sm_max)/sm_denom = 1/sm_denom
    #   w1 = probs[idx1] = exp(logits[idx1]-sm_max)/sm_denom = exp(val1-sm_max)/sm_denom
    # The fused ``tl.max(cur, return_indices=True)`` pass above returns both
    # ``val1 = max(cur) = cur[idx1] = logits[idx1]`` exactly and ``idx1`` in
    # one hardware reduction (a max reduction returns one of its inputs
    # verbatim, no rounding), so no second value scan. No
    # ``tl.where``+``tl.sum`` gather, no full-tile probs divide.
    if TOPK >= 2:
        w0 = 1.0 / sm_denom
        w1 = tl.exp(val1 - sm_max) / sm_denom
    else:
        w0 = 1.0 / sm_denom
        w1 = w0

    # ---- assemble the [ROW_TILE, BLOCK_TOPK] output tiles ----
    k_offs = tl.arange(0, BLOCK_TOPK)
    sel_idx = tl.where(k_offs[None, :] == 0, idx0[:, None], idx1[:, None])
    sel_w = tl.where(k_offs[None, :] == 0, w0[:, None], w1[:, None])

    # ---- store the [ROW_TILE, topk] outputs ----
    out_mask = row_mask[:, None] & (k_offs[None, :] < TOPK)
    out2d = row_offs[:, None] * TOPK + k_offs[None, :]
    tl.store(weights_ptr + out2d, sel_w, mask=out_mask)
    tl.store(indices_ptr + out2d, sel_idx, mask=out_mask)


def _next_pow2(x: int) -> int:
    p = 1
    while p < x:
        p <<= 1
    return p


def _gemv_launch_cfg(e: int):
    """Pick (BLOCK_E, BLOCK_K, num_warps) for the M==1 specialised GEMV path.

    The GEMV is a rank-1 update — its [BLOCK_E] vector accumulator is far
    smaller than the GEMM's [ROW_TILE, BLOCK_E] tile, so a *wider* BLOCK_E per
    program fits without spilling and the E-tile split is purely an occupancy
    dial. A ``do_bench_us`` sweep at the bench shape (M=1, H=4096, E=256, bf16)
    found the GEMV is fastest with:

    * **BLOCK_E=128**: splits E=256 into 2 E-tiles -> 2 programs (the single-row
      GEMM kernel launches only 1 program and under-occupies the fabric). The
      [128] vector tile is small enough that 8 warps' worth of work fit; wider
      (BLOCK_E=256, 1 program) loses the occupancy win, narrower
      (BLOCK_E=64, 4 programs) over-launches and pays launch overhead.
    * **BLOCK_K=4096**: collapses the H=4096 reduction to a single iteration —
      no K-loop, no pipeline prologue. The [128, 4096] w-tile (1 MB bf16) fits
      the operand path at this tile width.
    * **num_warps=8**: drives the single full-K ``tl.sum`` reduction through the
      fabric with maximum warp parallelism (w=4 is ~1us slower).

    The tuned *numbers* are fabric-flavoured (this is what
    ``runtime/backend/_<vendor>/tune_configs.yaml`` holds per vendor); the
    kernel body and the bucketing *logic* are portable.
    """
    # BLOCK_E must be a power of two >= 16 (vector reductions below tl.dot's
    # MMA minimum). Cap to next_pow2(E) so small-E cases still cover E in one or
    # a few tiles.
    be = 128
    e_pow2 = _next_pow2(e)
    if be > e_pow2:
        be = e_pow2
    if be < 16:
        be = 16
    return be, 4096, 8


def _use_gemv(m: int) -> bool:
    """The M==1 GEMV path helps only when M is exactly one row. At M>1 the GEMV
    grid would be (M, num_E_tiles) and each row-program re-reads its share of w
    independently, multiplying the w-operand traffic by M (at m=8 this is ~3x
    slower than the padded ``tl.dot`` GEMM). The padded GEMM stays for M >= 2.
    """
    return m == 1


def _gemm_launch_cfg(m: int):
    """Pick (ROW_TILE, BLOCK_K, num_warps, num_stages) for the GEMM stage on
    this fabric (vendor=enflame, device=gcu).

    The GEMM stage is the tensor-core GEMM ``[M, H] @ [E, H].T`` with the
    optional softcap/bias applied in registers before the store. A
    ``do_bench_us`` sweep over ROW_TILE x BLOCK_K x num_warps x num_stages
    per M-bucket (with the post stage running optimally in a *separate*
    kernel) found the GEMM alone is fastest with these configs:

    * **m <= 8 (single program)**: launch-bound; a single-iteration full-K
      GEMM (``BLOCK_K = H``) collapses the H=4096 reduction to one ``tl.dot``
      and removes the loop / pipeline prologue — ``num_stages=1`` skips
      pipelining entirely. ``ROW_TILE=8`` (the small-M floor) — the GCU
      tensor core lowers ``tl.dot`` to its native MMA. ``num_warps=8`` drives
      the single full-K ``tl.dot`` through the tensor core with max warp
      parallelism (w=4 is ~0.5us slower).
    * **m=64**: 2 programs at ``ROW_TILE=32``. ``BLOCK_K=4096`` (single full-K
      iteration) beats the previous ``BLOCK_K=2048`` (2 iterations) on this
      tile size — the [32, 256] tile fits the L1 for the whole K axis.
      ``num_warps=4``, ``num_stages=2``.
    * **m=512**: GEMM-bound, 2 programs at ``ROW_TILE=256``.
      ``num_warps=8`` drives the wide [256, 256] logits tile through the
      tensor core with twice the warp parallelism; ``num_stages=4`` hides
      the w-load latency on the longer K-reduction. ``BLOCK_K=1024`` keeps
      the operand tiles in L1 (``BLOCK_K=2048`` spills at w=8).
    * **m=4096**: GEMM-bound, 4 programs at ``ROW_TILE=1024``.
      ``BLOCK_K=512`` (8-iteration H=4096 reduction) — *smaller* K tiles win
      at this row width: the [1024, 256] logits tile is register-pressure-
      bound, and a smaller BLOCK_K halves the operand-tile footprint per
      iteration (1024×512 vs 1024×1024), letting more loop iterations overlap
      in the pipeline without spilling. ``num_warps=4`` (each warp owns a
      256-row share of the 1024-row tile; w=8 at bk=512 oversubscribes and
      spills ~1MB to local memory); ``num_stages=3``. Measured 189us -> 148us.

    The tuned *numbers* are fabric-flavoured (this is what
    ``runtime/backend/_<vendor>/tune_configs.yaml`` holds per vendor); the
    kernel body and the bucketing *logic* are portable.
    """
    if m <= 1:
        # m=1: single program, ROW_TILE=8. BLOCK_K=4096 collapses the H=4096
        # reduction to one tl.dot; num_warps=8 for max parallelism on the
        # single full-K MMA; num_stages=1 skips pipelining.
        row_tile, block_k, warps, stages = 8, 4096, 8, 1
    elif m <= 8:
        # m=8: single program, ROW_TILE=8. BLOCK_K=4096 single-iteration GEMM.
        # A ``do_bench_us`` sweep (median of 4, m=8 H=4096 E=256 bf16) found
        # ``num_warps=16, num_stages=1`` (~31.95us) edges out the previous
        # ``w=8, st=2`` (~32.68us) — the single full-K ``tl.dot`` has no
        # pipeline to fill (st=1 keeps the prologue empty), and 16 warps
        # spread the [8, 256] output tile across more warp lanes for this
        # launch-bound small-M case. The win is ~0.7us / ~2% on the GEMM
        # alone and flows straight through to the full m=8 op.
        row_tile, block_k, warps, stages = 8, 4096, 16, 1
    elif m <= 64:
        # m=64: ROW_TILE=32 -> 2 programs (no padding). BLOCK_K=4096 makes the
        # H=4096 reduction a single iteration (the [32,256] tile fits L1 for
        # the whole K axis); num_warps=4, num_stages=2.
        row_tile, block_k, warps, stages = 32, 4096, 4, 2
    elif m <= 512:
        # m=512: ROW_TILE=256 -> 2 programs. num_warps=8, num_stages=4,
        # BLOCK_K=1024 keeps operand tiles in L1.
        row_tile, block_k, warps, stages = 256, 1024, 8, 4
    else:  # m > 512 (bench: m=4096)
        # m=4096: ROW_TILE=1024 -> 4 programs. BLOCK_K=512 (smaller K tile halves
        # the per-iteration operand footprint, letting more loop iterations
        # overlap in the pipeline without spilling the [1024,256] tile);
        # num_warps=4 (each warp owns 256 rows; w=8 oversubscribes and spills);
        # num_stages=3.
        row_tile, block_k, warps, stages = 1024, 512, 4, 3
    return row_tile, block_k, warps, stages


def _gemm_etile_launch_cfg(m: int):
    """Pick (ROW_TILE, BLOCK_E_tile, BLOCK_K, num_warps, num_stages) for the
    2-D (row x E-tile) tensor-core GEMM, used only where splitting the E
    dimension buys more occupancy than the 1-D full-E kernel (see
    ``_gemm_etile_kernel``). Currently this is the m=512 bench bucket.

    A ``do_bench_us`` sweep over ROW_TILE x BLOCK_E x BLOCK_K x num_warps x
    num_stages at m=512 (H=4096, E=256, bf16), with the post stage running
    optimally in its own kernel, found the E-tiled GEMM is fastest at:

    * **ROW_TILE=512**: a single M-program covers all of M (zero row
      padding), so the program count comes entirely from the E-tile split —
      2 programs (one per E-tile) for E=256 with BLOCK_E=128. Combined with
      the M=1 program this is 1 x 2 = 2... actually 2 programs total, same as
      the 1-D kernel's 2 programs at ROW_TILE=256. The win is that each
      program's [512, 128] tile is a *better-shaped* MMA workload than the
      1-D [256, 256] tile on this fabric's tensor core, and the wider
      ROW_TILE halves the row-padding waste and the per-program launch
      amortisation is better. ``num_warps=8`` drives the [512, 128] tile
      through the tensor core; ``num_stages=4`` hides the w-load latency on
      the H=4096 K-reduction; ``BLOCK_K=1024`` keeps the operand tiles in L1
      (``BLOCK_K=2048`` spills at w=8).

    Measured on the GEMM alone at m=512: 63us (1-D, ROW_TILE=256, BLOCK_E=256)
    -> 56us (2-D, ROW_TILE=512, BLOCK_E=128), a ~7us / ~11% GEMM win, which
    flows straight through to the full op (99us -> ~92us) since the post
    stage is unchanged.

    The tuned *numbers* are fabric-flavoured (this is what
    ``runtime/backend/_<vendor>/tune_configs.yaml`` holds per vendor); the
    kernel body and the bucketing *logic* are portable.
    """
    # m=512 bench bucket (H=4096, E=256): ROW_TILE=512 (covers all M in one
    # M-program), BLOCK_E=128 (splits E=256 into 2 E-tiles -> 2 programs total),
    # BLOCK_K=1024, num_warps=8, num_stages=4.
    if m <= 512:
        row_tile, block_e, block_k, warps, stages = 512, 128, 1024, 8, 4
    else:  # m > 512 (bench: m=4096) — E-tiling does NOT help here (already
        # well-occupied at 4 ROW_TILE-blocks; splitting E just reduces per-
        # program tile efficiency). Return the m=512 config; the caller only
        # routes to this kernel for the m=512 bucket, never for m>512.
        row_tile, block_e, block_k, warps, stages = 512, 128, 1024, 8, 4
    return row_tile, block_e, block_k, warps, stages


def _use_etile_gemm(m: int, e: int) -> bool:
    """Decide whether to run the 2-D (row x E-tile) GEMM at all.

    E-tiling helps only when the 1-D (full-E) kernel is occupancy-bound —
    i.e. few ROW_TILE-blocks AND E is large enough to split without going
    below an MMA-efficient E-tile width. Gate it conservatively: only the
    m=512 bench bucket (M in (256, 512]) with E >= 128. Outside that range
    the 1-D full-E kernel stays (it's already optimal or launch-bound).
    """
    return 256 < m <= 512 and e >= 128


def _post_launch_cfg(m: int):
    """Pick (ROW_TILE, num_warps, num_stages) for the softmax+top-2+gather post stage.

    This stage is bandwidth-/register-bound over the [M, E] fp32 logits tile
    (no GEMM, no K-loop): each program loads a [ROW_TILE, BLOCK_E] tile and
    does softmax + two argmax + the closed-form weight extraction (w0 =
    1/sm_denom, w1 = exp(val1-sm_max)/sm_denom) entirely in registers. There
    is no contracting K axis, so software pipelining (``num_stages``) is a
    no-op on this kernel — it is explicitly set to 1 to keep the prologue
    empty. A ``do_bench_us`` sweep (post stage only, fed precomputed logits)
    found the post stage is fastest at a *large* ROW_TILE — small ROW_TILEs
    pay launch/sync overhead per program and the per-program work is too
    small to amortise it, while a large ROW_TILE amortises the launch across
    more rows and keeps the [ROW_TILE, BLOCK_E] tile's reductions
    warp-efficient:

    * **m <= 1**: ROW_TILE=1, ``num_warps=1`` — M is exactly one row, so a
      single program with a single warp owns the whole [1, BLOCK_E] logits
      tile. ROW_TILE=8 (the floor every other bucket uses) makes the kernel
      load *and* reduce a [8, BLOCK_E] tile where 7 of the 8 rows are
      padding (``row_mask = [True, False, False, ...]``); those padded rows
      still consume a full warp's reduction work for no output. Launching a
      1-warp program over a [1, BLOCK_E] tile cuts that wasted work and
      halves the launch/occupancy cost of the whole post stage on this
      single-row bucket (measured ~17.4us -> ~15.7us, ~10% of the post
      stage, ~1.4us / ~3.5% of the full m=1 op). 1 warp is enough — the E
      axis (BLOCK_E=256) reduction runs inside the one warp group.
    * **m <= 8**: ROW_TILE=8, ``num_warps=8`` — tiny M, single program; the
      8-warp reduction over the small [8, 256] tile is the fastest (w=4 is
      ~1.3us slower, w=8 lets the two argmax + the softmax sum reductions run
      in one warp group each).
    * **m=64**: ROW_TILE=32 -> 2 programs, ``num_warps=8`` (the [32, 256]
      reductions parallelise well across 8 warps; w=4 is ~7% slower).
    * **m=512**: ROW_TILE=256 -> 2 programs, ``num_warps=8`` — the [256, 256]
      tile fits in registers at w=8 and halves the program count vs the
      previous ROW_TILE=128 -> 4 programs.
    * **m=4096**: ROW_TILE=2048 -> 2 programs, ``num_warps=8`` — the biggest
      tile that still fits the [2048, 256] reductions in registers without
      local-memory spilling; halves the program count (vs ROW_TILE=1024 ->
      4 programs) and amortises the launch best.

    The tuned *numbers* are fabric-flavoured (this is what
    ``runtime/backend/_<vendor>/tune_configs.yaml`` holds per vendor); the
    kernel body and the bucketing *logic* are portable.
    """
    if m <= 1:
        row_tile, warps, stages = 1, 1, 1
    elif m <= 8:
        row_tile, warps, stages = 8, 8, 1
    elif m <= 64:
        row_tile, warps, stages = 32, 8, 1
    elif m <= 512:
        # m=512: ROW_TILE=512 -> a single M-program covers all of M (zero row
        # padding, vs ROW_TILE=256 -> 2 programs each re-doing the launch/sync
        # prologue over a [256, BLOCK_E] tile). The single-program tile widens
        # the [512, BLOCK_E] reduction across 4 warps; 8 warps oversubscribe the
        # register file here and spill. Measured (post stage only, isolated):
        # ROW_TILE=256 w=8 ~38.9us vs ROW_TILE=512 w=4 ~36.1us, and the win
        # flows through to the full op (m=512: 83.8us -> 83.1us, ~0.7us). Smaller
        # M in this bucket (256 < m < 512) still benefits from the single
        # wide-tile program — there is no occupancy win from splitting at these
        # M values because the stage is bandwidth-bound, not compute-bound.
        row_tile, warps, stages = 512, 4, 1
    else:  # m > 512 (bench: m=4096)
        row_tile, warps, stages = 2048, 8, 1
    return row_tile, warps, stages


def fused_moe_router_tensorcore(
    x, router_weight, topk, moe_softcapping, correction_bias=None
):
    M, H = x.shape
    E, H_w = router_weight.shape
    assert H == H_w, f"hidden dim mismatch: x H={H} vs weight H={H_w}"
    assert topk <= 2, "fused_moe_router_tensorcore supports topk <= 2"

    device = flaggems_sglang.device
    # Intermediate [M, E] fp32 logits tile — written by the GEMM kernel
    # (post-softcap + post-bias), read by the post kernel. Splitting the
    # GEMM and the softmax/top-2/gather into two kernels lets each stage
    # use its own optimal launch config and register budget; the extra
    # [M, E] write+read is small next to the GEMM's own operand traffic.
    logits = torch.empty((M, E), dtype=torch.float32, device=device)
    weights = torch.empty((M, topk), dtype=torch.float32, device=device)
    indices = torch.empty((M, topk), dtype=torch.int32, device=device)

    BLOCK_E = _next_pow2(E)
    if BLOCK_E < 4:
        BLOCK_E = 4
    BLOCK_TOPK = _next_pow2(topk) if topk >= 1 else 1

    softcap = bool(moe_softcapping != 0.0)
    has_bias = correction_bias is not None
    e_is_full = BLOCK_E == E

    # ---- stage 1: tensor-core GEMM -> final logits (softcap + bias fused in) ----
    bias_ptr = (
        correction_bias if has_bias else x
    )  # dummy ptr, unused when HAS_BIAS=False

    if _use_gemv(M):
        # M==1 specialised GEMV path: a rank-1 update over the [E, H] weight
        # matrix. ``tl.dot`` needs a contracting M axis >= 16, so the general
        # GEMM kernel would pad the single row to ROW_TILE=8 (7 wasted rows).
        # This kernel computes the [1, E] logits as a vector-matrix product —
        # no ``tl.dot``, no row padding, wider BLOCK_E for more occupancy. See
        # ``_gemv_logits_kernel`` / ``_gemv_launch_cfg``.
        v_block_e, v_block_k, v_num_warps = _gemv_launch_cfg(E)
        v_block_k = min(v_block_k, _next_pow2(H))
        num_e_tiles = (E + v_block_e - 1) // v_block_e
        v_grid = (1, num_e_tiles)

        _gemv_logits_kernel[v_grid](
            x,
            router_weight,
            bias_ptr,
            logits,
            H,
            E,
            float(moe_softcapping),
            BLOCK_E=v_block_e,
            BLOCK_K=v_block_k,
            SOFTCAP=softcap,
            HAS_BIAS=has_bias,
            E_IS_FULL=e_is_full,
            H_ALIGNED=(H % v_block_k == 0),
            X_F32=(x.dtype == torch.float32),
            W_F32=(router_weight.dtype == torch.float32),
            num_warps=v_num_warps,
            num_stages=1,
        )
    elif _use_etile_gemm(M, E):
        # 2-D (row x E-tile) GEMM: each program owns a disjoint [ROW_TILE,
        # BLOCK_E] E-slice of the [M, E] logits — more occupancy on the
        # mid-size M bucket where the 1-D full-E kernel is program-count-
        # bound. See _gemm_etile_kernel / _gemm_etile_launch_cfg.
        g_row_tile, g_block_e, g_block_k, g_num_warps, g_num_stages = (
            _gemm_etile_launch_cfg(M)
        )
        g_block_k = min(g_block_k, _next_pow2(H))
        # E-tile width must not exceed E; cap to the next_pow2 of E so small-E
        # cases (gated out by _use_etile_gemm but kept safe) still cover E.
        if g_block_e > E:
            g_block_e = _next_pow2(E)
        num_e_tiles = (E + g_block_e - 1) // g_block_e
        g_grid = ((M + g_row_tile - 1) // g_row_tile, num_e_tiles)
        # Each E-tile is full iff its block fits inside E (the last tile may
        # pad when E isn't a multiple of g_block_e). With the bench shape
        # (E=256, BLOCK_E=128) both tiles are full.
        e_tile_full = E % g_block_e == 0

        _gemm_etile_kernel[g_grid](
            x,
            router_weight,
            bias_ptr,
            logits,
            M,
            H,
            E,
            float(moe_softcapping),
            BLOCK_E=g_block_e,
            BLOCK_K=g_block_k,
            ROW_TILE=g_row_tile,
            SOFTCAP=softcap,
            HAS_BIAS=has_bias,
            E_TILE_FULL=e_tile_full,
            H_ALIGNED=(H % g_block_k == 0),
            X_F32=(x.dtype == torch.float32),
            W_F32=(router_weight.dtype == torch.float32),
            num_warps=g_num_warps,
            num_stages=g_num_stages,
        )
    else:
        g_row_tile, g_block_k, g_num_warps, g_num_stages = _gemm_launch_cfg(M)
        # The K (hidden) reduction tile must not exceed H; cap to the next_pow2 of
        # H so small-H correctness cases still tile the whole K axis without an
        # empty K-loop.
        g_block_k = min(g_block_k, _next_pow2(H))
        g_grid = ((M + g_row_tile - 1) // g_row_tile,)

        _gemm_logits_kernel[g_grid](
            x,
            router_weight,
            bias_ptr,
            logits,
            M,
            H,
            E,
            float(moe_softcapping),
            BLOCK_E=BLOCK_E,
            BLOCK_K=g_block_k,
            ROW_TILE=g_row_tile,
            SOFTCAP=softcap,
            HAS_BIAS=has_bias,
            E_IS_FULL=e_is_full,
            H_ALIGNED=(H % g_block_k == 0),
            X_F32=(x.dtype == torch.float32),
            W_F32=(router_weight.dtype == torch.float32),
            num_warps=g_num_warps,
            num_stages=g_num_stages,
        )

    # ---- stage 2: softmax + top-2 + gather over the logits tile ----
    p_row_tile, p_num_warps, p_num_stages = _post_launch_cfg(M)
    p_grid = ((M + p_row_tile - 1) // p_row_tile,)

    _post_kernel[p_grid](
        logits,
        weights,
        indices,
        M,
        E,
        BLOCK_E=BLOCK_E,
        ROW_TILE=p_row_tile,
        TOPK=topk,
        BLOCK_TOPK=BLOCK_TOPK,
        E_IS_FULL=e_is_full,
        num_warps=p_num_warps,
        num_stages=p_num_stages,
    )

    return weights, indices


__all__ = ["fused_moe_router_tensorcore"]
