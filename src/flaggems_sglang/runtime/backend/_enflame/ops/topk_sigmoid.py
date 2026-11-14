# Copyright 2026, The FlagOS Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
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

import torch
import triton
import triton.language as tl

# Why this diverges from the generic implementation (enflame GCU rewrite)
# -----------------------------------------------------------------------
# The generic file keeps a `_topk_sigmoid_tuple_kernel` (per-lane sort-4
# built from tl.reshape + tl.split, a cross-lane `tl.reduce` with the
# 8-tuple `_merge_top8` combine, then rank/byte repacking) dispatched for
# n_experts <= 256 and num_tokens <= 512.  That kernel is unusable on the
# enflame backend:
#   * 2026-11-07+ toolchain: it fails to compile — `memref_ext.slice_start`
#     operand-segment size mismatch (same cluster as
#     compiler_issues/vllm_grouped_topk.md; see compiler_issues/
#     sglang_topk_sigmoid.md).
#   * pre-reinstall toolchain (baseline): it compiled but silently returned
#     wrong ids/weights for k < KSLOT configs.
# `tl.max(..., return_indices=True)` over a 2D tile hits the same
# `memref_ext.slice_pad_start` cluster, and in 1D it measures ~2x SLOWER
# than the plain max + leftmost min-where pair, so this copy routes every
# config through the 1D iterative-max selection whose primitives (plain
# 1D tl.max/tl.where, masked 1D loads) are proven on GCU.
#
# Perf design (2026-11-08 + 2026-11-12 E-1 round, core bench t=1/8/64/512
# @ E=256 k=8 fp32 logits, renorm: 10.24/10.53/22.54/129.00us at import ->
# 4.86/5.26/6.39/23.20us in round 1 -> 4.9/5.3/6.4/18.0us in round 2):
#   * No int64 anywhere in the selection loop: the generic FAST32 path
#     byte-packs k lanes into an int64 [BLOCK] accumulator with a per-round
#     OR and a final `tl.sum`; dropping it for a [KSLOT] int32 accumulator
#     (`tl.where(kslots == j, lane, idx)`) alone is -13% (int64 vector ops
#     are expensive on GCU; the packed int64 single-reduction variant of
#     the same loop measures 6x slower still, re-checked against the vllm
#     sibling whose int64 packed tree only wins at e<=128 / num_warps=4).
#   * The selection loop is latency-bound per round but issue-bound per
#     program: each round is a serial chain max -> tie-where -> min ->
#     invalidate-where over [BLOCK], so ROWS rows per program are processed
#     with their chains interleaved in the round body.  The per-row cost
#     keeps dropping as independent chains fill the issue slots: 2.14us at
#     ROWS=4, 1.87 at 6, 1.81 at 8, 1.63 at 11; ROWS=12 regresses (1.65,
#     register-pressure knee).  The ladder therefore tops out at 11.
#   * `_pick_rows` is wave-aware: with 48 SPs the makespan is
#     ceil(ctas/48) * f(ROWS), so mid sizes want the ROWS whose CTA count
#     fills whole waves (t=512: 128 CTAs of ROWS=4 = 2.67 waves = 23.2us,
#     but 47 CTAs of ROWS=11 = 1 wave = 18.0us), and large sizes want the
#     flattest per-row ROWS=11 (t=4096: 153.0 -> 133.5us).  The dispatch
#     table was verified against a 21-shape sweep (t=48..4096).
# The leftmost tie-break (`tl.min(tl.where(key == pm, cols, BLOCK))`) is
# preserved bit-exactly: ids match the previous enflame kernel bitwise on
# every probed shape, including bf16 duplicate-heavy rows and fp32
# sigmoid-rounding ties.


@triton.jit
def _mono(bits):
    """Signed-int image preserving IEEE-754 order; its own inverse."""
    return tl.where(bits >= 0, bits, bits ^ 0x7FFFFFFF)


@triton.jit
def _load_key(
    x_ptr,
    row,
    cols,
    stride_x,
    n_experts,
    NOMASK: tl.constexpr,
    I32LOAD: tl.constexpr,
):
    # fp32 input: the monotonic map consumes the raw bit pattern, so skip
    # the float convert entirely.  Padded lanes get the -inf image, which
    # is never selected ahead of a finite logit.
    if I32LOAD:
        b32 = x_ptr.to(tl.pointer_type(tl.int32))
        if NOMASK:
            return _mono(tl.load(b32 + row * stride_x + cols))
        else:
            return _mono(
                tl.load(
                    b32 + row * stride_x + cols,
                    mask=cols < n_experts,
                    other=-8388608,
                )
            )
    else:
        if NOMASK:
            s = tl.load(x_ptr + row * stride_x + cols).to(tl.float32)
        else:
            s = tl.load(
                x_ptr + row * stride_x + cols,
                mask=cols < n_experts,
                other=float("-inf"),
            ).to(tl.float32)
        return _mono(s.to(tl.int32, bitcast=True))


@triton.jit
def _topk_sigmoid_kernel(
    x_ptr,
    w_ptr,
    i_ptr,
    n_tokens,
    n_experts,
    scale,
    stride_x,
    stride_w,
    stride_i,
    k: tl.constexpr,
    RENORM: tl.constexpr,
    ROWS: tl.constexpr,
    BLOCK: tl.constexpr,
    KSLOT: tl.constexpr,
    NOMASK: tl.constexpr,
    I32LOAD: tl.constexpr,
):
    pid = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    kslots = tl.arange(0, KSLOT)
    km = kslots < k

    # Rows past the tail are load-clamped onto the last valid row and
    # store-masked, so every program issues the same unmasked row load.
    base = pid * ROWS
    rA = tl.minimum(base, n_tokens - 1).to(tl.int64)
    kA = _load_key(x_ptr, rA, cols, stride_x, n_experts, NOMASK, I32LOAD)
    if ROWS >= 2:
        rB = tl.minimum(base + 1, n_tokens - 1).to(tl.int64)
        kB = _load_key(x_ptr, rB, cols, stride_x, n_experts, NOMASK, I32LOAD)
    if ROWS >= 4:
        rC = tl.minimum(base + 2, n_tokens - 1).to(tl.int64)
        kC = _load_key(x_ptr, rC, cols, stride_x, n_experts, NOMASK, I32LOAD)
        rD = tl.minimum(base + 3, n_tokens - 1).to(tl.int64)
        kD = _load_key(x_ptr, rD, cols, stride_x, n_experts, NOMASK, I32LOAD)
    if ROWS >= 6:
        rE = tl.minimum(base + 4, n_tokens - 1).to(tl.int64)
        kE = _load_key(x_ptr, rE, cols, stride_x, n_experts, NOMASK, I32LOAD)
        rF = tl.minimum(base + 5, n_tokens - 1).to(tl.int64)
        kF = _load_key(x_ptr, rF, cols, stride_x, n_experts, NOMASK, I32LOAD)
    if ROWS >= 8:
        rG = tl.minimum(base + 6, n_tokens - 1).to(tl.int64)
        kG = _load_key(x_ptr, rG, cols, stride_x, n_experts, NOMASK, I32LOAD)
        rH = tl.minimum(base + 7, n_tokens - 1).to(tl.int64)
        kH = _load_key(x_ptr, rH, cols, stride_x, n_experts, NOMASK, I32LOAD)
    if ROWS >= 11:
        rI = tl.minimum(base + 8, n_tokens - 1).to(tl.int64)
        kI = _load_key(x_ptr, rI, cols, stride_x, n_experts, NOMASK, I32LOAD)
        rJ = tl.minimum(base + 9, n_tokens - 1).to(tl.int64)
        kJ = _load_key(x_ptr, rJ, cols, stride_x, n_experts, NOMASK, I32LOAD)
        rK = tl.minimum(base + 10, n_tokens - 1).to(tl.int64)
        kK = _load_key(x_ptr, rK, cols, stride_x, n_experts, NOMASK, I32LOAD)

    valsA = tl.zeros([KSLOT], dtype=tl.float32)
    idxA = tl.zeros([KSLOT], dtype=tl.int32)
    if ROWS >= 2:
        valsB = tl.zeros([KSLOT], dtype=tl.float32)
        idxB = tl.zeros([KSLOT], dtype=tl.int32)
    if ROWS >= 4:
        valsC = tl.zeros([KSLOT], dtype=tl.float32)
        idxC = tl.zeros([KSLOT], dtype=tl.int32)
        valsD = tl.zeros([KSLOT], dtype=tl.float32)
        idxD = tl.zeros([KSLOT], dtype=tl.int32)
    if ROWS >= 6:
        valsE = tl.zeros([KSLOT], dtype=tl.float32)
        idxE = tl.zeros([KSLOT], dtype=tl.int32)
        valsF = tl.zeros([KSLOT], dtype=tl.float32)
        idxF = tl.zeros([KSLOT], dtype=tl.int32)
    if ROWS >= 8:
        valsG = tl.zeros([KSLOT], dtype=tl.float32)
        idxG = tl.zeros([KSLOT], dtype=tl.int32)
        valsH = tl.zeros([KSLOT], dtype=tl.float32)
        idxH = tl.zeros([KSLOT], dtype=tl.int32)
    if ROWS >= 11:
        valsI = tl.zeros([KSLOT], dtype=tl.float32)
        idxI = tl.zeros([KSLOT], dtype=tl.int32)
        valsJ = tl.zeros([KSLOT], dtype=tl.float32)
        idxJ = tl.zeros([KSLOT], dtype=tl.int32)
        valsK = tl.zeros([KSLOT], dtype=tl.float32)
        idxK = tl.zeros([KSLOT], dtype=tl.int32)

    for j in tl.static_range(0, k):
        # Independent per-row chains interleaved in the round body: the
        # serial max -> where -> min -> where chain is latency-bound, so
        # ROWS rows pipeline on the SP instead of stalling each other.
        pmA = tl.max(kA, axis=0)
        if ROWS >= 2:
            pmB = tl.max(kB, axis=0)
        if ROWS >= 4:
            pmC = tl.max(kC, axis=0)
            pmD = tl.max(kD, axis=0)
        if ROWS >= 6:
            pmE = tl.max(kE, axis=0)
            pmF = tl.max(kF, axis=0)
        if ROWS >= 8:
            pmG = tl.max(kG, axis=0)
            pmH = tl.max(kH, axis=0)
        if ROWS >= 11:
            pmI = tl.max(kI, axis=0)
            pmJ = tl.max(kJ, axis=0)
            pmK = tl.max(kK, axis=0)
        # Tied maxima: take the leftmost lane.  Matching every tied lane
        # instead (as the generic file does) corrupts ids whenever the
        # input dtype has duplicate scores (common for bf16 at 128+
        # experts).
        laneA = tl.min(tl.where(kA == pmA, cols, BLOCK), axis=0)
        if ROWS >= 2:
            laneB = tl.min(tl.where(kB == pmB, cols, BLOCK), axis=0)
        if ROWS >= 4:
            laneC = tl.min(tl.where(kC == pmC, cols, BLOCK), axis=0)
            laneD = tl.min(tl.where(kD == pmD, cols, BLOCK), axis=0)
        if ROWS >= 6:
            laneE = tl.min(tl.where(kE == pmE, cols, BLOCK), axis=0)
            laneF = tl.min(tl.where(kF == pmF, cols, BLOCK), axis=0)
        if ROWS >= 8:
            laneG = tl.min(tl.where(kG == pmG, cols, BLOCK), axis=0)
            laneH = tl.min(tl.where(kH == pmH, cols, BLOCK), axis=0)
        if ROWS >= 11:
            laneI = tl.min(tl.where(kI == pmI, cols, BLOCK), axis=0)
            laneJ = tl.min(tl.where(kJ == pmJ, cols, BLOCK), axis=0)
            laneK = tl.min(tl.where(kK == pmK, cols, BLOCK), axis=0)
        valsA = tl.where(
            kslots == j, _mono(pmA).to(tl.float32, bitcast=True), valsA
        )
        idxA = tl.where(kslots == j, laneA.to(tl.int32), idxA)
        if ROWS >= 2:
            valsB = tl.where(
                kslots == j, _mono(pmB).to(tl.float32, bitcast=True), valsB
            )
            idxB = tl.where(kslots == j, laneB.to(tl.int32), idxB)
        if ROWS >= 4:
            valsC = tl.where(
                kslots == j, _mono(pmC).to(tl.float32, bitcast=True), valsC
            )
            idxC = tl.where(kslots == j, laneC.to(tl.int32), idxC)
            valsD = tl.where(
                kslots == j, _mono(pmD).to(tl.float32, bitcast=True), valsD
            )
            idxD = tl.where(kslots == j, laneD.to(tl.int32), idxD)
        if ROWS >= 6:
            valsE = tl.where(
                kslots == j, _mono(pmE).to(tl.float32, bitcast=True), valsE
            )
            idxE = tl.where(kslots == j, laneE.to(tl.int32), idxE)
            valsF = tl.where(
                kslots == j, _mono(pmF).to(tl.float32, bitcast=True), valsF
            )
            idxF = tl.where(kslots == j, laneF.to(tl.int32), idxF)
        if ROWS >= 8:
            valsG = tl.where(
                kslots == j, _mono(pmG).to(tl.float32, bitcast=True), valsG
            )
            idxG = tl.where(kslots == j, laneG.to(tl.int32), idxG)
            valsH = tl.where(
                kslots == j, _mono(pmH).to(tl.float32, bitcast=True), valsH
            )
            idxH = tl.where(kslots == j, laneH.to(tl.int32), idxH)
        if ROWS >= 11:
            valsI = tl.where(
                kslots == j, _mono(pmI).to(tl.float32, bitcast=True), valsI
            )
            idxI = tl.where(kslots == j, laneI.to(tl.int32), idxI)
            valsJ = tl.where(
                kslots == j, _mono(pmJ).to(tl.float32, bitcast=True), valsJ
            )
            idxJ = tl.where(kslots == j, laneJ.to(tl.int32), idxJ)
            valsK = tl.where(
                kslots == j, _mono(pmK).to(tl.float32, bitcast=True), valsK
            )
            idxK = tl.where(kslots == j, laneK.to(tl.int32), idxK)
        kA = tl.where(cols == laneA, -2147483648, kA)
        if ROWS >= 2:
            kB = tl.where(cols == laneB, -2147483648, kB)
        if ROWS >= 4:
            kC = tl.where(cols == laneC, -2147483648, kC)
            kD = tl.where(cols == laneD, -2147483648, kD)
        if ROWS >= 6:
            kE = tl.where(cols == laneE, -2147483648, kE)
            kF = tl.where(cols == laneF, -2147483648, kF)
        if ROWS >= 8:
            kG = tl.where(cols == laneG, -2147483648, kG)
            kH = tl.where(cols == laneH, -2147483648, kH)
        if ROWS >= 11:
            kI = tl.where(cols == laneI, -2147483648, kI)
            kJ = tl.where(cols == laneJ, -2147483648, kJ)
            kK = tl.where(cols == laneK, -2147483648, kK)

    wA = tl.sigmoid(valsA)
    if RENORM:
        tA = tl.sum(tl.where(km, wA, 0.0), axis=0) + 1e-20
        wA = tl.where(km, wA * (scale / tA), 0.0)
    if KSLOT == k:
        tl.store(w_ptr + rA * stride_w + kslots, wA)
        tl.store(i_ptr + rA * stride_i + kslots, idxA)
    else:
        tl.store(w_ptr + rA * stride_w + kslots, wA, mask=km)
        tl.store(i_ptr + rA * stride_i + kslots, idxA, mask=km)

    if ROWS >= 2:
        wB = tl.sigmoid(valsB)
        if RENORM:
            tB = tl.sum(tl.where(km, wB, 0.0), axis=0) + 1e-20
            wB = tl.where(km, wB * (scale / tB), 0.0)
        mB = km & (base + 1 < n_tokens)
        tl.store(w_ptr + rB * stride_w + kslots, wB, mask=mB)
        tl.store(i_ptr + rB * stride_i + kslots, idxB, mask=mB)

    if ROWS >= 4:
        wC = tl.sigmoid(valsC)
        if RENORM:
            tC = tl.sum(tl.where(km, wC, 0.0), axis=0) + 1e-20
            wC = tl.where(km, wC * (scale / tC), 0.0)
        mC = km & (base + 2 < n_tokens)
        tl.store(w_ptr + rC * stride_w + kslots, wC, mask=mC)
        tl.store(i_ptr + rC * stride_i + kslots, idxC, mask=mC)

        wD = tl.sigmoid(valsD)
        if RENORM:
            tD = tl.sum(tl.where(km, wD, 0.0), axis=0) + 1e-20
            wD = tl.where(km, wD * (scale / tD), 0.0)
        mD = km & (base + 3 < n_tokens)
        tl.store(w_ptr + rD * stride_w + kslots, wD, mask=mD)
        tl.store(i_ptr + rD * stride_i + kslots, idxD, mask=mD)

    if ROWS >= 6:
        wE = tl.sigmoid(valsE)
        if RENORM:
            tE = tl.sum(tl.where(km, wE, 0.0), axis=0) + 1e-20
            wE = tl.where(km, wE * (scale / tE), 0.0)
        mE = km & (base + 4 < n_tokens)
        tl.store(w_ptr + rE * stride_w + kslots, wE, mask=mE)
        tl.store(i_ptr + rE * stride_i + kslots, idxE, mask=mE)

        wF = tl.sigmoid(valsF)
        if RENORM:
            tF = tl.sum(tl.where(km, wF, 0.0), axis=0) + 1e-20
            wF = tl.where(km, wF * (scale / tF), 0.0)
        mF = km & (base + 5 < n_tokens)
        tl.store(w_ptr + rF * stride_w + kslots, wF, mask=mF)
        tl.store(i_ptr + rF * stride_i + kslots, idxF, mask=mF)

    if ROWS >= 8:
        wG = tl.sigmoid(valsG)
        if RENORM:
            tG = tl.sum(tl.where(km, wG, 0.0), axis=0) + 1e-20
            wG = tl.where(km, wG * (scale / tG), 0.0)
        mG = km & (base + 6 < n_tokens)
        tl.store(w_ptr + rG * stride_w + kslots, wG, mask=mG)
        tl.store(i_ptr + rG * stride_i + kslots, idxG, mask=mG)

        wH = tl.sigmoid(valsH)
        if RENORM:
            tH = tl.sum(tl.where(km, wH, 0.0), axis=0) + 1e-20
            wH = tl.where(km, wH * (scale / tH), 0.0)
        mH = km & (base + 7 < n_tokens)
        tl.store(w_ptr + rH * stride_w + kslots, wH, mask=mH)
        tl.store(i_ptr + rH * stride_i + kslots, idxH, mask=mH)

    if ROWS >= 11:
        wI = tl.sigmoid(valsI)
        if RENORM:
            tI = tl.sum(tl.where(km, wI, 0.0), axis=0) + 1e-20
            wI = tl.where(km, wI * (scale / tI), 0.0)
        mI = km & (base + 8 < n_tokens)
        tl.store(w_ptr + rI * stride_w + kslots, wI, mask=mI)
        tl.store(i_ptr + rI * stride_i + kslots, idxI, mask=mI)

        wJ = tl.sigmoid(valsJ)
        if RENORM:
            tJ = tl.sum(tl.where(km, wJ, 0.0), axis=0) + 1e-20
            wJ = tl.where(km, wJ * (scale / tJ), 0.0)
        mJ = km & (base + 9 < n_tokens)
        tl.store(w_ptr + rJ * stride_w + kslots, wJ, mask=mJ)
        tl.store(i_ptr + rJ * stride_i + kslots, idxJ, mask=mJ)

        wK = tl.sigmoid(valsK)
        if RENORM:
            tK = tl.sum(tl.where(km, wK, 0.0), axis=0) + 1e-20
            wK = tl.where(km, wK * (scale / tK), 0.0)
        mK = km & (base + 10 < n_tokens)
        tl.store(w_ptr + rK * stride_w + kslots, wK, mask=mK)
        tl.store(i_ptr + rK * stride_i + kslots, idxK, mask=mK)


# One 48-CTA-wave worth of per-ROWS program time, in microseconds, measured
# on GCU (e=256 k=8 fp32, do_bench median, 2026-11-12; see the file header).
# The selection loop is issue-bound per program, so the per-row cost falls
# from 3.18us (ROWS=2) to 1.63us (ROWS=11); ROWS=12 regresses and is not on
# the ladder.  Re-measure this table when the toolchain moves.
_ROW_COST_US = (
    (1, 6.29),
    (2, 6.36),
    (4, 8.54),
    (6, 11.34),
    (8, 14.61),
    (11, 17.90),
)
_SP_COUNT = 48
# Interleaved-chain budget: rows_per_program * BLOCK is capped at the
# measured-no-spill point (11 * 256 lanes) so wide expert tiles dispatch to
# smaller ROWS instead of spilling registers.
_CHAIN_BUDGET = 11 * 256


def _pick_rows(num_tokens, block):
    # Wave-aware dispatch: the kernel is small enough that CTA waves
    # quantize the makespan, so pick the ladder rung minimizing
    # ceil(ctas / 48) * f(ROWS).  Mid sizes want a rung whose CTA count
    # fills whole waves (t=512: ROWS=4 -> 128 CTAs -> 3 waves -> 23.2us,
    # ROWS=11 -> 47 CTAs -> 1 wave -> 18.0us); large sizes converge on the
    # flattest per-row rung (ROWS=11).  Small sizes keep the latency-optimal
    # one/two-row programs of the previous dispatch.
    best_rows = 1
    best_cost = None
    for rows, cost_us in _ROW_COST_US:
        if rows * block > _CHAIN_BUDGET:
            break
        ctas = -(-num_tokens // rows)
        waves = -(-ctas // _SP_COUNT)
        cost = waves * cost_us
        if best_cost is None or cost < best_cost - 1e-9:
            best_rows, best_cost = rows, cost
    return best_rows


def topk_sigmoid(
    topk_weights, topk_ids, gating_output, renormalize, routed_scaling_factor
):
    num_tokens, n_experts = gating_output.shape
    k = topk_weights.shape[1]
    if num_tokens == 0 or k == 0:
        return topk_weights, topk_ids

    x = (
        gating_output
        if gating_output.is_contiguous()
        else gating_output.contiguous()
    )

    scale = float(routed_scaling_factor)
    stride_x = x.stride(0)
    stride_w = topk_weights.stride(0)
    stride_i = topk_ids.stride(0)
    renorm = bool(renormalize)

    i32load = x.dtype == torch.float32

    BLOCK = max(16, triton.next_power_of_2(n_experts))
    KSLOT = triton.next_power_of_2(k)
    NOMASK = n_experts == BLOCK
    rows = _pick_rows(num_tokens, BLOCK)

    grid = (triton.cdiv(num_tokens, rows),)
    _topk_sigmoid_kernel[grid](
        x,
        topk_weights,
        topk_ids,
        num_tokens,
        n_experts,
        scale,
        stride_x,
        stride_w,
        stride_i,
        k=k,
        RENORM=renorm,
        ROWS=rows,
        BLOCK=BLOCK,
        KSLOT=KSLOT,
        NOMASK=NOMASK,
        I32LOAD=i32load,
        num_warps=1,
        num_stages=1,
    )
    return topk_weights, topk_ids


__all__ = ["topk_sigmoid"]
