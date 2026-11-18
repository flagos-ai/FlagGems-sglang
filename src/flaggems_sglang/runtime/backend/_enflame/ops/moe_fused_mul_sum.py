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


"""moe_fused_mul_sum: out[t, h] = sum_k inputs[t, k, h] * w[t, k], fp32 accumulation.

w = topk_weights * routed_scaling_factor, zeroed for experts the routing masks
out.  There are two masking modes and they are mutually exclusive in the
reference: an `expert_map` lookup (`expert_map[topk_ids] >= 0`) takes priority,
otherwise `is_ep` masks on `topk_ids >= 0`.  Both are folded into the weight
before the accumulation, so the inner loop is a plain weighted sum.

GCU (enflame) rewrite of the generic kernel.  The generic version walks a
flattened (token, hidden-tile) index space with up to 4096 tiny programs, a
runtime `top_k` loop whose per-k weight/id/map scalar loads sit on the critical
path, and an always-executed masked load.  On GCU400 that costs: (a) the
~50 ns/CTA dispatch floor times thousands of programs, (b) unpipelined K
loads (the dynamic-range loop does not software-pipeline, measured 0.22 TB/s
in the moe_sum family), and (c) predicate tax on every load.  This copy keeps
the exact per-k arithmetic and k-ascending accumulation order (bitwise equal
to the generic kernel on every shape checked) but reorganizes the launch:

* constexpr TOPK with `tl.static_range` so the K row loads issue together
  (playbook "constexpr TOPK 保序展开"; 2.2~4.5x on moe_sum), and the K weight
  scalars become independent back-to-back loads off the critical path.
* constexpr MODE so the id/map loads exist only in the arm that needs them
  (mode 0 pays no masking loads at all).
* EXACT arm (BLOCK_H divides hidden) drops the hidden predicate; CONTIG is
  guaranteed by the wrapper's `.contiguous()` calls.
* one program per SP (48) striding over the flattened tile space, or a flat
  launch when the whole tile count fits in ~two SP waves; the grid can never
  approach the GCU launch limits by construction.
* mode 0 passes the (never dereferenced) dummy pointer slots as `inp`, so the
  hot path allocates nothing and runs no memset.

Second pass (2026-11-12, L600 probe ladder with do_bench): the per-k weight
scalar loads sit on the FMA critical path and cost ~19% at T=4096 (full
171us vs read+accumulate floor 139us). Two changes attack exactly that:

* SCALE_ONE constexpr: `wgt * scale` is skipped when scale == 1.0 (the fp
  multiply by 1.0 is an exact identity, so results stay bitwise equal); the
  bench shapes all pass routed_scaling_factor=None -> scale 1.0.
* PAIR arm (mode 0, EXACT, TOPK <= 16, tile count > 32): two adjacent
  (token, hidden-block) items per iteration with the A/B scalar and vector
  loads interleaved inside one K loop, so every k segment issues 2 scalar +
  2 vector loads back-to-back (moe_sum static_pair deep-issue precedent).
  The odd tail clamps `wB = min(wA+1, n_work-1)`; the duplicate recompute is
  an idempotent overwrite, no mask and no OOB.  num_warps=2 is part of the
  recipe: the pair issue pattern and warp count are coupled (pair@nw4 is
  +21%, mirror of the hc_gate_mix finding).  Per-token k-ascending order and
  per-element arithmetic are unchanged, so the pair arm is bitwise equal to
  the single-item arm as well.

Measured on [T, 8, 4096] bf16 (do_bench kernel mode, median):
T=1/8/64/512/4096  10.3/14.4/63.5/476/2793 us -> 4.1/4.7/8.9/26.7/171 us
(first pass) -> 4.2/4.3/7.3/24.6/148.6 us (second pass).
"""

import torch
import triton
import triton.language as tl

# GCU400 dispatch floor is ~50 ns/CTA, so keep the launch at one program per
# SP and let each program stride over the flattened (token, hidden-block)
# tile space (moe_sum family precedent: 48 -> 8x faster than 4096 programs).
_GRID_CAP = 48
# Below ~two SP waves a flat launch (one program per tile) wins: no stride
# loop, full parallelism, and the dispatch tail is still tiny.
_FLAT_MAX = 96
# Widest hidden block: one [BLOCK_H] load per expert row.  Wider blocks
# amortize the per-tile fixed costs (weight scalars, decode, store) --
# measured at H=4096: BLOCK 1024/2048/4096 -> 636/381/171 us @ T=4096.
_MAX_BLOCK = 4096
# K-unroll depth for top_k > 16 (chunked static_range with a dynamic tail).
# Bounds the in-flight bytes: 16 * 2048 * (itemsize + 4) <= 256 KB, well
# under the 448 KB double-program DSM line (host clamps BLOCK to 2048 when
# top_k > 16).
_CHUNK = 16
# Adjacent-item pairing only pays once halving the program count still
# covers the SPs: measured T=8 (8 tiles -> 4 programs) is +11%, T=64
# (64 tiles -> 32 programs) is -17.6%.
_PAIR_MIN = 32


@triton.jit
def _moe_fused_mul_sum_kernel(
    in_ptr,
    w_ptr,
    ids_ptr,
    map_ptr,
    out_ptr,
    scale,
    num_tokens,
    TOPK: tl.constexpr,
    HIDDEN: tl.constexpr,
    MODE: tl.constexpr,
    BLOCK_H: tl.constexpr,
    U: tl.constexpr,
    SCALE_ONE: tl.constexpr,
    PAIR: tl.constexpr,
):
    # (token, hidden block) tile space flattened to one index; NUM_H_BLOCKS
    # and EXACT fold at compile time, and the k-ascending accumulation keeps
    # the fp32 result bitwise equal to the generic kernel.
    NT: tl.constexpr = (HIDDEN + BLOCK_H - 1) // BLOCK_H
    EXACT: tl.constexpr = HIDDEN % BLOCK_H == 0
    n_work = num_tokens * NT
    if PAIR:
        # Host gates this arm to MODE == 0, EXACT and TOPK <= 16: two
        # adjacent items per iteration, A/B loads interleaved in one K loop
        # (deep issue).  Odd tail clamps to a recompute (idempotent store).
        n_pair = (n_work + 1) // 2
        for pw in range(tl.program_id(0), n_pair, tl.num_programs(0)):
            wA = pw * 2
            wB = tl.minimum(wA + 1, n_work - 1)
            tA = (wA // NT).to(tl.int64)
            offsA = (wA % NT) * BLOCK_H + tl.arange(0, BLOCK_H)
            offsA = tl.max_contiguous(tl.multiple_of(offsA, BLOCK_H), BLOCK_H)
            baseA = in_ptr + tA * (TOPK * HIDDEN) + offsA
            wbaseA = w_ptr + tA * TOPK
            tB = (wB // NT).to(tl.int64)
            offsB = (wB % NT) * BLOCK_H + tl.arange(0, BLOCK_H)
            offsB = tl.max_contiguous(tl.multiple_of(offsB, BLOCK_H), BLOCK_H)
            baseB = in_ptr + tB * (TOPK * HIDDEN) + offsB
            wbaseB = w_ptr + tB * TOPK
            accA = tl.zeros((BLOCK_H,), dtype=tl.float32)
            accB = tl.zeros((BLOCK_H,), dtype=tl.float32)
            for k in tl.static_range(TOPK):
                wgtA = tl.load(wbaseA + k).to(tl.float32)
                wgtB = tl.load(wbaseB + k).to(tl.float32)
                if not SCALE_ONE:
                    wgtA = wgtA * scale
                    wgtB = wgtB * scale
                xA = tl.load(baseA + k * HIDDEN).to(tl.float32)
                xB = tl.load(baseB + k * HIDDEN).to(tl.float32)
                accA += xA * wgtA
                accB += xB * wgtB
            tl.store(
                out_ptr + tA * HIDDEN + offsA,
                accA.to(out_ptr.dtype.element_ty),
            )
            tl.store(
                out_ptr + tB * HIDDEN + offsB,
                accB.to(out_ptr.dtype.element_ty),
            )
    else:
        for w in range(tl.program_id(0), n_work, tl.num_programs(0)):
            t = (w // NT).to(tl.int64)
            hb = w % NT
            offs_h = hb * BLOCK_H + tl.arange(0, BLOCK_H)
            offs_h = tl.max_contiguous(
                tl.multiple_of(offs_h, BLOCK_H), BLOCK_H
            )
            base = in_ptr + t * (TOPK * HIDDEN) + offs_h
            wbase = w_ptr + t * TOPK
            idbase = ids_ptr + t * TOPK
            acc = tl.zeros((BLOCK_H,), dtype=tl.float32)
            if EXACT:
                if TOPK <= U:
                    for k in tl.static_range(TOPK):
                        wgt = tl.load(wbase + k).to(tl.float32)
                        if not SCALE_ONE:
                            wgt = wgt * scale
                        if MODE == 1:
                            e = tl.load(idbase + k)
                            bad = tl.load(map_ptr + tl.maximum(e, 0)) < 0
                            wgt = tl.where(bad, 0.0, wgt)
                        if MODE == 2:
                            e = tl.load(idbase + k)
                            wgt = tl.where(e < 0, 0.0, wgt)
                        acc += tl.load(base + k * HIDDEN).to(tl.float32) * wgt
                else:
                    for c in range(0, TOPK // U):
                        kb = c * U
                        for u in tl.static_range(U):
                            wgt = tl.load(wbase + kb + u).to(tl.float32)
                            if not SCALE_ONE:
                                wgt = wgt * scale
                            if MODE == 1:
                                e = tl.load(idbase + kb + u)
                                bad = tl.load(map_ptr + tl.maximum(e, 0)) < 0
                                wgt = tl.where(bad, 0.0, wgt)
                            if MODE == 2:
                                e = tl.load(idbase + kb + u)
                                wgt = tl.where(e < 0, 0.0, wgt)
                            acc += (
                                tl.load(base + (kb + u) * HIDDEN).to(
                                    tl.float32
                                )
                                * wgt
                            )
                    if TOPK % U != 0:
                        for k in range((TOPK // U) * U, TOPK):
                            wgt = tl.load(wbase + k).to(tl.float32)
                            if not SCALE_ONE:
                                wgt = wgt * scale
                            if MODE == 1:
                                e = tl.load(idbase + k)
                                bad = tl.load(map_ptr + tl.maximum(e, 0)) < 0
                                wgt = tl.where(bad, 0.0, wgt)
                            if MODE == 2:
                                e = tl.load(idbase + k)
                                wgt = tl.where(e < 0, 0.0, wgt)
                            acc += (
                                tl.load(base + k * HIDDEN).to(tl.float32) * wgt
                            )
                tl.store(
                    out_ptr + t * HIDDEN + offs_h,
                    acc.to(out_ptr.dtype.element_ty),
                )
            else:
                mask = offs_h < HIDDEN
                if TOPK <= U:
                    for k in tl.static_range(TOPK):
                        wgt = tl.load(wbase + k).to(tl.float32)
                        if not SCALE_ONE:
                            wgt = wgt * scale
                        if MODE == 1:
                            e = tl.load(idbase + k)
                            bad = tl.load(map_ptr + tl.maximum(e, 0)) < 0
                            wgt = tl.where(bad, 0.0, wgt)
                        if MODE == 2:
                            e = tl.load(idbase + k)
                            wgt = tl.where(e < 0, 0.0, wgt)
                        acc += (
                            tl.load(
                                base + k * HIDDEN, mask=mask, other=0.0
                            ).to(tl.float32)
                            * wgt
                        )
                else:
                    for c in range(0, TOPK // U):
                        kb = c * U
                        for u in tl.static_range(U):
                            wgt = tl.load(wbase + kb + u).to(tl.float32)
                            if not SCALE_ONE:
                                wgt = wgt * scale
                            if MODE == 1:
                                e = tl.load(idbase + kb + u)
                                bad = tl.load(map_ptr + tl.maximum(e, 0)) < 0
                                wgt = tl.where(bad, 0.0, wgt)
                            if MODE == 2:
                                e = tl.load(idbase + kb + u)
                                wgt = tl.where(e < 0, 0.0, wgt)
                            acc += (
                                tl.load(
                                    base + (kb + u) * HIDDEN,
                                    mask=mask,
                                    other=0.0,
                                ).to(tl.float32)
                                * wgt
                            )
                    if TOPK % U != 0:
                        for k in range((TOPK // U) * U, TOPK):
                            wgt = tl.load(wbase + k).to(tl.float32)
                            if not SCALE_ONE:
                                wgt = wgt * scale
                            if MODE == 1:
                                e = tl.load(idbase + k)
                                bad = tl.load(map_ptr + tl.maximum(e, 0)) < 0
                                wgt = tl.where(bad, 0.0, wgt)
                            if MODE == 2:
                                e = tl.load(idbase + k)
                                wgt = tl.where(e < 0, 0.0, wgt)
                            acc += (
                                tl.load(
                                    base + k * HIDDEN, mask=mask, other=0.0
                                ).to(tl.float32)
                                * wgt
                            )
                tl.store(
                    out_ptr + t * HIDDEN + offs_h,
                    acc.to(out_ptr.dtype.element_ty),
                    mask=mask,
                )


def _pick_block(hidden: int, top_k: int) -> int:
    # Largest power-of-two divisor of hidden (up to _MAX_BLOCK): EXACT holds
    # and each expert row is one wide contiguous load.  Non-divisible hidden
    # falls back to next_pow2(hidden) with a masked tail block.
    block = 0
    for candidate in (_MAX_BLOCK, 2048, 1024, 512, 256, 128, 64):
        if hidden % candidate == 0:
            block = candidate
            break
    if block == 0:
        block = min(_MAX_BLOCK, triton.next_power_of_2(max(hidden, 16)))
    if top_k > _CHUNK and block > 2048:
        # Keep the U-deep in-flight window inside the DSM budget for the
        # chunked path (16 * 2048 * (itemsize + 4) <= 256 KB).
        block = 2048
    return block


def moe_fused_mul_sum(
    inputs,
    topk_weights,
    topk_ids=None,
    expert_map=None,
    routed_scaling_factor=None,
    is_ep=False,
):
    inp = inputs.contiguous()
    num_tokens, top_k, hidden = inp.shape
    out = torch.empty((num_tokens, hidden), dtype=inp.dtype, device=inp.device)
    if num_tokens == 0 or hidden == 0:
        return out
    if top_k == 0:
        out.zero_()
        return out

    w = topk_weights.contiguous()
    mask_map = expert_map is not None
    mask_ep = (not mask_map) and is_ep
    mode = 1 if mask_map else (2 if mask_ep else 0)
    if topk_ids is None:
        # MODE == 0 never dereferences ids_ptr/map_ptr, so `inp` doubles as a
        # typed dummy pointer and the hot path allocates nothing.  The other
        # modes always receive real ids in practice; the zeros fallback only
        # mirrors the generic kernel's defensive dummy.
        ids = (
            inp
            if mode == 0
            else torch.zeros(1, dtype=torch.int32, device=inp.device)
        )
    else:
        ids = topk_ids if topk_ids.is_contiguous() else topk_ids.contiguous()
    emap = expert_map if mask_map else ids
    if mask_map and not emap.is_contiguous():
        emap = expert_map.contiguous()

    block_h = _pick_block(hidden, top_k)
    nt = triton.cdiv(hidden, block_h)
    total = num_tokens * nt
    scale = (
        1.0 if routed_scaling_factor is None else float(routed_scaling_factor)
    )
    # Adjacent-item pairing (deep issue) pays only on the unmasked, unchunked
    # hot path and only once the halved program count still covers the SPs;
    # measured T=8 (8 tiles) +11%, T=64 (64 tiles) -17.6%, T=4096 -13%.
    use_pair = (
        mode == 0
        and top_k <= _CHUNK
        and hidden % block_h == 0
        and total > _PAIR_MIN
    )
    if use_pair:
        n_pair = (total + 1) // 2
        nprog = n_pair if n_pair <= _FLAT_MAX else _GRID_CAP
        num_warps = 2
    else:
        nprog = total if total <= _FLAT_MAX else _GRID_CAP
        num_warps = 4 if block_h >= 2048 else 2
    _moe_fused_mul_sum_kernel[(nprog,)](
        inp,
        w,
        ids,
        emap,
        out,
        scale,
        num_tokens,
        TOPK=top_k,
        HIDDEN=hidden,
        MODE=mode,
        BLOCK_H=block_h,
        U=_CHUNK,
        SCALE_ONE=scale == 1.0,
        PAIR=use_pair,
        num_warps=num_warps,
    )
    return out


__all__ = ["moe_fused_mul_sum"]
