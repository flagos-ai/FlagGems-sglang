# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""SGLang MoE sigmoid-gate top-k renorm — GCU perf rewrite.

The generic implementation launches one CTA per token row.  On this device
class every such CTA pays a ~2.7us dispatch+load+store serial chain, so the
4096-row bench shape wave-quantizes to ~380us (slower than the eager torch
reference).  Its K-pass selection also builds every argmax from three
dependent reduction trees, and the "obvious" fusion
``tl.max(..., return_indices=True)`` measured ~2.5x SLOWER than the
explicit three-tree form on this backend (grid=(M,) A/B at M=4096:
605us vs 379us), so this rewrite keeps max + where + min.

Structure (core bench shapes, bf16, do_bench median, GCU):

* persistent grid of min(48, ceil(M/U)) programs striding over rows in a
  pipelined ``tl.range`` loop — amortizes the per-CTA dispatch chain
  (M=4096 floor 233us -> 90us);
* U rows per loop step as independent register chains; the scheduler
  interleaves their selection trees, cutting the serial K-loop cost per
  row.  Round 1 shipped U in {1,2,4}; round 2 measured the issue-saturation
  curve to its flat part (per-row 2.39us at U4 -> 1.55 at U11 -> 1.34 at
  U22; no register knee up to 22 rows x BLOCK 128 = 2816 lanes/vector,
  same per-vector budget as the topk_sigmoid 11x256 no-spill point), so
  the ladder now runs {1,2,4,6,8,11,14,16,18,20,22} and the host picks the
  rung by makespan: ceil(ctas/48) iterations x measured per-iteration cost
  (wave quantization dominates the mid range — M=512: U4 3 iterations
  20.3us vs U11 one wave 17.0us; M=4096: 125.5 -> 103.4us at U22);
* selection writes a per-expert ``rank`` instead of extracting each picked
  probability through a where+max tree; outputs scatter-store straight
  from the rank and one masked sum feeds the denominator — removes K
  extraction trees and K scalar sigmoids per row;
* shared-expert loads/sigmoids issue before the selection loop (latency
  hides under it); non-row scalar scales load once per program.

Round-2 vetoes (measured, GCU bf16 N=128/S=3/K=5, do_bench median):
int16 rank accumulators +54..67%; num_warps=2 ~10x; num_stages 1/3 within
noise; phase-batched slot steps help U<=2 marginally but regress U>=6.

Semantics match the generic kernel: equal scores pick the lowest expert
index, output slots are ordered by descending score, ``routed_w`` carries
sigmoid(logit) rather than the biased score (verified against the device
reference on boundary-tie inputs and on every correctness shape).  One
kernel launch per call; no torch compute on device tensors beyond the
three output allocations.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _slot_step(s, rank, offs_n, N, slot):
    # Argmax of ``s`` with lowest-index tie-break, expressed as
    # max-tree + where + min-tree.  return_indices fusions are slower on
    # this backend (see module docstring).
    best = tl.max(s, axis=0)
    cand = tl.where(s == best, offs_n, N)
    bidx = tl.min(cand, axis=0)
    rank = tl.where(offs_n == bidx, slot, rank)
    s = tl.where(offs_n == bidx, -float("inf"), s)
    return s, rank


@triton.jit
def _sigmoid_gate_topk_renorm_kernel(
    logits,
    bias,
    routed_w,
    indices,
    shared_w,
    M,
    nprog,
    logits_stride_m,
    logits_stride_g,
    bias_stride,
    route_scale,
    global_scale,
    route_scale_stride_m,
    global_scale_stride_m,
    N: tl.constexpr,
    S: tl.constexpr,
    K: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_S: tl.constexpr,
    ROUTE_SCALE_IS_TENSOR: tl.constexpr,
    ROUTE_SCALE_IS_ROW: tl.constexpr,
    GLOBAL_SCALE_IS_TENSOR: tl.constexpr,
    GLOBAL_SCALE_IS_ROW: tl.constexpr,
    N_EXACT: tl.constexpr,
    UROWS: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_n = tl.arange(0, BLOCK_N)
    mask_n = offs_n < N
    if N_EXACT:
        bias_f = tl.load(bias + offs_n * bias_stride).to(tl.float32)
    else:
        bias_f = tl.load(
            bias + offs_n * bias_stride, mask=mask_n, other=-float("inf")
        ).to(tl.float32)
    offs_s = tl.arange(0, BLOCK_S)
    mask_s = offs_s < S

    if ROUTE_SCALE_IS_TENSOR:
        if not ROUTE_SCALE_IS_ROW:
            route_scale_h = tl.load(route_scale).to(tl.float32)
    if GLOBAL_SCALE_IS_TENSOR:
        if not GLOBAL_SCALE_IS_ROW:
            global_scale_h = tl.load(global_scale).to(tl.float32)

    for base in tl.range(pid * UROWS, M, nprog * UROWS, num_stages=2):

        r0 = base
        v0 = r0 < M
        b0 = r0 * logits_stride_m
        if UROWS >= 2:
            r1 = base + 1
            v1 = r1 < M
            b1 = r1 * logits_stride_m
        if UROWS >= 4:
            r2 = base + 2
            v2 = r2 < M
            b2 = r2 * logits_stride_m
            r3 = base + 3
            v3 = r3 < M
            b3 = r3 * logits_stride_m
        if UROWS >= 6:
            r4 = base + 4
            v4 = r4 < M
            b4 = r4 * logits_stride_m
            r5 = base + 5
            v5 = r5 < M
            b5 = r5 * logits_stride_m
        if UROWS >= 8:
            r6 = base + 6
            v6 = r6 < M
            b6 = r6 * logits_stride_m
            r7 = base + 7
            v7 = r7 < M
            b7 = r7 * logits_stride_m
        if UROWS >= 11:
            r8 = base + 8
            v8 = r8 < M
            b8 = r8 * logits_stride_m
            r9 = base + 9
            v9 = r9 < M
            b9 = r9 * logits_stride_m
            r10 = base + 10
            v10 = r10 < M
            b10 = r10 * logits_stride_m
        if UROWS >= 14:
            r11 = base + 11
            v11 = r11 < M
            b11 = r11 * logits_stride_m
            r12 = base + 12
            v12 = r12 < M
            b12 = r12 * logits_stride_m
            r13 = base + 13
            v13 = r13 < M
            b13 = r13 * logits_stride_m
        if UROWS >= 16:
            r14 = base + 14
            v14 = r14 < M
            b14 = r14 * logits_stride_m
            r15 = base + 15
            v15 = r15 < M
            b15 = r15 * logits_stride_m
        if UROWS >= 18:
            r16 = base + 16
            v16 = r16 < M
            b16 = r16 * logits_stride_m
            r17 = base + 17
            v17 = r17 < M
            b17 = r17 * logits_stride_m
        if UROWS >= 20:
            r18 = base + 18
            v18 = r18 < M
            b18 = r18 * logits_stride_m
            r19 = base + 19
            v19 = r19 < M
            b19 = r19 * logits_stride_m
        if UROWS >= 22:
            r20 = base + 20
            v20 = r20 < M
            b20 = r20 * logits_stride_m
            r21 = base + 21
            v21 = r21 < M
            b21 = r21 * logits_stride_m
        x0 = tl.load(
            logits + b0 + offs_n * logits_stride_g,
            mask=v0 & mask_n,
            other=-float("inf"),
        ).to(tl.float32)
        if UROWS >= 2:
            x1 = tl.load(
                logits + b1 + offs_n * logits_stride_g,
                mask=v1 & mask_n,
                other=-float("inf"),
            ).to(tl.float32)
        if UROWS >= 4:
            x2 = tl.load(
                logits + b2 + offs_n * logits_stride_g,
                mask=v2 & mask_n,
                other=-float("inf"),
            ).to(tl.float32)
            x3 = tl.load(
                logits + b3 + offs_n * logits_stride_g,
                mask=v3 & mask_n,
                other=-float("inf"),
            ).to(tl.float32)
        if UROWS >= 6:
            x4 = tl.load(
                logits + b4 + offs_n * logits_stride_g,
                mask=v4 & mask_n,
                other=-float("inf"),
            ).to(tl.float32)
            x5 = tl.load(
                logits + b5 + offs_n * logits_stride_g,
                mask=v5 & mask_n,
                other=-float("inf"),
            ).to(tl.float32)
        if UROWS >= 8:
            x6 = tl.load(
                logits + b6 + offs_n * logits_stride_g,
                mask=v6 & mask_n,
                other=-float("inf"),
            ).to(tl.float32)
            x7 = tl.load(
                logits + b7 + offs_n * logits_stride_g,
                mask=v7 & mask_n,
                other=-float("inf"),
            ).to(tl.float32)
        if UROWS >= 11:
            x8 = tl.load(
                logits + b8 + offs_n * logits_stride_g,
                mask=v8 & mask_n,
                other=-float("inf"),
            ).to(tl.float32)
            x9 = tl.load(
                logits + b9 + offs_n * logits_stride_g,
                mask=v9 & mask_n,
                other=-float("inf"),
            ).to(tl.float32)
            x10 = tl.load(
                logits + b10 + offs_n * logits_stride_g,
                mask=v10 & mask_n,
                other=-float("inf"),
            ).to(tl.float32)
        if UROWS >= 14:
            x11 = tl.load(
                logits + b11 + offs_n * logits_stride_g,
                mask=v11 & mask_n,
                other=-float("inf"),
            ).to(tl.float32)
            x12 = tl.load(
                logits + b12 + offs_n * logits_stride_g,
                mask=v12 & mask_n,
                other=-float("inf"),
            ).to(tl.float32)
            x13 = tl.load(
                logits + b13 + offs_n * logits_stride_g,
                mask=v13 & mask_n,
                other=-float("inf"),
            ).to(tl.float32)
        if UROWS >= 16:
            x14 = tl.load(
                logits + b14 + offs_n * logits_stride_g,
                mask=v14 & mask_n,
                other=-float("inf"),
            ).to(tl.float32)
            x15 = tl.load(
                logits + b15 + offs_n * logits_stride_g,
                mask=v15 & mask_n,
                other=-float("inf"),
            ).to(tl.float32)
        if UROWS >= 18:
            x16 = tl.load(
                logits + b16 + offs_n * logits_stride_g,
                mask=v16 & mask_n,
                other=-float("inf"),
            ).to(tl.float32)
            x17 = tl.load(
                logits + b17 + offs_n * logits_stride_g,
                mask=v17 & mask_n,
                other=-float("inf"),
            ).to(tl.float32)
        if UROWS >= 20:
            x18 = tl.load(
                logits + b18 + offs_n * logits_stride_g,
                mask=v18 & mask_n,
                other=-float("inf"),
            ).to(tl.float32)
            x19 = tl.load(
                logits + b19 + offs_n * logits_stride_g,
                mask=v19 & mask_n,
                other=-float("inf"),
            ).to(tl.float32)
        if UROWS >= 22:
            x20 = tl.load(
                logits + b20 + offs_n * logits_stride_g,
                mask=v20 & mask_n,
                other=-float("inf"),
            ).to(tl.float32)
            x21 = tl.load(
                logits + b21 + offs_n * logits_stride_g,
                mask=v21 & mask_n,
                other=-float("inf"),
            ).to(tl.float32)
        if S > 0:
            sh0 = tl.load(
                logits + b0 + (N + offs_s) * logits_stride_g,
                mask=v0 & mask_s,
                other=-float("inf"),
            ).to(tl.float32)
            if UROWS >= 2:
                sh1 = tl.load(
                    logits + b1 + (N + offs_s) * logits_stride_g,
                    mask=v1 & mask_s,
                    other=-float("inf"),
                ).to(tl.float32)
            if UROWS >= 4:
                sh2 = tl.load(
                    logits + b2 + (N + offs_s) * logits_stride_g,
                    mask=v2 & mask_s,
                    other=-float("inf"),
                ).to(tl.float32)
                sh3 = tl.load(
                    logits + b3 + (N + offs_s) * logits_stride_g,
                    mask=v3 & mask_s,
                    other=-float("inf"),
                ).to(tl.float32)
            if UROWS >= 6:
                sh4 = tl.load(
                    logits + b4 + (N + offs_s) * logits_stride_g,
                    mask=v4 & mask_s,
                    other=-float("inf"),
                ).to(tl.float32)
                sh5 = tl.load(
                    logits + b5 + (N + offs_s) * logits_stride_g,
                    mask=v5 & mask_s,
                    other=-float("inf"),
                ).to(tl.float32)
            if UROWS >= 8:
                sh6 = tl.load(
                    logits + b6 + (N + offs_s) * logits_stride_g,
                    mask=v6 & mask_s,
                    other=-float("inf"),
                ).to(tl.float32)
                sh7 = tl.load(
                    logits + b7 + (N + offs_s) * logits_stride_g,
                    mask=v7 & mask_s,
                    other=-float("inf"),
                ).to(tl.float32)
            if UROWS >= 11:
                sh8 = tl.load(
                    logits + b8 + (N + offs_s) * logits_stride_g,
                    mask=v8 & mask_s,
                    other=-float("inf"),
                ).to(tl.float32)
                sh9 = tl.load(
                    logits + b9 + (N + offs_s) * logits_stride_g,
                    mask=v9 & mask_s,
                    other=-float("inf"),
                ).to(tl.float32)
                sh10 = tl.load(
                    logits + b10 + (N + offs_s) * logits_stride_g,
                    mask=v10 & mask_s,
                    other=-float("inf"),
                ).to(tl.float32)
            if UROWS >= 14:
                sh11 = tl.load(
                    logits + b11 + (N + offs_s) * logits_stride_g,
                    mask=v11 & mask_s,
                    other=-float("inf"),
                ).to(tl.float32)
                sh12 = tl.load(
                    logits + b12 + (N + offs_s) * logits_stride_g,
                    mask=v12 & mask_s,
                    other=-float("inf"),
                ).to(tl.float32)
                sh13 = tl.load(
                    logits + b13 + (N + offs_s) * logits_stride_g,
                    mask=v13 & mask_s,
                    other=-float("inf"),
                ).to(tl.float32)
            if UROWS >= 16:
                sh14 = tl.load(
                    logits + b14 + (N + offs_s) * logits_stride_g,
                    mask=v14 & mask_s,
                    other=-float("inf"),
                ).to(tl.float32)
                sh15 = tl.load(
                    logits + b15 + (N + offs_s) * logits_stride_g,
                    mask=v15 & mask_s,
                    other=-float("inf"),
                ).to(tl.float32)
            if UROWS >= 18:
                sh16 = tl.load(
                    logits + b16 + (N + offs_s) * logits_stride_g,
                    mask=v16 & mask_s,
                    other=-float("inf"),
                ).to(tl.float32)
                sh17 = tl.load(
                    logits + b17 + (N + offs_s) * logits_stride_g,
                    mask=v17 & mask_s,
                    other=-float("inf"),
                ).to(tl.float32)
            if UROWS >= 20:
                sh18 = tl.load(
                    logits + b18 + (N + offs_s) * logits_stride_g,
                    mask=v18 & mask_s,
                    other=-float("inf"),
                ).to(tl.float32)
                sh19 = tl.load(
                    logits + b19 + (N + offs_s) * logits_stride_g,
                    mask=v19 & mask_s,
                    other=-float("inf"),
                ).to(tl.float32)
            if UROWS >= 22:
                sh20 = tl.load(
                    logits + b20 + (N + offs_s) * logits_stride_g,
                    mask=v20 & mask_s,
                    other=-float("inf"),
                ).to(tl.float32)
                sh21 = tl.load(
                    logits + b21 + (N + offs_s) * logits_stride_g,
                    mask=v21 & mask_s,
                    other=-float("inf"),
                ).to(tl.float32)
        p0 = tl.sigmoid(x0)
        if UROWS >= 2:
            p1 = tl.sigmoid(x1)
        if UROWS >= 4:
            p2 = tl.sigmoid(x2)
            p3 = tl.sigmoid(x3)
        if UROWS >= 6:
            p4 = tl.sigmoid(x4)
            p5 = tl.sigmoid(x5)
        if UROWS >= 8:
            p6 = tl.sigmoid(x6)
            p7 = tl.sigmoid(x7)
        if UROWS >= 11:
            p8 = tl.sigmoid(x8)
            p9 = tl.sigmoid(x9)
            p10 = tl.sigmoid(x10)
        if UROWS >= 14:
            p11 = tl.sigmoid(x11)
            p12 = tl.sigmoid(x12)
            p13 = tl.sigmoid(x13)
        if UROWS >= 16:
            p14 = tl.sigmoid(x14)
            p15 = tl.sigmoid(x15)
        if UROWS >= 18:
            p16 = tl.sigmoid(x16)
            p17 = tl.sigmoid(x17)
        if UROWS >= 20:
            p18 = tl.sigmoid(x18)
            p19 = tl.sigmoid(x19)
        if UROWS >= 22:
            p20 = tl.sigmoid(x20)
            p21 = tl.sigmoid(x21)
        if S > 0:
            q0 = tl.sigmoid(sh0)
            q0 = tl.where(mask_s, q0, 0.0)
            if UROWS >= 2:
                q1 = tl.sigmoid(sh1)
                q1 = tl.where(mask_s, q1, 0.0)
            if UROWS >= 4:
                q2 = tl.sigmoid(sh2)
                q2 = tl.where(mask_s, q2, 0.0)
                q3 = tl.sigmoid(sh3)
                q3 = tl.where(mask_s, q3, 0.0)
            if UROWS >= 6:
                q4 = tl.sigmoid(sh4)
                q4 = tl.where(mask_s, q4, 0.0)
                q5 = tl.sigmoid(sh5)
                q5 = tl.where(mask_s, q5, 0.0)
            if UROWS >= 8:
                q6 = tl.sigmoid(sh6)
                q6 = tl.where(mask_s, q6, 0.0)
                q7 = tl.sigmoid(sh7)
                q7 = tl.where(mask_s, q7, 0.0)
            if UROWS >= 11:
                q8 = tl.sigmoid(sh8)
                q8 = tl.where(mask_s, q8, 0.0)
                q9 = tl.sigmoid(sh9)
                q9 = tl.where(mask_s, q9, 0.0)
                q10 = tl.sigmoid(sh10)
                q10 = tl.where(mask_s, q10, 0.0)
            if UROWS >= 14:
                q11 = tl.sigmoid(sh11)
                q11 = tl.where(mask_s, q11, 0.0)
                q12 = tl.sigmoid(sh12)
                q12 = tl.where(mask_s, q12, 0.0)
                q13 = tl.sigmoid(sh13)
                q13 = tl.where(mask_s, q13, 0.0)
            if UROWS >= 16:
                q14 = tl.sigmoid(sh14)
                q14 = tl.where(mask_s, q14, 0.0)
                q15 = tl.sigmoid(sh15)
                q15 = tl.where(mask_s, q15, 0.0)
            if UROWS >= 18:
                q16 = tl.sigmoid(sh16)
                q16 = tl.where(mask_s, q16, 0.0)
                q17 = tl.sigmoid(sh17)
                q17 = tl.where(mask_s, q17, 0.0)
            if UROWS >= 20:
                q18 = tl.sigmoid(sh18)
                q18 = tl.where(mask_s, q18, 0.0)
                q19 = tl.sigmoid(sh19)
                q19 = tl.where(mask_s, q19, 0.0)
            if UROWS >= 22:
                q20 = tl.sigmoid(sh20)
                q20 = tl.where(mask_s, q20, 0.0)
                q21 = tl.sigmoid(sh21)
                q21 = tl.where(mask_s, q21, 0.0)
        s0 = p0 + bias_f
        rank0 = tl.full((BLOCK_N,), K, tl.int32)
        if not N_EXACT:
            s0 = tl.where(mask_n, s0, -float("inf"))
        if UROWS >= 2:
            s1 = p1 + bias_f
            rank1 = tl.full((BLOCK_N,), K, tl.int32)
            if not N_EXACT:
                s1 = tl.where(mask_n, s1, -float("inf"))
        if UROWS >= 4:
            s2 = p2 + bias_f
            rank2 = tl.full((BLOCK_N,), K, tl.int32)
            if not N_EXACT:
                s2 = tl.where(mask_n, s2, -float("inf"))
            s3 = p3 + bias_f
            rank3 = tl.full((BLOCK_N,), K, tl.int32)
            if not N_EXACT:
                s3 = tl.where(mask_n, s3, -float("inf"))
        if UROWS >= 6:
            s4 = p4 + bias_f
            rank4 = tl.full((BLOCK_N,), K, tl.int32)
            if not N_EXACT:
                s4 = tl.where(mask_n, s4, -float("inf"))
            s5 = p5 + bias_f
            rank5 = tl.full((BLOCK_N,), K, tl.int32)
            if not N_EXACT:
                s5 = tl.where(mask_n, s5, -float("inf"))
        if UROWS >= 8:
            s6 = p6 + bias_f
            rank6 = tl.full((BLOCK_N,), K, tl.int32)
            if not N_EXACT:
                s6 = tl.where(mask_n, s6, -float("inf"))
            s7 = p7 + bias_f
            rank7 = tl.full((BLOCK_N,), K, tl.int32)
            if not N_EXACT:
                s7 = tl.where(mask_n, s7, -float("inf"))
        if UROWS >= 11:
            s8 = p8 + bias_f
            rank8 = tl.full((BLOCK_N,), K, tl.int32)
            if not N_EXACT:
                s8 = tl.where(mask_n, s8, -float("inf"))
            s9 = p9 + bias_f
            rank9 = tl.full((BLOCK_N,), K, tl.int32)
            if not N_EXACT:
                s9 = tl.where(mask_n, s9, -float("inf"))
            s10 = p10 + bias_f
            rank10 = tl.full((BLOCK_N,), K, tl.int32)
            if not N_EXACT:
                s10 = tl.where(mask_n, s10, -float("inf"))
        if UROWS >= 14:
            s11 = p11 + bias_f
            rank11 = tl.full((BLOCK_N,), K, tl.int32)
            if not N_EXACT:
                s11 = tl.where(mask_n, s11, -float("inf"))
            s12 = p12 + bias_f
            rank12 = tl.full((BLOCK_N,), K, tl.int32)
            if not N_EXACT:
                s12 = tl.where(mask_n, s12, -float("inf"))
            s13 = p13 + bias_f
            rank13 = tl.full((BLOCK_N,), K, tl.int32)
            if not N_EXACT:
                s13 = tl.where(mask_n, s13, -float("inf"))
        if UROWS >= 16:
            s14 = p14 + bias_f
            rank14 = tl.full((BLOCK_N,), K, tl.int32)
            if not N_EXACT:
                s14 = tl.where(mask_n, s14, -float("inf"))
            s15 = p15 + bias_f
            rank15 = tl.full((BLOCK_N,), K, tl.int32)
            if not N_EXACT:
                s15 = tl.where(mask_n, s15, -float("inf"))
        if UROWS >= 18:
            s16 = p16 + bias_f
            rank16 = tl.full((BLOCK_N,), K, tl.int32)
            if not N_EXACT:
                s16 = tl.where(mask_n, s16, -float("inf"))
            s17 = p17 + bias_f
            rank17 = tl.full((BLOCK_N,), K, tl.int32)
            if not N_EXACT:
                s17 = tl.where(mask_n, s17, -float("inf"))
        if UROWS >= 20:
            s18 = p18 + bias_f
            rank18 = tl.full((BLOCK_N,), K, tl.int32)
            if not N_EXACT:
                s18 = tl.where(mask_n, s18, -float("inf"))
            s19 = p19 + bias_f
            rank19 = tl.full((BLOCK_N,), K, tl.int32)
            if not N_EXACT:
                s19 = tl.where(mask_n, s19, -float("inf"))
        if UROWS >= 22:
            s20 = p20 + bias_f
            rank20 = tl.full((BLOCK_N,), K, tl.int32)
            if not N_EXACT:
                s20 = tl.where(mask_n, s20, -float("inf"))
            s21 = p21 + bias_f
            rank21 = tl.full((BLOCK_N,), K, tl.int32)
            if not N_EXACT:
                s21 = tl.where(mask_n, s21, -float("inf"))
        for slot in tl.static_range(0, K):
            s0, rank0 = _slot_step(s0, rank0, offs_n, N, slot)
            if UROWS >= 2:
                s1, rank1 = _slot_step(s1, rank1, offs_n, N, slot)
            if UROWS >= 4:
                s2, rank2 = _slot_step(s2, rank2, offs_n, N, slot)
                s3, rank3 = _slot_step(s3, rank3, offs_n, N, slot)
            if UROWS >= 6:
                s4, rank4 = _slot_step(s4, rank4, offs_n, N, slot)
                s5, rank5 = _slot_step(s5, rank5, offs_n, N, slot)
            if UROWS >= 8:
                s6, rank6 = _slot_step(s6, rank6, offs_n, N, slot)
                s7, rank7 = _slot_step(s7, rank7, offs_n, N, slot)
            if UROWS >= 11:
                s8, rank8 = _slot_step(s8, rank8, offs_n, N, slot)
                s9, rank9 = _slot_step(s9, rank9, offs_n, N, slot)
                s10, rank10 = _slot_step(s10, rank10, offs_n, N, slot)
            if UROWS >= 14:
                s11, rank11 = _slot_step(s11, rank11, offs_n, N, slot)
                s12, rank12 = _slot_step(s12, rank12, offs_n, N, slot)
                s13, rank13 = _slot_step(s13, rank13, offs_n, N, slot)
            if UROWS >= 16:
                s14, rank14 = _slot_step(s14, rank14, offs_n, N, slot)
                s15, rank15 = _slot_step(s15, rank15, offs_n, N, slot)
            if UROWS >= 18:
                s16, rank16 = _slot_step(s16, rank16, offs_n, N, slot)
                s17, rank17 = _slot_step(s17, rank17, offs_n, N, slot)
            if UROWS >= 20:
                s18, rank18 = _slot_step(s18, rank18, offs_n, N, slot)
                s19, rank19 = _slot_step(s19, rank19, offs_n, N, slot)
            if UROWS >= 22:
                s20, rank20 = _slot_step(s20, rank20, offs_n, N, slot)
                s21, rank21 = _slot_step(s21, rank21, offs_n, N, slot)
        sel0 = rank0 < K
        d0 = tl.sum(tl.where(sel0, p0, 0.0), axis=0)
        if S > 0:
            d0 += tl.sum(q0, axis=0)
        if ROUTE_SCALE_IS_TENSOR:
            if ROUTE_SCALE_IS_ROW:
                rs0 = tl.load(
                    route_scale + r0 * route_scale_stride_m, mask=v0, other=0.0
                ).to(tl.float32)
            else:
                rs0 = route_scale_h
        else:
            rs0 = route_scale
        if GLOBAL_SCALE_IS_TENSOR:
            if GLOBAL_SCALE_IS_ROW:
                gs0 = tl.load(
                    global_scale + r0 * global_scale_stride_m,
                    mask=v0,
                    other=0.0,
                ).to(tl.float32)
            else:
                gs0 = global_scale_h
        else:
            gs0 = global_scale
        sd0 = rs0 * gs0 / d0
        tl.store(routed_w + r0 * K + rank0, p0 * sd0, mask=sel0 & v0)
        tl.store(indices + r0 * K + rank0, offs_n.to(tl.int32), mask=sel0 & v0)
        if S > 0:
            tl.store(shared_w + r0 * S + offs_s, q0 * sd0, mask=mask_s & v0)
        if UROWS >= 2:
            sel1 = rank1 < K
            d1 = tl.sum(tl.where(sel1, p1, 0.0), axis=0)
            if S > 0:
                d1 += tl.sum(q1, axis=0)
            if ROUTE_SCALE_IS_TENSOR:
                if ROUTE_SCALE_IS_ROW:
                    rs1 = tl.load(
                        route_scale + r1 * route_scale_stride_m,
                        mask=v1,
                        other=0.0,
                    ).to(tl.float32)
                else:
                    rs1 = route_scale_h
            else:
                rs1 = route_scale
            if GLOBAL_SCALE_IS_TENSOR:
                if GLOBAL_SCALE_IS_ROW:
                    gs1 = tl.load(
                        global_scale + r1 * global_scale_stride_m,
                        mask=v1,
                        other=0.0,
                    ).to(tl.float32)
                else:
                    gs1 = global_scale_h
            else:
                gs1 = global_scale
            sd1 = rs1 * gs1 / d1
            tl.store(routed_w + r1 * K + rank1, p1 * sd1, mask=sel1 & v1)
            tl.store(
                indices + r1 * K + rank1, offs_n.to(tl.int32), mask=sel1 & v1
            )
            if S > 0:
                tl.store(
                    shared_w + r1 * S + offs_s, q1 * sd1, mask=mask_s & v1
                )
        if UROWS >= 4:
            sel2 = rank2 < K
            d2 = tl.sum(tl.where(sel2, p2, 0.0), axis=0)
            if S > 0:
                d2 += tl.sum(q2, axis=0)
            if ROUTE_SCALE_IS_TENSOR:
                if ROUTE_SCALE_IS_ROW:
                    rs2 = tl.load(
                        route_scale + r2 * route_scale_stride_m,
                        mask=v2,
                        other=0.0,
                    ).to(tl.float32)
                else:
                    rs2 = route_scale_h
            else:
                rs2 = route_scale
            if GLOBAL_SCALE_IS_TENSOR:
                if GLOBAL_SCALE_IS_ROW:
                    gs2 = tl.load(
                        global_scale + r2 * global_scale_stride_m,
                        mask=v2,
                        other=0.0,
                    ).to(tl.float32)
                else:
                    gs2 = global_scale_h
            else:
                gs2 = global_scale
            sd2 = rs2 * gs2 / d2
            tl.store(routed_w + r2 * K + rank2, p2 * sd2, mask=sel2 & v2)
            tl.store(
                indices + r2 * K + rank2, offs_n.to(tl.int32), mask=sel2 & v2
            )
            if S > 0:
                tl.store(
                    shared_w + r2 * S + offs_s, q2 * sd2, mask=mask_s & v2
                )
            sel3 = rank3 < K
            d3 = tl.sum(tl.where(sel3, p3, 0.0), axis=0)
            if S > 0:
                d3 += tl.sum(q3, axis=0)
            if ROUTE_SCALE_IS_TENSOR:
                if ROUTE_SCALE_IS_ROW:
                    rs3 = tl.load(
                        route_scale + r3 * route_scale_stride_m,
                        mask=v3,
                        other=0.0,
                    ).to(tl.float32)
                else:
                    rs3 = route_scale_h
            else:
                rs3 = route_scale
            if GLOBAL_SCALE_IS_TENSOR:
                if GLOBAL_SCALE_IS_ROW:
                    gs3 = tl.load(
                        global_scale + r3 * global_scale_stride_m,
                        mask=v3,
                        other=0.0,
                    ).to(tl.float32)
                else:
                    gs3 = global_scale_h
            else:
                gs3 = global_scale
            sd3 = rs3 * gs3 / d3
            tl.store(routed_w + r3 * K + rank3, p3 * sd3, mask=sel3 & v3)
            tl.store(
                indices + r3 * K + rank3, offs_n.to(tl.int32), mask=sel3 & v3
            )
            if S > 0:
                tl.store(
                    shared_w + r3 * S + offs_s, q3 * sd3, mask=mask_s & v3
                )
        if UROWS >= 6:
            sel4 = rank4 < K
            d4 = tl.sum(tl.where(sel4, p4, 0.0), axis=0)
            if S > 0:
                d4 += tl.sum(q4, axis=0)
            if ROUTE_SCALE_IS_TENSOR:
                if ROUTE_SCALE_IS_ROW:
                    rs4 = tl.load(
                        route_scale + r4 * route_scale_stride_m,
                        mask=v4,
                        other=0.0,
                    ).to(tl.float32)
                else:
                    rs4 = route_scale_h
            else:
                rs4 = route_scale
            if GLOBAL_SCALE_IS_TENSOR:
                if GLOBAL_SCALE_IS_ROW:
                    gs4 = tl.load(
                        global_scale + r4 * global_scale_stride_m,
                        mask=v4,
                        other=0.0,
                    ).to(tl.float32)
                else:
                    gs4 = global_scale_h
            else:
                gs4 = global_scale
            sd4 = rs4 * gs4 / d4
            tl.store(routed_w + r4 * K + rank4, p4 * sd4, mask=sel4 & v4)
            tl.store(
                indices + r4 * K + rank4, offs_n.to(tl.int32), mask=sel4 & v4
            )
            if S > 0:
                tl.store(
                    shared_w + r4 * S + offs_s, q4 * sd4, mask=mask_s & v4
                )
            sel5 = rank5 < K
            d5 = tl.sum(tl.where(sel5, p5, 0.0), axis=0)
            if S > 0:
                d5 += tl.sum(q5, axis=0)
            if ROUTE_SCALE_IS_TENSOR:
                if ROUTE_SCALE_IS_ROW:
                    rs5 = tl.load(
                        route_scale + r5 * route_scale_stride_m,
                        mask=v5,
                        other=0.0,
                    ).to(tl.float32)
                else:
                    rs5 = route_scale_h
            else:
                rs5 = route_scale
            if GLOBAL_SCALE_IS_TENSOR:
                if GLOBAL_SCALE_IS_ROW:
                    gs5 = tl.load(
                        global_scale + r5 * global_scale_stride_m,
                        mask=v5,
                        other=0.0,
                    ).to(tl.float32)
                else:
                    gs5 = global_scale_h
            else:
                gs5 = global_scale
            sd5 = rs5 * gs5 / d5
            tl.store(routed_w + r5 * K + rank5, p5 * sd5, mask=sel5 & v5)
            tl.store(
                indices + r5 * K + rank5, offs_n.to(tl.int32), mask=sel5 & v5
            )
            if S > 0:
                tl.store(
                    shared_w + r5 * S + offs_s, q5 * sd5, mask=mask_s & v5
                )
        if UROWS >= 8:
            sel6 = rank6 < K
            d6 = tl.sum(tl.where(sel6, p6, 0.0), axis=0)
            if S > 0:
                d6 += tl.sum(q6, axis=0)
            if ROUTE_SCALE_IS_TENSOR:
                if ROUTE_SCALE_IS_ROW:
                    rs6 = tl.load(
                        route_scale + r6 * route_scale_stride_m,
                        mask=v6,
                        other=0.0,
                    ).to(tl.float32)
                else:
                    rs6 = route_scale_h
            else:
                rs6 = route_scale
            if GLOBAL_SCALE_IS_TENSOR:
                if GLOBAL_SCALE_IS_ROW:
                    gs6 = tl.load(
                        global_scale + r6 * global_scale_stride_m,
                        mask=v6,
                        other=0.0,
                    ).to(tl.float32)
                else:
                    gs6 = global_scale_h
            else:
                gs6 = global_scale
            sd6 = rs6 * gs6 / d6
            tl.store(routed_w + r6 * K + rank6, p6 * sd6, mask=sel6 & v6)
            tl.store(
                indices + r6 * K + rank6, offs_n.to(tl.int32), mask=sel6 & v6
            )
            if S > 0:
                tl.store(
                    shared_w + r6 * S + offs_s, q6 * sd6, mask=mask_s & v6
                )
            sel7 = rank7 < K
            d7 = tl.sum(tl.where(sel7, p7, 0.0), axis=0)
            if S > 0:
                d7 += tl.sum(q7, axis=0)
            if ROUTE_SCALE_IS_TENSOR:
                if ROUTE_SCALE_IS_ROW:
                    rs7 = tl.load(
                        route_scale + r7 * route_scale_stride_m,
                        mask=v7,
                        other=0.0,
                    ).to(tl.float32)
                else:
                    rs7 = route_scale_h
            else:
                rs7 = route_scale
            if GLOBAL_SCALE_IS_TENSOR:
                if GLOBAL_SCALE_IS_ROW:
                    gs7 = tl.load(
                        global_scale + r7 * global_scale_stride_m,
                        mask=v7,
                        other=0.0,
                    ).to(tl.float32)
                else:
                    gs7 = global_scale_h
            else:
                gs7 = global_scale
            sd7 = rs7 * gs7 / d7
            tl.store(routed_w + r7 * K + rank7, p7 * sd7, mask=sel7 & v7)
            tl.store(
                indices + r7 * K + rank7, offs_n.to(tl.int32), mask=sel7 & v7
            )
            if S > 0:
                tl.store(
                    shared_w + r7 * S + offs_s, q7 * sd7, mask=mask_s & v7
                )
        if UROWS >= 11:
            sel8 = rank8 < K
            d8 = tl.sum(tl.where(sel8, p8, 0.0), axis=0)
            if S > 0:
                d8 += tl.sum(q8, axis=0)
            if ROUTE_SCALE_IS_TENSOR:
                if ROUTE_SCALE_IS_ROW:
                    rs8 = tl.load(
                        route_scale + r8 * route_scale_stride_m,
                        mask=v8,
                        other=0.0,
                    ).to(tl.float32)
                else:
                    rs8 = route_scale_h
            else:
                rs8 = route_scale
            if GLOBAL_SCALE_IS_TENSOR:
                if GLOBAL_SCALE_IS_ROW:
                    gs8 = tl.load(
                        global_scale + r8 * global_scale_stride_m,
                        mask=v8,
                        other=0.0,
                    ).to(tl.float32)
                else:
                    gs8 = global_scale_h
            else:
                gs8 = global_scale
            sd8 = rs8 * gs8 / d8
            tl.store(routed_w + r8 * K + rank8, p8 * sd8, mask=sel8 & v8)
            tl.store(
                indices + r8 * K + rank8, offs_n.to(tl.int32), mask=sel8 & v8
            )
            if S > 0:
                tl.store(
                    shared_w + r8 * S + offs_s, q8 * sd8, mask=mask_s & v8
                )
            sel9 = rank9 < K
            d9 = tl.sum(tl.where(sel9, p9, 0.0), axis=0)
            if S > 0:
                d9 += tl.sum(q9, axis=0)
            if ROUTE_SCALE_IS_TENSOR:
                if ROUTE_SCALE_IS_ROW:
                    rs9 = tl.load(
                        route_scale + r9 * route_scale_stride_m,
                        mask=v9,
                        other=0.0,
                    ).to(tl.float32)
                else:
                    rs9 = route_scale_h
            else:
                rs9 = route_scale
            if GLOBAL_SCALE_IS_TENSOR:
                if GLOBAL_SCALE_IS_ROW:
                    gs9 = tl.load(
                        global_scale + r9 * global_scale_stride_m,
                        mask=v9,
                        other=0.0,
                    ).to(tl.float32)
                else:
                    gs9 = global_scale_h
            else:
                gs9 = global_scale
            sd9 = rs9 * gs9 / d9
            tl.store(routed_w + r9 * K + rank9, p9 * sd9, mask=sel9 & v9)
            tl.store(
                indices + r9 * K + rank9, offs_n.to(tl.int32), mask=sel9 & v9
            )
            if S > 0:
                tl.store(
                    shared_w + r9 * S + offs_s, q9 * sd9, mask=mask_s & v9
                )
            sel10 = rank10 < K
            d10 = tl.sum(tl.where(sel10, p10, 0.0), axis=0)
            if S > 0:
                d10 += tl.sum(q10, axis=0)
            if ROUTE_SCALE_IS_TENSOR:
                if ROUTE_SCALE_IS_ROW:
                    rs10 = tl.load(
                        route_scale + r10 * route_scale_stride_m,
                        mask=v10,
                        other=0.0,
                    ).to(tl.float32)
                else:
                    rs10 = route_scale_h
            else:
                rs10 = route_scale
            if GLOBAL_SCALE_IS_TENSOR:
                if GLOBAL_SCALE_IS_ROW:
                    gs10 = tl.load(
                        global_scale + r10 * global_scale_stride_m,
                        mask=v10,
                        other=0.0,
                    ).to(tl.float32)
                else:
                    gs10 = global_scale_h
            else:
                gs10 = global_scale
            sd10 = rs10 * gs10 / d10
            tl.store(routed_w + r10 * K + rank10, p10 * sd10, mask=sel10 & v10)
            tl.store(
                indices + r10 * K + rank10,
                offs_n.to(tl.int32),
                mask=sel10 & v10,
            )
            if S > 0:
                tl.store(
                    shared_w + r10 * S + offs_s, q10 * sd10, mask=mask_s & v10
                )
        if UROWS >= 14:
            sel11 = rank11 < K
            d11 = tl.sum(tl.where(sel11, p11, 0.0), axis=0)
            if S > 0:
                d11 += tl.sum(q11, axis=0)
            if ROUTE_SCALE_IS_TENSOR:
                if ROUTE_SCALE_IS_ROW:
                    rs11 = tl.load(
                        route_scale + r11 * route_scale_stride_m,
                        mask=v11,
                        other=0.0,
                    ).to(tl.float32)
                else:
                    rs11 = route_scale_h
            else:
                rs11 = route_scale
            if GLOBAL_SCALE_IS_TENSOR:
                if GLOBAL_SCALE_IS_ROW:
                    gs11 = tl.load(
                        global_scale + r11 * global_scale_stride_m,
                        mask=v11,
                        other=0.0,
                    ).to(tl.float32)
                else:
                    gs11 = global_scale_h
            else:
                gs11 = global_scale
            sd11 = rs11 * gs11 / d11
            tl.store(routed_w + r11 * K + rank11, p11 * sd11, mask=sel11 & v11)
            tl.store(
                indices + r11 * K + rank11,
                offs_n.to(tl.int32),
                mask=sel11 & v11,
            )
            if S > 0:
                tl.store(
                    shared_w + r11 * S + offs_s, q11 * sd11, mask=mask_s & v11
                )
            sel12 = rank12 < K
            d12 = tl.sum(tl.where(sel12, p12, 0.0), axis=0)
            if S > 0:
                d12 += tl.sum(q12, axis=0)
            if ROUTE_SCALE_IS_TENSOR:
                if ROUTE_SCALE_IS_ROW:
                    rs12 = tl.load(
                        route_scale + r12 * route_scale_stride_m,
                        mask=v12,
                        other=0.0,
                    ).to(tl.float32)
                else:
                    rs12 = route_scale_h
            else:
                rs12 = route_scale
            if GLOBAL_SCALE_IS_TENSOR:
                if GLOBAL_SCALE_IS_ROW:
                    gs12 = tl.load(
                        global_scale + r12 * global_scale_stride_m,
                        mask=v12,
                        other=0.0,
                    ).to(tl.float32)
                else:
                    gs12 = global_scale_h
            else:
                gs12 = global_scale
            sd12 = rs12 * gs12 / d12
            tl.store(routed_w + r12 * K + rank12, p12 * sd12, mask=sel12 & v12)
            tl.store(
                indices + r12 * K + rank12,
                offs_n.to(tl.int32),
                mask=sel12 & v12,
            )
            if S > 0:
                tl.store(
                    shared_w + r12 * S + offs_s, q12 * sd12, mask=mask_s & v12
                )
            sel13 = rank13 < K
            d13 = tl.sum(tl.where(sel13, p13, 0.0), axis=0)
            if S > 0:
                d13 += tl.sum(q13, axis=0)
            if ROUTE_SCALE_IS_TENSOR:
                if ROUTE_SCALE_IS_ROW:
                    rs13 = tl.load(
                        route_scale + r13 * route_scale_stride_m,
                        mask=v13,
                        other=0.0,
                    ).to(tl.float32)
                else:
                    rs13 = route_scale_h
            else:
                rs13 = route_scale
            if GLOBAL_SCALE_IS_TENSOR:
                if GLOBAL_SCALE_IS_ROW:
                    gs13 = tl.load(
                        global_scale + r13 * global_scale_stride_m,
                        mask=v13,
                        other=0.0,
                    ).to(tl.float32)
                else:
                    gs13 = global_scale_h
            else:
                gs13 = global_scale
            sd13 = rs13 * gs13 / d13
            tl.store(routed_w + r13 * K + rank13, p13 * sd13, mask=sel13 & v13)
            tl.store(
                indices + r13 * K + rank13,
                offs_n.to(tl.int32),
                mask=sel13 & v13,
            )
            if S > 0:
                tl.store(
                    shared_w + r13 * S + offs_s, q13 * sd13, mask=mask_s & v13
                )
        if UROWS >= 16:
            sel14 = rank14 < K
            d14 = tl.sum(tl.where(sel14, p14, 0.0), axis=0)
            if S > 0:
                d14 += tl.sum(q14, axis=0)
            if ROUTE_SCALE_IS_TENSOR:
                if ROUTE_SCALE_IS_ROW:
                    rs14 = tl.load(
                        route_scale + r14 * route_scale_stride_m,
                        mask=v14,
                        other=0.0,
                    ).to(tl.float32)
                else:
                    rs14 = route_scale_h
            else:
                rs14 = route_scale
            if GLOBAL_SCALE_IS_TENSOR:
                if GLOBAL_SCALE_IS_ROW:
                    gs14 = tl.load(
                        global_scale + r14 * global_scale_stride_m,
                        mask=v14,
                        other=0.0,
                    ).to(tl.float32)
                else:
                    gs14 = global_scale_h
            else:
                gs14 = global_scale
            sd14 = rs14 * gs14 / d14
            tl.store(routed_w + r14 * K + rank14, p14 * sd14, mask=sel14 & v14)
            tl.store(
                indices + r14 * K + rank14,
                offs_n.to(tl.int32),
                mask=sel14 & v14,
            )
            if S > 0:
                tl.store(
                    shared_w + r14 * S + offs_s, q14 * sd14, mask=mask_s & v14
                )
            sel15 = rank15 < K
            d15 = tl.sum(tl.where(sel15, p15, 0.0), axis=0)
            if S > 0:
                d15 += tl.sum(q15, axis=0)
            if ROUTE_SCALE_IS_TENSOR:
                if ROUTE_SCALE_IS_ROW:
                    rs15 = tl.load(
                        route_scale + r15 * route_scale_stride_m,
                        mask=v15,
                        other=0.0,
                    ).to(tl.float32)
                else:
                    rs15 = route_scale_h
            else:
                rs15 = route_scale
            if GLOBAL_SCALE_IS_TENSOR:
                if GLOBAL_SCALE_IS_ROW:
                    gs15 = tl.load(
                        global_scale + r15 * global_scale_stride_m,
                        mask=v15,
                        other=0.0,
                    ).to(tl.float32)
                else:
                    gs15 = global_scale_h
            else:
                gs15 = global_scale
            sd15 = rs15 * gs15 / d15
            tl.store(routed_w + r15 * K + rank15, p15 * sd15, mask=sel15 & v15)
            tl.store(
                indices + r15 * K + rank15,
                offs_n.to(tl.int32),
                mask=sel15 & v15,
            )
            if S > 0:
                tl.store(
                    shared_w + r15 * S + offs_s, q15 * sd15, mask=mask_s & v15
                )
        if UROWS >= 18:
            sel16 = rank16 < K
            d16 = tl.sum(tl.where(sel16, p16, 0.0), axis=0)
            if S > 0:
                d16 += tl.sum(q16, axis=0)
            if ROUTE_SCALE_IS_TENSOR:
                if ROUTE_SCALE_IS_ROW:
                    rs16 = tl.load(
                        route_scale + r16 * route_scale_stride_m,
                        mask=v16,
                        other=0.0,
                    ).to(tl.float32)
                else:
                    rs16 = route_scale_h
            else:
                rs16 = route_scale
            if GLOBAL_SCALE_IS_TENSOR:
                if GLOBAL_SCALE_IS_ROW:
                    gs16 = tl.load(
                        global_scale + r16 * global_scale_stride_m,
                        mask=v16,
                        other=0.0,
                    ).to(tl.float32)
                else:
                    gs16 = global_scale_h
            else:
                gs16 = global_scale
            sd16 = rs16 * gs16 / d16
            tl.store(routed_w + r16 * K + rank16, p16 * sd16, mask=sel16 & v16)
            tl.store(
                indices + r16 * K + rank16,
                offs_n.to(tl.int32),
                mask=sel16 & v16,
            )
            if S > 0:
                tl.store(
                    shared_w + r16 * S + offs_s, q16 * sd16, mask=mask_s & v16
                )
            sel17 = rank17 < K
            d17 = tl.sum(tl.where(sel17, p17, 0.0), axis=0)
            if S > 0:
                d17 += tl.sum(q17, axis=0)
            if ROUTE_SCALE_IS_TENSOR:
                if ROUTE_SCALE_IS_ROW:
                    rs17 = tl.load(
                        route_scale + r17 * route_scale_stride_m,
                        mask=v17,
                        other=0.0,
                    ).to(tl.float32)
                else:
                    rs17 = route_scale_h
            else:
                rs17 = route_scale
            if GLOBAL_SCALE_IS_TENSOR:
                if GLOBAL_SCALE_IS_ROW:
                    gs17 = tl.load(
                        global_scale + r17 * global_scale_stride_m,
                        mask=v17,
                        other=0.0,
                    ).to(tl.float32)
                else:
                    gs17 = global_scale_h
            else:
                gs17 = global_scale
            sd17 = rs17 * gs17 / d17
            tl.store(routed_w + r17 * K + rank17, p17 * sd17, mask=sel17 & v17)
            tl.store(
                indices + r17 * K + rank17,
                offs_n.to(tl.int32),
                mask=sel17 & v17,
            )
            if S > 0:
                tl.store(
                    shared_w + r17 * S + offs_s, q17 * sd17, mask=mask_s & v17
                )
        if UROWS >= 20:
            sel18 = rank18 < K
            d18 = tl.sum(tl.where(sel18, p18, 0.0), axis=0)
            if S > 0:
                d18 += tl.sum(q18, axis=0)
            if ROUTE_SCALE_IS_TENSOR:
                if ROUTE_SCALE_IS_ROW:
                    rs18 = tl.load(
                        route_scale + r18 * route_scale_stride_m,
                        mask=v18,
                        other=0.0,
                    ).to(tl.float32)
                else:
                    rs18 = route_scale_h
            else:
                rs18 = route_scale
            if GLOBAL_SCALE_IS_TENSOR:
                if GLOBAL_SCALE_IS_ROW:
                    gs18 = tl.load(
                        global_scale + r18 * global_scale_stride_m,
                        mask=v18,
                        other=0.0,
                    ).to(tl.float32)
                else:
                    gs18 = global_scale_h
            else:
                gs18 = global_scale
            sd18 = rs18 * gs18 / d18
            tl.store(routed_w + r18 * K + rank18, p18 * sd18, mask=sel18 & v18)
            tl.store(
                indices + r18 * K + rank18,
                offs_n.to(tl.int32),
                mask=sel18 & v18,
            )
            if S > 0:
                tl.store(
                    shared_w + r18 * S + offs_s, q18 * sd18, mask=mask_s & v18
                )
            sel19 = rank19 < K
            d19 = tl.sum(tl.where(sel19, p19, 0.0), axis=0)
            if S > 0:
                d19 += tl.sum(q19, axis=0)
            if ROUTE_SCALE_IS_TENSOR:
                if ROUTE_SCALE_IS_ROW:
                    rs19 = tl.load(
                        route_scale + r19 * route_scale_stride_m,
                        mask=v19,
                        other=0.0,
                    ).to(tl.float32)
                else:
                    rs19 = route_scale_h
            else:
                rs19 = route_scale
            if GLOBAL_SCALE_IS_TENSOR:
                if GLOBAL_SCALE_IS_ROW:
                    gs19 = tl.load(
                        global_scale + r19 * global_scale_stride_m,
                        mask=v19,
                        other=0.0,
                    ).to(tl.float32)
                else:
                    gs19 = global_scale_h
            else:
                gs19 = global_scale
            sd19 = rs19 * gs19 / d19
            tl.store(routed_w + r19 * K + rank19, p19 * sd19, mask=sel19 & v19)
            tl.store(
                indices + r19 * K + rank19,
                offs_n.to(tl.int32),
                mask=sel19 & v19,
            )
            if S > 0:
                tl.store(
                    shared_w + r19 * S + offs_s, q19 * sd19, mask=mask_s & v19
                )
        if UROWS >= 22:
            sel20 = rank20 < K
            d20 = tl.sum(tl.where(sel20, p20, 0.0), axis=0)
            if S > 0:
                d20 += tl.sum(q20, axis=0)
            if ROUTE_SCALE_IS_TENSOR:
                if ROUTE_SCALE_IS_ROW:
                    rs20 = tl.load(
                        route_scale + r20 * route_scale_stride_m,
                        mask=v20,
                        other=0.0,
                    ).to(tl.float32)
                else:
                    rs20 = route_scale_h
            else:
                rs20 = route_scale
            if GLOBAL_SCALE_IS_TENSOR:
                if GLOBAL_SCALE_IS_ROW:
                    gs20 = tl.load(
                        global_scale + r20 * global_scale_stride_m,
                        mask=v20,
                        other=0.0,
                    ).to(tl.float32)
                else:
                    gs20 = global_scale_h
            else:
                gs20 = global_scale
            sd20 = rs20 * gs20 / d20
            tl.store(routed_w + r20 * K + rank20, p20 * sd20, mask=sel20 & v20)
            tl.store(
                indices + r20 * K + rank20,
                offs_n.to(tl.int32),
                mask=sel20 & v20,
            )
            if S > 0:
                tl.store(
                    shared_w + r20 * S + offs_s, q20 * sd20, mask=mask_s & v20
                )
            sel21 = rank21 < K
            d21 = tl.sum(tl.where(sel21, p21, 0.0), axis=0)
            if S > 0:
                d21 += tl.sum(q21, axis=0)
            if ROUTE_SCALE_IS_TENSOR:
                if ROUTE_SCALE_IS_ROW:
                    rs21 = tl.load(
                        route_scale + r21 * route_scale_stride_m,
                        mask=v21,
                        other=0.0,
                    ).to(tl.float32)
                else:
                    rs21 = route_scale_h
            else:
                rs21 = route_scale
            if GLOBAL_SCALE_IS_TENSOR:
                if GLOBAL_SCALE_IS_ROW:
                    gs21 = tl.load(
                        global_scale + r21 * global_scale_stride_m,
                        mask=v21,
                        other=0.0,
                    ).to(tl.float32)
                else:
                    gs21 = global_scale_h
            else:
                gs21 = global_scale
            sd21 = rs21 * gs21 / d21
            tl.store(routed_w + r21 * K + rank21, p21 * sd21, mask=sel21 & v21)
            tl.store(
                indices + r21 * K + rank21,
                offs_n.to(tl.int32),
                mask=sel21 & v21,
            )
            if S > 0:
                tl.store(
                    shared_w + r21 * S + offs_s, q21 * sd21, mask=mask_s & v21
                )


def _pow2_at_least_1(x):
    x = int(x)
    if x <= 1:
        return 1
    return triton.next_power_of_2(x)


def _row_stride_or_zero(x):
    if torch.is_tensor(x) and x.dim() > 0:
        return x.stride(0)
    return 0


def _is_row_tensor(x):
    return torch.is_tensor(x) and x.numel() > 1


# Persistent-program count: one program per SPU on this device class
# (48 SPUs, num_warps=1).
#
# Wave-aware row-group ladder (2026-11-13 round 2, GCU bf16 N=128/S=3/K=5,
# do_bench median; re-measure when the toolchain moves):
#   _ROW_ISO_US — one 48-CTA-wave worth of per-UROWS program time for the
#   FIRST loop iteration (cold pipeline + prologue); _ROW_SS_US — the
#   steady per-iteration cost with num_stages=2 overlap.  The selection
#   loop is issue-bound per program, so the per-row cost keeps falling
#   with the chain count (2.39us at U4, 1.55 at U11, 1.34 at U22) and the
#   makespan is ceil(ctas/48) iterations, so mid sizes want the rung
#   whose CTA count fills whole waves (M=512: U11 one wave 17.0us vs U4
#   three iterations 20.3us) while large sizes ride the flattest per-row
#   rung their wave count allows (M=4096: U22 103.4us vs U4 125.5us).
_PERSIST_SPC = 48
_ROW_ISO_US = (
    (1, 5.91),
    (2, 7.29),
    (4, 9.54),
    (6, 11.54),
    (8, 13.77),
    (11, 17.03),
    (14, 20.39),
    (16, 22.59),
    (18, 25.14),
    (20, 27.24),
    (22, 29.50),
)
_ROW_SS_US = {
    1: 2.28,
    2: 3.27,
    4: 5.52,
    6: 7.46,
    8: 9.55,
    11: 12.81,
    14: 15.88,
    16: 18.10,
    18: 20.73,
    20: 21.81,
    22: 24.62,
}
# Interleaved-chain budget: rows_per_program * BLOCK_N stays at or below
# the measured no-spill point (22 * 128 = 2816 lanes per chain vector; the
# topk_sigmoid sibling pinned the same 2816 budget at 11 * 256), so wide
# expert tiles dispatch to smaller row groups instead of spilling.
_CHAIN_BUDGET = 22 * 128


def _pick_rows(m, block_n):
    # Minimize ceil(ctas / 48) iterations x per-iteration cost over the
    # ladder rungs that fit the chain budget.  Small sizes keep the
    # latency-optimal one/two-row programs; the table only needs to bound
    # the estimate from above (median model error 1.4% over a 125-point
    # probe sweep, and every probe-measured M resolved to its measured
    # best rung or within 5% of it).
    best_rows = 1
    best_cost = None
    for rows, iso_us in _ROW_ISO_US:
        if rows * block_n > _CHAIN_BUDGET:
            break
        ctas = -(-m // rows)
        iters = -(-ctas // _PERSIST_SPC)
        cost = iso_us + (iters - 1) * _ROW_SS_US[rows]
        if best_cost is None or cost < best_cost - 1e-9:
            best_rows, best_cost = rows, cost
    return best_rows


def sigmoid_gate_topk_renorm(
    logits, k, n_shared_experts, route_scale, global_scale, bias
):
    M, G = logits.shape
    K = int(k)
    S = int(n_shared_experts)
    N = G - S

    routed_w = torch.empty((M, K), device=logits.device, dtype=logits.dtype)
    indices = torch.empty((M, K), device=logits.device, dtype=torch.int32)
    shared_w = torch.empty((M, S), device=logits.device, dtype=logits.dtype)

    if M == 0:
        return routed_w, indices, shared_w

    block_n = _pow2_at_least_1(N)
    block_s = _pow2_at_least_1(S)
    urows = _pick_rows(M, block_n)
    nprog = min(_PERSIST_SPC, triton.cdiv(M, urows))

    _sigmoid_gate_topk_renorm_kernel[(nprog,)](
        logits,
        bias,
        routed_w,
        indices,
        shared_w,
        M,
        nprog,
        logits.stride(0),
        logits.stride(1),
        bias.stride(0),
        route_scale,
        global_scale,
        _row_stride_or_zero(route_scale),
        _row_stride_or_zero(global_scale),
        N,
        S,
        K,
        BLOCK_N=block_n,
        BLOCK_S=block_s,
        ROUTE_SCALE_IS_TENSOR=torch.is_tensor(route_scale),
        ROUTE_SCALE_IS_ROW=_is_row_tensor(route_scale),
        GLOBAL_SCALE_IS_TENSOR=torch.is_tensor(global_scale),
        GLOBAL_SCALE_IS_ROW=_is_row_tensor(global_scale),
        N_EXACT=block_n == N,
        UROWS=urows,
        num_warps=1,
        num_stages=2,
    )

    return routed_w, indices, shared_w


__all__ = ["sigmoid_gate_topk_renorm"]
