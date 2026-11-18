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
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Triton fused MoE experts (unquantized path), SGLang fused_experts semantics.

Sorted grouped-GEMM formulation (same family as vLLM/SGLang's fused_moe):

  1. _reset_kernel     zero the per-expert counters/cursors, zero-fill ids
                       (padding rows carry slot 0 -- a valid x row -- so the
                       GEMM A loads run unmasked; validity is recomputed in
                       the GEMMs from the per-expert off/cnt scalars)
  2. _hist_kernel      per-expert slot histogram (atomic; _hist2_kernel for
                       >= 64 slots: one-hot local reduce + one vector atomic
                       per program)
  3. _prefix_kernel    block-aligned exclusive prefix scan -> row base per
                       expert + the padded row count (stays on device)
  4. _scatter_kernel   rank slots inside their expert -> sorted_ids[row] = slot
  5. _expertids_kernel expert id per M-block (one vector scan over the
                       padded row bases -- the old binary-lifting search
                       chained log2(E) dependent scalar loads)
  6. _gate_up_kernel   grouped GEMM (gate || up) + activation  -> C1
  7. _down_kernel      grouped GEMM down projection + routing weight -> C3
  8. _combine_kernel   sum over topk slots, routing scale, one cast -> out

Everything is allocated per call from static bounds; the only data dependent
quantity (the padded row count) never leaves the device, so run() performs no
host synchronisation.

GCU (enflame) specifics vs the generic implementation:

  * A directly-loaded bf16/fp16 tile fed straight into ``tl.dot`` hits a DSM
    staging pathology on the enflame backend (the ``bf16 A 操作数直进 dot``
    entry in the GCU compiler-trap list): the grouped GEMMs run ~40x slower
    than their memory floor.  Routing the A operand through a lossless
    ``int16`` bitcast round-trip after the load takes the integer/computed
    path instead and restores full speed (measured on the bench shapes:
    gate_up 114.2ms -> 2.0ms at num_tokens=512).
  * B operands are loaded directly as ``[BLOCK_K, BLOCK_N]`` tiles (K axis
    contiguous) and fed to the 3-operand ``tl.dot(a, b, acc)`` form; no
    ``tl.trans`` and no separate accumulate add per K-iteration.
  * The expert base is hoisted into a scalar pointer and all vector offsets
    stay int32 (a WIDE constexpr keeps int64 vector math for buffers larger
    than 2**31 elements).
  * bf16/fp16 inputs skip the Neumaier compensation and the hi/lo split of
    the generic code and keep the activation/combine intermediates in the
    input dtype: the products are exact in fp32 and the bf16/fp16 test
    tolerances (2e-2 / 1e-2) are orders of magnitude above the introduced
    rounding.  fp32 inputs keep the generic compensated semantics.
  * Tile configs are DSM-budget ladders; a launch that exceeds the hardware
    DSM limit falls through to the next smaller candidate.
"""

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# bookkeeping kernels
# ---------------------------------------------------------------------------
@triton.jit
def _reset_kernel(
    cnt_ptr, cur_ptr, sorted_ptr, E, FILL_ROWS, BLOCK: tl.constexpr
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < E
    tl.store(cnt_ptr + offs, tl.zeros((BLOCK,), tl.int32), mask=m)
    tl.store(cur_ptr + offs, tl.zeros((BLOCK,), tl.int32), mask=m)
    m2 = offs < FILL_ROWS
    # Padding rows carry slot id 0: a valid row of x, so the grouped-GEMM A
    # loads can drop their row masks entirely (a masked 2D A load with mostly
    # padding rows drags the whole dot pipeline into the element-wise slow
    # path on this backend).  Row validity is recomputed in the GEMMs from
    # the per-expert off/cnt scalars instead of from the slot id.
    tl.store(sorted_ptr + offs, tl.zeros((BLOCK,), tl.int32), mask=m2)


@triton.jit
def _hist_kernel(ids_ptr, cnt_ptr, NUM_SLOTS, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < NUM_SLOTS
    e = tl.load(ids_ptr + offs, mask=m, other=0)
    tl.atomic_add(cnt_ptr + e, 1, mask=m)


@triton.jit
def _hist2_kernel(
    ids_ptr, cnt_ptr, NUM_SLOTS, E, BLOCK: tl.constexpr, BLOCK_E: tl.constexpr
):
    # Local-reduce histogram: per-program one-hot sum over the expert bins,
    # then a single masked vector atomic merge per program.  Cuts the atomic
    # count from NUM_SLOTS to BLOCK_E * num_programs, which wins once there
    # are enough slots for cross-program contention to serialise the scalar
    # atomics (measured on gcu400: 4.44x at num_tokens=512, 1.28x at 8,
    # 0.62x at 1 -- the host dispatches on NUM_SLOTS >= 64).
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < NUM_SLOTS
    e = tl.load(ids_ptr + offs, mask=m, other=E)
    bins = tl.arange(0, BLOCK_E)
    onehot = (e[:, None] == bins[None, :]).to(tl.int32)
    local = tl.sum(onehot, axis=0)
    tl.atomic_add(cnt_ptr + bins, local, mask=bins < E)


@triton.jit
def _prefix_kernel(
    cnt_ptr,
    off_ptr,
    num_pad_ptr,
    cur_ptr,
    E,
    BLOCK_M: tl.constexpr,
    BLOCK_E: tl.constexpr,
):
    offs = tl.arange(0, BLOCK_E)
    m = offs < E
    c = tl.load(cnt_ptr + offs, mask=m, other=0)
    pad = ((c + BLOCK_M - 1) // BLOCK_M) * BLOCK_M
    pad = tl.where(m, pad, 0)
    incl = tl.cumsum(pad, axis=0)
    excl = incl - pad
    tl.store(off_ptr + offs, excl, mask=m)
    total = tl.sum(pad, axis=0)
    tl.store(off_ptr + E, total)
    tl.store(num_pad_ptr, total)
    tl.store(cur_ptr + offs, tl.zeros((BLOCK_E,), tl.int32), mask=m)


@triton.jit
def _scatter_kernel(
    ids_ptr, off_ptr, cur_ptr, sorted_ptr, NUM_SLOTS, BLOCK: tl.constexpr
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < NUM_SLOTS
    e = tl.load(ids_ptr + offs, mask=m, other=0)
    base = tl.load(off_ptr + e, mask=m, other=0)
    rank = tl.atomic_add(cur_ptr + e, 1, mask=m)
    tl.store(sorted_ptr + base + rank, offs, mask=m)


@triton.jit
def _expertids_kernel(
    off_ptr, eids_ptr, E, BLOCK_E: tl.constexpr, BLOCK_M: tl.constexpr
):
    # Owning expert of each M-block: one masked vector load + a compare/sum
    # scan over the padded row bases.  The old binary-lifting search chained
    # log2(E) dependent scalar loads per program and measured 14-27us.
    pid = tl.program_id(0)
    row = pid * BLOCK_M
    bins = tl.arange(0, BLOCK_E)
    off_vec = tl.load(off_ptr + bins, mask=bins < E, other=2147483647)
    e = tl.sum((off_vec <= row).to(tl.int32), axis=0) - 1
    tl.store(eids_ptr + pid, e)


# ---------------------------------------------------------------------------
# grouped GEMM 1: gate/up projection + activation
# ---------------------------------------------------------------------------
@triton.jit
def _gate_up_kernel(
    x_ptr,
    w1_ptr,
    sorted_ptr,
    eids_ptr,
    off_ptr,
    cnt_ptr,
    num_pad_ptr,
    c1_ptr,
    H,
    I,
    NUM_SLOTS,
    stride_xm,
    stride_we,
    num_pid_m,
    num_pid_n,
    TOPK: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    ACT: tl.constexpr,
    EVEN_K: tl.constexpr,
    EVEN_N: tl.constexpr,
    COMPENSATE: tl.constexpr,
    A_FP16: tl.constexpr,
    WIDE: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    start_row = pid_m * BLOCK_M
    n_pad = tl.load(num_pad_ptr)
    if start_row < n_pad:
        e = tl.load(eids_ptr + pid_m)
        offs_m = start_row + tl.arange(0, BLOCK_M)
        j = tl.load(sorted_ptr + offs_m)
        tok = j // TOPK
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)

        if WIDE:
            x_row = tok.to(tl.int64) * stride_xm
        else:
            x_row = tok * stride_xm
        x_base = x_ptr + x_row[:, None]
        w_e = w1_ptr + e.to(tl.int64) * stride_we
        n_gate = offs_n * H
        n_up = (offs_n + I) * H

        acc_g = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        acc_u = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        if COMPENSATE:
            cmp_g = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            cmp_u = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            if EVEN_K:
                for k in range(0, H, BLOCK_K):
                    a = tl.load(x_base + (k + offs_k)[None, :])
                    if EVEN_N:
                        bg = tl.load(
                            w_e + n_gate[None, :] + (k + offs_k)[:, None]
                        )
                        bu = tl.load(
                            w_e + n_up[None, :] + (k + offs_k)[:, None]
                        )
                    else:
                        nm = offs_n < I
                        bg = tl.load(
                            w_e + n_gate[None, :] + (k + offs_k)[:, None],
                            mask=nm[None, :],
                            other=0.0,
                        )
                        bu = tl.load(
                            w_e + n_up[None, :] + (k + offs_k)[:, None],
                            mask=nm[None, :],
                            other=0.0,
                        )
                    dg = tl.dot(a, bg, out_dtype=tl.float32)
                    du = tl.dot(a, bu, out_dtype=tl.float32)
                    tg = acc_g + dg
                    tu = acc_u + du
                    cmp_g += tl.where(
                        tl.abs(acc_g) >= tl.abs(dg),
                        (acc_g - tg) + dg,
                        (dg - tg) + acc_g,
                    )
                    cmp_u += tl.where(
                        tl.abs(acc_u) >= tl.abs(du),
                        (acc_u - tu) + du,
                        (du - tu) + acc_u,
                    )
                    acc_g = tg
                    acc_u = tu
            else:
                for k in range(0, H, BLOCK_K):
                    km = (k + offs_k) < H
                    a = tl.load(
                        x_base + (k + offs_k)[None, :],
                        mask=km[None, :],
                        other=0.0,
                    )
                    if EVEN_N:
                        bg = tl.load(
                            w_e + n_gate[None, :] + (k + offs_k)[:, None],
                            mask=km[:, None],
                            other=0.0,
                        )
                        bu = tl.load(
                            w_e + n_up[None, :] + (k + offs_k)[:, None],
                            mask=km[:, None],
                            other=0.0,
                        )
                    else:
                        nm = offs_n < I
                        bg = tl.load(
                            w_e + n_gate[None, :] + (k + offs_k)[:, None],
                            mask=km[:, None] & nm[None, :],
                            other=0.0,
                        )
                        bu = tl.load(
                            w_e + n_up[None, :] + (k + offs_k)[:, None],
                            mask=km[:, None] & nm[None, :],
                            other=0.0,
                        )
                    dg = tl.dot(a, bg, out_dtype=tl.float32)
                    du = tl.dot(a, bu, out_dtype=tl.float32)
                    tg = acc_g + dg
                    tu = acc_u + du
                    cmp_g += tl.where(
                        tl.abs(acc_g) >= tl.abs(dg),
                        (acc_g - tg) + dg,
                        (dg - tg) + acc_g,
                    )
                    cmp_u += tl.where(
                        tl.abs(acc_u) >= tl.abs(du),
                        (acc_u - tu) + du,
                        (du - tu) + acc_u,
                    )
                    acc_g = tg
                    acc_u = tu
            acc_g = acc_g + cmp_g
            acc_u = acc_u + cmp_u
        else:
            if EVEN_K:
                for k in range(0, H, BLOCK_K):
                    a = tl.load(x_base + (k + offs_k)[None, :])
                    if A_FP16:
                        a = a.to(tl.int16, bitcast=True).to(
                            tl.float16, bitcast=True
                        )
                    else:
                        a = a.to(tl.int16, bitcast=True).to(
                            tl.bfloat16, bitcast=True
                        )
                    if EVEN_N:
                        bg = tl.load(
                            w_e + n_gate[None, :] + (k + offs_k)[:, None]
                        )
                        bu = tl.load(
                            w_e + n_up[None, :] + (k + offs_k)[:, None]
                        )
                    else:
                        nm = offs_n < I
                        bg = tl.load(
                            w_e + n_gate[None, :] + (k + offs_k)[:, None],
                            mask=nm[None, :],
                            other=0.0,
                        )
                        bu = tl.load(
                            w_e + n_up[None, :] + (k + offs_k)[:, None],
                            mask=nm[None, :],
                            other=0.0,
                        )
                    acc_g = tl.dot(a, bg, acc_g, out_dtype=tl.float32)
                    acc_u = tl.dot(a, bu, acc_u, out_dtype=tl.float32)
            else:
                for k in range(0, H, BLOCK_K):
                    km = (k + offs_k) < H
                    a = tl.load(
                        x_base + (k + offs_k)[None, :],
                        mask=km[None, :],
                        other=0.0,
                    )
                    if A_FP16:
                        a = a.to(tl.int16, bitcast=True).to(
                            tl.float16, bitcast=True
                        )
                    else:
                        a = a.to(tl.int16, bitcast=True).to(
                            tl.bfloat16, bitcast=True
                        )
                    if EVEN_N:
                        bg = tl.load(
                            w_e + n_gate[None, :] + (k + offs_k)[:, None],
                            mask=km[:, None],
                            other=0.0,
                        )
                        bu = tl.load(
                            w_e + n_up[None, :] + (k + offs_k)[:, None],
                            mask=km[:, None],
                            other=0.0,
                        )
                    else:
                        nm = offs_n < I
                        bg = tl.load(
                            w_e + n_gate[None, :] + (k + offs_k)[:, None],
                            mask=km[:, None] & nm[None, :],
                            other=0.0,
                        )
                        bu = tl.load(
                            w_e + n_up[None, :] + (k + offs_k)[:, None],
                            mask=km[:, None] & nm[None, :],
                            other=0.0,
                        )
                    acc_g = tl.dot(a, bg, acc_g, out_dtype=tl.float32)
                    acc_u = tl.dot(a, bu, acc_u, out_dtype=tl.float32)

        g = acc_g
        if ACT == 0:
            act = g * tl.sigmoid(g)
        else:
            act = 0.5 * g * (1.0 + tl.erf(g * 0.7071067811865476))
        res = (act * acc_u).to(c1_ptr.dtype.element_ty)

        if EVEN_N:
            tl.store(c1_ptr + offs_m[:, None] * I + offs_n[None, :], res)
        else:
            nm = offs_n < I
            tl.store(
                c1_ptr + offs_m[:, None] * I + offs_n[None, :],
                res,
                mask=nm[None, :],
            )


# ---------------------------------------------------------------------------
# grouped GEMM 2: down projection + routing weight
# ---------------------------------------------------------------------------
@triton.jit
def _down_kernel(
    c1_ptr,
    w2_ptr,
    sorted_ptr,
    eids_ptr,
    off_ptr,
    cnt_ptr,
    num_pad_ptr,
    w_ptr,
    c3_ptr,
    H,
    I,
    NUM_SLOTS,
    stride_we,
    stride_wh,
    num_pid_m,
    num_pid_n,
    TOPK: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    EVEN_K: tl.constexpr,
    EVEN_N: tl.constexpr,
    SPLIT: tl.constexpr,
    SPLIT_F16: tl.constexpr,
    A_FP16: tl.constexpr,
    WIDE: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    start_row = pid_m * BLOCK_M
    n_pad = tl.load(num_pad_ptr)
    if start_row < n_pad:
        e = tl.load(eids_ptr + pid_m)
        offs_m = start_row + tl.arange(0, BLOCK_M)
        j = tl.load(sorted_ptr + offs_m)
        off_e = tl.load(off_ptr + e)
        cnt_e = tl.load(cnt_ptr + e)
        row_valid = offs_m < (off_e + cnt_e)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)

        if WIDE:
            c1_off = offs_m.to(tl.int64) * I
            c3_off = j.to(tl.int64) * H
        else:
            c1_off = offs_m * I
            c3_off = j * H
        w_e = w2_ptr + e.to(tl.int64) * stride_we
        n_w = offs_n * stride_wh

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        if EVEN_K:
            for k in range(0, I, BLOCK_K):
                a = tl.load(c1_ptr + c1_off[:, None] + (k + offs_k)[None, :])
                if EVEN_N:
                    b = tl.load(w_e + n_w[None, :] + (k + offs_k)[:, None])
                else:
                    nm = offs_n < H
                    b = tl.load(
                        w_e + n_w[None, :] + (k + offs_k)[:, None],
                        mask=nm[None, :],
                        other=0.0,
                    )
                if SPLIT:
                    if SPLIT_F16:
                        a_hi = a.to(tl.float16)
                        a_lo = (a - a_hi.to(tl.float32)).to(tl.float16)
                    else:
                        a_hi = a.to(tl.bfloat16)
                        a_lo = (a - a_hi.to(tl.float32)).to(tl.bfloat16)
                    acc = tl.dot(a_hi, b, acc, out_dtype=tl.float32)
                    acc = tl.dot(a_lo, b, acc, out_dtype=tl.float32)
                else:
                    if A_FP16:
                        a = a.to(tl.int16, bitcast=True).to(
                            tl.float16, bitcast=True
                        )
                    else:
                        a = a.to(tl.int16, bitcast=True).to(
                            tl.bfloat16, bitcast=True
                        )
                    acc = tl.dot(a, b, acc, out_dtype=tl.float32)
        else:
            for k in range(0, I, BLOCK_K):
                km = (k + offs_k) < I
                a = tl.load(
                    c1_ptr + c1_off[:, None] + (k + offs_k)[None, :],
                    mask=km[None, :],
                    other=0.0,
                )
                if EVEN_N:
                    b = tl.load(
                        w_e + n_w[None, :] + (k + offs_k)[:, None],
                        mask=km[:, None],
                        other=0.0,
                    )
                else:
                    nm = offs_n < H
                    b = tl.load(
                        w_e + n_w[None, :] + (k + offs_k)[:, None],
                        mask=km[:, None] & nm[None, :],
                        other=0.0,
                    )
                if SPLIT:
                    if SPLIT_F16:
                        a_hi = a.to(tl.float16)
                        a_lo = (a - a_hi.to(tl.float32)).to(tl.float16)
                    else:
                        a_hi = a.to(tl.bfloat16)
                        a_lo = (a - a_hi.to(tl.float32)).to(tl.bfloat16)
                    acc = tl.dot(a_hi, b, acc, out_dtype=tl.float32)
                    acc = tl.dot(a_lo, b, acc, out_dtype=tl.float32)
                else:
                    if A_FP16:
                        a = a.to(tl.int16, bitcast=True).to(
                            tl.float16, bitcast=True
                        )
                    else:
                        a = a.to(tl.int16, bitcast=True).to(
                            tl.bfloat16, bitcast=True
                        )
                    acc = tl.dot(a, b, acc, out_dtype=tl.float32)

        wv = tl.load(w_ptr + j, mask=row_valid, other=0.0).to(tl.float32)
        res = acc * wv[:, None]
        if EVEN_N:
            tl.store(
                c3_ptr + c3_off[:, None] + offs_n[None, :],
                res,
                mask=row_valid[:, None],
            )
        else:
            nm = offs_n < H
            tl.store(
                c3_ptr + c3_off[:, None] + offs_n[None, :],
                res,
                mask=row_valid[:, None] & nm[None, :],
            )


# ---------------------------------------------------------------------------
# topk combine
# ---------------------------------------------------------------------------
@triton.jit
def _combine_kernel(
    c3_ptr,
    out_ptr,
    T,
    H,
    RSF,
    stride_om,
    TOPK: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_H: tl.constexpr,
    EVEN_H: tl.constexpr,
    EVEN_T: tl.constexpr,
    WIDE: tl.constexpr,
):
    pid_t = tl.program_id(0)
    pid_h = tl.program_id(1)
    offs_t = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    tm = offs_t < T
    acc = tl.zeros((BLOCK_T, BLOCK_H), dtype=tl.float32)
    if WIDE:
        base = offs_t.to(tl.int64)[:, None] * (TOPK * H)
        o_off = offs_t.to(tl.int64)[:, None] * stride_om + offs_h[None, :]
    else:
        base = offs_t[:, None] * (TOPK * H)
        o_off = offs_t[:, None] * stride_om + offs_h[None, :]
    # The 2D loads stay unpredicated when T fills the token block exactly:
    # a predicated 2D load with (nearly) all-true rows still pays the masked
    # lowering tax and measured ~10x below the streaming floor here.
    for s in tl.static_range(TOPK):
        if EVEN_T:
            acc += tl.load(c3_ptr + base + s * H + offs_h[None, :])
        else:
            acc += tl.load(
                c3_ptr + base + s * H + offs_h[None, :],
                mask=tm[:, None],
                other=0.0,
            )
    out = (acc * RSF).to(out_ptr.dtype.element_ty)
    if EVEN_T:
        if EVEN_H:
            tl.store(out_ptr + o_off, out)
        else:
            nm = offs_h < H
            tl.store(out_ptr + o_off, out, mask=nm[None, :])
    else:
        if EVEN_H:
            tl.store(out_ptr + o_off, out, mask=tm[:, None])
        else:
            nm = offs_h < H
            tl.store(out_ptr + o_off, out, mask=tm[:, None] & nm[None, :])


# ---------------------------------------------------------------------------
# host side
# ---------------------------------------------------------------------------
def _pick_block_m(num_slots, E):
    """Padding granularity for the sorted rows.

    BLOCK_M is both the padding unit (at most BLOCK_M-1 wasted rows per expert)
    and the granularity of the weight re-reads: every extra M-block per expert
    re-reads that expert's whole w1/w2 slice.  Measured interleaved on the
    timing shapes, BLOCK_M=64 beat BLOCK_M=128 by 1.6x on mixtral-b8192 and
    1.7x on qwen3moe-b8192 (both of which the old >=64 rule sent to 128), while
    the small-token shapes keep their 16/32 choice, so 64 is the cap.
    """
    avg = (num_slots + E - 1) // E
    if avg <= 8:
        return 16
    if avg <= 32:
        return 32
    return 64


# Tile ladders for the grouped GEMMs, tuned on the sglang bench shapes
# (num_tokens 1/8/64/512, hidden 2048, inter 1408, 64 experts, topk 8, bf16;
# gcu400, 2026-11-08, retuned 2026-11-13).  The first entry is the tuned
# optimum per BLOCK_M padding granularity; a launch that exceeds the hardware
# DSM budget falls through to the next candidate via the OutOfResources retry
# in fused_moe().  The fp32 (COMPENSATE / SPLIT) paths start from the generic
# 64x64x64 tiles instead.
# 2026-11-13 retune rounds:
#   round 2 (do_bench w1000/r100, 3 interleaved A/B rounds): the down GEMM
#     prefers a wide 512x256 tile (512B contiguous K-segments per weight
#     column, half the c1 re-reads) -- 1.16/1.93/1.87/1.63x at num_tokens
#     512/64/8/1 vs the old 256x128.
#   round 3 (rejected): BLOCK_K=512 tiles are no win -- gate_up (128,512)
#     runs 19% slower than (256,256) at T=512, down (256,512) exceeds the DSM
#     budget (required 970752 > 917504); adding a full/tail K-loop split
#     also inflates DSM for the even-K path (branch-tile union), pushing the
#     previously-fitting (512,256) to OOR, and each OOR ladder fall
#     re-compiles the failed tile at every call (~0.5s/call).
#   round 4 (after the unmasked-A change, per-BLOCK_M sweep with the real
#     launch grids): wider BLOCK_N wins at the larger paddings -- gate_up
#     (512,128) at BLOCK_M=64 (-10% vs (256,256)), (256,128) at BLOCK_M=32;
#     down flips to (256,256) at BLOCK_M=64 (-6% vs (512,256)).  BLOCK_M=128
#     rejected: gate_up (256,256)/(512,128) go OOR and the fitting tiles run
#     15-30% slower than BLOCK_M=64.
_GATE_UP_TILES_BY_BM = {
    16: ((256, 256), (128, 256), (256, 128), (128, 128), (64, 64)),
    32: ((256, 256), (128, 256), (256, 128), (128, 128), (64, 64)),
    64: ((512, 128), (128, 256), (256, 256), (128, 128), (64, 64)),
}
_DOWN_TILES_BY_BM = {
    16: ((512, 256), (256, 128), (128, 128), (64, 64)),
    32: ((512, 256), (256, 128), (128, 128), (64, 64)),
    64: ((256, 256), (512, 128), (256, 128), (128, 128), (64, 64)),
}
_GATE_UP_TILES_F32 = ((64, 64), (64, 32), (32, 32))
_DOWN_TILES_F32 = ((64, 64), (64, 32), (32, 32))

# OutOfResources memo: a tile whose compile exceeds the DSM budget leaves no
# Triton cache entry, so every subsequent call would re-attempt the ~0.5s
# compile before falling through the ladder.  Remember failed specializations
# (the DSM requirement is a function of the constexpr key only) and skip
# them outright on later calls.
_oor_memo = set()


def fused_moe(
    hidden_states,
    w1,
    w2,
    topk_output,
    activation="silu",
    routed_scaling_factor=1.0,
):
    topk_weights, topk_ids, _ = topk_output

    T, H = hidden_states.shape
    E, N, _ = w1.shape
    inter = N // 2
    topk = topk_ids.shape[1]
    NUM_SLOTS = T * topk
    dev = hidden_states.device
    dtype = hidden_states.dtype

    if activation == "silu":
        ACT = 0
    elif activation == "gelu":
        ACT = 1
    else:
        raise ValueError("Unsupported activation: %s" % activation)

    ids = topk_ids.reshape(-1)
    wflat = topk_weights.reshape(-1)
    E = int(E)
    TOPK = int(topk)

    BLOCK_M = _pick_block_m(NUM_SLOTS, E)

    # Low-precision inputs take the bitcast fast path with input-dtype
    # intermediates; fp32 keeps the generic compensated semantics.
    LOWP = dtype in (torch.bfloat16, torch.float16)
    COMPENSATE = not LOWP
    SPLIT = not LOWP
    c1_dtype = dtype if LOWP else torch.float32
    c3_dtype = dtype if LOWP else torch.float32
    if LOWP:
        gate_tiles = _GATE_UP_TILES_BY_BM.get(
            BLOCK_M, _GATE_UP_TILES_BY_BM[64]
        )
        down_tiles = _DOWN_TILES_BY_BM.get(BLOCK_M, _DOWN_TILES_BY_BM[64])
    else:
        gate_tiles = _GATE_UP_TILES_F32
        down_tiles = _DOWN_TILES_F32
    warps = 8 if LOWP else 4

    MAXPAD = NUM_SLOTS + E * (BLOCK_M - 1)
    num_pid_m = triton.cdiv(MAXPAD, BLOCK_M)
    # Round the row buffers up to the M-grid so the unmasked A loads and the
    # unmasked c1 stores of the GEMMs never touch memory past the tensors
    # (the last M-block can overshoot MAXPAD by up to BLOCK_M-1 rows).
    alloc_rows = num_pid_m * BLOCK_M

    wide = (
        w1.numel() >= 2**31
        or w2.numel() >= 2**31
        or hidden_states.numel() >= 2**31
        or alloc_rows * inter >= 2**31
        or NUM_SLOTS * H >= 2**31
        or T * TOPK * H >= 2**31
    )

    cnt = torch.empty(E, dtype=torch.int32, device=dev)
    cur = torch.empty(E, dtype=torch.int32, device=dev)
    off = torch.empty(E + 1, dtype=torch.int32, device=dev)
    sorted_ids = torch.empty(alloc_rows, dtype=torch.int32, device=dev)
    eids = torch.empty(num_pid_m, dtype=torch.int32, device=dev)
    nbuf = torch.empty(1, dtype=torch.int32, device=dev)
    c1 = torch.empty(alloc_rows, inter, dtype=c1_dtype, device=dev)
    c3 = torch.empty(NUM_SLOTS, H, dtype=c3_dtype, device=dev)
    out = torch.empty(T, H, dtype=dtype, device=dev)

    RB = 1024
    _reset_kernel[(triton.cdiv(max(E, alloc_rows), RB),)](
        cnt, cur, sorted_ids, E, alloc_rows, RB
    )

    BLOCK_E = max(16, triton.next_power_of_2(max(E, 1)))
    if NUM_SLOTS >= 64:
        _hist2_kernel[(triton.cdiv(NUM_SLOTS, 512),)](
            ids, cnt, NUM_SLOTS, E, 512, BLOCK_E
        )
    else:
        _hist_kernel[(triton.cdiv(NUM_SLOTS, 512),)](ids, cnt, NUM_SLOTS, 512)

    _prefix_kernel[(1,)](cnt, off, nbuf, cur, E, BLOCK_M, BLOCK_E)

    _scatter_kernel[(triton.cdiv(NUM_SLOTS, 256),)](
        ids, off, cur, sorted_ids, NUM_SLOTS, 256
    )

    _expertids_kernel[(num_pid_m,)](off, eids, E, BLOCK_E, BLOCK_M)

    for bn, bk in gate_tiles:
        bk = min(bk, triton.next_power_of_2(max(H, 1)))
        num_pid_n1 = triton.cdiv(inter, bn)
        gkey = (
            "gate_up",
            bn,
            bk,
            BLOCK_M,
            TOPK,
            ACT,
            H,
            inter,
            warps,
            (H % bk == 0),
            (inter % bn == 0),
            COMPENSATE,
            (dtype == torch.float16),
            wide,
        )
        if gkey in _oor_memo:
            continue
        try:
            _gate_up_kernel[(num_pid_m * num_pid_n1,)](
                hidden_states,
                w1,
                sorted_ids,
                eids,
                off,
                cnt,
                nbuf,
                c1,
                H,
                inter,
                NUM_SLOTS,
                hidden_states.stride(0),
                w1.stride(0),
                num_pid_m,
                num_pid_n1,
                TOPK=TOPK,
                BLOCK_M=BLOCK_M,
                BLOCK_N=bn,
                BLOCK_K=bk,
                ACT=ACT,
                EVEN_K=(H % bk == 0),
                EVEN_N=(inter % bn == 0),
                COMPENSATE=COMPENSATE,
                A_FP16=(dtype == torch.float16),
                WIDE=wide,
                num_warps=warps,
                num_stages=2,
            )
            break
        except triton.runtime.errors.OutOfResources:
            _oor_memo.add(gkey)
            continue

    for bn, bk in down_tiles:
        bk = min(bk, triton.next_power_of_2(max(inter, 1)))
        num_pid_n2 = triton.cdiv(H, bn)
        dkey = (
            "down",
            bn,
            bk,
            BLOCK_M,
            TOPK,
            H,
            inter,
            warps,
            (inter % bk == 0),
            (H % bn == 0),
            SPLIT,
            (dtype == torch.float16) and SPLIT,
            (dtype == torch.float16),
            wide,
        )
        if dkey in _oor_memo:
            continue
        try:
            _down_kernel[(num_pid_m * num_pid_n2,)](
                c1,
                w2,
                sorted_ids,
                eids,
                off,
                cnt,
                nbuf,
                wflat,
                c3,
                H,
                inter,
                NUM_SLOTS,
                w2.stride(0),
                w2.stride(1),
                num_pid_m,
                num_pid_n2,
                TOPK=TOPK,
                BLOCK_M=BLOCK_M,
                BLOCK_N=bn,
                BLOCK_K=bk,
                EVEN_K=(inter % bk == 0),
                EVEN_N=(H % bn == 0),
                SPLIT=SPLIT,
                SPLIT_F16=(dtype == torch.float16) and SPLIT,
                A_FP16=(dtype == torch.float16),
                WIDE=wide,
                num_warps=warps,
                num_stages=2,
            )
            break
        except triton.runtime.errors.OutOfResources:
            _oor_memo.add(dkey)
            continue

    # Combine: wide BH keeps the c3 row segments (the DMA descriptor length)
    # >= 1KB once there is enough token parallelism; small T keeps the 512B
    # form that keeps the grid occupied (measured at T=512: 78us -> 42us at
    # BH=512, 32us at BH=1024; the segment-issue rate, not the mask, is the
    # limiter -- EVEN_T changed nothing).
    BT = 16
    if T >= 256:
        BH = min(1024, triton.next_power_of_2(max(H, 1)))
    else:
        BH = 256
    _combine_kernel[(triton.cdiv(T, BT), triton.cdiv(H, BH))](
        c3,
        out,
        T,
        H,
        float(routed_scaling_factor),
        out.stride(0),
        TOPK=TOPK,
        BLOCK_T=BT,
        BLOCK_H=BH,
        EVEN_H=(H % BH == 0),
        EVEN_T=(T % BT == 0),
        WIDE=wide,
        num_warps=4,
    )

    return out


__all__ = ["fused_moe"]
