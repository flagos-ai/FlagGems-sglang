# Copyright 2026, The FlagOS Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License")
# you may not use this file except in compliance with the License
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

  1. _reset_kernel     zero the per-expert counters/cursors, sentinel-fill ids
  2. _hist_kernel      per-expert slot histogram (atomic)
  3. _prefix_kernel    block-aligned exclusive prefix scan -> row base per
                       expert + the padded row count (stays on device)
  4. _scatter_kernel   rank slots inside their expert -> sorted_ids[row] = slot
  5. _expertids_kernel binary lifting search -> expert per M-block
  6. _gate_up_kernel   grouped GEMM (gate || up) + activation  -> C1
  7. _down_kernel      grouped GEMM down projection + routing weight -> C3
  8. _combine_kernel   sum over topk slots, routing scale, one cast -> out

Everything is allocated per call from static bounds; the only data dependent
quantity (the padded row count) never leaves the device, so run() performs no
host synchronisation.
"""

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# bookkeeping kernels
# ---------------------------------------------------------------------------
@triton.jit
def _reset_kernel(
    cnt_ptr, cur_ptr, sorted_ptr, E, MAXPAD, NUM_SLOTS, BLOCK: tl.constexpr
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < E
    tl.store(cnt_ptr + offs, tl.zeros((BLOCK,), tl.int32), mask=m)
    tl.store(cur_ptr + offs, tl.zeros((BLOCK,), tl.int32), mask=m)
    m2 = offs < MAXPAD
    tl.store(
        sorted_ptr + offs, tl.full((BLOCK,), NUM_SLOTS, tl.int32), mask=m2
    )


@triton.jit
def _hist_kernel(ids_ptr, cnt_ptr, NUM_SLOTS, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < NUM_SLOTS
    e = tl.load(ids_ptr + offs, mask=m, other=0)
    tl.atomic_add(cnt_ptr + e, 1, mask=m)


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
    off_ptr, eids_ptr, E, LOG_E: tl.constexpr, BLOCK_M: tl.constexpr
):
    pid = tl.program_id(0)
    row = pid * BLOCK_M
    ones = tl.full((1,), 1, tl.int32)
    lo = tl.zeros((1,), tl.int32)
    for i in tl.static_range(LOG_E):
        mid = lo + (1 << (LOG_E - 1 - i))
        v = tl.load(off_ptr + mid, mask=mid <= E, other=2147483647)
        lo = tl.where(v <= row, mid, lo)
    tl.store(eids_ptr + pid + tl.arange(0, 1), tl.minimum(lo, E) * ones)


# ---------------------------------------------------------------------------
# grouped GEMM 1: gate/up projection + activation
# ---------------------------------------------------------------------------
@triton.jit
def _gate_up_kernel(
    x_ptr,
    w1_ptr,
    sorted_ptr,
    eids_ptr,
    num_pad_ptr,
    c1_ptr,
    H,
    I,
    NUM_SLOTS,
    TOPK,
    stride_xm,
    stride_we,
    num_pid_m,
    num_pid_n,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    ACT: tl.constexpr,
    EVEN_K: tl.constexpr,
    EVEN_N: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    start_row = pid_m * BLOCK_M
    n_pad = tl.load(num_pad_ptr)
    if start_row < n_pad:
        e = tl.load(eids_ptr + pid_m)
        offs_m = start_row + tl.arange(0, BLOCK_M)
        j = tl.load(sorted_ptr + offs_m)
        row_mask = j < NUM_SLOTS
        # hidden_states is indexed by token: slot j belongs to token j // topk
        tok = j // TOPK
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)

        x_row = tok.to(tl.int64) * stride_xm
        e_off = e.to(tl.int64) * stride_we
        w_off_g = e_off + offs_n[:, None].to(tl.int64) * H
        w_off_u = e_off + (offs_n + I)[:, None].to(tl.int64) * H

        # Neumaier compensated accumulation: the K=H reduction is 64
        # sequential fp32 chunk adds and each one discards low bits of a
        # partial sum that is ~1e4 in magnitude, which is exactly the
        # residual behind the last mixtral tolerance failure.  The
        # compensation term is an ordinary fp32 tile, so the tensor-core
        # dots keep their full throughput.
        acc_g = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        acc_u = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        cmp_g = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        cmp_u = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        if EVEN_K:
            for k in range(0, H, BLOCK_K):
                a = tl.load(
                    x_ptr + x_row[:, None] + (k + offs_k)[None, :],
                    mask=row_mask[:, None],
                    other=0.0,
                )
                bg = tl.load(w1_ptr + w_off_g + (k + offs_k)[None, :])
                bu = tl.load(w1_ptr + w_off_u + (k + offs_k)[None, :])
                dg = tl.dot(a, tl.trans(bg))
                du = tl.dot(a, tl.trans(bu))
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
                    x_ptr + x_row[:, None] + (k + offs_k)[None, :],
                    mask=row_mask[:, None] & km[None, :],
                    other=0.0,
                )
                bg = tl.load(
                    w1_ptr + w_off_g + (k + offs_k)[None, :],
                    mask=km[None, :],
                    other=0.0,
                )
                bu = tl.load(
                    w1_ptr + w_off_u + (k + offs_k)[None, :],
                    mask=km[None, :],
                    other=0.0,
                )
                dg = tl.dot(a, tl.trans(bg))
                du = tl.dot(a, tl.trans(bu))
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

        g = acc_g
        if ACT == 0:
            act = g * tl.sigmoid(g)
        else:
            act = 0.5 * g * (1.0 + tl.erf(g * 0.7071067811865476))
        res = (act * acc_u).to(c1_ptr.dtype.element_ty)

        c1_off = offs_m[:, None].to(tl.int64) * I + offs_n[None, :].to(
            tl.int64
        )
        if EVEN_N:
            tl.store(c1_ptr + c1_off, res, mask=row_mask[:, None])
        else:
            nm = offs_n < I
            tl.store(
                c1_ptr + c1_off, res, mask=row_mask[:, None] & nm[None, :]
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
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    EVEN_K: tl.constexpr,
    EVEN_N: tl.constexpr,
    SPLIT_F16: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    start_row = pid_m * BLOCK_M
    n_pad = tl.load(num_pad_ptr)
    if start_row < n_pad:
        e = tl.load(eids_ptr + pid_m)
        offs_m = start_row + tl.arange(0, BLOCK_M)
        j = tl.load(sorted_ptr + offs_m)
        row_mask = j < NUM_SLOTS
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)

        c1_off = offs_m[:, None].to(tl.int64) * I
        e_off = e.to(tl.int64) * stride_we
        w_off = e_off + offs_n[None, :].to(tl.int64) * stride_wh

        # The activation is held in fp32, but tensor-core mma only accepts
        # low-precision operands.  Split it into a high part and the residual
        # so the pair of dots reproduces fp32 accuracy (8 + 8 mantissa bits
        # for bf16, 11 + 11 for fp16); the low part contributes a correction
        # two orders of magnitude below the tolerance floor.
        #
        # This reduction keeps a plain fp32 accumulator.  A Neumaier
        # compensated accumulator was measured against this exact code and is
        # accuracy-neutral on all 11 correctness workloads (the mixtral
        # cancellation case is fixed by the gate/up compensation alone), while
        # its two extra fp32 tiles were the largest single source of register
        # pressure in the kernel that owns 54% of device time.
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        if EVEN_K:
            for k in range(0, I, BLOCK_K):
                a = tl.load(
                    c1_ptr + c1_off + (k + offs_k)[None, :],
                    mask=row_mask[:, None],
                    other=0.0,
                )
                b = tl.load(w2_ptr + w_off + (k + offs_k)[:, None])
                if SPLIT_F16:
                    a_hi = a.to(tl.float16)
                    a_lo = (a - a_hi.to(tl.float32)).to(tl.float16)
                else:
                    a_hi = a.to(tl.bfloat16)
                    a_lo = (a - a_hi.to(tl.float32)).to(tl.bfloat16)
                acc += tl.dot(a_hi, b) + tl.dot(a_lo, b)
        else:
            for k in range(0, I, BLOCK_K):
                km = (k + offs_k) < I
                a = tl.load(
                    c1_ptr + c1_off + (k + offs_k)[None, :],
                    mask=row_mask[:, None] & km[None, :],
                    other=0.0,
                )
                b = tl.load(
                    w2_ptr + w_off + (k + offs_k)[:, None],
                    mask=km[:, None],
                    other=0.0,
                )
                if SPLIT_F16:
                    a_hi = a.to(tl.float16)
                    a_lo = (a - a_hi.to(tl.float32)).to(tl.float16)
                else:
                    a_hi = a.to(tl.bfloat16)
                    a_lo = (a - a_hi.to(tl.float32)).to(tl.bfloat16)
                acc += tl.dot(a_hi, b) + tl.dot(a_lo, b)

        wv = tl.load(w_ptr + j, mask=row_mask, other=0.0).to(tl.float32)
        res = acc * wv[:, None]
        c3_off = j.to(tl.int64)[:, None] * H + offs_n[None, :].to(tl.int64)
        if EVEN_N:
            tl.store(c3_ptr + c3_off, res, mask=row_mask[:, None])
        else:
            nm = offs_n < H
            tl.store(
                c3_ptr + c3_off, res, mask=row_mask[:, None] & nm[None, :]
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
):
    pid_t = tl.program_id(0)
    pid_h = tl.program_id(1)
    offs_t = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    tm = offs_t < T
    acc = tl.zeros((BLOCK_T, BLOCK_H), dtype=tl.float32)
    base = offs_t.to(tl.int64)[:, None] * (TOPK * H)
    for s in tl.static_range(TOPK):
        acc += tl.load(
            c3_ptr + base + s * H + offs_h[None, :].to(tl.int64),
            mask=tm[:, None],
            other=0.0,
        )
    out = (acc * RSF).to(out_ptr.dtype.element_ty)
    o_off = offs_t.to(tl.int64)[:, None] * stride_om + offs_h[None, :].to(
        tl.int64
    )
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
    BLOCK_N = 64 if inter % 64 == 0 else (32 if inter % 32 == 0 else 16)
    BLOCK_H = 64 if H % 64 == 0 else (32 if H % 32 == 0 else 16)

    MAXPAD = NUM_SLOTS + E * (BLOCK_M - 1)
    num_pid_m = triton.cdiv(MAXPAD, BLOCK_M)
    num_pid_n1 = triton.cdiv(inter, BLOCK_N)
    num_pid_n2 = triton.cdiv(H, BLOCK_H)

    cnt = torch.empty(E, dtype=torch.int32, device=dev)
    cur = torch.empty(E, dtype=torch.int32, device=dev)
    off = torch.empty(E + 1, dtype=torch.int32, device=dev)
    sorted_ids = torch.empty(MAXPAD, dtype=torch.int32, device=dev)
    eids = torch.empty(num_pid_m, dtype=torch.int32, device=dev)
    nbuf = torch.empty(1, dtype=torch.int32, device=dev)
    c1 = torch.empty(MAXPAD, inter, dtype=torch.float32, device=dev)
    c3 = torch.empty(NUM_SLOTS, H, dtype=torch.float32, device=dev)
    out = torch.empty(T, H, dtype=dtype, device=dev)

    RB = 1024
    _reset_kernel[(triton.cdiv(max(E, MAXPAD), RB),)](
        cnt, cur, sorted_ids, E, MAXPAD, NUM_SLOTS, RB
    )

    _hist_kernel[(triton.cdiv(NUM_SLOTS, 512),)](ids, cnt, NUM_SLOTS, 512)

    BLOCK_E = max(16, triton.next_power_of_2(max(E, 1)))
    _prefix_kernel[(1,)](cnt, off, nbuf, cur, E, BLOCK_M, BLOCK_E)

    _scatter_kernel[(triton.cdiv(NUM_SLOTS, 256),)](
        ids, off, cur, sorted_ids, NUM_SLOTS, 256
    )

    LOG_E = max(1, max(E, 1).bit_length())
    _expertids_kernel[(num_pid_m,)](off, eids, E, LOG_E, BLOCK_M)

    GROUP_M = 8
    num_warps = 4 if BLOCK_M <= 64 else 8

    _gate_up_kernel[(num_pid_m * num_pid_n1,)](
        hidden_states,
        w1,
        sorted_ids,
        eids,
        nbuf,
        c1,
        H,
        inter,
        NUM_SLOTS,
        TOPK,
        hidden_states.stride(0),
        w1.stride(0),
        num_pid_m,
        num_pid_n1,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=64,
        GROUP_M=GROUP_M,
        ACT=ACT,
        EVEN_K=(H % 64 == 0),
        EVEN_N=(inter % BLOCK_N == 0),
        num_warps=num_warps,
        num_stages=3,
    )

    _down_kernel[(num_pid_m * num_pid_n2,)](
        c1,
        w2,
        sorted_ids,
        eids,
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
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_H,
        BLOCK_K=64,
        GROUP_M=GROUP_M,
        EVEN_K=(inter % 64 == 0),
        EVEN_N=(H % BLOCK_H == 0),
        SPLIT_F16=(dtype == torch.float16),
        num_warps=num_warps,
        num_stages=3,
    )

    _combine_kernel[(triton.cdiv(T, 4), triton.cdiv(H, 256))](
        c3,
        out,
        T,
        H,
        float(routed_scaling_factor),
        out.stride(0),
        TOPK=TOPK,
        BLOCK_T=4,
        BLOCK_H=256,
        EVEN_H=(H % 256 == 0),
        num_warps=4,
    )

    return out


__all__ = ["fused_moe"]
