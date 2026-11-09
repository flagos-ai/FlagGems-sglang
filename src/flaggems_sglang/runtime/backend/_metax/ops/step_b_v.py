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

import torch
import triton
import triton.language as tl


@triton.jit
def _step_b_v_kernel(PTRS, META: tl.constexpr):
    X = PTRS[0]
    W = PTRS[1]
    BASE = PTRS[2]
    OUT = PTRS[3]
    SEG = PTRS[4]
    WIDX = PTRS[5]
    RANKS = PTRS[6]
    SCALES = PTRS[7]
    PERM = PTRS[8]
    N: tl.constexpr = META[1]
    K: tl.constexpr = META[2]
    F: tl.constexpr = META[3]
    Q: tl.constexpr = META[4]
    S: tl.constexpr = META[5]
    PL: tl.constexpr = META[6]
    BS: tl.constexpr = META[7]
    HAS_PERM: tl.constexpr = META[8]
    SMALL: tl.constexpr = META[9]
    XS0: tl.constexpr = META[10]
    XS1: tl.constexpr = META[11]
    XS2: tl.constexpr = META[12]
    WS0: tl.constexpr = META[13]
    WS1: tl.constexpr = META[14]
    WS2: tl.constexpr = META[15]
    GS0: tl.constexpr = META[16]
    GS1: tl.constexpr = META[17]
    GS2: tl.constexpr = META[18]
    BO0: tl.constexpr = META[19]
    BO1: tl.constexpr = META[20]
    BO2: tl.constexpr = META[21]
    BM: tl.constexpr = META[22]
    BN: tl.constexpr = META[23]
    BK: tl.constexpr = META[24]
    SM1: tl.constexpr = META[25]
    pid_s = tl.program_id(0)
    h = tl.program_id(1)
    b = tl.program_id(2)
    if (b == 0) & (pid_s == 0):
        lim = tl.load(SEG + BS)
        if not HAS_PERM or PL == S:
            prefix = tl.load(SEG)
            if prefix > 0:
                pn = tl.arange(0, BN)
                pm = pn < N
                for pr in range(0, prefix):
                    if HAS_PERM:
                        prow = tl.load(PERM + pr).to(tl.int32)
                    else:
                        prow = pr
                    pv = tl.load(
                        BASE + prow * BO0 + h * BO1 + pn * BO2, pm, other=0.0
                    )
                    tl.store(OUT + prow * GS0 + h * GS1 + pn * GS2, pv, pm)
            if lim < S:
                offs_n = tl.arange(0, BN)
                nm = offs_n < N
                for r in range(lim, S):
                    if HAS_PERM:
                        row = tl.load(PERM + r).to(tl.int32)
                    else:
                        row = r
                    bp = row * BO0 + h * BO1 + offs_n * BO2
                    v = tl.load(BASE + bp, mask=nm, other=0.0)
                    tl.store(
                        OUT + row * GS0 + h * GS1 + offs_n * GS2, v, mask=nm
                    )
    lo = tl.load(SEG + b)
    hi = tl.load(SEG + b + 1)
    if hi == lo:
        return
    seg_len = hi - lo
    i0 = pid_s * BM
    if i0 >= seg_len:
        return
    w = tl.load(WIDX + b)
    offs_n = tl.arange(0, BN)
    offs_k = tl.arange(0, BK)
    nmask = offs_n < N
    kmask = offs_k < K
    j = tl.arange(0, BM)
    if HAS_PERM:
        idx = lo + i0 + j
        rows = tl.load(PERM + tl.minimum(idx, PL - 1)).to(tl.int32)
        rmask = i0 + j < seg_len
        rx = tl.minimum(rows, SM1)
    elif SMALL:
        rows = lo + i0 + j
        rmask = i0 + j < seg_len
        rx = tl.minimum(rows, SM1)
    else:
        i0s = tl.maximum(tl.minimum(i0, seg_len - BM), 0)
        r0 = tl.maximum(tl.minimum(lo + i0s, S - BM), 0)
        off = r0 - (lo + i0)
        rows = r0 + j
        rmask = (j >= -off) & (j < seg_len - i0 - off)
        rx = rows
    x = tl.load(
        X + rx[:, None] * XS0 + h * XS1 + offs_k[None, :] * XS2,
        mask=rmask[:, None] & kmask[None, :],
        other=0.0,
    )
    wb = tl.load(
        W
        + w * WS0
        + (h * F + Q + offs_n[None, :]) * WS1
        + offs_k[:, None] * WS2,
        mask=kmask[:, None] & nmask[None, :],
        other=0.0,
    )
    active = tl.load(RANKS + w) > 0
    scale = tl.load(SCALES + w)
    if X.dtype.element_ty == tl.float32:
        raw = tl.dot(x, wb, input_precision="ieee")
    else:
        raw = tl.dot(x, wb)
    acc = tl.where(active, raw * scale, 0.0)
    m2 = rmask[:, None] & nmask[None, :]
    op = OUT + rx[:, None] * GS0 + h * GS1 + offs_n[None, :] * GS2
    base = tl.load(
        BASE + rx[:, None] * BO0 + h * BO1 + offs_n[None, :] * BO2,
        mask=m2,
        other=0.0,
    ).to(tl.float32)
    tl.store(op, (base + acc).to(OUT.dtype.element_ty), mask=m2)


@triton.jit
def _copy_base_kernel(PTRS, META: tl.constexpr):
    BASE = PTRS[0]
    OUT = PTRS[1]
    TOTAL: tl.constexpr = META[0]
    BLOCK: tl.constexpr = META[1]
    if TOTAL == 0:
        return
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    value = tl.load(BASE + i, i < TOTAL, other=0.0).to(tl.float32)
    tl.store(OUT + i, value, i < TOTAL)


def _optimized_step_b_v(
    attn_lora_a, B_buf, batch_info, base_output, qk_nope_head_dim, v_head_dim
):
    if batch_info.bs == 0 or base_output.numel() == 0:
        base = base_output.contiguous()
        out = torch.empty_like(base)
        total = base.numel()
        _copy_base_kernel.run(
            (base, out),
            (total, 1024),
            grid=(max(1, (total + 1024 - 1) // 1024),),
            warmup=False,
        )
        return out
    S, H, K = attn_lora_a.shape
    N = v_head_dim
    F = qk_nope_head_dim + N
    perm = batch_info.permutation
    if perm is not None:
        perm = perm.contiguous()
    has_perm = perm is not None
    bn = max(16, 0 if N == 0 else 1 << (N - 1).bit_length())
    bk = max(16, 0 if K == 0 else 1 << (K - 1).bit_length())
    ml = batch_info.max_len
    if not ml:
        ml = S
    bm = min(64, max(16, 0 if ml == 0 else 1 << (ml - 1).bit_length()))
    pl = perm.numel() if has_perm else S
    if has_perm and pl != S:
        out = base_output.clone(memory_format=torch.contiguous_format)
    else:
        out = base_output.new_empty((S + bm, H, N))
    if not has_perm:
        perm = attn_lora_a
    _step_b_v_kernel.run(
        (
            attn_lora_a,
            B_buf,
            base_output,
            out,
            batch_info.seg_indptr.contiguous(),
            batch_info.weight_indices.contiguous(),
            batch_info.lora_ranks.contiguous(),
            batch_info.scalings.contiguous(),
            perm,
        ),
        (
            H,
            N,
            K,
            F,
            qk_nope_head_dim,
            S,
            pl,
            batch_info.bs,
            has_perm,
            S < bm,
            *attn_lora_a.stride(),
            *B_buf.stride(),
            *out.stride(),
            *base_output.stride(),
            bm,
            bn,
            bk,
            S - 1,
        ),
        num_warps=4,
        num_stages=2,
        pipeline="cpasync",
        scenario="fullstage",
        grid=((ml + bm - 1) // bm, H, batch_info.bs),
        warmup=False,
    )
    return out[:S]


@triton.jit
def _bounded_rows(PTRS, META: tl.constexpr):
    X = PTRS[0]
    W = PTRS[1]
    OUT = PTRS[2]
    SEG = PTRS[3]
    WI = PTRS[4]
    RANKS = PTRS[5]
    SCALES = PTRS[6]
    PERM = PTRS[7]
    H: tl.constexpr = META[0]
    N: tl.constexpr = META[1]
    K: tl.constexpr = META[2]
    F: tl.constexpr = META[3]
    Q: tl.constexpr = META[4]
    ML: tl.constexpr = META[5]
    NT: tl.constexpr = META[6]
    TOTAL: tl.constexpr = META[7]
    GRID: tl.constexpr = META[8]
    PL: tl.constexpr = META[9]
    HAS_PERM: tl.constexpr = META[10]
    X0: tl.constexpr = META[11]
    X1: tl.constexpr = META[12]
    X2: tl.constexpr = META[13]
    W0: tl.constexpr = META[14]
    W1: tl.constexpr = META[15]
    W2: tl.constexpr = META[16]
    BK: tl.constexpr = META[17]
    BN: tl.constexpr = META[18]
    for work in range(tl.program_id(0), TOTAL, GRID):
        panel = work % NT
        h = work // NT % H
        m = work // (NT * H) % ML
        b = work // (NT * H * ML)
        lo = tl.load(SEG + b)
        hi = tl.load(SEG + b + 1)
        logical = lo + m
        if (logical < hi) & (logical < PL):
            w = tl.load(WI + b)
            if tl.load(RANKS + w) != 0:
                if HAS_PERM:
                    row = tl.load(PERM + logical).to(tl.int32)
                else:
                    row = logical
                n = panel * BN + tl.arange(0, BN)
                ks = tl.arange(0, BK)
                correction = tl.zeros((BN,), tl.float32)
                for chunk in range(tl.cdiv(K, BK)):
                    k = chunk * BK + ks
                    x = tl.load(
                        X + row * X0 + h * X1 + k * X2, k < K, other=0.0
                    ).to(tl.float32)
                    weight = tl.load(
                        W
                        + w * W0
                        + (h * F + Q + n[None, :]) * W1
                        + k[:, None] * W2,
                        (k < K)[:, None] & (n < N)[None, :],
                        other=0.0,
                    ).to(tl.float32)
                    correction += tl.sum(x[:, None] * weight, axis=0)
                ptr = OUT + (row * H + h) * N + n
                base = tl.load(ptr, n < N, other=0.0).to(tl.float32)
                scale = tl.load(SCALES + w).to(tl.float32)
                tl.store(ptr, base + correction * scale, n < N)


def step_b_v(
    attn_lora_a, B_buf, batch_info, base_output, qk_nope_head_dim, v_head_dim
):
    if (
        v_head_dim <= 128
        and attn_lora_a.shape[-1] * attn_lora_a.element_size() <= 256
        or base_output.numel() == 0
        or batch_info.bs == 0
        or (B_buf.shape[0] == 0)
    ):
        return _optimized_step_b_v(
            attn_lora_a,
            B_buf,
            batch_info,
            base_output,
            qk_nope_head_dim,
            v_head_dim,
        )
    x = attn_lora_a.contiguous()
    weight = B_buf.contiguous()
    S, H, K = x.shape
    N = v_head_dim
    ml = batch_info.max_len or S
    nt = (N + 32 - 1) // 32
    total = ml * H * batch_info.bs * nt
    grid = max(1, min(8192, total))
    out = base_output.clone(memory_format=torch.contiguous_format)
    perm = batch_info.permutation
    p = perm.contiguous() if perm is not None else batch_info.seg_indptr
    _bounded_rows.run(
        (
            x,
            weight,
            out,
            batch_info.seg_indptr.contiguous(),
            batch_info.weight_indices.contiguous(),
            batch_info.lora_ranks.contiguous(),
            batch_info.scalings.contiguous(),
            p,
        ),
        (
            H,
            N,
            K,
            qk_nope_head_dim + N,
            qk_nope_head_dim,
            ml,
            nt,
            total,
            grid,
            perm.numel() if perm is not None else S,
            perm is not None,
            *x.stride(),
            *weight.stride(),
            32,
            32,
        ),
        grid=(grid,),
        warmup=False,
    )
    return out


__all__ = ["step_b_v"]
