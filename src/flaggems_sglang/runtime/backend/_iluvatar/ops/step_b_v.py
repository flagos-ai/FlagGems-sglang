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
def _step_b_v_kernel_flat(
    X, W, BASE, OUT, SEG, WIDX, RANKS, SCALES, PERM, META: tl.constexpr
):
    H: tl.constexpr = META[0]
    N: tl.constexpr = META[1]
    S: tl.constexpr = META[2]
    K: tl.constexpr = META[3]
    F: tl.constexpr = META[4]
    Q: tl.constexpr = META[5]
    BS: tl.constexpr = META[6]
    XS0: tl.constexpr = META[7]
    XS1: tl.constexpr = META[8]
    XS2: tl.constexpr = META[9]
    WS0: tl.constexpr = META[10]
    WS1: tl.constexpr = META[11]
    WS2: tl.constexpr = META[12]
    GS0: tl.constexpr = META[13]
    GS1: tl.constexpr = META[14]
    GS2: tl.constexpr = META[15]
    BM: tl.constexpr = META[16]
    BN: tl.constexpr = META[17]
    BK: tl.constexpr = META[18]
    HAS_PERM: tl.constexpr = META[19]
    PL: tl.constexpr = META[20]
    EXACT: tl.constexpr = META[21]
    NFULL: tl.constexpr = META[22]
    KFULL: tl.constexpr = META[23]
    h = tl.program_id(0)
    i0 = tl.program_id(1) * BM
    idx = i0 + tl.arange(0, BM)
    if HAS_PERM:
        keep = idx < PL
        rows = tl.load(PERM + idx, mask=keep, other=0).to(tl.int32)
    else:
        keep = idx < S
        rows = idx
    offs_k = tl.arange(0, BK)
    offs_n = tl.arange(0, BN)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    rscale = tl.zeros((BM,), dtype=tl.float32)
    owner = tl.full((), -1, tl.int32)
    if not HAS_PERM and S >= BM:
        for segment in range(BS):
            lower = tl.load(SEG + segment)
            upper = tl.load(SEG + segment + 1)
            if (lower <= i0) & (upper >= i0 + BM):
                owner = segment
    if owner >= 0:
        w = tl.load(WIDX + owner)
        if tl.load(RANKS + w) > 0:
            xp = X + rows[:, None] * XS0 + h * XS1 + offs_k[None, :] * XS2
            wp = (
                W
                + w * WS0
                + (h * F + Q + offs_n[None, :]) * WS1
                + offs_k[:, None] * WS2
            )
            if KFULL:
                x = tl.load(xp)
                if NFULL:
                    wb = tl.load(wp, eviction_policy="evict_last")
                else:
                    wb = tl.load(
                        wp,
                        mask=(offs_n < N)[None, :],
                        other=0.0,
                        eviction_policy="evict_last",
                    )
            else:
                x = tl.load(xp, mask=(offs_k < K)[None, :], other=0.0)
                wb = tl.load(
                    wp,
                    mask=(offs_k < K)[:, None] & (offs_n < N)[None, :],
                    other=0.0,
                    eviction_policy="evict_last",
                )
            if X.dtype.element_ty == tl.float32:
                acc = tl.dot(x, wb, input_precision="ieee")
            else:
                acc = tl.dot(x, wb)
            scale = tl.load(SCALES + w)
            bp = BASE + rows[:, None] * GS0 + h * GS1 + offs_n[None, :] * GS2
            op = OUT + rows[:, None] * (H * N) + h * N + offs_n[None, :]
            if EXACT:
                base = tl.load(bp, eviction_policy="evict_first").to(
                    tl.float32
                )
                tl.store(op, (base + scale * acc).to(OUT.dtype.element_ty))
            else:
                if NFULL:
                    m2 = keep[:, None]
                else:
                    m2 = keep[:, None] & (offs_n < N)[None, :]
                base = tl.load(
                    bp, mask=m2, other=0.0, eviction_policy="evict_first"
                ).to(tl.float32)
                tl.store(
                    op, (base + scale * acc).to(OUT.dtype.element_ty), mask=m2
                )
        else:
            bp = BASE + rows[:, None] * GS0 + h * GS1 + offs_n[None, :] * GS2
            op = OUT + rows[:, None] * (H * N) + h * N + offs_n[None, :]
            if EXACT:
                base = tl.load(bp, eviction_policy="evict_first").to(
                    tl.float32
                )
                tl.store(op, base)
            else:
                if NFULL:
                    m2 = keep[:, None]
                else:
                    m2 = keep[:, None] & (offs_n < N)[None, :]
                base = tl.load(
                    bp, mask=m2, other=0.0, eviction_policy="evict_first"
                ).to(tl.float32)
                tl.store(op, base, mask=m2)
        return
    else:
        for b in range(BS):
            lo = tl.load(SEG + b)
            hi = tl.load(SEG + b + 1)
            if (lo < i0 + BM) & (hi > i0):
                w = tl.load(WIDX + b)
                if tl.load(RANKS + w) > 0:
                    rmask = (idx >= lo) & (idx < hi)
                    if not EXACT:
                        rmask = rmask & keep
                    xp = (
                        X
                        + rows[:, None] * XS0
                        + h * XS1
                        + offs_k[None, :] * XS2
                    )
                    wp = (
                        W
                        + w * WS0
                        + (h * F + Q + offs_n[None, :]) * WS1
                        + offs_k[:, None] * WS2
                    )
                    if KFULL:
                        x = tl.load(xp, mask=rmask[:, None], other=0.0)
                        if NFULL:
                            wb = tl.load(wp, eviction_policy="evict_last")
                        else:
                            wb = tl.load(
                                wp,
                                mask=(offs_n < N)[None, :],
                                other=0.0,
                                eviction_policy="evict_last",
                            )
                    else:
                        x = tl.load(
                            xp,
                            mask=rmask[:, None] & (offs_k < K)[None, :],
                            other=0.0,
                        )
                        wb = tl.load(
                            wp,
                            mask=(offs_k < K)[:, None] & (offs_n < N)[None, :],
                            other=0.0,
                            eviction_policy="evict_last",
                        )
                    if X.dtype.element_ty == tl.float32:
                        updated = tl.dot(x, wb, acc, input_precision="ieee")
                    else:
                        updated = tl.dot(x, wb, acc)
                    if (lo <= i0) & (hi >= i0 + BM):
                        acc = updated
                    else:
                        acc = tl.where(rmask[:, None], updated, acc)
                    rscale = tl.where(rmask, tl.load(SCALES + w), rscale)
    bp = BASE + rows[:, None] * GS0 + h * GS1 + offs_n[None, :] * GS2
    op = OUT + rows[:, None] * (H * N) + h * N + offs_n[None, :]
    if EXACT:
        base = tl.load(bp, eviction_policy="evict_first").to(tl.float32)
        tl.store(op, (base + rscale[:, None] * acc).to(OUT.dtype.element_ty))
    else:
        if NFULL:
            m2 = keep[:, None]
        else:
            m2 = keep[:, None] & (offs_n < N)[None, :]
        base = tl.load(
            bp, mask=m2, other=0.0, eviction_policy="evict_first"
        ).to(tl.float32)
        tl.store(
            op,
            (base + rscale[:, None] * acc).to(OUT.dtype.element_ty),
            mask=m2,
        )


def step_b_v_flat(
    attn_lora_a, B_buf, batch_info, base_output, qk_nope_head_dim, v_head_dim
):
    S, H, K = attn_lora_a.shape
    N = v_head_dim
    perm = batch_info.permutation
    if perm is not None:
        perm = perm.contiguous()
    has_perm = perm is not None
    bn = max(16, 0 if N == 0 else 1 << (N - 1).bit_length())
    bk = max(16, 0 if K == 0 else 1 << (K - 1).bit_length())
    bm = max(16, 4096 // bn)
    if has_perm and perm.numel() != S:
        out = base_output.clone(memory_format=torch.contiguous_format)
    else:
        out = base_output.new_empty((S, H, N))
    if not has_perm:
        perm = attn_lora_a
    _step_b_v_kernel_flat.run(
        attn_lora_a,
        B_buf,
        base_output,
        out,
        batch_info.seg_indptr.contiguous(),
        batch_info.weight_indices.contiguous(),
        batch_info.lora_ranks.contiguous(),
        batch_info.scalings.contiguous(),
        perm,
        (
            H,
            N,
            S,
            K,
            qk_nope_head_dim + N,
            qk_nope_head_dim,
            batch_info.bs,
            *attn_lora_a.stride(),
            *B_buf.stride(),
            *base_output.stride(),
            bm,
            bn,
            bk,
            has_perm,
            perm.numel() if has_perm else S,
            not has_perm and S % bm == 0 and (N == bn),
            N == bn,
            K == bk,
        ),
        num_warps=4,
        num_stages=2,
        grid=(H, (S + bm - 1) // bm),
        warmup=False,
    )
    return out


@triton.jit
def _step_b_v_kernel_segment(
    X, W, BASE, OUT, SEG, WIDX, RANKS, SCALES, PERM, META: tl.constexpr
):
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
    pid_s = tl.program_id(1) if K == 16 or K == 32 else tl.program_id(0)
    h = tl.program_id(0) if K == 16 or K == 32 else tl.program_id(1)
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
    if K == BK and N == BN and (not HAS_PERM):
        if i0 + BM <= seg_len:
            fast_rows = lo + i0 + j
            fast_bp = (
                BASE
                + fast_rows[:, None] * BO0
                + h * BO1
                + offs_n[None, :] * BO2
            )
            fast_op = (
                OUT
                + fast_rows[:, None] * GS0
                + h * GS1
                + offs_n[None, :] * GS2
            )
            if tl.load(RANKS + w) > 0:
                fast_base = tl.load(fast_bp, eviction_policy="evict_first")
                fast_x = tl.load(
                    X
                    + fast_rows[:, None] * XS0
                    + h * XS1
                    + offs_k[None, :] * XS2
                )
                fast_w = tl.load(
                    W
                    + w * WS0
                    + (h * F + Q + offs_n[None, :]) * WS1
                    + offs_k[:, None] * WS2,
                    eviction_policy="evict_last",
                )
                if X.dtype.element_ty == tl.float32:
                    fast_acc = tl.dot(fast_x, fast_w, input_precision="ieee")
                else:
                    fast_acc = tl.dot(fast_x, fast_w)
                fast_scale = tl.load(SCALES + w)
                tl.store(
                    fast_op,
                    (fast_base.to(tl.float32) + fast_acc * fast_scale).to(
                        OUT.dtype.element_ty
                    ),
                )
            else:
                inactive_base = tl.load(fast_bp, eviction_policy="evict_first")
                tl.store(fast_op, inactive_base)
            return
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
    if tl.load(RANKS + w) == 0:
        m2 = rmask[:, None] & nmask[None, :]
        op = OUT + rx[:, None] * GS0 + h * GS1 + offs_n[None, :] * GS2
        base = tl.load(
            BASE + rx[:, None] * BO0 + h * BO1 + offs_n[None, :] * BO2,
            mask=m2,
            other=0.0,
            eviction_policy="evict_first",
        )
        tl.store(op, base, mask=m2)
        return
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
        eviction_policy="evict_last",
    )
    scale = tl.where(tl.load(RANKS + w) > 0, tl.load(SCALES + w), 0.0)
    if X.dtype.element_ty == tl.float32:
        acc = tl.dot(x, wb, input_precision="ieee") * scale
    else:
        acc = tl.dot(x, wb) * scale
    m2 = rmask[:, None] & nmask[None, :]
    op = OUT + rx[:, None] * GS0 + h * GS1 + offs_n[None, :] * GS2
    base = tl.load(
        BASE + rx[:, None] * BO0 + h * BO1 + offs_n[None, :] * BO2,
        mask=m2,
        other=0.0,
        eviction_policy="evict_first",
    ).to(tl.float32)
    tl.store(op, (base + acc).to(OUT.dtype.element_ty), mask=m2)


@triton.jit(do_not_specialize=["K_RUNTIME"])
def _step_b_v_kernel_segment_kloop(
    X,
    W,
    BASE,
    OUT,
    SEG,
    WIDX,
    RANKS,
    SCALES,
    PERM,
    K_RUNTIME,
    META: tl.constexpr,
):
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
    pid_s = tl.program_id(1) if K == 16 or K == 32 else tl.program_id(0)
    h = tl.program_id(0) if K == 16 or K == 32 else tl.program_id(1)
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
    if K == BK and N == BN and (not HAS_PERM):
        if i0 + BM <= seg_len:
            fast_rows = lo + i0 + j
            fast_bp = (
                BASE
                + fast_rows[:, None] * BO0
                + h * BO1
                + offs_n[None, :] * BO2
            )
            fast_op = (
                OUT
                + fast_rows[:, None] * GS0
                + h * GS1
                + offs_n[None, :] * GS2
            )
            if tl.load(RANKS + w) > 0:
                fast_acc = tl.zeros((BM, BN), tl.float32)
                xbase = X + (lo + i0) * XS0 + h * XS1
                wbase = W + w * WS0 + (h * F + Q) * WS1
                for chunk in range(tl.cdiv(K_RUNTIME, BK)):
                    kr = chunk * BK + offs_k
                    fast_x = tl.load(
                        xbase + j[:, None] * XS0 + kr[None, :] * XS2
                    )
                    fast_w = tl.load(
                        wbase + offs_n[None, :] * WS1 + kr[:, None] * WS2,
                        eviction_policy="evict_last",
                    )
                    if X.dtype.element_ty == tl.float32:
                        fast_acc += tl.dot(
                            fast_x, fast_w, input_precision="ieee"
                        )
                    else:
                        fast_acc += tl.dot(fast_x, fast_w)
                fast_scale = tl.load(SCALES + w)
                fast_base = tl.load(fast_bp, eviction_policy="evict_first").to(
                    tl.float32
                )
                tl.store(
                    fast_op,
                    (fast_base + fast_acc * fast_scale).to(
                        OUT.dtype.element_ty
                    ),
                )
            else:
                inactive_base = tl.load(fast_bp, eviction_policy="evict_first")
                tl.store(fast_op, inactive_base)
            return
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
    if tl.load(RANKS + w) == 0:
        m2 = rmask[:, None] & nmask[None, :]
        op = OUT + rx[:, None] * GS0 + h * GS1 + offs_n[None, :] * GS2
        base = tl.load(
            BASE + rx[:, None] * BO0 + h * BO1 + offs_n[None, :] * BO2,
            mask=m2,
            other=0.0,
            eviction_policy="evict_first",
        )
        tl.store(op, base, mask=m2)
        return
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
        eviction_policy="evict_last",
    )
    scale = tl.where(tl.load(RANKS + w) > 0, tl.load(SCALES + w), 0.0)
    if X.dtype.element_ty == tl.float32:
        acc = tl.dot(x, wb, input_precision="ieee") * scale
    else:
        acc = tl.dot(x, wb) * scale
    m2 = rmask[:, None] & nmask[None, :]
    op = OUT + rx[:, None] * GS0 + h * GS1 + offs_n[None, :] * GS2
    base = tl.load(
        BASE + rx[:, None] * BO0 + h * BO1 + offs_n[None, :] * BO2,
        mask=m2,
        other=0.0,
        eviction_policy="evict_first",
    ).to(tl.float32)
    tl.store(op, (base + acc).to(OUT.dtype.element_ty), mask=m2)


def _large_step_b_v_segment(
    attn_lora_a, B_buf, batch_info, base_output, qk_nope_head_dim, v_head_dim
):
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
    bm = min(
        32 if K in (16, 32) else 128,
        max(16, 0 if ml == 0 else 1 << (ml - 1).bit_length()),
    )
    pl = perm.numel() if has_perm else S
    if has_perm and pl != S:
        out = base_output.clone(memory_format=torch.contiguous_format)
    else:
        out = base_output.new_empty((S + bm, H, N))
    if not has_perm:
        perm = attn_lora_a
    if K == 32:
        _step_b_v_kernel_segment_kloop.run(
            attn_lora_a,
            B_buf,
            base_output,
            out,
            batch_info.seg_indptr.contiguous(),
            batch_info.weight_indices.contiguous(),
            batch_info.lora_ranks.contiguous(),
            batch_info.scalings.contiguous(),
            perm,
            K,
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
            grid=(
                (H, (ml + bm - 1) // bm, batch_info.bs)
                if K in (16, 32)
                else ((ml + bm - 1) // bm, H, batch_info.bs)
            ),
            warmup=False,
        )
    else:
        _step_b_v_kernel_segment.run(
            attn_lora_a,
            B_buf,
            base_output,
            out,
            batch_info.seg_indptr.contiguous(),
            batch_info.weight_indices.contiguous(),
            batch_info.lora_ranks.contiguous(),
            batch_info.scalings.contiguous(),
            perm,
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
            grid=(
                (H, (ml + bm - 1) // bm, batch_info.bs)
                if K in (16, 32)
                else ((ml + bm - 1) // bm, H, batch_info.bs)
            ),
            warmup=False,
        )
    return out[:S]


@triton.jit
def _copy_base_kernel_segment(BASE, OUT, META: tl.constexpr):
    TOTAL: tl.constexpr = META[0]
    BLOCK: tl.constexpr = META[1]
    if TOTAL == 0:
        return
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    value = tl.load(BASE + i, i < TOTAL, other=0.0).to(tl.float32)
    tl.store(OUT + i, value, i < TOTAL)


def step_b_v_segment(
    attn_lora_a, B_buf, batch_info, base_output, qk_nope_head_dim, v_head_dim
):
    if batch_info.bs == 0 or base_output.numel() == 0:
        base = base_output.contiguous()
        out = torch.empty_like(base)
        total = base.numel()
        _copy_base_kernel_segment.run(
            base,
            out,
            (total, 1024),
            grid=(max(1, (total + 1024 - 1) // 1024),),
            warmup=False,
        )
        return out
    return _large_step_b_v_segment(
        attn_lora_a,
        B_buf,
        batch_info,
        base_output,
        qk_nope_head_dim,
        v_head_dim,
    )


def _optimized_step_b_v(
    attn_lora_a, B_buf, batch_info, base_output, qk_nope_head_dim, v_head_dim
):
    args = (
        attn_lora_a,
        B_buf,
        batch_info,
        base_output,
        qk_nope_head_dim,
        v_head_dim,
    )
    if (
        attn_lora_a.shape[-1] in (16, 32)
        and attn_lora_a.shape[0] >= 64
        and (batch_info.permutation is None)
    ):
        return step_b_v_segment(*args)
    return step_b_v_flat(*args)


@triton.jit
def _bounded_rows(X, W, OUT, SEG, WI, RANKS, SCALES, PERM, META: tl.constexpr):
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
        x,
        weight,
        out,
        batch_info.seg_indptr.contiguous(),
        batch_info.weight_indices.contiguous(),
        batch_info.lora_ranks.contiguous(),
        batch_info.scalings.contiguous(),
        p,
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
