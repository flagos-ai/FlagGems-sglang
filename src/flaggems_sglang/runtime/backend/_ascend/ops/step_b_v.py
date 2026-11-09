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
def _step_b_v_kernel(
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
    if tl.load(RANKS + w) == 0:
        m2 = rmask[:, None] & nmask[None, :]
        op = OUT + rx[:, None] * GS0 + h * GS1 + offs_n[None, :] * GS2
        base = tl.load(
            BASE + rx[:, None] * BO0 + h * BO1 + offs_n[None, :] * BO2,
            mask=m2,
            other=0.0,
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
    )
    scale = tl.where(tl.load(RANKS + w) > 0, tl.load(SCALES + w), 0.0)
    acc = tl.dot(x, wb) * scale
    m2 = rmask[:, None] & nmask[None, :]
    op = OUT + rx[:, None] * GS0 + h * GS1 + offs_n[None, :] * GS2
    base = tl.load(
        BASE + rx[:, None] * BO0 + h * BO1 + offs_n[None, :] * BO2,
        mask=m2,
        other=0.0,
    ).to(tl.float32)
    tl.store(op, (base + acc).to(OUT.dtype.element_ty), mask=m2)


def _large_step_b_v(
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
    bm = min(128, max(16, 0 if ml == 0 else 1 << (ml - 1).bit_length()))
    pl = perm.numel() if has_perm else S
    if has_perm and pl != S:
        out = base_output.clone(memory_format=torch.contiguous_format)
    else:
        out = base_output.new_empty((S + bm, H, N))
    if not has_perm:
        perm = None
    _step_b_v_kernel.run(
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
        grid=((ml + bm - 1) // bm, H, batch_info.bs),
        warmup=False,
    )
    return out[:S]


@triton.jit
def _batched_heads(
    X, W, BASE, OUT, SEG, WIDX, RANKS, SCALES, META: tl.constexpr
):
    S: tl.constexpr = META[0]
    H: tl.constexpr = META[1]
    K: tl.constexpr = META[2]
    N: tl.constexpr = META[3]
    Q: tl.constexpr = META[4]
    F: tl.constexpr = META[5]
    BS: tl.constexpr = META[6]
    GH: tl.constexpr = META[7]
    BM: tl.constexpr = META[8]
    BN: tl.constexpr = META[9]
    BK: tl.constexpr = META[10]
    XS0: tl.constexpr = META[11]
    XS1: tl.constexpr = META[12]
    XS2: tl.constexpr = META[13]
    WS0: tl.constexpr = META[14]
    WS1: tl.constexpr = META[15]
    WS2: tl.constexpr = META[16]
    BO0: tl.constexpr = META[17]
    BO1: tl.constexpr = META[18]
    BO2: tl.constexpr = META[19]
    tile = tl.program_id(0)
    h0 = tl.program_id(1) * GH
    b = tl.program_id(2)
    lo = tl.load(SEG + b)
    hi = tl.load(SEG + b + 1)
    if tile == 0 and b == 0:
        lim = tl.load(SEG + BS)
        c = tl.arange(0, GH * BN)
        ch = h0 + c // BN
        cn = c % BN
        prefix = tl.load(SEG)
        for prow in range(0, prefix):
            pv = tl.load(
                BASE + prow * BO0 + ch * BO1 + cn * BO2, cn < N, other=0.0
            )
            tl.store(OUT + (prow * H + ch) * N + cn, pv, cn < N)
        for row in range(lim, S):
            value = tl.load(
                BASE + row * BO0 + ch * BO1 + cn * BO2, cn < N, other=0.0
            )
            tl.store(OUT + (row * H + ch) * N + cn, value, cn < N)
    if lo == hi or tile * BM >= hi - lo:
        return
    w = tl.load(WIDX + b)
    h = h0 + tl.arange(0, GH)
    m = lo + tile * BM + tl.arange(0, BM)
    k = tl.arange(0, BK)
    n = tl.arange(0, BN)
    if tl.load(RANKS + w) == 0:
        bp = (
            BASE
            + m[:, None, None] * BO0
            + h[None, :, None] * BO1
            + n[None, None, :] * BO2
        )
        op = (
            OUT
            + (m[:, None, None] * H + h[None, :, None]) * N
            + n[None, None, :]
        )
        mask = (m < hi)[:, None, None] & (n < N)[None, None, :]
        base = tl.load(bp, mask, other=0.0)
        tl.store(op, base, mask)
        return
    x = tl.load(
        X
        + h[:, None, None] * XS1
        + m[None, :, None] * XS0
        + k[None, None, :] * XS2,
        (m < hi)[None, :, None] & (k < K)[None, None, :],
        other=0.0,
    )
    wb = tl.load(
        W
        + w * WS0
        + (h[:, None, None] * F + Q + n[None, None, :]) * WS1
        + k[None, :, None] * WS2,
        (k < K)[None, :, None] & (n < N)[None, None, :],
        other=0.0,
    )
    acc = tl.dot(x, wb)
    acc = tl.trans(acc, (1, 0, 2))
    scale = tl.where(tl.load(RANKS + w) > 0, tl.load(SCALES + w), 0.0)
    bp = (
        BASE
        + m[:, None, None] * BO0
        + h[None, :, None] * BO1
        + n[None, None, :] * BO2
    )
    op = OUT + (m[:, None, None] * H + h[None, :, None]) * N + n[None, None, :]
    mask = (m < hi)[:, None, None] & (n < N)[None, None, :]
    base = tl.load(bp, mask, other=0.0).to(tl.float32)
    tl.store(op, base + acc * scale, mask)


@triton.jit
def _copy_base_kernel(BASE, OUT, META: tl.constexpr):
    TOTAL: tl.constexpr = META[0]
    BLOCK: tl.constexpr = META[1]
    if TOTAL == 0:
        return
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    value = tl.load(BASE + i, i < TOTAL, other=0.0).to(tl.float32)
    tl.store(OUT + i, value, i < TOTAL)


def _previous_step_b_v(
    attn_lora_a, B_buf, batch_info, base_output, qk_nope_head_dim, v_head_dim
):
    if batch_info.bs == 0 or base_output.numel() == 0:
        base = base_output.contiguous()
        out = torch.empty_like(base)
        total = base.numel()
        _copy_base_kernel.run(
            base,
            out,
            (total, 1024),
            grid=(max(1, (total + 1024 - 1) // 1024),),
            warmup=False,
        )
        return out
    if attn_lora_a.stride(-1) != 1:
        attn_lora_a = attn_lora_a.contiguous()
    S, H, K = attn_lora_a.shape
    if S < 64 or K != 32 or H % 2 != 0 or (batch_info.permutation is not None):
        return _large_step_b_v(
            attn_lora_a,
            B_buf,
            batch_info,
            base_output,
            qk_nope_head_dim,
            v_head_dim,
        )
    N = v_head_dim
    out = base_output.new_empty((S, H, N))
    ml = batch_info.max_len or S
    bm = 64
    _batched_heads.run(
        attn_lora_a,
        B_buf,
        base_output,
        out,
        batch_info.seg_indptr.contiguous(),
        batch_info.weight_indices.contiguous(),
        batch_info.lora_ranks.contiguous(),
        batch_info.scalings.contiguous(),
        (
            S,
            H,
            K,
            N,
            qk_nope_head_dim,
            qk_nope_head_dim + N,
            batch_info.bs,
            2,
            bm,
            max(16, 0 if N == 0 else 1 << (N - 1).bit_length()),
            max(16, 0 if K == 0 else 1 << (K - 1).bit_length()),
            *attn_lora_a.stride(),
            *B_buf.stride(),
            *base_output.stride(),
        ),
        grid=((ml + bm - 1) // bm, H // 2, batch_info.bs),
        warmup=False,
    )
    return out


@triton.jit
def _reuse_kernel(
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
    w = tl.load(WIDX + b)
    offs_n = tl.arange(0, BN)
    offs_k = tl.arange(0, BK)
    nmask = offs_n < N
    kmask = offs_k < K
    j = tl.arange(0, BM)
    wb = tl.load(
        W
        + w * WS0
        + (h * F + Q + offs_n[None, :]) * WS1
        + offs_k[:, None] * WS2,
        mask=kmask[:, None] & nmask[None, :],
        other=0.0,
    )
    scale = tl.where(tl.load(RANKS + w) > 0, tl.load(SCALES + w), 0.0)
    for step in range(2):
        i0 = pid_s * (BM * 2) + step * BM
        if i0 < seg_len:
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
                preserved = tl.load(
                    BASE + rx[:, None] * BO0 + h * BO1 + offs_n[None, :] * BO2,
                    mask=m2,
                    other=0.0,
                )
                tl.store(op, preserved, mask=m2)
            else:
                x = tl.load(
                    X + rx[:, None] * XS0 + h * XS1 + offs_k[None, :] * XS2,
                    mask=rmask[:, None] & kmask[None, :],
                    other=0.0,
                )
                acc = tl.dot(x, wb) * scale
                m2 = rmask[:, None] & nmask[None, :]
                op = OUT + rx[:, None] * GS0 + h * GS1 + offs_n[None, :] * GS2
                base = tl.load(
                    BASE + rx[:, None] * BO0 + h * BO1 + offs_n[None, :] * BO2,
                    mask=m2,
                    other=0.0,
                ).to(tl.float32)
                tl.store(op, (base + acc).to(OUT.dtype.element_ty), mask=m2)


def _current_step_b_v(
    attn_lora_a, B_buf, batch_info, base_output, qk_nope_head_dim, v_head_dim
):
    if attn_lora_a.stride(-1) != 1:
        attn_lora_a = attn_lora_a.contiguous()
    S, H, K = attn_lora_a.shape
    N = v_head_dim
    ml = batch_info.max_len or S
    if (
        batch_info.bs == 0
        or K != 16
        or S < 256
        or (ml < 128)
        or (batch_info.permutation is not None)
    ):
        return _previous_step_b_v(
            attn_lora_a,
            B_buf,
            batch_info,
            base_output,
            qk_nope_head_dim,
            v_head_dim,
        )
    bm = 128
    bn = max(16, 0 if N == 0 else 1 << (N - 1).bit_length())
    bk = max(16, 0 if K == 0 else 1 << (K - 1).bit_length())
    out = base_output.new_empty((S + bm, H, N))
    _reuse_kernel.run(
        attn_lora_a,
        B_buf,
        base_output,
        out,
        batch_info.seg_indptr.contiguous(),
        batch_info.weight_indices.contiguous(),
        batch_info.lora_ranks.contiguous(),
        batch_info.scalings.contiguous(),
        attn_lora_a,
        (
            H,
            N,
            K,
            qk_nope_head_dim + N,
            qk_nope_head_dim,
            S,
            S,
            batch_info.bs,
            False,
            False,
            *attn_lora_a.stride(),
            *B_buf.stride(),
            *out.stride(),
            *base_output.stride(),
            bm,
            bn,
            bk,
            S - 1,
        ),
        grid=((ml + bm * 2 - 1) // (bm * 2), H, batch_info.bs),
        warmup=False,
    )
    return out[:S]


@triton.jit
def _adapter_heads(
    X, W, BASE, OUT, SEG, WIDX, RANKS, SCALES, META: tl.constexpr
):
    S: tl.constexpr = META[0]
    H: tl.constexpr = META[1]
    K: tl.constexpr = META[2]
    N: tl.constexpr = META[3]
    Q: tl.constexpr = META[4]
    F: tl.constexpr = META[5]
    BS: tl.constexpr = META[6]
    GH: tl.constexpr = META[7]
    BM: tl.constexpr = META[8]
    BN: tl.constexpr = META[9]
    BK: tl.constexpr = META[10]
    ML: tl.constexpr = META[11]
    XS0: tl.constexpr = META[12]
    XS1: tl.constexpr = META[13]
    XS2: tl.constexpr = META[14]
    WS0: tl.constexpr = META[15]
    WS1: tl.constexpr = META[16]
    WS2: tl.constexpr = META[17]
    BO0: tl.constexpr = META[18]
    BO1: tl.constexpr = META[19]
    BO2: tl.constexpr = META[20]
    tile = tl.program_id(0)
    h0 = tl.program_id(1) * GH
    w = tl.program_id(2)
    if tile == 0 and w == 0:
        lim = tl.load(SEG + BS)
        prefix = tl.load(SEG)
        c = tl.arange(0, GH * BN)
        ch = h0 + c // BN
        cn = c % BN
        for row in range(0, prefix):
            preserved = tl.load(
                BASE + row * BO0 + ch * BO1 + cn * BO2, cn < N, other=0.0
            )
            tl.store(OUT + (row * H + ch) * N + cn, preserved, cn < N)
        for row in range(lim, S):
            preserved = tl.load(
                BASE + row * BO0 + ch * BO1 + cn * BO2, cn < N, other=0.0
            )
            tl.store(OUT + (row * H + ch) * N + cn, preserved, cn < N)
    h = h0 + tl.arange(0, GH)
    k = tl.arange(0, BK)
    n = tl.arange(0, BN)
    wb = tl.load(
        W
        + w * WS0
        + (h[:, None, None] * F + Q + n[None, None, :]) * WS1
        + k[None, :, None] * WS2,
        (k < K)[None, :, None] & (n < N)[None, None, :],
        other=0.0,
    )
    active = tl.load(RANKS + w) != 0
    scale = tl.load(SCALES + w)
    for b in range(BS):
        if tl.load(WIDX + b) == w:
            lo = tl.load(SEG + b)
            hi = tl.load(SEG + b + 1)
            for step in range(ML):
                offset = (tile * ML + step) * BM
                if offset < hi - lo:
                    m = lo + offset + tl.arange(0, BM)
                    bp = (
                        BASE
                        + m[:, None, None] * BO0
                        + h[None, :, None] * BO1
                        + n[None, None, :] * BO2
                    )
                    op = (
                        OUT
                        + (m[:, None, None] * H + h[None, :, None]) * N
                        + n[None, None, :]
                    )
                    mask = (m < hi)[:, None, None] & (n < N)[None, None, :]
                    if active:
                        x = tl.load(
                            X
                            + h[:, None, None] * XS1
                            + m[None, :, None] * XS0
                            + k[None, None, :] * XS2,
                            (m < hi)[None, :, None] & (k < K)[None, None, :],
                            other=0.0,
                        )
                        acc = tl.trans(tl.dot(x, wb), (1, 0, 2))
                        base_value = tl.load(bp, mask, other=0.0).to(
                            tl.float32
                        )
                        tl.store(op, base_value + acc * scale, mask)
                    else:
                        preserved_value = tl.load(bp, mask, other=0.0)
                        tl.store(op, preserved_value, mask)


def _k32_adapter_step_b_v(
    attn_lora_a, B_buf, batch_info, base_output, qk_nope_head_dim, v_head_dim
):
    S, H, K = attn_lora_a.shape
    na = B_buf.shape[0]
    if (
        batch_info.bs == 0
        or na > batch_info.bs
        or S < 64
        or (K != 32)
        or (H % 2 != 0)
        or (batch_info.permutation is not None)
    ):
        return _current_step_b_v(
            attn_lora_a,
            B_buf,
            batch_info,
            base_output,
            qk_nope_head_dim,
            v_head_dim,
        )
    if attn_lora_a.stride(-1) != 1:
        attn_lora_a = attn_lora_a.contiguous()
    N = v_head_dim
    out = base_output.new_empty((S, H, N))
    ml = batch_info.max_len or S
    bm = 64
    _adapter_heads.run(
        attn_lora_a,
        B_buf,
        base_output,
        out,
        batch_info.seg_indptr.contiguous(),
        batch_info.weight_indices.contiguous(),
        batch_info.lora_ranks.contiguous(),
        batch_info.scalings.contiguous(),
        (
            S,
            H,
            K,
            N,
            qk_nope_head_dim,
            qk_nope_head_dim + N,
            batch_info.bs,
            2,
            bm,
            max(16, 0 if N == 0 else 1 << (N - 1).bit_length()),
            max(16, 0 if K == 0 else 1 << (K - 1).bit_length()),
            1,
            *attn_lora_a.stride(),
            *B_buf.stride(),
            *base_output.stride(),
        ),
        grid=((ml + bm - 1) // bm, H // 2, na),
        warmup=False,
    )
    return out


@triton.jit
def _adapter_reuse_kernel(
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
    pid_s = tl.program_id(0)
    h = tl.program_id(1)
    w = tl.program_id(2)
    if (w == 0) & (pid_s == 0):
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
    offs_n = tl.arange(0, BN)
    offs_k = tl.arange(0, BK)
    nmask = offs_n < N
    kmask = offs_k < K
    j = tl.arange(0, BM)
    wb = tl.load(
        W
        + w * WS0
        + (h * F + Q + offs_n[None, :]) * WS1
        + offs_k[:, None] * WS2,
        mask=kmask[:, None] & nmask[None, :],
        other=0.0,
    )
    scale = tl.where(tl.load(RANKS + w) > 0, tl.load(SCALES + w), 0.0)
    for b in range(BS):
        if tl.load(WIDX + b) == w:
            lo = tl.load(SEG + b)
            hi = tl.load(SEG + b + 1)
            seg_len = hi - lo
            for step in range(2):
                i0 = pid_s * (BM * 2) + step * BM
                if i0 < seg_len:
                    if HAS_PERM:
                        idx = lo + i0 + j
                        rows = tl.load(PERM + tl.minimum(idx, PL - 1)).to(
                            tl.int32
                        )
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
                        op = (
                            OUT
                            + rx[:, None] * GS0
                            + h * GS1
                            + offs_n[None, :] * GS2
                        )
                        preserved = tl.load(
                            BASE
                            + rx[:, None] * BO0
                            + h * BO1
                            + offs_n[None, :] * BO2,
                            mask=m2,
                            other=0.0,
                        )
                        tl.store(op, preserved, mask=m2)
                    else:
                        x = tl.load(
                            X
                            + rx[:, None] * XS0
                            + h * XS1
                            + offs_k[None, :] * XS2,
                            mask=rmask[:, None] & kmask[None, :],
                            other=0.0,
                        )
                        acc = tl.dot(x, wb) * scale
                        m2 = rmask[:, None] & nmask[None, :]
                        op = (
                            OUT
                            + rx[:, None] * GS0
                            + h * GS1
                            + offs_n[None, :] * GS2
                        )
                        base = tl.load(
                            BASE
                            + rx[:, None] * BO0
                            + h * BO1
                            + offs_n[None, :] * BO2,
                            mask=m2,
                            other=0.0,
                        ).to(tl.float32)
                        tl.store(
                            op, (base + acc).to(OUT.dtype.element_ty), mask=m2
                        )


def _optimized_step_b_v(
    attn_lora_a, B_buf, batch_info, base_output, qk_nope_head_dim, v_head_dim
):
    if B_buf.shape[0] == 0:
        base = base_output.contiguous()
        out = torch.empty_like(base)
        total = base.numel()
        _copy_base_kernel.run(
            base,
            out,
            (total, 1024),
            grid=(max(1, (total + 1024 - 1) // 1024),),
            warmup=False,
        )
        return out
    S, H, K = attn_lora_a.shape
    ml = batch_info.max_len or S
    na = B_buf.shape[0]
    if (
        batch_info.bs == 0
        or na > batch_info.bs
        or K != 16
        or (S < 256)
        or (ml < 128)
        or (batch_info.permutation is not None)
    ):
        return _k32_adapter_step_b_v(
            attn_lora_a,
            B_buf,
            batch_info,
            base_output,
            qk_nope_head_dim,
            v_head_dim,
        )
    if attn_lora_a.stride(-1) != 1:
        attn_lora_a = attn_lora_a.contiguous()
    N = v_head_dim
    bm = 128
    bn = max(16, 0 if N == 0 else 1 << (N - 1).bit_length())
    bk = max(16, 0 if K == 0 else 1 << (K - 1).bit_length())
    out = base_output.new_empty((S + bm, H, N))
    _adapter_reuse_kernel.run(
        attn_lora_a,
        B_buf,
        base_output,
        out,
        batch_info.seg_indptr.contiguous(),
        batch_info.weight_indices.contiguous(),
        batch_info.lora_ranks.contiguous(),
        batch_info.scalings.contiguous(),
        attn_lora_a,
        (
            H,
            N,
            K,
            qk_nope_head_dim + N,
            qk_nope_head_dim,
            S,
            S,
            batch_info.bs,
            False,
            False,
            *attn_lora_a.stride(),
            *B_buf.stride(),
            *out.stride(),
            *base_output.stride(),
            bm,
            bn,
            bk,
            S - 1,
        ),
        grid=((ml + bm * 2 - 1) // (bm * 2), H, na),
        warmup=False,
    )
    return out[:S]


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


_control_step_b_v = step_b_v


@triton.jit
def _pair_select_k(
    X, W, BASE, OUT, SEG, WI, RANKS, SCALES, META: tl.constexpr
):
    S: tl.constexpr = META[0]
    H: tl.constexpr = META[1]
    N: tl.constexpr = META[2]
    K: tl.constexpr = META[3]
    F: tl.constexpr = META[4]
    Q: tl.constexpr = META[5]
    BS: tl.constexpr = META[6]
    NA: tl.constexpr = META[7]
    BM: tl.constexpr = META[8]
    GRID: tl.constexpr = META[9]
    BALANCED: tl.constexpr = META[10]
    pid = tl.program_id(0)
    if pid == 0:
        prefix = tl.load(SEG) * H * N
        tail = tl.load(SEG + BS) * H * N
        flat = tl.arange(0, 1024)
        for block in range(tl.cdiv(prefix, 1024)):
            off = block * 1024 + flat
            value = tl.load(BASE + off, off < prefix, other=0.0)
            tl.store(OUT + off, value, off < prefix)
        for block in range(tl.cdiv(S * H * N - tail, 1024)):
            off = tail + block * 1024 + flat
            value = tl.load(BASE + off, off < S * H * N, other=0.0)
            tl.store(OUT + off, value, off < S * H * N)
    total = NA * (H // 2)
    if BALANCED:
        first = pid * total // GRID
        limit = (pid + 1) * total // GRID
        stride = 1
    else:
        first = pid
        limit = total
        stride = GRID
    for job in range(first, limit, stride):
        w = job // (H // 2)
        h = job % (H // 2) * 2
        ns = tl.arange(0, 2 * N)
        if tl.load(RANKS + w) == 0:
            for b in range(BS):
                if tl.load(WI + b) == w:
                    lo = tl.load(SEG + b)
                    hi = tl.load(SEG + b + 1)
                    for row in range(lo, hi):
                        off = (row * H + h) * N + ns
                        value = tl.load(BASE + off)
                        tl.store(OUT + off, value)
        else:
            nn = tl.arange(0, N)
            kk = tl.arange(0, K)
            w0 = tl.load(
                W
                + w * (H * F * K)
                + (h * F + Q + nn[None, :]) * K
                + kk[:, None]
            )
            w1 = tl.load(
                W
                + w * (H * F * K)
                + ((h + 1) * F + Q + nn[None, :]) * K
                + kk[:, None]
            )
            scale = tl.load(SCALES + w).to(tl.float32)
            for b in range(BS):
                if tl.load(WI + b) == w:
                    lo = tl.load(SEG + b)
                    hi = tl.load(SEG + b + 1)
                    for tile in range(tl.cdiv(hi - lo, BM)):
                        start = lo + tile * BM
                        row0 = tl.minimum(start, S - BM)
                        rows = row0 + tl.arange(0, BM)
                        keep = (rows >= start) & (rows < hi)
                        x0 = tl.load(
                            X + (rows[:, None] * H + h) * K + kk[None, :]
                        )
                        x1 = tl.load(
                            X + (rows[:, None] * H + h + 1) * K + kk[None, :]
                        )
                        acc0 = tl.dot(x0, w0) * scale
                        acc1 = tl.dot(x1, w1) * scale
                        off0 = (rows[:, None] * H + h) * N + nn[None, :]
                        off1 = (rows[:, None] * H + h + 1) * N + nn[None, :]
                        base0 = tl.load(
                            BASE + off0, keep[:, None], other=0.0
                        ).to(tl.float32)
                        base1 = tl.load(
                            BASE + off1, keep[:, None], other=0.0
                        ).to(tl.float32)
                        tl.store(OUT + off0, base0 + acc0, keep[:, None])
                        tl.store(OUT + off1, base1 + acc1, keep[:, None])


def step_b_v_pair(
    attn_lora_a, B_buf, batch_info, base_output, qk_nope_head_dim, v_head_dim
):
    S, H, K = attn_lora_a.shape
    N = v_head_dim
    na = B_buf.shape[0]
    ml = batch_info.max_len or S
    if (
        K not in (16, 32)
        or H % 2
        or S < 256
        or (ml > 256)
        or (N != 128)
        or (na == 0)
        or (na > batch_info.bs)
        or (batch_info.permutation is not None)
        or (attn_lora_a.dtype != torch.bfloat16)
        or (base_output.numel() == 0)
    ):
        return _control_step_b_v(
            attn_lora_a,
            B_buf,
            batch_info,
            base_output,
            qk_nope_head_dim,
            v_head_dim,
        )
    x = attn_lora_a.contiguous()
    raw = B_buf.contiguous()
    base = base_output.contiguous()
    out = torch.empty_like(base)
    if K == 32:
        _pair_select_k_prefetch.run(
            x,
            raw,
            base,
            out,
            batch_info.seg_indptr.contiguous(),
            batch_info.weight_indices.contiguous(),
            batch_info.lora_ranks.contiguous(),
            batch_info.scalings.contiguous(),
            (
                S,
                H,
                N,
                K,
                qk_nope_head_dim + N,
                qk_nope_head_dim,
                batch_info.bs,
                na,
                64,
            ),
            num_stages=2,
            grid=(na * (H // 2) + 1,),
            warmup=False,
        )
    else:
        grid = min(20, na * (H // 2))
        _pair_select_k.run(
            x,
            raw,
            base,
            out,
            batch_info.seg_indptr.contiguous(),
            batch_info.weight_indices.contiguous(),
            batch_info.lora_ranks.contiguous(),
            batch_info.scalings.contiguous(),
            (
                S,
                H,
                N,
                K,
                qk_nope_head_dim + N,
                qk_nope_head_dim,
                batch_info.bs,
                na,
                64,
                grid,
                False,
            ),
            num_stages=1,
            grid=(grid,),
            warmup=False,
        )
    return out


def step_b_v(
    attn_lora_a, B_buf, batch_info, base_output, qk_nope_head_dim, v_head_dim
):
    return step_b_v_pair(
        attn_lora_a,
        B_buf,
        batch_info,
        base_output,
        qk_nope_head_dim,
        v_head_dim,
    )


@triton.jit
def _pair_select_k_prefetch(
    X, W, BASE, OUT, SEG, WI, RANKS, SCALES, META: tl.constexpr
):
    S: tl.constexpr = META[0]
    H: tl.constexpr = META[1]
    N: tl.constexpr = META[2]
    K: tl.constexpr = META[3]
    F: tl.constexpr = META[4]
    Q: tl.constexpr = META[5]
    BS: tl.constexpr = META[6]
    NA: tl.constexpr = META[7]
    BM: tl.constexpr = META[8]
    pid = tl.program_id(0)
    if pid == NA * (H // 2):
        prefix = tl.load(SEG) * H * N
        tail = tl.load(SEG + BS) * H * N
        flat = tl.arange(0, 1024)
        for block in range(tl.cdiv(prefix, 1024)):
            off = block * 1024 + flat
            value = tl.load(BASE + off, off < prefix, other=0.0)
            tl.store(OUT + off, value, off < prefix)
        for block in range(tl.cdiv(S * H * N - tail, 1024)):
            off = tail + block * 1024 + flat
            value = tl.load(BASE + off, off < S * H * N, other=0.0)
            tl.store(OUT + off, value, off < S * H * N)
        return
    w = pid // (H // 2)
    h = pid % (H // 2) * 2
    ns = tl.arange(0, 2 * N)
    if tl.load(RANKS + w) == 0:
        for b in range(BS):
            if tl.load(WI + b) == w:
                lo = tl.load(SEG + b)
                hi = tl.load(SEG + b + 1)
                for row in range(lo, hi):
                    off = (row * H + h) * N + ns
                    value = tl.load(BASE + off)
                    tl.store(OUT + off, value)
        return
    nn = tl.arange(0, N)
    kk = tl.arange(0, K)
    w0 = tl.load(
        W + w * (H * F * K) + (h * F + Q + nn[None, :]) * K + kk[:, None]
    )
    w1 = tl.load(
        W + w * (H * F * K) + ((h + 1) * F + Q + nn[None, :]) * K + kk[:, None]
    )
    scale = tl.load(SCALES + w).to(tl.float32)
    for b in range(BS):
        if tl.load(WI + b) == w:
            lo = tl.load(SEG + b)
            hi = tl.load(SEG + b + 1)
            for tile in range(tl.cdiv(hi - lo, BM)):
                start = lo + tile * BM
                row0 = tl.minimum(start, S - BM)
                rows = row0 + tl.arange(0, BM)
                keep = (rows >= start) & (rows < hi)
                off0 = (rows[:, None] * H + h) * N + nn[None, :]
                off1 = (rows[:, None] * H + h + 1) * N + nn[None, :]
                base0 = tl.load(BASE + off0, keep[:, None], other=0.0)
                base1 = tl.load(BASE + off1, keep[:, None], other=0.0)
                x0 = tl.load(X + (rows[:, None] * H + h) * K + kk[None, :])
                x1 = tl.load(X + (rows[:, None] * H + h + 1) * K + kk[None, :])
                acc0 = tl.dot(x0, w0) * scale
                acc1 = tl.dot(x1, w1) * scale
                tl.store(
                    OUT + off0, base0.to(tl.float32) + acc0, keep[:, None]
                )
                tl.store(
                    OUT + off1, base1.to(tl.float32) + acc1, keep[:, None]
                )


__all__ = ["step_b_v"]
