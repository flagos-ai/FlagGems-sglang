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

import math

import torch
import triton
import triton.language as tl


@triton.jit
def _gen_lean(
    X, A, B, OUT, SEG, WIDX, RANKS, SCALES, PERM, META: tl.constexpr
):
    H: tl.constexpr = META[0]
    R: tl.constexpr = META[1]
    N: tl.constexpr = META[2]
    PM: tl.constexpr = META[3]
    BS: tl.constexpr = META[4]
    NT: tl.constexpr = META[5]
    BM: tl.constexpr = META[6]
    BN: tl.constexpr = META[7]
    BK: tl.constexpr = META[8]
    FLAT: tl.constexpr = META[9]
    HAS_PERM: tl.constexpr = META[10]
    MEXACT: tl.constexpr = META[11]
    offs_k = tl.arange(0, BK)
    pid = tl.program_id(0)
    m = pid // NT * BM + tl.arange(0, BM)
    if FLAT:
        pos = m
        ok = m < PM * H
        if MEXACT:
            xoff = m[:, None] * R
            ooff = m[:, None] * N
        else:
            ms = tl.minimum(m, PM * H - 1)
            xoff = ms[:, None] * R
            ooff = ms[:, None] * N
    else:
        idx = m // H
        h = m % H
        pos = idx
        ok = idx < PM
        if HAS_PERM:
            rows = tl.load(PERM + idx, mask=idx < PM, other=0).to(tl.int32)
        else:
            rows = tl.minimum(idx, PM - 1)
        xoff = rows[:, None] * (H * R) + h[:, None] * R
        ooff = rows[:, None] * (H * N) + h[:, None] * N
    offs_n = pid % NT * BN + tl.arange(0, BN)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for b in tl.static_range(BS):
        if FLAT:
            lo = tl.load(SEG + b) * H
            hi = tl.load(SEG + b + 1) * H
        else:
            lo = tl.load(SEG + b)
            hi = tl.load(SEG + b + 1)
        if lo < hi:
            w = tl.load(WIDX + b)
            if tl.load(RANKS + w) > 0:
                if MEXACT:
                    rmask = (pos >= lo) & (pos < hi)
                else:
                    rmask = (pos >= lo) & (pos < hi) & ok
                x = tl.load(
                    X + xoff + offs_k[None, :],
                    mask=rmask[:, None] & (offs_k[None, :] < R),
                    other=0.0,
                )
                a = tl.load(
                    A + w * (R * N) + offs_k[:, None] * N + offs_n[None, :]
                )
                partial = tl.dot(x, a) * tl.load(SCALES + w)
                acc = acc + tl.where(rmask[:, None], partial, 0.0)
    off = ooff + offs_n[None, :]
    if MEXACT:
        base = tl.load(B + off).to(tl.float32)
        tl.store(OUT + off, (base + acc).to(OUT.dtype.element_ty))
    else:
        base = tl.load(B + off, mask=ok[:, None], other=0.0).to(tl.float32)
        tl.store(
            OUT + off, (base + acc).to(OUT.dtype.element_ty), mask=ok[:, None]
        )


@triton.jit
def _permrow_lean(
    X, A, B, OUT, SEG, WIDX, RANKS, SCALES, PERM, META: tl.constexpr
):
    H: tl.constexpr = META[0]
    R: tl.constexpr = META[1]
    N: tl.constexpr = META[2]
    BS: tl.constexpr = META[3]
    BSP: tl.constexpr = META[4]
    BK: tl.constexpr = META[5]
    offs_k = tl.arange(0, BK)
    offs_n = tl.arange(0, N)
    h = tl.arange(0, H)
    bidx = tl.arange(0, BSP)
    bvalid = bidx < BS
    hi_all = tl.load(SEG + 1 + bidx, mask=bvalid, other=0)
    p = tl.program_id(0)
    row = tl.load(PERM + p).to(tl.int32)
    b0 = tl.minimum(tl.sum(((hi_all <= p) & bvalid).to(tl.int32), 0), BS - 1)
    w = tl.load(WIDX + b0)
    live = (
        (p >= tl.load(SEG + b0))
        & (p < tl.load(SEG + b0 + 1))
        & (tl.load(RANKS + w) > 0)
    )
    off = row * (H * N) + h[:, None] * N + offs_n[None, :]
    if not live:
        tl.store(OUT + off, tl.load(B + off))
        return
    x = tl.load(
        X + row * (H * R) + h[:, None] * R + offs_k[None, :],
        mask=(h[:, None] < H) & (offs_k[None, :] < R),
        other=0.0,
    )
    a = tl.load(A + w * (R * N) + offs_k[:, None] * N + offs_n[None, :])
    acc = tl.dot(x, a) * tl.load(SCALES + w)
    base = tl.load(B + off).to(tl.float32)
    tl.store(OUT + off, (base + acc).to(OUT.dtype.element_ty))


@triton.jit
def _gen(X, A, B, OUT, SEG, WIDX, RANKS, SCALES, PERM, META: tl.constexpr):
    H: tl.constexpr = META[0]
    R: tl.constexpr = META[1]
    N: tl.constexpr = META[2]
    PM: tl.constexpr = META[3]
    BS: tl.constexpr = META[4]
    NT: tl.constexpr = META[5]
    XS0: tl.constexpr = META[8]
    XS1: tl.constexpr = META[9]
    XS2: tl.constexpr = META[10]
    AS0: tl.constexpr = META[11]
    AS1: tl.constexpr = META[12]
    AS2: tl.constexpr = META[13]
    OS0: tl.constexpr = META[14]
    OS1: tl.constexpr = META[15]
    OS2: tl.constexpr = META[16]
    BM: tl.constexpr = META[17]
    BN: tl.constexpr = META[18]
    BK: tl.constexpr = META[19]
    FLAT: tl.constexpr = META[20]
    HAS_PERM: tl.constexpr = META[21]
    MEXACT: tl.constexpr = META[22]
    offs_k = tl.arange(0, BK)
    pid = tl.program_id(0)
    m = pid // NT * BM + tl.arange(0, BM)
    if FLAT:
        pos = m
        ok = m < PM * H
        if MEXACT:
            xoff = m[:, None] * XS1
            ooff = m[:, None] * OS1
        else:
            ms = tl.minimum(m, PM * H - 1)
            xoff = ms[:, None] * XS1
            ooff = ms[:, None] * OS1
    else:
        idx = m // H
        h = m % H
        pos = idx
        ok = idx < PM
        if HAS_PERM:
            rows = tl.load(PERM + idx, mask=idx < PM, other=0).to(tl.int32)
        else:
            rows = tl.minimum(idx, PM - 1)
        xoff = rows[:, None] * XS0 + h[:, None] * XS1
        ooff = rows[:, None] * OS0 + h[:, None] * OS1
    offs_n = pid % NT * BN + tl.arange(0, BN)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for b in tl.static_range(BS):
        if FLAT:
            lo = tl.load(SEG + b) * H
            hi = tl.load(SEG + b + 1) * H
        else:
            lo = tl.load(SEG + b)
            hi = tl.load(SEG + b + 1)
        if lo < hi:
            w = tl.load(WIDX + b)
            if tl.load(RANKS + w) > 0:
                if MEXACT:
                    rmask = (pos >= lo) & (pos < hi)
                else:
                    rmask = (pos >= lo) & (pos < hi) & ok
                x = tl.load(
                    X + xoff + offs_k[None, :] * XS2,
                    mask=rmask[:, None] & (offs_k[None, :] < R),
                    other=0.0,
                )
                a = tl.load(
                    A
                    + w * AS0
                    + offs_k[:, None] * AS1
                    + offs_n[None, :] * AS2,
                    mask=(offs_k[:, None] < R) & (offs_n[None, :] < N),
                    other=0.0,
                )
                partial = tl.dot(x, a) * tl.load(SCALES + w)
                acc = acc + tl.where(rmask[:, None], partial, 0.0)
    off = ooff + offs_n[None, :] * OS2
    if MEXACT and N % BN == 0:
        base = tl.load(B + off).to(tl.float32)
        tl.store(OUT + off, (base + acc).to(OUT.dtype.element_ty))
    else:
        valid = ok[:, None] & (offs_n[None, :] < N)
        base = tl.load(B + off, mask=valid, other=0.0).to(tl.float32)
        tl.store(OUT + off, (base + acc).to(OUT.dtype.element_ty), mask=valid)


@triton.jit
def _permrow(X, A, B, OUT, SEG, WIDX, RANKS, SCALES, PERM, META: tl.constexpr):
    H: tl.constexpr = META[0]
    R: tl.constexpr = META[1]
    N: tl.constexpr = META[2]
    BS: tl.constexpr = META[3]
    BSP: tl.constexpr = META[4]
    XS0: tl.constexpr = META[7]
    XS1: tl.constexpr = META[8]
    XS2: tl.constexpr = META[9]
    AS0: tl.constexpr = META[10]
    AS1: tl.constexpr = META[11]
    AS2: tl.constexpr = META[12]
    OS0: tl.constexpr = META[13]
    OS1: tl.constexpr = META[14]
    OS2: tl.constexpr = META[15]
    BK: tl.constexpr = META[16]
    offs_k = tl.arange(0, BK)
    offs_n = tl.arange(0, N)
    h = tl.arange(0, H)
    bidx = tl.arange(0, BSP)
    bvalid = bidx < BS
    hi_all = tl.load(SEG + 1 + bidx, mask=bvalid, other=0)
    p = tl.program_id(0)
    row = tl.load(PERM + p).to(tl.int32)
    b0 = tl.minimum(tl.sum(((hi_all <= p) & bvalid).to(tl.int32), 0), BS - 1)
    w = tl.load(WIDX + b0)
    live = (
        (p >= tl.load(SEG + b0))
        & (p < tl.load(SEG + b0 + 1))
        & (tl.load(RANKS + w) > 0)
    )
    off = row * OS0 + h[:, None] * OS1 + offs_n[None, :] * OS2
    if not live:
        tl.store(OUT + off, tl.load(B + off))
        return
    x = tl.load(
        X + row * XS0 + h[:, None] * XS1 + offs_k[None, :] * XS2,
        mask=(h[:, None] < H) & (offs_k[None, :] < R),
        other=0.0,
    )
    a = tl.load(A + w * AS0 + offs_k[:, None] * AS1 + offs_n[None, :] * AS2)
    acc = tl.dot(x, a) * tl.load(SCALES + w)
    base = tl.load(B + off).to(tl.float32)
    tl.store(OUT + off, (base + acc).to(OUT.dtype.element_ty))


@triton.jit
def _aligned(X, A, B, OUT, SEG, WIDX, RANKS, SCALES, META: tl.constexpr):
    H: tl.constexpr = META[0]
    BS: tl.constexpr = META[1]
    BSP: tl.constexpr = META[2]
    NT: tl.constexpr = META[3]
    XS1: tl.constexpr = META[6]
    XS2: tl.constexpr = META[7]
    AS0: tl.constexpr = META[8]
    AS1: tl.constexpr = META[9]
    AS2: tl.constexpr = META[10]
    OS1: tl.constexpr = META[11]
    OS2: tl.constexpr = META[12]
    BM: tl.constexpr = META[13]
    BN: tl.constexpr = META[14]
    BK: tl.constexpr = META[15]
    offs_k = tl.arange(0, BK)
    bidx = tl.arange(0, BSP)
    bvalid = bidx < BS
    hi_all = tl.load(SEG + 1 + bidx, mask=bvalid, other=0) * H
    pid = tl.program_id(0)
    m0 = pid // NT * BM
    m = m0 + tl.arange(0, BM)
    offs_n = pid % NT * BN + tl.arange(0, BN)
    b0 = tl.minimum(tl.sum(((hi_all <= m0) & bvalid).to(tl.int32), 0), BS - 1)
    w = tl.load(WIDX + b0)
    live = (
        (m0 >= tl.load(SEG + b0) * H)
        & (m0 < tl.load(SEG + b0 + 1) * H)
        & (tl.load(RANKS + w) > 0)
    )
    if not live:
        off = m[:, None] * OS1 + offs_n[None, :] * OS2
        tl.store(OUT + off, tl.load(B + off))
        return
    x = tl.load(X + m[:, None] * XS1 + offs_k[None, :] * XS2)
    a = tl.load(A + w * AS0 + offs_k[:, None] * AS1 + offs_n[None, :] * AS2)
    acc = tl.dot(x, a) * tl.load(SCALES + w)
    off = m[:, None] * OS1 + offs_n[None, :] * OS2
    base = tl.load(B + off).to(tl.float32)
    tl.store(OUT + off, (base + acc).to(OUT.dtype.element_ty))


@triton.jit
def _seggrid(X, A, B, OUT, SEG, WIDX, RANKS, SCALES, META: tl.constexpr):
    H: tl.constexpr = META[0]
    BS: tl.constexpr = META[1]
    S: tl.constexpr = META[2]
    N: tl.constexpr = META[3]
    XS1: tl.constexpr = META[4]
    XS2: tl.constexpr = META[5]
    AS0: tl.constexpr = META[6]
    AS1: tl.constexpr = META[7]
    AS2: tl.constexpr = META[8]
    OS1: tl.constexpr = META[9]
    OS2: tl.constexpr = META[10]
    BM: tl.constexpr = META[11]
    BN: tl.constexpr = META[12]
    BK: tl.constexpr = META[13]
    offs_k = tl.arange(0, BK)
    pid = tl.program_id(0)
    b = tl.program_id(1)
    if (b == 0) & (pid == 0):
        first = tl.load(SEG)
        if first > 0:
            for r in range(first * H):
                off = r * N + tl.arange(0, N)
                tl.store(OUT + off, tl.load(B + off))
        lim = tl.load(SEG + BS)
        if lim < S:
            for r in range(lim * H, S * H):
                off = r * N + tl.arange(0, N)
                tl.store(OUT + off, tl.load(B + off))
    lo = tl.load(SEG + b) * H
    hi = tl.load(SEG + b + 1) * H
    m0 = lo + pid * BM
    if m0 >= hi:
        return
    w = tl.load(WIDX + b)
    act = tl.load(RANKS + w) > 0
    m = m0 + tl.arange(0, BM)
    offs_n = tl.arange(0, BN)
    live = m < hi
    if not act:
        off = m[:, None] * OS1 + offs_n[None, :] * OS2
        tl.store(
            OUT + off,
            tl.load(B + off, mask=live[:, None], other=0.0),
            mask=live[:, None],
        )
        return
    x = tl.load(
        X + m[:, None] * XS1 + offs_k[None, :] * XS2,
        mask=live[:, None],
        other=0.0,
    )
    a = tl.load(A + w * AS0 + offs_k[:, None] * AS1 + offs_n[None, :] * AS2)
    acc = tl.dot(x, a) * tl.load(SCALES + w)
    off = m[:, None] * OS1 + offs_n[None, :] * OS2
    base = tl.load(B + off, mask=live[:, None], other=0.0).to(tl.float32)
    tl.store(
        OUT + off, (base + acc).to(OUT.dtype.element_ty), mask=live[:, None]
    )


def _base(
    q_lora_a, A_buf, batch_info, base_output, seg, widx, ranks, scales, perm
):
    if not q_lora_a.is_contiguous():
        q_lora_a = q_lora_a.contiguous()
    if not A_buf.is_contiguous():
        A_buf = A_buf.contiguous()
    if not base_output.is_contiguous():
        base_output = base_output.contiguous()
    S, H, R = q_lora_a.shape
    N = A_buf.shape[2]
    xs = q_lora_a.stride()
    os = base_output.stride()
    bs = batch_info.bs
    bsp = max(2, 0 if bs == 0 else 1 << (bs - 1).bit_length())
    bk = max(16, 0 if R == 0 else 1 << (R - 1).bit_length())
    flat = perm is None and xs[0] == H * xs[1] and (os[0] == H * os[1])
    ml = batch_info.max_len
    if (
        bs > 0
        and flat
        and (N == (0 if N == 0 else 1 << (N - 1).bit_length()))
        and (N <= 512)
        and (bk == R)
        and (xs == (H * R, R, 1))
        and (os == (H * N, N, 1))
        and (A_buf.stride() == (R * N, N, 1))
        and (ml is not None)
    ):
        bm = 64 if H >= 64 else 16
        bn = N
        out = torch.empty_like(base_output)
        maxt = max(1, (ml * H + bm - 1) // bm)
        _seggrid.run(
            q_lora_a,
            A_buf,
            base_output,
            out,
            seg,
            widx,
            ranks,
            scales,
            (
                H,
                bs,
                S,
                N,
                xs[1],
                xs[2],
                *A_buf.stride(),
                os[1],
                os[2],
                bm,
                bn,
                bk,
            ),
            num_warps=4,
            num_stages=2,
            grid=(maxt, bs),
            warmup=False,
        )
        return out
    contig = (
        xs == (H * R, R, 1)
        and os == (H * N, N, 1)
        and (A_buf.stride() == (R * N, N, 1))
    )
    if (
        contig
        and perm is None
        and (H == (0 if H == 0 else 1 << (H - 1).bit_length()))
        and (bk == R)
        and (N == (0 if N == 0 else 1 << (N - 1).bit_length()))
    ):
        bm, bn = (16, N)
        out = torch.empty_like(base_output)
        nt = N // bn
        total = (S * H + bm - 1) // bm * nt
        _gen_lean.run(
            q_lora_a,
            A_buf,
            base_output,
            out,
            seg,
            widx,
            ranks,
            scales,
            None,
            (H, R, N, S, bs, nt, bm, bn, bk, True, False, S * H % bm == 0),
            num_warps=4,
            num_stages=2,
            grid=(total,),
            warmup=False,
        )
        return out
    if (
        bs > 0
        and contig
        and (perm is not None)
        and (perm.numel() == S)
        and (H >= 16)
        and (H == (0 if H == 0 else 1 << (H - 1).bit_length()))
        and (bk == R)
        and (N == (0 if N == 0 else 1 << (N - 1).bit_length()))
        and (H <= 16)
    ):
        out = torch.empty_like(base_output)
        _permrow_lean.run(
            q_lora_a,
            A_buf,
            base_output,
            out,
            seg,
            widx,
            ranks,
            scales,
            perm,
            (H, R, N, bs, bsp, bk),
            num_warps=4,
            num_stages=2,
            grid=(S,),
            warmup=False,
        )
        return out
    if (
        bs > 0
        and perm is not None
        and (perm.numel() == S)
        and (H >= 16)
        and (H == (0 if H == 0 else 1 << (H - 1).bit_length()))
        and (bk == R)
        and (os[0] == H * os[1])
        and (xs[0] == H * xs[1])
        and (N == (0 if N == 0 else 1 << (N - 1).bit_length()))
        and (H <= 16)
    ):
        out = torch.empty_like(base_output)
        total = S
        np = total
        _permrow.run(
            q_lora_a,
            A_buf,
            base_output,
            out,
            seg,
            widx,
            ranks,
            scales,
            perm,
            (H, R, N, bs, bsp, total, np, *xs, *A_buf.stride(), *os, bk),
            num_warps=4,
            num_stages=2,
            grid=(np,),
            warmup=False,
        )
        return out
    bm = 16
    bn = N if N == (0 if N == 0 else 1 << (N - 1).bit_length()) else 64
    has_perm = perm is not None
    if has_perm:
        pm = perm.numel()
        out = base_output.clone() if pm != S else torch.empty_like(base_output)
    else:
        pm = S
        out = torch.empty_like(base_output)
    nt = (N + bn - 1) // bn
    total = (pm * H + bm - 1) // bm * nt
    np = total
    _gen.run(
        q_lora_a,
        A_buf,
        base_output,
        out,
        seg,
        widx,
        ranks,
        scales,
        perm,
        (
            H,
            R,
            N,
            pm,
            bs,
            nt,
            total,
            np,
            *xs,
            *A_buf.stride(),
            *os,
            bm,
            bn,
            bk,
            flat,
            has_perm,
            pm * H % bm == 0,
        ),
        num_warps=4,
        num_stages=2,
        grid=(np,),
        warmup=False,
    )
    return out


@triton.jit
def _reuse(X, A, B, OUT, SEG, WIDX, RANKS, SCALES, META: tl.constexpr):
    H: tl.constexpr = META[0]
    R: tl.constexpr = META[1]
    N: tl.constexpr = META[2]
    S: tl.constexpr = META[3]
    BS: tl.constexpr = META[4]
    BM: tl.constexpr = META[5]
    ML: tl.constexpr = META[6]
    MT: tl.constexpr = META[7]
    p = tl.program_id(0)
    b = tl.program_id(1)
    if (b == 0) & (p == 0):
        first = tl.load(SEG)
        for row in range(first * H):
            j = row * N + tl.arange(0, N)
            tl.store(OUT + j, tl.load(B + j))
        lim = tl.load(SEG + BS)
        for row in range(lim * H, S * H):
            j = row * N + tl.arange(0, N)
            tl.store(OUT + j, tl.load(B + j))
    lo = tl.load(SEG + b) * H
    hi = tl.load(SEG + b + 1) * H
    panels = (hi - lo) // BM
    start_panel = p * panels // MT
    end_panel = (p + 1) * panels // MT
    m0 = lo + start_panel * BM
    hi = lo + end_panel * BM
    if m0 >= hi:
        return
    w = tl.load(WIDX + b)
    active = tl.load(RANKS + w) > 0
    scale = tl.load(SCALES + w)
    k = tl.arange(0, R)
    n = tl.arange(0, N)
    if not active:
        for i in range(ML):
            if m0 + i * BM < hi:
                m = m0 + i * BM + tl.arange(0, BM)
                off = m[:, None] * N + n[None, :]
                tl.store(OUT + off, tl.load(B + off))
        return
    a = tl.load(A + w * (R * N) + k[:, None] * N + n[None, :])
    for i in range(ML):
        if m0 + i * BM < hi:
            m = m0 + i * BM + tl.arange(0, BM)
            off = m[:, None] * N + n[None, :]
            base = tl.load(B + off)
            x = tl.load(X + m[:, None] * R + k[None, :])
            acc = tl.dot(x, a) * scale
            tl.store(
                OUT + off, (base.to(tl.float32) + acc).to(OUT.dtype.element_ty)
            )


@triton.jit
def _copy_base(B, OUT, META: tl.constexpr):
    TOTAL: tl.constexpr = META[0]
    BLOCK: tl.constexpr = META[1]
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(OUT + i, tl.load(B + i, mask=i < TOTAL, other=0), mask=i < TOTAL)


@triton.jit
def _bounded_head(
    X, A, B, OUT, SEG, WIDX, RANKS, SCALES, PERM, META: tl.constexpr
):
    H: tl.constexpr = META[0]
    R: tl.constexpr = META[1]
    N: tl.constexpr = META[2]
    PM: tl.constexpr = META[3]
    HAS_PERM: tl.constexpr = META[4]
    BM: tl.constexpr = META[5]
    BN: tl.constexpr = META[6]
    BK: tl.constexpr = META[7]
    b = tl.program_id(1)
    lo = tl.load(SEG + b)
    hi = tl.load(SEG + b + 1)
    p = lo + tl.program_id(0) // tl.cdiv(H, BM)
    if p >= hi or p >= PM:
        return
    w = tl.load(WIDX + b)
    if tl.load(RANKS + w) == 0:
        return
    if HAS_PERM:
        row = tl.load(PERM + p)
    else:
        row = p
    h = tl.program_id(0) % tl.cdiv(H, BM) * BM + tl.arange(0, BM)
    n = tl.program_id(2) * BN + tl.arange(0, BN)
    ks = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), tl.float32)
    for chunk in range(tl.cdiv(R, BK)):
        k = chunk * BK + ks
        x = tl.load(
            X + row * H * R + h[:, None] * R + k[None, :],
            mask=(h[:, None] < H) & (k[None, :] < R),
            other=0.0,
        )
        a = tl.load(
            A + w * R * N + k[:, None] * N + n[None, :],
            mask=(k[:, None] < R) & (n[None, :] < N),
            other=0.0,
        )
        acc = tl.dot(x, a, acc, input_precision="ieee")
    off = row * H * N + h[:, None] * N + n[None, :]
    mask = (h[:, None] < H) & (n[None, :] < N)
    base = tl.load(B + off, mask=mask, other=0.0).to(tl.float32)
    value = base + acc * tl.load(SCALES + w)
    tl.store(OUT + off, value.to(OUT.dtype.element_ty), mask=mask)


def _core_entry(q_lora_a, A_buf, batch_info, base_output):
    seg = batch_info.seg_indptr.contiguous()
    widx = batch_info.weight_indices.contiguous()
    ranks = batch_info.lora_ranks.contiguous()
    scales = batch_info.scalings.contiguous()
    perm = batch_info.permutation
    if perm is not None:
        perm = perm.contiguous()
    S, H, R = q_lora_a.shape
    adapter_count, _, N = A_buf.shape
    if S * H * N == 0:
        return torch.empty_like(base_output)
    if (
        R == 0
        or batch_info.bs == 0
        or adapter_count == 0
        or (perm is not None and perm.numel() == 0)
    ):
        base_output = base_output.contiguous()
        out = torch.empty_like(base_output)
        _copy_base.run(
            base_output,
            out,
            (S * H * N, 1024),
            grid=((S * H * N + 1024 - 1) // 1024,),
            warmup=False,
        )
        return out
    if R > 128:
        q_lora_a = q_lora_a.contiguous()
        A_buf = A_buf.contiguous()
        base_output = base_output.contiguous()
        out = torch.empty_like(base_output)
        _copy_base.run(
            base_output,
            out,
            (S * H * N, 1024),
            grid=((S * H * N + 1024 - 1) // 1024,),
            warmup=False,
        )
        pm = S if perm is None else perm.numel()
        longest = batch_info.max_len or S
        if pm:
            _bounded_head.run(
                q_lora_a,
                A_buf,
                base_output,
                out,
                seg,
                widx,
                ranks,
                scales,
                perm,
                (H, R, N, pm, perm is not None, 32, 64, 64),
                num_warps=4,
                num_stages=1,
                grid=(
                    longest * ((H + 32 - 1) // 32),
                    batch_info.bs,
                    (N + 64 - 1) // 64,
                ),
                warmup=False,
            )
        return out
    bm = 64
    if (
        batch_info.bs > 0
        and perm is None
        and (batch_info.max_len is not None)
        and (H % bm == 0)
        and (R >= 16)
        and (R == (0 if R == 0 else 1 << (R - 1).bit_length()))
        and (N == (0 if N == 0 else 1 << (N - 1).bit_length()))
        and (N <= 512)
        and q_lora_a.is_contiguous()
        and A_buf.is_contiguous()
        and base_output.is_contiguous()
    ):
        panels = max(1, (batch_info.max_len * H + bm - 1) // bm)
        cores = triton.runtime.driver.active.utils.get_device_properties(
            q_lora_a.device.index
        )["num_aicore"]
        target_tiles = cores // math.gcd(cores, batch_info.bs)
        ml = max(1, (panels + target_tiles - 1) // target_tiles)
        out = torch.empty_like(base_output)
        _reuse.run(
            q_lora_a,
            A_buf,
            base_output,
            out,
            seg,
            widx,
            ranks,
            scales,
            (H, R, N, S, batch_info.bs, bm, ml, target_tiles),
            num_warps=4,
            num_stages=2,
            grid=(target_tiles, batch_info.bs),
            warmup=False,
        )
        return out
    return _base(
        q_lora_a,
        A_buf,
        batch_info,
        base_output,
        seg,
        widx,
        ranks,
        scales,
        perm,
    )


@triton.jit
def _small_row_dot(
    X, A, B, OUT, SEG, WIDX, RANKS, SCALES, PERM, META: tl.constexpr
):
    H: tl.constexpr = META[0]
    R: tl.constexpr = META[1]
    N: tl.constexpr = META[2]
    BS: tl.constexpr = META[3]
    HAS_PERM: tl.constexpr = META[4]
    BH: tl.constexpr = META[5]
    BK: tl.constexpr = META[6]
    BN: tl.constexpr = META[7]
    logical = tl.program_id(0)
    row = tl.load(PERM + logical).to(tl.int32) if HAS_PERM else logical
    head = tl.arange(0, BH)
    n = tl.program_id(1) * BN + tl.arange(0, BN)
    offset = (row * H + head[:, None]) * N + n[None, :]
    live = (head[:, None] < H) & (n[None, :] < N)
    w = 0
    active = False
    for segment in range(BS):
        start = tl.load(SEG + segment)
        end = tl.load(SEG + segment + 1)
        if start < end:
            adapter = tl.load(WIDX + segment)
            hit = (
                (logical >= start)
                & (logical < end)
                & (tl.load(RANKS + adapter) > 0)
            )
            w = tl.where(hit, adapter, w)
            active = active | hit
    if active:
        k = tl.arange(0, BK)
        x = tl.load(
            X + (row * H + head[:, None]) * R + k[None, :],
            mask=(head[:, None] < H) & (k[None, :] < R),
            other=0,
        )
        a = tl.load(
            A + w * R * N + k[:, None] * N + n[None, :],
            mask=(k[:, None] < R) & (n[None, :] < N),
            other=0,
        )
        update = tl.dot(
            x,
            a,
            input_precision=(
                "ieee" if X.dtype.element_ty == tl.float32 else "tf32"
            ),
        ) * tl.load(SCALES + w)
        base = tl.load(B + offset, mask=live, other=0).to(tl.float32)
        tl.store(
            OUT + offset, (base + update).to(OUT.dtype.element_ty), mask=live
        )
    else:
        tl.store(
            OUT + offset, tl.load(B + offset, mask=live, other=0), mask=live
        )


def step_b_q(q_lora_a, A_buf, batch_info, base_output):
    S, H, R = q_lora_a.shape
    N = A_buf.shape[2]
    perm = batch_info.permutation
    if (
        0 < S * H <= 256
        and 0 < H <= 16
        and (0 < R <= 32)
        and (N > 0)
        and (A_buf.shape[0] > 0)
        and (batch_info.bs > 0)
        and (perm is None or perm.numel() == S)
        and q_lora_a.is_contiguous()
        and A_buf.is_contiguous()
        and base_output.is_contiguous()
        and batch_info.seg_indptr.is_contiguous()
        and batch_info.weight_indices.is_contiguous()
        and batch_info.lora_ranks.is_contiguous()
        and batch_info.scalings.is_contiguous()
        and (perm is None or perm.is_contiguous())
    ):
        out = torch.empty_like(base_output)
        _small_row_dot.run(
            q_lora_a,
            A_buf,
            base_output,
            out,
            batch_info.seg_indptr,
            batch_info.weight_indices,
            batch_info.lora_ranks,
            batch_info.scalings,
            perm,
            (
                H,
                R,
                N,
                batch_info.bs,
                perm is not None,
                16,
                max(16, 0 if R == 0 else 1 << (R - 1).bit_length()),
                64,
            ),
            num_warps=4,
            num_stages=2,
            grid=(S, (N + 64 - 1) // 64),
            warmup=False,
        )
        return out
    return _core_entry(q_lora_a, A_buf, batch_info, base_output)


__all__ = ["step_b_q"]
