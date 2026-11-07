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
def _step_b_q_kernel(
    X, A, B, OUT, SEG, WIDX, RANKS, SCALES, PERM, META: tl.constexpr
):
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
    offs_k = tl.arange(0, BK)
    kmask = offs_k < R
    pid = tl.program_id(0)
    m = pid // NT * BM + tl.arange(0, BM)
    if FLAT:
        keep = m < PM * H
        xoff = m[:, None] * XS1
        ooff = m[:, None] * OS1
        pos = m
    else:
        idx = m // H
        h = m % H
        keep = idx < PM
        if HAS_PERM:
            rows = tl.load(PERM + idx, mask=keep, other=0).to(tl.int32)
        else:
            rows = idx
        xoff = rows[:, None] * XS0 + h[:, None] * XS1
        ooff = rows[:, None] * OS0 + h[:, None] * OS1
        pos = idx
    offs_n = pid % NT * BN + tl.arange(0, BN)
    nmask = offs_n < N
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    rs = tl.zeros((BM,), dtype=tl.float32)
    for b in range(BS):
        if FLAT:
            lo = tl.load(SEG + b) * H
            hi = tl.load(SEG + b + 1) * H
        else:
            lo = tl.load(SEG + b)
            hi = tl.load(SEG + b + 1)
        if lo < hi:
            w = tl.load(WIDX + b)
            if tl.load(RANKS + w) > 0:
                rmask = keep & (pos >= lo) & (pos < hi)
                x = tl.load(
                    X + xoff + offs_k[None, :] * XS2,
                    mask=rmask[:, None] & kmask[None, :],
                    other=0.0,
                )
                a = tl.load(
                    A
                    + w * AS0
                    + offs_k[:, None] * AS1
                    + offs_n[None, :] * AS2,
                    mask=kmask[:, None] & nmask[None, :],
                    other=0.0,
                )
                partial = tl.dot(
                    x,
                    a,
                    input_precision=(
                        "ieee" if X.dtype.element_ty == tl.float32 else "tf32"
                    ),
                )
                acc = acc + tl.where(rmask[:, None], partial, 0.0)
                rs = tl.where(rmask, tl.load(SCALES + w), rs)
    off = ooff + offs_n[None, :] * OS2
    mask = keep[:, None] & nmask[None, :]
    base = tl.load(B + off, mask=mask, other=0.0).to(tl.float32)
    tl.store(
        OUT + off,
        (base + acc * rs[:, None]).to(OUT.dtype.element_ty),
        mask=mask,
    )


@triton.jit
def _step_b_q_aligned(
    X, A, B, OUT, SEG, WIDX, RANKS, SCALES, META: tl.constexpr
):
    H: tl.constexpr = META[0]
    BS: tl.constexpr = META[1]
    NT: tl.constexpr = META[2]
    XS1: tl.constexpr = META[5]
    XS2: tl.constexpr = META[6]
    AS0: tl.constexpr = META[7]
    AS1: tl.constexpr = META[8]
    AS2: tl.constexpr = META[9]
    OS1: tl.constexpr = META[10]
    OS2: tl.constexpr = META[11]
    BM: tl.constexpr = META[12]
    BN: tl.constexpr = META[13]
    BK: tl.constexpr = META[14]
    offs_k = tl.arange(0, BK)
    pid = tl.program_id(0)
    m0 = pid // NT * BM
    m = m0 + tl.arange(0, BM)
    offs_n = pid % NT * BN + tl.arange(0, BN)
    off = m[:, None] * OS1 + offs_n[None, :] * OS2
    w = 0
    act = False
    for b in range(BS):
        wb = tl.load(WIDX + b)
        hit = (
            (m0 >= tl.load(SEG + b) * H)
            & (m0 < tl.load(SEG + b + 1) * H)
            & (tl.load(RANKS + wb) > 0)
        )
        w = tl.where(hit, wb, w)
        act = act | hit
    if act:
        x = tl.load(X + m[:, None] * XS1 + offs_k[None, :] * XS2)
        a = tl.load(
            A + w * AS0 + offs_k[:, None] * AS1 + offs_n[None, :] * AS2
        )
        acc = tl.dot(
            x,
            a,
            input_precision=(
                "ieee" if X.dtype.element_ty == tl.float32 else "tf32"
            ),
        ) * tl.load(SCALES + w)
        base = tl.load(B + off).to(tl.float32)
        tl.store(OUT + off, (base + acc).to(OUT.dtype.element_ty))
    else:
        tl.store(OUT + off, tl.load(B + off))


@triton.jit
def _seggrid(X, A, B, OUT, SEG, WIDX, RANKS, SCALES, META: tl.constexpr):
    H: tl.constexpr = META[0]
    BS: tl.constexpr = META[1]
    NT: tl.constexpr = META[2]
    S: tl.constexpr = META[3]
    N: tl.constexpr = META[4]
    XS1: tl.constexpr = META[5]
    XS2: tl.constexpr = META[6]
    AS0: tl.constexpr = META[7]
    AS1: tl.constexpr = META[8]
    AS2: tl.constexpr = META[9]
    OS1: tl.constexpr = META[10]
    OS2: tl.constexpr = META[11]
    BM: tl.constexpr = META[12]
    BN: tl.constexpr = META[13]
    BK: tl.constexpr = META[14]
    offs_k = tl.arange(0, BK)
    p = tl.program_id(0)
    b = tl.program_id(1)
    if (b == 0) & (p == 0):
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
    x_base = X + lo * XS1
    hi = tl.load(SEG + b + 1) * H
    m0 = lo + p // NT * BM
    if m0 >= hi:
        return
    w = tl.load(WIDX + b)
    act = tl.load(RANKS + w) > 0
    m = m0 + tl.arange(0, BM)
    offs_n = p % NT * BN + tl.arange(0, BN)
    if not act:
        off = m[:, None] * OS1 + offs_n[None, :] * OS2
        tl.store(OUT + off, tl.load(B + off))
        return
    x = tl.load(
        x_base
        + (p // NT * BM + tl.arange(0, BM))[:, None] * XS1
        + offs_k[None, :] * XS2
    )
    a = tl.load(A + w * AS0 + offs_k[:, None] * AS1 + offs_n[None, :] * AS2)
    acc = tl.dot(
        x,
        a,
        input_precision="ieee" if X.dtype.element_ty == tl.float32 else "tf32",
    ) * tl.load(SCALES + w)
    off = m[:, None] * OS1 + offs_n[None, :] * OS2
    base = tl.load(B + off).to(tl.float32)
    tl.store(OUT + off, (base + acc).to(OUT.dtype.element_ty))


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


def step_b_q(q_lora_a, A_buf, batch_info, base_output):
    seg = batch_info.seg_indptr.contiguous()
    widx = batch_info.weight_indices.contiguous()
    ranks = batch_info.lora_ranks.contiguous()
    scales = batch_info.scalings.contiguous()
    perm = batch_info.permutation
    if perm is not None:
        perm = perm.contiguous()
    if not q_lora_a.is_contiguous():
        q_lora_a = q_lora_a.contiguous()
    if not A_buf.is_contiguous():
        A_buf = A_buf.contiguous()
    if not base_output.is_contiguous():
        base_output = base_output.contiguous()
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
                seg if perm is None else perm,
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
    xs = q_lora_a.stride()
    os = base_output.stride()
    bk = max(16, 0 if R == 0 else 1 << (R - 1).bit_length())
    bs = batch_info.bs
    flat = perm is None and xs[0] == H * xs[1] and (os[0] == H * os[1])
    ml = batch_info.max_len
    if ml is not None and ml < 128:
        bm, bn = (32, 256)
    else:
        bm, bn = (64, 128)
    if (
        bs > 0
        and flat
        and (H % bm == 0)
        and (N % bn == 0)
        and (N & N - 1 == 0)
        and (bk == R)
        and (xs == (H * R, R, 1))
        and (os == (H * N, N, 1))
        and (A_buf.stride() == (R * N, N, 1))
        and (ml is not None)
    ):
        out = torch.empty_like(base_output)
        nt = N // bn
        maxt = max(1, (ml * H + bm - 1) // bm)
        if bm == 64 and H % (2 * bm) == 0:
            maxt = max(1, (ml * H + bm * 8 - 1) // (bm * 8))
            _reuse_seggrid.run(
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
                    nt,
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
                    8,
                    maxt,
                ),
                num_warps=8,
                num_stages=2,
                grid=(maxt * nt, bs),
                warmup=False,
            )
        else:
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
                    nt,
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
                num_warps=8,
                num_stages=2,
                use_sme=(
                    3
                    if q_lora_a.dtype in (torch.float16, torch.bfloat16)
                    else 0
                ),
                grid=(maxt * nt, bs),
                warmup=False,
            )
        return out
    bm, bn = (16, 64)
    has_perm = perm is not None
    if has_perm:
        pm = perm.numel()
        out = base_output.clone() if pm != S else torch.empty_like(base_output)
    else:
        pm = S
        out = torch.empty_like(base_output)
        perm = base_output
    nt = (N + bn - 1) // bn
    total = (pm * H + bm - 1) // bm * nt
    np = total
    _step_b_q_kernel.run(
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
            batch_info.bs,
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
        ),
        num_warps=4,
        num_stages=2,
        grid=(np,),
        warmup=False,
    )
    return out


@triton.jit
def _reuse_seggrid(X, A, B, OUT, SEG, WIDX, RANKS, SCALES, META: tl.constexpr):
    H: tl.constexpr = META[0]
    BS: tl.constexpr = META[1]
    NT: tl.constexpr = META[2]
    S: tl.constexpr = META[3]
    N: tl.constexpr = META[4]
    XS1: tl.constexpr = META[5]
    XS2: tl.constexpr = META[6]
    AS0: tl.constexpr = META[7]
    AS1: tl.constexpr = META[8]
    AS2: tl.constexpr = META[9]
    OS1: tl.constexpr = META[10]
    OS2: tl.constexpr = META[11]
    BM: tl.constexpr = META[12]
    BN: tl.constexpr = META[13]
    BK: tl.constexpr = META[14]
    ML: tl.constexpr = META[15]
    MT: tl.constexpr = META[16]
    offs_k = tl.arange(0, BK)
    raw_p = tl.program_id(0)
    group = raw_p // (8 * NT)
    first_m = group * 8
    group_m = tl.minimum(MT - first_m, 8)
    within = raw_p % (8 * NT)
    p = (first_m + within % group_m) * NT + within // group_m
    b = tl.program_id(1)
    if (b == 0) & (p == 0):
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
    m0 = lo + p // NT * BM * ML
    if m0 >= hi:
        return
    w = tl.load(WIDX + b)
    act = tl.load(RANKS + w) > 0
    offs_n = p % NT * BN + tl.arange(0, BN)
    if not act:
        for i in range(tl.minimum(ML, (hi - m0) // BM)):
            m = m0 + i * BM + tl.arange(0, BM)
            off = m[:, None] * OS1 + offs_n[None, :] * OS2
            tl.store(OUT + off, tl.load(B + off))
        return
    a = tl.load(A + w * AS0 + offs_k[:, None] * AS1 + offs_n[None, :] * AS2)
    for i in range(tl.minimum(ML, (hi - m0) // BM)):
        m = m0 + i * BM + tl.arange(0, BM)
        x = tl.load(X + m[:, None] * XS1 + offs_k[None, :] * XS2)
        acc = tl.dot(
            x,
            a,
            input_precision=(
                "ieee" if X.dtype.element_ty == tl.float32 else "tf32"
            ),
        ) * tl.load(SCALES + w)
        off = m[:, None] * OS1 + offs_n[None, :] * OS2
        base = tl.load(B + off).to(tl.float32)
        tl.store(OUT + off, (base + acc).to(OUT.dtype.element_ty))


__all__ = ["step_b_q"]
