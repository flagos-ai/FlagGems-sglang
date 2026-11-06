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
    XS0: tl.constexpr = META[6]
    XS1: tl.constexpr = META[7]
    XS2: tl.constexpr = META[8]
    AS0: tl.constexpr = META[9]
    AS1: tl.constexpr = META[10]
    AS2: tl.constexpr = META[11]
    OS0: tl.constexpr = META[12]
    OS1: tl.constexpr = META[13]
    OS2: tl.constexpr = META[14]
    BM: tl.constexpr = META[15]
    BN: tl.constexpr = META[16]
    BK: tl.constexpr = META[17]
    FLAT: tl.constexpr = META[18]
    HAS_PERM: tl.constexpr = META[19]
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
                partial = tl.dot(x, a)
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
    XS1: tl.constexpr = META[3]
    XS2: tl.constexpr = META[4]
    AS0: tl.constexpr = META[5]
    AS1: tl.constexpr = META[6]
    AS2: tl.constexpr = META[7]
    OS1: tl.constexpr = META[8]
    OS2: tl.constexpr = META[9]
    BM: tl.constexpr = META[10]
    BN: tl.constexpr = META[11]
    BK: tl.constexpr = META[12]
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
        xp = tl.make_block_ptr(
            base=X + m0 * XS1,
            shape=(BM, BK),
            strides=(XS1, XS2),
            offsets=(0, 0),
            block_shape=(BM, BK),
            order=(1, 0) if XS2 <= XS1 else (0, 1),
        )
        ap = tl.make_block_ptr(
            base=A + w * AS0 + pid % NT * BN * AS2,
            shape=(BK, BN),
            strides=(AS1, AS2),
            offsets=(0, 0),
            block_shape=(BK, BN),
            order=(1, 0) if AS2 <= AS1 else (0, 1),
        )
        x = tl.load(xp)
        a = tl.load(ap)
        acc = tl.dot(x, a) * tl.load(SCALES + w)
        base = tl.load(B + off, cache_modifier=".cg").to(tl.float32)
        tl.store(OUT + off, (base + acc).to(OUT.dtype.element_ty))
    else:
        tl.store(OUT + off, tl.load(B + off))


@triton.jit
def _copy_base(B, OUT, TOTAL: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(OUT + i, tl.load(B + i, mask=i < TOTAL, other=0), mask=i < TOTAL)


@triton.jit
def _large_rank_segment(
    X,
    A,
    B,
    O,
    SEG,
    WIDX,
    RANKS,
    SCALES,
    PERM,
    H: tl.constexpr,
    R: tl.constexpr,
    N: tl.constexpr,
    PM: tl.constexpr,
    HAS_PERM: tl.constexpr,
    XS0: tl.constexpr,
    XS1: tl.constexpr,
    XS2: tl.constexpr,
    AS0: tl.constexpr,
    AS1: tl.constexpr,
    AS2: tl.constexpr,
    OS0: tl.constexpr,
    OS1: tl.constexpr,
    OS2: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    INPUT_PRECISION: tl.constexpr,
):
    b = tl.program_id(1)
    lo = tl.load(SEG + b)
    hi = tl.load(SEG + b + 1)
    if lo < hi:
        w = tl.load(WIDX + b)
        if tl.load(RANKS + w) > 0:
            m = lo * H + tl.program_id(0) * BM + tl.arange(0, BM)
            pos, heads = (m // H, m % H)
            keep = (pos < hi) & (pos < PM)
            if HAS_PERM:
                rows = tl.load(PERM + pos, mask=keep, other=0).to(tl.int32)
            else:
                rows = pos
            ns = tl.program_id(2) * BN + tl.arange(0, BN)
            ks = tl.arange(0, BK)
            acc = tl.zeros((BM, BN), tl.float32)
            for chunk in range(tl.cdiv(R, BK)):
                k = chunk * BK + ks
                x = tl.load(
                    X
                    + rows[:, None] * XS0
                    + heads[:, None] * XS1
                    + k[None, :] * XS2,
                    mask=keep[:, None] & (k[None, :] < R),
                    other=0,
                )
                a = tl.load(
                    A + w * AS0 + k[:, None] * AS1 + ns[None, :] * AS2,
                    mask=(k[:, None] < R) & (ns[None, :] < N),
                    other=0,
                )
                acc = tl.dot(x, a, acc, input_precision=INPUT_PRECISION)
            off = (
                rows[:, None] * OS0 + heads[:, None] * OS1 + ns[None, :] * OS2
            )
            mask = keep[:, None] & (ns[None, :] < N)
            base = tl.load(B + off, mask=mask, other=0).to(tl.float32)
            result = base + acc * tl.load(SCALES + w)
            tl.store(O + off, result.to(O.dtype.element_ty), mask=mask)


@triton.jit
def _rank_segment_fused(
    X,
    A,
    B,
    O,
    SEG,
    WIDX,
    RANKS,
    SCALES,
    PERM,
    H: tl.constexpr,
    R: tl.constexpr,
    N: tl.constexpr,
    PM: tl.constexpr,
    BS: tl.constexpr,
    HAS_PERM: tl.constexpr,
    XS0: tl.constexpr,
    XS1: tl.constexpr,
    XS2: tl.constexpr,
    AS0: tl.constexpr,
    AS1: tl.constexpr,
    AS2: tl.constexpr,
    OS0: tl.constexpr,
    OS1: tl.constexpr,
    OS2: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    INPUT_PRECISION: tl.constexpr,
):
    b = tl.program_id(1)
    ns = tl.program_id(2) * BN + tl.arange(0, BN)
    if b == BS:
        first = tl.load(SEG)
        last = tl.load(SEG + BS)
        if first > 0 or last < PM:
            for begin in range(
                tl.program_id(0) * BM, PM * H, tl.num_programs(0) * BM
            ):
                m = begin + tl.arange(0, BM)
                pos, heads = (m // H, m % H)
                keep = (pos < PM) & ((pos < first) | (pos >= last))
                if HAS_PERM:
                    rows = tl.load(PERM + pos, mask=keep, other=0).to(tl.int32)
                else:
                    rows = pos
                off = (
                    rows[:, None] * OS0
                    + heads[:, None] * OS1
                    + ns[None, :] * OS2
                )
                mask = keep[:, None] & (ns[None, :] < N)
                tl.store(
                    O + off, tl.load(B + off, mask=mask, other=0), mask=mask
                )
    else:
        lo = tl.load(SEG + b)
        hi = tl.load(SEG + b + 1)
        m0 = lo * H + tl.program_id(0) * BM
        if m0 < hi * H:
            w = tl.load(WIDX + b)
            m = m0 + tl.arange(0, BM)
            pos, heads = (m // H, m % H)
            keep = (pos < hi) & (pos < PM)
            if HAS_PERM:
                rows = tl.load(PERM + pos, mask=keep, other=0).to(tl.int32)
            else:
                rows = pos
            off = (
                rows[:, None] * OS0 + heads[:, None] * OS1 + ns[None, :] * OS2
            )
            mask = keep[:, None] & (ns[None, :] < N)
            if tl.load(RANKS + w) > 0:
                ks = tl.arange(0, BK)
                acc = tl.zeros((BM, BN), tl.float32)
                for chunk in range(tl.cdiv(R, BK)):
                    k = chunk * BK + ks
                    x = tl.load(
                        X
                        + rows[:, None] * XS0
                        + heads[:, None] * XS1
                        + k[None, :] * XS2,
                        mask=keep[:, None] & (k[None, :] < R),
                        other=0,
                    )
                    a = tl.load(
                        A + w * AS0 + k[:, None] * AS1 + ns[None, :] * AS2,
                        mask=(k[:, None] < R) & (ns[None, :] < N),
                        other=0,
                    )
                    acc = tl.dot(x, a, acc, input_precision=INPUT_PRECISION)
                acc = acc * tl.load(SCALES + w)
                base = tl.load(B + off, mask=mask, other=0).to(tl.float32)
                tl.store(
                    O + off, (base + acc).to(O.dtype.element_ty), mask=mask
                )
            else:
                tl.store(
                    O + off, tl.load(B + off, mask=mask, other=0), mask=mask
                )


def step_b_q(q_lora_a, A_buf, batch_info, base_output):
    seg = batch_info.seg_indptr
    widx = batch_info.weight_indices
    ranks = batch_info.lora_ranks
    scales = batch_info.scalings
    perm = batch_info.permutation
    if not (
        seg.is_contiguous()
        and widx.is_contiguous()
        and ranks.is_contiguous()
        and scales.is_contiguous()
        and (perm is None or perm.is_contiguous())
    ):
        return _strided_metadata_entry(
            q_lora_a, A_buf, batch_info, base_output
        )
    q_contiguous = q_lora_a.is_contiguous()
    a_contiguous = A_buf.is_contiguous()
    if (
        not q_contiguous
        and q_lora_a.dtype == torch.float32
        and (q_lora_a.stride(2) != 1)
    ):
        q_lora_a = q_lora_a.contiguous()
    if not a_contiguous and A_buf.dtype == torch.float32:
        A_buf = A_buf.contiguous()
    if not base_output.is_contiguous():
        if (
            (q_lora_a.shape[2] > 128 or q_lora_a.dtype == torch.float32)
            and q_lora_a.shape[0] > 0
            and (q_lora_a.shape[1] > 0)
            and (q_lora_a.shape[2] > 0)
            and (A_buf.shape[0] > 0)
            and (A_buf.shape[2] > 0)
            and (batch_info.bs > 0)
            and (perm is None or perm.numel() > 0)
        ):
            return _rank_base_view_entry(
                q_lora_a, A_buf, batch_info, base_output
            )
        if (
            q_lora_a.dtype in (torch.float16, torch.bfloat16)
            and perm is None
            and (q_lora_a.shape[0] > 0)
            and (q_lora_a.shape[1] > 0)
            and (q_lora_a.shape[1] % 64 == 0)
            and (A_buf.shape[2] > 0)
            and (A_buf.shape[2] % 128 == 0)
            and (q_lora_a.shape[2] in (16, 32, 64, 128))
            and (A_buf.shape[0] > 0)
            and (batch_info.bs > 0)
        ):
            return _aligned_base_view_entry(
                q_lora_a, A_buf, batch_info, base_output
            )
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
            S * H * N,
            1024,
            grid=((S * H * N + 1024 - 1) // 1024,),
            warmup=False,
        )
        return out
    if (
        perm is None
        and H % 64 == 0
        and (N % 128 == 0)
        and (R in (16, 32, 64, 128))
        and (q_lora_a.dtype != torch.float32)
    ):
        out = torch.empty_like(base_output)
        nt = N // 128
        bm = 32 if R == 16 else 64
        if R == 16 and torch.version.hip is not None:
            ml = batch_info.max_len
            if (
                q_contiguous
                and a_contiguous
                and (ml is not None)
                and (N & N - 1 == 0)
            ):
                maxt = max(1, (ml * H + 15) // 16)
                _generic_seggrid.run(
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
                        batch_info.bs,
                        nt,
                        S,
                        N,
                        R,
                        1,
                        R * N,
                        N,
                        1,
                        N,
                        1,
                        16,
                        128,
                        R,
                    ),
                    num_warps=4,
                    num_stages=1,
                    grid=(maxt * nt, batch_info.bs),
                    warmup=False,
                )
                return out
        total = S * H // bm * nt
        if q_contiguous and a_contiguous:
            _step_b_q_aligned.run(
                q_lora_a,
                A_buf,
                base_output,
                out,
                seg,
                widx,
                ranks,
                scales,
                (H, batch_info.bs, nt, R, 1, R * N, N, 1, N, 1, bm, 128, R),
                num_warps=4,
                num_stages=1,
                grid=(total,),
                warmup=False,
            )
        else:
            xs = q_lora_a.stride()
            if xs[0] == H * xs[1]:
                _step_b_q_aligned.run(
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
                        batch_info.bs,
                        nt,
                        xs[1],
                        xs[2],
                        *A_buf.stride(),
                        N,
                        1,
                        bm,
                        128,
                        R,
                    ),
                    num_warps=4,
                    num_stages=1,
                    grid=(total,),
                    warmup=False,
                )
            else:
                _step_b_q_aligned_views.run(
                    q_lora_a,
                    A_buf,
                    base_output,
                    out,
                    seg,
                    widx,
                    ranks,
                    scales,
                    H,
                    batch_info.bs,
                    nt,
                    xs[1],
                    xs[2],
                    *A_buf.stride(),
                    N,
                    1,
                    bm,
                    128,
                    R,
                    xs[0],
                    num_warps=4,
                    num_stages=1,
                    grid=(total,),
                    warmup=False,
                )
        return out
    xs = q_lora_a.stride()
    os = base_output.stride()
    if R > 128 or q_lora_a.dtype == torch.float32:
        pm = S if perm is None else perm.numel()
        out = base_output.clone() if pm != S else torch.empty_like(base_output)
        if pm:
            bound = batch_info.max_len or S
            _rank_segment_fused.run(
                q_lora_a,
                A_buf,
                base_output,
                out,
                seg,
                widx,
                ranks,
                scales,
                perm,
                H,
                R,
                N,
                pm,
                batch_info.bs,
                perm is not None,
                *xs,
                *A_buf.stride(),
                *os,
                32,
                64,
                64,
                (
                    "tf32x3"
                    if q_lora_a.dtype == torch.float32
                    and torch.version.hip is None
                    else "ieee"
                ),
                num_warps=4,
                num_stages=1,
                grid=(
                    max(1, (bound * H + 32 - 1) // 32),
                    batch_info.bs + 1,
                    (N + 64 - 1) // 64,
                ),
                warmup=False,
            )
        return out
    bk = max(16, 0 if R == 0 else 1 << (R - 1).bit_length())
    flat = perm is None and xs[0] == H * xs[1] and (os[0] == H * os[1])
    bm, bn = (16, 64)
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
def _step_b_q_kernel_strided(
    X,
    A,
    B,
    O,
    SEG,
    WIDX,
    RANKS,
    SCALES,
    PERM,
    H: tl.constexpr,
    R: tl.constexpr,
    N: tl.constexpr,
    PM: tl.constexpr,
    BS: tl.constexpr,
    NT: tl.constexpr,
    XS0: tl.constexpr,
    XS1: tl.constexpr,
    XS2: tl.constexpr,
    AS0: tl.constexpr,
    AS1: tl.constexpr,
    AS2: tl.constexpr,
    OS0: tl.constexpr,
    OS1: tl.constexpr,
    OS2: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    FLAT: tl.constexpr,
    HAS_PERM: tl.constexpr,
    SEG_STRIDE: tl.constexpr,
    WIDX_STRIDE: tl.constexpr,
    RANKS_STRIDE: tl.constexpr,
    SCALES_STRIDE: tl.constexpr,
    PERM_STRIDE: tl.constexpr,
):
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
            rows = tl.load(
                PERM + (0 + idx) * PERM_STRIDE, mask=keep, other=0
            ).to(tl.int32)
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
            lo = tl.load(SEG + (0 + b) * SEG_STRIDE) * H
            hi = tl.load(SEG + (0 + b + 1) * SEG_STRIDE) * H
        else:
            lo = tl.load(SEG + (0 + b) * SEG_STRIDE)
            hi = tl.load(SEG + (0 + b + 1) * SEG_STRIDE)
        if lo < hi:
            w = tl.load(WIDX + (0 + b) * WIDX_STRIDE)
            if tl.load(RANKS + (0 + w) * RANKS_STRIDE) > 0:
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
                partial = tl.dot(x, a)
                acc = acc + tl.where(rmask[:, None], partial, 0.0)
                rs = tl.where(
                    rmask, tl.load(SCALES + (0 + w) * SCALES_STRIDE), rs
                )
    off = ooff + offs_n[None, :] * OS2
    mask = keep[:, None] & nmask[None, :]
    base = tl.load(B + off, mask=mask, other=0.0).to(tl.float32)
    tl.store(
        O + off, (base + acc * rs[:, None]).to(O.dtype.element_ty), mask=mask
    )


@triton.jit
def _step_b_q_aligned_strided(
    X,
    A,
    B,
    O,
    SEG,
    WIDX,
    RANKS,
    SCALES,
    H: tl.constexpr,
    BS: tl.constexpr,
    NT: tl.constexpr,
    XS1: tl.constexpr,
    XS2: tl.constexpr,
    AS0: tl.constexpr,
    AS1: tl.constexpr,
    AS2: tl.constexpr,
    OS1: tl.constexpr,
    OS2: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    SEG_STRIDE: tl.constexpr,
    WIDX_STRIDE: tl.constexpr,
    RANKS_STRIDE: tl.constexpr,
    SCALES_STRIDE: tl.constexpr,
    PERM_STRIDE: tl.constexpr,
):
    offs_k = tl.arange(0, BK)
    pid = tl.program_id(0)
    m0 = pid // NT * BM
    m = m0 + tl.arange(0, BM)
    offs_n = pid % NT * BN + tl.arange(0, BN)
    off = m[:, None] * OS1 + offs_n[None, :] * OS2
    w = 0
    act = False
    for b in range(BS):
        wb = tl.load(WIDX + (0 + b) * WIDX_STRIDE)
        hit = (
            (m0 >= tl.load(SEG + (0 + b) * SEG_STRIDE) * H)
            & (m0 < tl.load(SEG + (0 + b + 1) * SEG_STRIDE) * H)
            & (tl.load(RANKS + (0 + wb) * RANKS_STRIDE) > 0)
        )
        w = tl.where(hit, wb, w)
        act = act | hit
    if act:
        x = tl.load(X + m[:, None] * XS1 + offs_k[None, :] * XS2)
        a = tl.load(
            A + w * AS0 + offs_k[:, None] * AS1 + offs_n[None, :] * AS2
        )
        acc = tl.dot(x, a) * tl.load(SCALES + (0 + w) * SCALES_STRIDE)
        base = tl.load(B + off, cache_modifier=".cg").to(tl.float32)
        tl.store(O + off, (base + acc).to(O.dtype.element_ty))
    else:
        tl.store(O + off, tl.load(B + off))


@triton.jit
def _rank_segment_fused_strided(
    X,
    A,
    B,
    O,
    SEG,
    WIDX,
    RANKS,
    SCALES,
    PERM,
    H: tl.constexpr,
    R: tl.constexpr,
    N: tl.constexpr,
    PM: tl.constexpr,
    BS: tl.constexpr,
    HAS_PERM: tl.constexpr,
    XS0: tl.constexpr,
    XS1: tl.constexpr,
    XS2: tl.constexpr,
    AS0: tl.constexpr,
    AS1: tl.constexpr,
    AS2: tl.constexpr,
    OS0: tl.constexpr,
    OS1: tl.constexpr,
    OS2: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    INPUT_PRECISION: tl.constexpr,
    SEG_STRIDE: tl.constexpr,
    WIDX_STRIDE: tl.constexpr,
    RANKS_STRIDE: tl.constexpr,
    SCALES_STRIDE: tl.constexpr,
    PERM_STRIDE: tl.constexpr,
):
    b = tl.program_id(1)
    ns = tl.program_id(2) * BN + tl.arange(0, BN)
    if b == BS:
        first = tl.load(SEG + 0 * SEG_STRIDE)
        last = tl.load(SEG + (0 + BS) * SEG_STRIDE)
        if first > 0 or last < PM:
            for begin in range(
                tl.program_id(0) * BM, PM * H, tl.num_programs(0) * BM
            ):
                m = begin + tl.arange(0, BM)
                pos, heads = (m // H, m % H)
                keep = (pos < PM) & ((pos < first) | (pos >= last))
                if HAS_PERM:
                    rows = tl.load(
                        PERM + (0 + pos) * PERM_STRIDE, mask=keep, other=0
                    ).to(tl.int32)
                else:
                    rows = pos
                off = (
                    rows[:, None] * OS0
                    + heads[:, None] * OS1
                    + ns[None, :] * OS2
                )
                mask = keep[:, None] & (ns[None, :] < N)
                tl.store(
                    O + off, tl.load(B + off, mask=mask, other=0), mask=mask
                )
    else:
        lo = tl.load(SEG + (0 + b) * SEG_STRIDE)
        hi = tl.load(SEG + (0 + b + 1) * SEG_STRIDE)
        m0 = lo * H + tl.program_id(0) * BM
        if m0 < hi * H:
            w = tl.load(WIDX + (0 + b) * WIDX_STRIDE)
            m = m0 + tl.arange(0, BM)
            pos, heads = (m // H, m % H)
            keep = (pos < hi) & (pos < PM)
            if HAS_PERM:
                rows = tl.load(
                    PERM + (0 + pos) * PERM_STRIDE, mask=keep, other=0
                ).to(tl.int32)
            else:
                rows = pos
            off = (
                rows[:, None] * OS0 + heads[:, None] * OS1 + ns[None, :] * OS2
            )
            mask = keep[:, None] & (ns[None, :] < N)
            if tl.load(RANKS + (0 + w) * RANKS_STRIDE) > 0:
                ks = tl.arange(0, BK)
                acc = tl.zeros((BM, BN), tl.float32)
                for chunk in range(tl.cdiv(R, BK)):
                    k = chunk * BK + ks
                    x = tl.load(
                        X
                        + rows[:, None] * XS0
                        + heads[:, None] * XS1
                        + k[None, :] * XS2,
                        mask=keep[:, None] & (k[None, :] < R),
                        other=0,
                    )
                    a = tl.load(
                        A + w * AS0 + k[:, None] * AS1 + ns[None, :] * AS2,
                        mask=(k[:, None] < R) & (ns[None, :] < N),
                        other=0,
                    )
                    acc = tl.dot(x, a, acc, input_precision=INPUT_PRECISION)
                acc = acc * tl.load(SCALES + (0 + w) * SCALES_STRIDE)
                base = tl.load(B + off, mask=mask, other=0).to(tl.float32)
                tl.store(
                    O + off, (base + acc).to(O.dtype.element_ty), mask=mask
                )
            else:
                tl.store(
                    O + off, tl.load(B + off, mask=mask, other=0), mask=mask
                )


def _strided_metadata_entry(q_lora_a, A_buf, batch_info, base_output):
    seg = batch_info.seg_indptr
    widx = batch_info.weight_indices
    ranks = batch_info.lora_ranks
    scales = batch_info.scalings
    perm = batch_info.permutation
    if perm is not None:
        perm = perm
    if (
        not q_lora_a.is_contiguous()
        and q_lora_a.dtype == torch.float32
        and (q_lora_a.stride(2) != 1)
    ):
        q_lora_a = q_lora_a.contiguous()
    if not A_buf.is_contiguous() and A_buf.dtype == torch.float32:
        A_buf = A_buf.contiguous()
    if not base_output.is_contiguous():
        if (
            (q_lora_a.shape[2] > 128 or q_lora_a.dtype == torch.float32)
            and q_lora_a.shape[0] > 0
            and (q_lora_a.shape[1] > 0)
            and (q_lora_a.shape[2] > 0)
            and (A_buf.shape[0] > 0)
            and (A_buf.shape[2] > 0)
            and (batch_info.bs > 0)
            and (perm is None or perm.numel() > 0)
        ):
            return _rank_base_view_entry(
                q_lora_a, A_buf, batch_info, base_output
            )
        if (
            q_lora_a.dtype in (torch.float16, torch.bfloat16)
            and perm is None
            and (q_lora_a.shape[0] > 0)
            and (q_lora_a.shape[1] > 0)
            and (q_lora_a.shape[1] % 64 == 0)
            and (A_buf.shape[2] > 0)
            and (A_buf.shape[2] % 128 == 0)
            and (q_lora_a.shape[2] in (16, 32, 64, 128))
            and (A_buf.shape[0] > 0)
            and (batch_info.bs > 0)
        ):
            return _aligned_base_view_entry(
                q_lora_a, A_buf, batch_info, base_output
            )
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
            S * H * N,
            1024,
            grid=((S * H * N + 1024 - 1) // 1024,),
            warmup=False,
        )
        return out
    xs = q_lora_a.stride()
    os = base_output.stride()
    if R > 128 or q_lora_a.dtype == torch.float32:
        pm = S if perm is None else perm.numel()
        out = base_output.clone() if pm != S else torch.empty_like(base_output)
        if pm:
            bound = batch_info.max_len or S
            _rank_segment_fused_strided.run(
                q_lora_a,
                A_buf,
                base_output,
                out,
                seg,
                widx,
                ranks,
                scales,
                perm,
                H,
                R,
                N,
                pm,
                batch_info.bs,
                perm is not None,
                *xs,
                *A_buf.stride(),
                *os,
                32,
                64,
                64,
                (
                    "tf32x3"
                    if q_lora_a.dtype == torch.float32
                    and torch.version.hip is None
                    else "ieee"
                ),
                seg.stride(0),
                widx.stride(0),
                ranks.stride(0),
                scales.stride(0),
                0 if perm is None else perm.stride(0),
                num_warps=4,
                num_stages=1,
                grid=(
                    max(1, (bound * H + 32 - 1) // 32),
                    batch_info.bs + 1,
                    (N + 64 - 1) // 64,
                ),
                warmup=False,
            )
        return out
    bk = max(16, 0 if R == 0 else 1 << (R - 1).bit_length())
    flat = perm is None and xs[0] == H * xs[1] and (os[0] == H * os[1])
    bm, bn = (64, 128)
    if (
        perm is None
        and os[0] == H * os[1]
        and (H % bm == 0)
        and (N % bn == 0)
        and (bk == R)
    ):
        out = torch.empty_like(base_output)
        nt = N // bn
        total = (S * H + bm - 1) // bm * nt
        if xs[0] == H * xs[1]:
            _step_b_q_aligned_strided.run(
                q_lora_a,
                A_buf,
                base_output,
                out,
                seg,
                widx,
                ranks,
                scales,
                H,
                batch_info.bs,
                nt,
                xs[1],
                xs[2],
                *A_buf.stride(),
                os[1],
                os[2],
                bm,
                bn,
                bk,
                seg.stride(0),
                widx.stride(0),
                ranks.stride(0),
                scales.stride(0),
                0 if perm is None else perm.stride(0),
                num_warps=4,
                num_stages=1,
                grid=(total,),
                warmup=False,
            )
        else:
            _step_b_q_aligned_strided_views.run(
                q_lora_a,
                A_buf,
                base_output,
                out,
                seg,
                widx,
                ranks,
                scales,
                H,
                batch_info.bs,
                nt,
                xs[1],
                xs[2],
                *A_buf.stride(),
                os[1],
                os[2],
                bm,
                bn,
                bk,
                seg.stride(0),
                widx.stride(0),
                ranks.stride(0),
                scales.stride(0),
                0 if perm is None else perm.stride(0),
                xs[0],
                num_warps=4,
                num_stages=1,
                grid=(total,),
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
    nt = (N + bn - 1) // bn
    total = (pm * H + bm - 1) // bm * nt
    np = total
    _step_b_q_kernel_strided.run(
        q_lora_a,
        A_buf,
        base_output,
        out,
        seg,
        widx,
        ranks,
        scales,
        perm,
        H,
        R,
        N,
        pm,
        batch_info.bs,
        nt,
        *xs,
        *A_buf.stride(),
        *os,
        bm,
        bn,
        bk,
        flat,
        has_perm,
        seg.stride(0),
        widx.stride(0),
        ranks.stride(0),
        scales.stride(0),
        0 if perm is None else perm.stride(0),
        num_warps=4,
        num_stages=2,
        grid=(np,),
        warmup=False,
    )
    return out


@triton.jit
def _step_b_q_aligned_views(
    X,
    A,
    B,
    O,
    SEG,
    WIDX,
    RANKS,
    SCALES,
    H: tl.constexpr,
    BS: tl.constexpr,
    NT: tl.constexpr,
    XS1: tl.constexpr,
    XS2: tl.constexpr,
    AS0: tl.constexpr,
    AS1: tl.constexpr,
    AS2: tl.constexpr,
    OS1: tl.constexpr,
    OS2: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    XS0: tl.constexpr,
):
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
        x = tl.load(
            X
            + (m[:, None] // H * XS0 + m[:, None] % H * XS1)
            + offs_k[None, :] * XS2
        )
        a = tl.load(
            A + w * AS0 + offs_k[:, None] * AS1 + offs_n[None, :] * AS2
        )
        acc = tl.dot(x, a) * tl.load(SCALES + w)
        base = tl.load(B + off, cache_modifier=".cg").to(tl.float32)
        tl.store(O + off, (base + acc).to(O.dtype.element_ty))
    else:
        tl.store(O + off, tl.load(B + off))


@triton.jit
def _step_b_q_aligned_strided_views(
    X,
    A,
    B,
    O,
    SEG,
    WIDX,
    RANKS,
    SCALES,
    H: tl.constexpr,
    BS: tl.constexpr,
    NT: tl.constexpr,
    XS1: tl.constexpr,
    XS2: tl.constexpr,
    AS0: tl.constexpr,
    AS1: tl.constexpr,
    AS2: tl.constexpr,
    OS1: tl.constexpr,
    OS2: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    SEG_STRIDE: tl.constexpr,
    WIDX_STRIDE: tl.constexpr,
    RANKS_STRIDE: tl.constexpr,
    SCALES_STRIDE: tl.constexpr,
    PERM_STRIDE: tl.constexpr,
    XS0: tl.constexpr,
):
    offs_k = tl.arange(0, BK)
    pid = tl.program_id(0)
    m0 = pid // NT * BM
    m = m0 + tl.arange(0, BM)
    offs_n = pid % NT * BN + tl.arange(0, BN)
    off = m[:, None] * OS1 + offs_n[None, :] * OS2
    w = 0
    act = False
    for b in range(BS):
        wb = tl.load(WIDX + (0 + b) * WIDX_STRIDE)
        hit = (
            (m0 >= tl.load(SEG + (0 + b) * SEG_STRIDE) * H)
            & (m0 < tl.load(SEG + (0 + b + 1) * SEG_STRIDE) * H)
            & (tl.load(RANKS + (0 + wb) * RANKS_STRIDE) > 0)
        )
        w = tl.where(hit, wb, w)
        act = act | hit
    if act:
        x = tl.load(
            X
            + (m[:, None] // H * XS0 + m[:, None] % H * XS1)
            + offs_k[None, :] * XS2
        )
        a = tl.load(
            A + w * AS0 + offs_k[:, None] * AS1 + offs_n[None, :] * AS2
        )
        acc = tl.dot(x, a) * tl.load(SCALES + (0 + w) * SCALES_STRIDE)
        base = tl.load(B + off, cache_modifier=".cg").to(tl.float32)
        tl.store(O + off, (base + acc).to(O.dtype.element_ty))
    else:
        tl.store(O + off, tl.load(B + off))


@triton.jit
def _aligned_base_views(
    X,
    A,
    B,
    O,
    SEG,
    WIDX,
    RANKS,
    SCALES,
    H: tl.constexpr,
    BS: tl.constexpr,
    NT: tl.constexpr,
    XS1: tl.constexpr,
    XS2: tl.constexpr,
    AS0: tl.constexpr,
    AS1: tl.constexpr,
    AS2: tl.constexpr,
    OS1: tl.constexpr,
    OS2: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    SEG_STRIDE: tl.constexpr,
    WIDX_STRIDE: tl.constexpr,
    RANKS_STRIDE: tl.constexpr,
    SCALES_STRIDE: tl.constexpr,
    PERM_STRIDE: tl.constexpr,
    XS0: tl.constexpr,
    BS0: tl.constexpr,
    BS1: tl.constexpr,
    BS2: tl.constexpr,
):
    offs_k = tl.arange(0, BK)
    pid = tl.program_id(0)
    m0 = pid // NT * BM
    m = m0 + tl.arange(0, BM)
    offs_n = pid % NT * BN + tl.arange(0, BN)
    off = m[:, None] * OS1 + offs_n[None, :] * OS2
    w = 0
    act = False
    for b in range(BS):
        wb = tl.load(WIDX + (0 + b) * WIDX_STRIDE)
        hit = (
            (m0 >= tl.load(SEG + (0 + b) * SEG_STRIDE) * H)
            & (m0 < tl.load(SEG + (0 + b + 1) * SEG_STRIDE) * H)
            & (tl.load(RANKS + (0 + wb) * RANKS_STRIDE) > 0)
        )
        w = tl.where(hit, wb, w)
        act = act | hit
    if act:
        x = tl.load(
            X
            + (m[:, None] // H * XS0 + m[:, None] % H * XS1)
            + offs_k[None, :] * XS2
        )
        a = tl.load(
            A + w * AS0 + offs_k[:, None] * AS1 + offs_n[None, :] * AS2
        )
        acc = tl.dot(x, a) * tl.load(SCALES + (0 + w) * SCALES_STRIDE)
        base = tl.load(
            B
            + m[:, None] // H * BS0
            + m[:, None] % H * BS1
            + offs_n[None, :] * BS2,
            cache_modifier=".cg",
        ).to(tl.float32)
        tl.store(O + off, (base + acc).to(O.dtype.element_ty))
    else:
        tl.store(
            O + off,
            tl.load(
                B
                + m[:, None] // H * BS0
                + m[:, None] % H * BS1
                + offs_n[None, :] * BS2
            ),
        )


def _aligned_base_view_entry(q_lora_a, A_buf, batch_info, base_output):
    S, H, R = q_lora_a.shape
    N = A_buf.shape[2]
    out = torch.empty(
        base_output.shape, dtype=base_output.dtype, device=base_output.device
    )
    seg = batch_info.seg_indptr
    widx = batch_info.weight_indices
    ranks = batch_info.lora_ranks
    scales = batch_info.scalings
    xs = q_lora_a.stride()
    _aligned_base_views.run(
        q_lora_a,
        A_buf,
        base_output,
        out,
        seg,
        widx,
        ranks,
        scales,
        H,
        batch_info.bs,
        N // 128,
        xs[1],
        xs[2],
        *A_buf.stride(),
        N,
        1,
        64,
        128,
        R,
        seg.stride(0),
        widx.stride(0),
        ranks.stride(0),
        scales.stride(0),
        0,
        xs[0],
        *base_output.stride(),
        num_warps=4,
        num_stages=1,
        grid=(S * H // 64 * (N // 128),),
        warmup=False,
    )
    return out


@triton.jit
def _rank_base_views(
    X,
    A,
    B,
    O,
    SEG,
    WIDX,
    RANKS,
    SCALES,
    PERM,
    H: tl.constexpr,
    R: tl.constexpr,
    N: tl.constexpr,
    PM: tl.constexpr,
    BS: tl.constexpr,
    HAS_PERM: tl.constexpr,
    XS0: tl.constexpr,
    XS1: tl.constexpr,
    XS2: tl.constexpr,
    AS0: tl.constexpr,
    AS1: tl.constexpr,
    AS2: tl.constexpr,
    OS0: tl.constexpr,
    OS1: tl.constexpr,
    OS2: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    INPUT_PRECISION: tl.constexpr,
    SEG_STRIDE: tl.constexpr,
    WIDX_STRIDE: tl.constexpr,
    RANKS_STRIDE: tl.constexpr,
    SCALES_STRIDE: tl.constexpr,
    PERM_STRIDE: tl.constexpr,
    BS0: tl.constexpr,
    BS1: tl.constexpr,
    BS2: tl.constexpr,
):
    b = tl.program_id(1)
    ns = tl.program_id(2) * BN + tl.arange(0, BN)
    if b == BS:
        first = tl.load(SEG + 0 * SEG_STRIDE)
        last = tl.load(SEG + (0 + BS) * SEG_STRIDE)
        if first > 0 or last < PM:
            for begin in range(
                tl.program_id(0) * BM, PM * H, tl.num_programs(0) * BM
            ):
                m = begin + tl.arange(0, BM)
                pos, heads = (m // H, m % H)
                keep = (pos < PM) & ((pos < first) | (pos >= last))
                if HAS_PERM:
                    rows = tl.load(
                        PERM + (0 + pos) * PERM_STRIDE, mask=keep, other=0
                    ).to(tl.int32)
                else:
                    rows = pos
                off = (
                    rows[:, None] * OS0
                    + heads[:, None] * OS1
                    + ns[None, :] * OS2
                )
                mask = keep[:, None] & (ns[None, :] < N)
                tl.store(
                    O + off,
                    tl.load(
                        B
                        + rows[:, None] * BS0
                        + heads[:, None] * BS1
                        + ns[None, :] * BS2,
                        mask=mask,
                        other=0,
                    ),
                    mask=mask,
                )
    else:
        lo = tl.load(SEG + (0 + b) * SEG_STRIDE)
        hi = tl.load(SEG + (0 + b + 1) * SEG_STRIDE)
        m0 = lo * H + tl.program_id(0) * BM
        if m0 < hi * H:
            w = tl.load(WIDX + (0 + b) * WIDX_STRIDE)
            m = m0 + tl.arange(0, BM)
            pos, heads = (m // H, m % H)
            keep = (pos < hi) & (pos < PM)
            if HAS_PERM:
                rows = tl.load(
                    PERM + (0 + pos) * PERM_STRIDE, mask=keep, other=0
                ).to(tl.int32)
            else:
                rows = pos
            off = (
                rows[:, None] * OS0 + heads[:, None] * OS1 + ns[None, :] * OS2
            )
            mask = keep[:, None] & (ns[None, :] < N)
            if tl.load(RANKS + (0 + w) * RANKS_STRIDE) > 0:
                ks = tl.arange(0, BK)
                acc = tl.zeros((BM, BN), tl.float32)
                for chunk in range(tl.cdiv(R, BK)):
                    k = chunk * BK + ks
                    x = tl.load(
                        X
                        + rows[:, None] * XS0
                        + heads[:, None] * XS1
                        + k[None, :] * XS2,
                        mask=keep[:, None] & (k[None, :] < R),
                        other=0,
                    )
                    a = tl.load(
                        A + w * AS0 + k[:, None] * AS1 + ns[None, :] * AS2,
                        mask=(k[:, None] < R) & (ns[None, :] < N),
                        other=0,
                    )
                    acc = tl.dot(x, a, acc, input_precision=INPUT_PRECISION)
                acc = acc * tl.load(SCALES + (0 + w) * SCALES_STRIDE)
                base = tl.load(
                    B
                    + rows[:, None] * BS0
                    + heads[:, None] * BS1
                    + ns[None, :] * BS2,
                    mask=mask,
                    other=0,
                ).to(tl.float32)
                tl.store(
                    O + off, (base + acc).to(O.dtype.element_ty), mask=mask
                )
            else:
                tl.store(
                    O + off,
                    tl.load(
                        B
                        + rows[:, None] * BS0
                        + heads[:, None] * BS1
                        + ns[None, :] * BS2,
                        mask=mask,
                        other=0,
                    ),
                    mask=mask,
                )


@triton.jit
def _copy_strided_base(
    B,
    O,
    H: tl.constexpr,
    N: tl.constexpr,
    TOTAL: tl.constexpr,
    BS0: tl.constexpr,
    BS1: tl.constexpr,
    BS2: tl.constexpr,
):
    i = tl.program_id(0) * 1024 + tl.arange(0, 1024)
    off = i // (H * N) * BS0 + i // N % H * BS1 + i % N * BS2
    tl.store(O + i, tl.load(B + off, mask=i < TOTAL, other=0), mask=i < TOTAL)


def _rank_base_view_entry(q_lora_a, A_buf, batch_info, base_output):
    S, H, R = q_lora_a.shape
    N = A_buf.shape[2]
    out = torch.empty(
        base_output.shape, dtype=base_output.dtype, device=base_output.device
    )
    seg = batch_info.seg_indptr
    widx = batch_info.weight_indices
    ranks = batch_info.lora_ranks
    scales = batch_info.scalings
    perm = batch_info.permutation
    pm = S if perm is None else perm.numel()
    if pm != S:
        _copy_strided_base.run(
            base_output,
            out,
            H,
            N,
            S * H * N,
            *base_output.stride(),
            grid=((S * H * N + 1024 - 1) // 1024,),
            warmup=False,
        )
    bound = batch_info.max_len or S
    _rank_base_views.run(
        q_lora_a,
        A_buf,
        base_output,
        out,
        seg,
        widx,
        ranks,
        scales,
        perm,
        H,
        R,
        N,
        pm,
        batch_info.bs,
        perm is not None,
        *q_lora_a.stride(),
        *A_buf.stride(),
        H * N,
        N,
        1,
        32,
        64,
        64,
        (
            "tf32x3"
            if q_lora_a.dtype == torch.float32 and torch.version.hip is None
            else "ieee"
        ),
        seg.stride(0),
        widx.stride(0),
        ranks.stride(0),
        scales.stride(0),
        0 if perm is None else perm.stride(0),
        *base_output.stride(),
        num_warps=4,
        num_stages=1,
        grid=(
            max(1, (bound * H + 32 - 1) // 32),
            batch_info.bs + 1,
            (N + 64 - 1) // 64,
        ),
        warmup=False,
    )
    return out


@triton.jit
def _generic_seggrid(
    X, A, B, OUT, SEG, WIDX, RANKS, SCALES, META: tl.constexpr
):
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
    xp = tl.make_block_ptr(
        base=X + m0 * XS1,
        shape=(BM, BK),
        strides=(XS1, XS2),
        offsets=(0, 0),
        block_shape=(BM, BK),
        order=(1, 0) if XS2 <= XS1 else (0, 1),
    )
    ap = tl.make_block_ptr(
        base=A + w * AS0 + p % NT * BN * AS2,
        shape=(BK, BN),
        strides=(AS1, AS2),
        offsets=(0, 0),
        block_shape=(BK, BN),
        order=(1, 0) if AS2 <= AS1 else (0, 1),
    )
    x = tl.load(xp)
    a = tl.load(ap)
    acc = tl.dot(x, a) * tl.load(SCALES + w)
    off = m[:, None] * OS1 + offs_n[None, :] * OS2
    base = tl.load(B + off, cache_modifier=".cg").to(tl.float32)
    tl.store(OUT + off, (base + acc).to(OUT.dtype.element_ty))


__all__ = ["step_b_q"]
