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
def _seg(
    X,
    A,
    B,
    O,
    SEG,
    WIDX,
    RANKS,
    SCALES,
    H: tl.constexpr,
    R: tl.constexpr,
    N: tl.constexpr,
    NT: tl.constexpr,
    S: tl.constexpr,
    BS: tl.constexpr,
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
    ML: tl.constexpr,
    NEXACT: tl.constexpr,
    KEXACT: tl.constexpr,
):
    b = tl.program_id(1)
    pid = tl.program_id(0)
    if (b == 0) & (pid == 0):
        first_count = tl.load(SEG) * H * N
        v = tl.arange(0, 1024)
        for j in range(tl.cdiv(first_count, 1024)):
            off = j * 1024 + v
            tl.store(
                O + off,
                tl.load(B + off, mask=off < first_count, other=0),
                mask=off < first_count,
            )
        lim = tl.load(SEG + BS) * H * N
        for j in range(tl.cdiv(S * H * N - lim, 1024)):
            off = lim + j * 1024 + v
            tl.store(
                O + off,
                tl.load(B + off, mask=off < S * H * N, other=0),
                mask=off < S * H * N,
            )
    start = tl.load(SEG + b)
    end = tl.load(SEG + b + 1)
    if start == end:
        return
    seglen = (end - start) * H
    m0 = pid // NT * BM * ML
    if m0 >= seglen:
        return
    offs_n = pid % NT * BN + tl.arange(0, BN)
    offs_k = tl.arange(0, BK)
    w = tl.load(WIDX + b)
    sc = tl.where(tl.load(RANKS + w) > 0, tl.load(SCALES + w), 0.0)
    ap = A + w * AS0 + offs_k[:, None] * AS1 + offs_n[None, :] * AS2
    if tl.load(RANKS + w) == 0:
        for i in range(ML):
            if m0 + i * BM < seglen:
                m = m0 + i * BM + tl.arange(0, BM)
                pos = start * H + m
                op = pos[:, None] * OS1 + offs_n[None, :] * OS2
                valid = (m[:, None] < seglen) & (offs_n[None, :] < N)
                tl.store(
                    O + op, tl.load(B + op, mask=valid, other=0.0), mask=valid
                )
        return
    nok = offs_n < N
    kok = offs_k < R
    a = tl.load(ap, mask=kok[:, None] & nok[None, :], other=0.0)
    for i in range(ML):
        if m0 + i * BM < seglen:
            m = m0 + i * BM + tl.arange(0, BM)
            pos = start * H + m
            xp = X + pos[:, None] * XS1 + offs_k[None, :] * XS2
            op = pos[:, None] * OS1 + offs_n[None, :] * OS2
            mok = m < seglen
            x = tl.load(xp, mask=mok[:, None] & kok[None, :], other=0.0)
            acc = (
                tl.dot(
                    x,
                    a,
                    input_precision=(
                        "ieee" if X.dtype.element_ty == tl.float32 else "tf32"
                    ),
                )
                * sc
            )
            omask = mok[:, None] & nok[None, :]
            base = tl.load(B + op, mask=omask, other=0.0).to(tl.float32)
            tl.store(O + op, (base + acc).to(O.dtype.element_ty), mask=omask)


@triton.jit
def _permseg(
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
    NT: tl.constexpr,
    BS: tl.constexpr,
    S: tl.constexpr,
    COPY_UNCOVERED: tl.constexpr,
    XS0: tl.constexpr,
    XS1: tl.constexpr,
    XS2: tl.constexpr,
    AS0: tl.constexpr,
    AS1: tl.constexpr,
    AS2: tl.constexpr,
    OS0: tl.constexpr,
    OS1: tl.constexpr,
    OS2: tl.constexpr,
    BH: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    HEXACT: tl.constexpr,
    NEXACT: tl.constexpr,
    KEXACT: tl.constexpr,
):
    b = tl.program_id(1)
    pid = tl.program_id(0)
    if COPY_UNCOVERED:
        if (b == 0) & (pid == 0):
            v = tl.arange(0, 1024)
            first = tl.load(SEG)
            lim = tl.load(SEG + BS)
            for p in range(first):
                row = tl.load(PERM + p).to(tl.int32)
                for j in range(tl.cdiv(H * N, 1024)):
                    off = j * 1024 + v
                    address = row * H * N + off
                    tl.store(
                        O + address,
                        tl.load(B + address, mask=off < H * N, other=0),
                        mask=off < H * N,
                    )
            for p in range(lim, S):
                row = tl.load(PERM + p).to(tl.int32)
                for j in range(tl.cdiv(H * N, 1024)):
                    off = j * 1024 + v
                    address = row * H * N + off
                    tl.store(
                        O + address,
                        tl.load(B + address, mask=off < H * N, other=0),
                        mask=off < H * N,
                    )
    start = tl.load(SEG + b)
    end = tl.load(SEG + b + 1)
    if start == end:
        return
    j = pid // NT
    if j >= end - start:
        return
    row = tl.load(PERM + start + j).to(tl.int32)
    h = tl.arange(0, BH)
    offs_n = pid % NT * BN + tl.arange(0, BN)
    offs_k = tl.arange(0, BK)
    w = tl.load(WIDX + b)
    sc = tl.where(tl.load(RANKS + w) > 0, tl.load(SCALES + w), 0.0)
    ap = A + w * AS0 + offs_k[:, None] * AS1 + offs_n[None, :] * AS2
    if tl.load(RANKS + w) == 0:
        if HEXACT and NEXACT:
            op = row * OS0 + h[:, None] * OS1 + offs_n[None, :] * OS2
            tl.store(O + op, tl.load(B + op))
        else:
            hs = h if HEXACT else tl.minimum(h, H - 1)
            op = row * OS0 + hs[:, None] * OS1 + offs_n[None, :] * OS2
            valid = (h[:, None] < H) & (offs_n[None, :] < N)
            tl.store(
                O + op, tl.load(B + op, mask=valid, other=0.0), mask=valid
            )
        return
    if HEXACT and NEXACT and KEXACT:
        xp = X + row * XS0 + h[:, None] * XS1 + offs_k[None, :] * XS2
        op = row * OS0 + h[:, None] * OS1 + offs_n[None, :] * OS2
        x = tl.load(xp)
        a = tl.load(ap)
        acc = (
            tl.dot(
                x,
                a,
                input_precision=(
                    "ieee" if X.dtype.element_ty == tl.float32 else "tf32"
                ),
            )
            * sc
        )
        base = tl.load(B + op).to(tl.float32)
        tl.store(O + op, (base + acc).to(O.dtype.element_ty))
    else:
        hs = h if HEXACT else tl.minimum(h, H - 1)
        hok = h < H
        nok = offs_n < N
        kok = offs_k < R
        xp = X + row * XS0 + hs[:, None] * XS1 + offs_k[None, :] * XS2
        op = row * OS0 + hs[:, None] * OS1 + offs_n[None, :] * OS2
        x = tl.load(xp, mask=hok[:, None] & kok[None, :], other=0.0)
        a = tl.load(ap, mask=kok[:, None] & nok[None, :], other=0.0)
        acc = (
            tl.dot(
                x,
                a,
                input_precision=(
                    "ieee" if X.dtype.element_ty == tl.float32 else "tf32"
                ),
            )
            * sc
        )
        omask = hok[:, None] & nok[None, :]
        base = tl.load(B + op, mask=omask, other=0.0).to(tl.float32)
        tl.store(O + op, (base + acc).to(O.dtype.element_ty), mask=omask)


@triton.jit
def _row_gemv(
    X,
    A,
    O,
    SEG,
    WI,
    RANKS,
    SCALES,
    PERM,
    H: tl.constexpr,
    R: tl.constexpr,
    N: tl.constexpr,
    BK: tl.constexpr,
    HAS_PERM: tl.constexpr,
):
    m = tl.program_id(0)
    h = tl.program_id(1)
    b = tl.program_id(2) // N
    n = tl.program_id(2) % N
    lo = tl.load(SEG + b)
    hi = tl.load(SEG + b + 1)
    if lo + m >= hi:
        return
    w = tl.load(WI + b)
    if tl.load(RANKS + w) == 0:
        return
    if HAS_PERM:
        row = tl.load(PERM + lo + m).to(tl.int32)
    else:
        row = lo + m
    k = tl.arange(0, BK)
    x = tl.load(X + (row * H + h) * R + k, mask=k < R, other=0.0).to(
        tl.float32
    )
    a = tl.load(A + w * R * N + k * N + n, mask=k < R, other=0.0).to(
        tl.float32
    )
    correction = tl.sum(x * a, 0) * tl.load(SCALES + w)
    ptr = O + (row * H + h) * N + n
    base = tl.load(ptr).to(tl.float32)
    tl.store(ptr, base + correction)


@triton.jit
def _copy_base(B, OUT, TOTAL: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(OUT + i, tl.load(B + i, mask=i < TOTAL, other=0), mask=i < TOTAL)


@triton.jit
def _bounded_head(
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
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
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
    tl.store(O + off, value.to(O.dtype.element_ty), mask=mask)


def _main_entry(q_lora_a, A_buf, batch_info, base_output):
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
    if not q_lora_a.is_contiguous():
        q_lora_a = q_lora_a.contiguous()
    if not A_buf.is_contiguous():
        A_buf = A_buf.contiguous()
    if not base_output.is_contiguous():
        base_output = base_output.contiguous()
    S, H, R = q_lora_a.shape
    original_h = H
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
        return out[:, :original_h, :] if H != original_h else out
    bs = batch_info.bs
    xs = q_lora_a.stride()
    os = base_output.stride()
    if R > 128:
        q_lora_a = q_lora_a.contiguous()
        A_buf = A_buf.contiguous()
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
                H,
                R,
                N,
                pm,
                perm is not None,
                32,
                64,
                64,
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
    bk = max(16, 0 if R == 0 else 1 << (R - 1).bit_length())
    pad_h = (
        max(16, 0 if H == 0 else 1 << (H - 1).bit_length())
        if perm is not None
        else H
    )
    if (
        N >= 32
        and bk <= 128
        and (perm is None or pad_h <= 128)
        and (R != bk or H != pad_h)
    ):
        packed_q = torch.zeros(
            (S, pad_h, bk), dtype=q_lora_a.dtype, device=q_lora_a.device
        )
        packed_b = (
            torch.zeros(
                (S, pad_h, N),
                dtype=base_output.dtype,
                device=base_output.device,
            )
            if pad_h != H
            else base_output
        )
        _pad_rows.run(
            q_lora_a,
            base_output,
            packed_q,
            packed_b,
            H,
            R,
            N,
            pad_h,
            bk,
            pad_h != H,
            num_stages=1,
            grid=(S * H, (N + 256 - 1) // 256 if pad_h != H else 1),
            warmup=False,
        )
        if R != bk:
            packed_a = torch.zeros(
                (adapter_count, bk, N), dtype=A_buf.dtype, device=A_buf.device
            )
            _pad_a.run(
                A_buf,
                packed_a,
                R,
                N,
                bk,
                num_stages=1,
                grid=(adapter_count, (R * N + 256 - 1) // 256),
                warmup=False,
            )
        else:
            packed_a = A_buf
        q_lora_a, A_buf, base_output = (packed_q, packed_a, packed_b)
        H, R = (pad_h, bk)
        xs = q_lora_a.stride()
        os = base_output.stride()
    wide_tile = N % 128 == 0
    bn = 128 if wide_tile else 32
    nt = (N + bn - 1) // bn
    nexact = wide_tile or N % 32 == 0
    kexact = bk == R
    longest = batch_info.max_len if batch_info.max_len is not None else S
    has_perm = perm is not None
    full_perm = has_perm and perm.numel() == S
    out = (
        base_output.clone()
        if has_perm and (not full_perm)
        else torch.empty_like(base_output)
    )
    if N < 32 or not kexact or (has_perm and H != pad_h):
        out = base_output.clone()
        if longest == 0:
            return out[:, :original_h, :] if H != original_h else out
        _row_gemv.run(
            q_lora_a,
            A_buf,
            out,
            seg,
            widx,
            ranks,
            scales,
            perm if has_perm else seg,
            H,
            R,
            N,
            bk,
            has_perm,
            num_stages=1,
            grid=(longest, H, bs * N),
            warmup=False,
        )
        return out[:, :original_h, :] if H != original_h else out
    if has_perm:
        bh = max(16, 0 if H == 0 else 1 << (H - 1).bit_length())
        _permseg.run(
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
            nt,
            bs,
            S,
            full_perm,
            *xs,
            *A_buf.stride(),
            *os,
            bh,
            bn,
            bk,
            bh == H,
            nexact,
            kexact,
            num_stages=1,
            grid=(max(1, longest) * nt, bs),
            warmup=False,
        )
        return out[:, :original_h, :] if H != original_h else out
    bm, ml = (512, 2)
    _seg.run(
        q_lora_a,
        A_buf,
        base_output,
        out,
        seg,
        widx,
        ranks,
        scales,
        H,
        R,
        N,
        nt,
        S,
        bs,
        xs[1],
        xs[2],
        *A_buf.stride(),
        os[1],
        os[2],
        bm,
        bn,
        bk,
        ml,
        nexact,
        kexact,
        num_stages=1,
        grid=(max(1, (longest * H + bm * ml - 1) // (bm * ml)) * nt, bs),
        warmup=False,
    )
    return out[:, :original_h, :] if H != original_h else out


@triton.jit
def _pad_rows(
    X,
    B,
    XP,
    BP,
    H: tl.constexpr,
    R: tl.constexpr,
    N: tl.constexpr,
    HP: tl.constexpr,
    RP: tl.constexpr,
    PAD_HEADS: tl.constexpr,
):
    row = tl.program_id(0)
    tile = tl.program_id(1)
    target = row // H * HP + row % H
    if tile == 0:
        k = tl.arange(0, RP)
        value = tl.load(X + row * R + k, mask=k < R, other=0.0)
        tl.store(XP + target * RP + k, value)
    if PAD_HEADS:
        n = tile * 256 + tl.arange(0, 256)
        value_b = tl.load(B + row * N + n, mask=n < N, other=0.0)
        tl.store(BP + target * N + n, value_b, mask=n < N)


@triton.jit
def _pad_a(A, AP, R: tl.constexpr, N: tl.constexpr, RP: tl.constexpr):
    w = tl.program_id(0)
    i = tl.program_id(1) * 256 + tl.arange(0, 256)
    value = tl.load(A + w * R * N + i, mask=i < R * N, other=0.0)
    tl.store(AP + w * RP * N + i, value, mask=i < RP * N)


@triton.jit
def _seg_strided(
    X,
    A,
    B,
    O,
    SEG,
    WIDX,
    RANKS,
    SCALES,
    H: tl.constexpr,
    R: tl.constexpr,
    N: tl.constexpr,
    NT: tl.constexpr,
    S: tl.constexpr,
    BS: tl.constexpr,
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
    ML: tl.constexpr,
    NEXACT: tl.constexpr,
    KEXACT: tl.constexpr,
    SEG_STRIDE: tl.constexpr,
    WIDX_STRIDE: tl.constexpr,
    RANKS_STRIDE: tl.constexpr,
    SCALES_STRIDE: tl.constexpr,
    PERM_STRIDE: tl.constexpr,
):
    b = tl.program_id(1)
    pid = tl.program_id(0)
    if (b == 0) & (pid == 0):
        first_count = tl.load(SEG + 0 * SEG_STRIDE) * H * N
        v = tl.arange(0, 1024)
        for j in range(tl.cdiv(first_count, 1024)):
            off = j * 1024 + v
            tl.store(
                O + off,
                tl.load(B + off, mask=off < first_count, other=0),
                mask=off < first_count,
            )
        lim = tl.load(SEG + (0 + BS) * SEG_STRIDE) * H * N
        for j in range(tl.cdiv(S * H * N - lim, 1024)):
            off = lim + j * 1024 + v
            tl.store(
                O + off,
                tl.load(B + off, mask=off < S * H * N, other=0),
                mask=off < S * H * N,
            )
    start = tl.load(SEG + (0 + b) * SEG_STRIDE)
    end = tl.load(SEG + (0 + b + 1) * SEG_STRIDE)
    if start == end:
        return
    seglen = (end - start) * H
    m0 = pid // NT * BM * ML
    if m0 >= seglen:
        return
    offs_n = pid % NT * BN + tl.arange(0, BN)
    offs_k = tl.arange(0, BK)
    w = tl.load(WIDX + (0 + b) * WIDX_STRIDE)
    sc = tl.where(
        tl.load(RANKS + (0 + w) * RANKS_STRIDE) > 0,
        tl.load(SCALES + (0 + w) * SCALES_STRIDE),
        0.0,
    )
    ap = A + w * AS0 + offs_k[:, None] * AS1 + offs_n[None, :] * AS2
    if tl.load(RANKS + (0 + w) * RANKS_STRIDE) == 0:
        for i in range(ML):
            if m0 + i * BM < seglen:
                m = m0 + i * BM + tl.arange(0, BM)
                pos = start * H + m
                op = pos[:, None] * OS1 + offs_n[None, :] * OS2
                valid = (m[:, None] < seglen) & (offs_n[None, :] < N)
                tl.store(
                    O + op, tl.load(B + op, mask=valid, other=0.0), mask=valid
                )
        return
    nok = offs_n < N
    kok = offs_k < R
    a = tl.load(ap, mask=kok[:, None] & nok[None, :], other=0.0)
    for i in range(ML):
        if m0 + i * BM < seglen:
            m = m0 + i * BM + tl.arange(0, BM)
            pos = start * H + m
            xp = X + pos[:, None] * XS1 + offs_k[None, :] * XS2
            op = pos[:, None] * OS1 + offs_n[None, :] * OS2
            mok = m < seglen
            x = tl.load(xp, mask=mok[:, None] & kok[None, :], other=0.0)
            acc = (
                tl.dot(
                    x,
                    a,
                    input_precision=(
                        "ieee" if X.dtype.element_ty == tl.float32 else "tf32"
                    ),
                )
                * sc
            )
            omask = mok[:, None] & nok[None, :]
            base = tl.load(B + op, mask=omask, other=0.0).to(tl.float32)
            tl.store(O + op, (base + acc).to(O.dtype.element_ty), mask=omask)


@triton.jit
def _permseg_strided(
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
    NT: tl.constexpr,
    BS: tl.constexpr,
    S: tl.constexpr,
    COPY_UNCOVERED: tl.constexpr,
    XS0: tl.constexpr,
    XS1: tl.constexpr,
    XS2: tl.constexpr,
    AS0: tl.constexpr,
    AS1: tl.constexpr,
    AS2: tl.constexpr,
    OS0: tl.constexpr,
    OS1: tl.constexpr,
    OS2: tl.constexpr,
    BH: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    HEXACT: tl.constexpr,
    NEXACT: tl.constexpr,
    KEXACT: tl.constexpr,
    SEG_STRIDE: tl.constexpr,
    WIDX_STRIDE: tl.constexpr,
    RANKS_STRIDE: tl.constexpr,
    SCALES_STRIDE: tl.constexpr,
    PERM_STRIDE: tl.constexpr,
):
    b = tl.program_id(1)
    pid = tl.program_id(0)
    if COPY_UNCOVERED:
        if (b == 0) & (pid == 0):
            v = tl.arange(0, 1024)
            first = tl.load(SEG + 0 * SEG_STRIDE)
            lim = tl.load(SEG + (0 + BS) * SEG_STRIDE)
            for p in range(first):
                row = tl.load(PERM + (0 + p) * PERM_STRIDE).to(tl.int32)
                for j in range(tl.cdiv(H * N, 1024)):
                    off = j * 1024 + v
                    address = row * H * N + off
                    tl.store(
                        O + address,
                        tl.load(B + address, mask=off < H * N, other=0),
                        mask=off < H * N,
                    )
            for p in range(lim, S):
                row = tl.load(PERM + (0 + p) * PERM_STRIDE).to(tl.int32)
                for j in range(tl.cdiv(H * N, 1024)):
                    off = j * 1024 + v
                    address = row * H * N + off
                    tl.store(
                        O + address,
                        tl.load(B + address, mask=off < H * N, other=0),
                        mask=off < H * N,
                    )
    start = tl.load(SEG + (0 + b) * SEG_STRIDE)
    end = tl.load(SEG + (0 + b + 1) * SEG_STRIDE)
    if start == end:
        return
    j = pid // NT
    if j >= end - start:
        return
    row = tl.load(PERM + (0 + start + j) * PERM_STRIDE).to(tl.int32)
    h = tl.arange(0, BH)
    offs_n = pid % NT * BN + tl.arange(0, BN)
    offs_k = tl.arange(0, BK)
    w = tl.load(WIDX + (0 + b) * WIDX_STRIDE)
    sc = tl.where(
        tl.load(RANKS + (0 + w) * RANKS_STRIDE) > 0,
        tl.load(SCALES + (0 + w) * SCALES_STRIDE),
        0.0,
    )
    ap = A + w * AS0 + offs_k[:, None] * AS1 + offs_n[None, :] * AS2
    if tl.load(RANKS + (0 + w) * RANKS_STRIDE) == 0:
        if HEXACT and NEXACT:
            op = row * OS0 + h[:, None] * OS1 + offs_n[None, :] * OS2
            tl.store(O + op, tl.load(B + op))
        else:
            hs = h if HEXACT else tl.minimum(h, H - 1)
            op = row * OS0 + hs[:, None] * OS1 + offs_n[None, :] * OS2
            valid = (h[:, None] < H) & (offs_n[None, :] < N)
            tl.store(
                O + op, tl.load(B + op, mask=valid, other=0.0), mask=valid
            )
        return
    if HEXACT and NEXACT and KEXACT:
        xp = X + row * XS0 + h[:, None] * XS1 + offs_k[None, :] * XS2
        op = row * OS0 + h[:, None] * OS1 + offs_n[None, :] * OS2
        x = tl.load(xp)
        a = tl.load(ap)
        acc = (
            tl.dot(
                x,
                a,
                input_precision=(
                    "ieee" if X.dtype.element_ty == tl.float32 else "tf32"
                ),
            )
            * sc
        )
        base = tl.load(B + op).to(tl.float32)
        tl.store(O + op, (base + acc).to(O.dtype.element_ty))
    else:
        hs = h if HEXACT else tl.minimum(h, H - 1)
        hok = h < H
        nok = offs_n < N
        kok = offs_k < R
        xp = X + row * XS0 + hs[:, None] * XS1 + offs_k[None, :] * XS2
        op = row * OS0 + hs[:, None] * OS1 + offs_n[None, :] * OS2
        x = tl.load(xp, mask=hok[:, None] & kok[None, :], other=0.0)
        a = tl.load(ap, mask=kok[:, None] & nok[None, :], other=0.0)
        acc = (
            tl.dot(
                x,
                a,
                input_precision=(
                    "ieee" if X.dtype.element_ty == tl.float32 else "tf32"
                ),
            )
            * sc
        )
        omask = hok[:, None] & nok[None, :]
        base = tl.load(B + op, mask=omask, other=0.0).to(tl.float32)
        tl.store(O + op, (base + acc).to(O.dtype.element_ty), mask=omask)


@triton.jit
def _row_gemv_strided(
    X,
    A,
    O,
    SEG,
    WI,
    RANKS,
    SCALES,
    PERM,
    H: tl.constexpr,
    R: tl.constexpr,
    N: tl.constexpr,
    BK: tl.constexpr,
    HAS_PERM: tl.constexpr,
    SEG_STRIDE: tl.constexpr,
    WIDX_STRIDE: tl.constexpr,
    RANKS_STRIDE: tl.constexpr,
    SCALES_STRIDE: tl.constexpr,
    PERM_STRIDE: tl.constexpr,
):
    m = tl.program_id(0)
    h = tl.program_id(1)
    b = tl.program_id(2) // N
    n = tl.program_id(2) % N
    lo = tl.load(SEG + (0 + b) * SEG_STRIDE)
    hi = tl.load(SEG + (0 + b + 1) * SEG_STRIDE)
    if lo + m >= hi:
        return
    w = tl.load(WI + (0 + b) * WIDX_STRIDE)
    if tl.load(RANKS + (0 + w) * RANKS_STRIDE) == 0:
        return
    if HAS_PERM:
        row = tl.load(PERM + (0 + lo + m) * PERM_STRIDE).to(tl.int32)
    else:
        row = lo + m
    k = tl.arange(0, BK)
    x = tl.load(X + (row * H + h) * R + k, mask=k < R, other=0.0).to(
        tl.float32
    )
    a = tl.load(A + w * R * N + k * N + n, mask=k < R, other=0.0).to(
        tl.float32
    )
    correction = tl.sum(x * a, 0) * tl.load(SCALES + (0 + w) * SCALES_STRIDE)
    ptr = O + (row * H + h) * N + n
    base = tl.load(ptr).to(tl.float32)
    tl.store(ptr, base + correction)


@triton.jit
def _bounded_head_strided(
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
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    SEG_STRIDE: tl.constexpr,
    WIDX_STRIDE: tl.constexpr,
    RANKS_STRIDE: tl.constexpr,
    SCALES_STRIDE: tl.constexpr,
    PERM_STRIDE: tl.constexpr,
):
    b = tl.program_id(1)
    lo = tl.load(SEG + (0 + b) * SEG_STRIDE)
    hi = tl.load(SEG + (0 + b + 1) * SEG_STRIDE)
    p = lo + tl.program_id(0) // tl.cdiv(H, BM)
    if p >= hi or p >= PM:
        return
    w = tl.load(WIDX + (0 + b) * WIDX_STRIDE)
    if tl.load(RANKS + (0 + w) * RANKS_STRIDE) == 0:
        return
    if HAS_PERM:
        row = tl.load(PERM + (0 + p) * PERM_STRIDE)
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
    value = base + acc * tl.load(SCALES + (0 + w) * SCALES_STRIDE)
    tl.store(O + off, value.to(O.dtype.element_ty), mask=mask)


def _strided_metadata_entry(q_lora_a, A_buf, batch_info, base_output):
    seg = batch_info.seg_indptr
    widx = batch_info.weight_indices
    ranks = batch_info.lora_ranks
    scales = batch_info.scalings
    perm = batch_info.permutation
    if perm is not None:
        perm = perm
    if not q_lora_a.is_contiguous():
        q_lora_a = q_lora_a.contiguous()
    if not A_buf.is_contiguous():
        A_buf = A_buf.contiguous()
    if not base_output.is_contiguous():
        base_output = base_output.contiguous()
    S, H, R = q_lora_a.shape
    original_h = H
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
        return out[:, :original_h, :] if H != original_h else out
    bs = batch_info.bs
    xs = q_lora_a.stride()
    os = base_output.stride()
    if R > 128:
        q_lora_a = q_lora_a.contiguous()
        A_buf = A_buf.contiguous()
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
        pm = S if perm is None else perm.numel()
        longest = batch_info.max_len or S
        if pm:
            _bounded_head_strided.run(
                q_lora_a,
                A_buf,
                base_output,
                out,
                seg,
                widx,
                ranks,
                scales,
                seg if perm is None else perm,
                H,
                R,
                N,
                pm,
                perm is not None,
                32,
                64,
                64,
                seg.stride(0),
                widx.stride(0),
                ranks.stride(0),
                scales.stride(0),
                0 if perm is None else perm.stride(0),
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
    bk = max(16, 0 if R == 0 else 1 << (R - 1).bit_length())
    pad_h = (
        max(16, 0 if H == 0 else 1 << (H - 1).bit_length())
        if perm is not None
        else H
    )
    if (
        N >= 32
        and bk <= 128
        and (perm is None or pad_h <= 128)
        and (R != bk or H != pad_h)
    ):
        packed_q = torch.zeros(
            (S, pad_h, bk), dtype=q_lora_a.dtype, device=q_lora_a.device
        )
        packed_b = (
            torch.zeros(
                (S, pad_h, N),
                dtype=base_output.dtype,
                device=base_output.device,
            )
            if pad_h != H
            else base_output
        )
        _pad_rows.run(
            q_lora_a,
            base_output,
            packed_q,
            packed_b,
            H,
            R,
            N,
            pad_h,
            bk,
            pad_h != H,
            num_stages=1,
            grid=(S * H, (N + 256 - 1) // 256 if pad_h != H else 1),
            warmup=False,
        )
        if R != bk:
            packed_a = torch.zeros(
                (adapter_count, bk, N), dtype=A_buf.dtype, device=A_buf.device
            )
            _pad_a.run(
                A_buf,
                packed_a,
                R,
                N,
                bk,
                num_stages=1,
                grid=(adapter_count, (R * N + 256 - 1) // 256),
                warmup=False,
            )
        else:
            packed_a = A_buf
        q_lora_a, A_buf, base_output = (packed_q, packed_a, packed_b)
        H, R = (pad_h, bk)
        xs = q_lora_a.stride()
        os = base_output.stride()
    wide_tile = N % 128 == 0
    bn = 128 if wide_tile else 32
    nt = (N + bn - 1) // bn
    nexact = wide_tile or N % 32 == 0
    kexact = bk == R
    longest = batch_info.max_len if batch_info.max_len is not None else S
    has_perm = perm is not None
    full_perm = has_perm and perm.numel() == S
    out = (
        base_output.clone()
        if has_perm and (not full_perm)
        else torch.empty_like(base_output)
    )
    if N < 32 or not kexact or (has_perm and H != pad_h):
        out = base_output.clone()
        if longest == 0:
            return out[:, :original_h, :] if H != original_h else out
        _row_gemv_strided.run(
            q_lora_a,
            A_buf,
            out,
            seg,
            widx,
            ranks,
            scales,
            perm if has_perm else seg,
            H,
            R,
            N,
            bk,
            has_perm,
            seg.stride(0),
            widx.stride(0),
            ranks.stride(0),
            scales.stride(0),
            0 if perm is None else perm.stride(0),
            num_stages=1,
            grid=(longest, H, bs * N),
            warmup=False,
        )
        return out[:, :original_h, :] if H != original_h else out
    if has_perm:
        bh = max(16, 0 if H == 0 else 1 << (H - 1).bit_length())
        _permseg_strided.run(
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
            nt,
            bs,
            S,
            full_perm,
            *xs,
            *A_buf.stride(),
            *os,
            bh,
            bn,
            bk,
            bh == H,
            nexact,
            kexact,
            seg.stride(0),
            widx.stride(0),
            ranks.stride(0),
            scales.stride(0),
            0 if perm is None else perm.stride(0),
            num_stages=1,
            grid=(max(1, longest) * nt, bs),
            warmup=False,
        )
        return out[:, :original_h, :] if H != original_h else out
    bm, ml = (512, 2)
    _seg_strided.run(
        q_lora_a,
        A_buf,
        base_output,
        out,
        seg,
        widx,
        ranks,
        scales,
        H,
        R,
        N,
        nt,
        S,
        bs,
        xs[1],
        xs[2],
        *A_buf.stride(),
        os[1],
        os[2],
        bm,
        bn,
        bk,
        ml,
        nexact,
        kexact,
        seg.stride(0),
        widx.stride(0),
        ranks.stride(0),
        scales.stride(0),
        0 if perm is None else perm.stride(0),
        num_stages=1,
        grid=(max(1, (longest * H + bm * ml - 1) // (bm * ml)) * nt, bs),
        warmup=False,
    )
    return out[:, :original_h, :] if H != original_h else out


@triton.jit
def _pack_small_n(
    A,
    B,
    AP,
    BP,
    H: tl.constexpr,
    R: tl.constexpr,
    N: tl.constexpr,
    ATOTAL: tl.constexpr,
    BTOTAL: tl.constexpr,
    ABLOCKS: tl.constexpr,
    AS0: tl.constexpr,
    AS1: tl.constexpr,
    AS2: tl.constexpr,
    BS0: tl.constexpr,
    BS1: tl.constexpr,
    BS2: tl.constexpr,
):
    pid = tl.program_id(0)
    lane = tl.arange(0, 1024)
    if pid < ABLOCKS:
        i = pid * 1024 + lane
        n = i % 32
        valid = (i < ATOTAL) & (n < N)
        value = tl.load(A + i // 32 * N + n, mask=valid, other=0)
        tl.store(AP + i, value, mask=i < ATOTAL)
    else:
        i = (pid - ABLOCKS) * 1024 + lane
        n = i % 32
        valid = (i < BTOTAL) & (n < N)
        value = tl.load(B + i // 32 * N + n, mask=valid, other=0)
        tl.store(BP + i, value, mask=i < BTOTAL)


def _original_entry(q_lora_a, A_buf, batch_info, base_output):
    S, H, R = q_lora_a.shape
    adapters, _, N = A_buf.shape
    perm = batch_info.permutation
    if (
        16 <= N < 32
        and 0 < R <= 128
        and (H > 128)
        and (S > 0)
        and (adapters > 0)
        and (batch_info.bs > 0)
        and (perm is not None)
        and (perm.numel() > 0)
    ):
        return _headpad_entry(q_lora_a, A_buf, batch_info, base_output)
    if (
        16 <= N < 32
        and 0 < R <= 128
        and (S * H > 0)
        and (adapters > 0)
        and (batch_info.bs > 0)
        and (
            perm is None
            or (
                perm.numel() > 0
                and (0 if H == 0 else 1 << (H - 1).bit_length()) <= 128
            )
        )
    ):
        A_buf = A_buf.contiguous()
        base_output = base_output.contiguous()
        ap = torch.zeros(
            (adapters, R, 32), dtype=A_buf.dtype, device=A_buf.device
        )
        bp = torch.zeros(
            (S, H, 32), dtype=base_output.dtype, device=base_output.device
        )
        blocks_a = (adapters * R * 32 + 1024 - 1) // 1024
        _pack_small_n.run(
            A_buf,
            base_output,
            ap,
            bp,
            H,
            R,
            N,
            adapters * R * 32,
            S * H * 32,
            blocks_a,
            *A_buf.stride(),
            *base_output.stride(),
            num_warps=4,
            num_stages=1,
            grid=(blocks_a + (S * H * 32 + 1024 - 1) // 1024,),
            warmup=False,
        )
        return _main_entry(q_lora_a, ap, batch_info, bp)[:, :, :N]
    return _main_entry(q_lora_a, A_buf, batch_info, base_output)


def _headpad_entry(q_lora_a, A_buf, batch_info, base_output):
    S, H, R = q_lora_a.shape
    adapters, _, N = A_buf.shape
    hp = (H + 32 - 1) // 32 * 32
    rp = max(64, 0 if R == 0 else 1 << (R - 1).bit_length())
    q = q_lora_a.contiguous()
    a = A_buf.contiguous()
    b = base_output.contiguous()
    apn = torch.zeros((adapters, R, 32), dtype=a.dtype, device=a.device)
    bpn = torch.zeros((S, H, 32), dtype=b.dtype, device=b.device)
    ab = (adapters * R * 32 + 1024 - 1) // 1024
    _pack_small_n.run(
        a,
        b,
        apn,
        bpn,
        H,
        R,
        N,
        adapters * R * 32,
        S * H * 32,
        ab,
        *a.stride(),
        *b.stride(),
        num_warps=4,
        num_stages=1,
        grid=(ab + (S * H * 32 + 1024 - 1) // 1024,),
        warmup=False,
    )
    qp = torch.zeros((S, hp, rp), dtype=q.dtype, device=q.device)
    ap = torch.zeros((adapters, rp, 32), dtype=a.dtype, device=a.device)
    bp = torch.zeros((S, hp, 32), dtype=b.dtype, device=b.device)
    _pad_rows.run(
        q,
        bpn,
        qp,
        bp,
        H,
        R,
        32,
        hp,
        rp,
        True,
        num_stages=1,
        grid=(S * H, 1),
        warmup=False,
    )
    _pad_a.run(
        apn,
        ap,
        R,
        32,
        rp,
        num_stages=1,
        grid=(adapters, (R * 32 + 256 - 1) // 256),
        warmup=False,
    )
    seg = batch_info.seg_indptr
    widx = batch_info.weight_indices
    ranks = batch_info.lora_ranks
    scales = batch_info.scalings
    perm = batch_info.permutation
    longest = batch_info.max_len or S
    _bounded_head_strided.run(
        qp,
        ap,
        bp,
        bp,
        seg,
        widx,
        ranks,
        scales,
        perm,
        hp,
        rp,
        32,
        perm.numel(),
        True,
        32,
        32,
        64,
        seg.stride(0),
        widx.stride(0),
        ranks.stride(0),
        scales.stride(0),
        perm.stride(0),
        num_warps=4,
        num_stages=1,
        grid=(longest * ((hp + 32 - 1) // 32), batch_info.bs, 1),
        warmup=False,
    )
    return bp[:, :H, :N]


@triton.jit
def _seg_nloop(X, A, B, OUT, SEG, WIDX, RANKS, SCALES, META: tl.constexpr):
    H: tl.constexpr = META[0]
    R: tl.constexpr = META[1]
    N: tl.constexpr = META[2]
    S: tl.constexpr = META[3]
    BS: tl.constexpr = META[4]
    BM: tl.constexpr = META[5]
    BN: tl.constexpr = META[6]
    BK: tl.constexpr = META[7]
    b = tl.program_id(1)
    pid = tl.program_id(0)
    if (b == 0) & (pid == 0):
        first_count = tl.load(SEG) * H * N
        v = tl.arange(0, 1024)
        for j in range(tl.cdiv(first_count, 1024)):
            off = j * 1024 + v
            tl.store(
                OUT + off,
                tl.load(B + off, mask=off < first_count, other=0),
                mask=off < first_count,
            )
        lim = tl.load(SEG + BS) * H * N
        for j in range(tl.cdiv(S * H * N - lim, 1024)):
            off = lim + j * 1024 + v
            tl.store(
                OUT + off,
                tl.load(B + off, mask=off < S * H * N, other=0),
                mask=off < S * H * N,
            )
    start = tl.load(SEG + b)
    end = tl.load(SEG + b + 1)
    if start == end:
        return
    seglen = (end - start) * H
    m0 = pid * BM
    if m0 >= seglen:
        return
    m = m0 + tl.arange(0, BM)
    pos = start * H + m
    k = tl.arange(0, BK)
    w = tl.load(WIDX + b)
    if tl.load(RANKS + w) == 0:
        for tile_n in range(tl.cdiv(N, BN)):
            n = tile_n * BN + tl.arange(0, BN)
            off = pos[:, None] * N + n[None, :]
            valid = (m[:, None] < seglen) & (n[None, :] < N)
            tl.store(
                OUT + off, tl.load(B + off, mask=valid, other=0.0), mask=valid
            )
        return
    x = tl.load(
        X + pos[:, None] * R + k[None, :],
        mask=(m[:, None] < seglen) & (k[None, :] < R),
        other=0.0,
    )
    scale = tl.load(SCALES + w)
    for tile_n in range(tl.cdiv(N, BN)):
        n = tile_n * BN + tl.arange(0, BN)
        a = tl.load(
            A + w * R * N + k[:, None] * N + n[None, :],
            mask=(k[:, None] < R) & (n[None, :] < N),
            other=0.0,
        )
        acc = (
            tl.dot(
                x,
                a,
                input_precision=(
                    "ieee" if X.dtype.element_ty == tl.float32 else "tf32"
                ),
            )
            * scale
        )
        off = pos[:, None] * N + n[None, :]
        valid = (m[:, None] < seglen) & (n[None, :] < N)
        base = tl.load(B + off, mask=valid, other=0.0).to(tl.float32)
        tl.store(OUT + off, (base + acc).to(OUT.dtype.element_ty), mask=valid)


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
    seg, widx = (batch_info.seg_indptr, batch_info.weight_indices)
    ranks, scales = (batch_info.lora_ranks, batch_info.scalings)
    if (
        batch_info.permutation is None
        and S > 0
        and (H > 0)
        and (R in (16, 32, 64, 128))
        and (N >= 128)
        and (N % 128 == 0)
        and (A_buf.shape[0] > 0)
        and (batch_info.bs > 0)
        and q_lora_a.is_contiguous()
        and A_buf.is_contiguous()
        and base_output.is_contiguous()
        and seg.is_contiguous()
        and widx.is_contiguous()
        and ranks.is_contiguous()
        and scales.is_contiguous()
    ):
        out = torch.empty_like(base_output)
        bound = batch_info.max_len if batch_info.max_len is not None else S
        _seg_nloop.run(
            q_lora_a,
            A_buf,
            base_output,
            out,
            seg,
            widx,
            ranks,
            scales,
            (H, R, N, S, batch_info.bs, 512, 128, R),
            num_stages=1,
            grid=(max(1, (bound * H + 512 - 1) // 512), batch_info.bs),
            warmup=False,
        )
        return out
    perm = batch_info.permutation
    if (
        0 < S * H <= 256
        and H in (16, 32, 64, 128)
        and (R in (16, 32))
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
            base_output if perm is None else perm,
            (H, R, N, batch_info.bs, perm is not None, H, R, 64),
            num_warps=4,
            num_stages=1,
            grid=(S, (N + 64 - 1) // 64),
            warmup=False,
        )
        return out
    return _original_entry(q_lora_a, A_buf, batch_info, base_output)


__all__ = ["step_b_q"]
