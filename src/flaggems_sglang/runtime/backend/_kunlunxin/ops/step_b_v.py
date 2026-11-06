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
def _step_b_v_kernel(X, W, OUT, SEG, WIDX, RANKS, SCALES, META: tl.constexpr):
    N: tl.constexpr = META[1]
    K: tl.constexpr = META[2]
    F: tl.constexpr = META[3]
    Q: tl.constexpr = META[4]
    XS0: tl.constexpr = META[5]
    XS1: tl.constexpr = META[6]
    XS2: tl.constexpr = META[7]
    WS0: tl.constexpr = META[8]
    WS1: tl.constexpr = META[9]
    WS2: tl.constexpr = META[10]
    GS0: tl.constexpr = META[11]
    GS1: tl.constexpr = META[12]
    GS2: tl.constexpr = META[13]
    BM: tl.constexpr = META[14]
    BN: tl.constexpr = META[15]
    BK: tl.constexpr = META[16]
    pid_s = tl.program_id(0)
    h = tl.program_id(1)
    b = tl.program_id(2)
    lo = tl.load(SEG + b)
    hi = tl.load(SEG + b + 1)
    if hi == lo:
        return
    w = tl.load(WIDX + b)
    if tl.load(RANKS + w) == 0:
        return
    seg_len = hi - lo
    i0 = pid_s * BM
    if i0 >= seg_len:
        return
    s_off = i0 + tl.arange(0, BM)
    offs_n = tl.arange(0, BN)
    offs_k = tl.arange(0, BK)
    rmask = s_off < seg_len
    nmask = offs_n < N
    kmask = offs_k < K
    rows = lo + s_off
    x = tl.load(
        X + rows[:, None] * XS0 + h * XS1 + offs_k[None, :] * XS2,
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
    acc = tl.dot(
        x, wb, input_precision="ieee" if x.dtype == tl.float32 else "tf32"
    ) * tl.load(SCALES + w).to(tl.float32)
    m2 = rmask[:, None] & nmask[None, :]
    op = OUT + rows[:, None] * GS0 + h * GS1 + offs_n[None, :] * GS2
    base = tl.load(op, mask=m2, other=0.0).to(tl.float32)
    tl.store(op, (base + acc).to(OUT.dtype.element_ty), mask=m2)


@triton.jit
def _pack_kernel(X, PERM, XC, DC, META: tl.constexpr):
    H: tl.constexpr = META[0]
    K: tl.constexpr = META[1]
    N: tl.constexpr = META[2]
    PT: tl.constexpr = META[3]
    PC: tl.constexpr = META[4]
    XS0: tl.constexpr = META[5]
    XS1: tl.constexpr = META[6]
    XS2: tl.constexpr = META[7]
    BK: tl.constexpr = META[8]
    pid = tl.program_id(0)
    work: tl.constexpr = PC * H
    if pid < work:
        i = pid // H
        h = pid % H
        row = tl.load(PERM + i, i < PT, other=0)
        k = tl.arange(0, BK)
        x = tl.load(
            X + row * XS0 + h * XS1 + k * XS2, (i < PT) & (k < K), other=0.0
        )
        tl.store(XC + (h * PC + i) * K + k, x, k < K)
    else:
        idx = (pid - work) * 1024 + tl.arange(0, 1024)
        tl.store(DC + idx, tl.full((1024,), 0.0, tl.float32), idx < PC * H * N)


@triton.jit
def _scatter_kernel(BASE, OUT, DC, PERM, META: tl.constexpr):
    H: tl.constexpr = META[0]
    N: tl.constexpr = META[1]
    GS0: tl.constexpr = META[2]
    GS1: tl.constexpr = META[3]
    GS2: tl.constexpr = META[4]
    BN: tl.constexpr = META[5]
    PC: tl.constexpr = META[6]
    i = tl.program_id(0)
    h = tl.program_id(1)
    row = tl.load(PERM + i)
    n = tl.arange(0, BN)
    delta = tl.load(DC + (h * PC + i) * N + n, n < N, other=0.0)
    bp = BASE + row * GS0 + h * GS1 + n * GS2
    base = tl.load(bp, n < N, other=0.0).to(tl.float32)
    op = OUT + (row * H + h) * N + n
    tl.store(op, (base + delta).to(OUT.dtype.element_ty), n < N)


@triton.jit
def _dot_compact_kernel(
    XC, W, DC, SEG, WIDX, RANKS, SCALES, META: tl.constexpr
):
    PC: tl.constexpr = META[0]
    N: tl.constexpr = META[2]
    K: tl.constexpr = META[3]
    F: tl.constexpr = META[4]
    Q: tl.constexpr = META[5]
    WS0: tl.constexpr = META[6]
    WS1: tl.constexpr = META[7]
    WS2: tl.constexpr = META[8]
    BM: tl.constexpr = META[9]
    BN: tl.constexpr = META[10]
    BK: tl.constexpr = META[11]
    pid_s = tl.program_id(0)
    h = tl.program_id(1)
    b = tl.program_id(2)
    lo = tl.load(SEG + b)
    hi = tl.load(SEG + b + 1)
    if hi == lo:
        return
    w = tl.load(WIDX + b)
    if tl.load(RANKS + w) == 0:
        return
    seg_len = hi - lo
    i0 = pid_s * BM
    if i0 >= seg_len:
        return
    s_off = i0 + tl.arange(0, BM)
    offs_n = tl.arange(0, BN)
    offs_k = tl.arange(0, BK)
    rmask = s_off < seg_len
    nmask = offs_n < N
    kmask = offs_k < K
    pos = lo + s_off
    x = tl.load(
        XC + (h * PC + pos[:, None]) * K + offs_k[None, :],
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
    acc = tl.dot(
        x, wb, input_precision="ieee" if x.dtype == tl.float32 else "tf32"
    ) * tl.load(SCALES + w).to(tl.float32)
    tl.store(
        DC + (h * PC + pos[:, None]) * N + offs_n[None, :],
        acc,
        mask=rmask[:, None] & nmask[None, :],
    )


@triton.jit
def _row_gemv(X, W, OUT, SEG, WI, RANKS, SCALES, PERM, META: tl.constexpr):
    H: tl.constexpr = META[0]
    N: tl.constexpr = META[1]
    K: tl.constexpr = META[2]
    F: tl.constexpr = META[3]
    Q: tl.constexpr = META[4]
    X0: tl.constexpr = META[5]
    X1: tl.constexpr = META[6]
    X2: tl.constexpr = META[7]
    W0: tl.constexpr = META[8]
    W1: tl.constexpr = META[9]
    W2: tl.constexpr = META[10]
    PS: tl.constexpr = META[11]
    BK: tl.constexpr = META[12]
    HAS_PERM: tl.constexpr = META[13]
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
        row = tl.load(PERM + (lo + m) * PS).to(tl.int32)
    else:
        row = lo + m
    k = tl.arange(0, BK)
    x = tl.load(X + row * X0 + h * X1 + k * X2, k < K, 0.0).to(tl.float32)
    wt = tl.load(W + w * W0 + (h * F + Q + n) * W1 + k * W2, k < K, 0.0).to(
        tl.float32
    )
    correction = tl.sum(x * wt, 0) * tl.load(SCALES + w)
    ptr = OUT + (row * H + h) * N + n
    base = tl.load(ptr).to(tl.float32)
    tl.store(ptr, base + correction)


@triton.jit
def _copy_base_kernel(BASE, OUT, META: tl.constexpr):
    TOTAL: tl.constexpr = META[0]
    BLOCK: tl.constexpr = META[1]
    if TOTAL == 0:
        return
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    value = tl.load(BASE + i, i < TOTAL, other=0.0).to(tl.float32)
    tl.store(OUT + i, value, i < TOTAL)


def _unfused_step_b_v(
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
    if B_buf.stride(-1) != 1:
        B_buf = B_buf.contiguous()
    S, H, K = attn_lora_a.shape
    N = v_head_dim
    F = qk_nope_head_dim + N
    perm = batch_info.permutation
    if perm is not None:
        perm = perm.contiguous()
    bn = max(16, 0 if N == 0 else 1 << (N - 1).bit_length())
    bk = max(16, 0 if K == 0 else 1 << (K - 1).bit_length())
    ml = batch_info.max_len
    if not ml:
        ml = S
    if (
        perm is not None
        and S <= 32
        and (ml <= S)
        and (0 < B_buf.shape[0] <= 4)
        and (K in (16, 32))
        and (N in (16, 32, 64, 128))
        and (H & H - 1 == 0)
        and (attn_lora_a.dtype != torch.float32)
    ):
        sp = max(16, 1 << (S - 1).bit_length())
        correction = torch.empty(
            (B_buf.shape[0], sp, H * N),
            dtype=torch.float32,
            device=attn_lora_a.device,
        )
        full = perm.numel() == S
        out = (
            base_output.new_empty((S, H, N))
            if full
            else base_output.clone(memory_format=torch.contiguous_format)
        )
        seg = batch_info.seg_indptr.contiguous()
        wi = batch_info.weight_indices.contiguous()
        ranks = batch_info.lora_ranks.contiguous()
        scales = batch_info.scalings.contiguous()
        _perm_all_rows.run(
            attn_lora_a,
            B_buf,
            correction,
            ranks,
            scales,
            S,
            (
                H,
                K,
                N,
                F,
                qk_nope_head_dim,
                sp,
                *attn_lora_a.stride(),
                *B_buf.stride(),
            ),
            num_warps=4,
            num_stages=1,
            grid=(H, B_buf.shape[0]),
            warmup=False,
        )
        _perm_choose.run(
            correction,
            base_output,
            seg,
            wi,
            ranks,
            perm,
            out,
            (H, N, sp, S, batch_info.bs, full, H * N, *base_output.stride()),
            num_warps=4,
            num_stages=1,
            grid=(S if full else ml, batch_info.bs + int(full)),
            warmup=False,
        )
        return out
    if N != bn or N < 16 or K < 16:
        out = base_output.clone(memory_format=torch.contiguous_format)
        p = perm if perm is not None else batch_info.seg_indptr
        _row_gemv.run(
            attn_lora_a,
            B_buf,
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
                F,
                qk_nope_head_dim,
                *attn_lora_a.stride(),
                *B_buf.stride(),
                p.stride(0),
                bk,
                perm is not None,
            ),
            grid=(ml, H, batch_info.bs * N),
            warmup=False,
        )
        return out
    bm = min(256, max(16, 0 if ml == 0 else 1 << (ml - 1).bit_length()))
    out = base_output.new_empty((S + bm, H, N))
    if perm is None or perm.numel() != S:
        out[:S].copy_(base_output)
    if perm is None:
        _step_b_v_kernel.run(
            attn_lora_a,
            B_buf,
            out,
            batch_info.seg_indptr.contiguous(),
            batch_info.weight_indices.contiguous(),
            batch_info.lora_ranks.contiguous(),
            batch_info.scalings.contiguous(),
            (
                H,
                N,
                K,
                F,
                qk_nope_head_dim,
                *attn_lora_a.stride(),
                *B_buf.stride(),
                *out.stride(),
                bm,
                bn,
                bk,
            ),
            grid=((ml + bm - 1) // bm, H, batch_info.bs),
            warmup=False,
        )
        return out[:S]
    rows_total = perm.numel()
    xc = attn_lora_a.new_empty((H, rows_total + bm, K))
    dc = torch.empty(
        (H, rows_total + bm, N), dtype=torch.float32, device=attn_lora_a.device
    )
    _pack_kernel.run(
        attn_lora_a,
        perm,
        xc,
        dc,
        (H, K, N, rows_total, rows_total + bm, *attn_lora_a.stride(), bk, bn),
        grid=(
            (rows_total + bm) * H
            + ((rows_total + bm) * H * N + 1024 - 1) // 1024,
        ),
        warmup=False,
    )
    _dot_compact_kernel.run(
        xc,
        B_buf,
        dc,
        batch_info.seg_indptr.contiguous(),
        batch_info.weight_indices.contiguous(),
        batch_info.lora_ranks.contiguous(),
        batch_info.scalings.contiguous(),
        (
            rows_total + bm,
            H,
            N,
            K,
            F,
            qk_nope_head_dim,
            *B_buf.stride(),
            bm,
            bn,
            bk,
        ),
        grid=((ml + bm - 1) // bm, H, batch_info.bs),
        warmup=False,
    )
    _scatter_kernel.run(
        base_output,
        out,
        dc,
        perm,
        (H, N, *base_output.stride(), bn, rows_total + bm),
        grid=(rows_total, H),
        warmup=False,
    )
    return out[:S]


@triton.jit
def _fused_nonperm_control(
    X, W, BASE, OUT, SEG, WIDX, RANKS, SCALES, META: tl.constexpr
):
    H: tl.constexpr = META[0]
    N: tl.constexpr = META[1]
    S: tl.constexpr = META[2]
    K: tl.constexpr = META[3]
    F: tl.constexpr = META[4]
    Q: tl.constexpr = META[5]
    BS: tl.constexpr = META[6]
    TILES: tl.constexpr = META[7]
    COVERS: tl.constexpr = META[8]
    XS0: tl.constexpr = META[9]
    XS1: tl.constexpr = META[10]
    XS2: tl.constexpr = META[11]
    WS0: tl.constexpr = META[12]
    WS1: tl.constexpr = META[13]
    WS2: tl.constexpr = META[14]
    BM: tl.constexpr = META[15]
    BN: tl.constexpr = META[16]
    BK: tl.constexpr = META[17]
    pid = tl.program_id(0)
    work = TILES * H * BS
    if pid >= work:
        cpid = pid - work
        offsets = tl.arange(0, 1024)
        first = tl.load(SEG) * H * N
        last = tl.load(SEG + BS) * H * N
        for start in range(cpid * 1024, first, COVERS * 1024):
            idx = start + offsets
            old_prefix = tl.load(BASE + idx, idx < first, other=0.0)
            tl.store(OUT + idx, old_prefix, idx < first)
        for start in range(last + cpid * 1024, S * H * N, COVERS * 1024):
            idx = start + offsets
            old_suffix = tl.load(BASE + idx, idx < S * H * N, other=0.0)
            tl.store(OUT + idx, old_suffix, idx < S * H * N)
        return
    pid_s = pid % TILES
    h = pid // TILES % H
    b = pid // (TILES * H)
    lo = tl.load(SEG + b)
    hi = tl.load(SEG + b + 1)
    if hi == lo:
        return
    i0 = pid_s * BM
    if i0 >= hi - lo:
        return
    w = tl.load(WIDX + b)
    if tl.load(RANKS + w) == 0:
        cn = tl.arange(0, BN)
        for cr in range(lo + i0, tl.minimum(hi, lo + i0 + BM)):
            co = (cr * H + h) * N + cn
            old_inactive = tl.load(BASE + co, cn < N, other=0.0)
            tl.store(OUT + co, old_inactive, cn < N)
        return
    s_off = i0 + tl.arange(0, BM)
    offs_n = tl.arange(0, BN)
    offs_k = tl.arange(0, BK)
    rmask = s_off < hi - lo
    nmask = offs_n < N
    kmask = offs_k < K
    rows = lo + s_off
    x = tl.load(
        X + rows[:, None] * XS0 + h * XS1 + offs_k[None, :] * XS2,
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
    acc = tl.dot(
        x, wb, input_precision="ieee" if x.dtype == tl.float32 else "tf32"
    ) * tl.load(SCALES + w).to(tl.float32)
    m2 = rmask[:, None] & nmask[None, :]
    off = (rows[:, None] * H + h) * N + offs_n[None, :]
    base = tl.load(BASE + off, mask=m2, other=0.0).to(tl.float32)
    tl.store(OUT + off, (base + acc).to(OUT.dtype.element_ty), mask=m2)


@triton.jit
def _fused_nonperm_flat(
    X, W, BASE, OUT, SEG, WIDX, RANKS, SCALES, META: tl.constexpr
):
    H: tl.constexpr = META[0]
    N: tl.constexpr = META[1]
    S: tl.constexpr = META[2]
    K: tl.constexpr = META[3]
    F: tl.constexpr = META[4]
    Q: tl.constexpr = META[5]
    BS: tl.constexpr = META[6]
    TILES: tl.constexpr = META[7]
    COVERS: tl.constexpr = META[8]
    XS0: tl.constexpr = META[9]
    XS1: tl.constexpr = META[10]
    XS2: tl.constexpr = META[11]
    WS0: tl.constexpr = META[12]
    WS1: tl.constexpr = META[13]
    WS2: tl.constexpr = META[14]
    BM: tl.constexpr = META[15]
    BN: tl.constexpr = META[16]
    BK: tl.constexpr = META[17]
    ML: tl.constexpr = META[18]
    pid = tl.program_id(0)
    work = TILES * H * BS
    if pid >= work:
        cpid = pid - work
        offsets = tl.arange(0, 1024)
        first = tl.load(SEG) * H * N
        last = tl.load(SEG + BS) * H * N
        for start in range(cpid * 1024, first, COVERS * 1024):
            idx = start + offsets
            old_prefix = tl.load(BASE + idx, idx < first, other=0.0)
            tl.store(OUT + idx, old_prefix, idx < first)
        for start in range(last + cpid * 1024, S * H * N, COVERS * 1024):
            idx = start + offsets
            old_suffix = tl.load(BASE + idx, idx < S * H * N, other=0.0)
            tl.store(OUT + idx, old_suffix, idx < S * H * N)
        return
    pid_s = pid % TILES
    h = pid // TILES % H
    b = pid // (TILES * H)
    lo = tl.load(SEG + b)
    hi = tl.load(SEG + b + 1)
    if hi == lo:
        return
    i0 = pid_s * BM * ML
    if i0 >= hi - lo:
        return
    w = tl.load(WIDX + b)
    if tl.load(RANKS + w) == 0:
        cn = tl.arange(0, BN)
        for cr in range(lo + i0, tl.minimum(hi, lo + i0 + BM * ML)):
            co = (cr * H + h) * N + cn
            old_inactive = tl.load(BASE + co, cn < N, other=0.0)
            tl.store(OUT + co, old_inactive, cn < N)
        return
    offs_n = tl.arange(0, BN)
    offs_k = tl.arange(0, BK)
    nmask = offs_n < N
    kmask = offs_k < K
    wb = tl.load(
        W
        + w * WS0
        + (h * F + Q + offs_n[None, :]) * WS1
        + offs_k[:, None] * WS2,
        mask=kmask[:, None] & nmask[None, :],
        other=0.0,
    )
    scale = tl.load(SCALES + w).to(tl.float32)
    for panel in range(ML):
        s_off = i0 + panel * BM + tl.arange(0, BM)
        rmask = s_off < hi - lo
        rows = lo + s_off
        x = tl.load(
            X + rows[:, None] * XS0 + h * XS1 + offs_k[None, :] * XS2,
            mask=rmask[:, None] & kmask[None, :],
            other=0.0,
        )
        acc = (
            tl.dot(
                x,
                wb,
                input_precision="ieee" if x.dtype == tl.float32 else "tf32",
            )
            * scale
        )
        m2 = rmask[:, None] & nmask[None, :]
        off = (rows[:, None] * H + h) * N + offs_n[None, :]
        base = tl.load(BASE + off, mask=m2, other=0.0).to(tl.float32)
        tl.store(OUT + off, (base + acc).to(OUT.dtype.element_ty), mask=m2)


def _mloop_step_b_v(
    attn_lora_a, B_buf, batch_info, base_output, qk_nope_head_dim, v_head_dim
):
    S, H, K = attn_lora_a.shape
    N = v_head_dim
    if base_output.numel() == 0:
        out = torch.empty_like(base_output)
        _copy_base_kernel.run(
            base_output, out, (0, 1024), grid=(1,), warmup=False
        )
        return out
    bn = max(16, 0 if N == 0 else 1 << (N - 1).bit_length())
    if (
        batch_info.bs == 0
        or batch_info.permutation is not None
        or N != bn
        or (N < 16)
        or (K < 16)
    ):
        return _unfused_step_b_v(
            attn_lora_a,
            B_buf,
            batch_info,
            base_output,
            qk_nope_head_dim,
            v_head_dim,
        )
    if attn_lora_a.stride(-1) != 1:
        attn_lora_a = attn_lora_a.contiguous()
    if B_buf.stride(-1) != 1:
        B_buf = B_buf.contiguous()
    base = base_output.contiguous()
    ml = batch_info.max_len or S
    bm = min(256, max(16, 0 if ml == 0 else 1 << (ml - 1).bit_length()))
    panels = 2 if K == 16 and bm == 256 else 1
    bm //= panels
    tiles = max(1, (ml + bm * panels - 1) // (bm * panels))
    covers = max(1, min(32, (S * H * N + 1048576 - 1) // 1048576))
    out = base.new_empty((S + bm, H, N))
    if panels == 1:
        _fused_nonperm_control.run(
            attn_lora_a,
            B_buf,
            base,
            out,
            batch_info.seg_indptr.contiguous(),
            batch_info.weight_indices.contiguous(),
            batch_info.lora_ranks.contiguous(),
            batch_info.scalings.contiguous(),
            (
                H,
                N,
                S,
                K,
                qk_nope_head_dim + N,
                qk_nope_head_dim,
                batch_info.bs,
                tiles,
                covers,
                *attn_lora_a.stride(),
                *B_buf.stride(),
                bm,
                bn,
                max(16, 0 if K == 0 else 1 << (K - 1).bit_length()),
            ),
            grid=(tiles * H * batch_info.bs + covers,),
            warmup=False,
        )
    else:
        _fused_nonperm_flat.run(
            attn_lora_a,
            B_buf,
            base,
            out,
            batch_info.seg_indptr.contiguous(),
            batch_info.weight_indices.contiguous(),
            batch_info.lora_ranks.contiguous(),
            batch_info.scalings.contiguous(),
            (
                H,
                N,
                S,
                K,
                qk_nope_head_dim + N,
                qk_nope_head_dim,
                batch_info.bs,
                tiles,
                covers,
                *attn_lora_a.stride(),
                *B_buf.stride(),
                bm,
                bn,
                max(16, 0 if K == 0 else 1 << (K - 1).bit_length()),
                panels,
            ),
            grid=(tiles * H * batch_info.bs + covers,),
            warmup=False,
        )
    return out[:S]


@triton.jit
def _fused_adapter(
    X, W, BASE, OUT, SEG, WIDX, RANKS, SCALES, META: tl.constexpr
):
    H: tl.constexpr = META[0]
    N: tl.constexpr = META[1]
    S: tl.constexpr = META[2]
    K: tl.constexpr = META[3]
    F: tl.constexpr = META[4]
    Q: tl.constexpr = META[5]
    BS: tl.constexpr = META[6]
    NA: tl.constexpr = META[7]
    TILES: tl.constexpr = META[8]
    COVERS: tl.constexpr = META[9]
    XS0: tl.constexpr = META[10]
    XS1: tl.constexpr = META[11]
    XS2: tl.constexpr = META[12]
    WS0: tl.constexpr = META[13]
    WS1: tl.constexpr = META[14]
    WS2: tl.constexpr = META[15]
    BM: tl.constexpr = META[16]
    BN: tl.constexpr = META[17]
    BK: tl.constexpr = META[18]
    ML: tl.constexpr = META[19]
    pid = tl.program_id(0)
    work = TILES * H * NA
    if K == 16:
        if pid < COVERS:
            cpid = pid
            offsets = tl.arange(0, 1024)
            first = tl.load(SEG) * H * N
            last = tl.load(SEG + BS) * H * N
            for start in range(cpid * 1024, first, COVERS * 1024):
                idx = start + offsets
                old_prefix = tl.load(BASE + idx, idx < first, other=0.0)
                tl.store(OUT + idx, old_prefix, idx < first)
            for start in range(last + cpid * 1024, S * H * N, COVERS * 1024):
                idx = start + offsets
                old_suffix = tl.load(BASE + idx, idx < S * H * N, other=0.0)
                tl.store(OUT + idx, old_suffix, idx < S * H * N)
        if pid < work:
            pid_s = pid % TILES
            h = pid // TILES % H
            w = pid // (TILES * H)
            i0 = pid_s * BM * ML
            if tl.load(RANKS + w) == 0:
                cn = tl.arange(0, BN)
                for b in range(BS):
                    if tl.load(WIDX + b) == w:
                        lo = tl.load(SEG + b)
                        hi = tl.load(SEG + b + 1)
                        for cr in range(
                            lo + i0, tl.minimum(hi, lo + i0 + BM * ML)
                        ):
                            co = (cr * H + h) * N + cn
                            old_inactive = tl.load(
                                BASE + co, cn < N, other=0.0
                            )
                            tl.store(OUT + co, old_inactive, cn < N)
            else:
                offs_n = tl.arange(0, BN)
                offs_k = tl.arange(0, BK)
                nmask = offs_n < N
                kmask = offs_k < K
                wb = tl.load(
                    W
                    + w * WS0
                    + (h * F + Q + offs_n[None, :]) * WS1
                    + offs_k[:, None] * WS2,
                    mask=kmask[:, None] & nmask[None, :],
                    other=0.0,
                )
                scale = tl.load(SCALES + w).to(tl.float32)
                for b in range(BS):
                    if tl.load(WIDX + b) == w:
                        lo = tl.load(SEG + b)
                        hi = tl.load(SEG + b + 1)
                        if i0 < hi - lo:
                            for panel in range(ML):
                                s_off = i0 + panel * BM + tl.arange(0, BM)
                                rmask = s_off < hi - lo
                                rows = lo + s_off
                                m2 = rmask[:, None] & nmask[None, :]
                                off = (rows[:, None] * H + h) * N + offs_n[
                                    None, :
                                ]
                                base = tl.load(BASE + off, mask=m2, other=0.0)
                                x = tl.load(
                                    X
                                    + rows[:, None] * XS0
                                    + h * XS1
                                    + offs_k[None, :] * XS2,
                                    mask=rmask[:, None] & kmask[None, :],
                                    other=0.0,
                                )
                                acc = (
                                    tl.dot(
                                        x,
                                        wb,
                                        input_precision=(
                                            "ieee"
                                            if x.dtype == tl.float32
                                            else "tf32"
                                        ),
                                    )
                                    * scale
                                )
                                tl.store(
                                    OUT + off,
                                    (base.to(tl.float32) + acc).to(
                                        OUT.dtype.element_ty
                                    ),
                                    mask=m2,
                                )
    elif pid >= work:
        cpid = pid - work
        offsets = tl.arange(0, 1024)
        first = tl.load(SEG) * H * N
        last = tl.load(SEG + BS) * H * N
        for start in range(cpid * 1024, first, COVERS * 1024):
            idx = start + offsets
            old_prefix = tl.load(BASE + idx, idx < first, other=0.0)
            tl.store(OUT + idx, old_prefix, idx < first)
        for start in range(last + cpid * 1024, S * H * N, COVERS * 1024):
            idx = start + offsets
            old_suffix = tl.load(BASE + idx, idx < S * H * N, other=0.0)
            tl.store(OUT + idx, old_suffix, idx < S * H * N)
    else:
        pid_s = pid % TILES
        h = pid // TILES % H
        w = pid // (TILES * H)
        i0 = pid_s * BM * ML
        if tl.load(RANKS + w) == 0:
            cn = tl.arange(0, BN)
            for b in range(BS):
                if tl.load(WIDX + b) == w:
                    lo = tl.load(SEG + b)
                    hi = tl.load(SEG + b + 1)
                    for cr in range(
                        lo + i0, tl.minimum(hi, lo + i0 + BM * ML)
                    ):
                        co = (cr * H + h) * N + cn
                        old_inactive = tl.load(BASE + co, cn < N, other=0.0)
                        tl.store(OUT + co, old_inactive, cn < N)
        else:
            offs_n = tl.arange(0, BN)
            offs_k = tl.arange(0, BK)
            nmask = offs_n < N
            kmask = offs_k < K
            wb = tl.load(
                W
                + w * WS0
                + (h * F + Q + offs_n[None, :]) * WS1
                + offs_k[:, None] * WS2,
                mask=kmask[:, None] & nmask[None, :],
                other=0.0,
            )
            scale = tl.load(SCALES + w).to(tl.float32)
            for b in range(BS):
                if tl.load(WIDX + b) == w:
                    lo = tl.load(SEG + b)
                    hi = tl.load(SEG + b + 1)
                    if i0 < hi - lo:
                        for panel in range(ML):
                            s_off = i0 + panel * BM + tl.arange(0, BM)
                            rmask = s_off < hi - lo
                            rows = lo + s_off
                            m2 = rmask[:, None] & nmask[None, :]
                            off = (rows[:, None] * H + h) * N + offs_n[None, :]
                            base = tl.load(BASE + off, mask=m2, other=0.0)
                            x = tl.load(
                                X
                                + rows[:, None] * XS0
                                + h * XS1
                                + offs_k[None, :] * XS2,
                                mask=rmask[:, None] & kmask[None, :],
                                other=0.0,
                            )
                            acc = (
                                tl.dot(
                                    x,
                                    wb,
                                    input_precision=(
                                        "ieee"
                                        if x.dtype == tl.float32
                                        else "tf32"
                                    ),
                                )
                                * scale
                            )
                            tl.store(
                                OUT + off,
                                (base.to(tl.float32) + acc).to(
                                    OUT.dtype.element_ty
                                ),
                                mask=m2,
                            )


def step_b_v(
    attn_lora_a, B_buf, batch_info, base_output, qk_nope_head_dim, v_head_dim
):
    S, H, K = attn_lora_a.shape
    N = v_head_dim
    na = B_buf.shape[0]
    if K not in (16, 32) or na >= batch_info.bs:
        return _mloop_step_b_v(
            attn_lora_a,
            B_buf,
            batch_info,
            base_output,
            qk_nope_head_dim,
            v_head_dim,
        )
    if base_output.numel() == 0:
        out = torch.empty_like(base_output)
        _copy_base_kernel.run(
            base_output, out, (0, 1024), grid=(1,), warmup=False
        )
        return out
    bn = max(16, 0 if N == 0 else 1 << (N - 1).bit_length())
    if (
        batch_info.bs == 0
        or batch_info.permutation is not None
        or N != bn
        or (N < 16)
        or (K < 16)
    ):
        return _unfused_step_b_v(
            attn_lora_a,
            B_buf,
            batch_info,
            base_output,
            qk_nope_head_dim,
            v_head_dim,
        )
    if attn_lora_a.stride(-1) != 1:
        attn_lora_a = attn_lora_a.contiguous()
    if B_buf.stride(-1) != 1:
        B_buf = B_buf.contiguous()
    base = base_output.contiguous()
    ml = batch_info.max_len or S
    bm = min(256, max(16, 0 if ml == 0 else 1 << (ml - 1).bit_length()))
    panels = 2 if K == 16 and bm == 256 else 1
    bm //= panels
    tiles = max(1, (ml + bm * panels - 1) // (bm * panels))
    covers = (
        min(
            max(1, min(32, (S * H * N + 1048576 - 1) // 1048576)),
            max(1, tiles * H * na),
        )
        if K == 16
        else max(1, min(32, (S * H * N + 1048576 - 1) // 1048576))
    )
    out = base.new_empty((S + bm, H, N))
    _fused_adapter.run(
        attn_lora_a,
        B_buf,
        base,
        out,
        batch_info.seg_indptr.contiguous(),
        batch_info.weight_indices.contiguous(),
        batch_info.lora_ranks.contiguous(),
        batch_info.scalings.contiguous(),
        (
            H,
            N,
            S,
            K,
            qk_nope_head_dim + N,
            qk_nope_head_dim,
            batch_info.bs,
            na,
            tiles,
            covers,
            *attn_lora_a.stride(),
            *B_buf.stride(),
            bm,
            bn,
            max(16, 0 if K == 0 else 1 << (K - 1).bit_length()),
            panels,
        ),
        grid=(
            (max(1, tiles * H * na),)
            if K == 16
            else (tiles * H * na + covers,)
        ),
        warmup=False,
    )
    return out[:S]


@triton.jit
def _perm_all_rows(X, W, C, RANKS, SCALES, SIZE, META: tl.constexpr):
    H: tl.constexpr = META[0]
    K: tl.constexpr = META[1]
    N: tl.constexpr = META[2]
    F: tl.constexpr = META[3]
    Q: tl.constexpr = META[4]
    SP: tl.constexpr = META[5]
    X0: tl.constexpr = META[6]
    X1: tl.constexpr = META[7]
    X2: tl.constexpr = META[8]
    W0: tl.constexpr = META[9]
    W1: tl.constexpr = META[10]
    W2: tl.constexpr = META[11]
    h = tl.program_id(0)
    w = tl.program_id(1)
    if tl.load(RANKS + w) == 0:
        return
    m = tl.arange(0, SP)
    k = tl.arange(0, K)
    n = tl.arange(0, N)
    x = tl.load(
        X + m[:, None] * X0 + h * X1 + k[None, :] * X2,
        (m < SIZE)[:, None] & (k < K)[None, :],
        other=0.0,
    )
    wt = tl.load(W + w * W0 + (h * F + Q + n[None, :]) * W1 + k[:, None] * W2)
    correction = tl.dot(x, wt) * tl.load(SCALES + w).to(tl.float32)
    tl.store(
        C + (w * SP + m[:, None]) * (H * N) + h * N + n[None, :],
        correction,
        (m < SP)[:, None] & (n < N)[None, :],
    )


@triton.jit
def _perm_choose(C, BASE, SEG, WI, RANKS, PERM, OUT, META: tl.constexpr):
    H: tl.constexpr = META[0]
    N: tl.constexpr = META[1]
    SP: tl.constexpr = META[2]
    BS: tl.constexpr = META[4]
    FULL: tl.constexpr = META[5]
    BLOCK: tl.constexpr = META[6]
    B0: tl.constexpr = META[7]
    B1: tl.constexpr = META[8]
    B2: tl.constexpr = META[9]
    m = tl.program_id(0)
    b = tl.program_id(1)
    off = tl.arange(0, BLOCK)
    if FULL:
        if b == BS:
            first = tl.load(SEG)
            last = tl.load(SEG + BS)
            if (m < first) | (m >= last):
                row = tl.load(PERM + m).to(tl.int32)
                base = tl.load(
                    BASE + row * B0 + off // N * B1 + off % N * B2,
                    off < H * N,
                    other=0.0,
                )
                tl.store(OUT + row * H * N + off, base, off < H * N)
            return
    lo = tl.load(SEG + b)
    hi = tl.load(SEG + b + 1)
    if m >= hi - lo:
        return
    w = tl.load(WI + b)
    row = tl.load(PERM + lo + m).to(tl.int32)
    if tl.load(RANKS + w) == 0:
        inactive = tl.load(
            BASE + row * B0 + off // N * B1 + off % N * B2,
            off < H * N,
            other=0.0,
        )
        tl.store(OUT + row * H * N + off, inactive, off < H * N)
        return
    base = tl.load(
        BASE + row * B0 + off // N * B1 + off % N * B2, off < H * N, other=0.0
    ).to(tl.float32)
    delta = tl.load(C + (w * SP + row) * (H * N) + off, off < H * N, other=0.0)
    tl.store(OUT + row * H * N + off, base + delta, off < H * N)


__all__ = ["step_b_v"]
