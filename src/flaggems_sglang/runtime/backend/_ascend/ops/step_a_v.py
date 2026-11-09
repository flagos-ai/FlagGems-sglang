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
def _perm_head_matrix(X, A, OUT, SEG, WI, RANKS, P, META: tl.constexpr):
    H: tl.constexpr = META[0]
    N: tl.constexpr = META[1]
    K: tl.constexpr = META[2]
    BS: tl.constexpr = META[3]
    X0: tl.constexpr = META[4]
    X1: tl.constexpr = META[5]
    X2: tl.constexpr = META[6]
    A0: tl.constexpr = META[7]
    A1: tl.constexpr = META[8]
    A2: tl.constexpr = META[9]
    logical_row = tl.program_id(0)
    row = tl.load(P + logical_row).to(tl.int32)
    h = tl.arange(0, H)
    n = tl.arange(0, N)
    k = tl.arange(0, K)
    selected = 0
    active = False
    for b in range(BS):
        lo = tl.load(SEG + b)
        hi = tl.load(SEG + b + 1)
        if (logical_row >= lo) & (logical_row < hi):
            selected = tl.load(WI + b)
            active = tl.load(RANKS + selected) > 0
    if active:
        x = tl.load(X + row * X0 + h[:, None] * X1 + k[None, :] * X2)
        a = tl.load(A + selected * A0 + n[None, :] * A1 + k[:, None] * A2)
        acc = tl.dot(x, a)
        tl.store(OUT + (row * H + h[:, None]) * N + n[None, :], acc)
    else:
        zero_i = tl.arange(0, H * N)
        tl.store(OUT + row * H * N + zero_i, 0.0)


@triton.jit
def _flat_kernel(X, A, OUT, SEG, WIDX, RANKS, META: tl.constexpr):
    H: tl.constexpr = META[0]
    N: tl.constexpr = META[1]
    M: tl.constexpr = META[2]
    K: tl.constexpr = META[3]
    XS1: tl.constexpr = META[4]
    XS2: tl.constexpr = META[5]
    AS0: tl.constexpr = META[6]
    AS1: tl.constexpr = META[7]
    AS2: tl.constexpr = META[8]
    BM: tl.constexpr = META[9]
    BN: tl.constexpr = META[10]
    BK: tl.constexpr = META[11]
    TAIL: tl.constexpr = META[12]
    BS: tl.constexpr = META[13]
    TILES: tl.constexpr = META[14]
    FOLD_ZERO: tl.constexpr = META[15]
    pid = tl.program_id(0)
    b = pid // TILES
    tile = pid % TILES
    if (
        FOLD_ZERO and pid == 0 or (not FOLD_ZERO and b == BS)
    ) and tl.program_id(1) == 0:
        first = tl.load(SEG) * H * N
        last = tl.load(SEG + BS) * H * N
        zero_i = tl.arange(0, 2048)
        for start in range(0, first, 2048):
            ix = start + zero_i
            tl.store(OUT + ix, 0.0, mask=ix < first)
        for start in range(last, M * N, 2048):
            ix = start + zero_i
            tl.store(OUT + ix, 0.0, mask=ix < M * N)
    if not FOLD_ZERO and b == BS or BS == 0:
        return
    lo = tl.load(SEG + b) * H
    hi = tl.load(SEG + b + 1) * H
    if lo >= hi:
        return
    w = tl.load(WIDX + b)
    m0 = lo + tile * BM
    if m0 >= hi:
        return
    if TAIL:
        m = m0 + tl.arange(0, BM)
        keep = m < hi
        ms = tl.minimum(m, M - 1)
    else:
        m = tl.minimum(m0, M - BM) + tl.arange(0, BM)
        keep = (m >= m0) & (m < hi)
        ms = m
    offs_n = tl.program_id(1) * BN + tl.arange(0, BN)
    offs_k = tl.arange(0, BK)
    nmask = offs_n < N
    if tl.load(RANKS + w) == 0:
        tl.store(
            OUT + ms[:, None] * N + offs_n[None, :],
            tl.full((BM, BN), 0, tl.float32),
            mask=keep[:, None] & nmask[None, :],
        )
        return
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    xp = X + ms[:, None] * XS1 + offs_k[None, :] * XS2
    ap = A + w * AS0 + offs_n[None, :] * AS1 + offs_k[:, None] * AS2
    for k0 in tl.static_range(0, K, BK):
        kmask = k0 + offs_k < K
        x = tl.load(xp, mask=keep[:, None] & kmask[None, :], other=0.0)
        a = tl.load(ap, mask=kmask[:, None] & nmask[None, :], other=0.0)
        acc = tl.dot(x, a, acc)
        xp += BK * XS2
        ap += BK * AS2
    tl.store(
        OUT + ms[:, None] * N + offs_n[None, :],
        acc.to(OUT.dtype.element_ty),
        mask=keep[:, None] & nmask[None, :],
    )


@triton.jit
def _pack_kernel(X, P, T, META: tl.constexpr):
    H: tl.constexpr = META[0]
    K: tl.constexpr = META[1]
    XS0: tl.constexpr = META[2]
    XS1: tl.constexpr = META[3]
    XS2: tl.constexpr = META[4]
    ROWS: tl.constexpr = META[5]
    B: tl.constexpr = META[6]
    HAS_PERM: tl.constexpr = META[7]
    ix = tl.program_id(0) * B + tl.arange(0, B)
    keep = ix < ROWS * H * K
    i = ix // (H * K)
    h = ix // K % H
    k = ix % K
    if HAS_PERM:
        row = tl.load(P + i, keep, 0).to(tl.int32)
    else:
        row = i
    v = tl.load(X + row * XS0 + h * XS1 + k * XS2, keep, 0)
    tl.store(T + ix, v, keep)


@triton.jit
def _scatter_kernel(Y, P, OUT, META: tl.constexpr):
    H: tl.constexpr = META[0]
    N: tl.constexpr = META[1]
    ROWS: tl.constexpr = META[2]
    B: tl.constexpr = META[3]
    HAS_PERM: tl.constexpr = META[4]
    ix = tl.program_id(0) * B + tl.arange(0, B)
    keep = ix < ROWS * H * N
    i = ix // (H * N)
    h = ix // N % H
    n = ix % N
    if HAS_PERM:
        row = tl.load(P + i, keep, 0).to(tl.int32)
    else:
        row = i
    v = tl.load(Y + ix, keep, 0)
    tl.store(OUT + (row * H + h) * N + n, v, keep)


@triton.jit
def _reuse_weight(X, A, OUT, SEG, WIDX, RANKS, META: tl.constexpr):
    H: tl.constexpr = META[0]
    N: tl.constexpr = META[1]
    K: tl.constexpr = META[2]
    XS1: tl.constexpr = META[3]
    AS0: tl.constexpr = META[4]
    AS1: tl.constexpr = META[5]
    BM: tl.constexpr = META[6]
    BN: tl.constexpr = META[7]
    ML: tl.constexpr = META[8]
    M: tl.constexpr = META[9]
    BS: tl.constexpr = META[10]
    TILES: tl.constexpr = META[11]
    pid = tl.program_id(0)
    b = pid // TILES
    tile = pid % TILES
    if b == BS:
        first = tl.load(SEG) * H * N
        last = tl.load(SEG + BS) * H * N
        z = tl.arange(0, 2048)
        for start in range(0, first, 2048):
            p = start + z
            tl.store(OUT + p, 0.0, mask=p < first)
        for start in range(last, M * N, 2048):
            p = start + z
            tl.store(OUT + p, 0.0, mask=p < M * N)
        return
    lo = tl.load(SEG + b) * H
    hi = tl.load(SEG + b + 1) * H
    panels = (hi - lo) // BM
    start_panel = tile * panels // TILES
    end_panel = (tile + 1) * panels // TILES
    m0 = lo + start_panel * BM
    if m0 >= hi or start_panel >= end_panel:
        return
    w = tl.load(WIDX + b)
    ns = tl.arange(0, BN)
    count = tl.minimum(ML, end_panel - start_panel)
    if tl.load(RANKS + w) == 0:
        for i in range(count):
            ms = m0 + i * BM + tl.arange(0, BM)
            tl.store(OUT + ms[:, None] * N + ns[None, :], 0.0)
        return
    ks = tl.arange(0, K)
    av = tl.load(A + w * AS0 + ks[:, None] + ns[None, :] * AS1)
    for i in range(count):
        ms = m0 + i * BM + tl.arange(0, BM)
        xv = tl.load(X + ms[:, None] * XS1 + ks[None, :])
        acc = tl.dot(xv, av)
        tl.store(OUT + ms[:, None] * N + ns[None, :], acc)


@triton.jit
def _reuse_global(X, A, OUT, SEG, WIDX, RANKS, META: tl.constexpr):
    H: tl.constexpr = META[0]
    N: tl.constexpr = META[1]
    K: tl.constexpr = META[2]
    XS1: tl.constexpr = META[3]
    AS0: tl.constexpr = META[4]
    AS1: tl.constexpr = META[5]
    BM: tl.constexpr = META[6]
    BN: tl.constexpr = META[7]
    M: tl.constexpr = META[8]
    BS: tl.constexpr = META[9]
    WORKERS: tl.constexpr = META[10]
    pid = tl.program_id(0)
    first = tl.load(SEG) * H
    last = tl.load(SEG + BS) * H
    if pid == WORKERS:
        z = tl.arange(0, 2048)
        for start in range(0, first * N, 2048):
            p = start + z
            tl.store(OUT + p, 0.0, mask=p < first * N)
        for start in range(last * N, M * N, 2048):
            p = start + z
            tl.store(OUT + p, 0.0, mask=p < M * N)
        return
    panels = (last - first) // BM
    begin = first + pid * panels // WORKERS * BM
    end = first + (pid + 1) * panels // WORKERS * BM
    ns = tl.arange(0, BN)
    ks = tl.arange(0, K)
    for b in range(BS):
        lo = tl.maximum(begin, tl.load(SEG + b) * H)
        hi = tl.minimum(end, tl.load(SEG + b + 1) * H)
        if lo < hi:
            count = (hi - lo) // BM
            w = tl.load(WIDX + b)
            if tl.load(RANKS + w) == 0:
                for i in range(count):
                    ms = lo + i * BM + tl.arange(0, BM)
                    tl.store(OUT + ms[:, None] * N + ns[None, :], 0.0)
            else:
                av = tl.load(A + w * AS0 + ks[:, None] + ns[None, :] * AS1)
                for i in range(count):
                    ms = lo + i * BM + tl.arange(0, BM)
                    xv = tl.load(X + ms[:, None] * XS1 + ks[None, :])
                    acc = tl.dot(xv, av)
                    tl.store(OUT + ms[:, None] * N + ns[None, :], acc)


def step_a_v(attn_output, A_buf, batch_info):
    S, H, K = attn_output.shape
    N = A_buf.shape[1]
    xs0, xs1, xs2 = attn_output.stride()
    seg = batch_info.seg_indptr
    widx = batch_info.weight_indices
    ranks = batch_info.lora_ranks
    perm = batch_info.permutation
    if seg.stride(0) != 1:
        seg = seg.contiguous()
    if widx.stride(0) != 1:
        widx = widx.contiguous()
    if ranks.stride(0) != 1:
        ranks = ranks.contiguous()
    if perm is not None and perm.stride(0) != 1:
        perm = perm.contiguous()
    has_perm = perm is not None
    bn = min(32, max(16, 0 if N == 0 else 1 << (N - 1).bit_length()))
    nt = (N + bn - 1) // bn
    kp = max(16, 0 if K == 0 else 1 << (K - 1).bit_length())
    bk = min(64, kp)
    fp32 = attn_output.dtype == torch.float32
    bm = 128 if fp32 else 256
    ml = batch_info.max_len
    if not isinstance(ml, int) or ml <= 0 or ml > S:
        ml = S
    bs = batch_info.bs
    rows = perm.numel() if has_perm else S
    if (
        has_perm
        and rows == S
        and (S > 0)
        and (not fp32)
        and (H in (16, 32))
        and (N in (16, 32))
        and (K == kp)
        and (K <= 512)
    ):
        if xs2 != 1:
            attn_output = attn_output.contiguous()
            xs0, xs1, xs2 = attn_output.stride()
        if A_buf.stride(2) != 1:
            A_buf = A_buf.contiguous()
        out = attn_output.new_empty((S, H, N))
        _perm_head_matrix.run(
            attn_output,
            A_buf,
            out,
            seg,
            widx,
            ranks,
            perm,
            (H, N, K, bs, xs0, xs1, xs2, *A_buf.stride()),
            grid=(S,),
            warmup=False,
            num_warps=4,
            num_stages=1,
        )
        return out
    if (
        not fp32
        and bs > 0
        and (N <= 32)
        and (not has_perm)
        and (xs0 == H * xs1)
        and (S * H > 4096)
        and (H % 64 == 0)
        and (K == kp)
        and (N == bn)
        and (xs2 == 1)
        and (A_buf.stride(2) == 1)
    ):
        out = attn_output.new_empty((S, H, N))
        cores = triton.runtime.driver.active.utils.get_device_properties(
            attn_output.device.index
        )["num_aicore"]
        if bs <= cores and cores % bs != 0:
            _reuse_global.run(
                attn_output,
                A_buf,
                out,
                seg,
                widx,
                ranks,
                (
                    H,
                    N,
                    K,
                    xs1,
                    A_buf.stride(0),
                    A_buf.stride(1),
                    64,
                    bn,
                    S * H,
                    bs,
                    cores,
                ),
                grid=(cores + 1,),
                warmup=False,
                num_warps=4,
                num_stages=2,
            )
            return out
        tiles = cores // math.gcd(cores, bs)
        panels_per_tile = max(1, (ml * H + 64 * tiles - 1) // (64 * tiles))
        _reuse_weight.run(
            attn_output,
            A_buf,
            out,
            seg,
            widx,
            ranks,
            (
                H,
                N,
                K,
                xs1,
                A_buf.stride(0),
                A_buf.stride(1),
                64,
                bn,
                panels_per_tile,
                S * H,
                bs,
                tiles,
            ),
            grid=(tiles * bs + 1,),
            warmup=False,
            num_warps=4,
            num_stages=2,
        )
        return out
    if bs == 0 or rows == 0 or (not has_perm and xs0 == H * xs1):
        fold_zero = (
            not fp32 and (not has_perm) and (S * H > 4096) and (N <= 32)
        )
        out = attn_output.new_empty((S, H, N))
        tiles = max(1, (ml * H + bm - 1) // bm)
        _flat_kernel.run(
            attn_output,
            A_buf,
            out,
            seg,
            widx,
            ranks,
            (
                H,
                N,
                S * H,
                K,
                xs1,
                xs2,
                *A_buf.stride(),
                bm,
                bn,
                bk,
                S * H < bm,
                bs,
                tiles,
                fold_zero,
            ),
            grid=(max(1, tiles * bs) if fold_zero else tiles * bs + 1, nt),
            warmup=False,
            num_warps=4,
            num_stages=2,
        )
        return out
    pk = attn_output.new_empty((rows * H, K))
    _pack_kernel.run(
        attn_output,
        perm if has_perm else attn_output,
        pk,
        (H, K, xs0, xs1, xs2, rows, 2048, has_perm),
        grid=((rows * H * K + 2048 - 1) // 2048,),
        warmup=False,
        num_warps=4,
        num_stages=2,
    )
    mid = attn_output.new_empty((rows * H, N))
    tiles = max(1, (min(ml, rows) * H + bm - 1) // bm)
    _flat_kernel.run(
        pk,
        A_buf,
        mid,
        seg,
        widx,
        ranks,
        (
            H,
            N,
            rows * H,
            K,
            K,
            1,
            *A_buf.stride(),
            bm,
            bn,
            bk,
            rows * H < bm,
            bs,
            tiles,
            False,
        ),
        grid=(tiles * bs + 1, nt),
        warmup=False,
        num_warps=4,
        num_stages=2,
    )
    if rows == S:
        out = attn_output.new_empty((S, H, N))
    else:
        out = torch.zeros(
            (S, H, N), dtype=attn_output.dtype, device=attn_output.device
        )
    _scatter_kernel.run(
        mid,
        perm if has_perm else attn_output,
        out,
        (H, N, rows, 2048, has_perm),
        grid=((rows * H * N + 2048 - 1) // 2048,),
        warmup=False,
        num_warps=4,
        num_stages=2,
    )
    return out


__all__ = ["step_a_v"]
