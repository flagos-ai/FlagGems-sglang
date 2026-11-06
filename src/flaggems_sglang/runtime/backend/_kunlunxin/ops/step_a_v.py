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
def _flat_nmajor(X, A, OUT, SEG, WIDX, RANKS, META: tl.constexpr):
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
    BS: tl.constexpr = META[12]
    TILES: tl.constexpr = META[13]
    FUSED_ZERO: tl.constexpr = META[14]
    pid = tl.program_id(0)
    b = pid // TILES
    tile = pid % TILES
    if FUSED_ZERO and b == BS:
        if tl.program_id(1) != 0:
            return
        first = tl.load(SEG) * H
        last = tl.load(SEG + BS) * H
        zero_m = tl.arange(0, BM)
        zero_n = tl.arange(0, BN)
        for start in range(0, first, BM):
            row = start + zero_m
            tl.store(
                OUT + zero_n[:, None] * M + row[None, :],
                tl.full((BN, BM), 0, tl.float32),
                mask=(zero_n[:, None] < N) & (row[None, :] < first),
            )
        for start in range(last, M, BM):
            row = start + zero_m
            tl.store(
                OUT + zero_n[:, None] * M + row[None, :],
                tl.full((BN, BM), 0, tl.float32),
                mask=(zero_n[:, None] < N) & (row[None, :] < M),
            )
        return
    lo = tl.load(SEG + b) * H
    hi = tl.load(SEG + b + 1) * H
    if lo >= hi:
        return
    w = tl.load(WIDX + b)
    m0 = lo + tile * BM
    if m0 >= hi:
        return
    m = m0 + tl.arange(0, BM)
    keep = m < hi
    offs_n = tl.program_id(1) * BN + tl.arange(0, BN)
    offs_k = tl.arange(0, BK)
    nmask = offs_n < N
    if tl.load(RANKS + w) == 0:
        if FUSED_ZERO:
            tl.store(
                OUT + offs_n[:, None] * M + m[None, :],
                tl.full((BN, BM), 0, tl.float32),
                mask=nmask[:, None] & keep[None, :],
            )
        return
    acc = tl.zeros((BN, BM), dtype=tl.float32)
    xp = X + m[None, :] * XS1 + offs_k[:, None] * XS2
    ap = A + w * AS0 + offs_n[:, None] * AS1 + offs_k[None, :] * AS2
    for k0 in range(0, K, BK):
        kmask = k0 + offs_k < K
        x = tl.load(xp, mask=keep[None, :] & kmask[:, None], other=0.0)
        a = tl.load(ap, mask=kmask[None, :] & nmask[:, None], other=0.0)
        acc = tl.dot(a, x, acc, input_precision="ieee")
        xp += BK * XS2
        ap += BK * AS2
    tl.store(
        OUT + m[None, :] + offs_n[:, None] * M,
        acc.to(OUT.dtype.element_ty),
        mask=keep[None, :] & nmask[:, None],
    )


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
        acc = tl.dot(x, a, input_precision="ieee")
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
    BS: tl.constexpr = META[12]
    TILES: tl.constexpr = META[13]
    FUSED_ZERO: tl.constexpr = META[14]
    pid = tl.program_id(0)
    b = pid // TILES
    tile = pid % TILES
    if FUSED_ZERO and b == BS:
        if tl.program_id(1) != 0:
            return
        first = tl.load(SEG) * H * N
        last = tl.load(SEG + BS) * H * N
        zero_i = tl.arange(0, 2048)
        for start in range(0, first, 2048):
            ix = start + zero_i
            tl.store(OUT + ix, 0.0, mask=ix < first)
        for start in range(last, M * N, 2048):
            ix = start + zero_i
            tl.store(OUT + ix, 0.0, mask=ix < M * N)
        return
    lo = tl.load(SEG + b) * H
    hi = tl.load(SEG + b + 1) * H
    if lo >= hi:
        return
    w = tl.load(WIDX + b)
    m0 = lo + tile * BM
    if m0 >= hi:
        return
    m = m0 + tl.arange(0, BM)
    keep = m < hi
    offs_n = tl.program_id(1) * BN + tl.arange(0, BN)
    offs_k = tl.arange(0, BK)
    nmask = offs_n < N
    if tl.load(RANKS + w) == 0:
        if FUSED_ZERO:
            tl.store(
                OUT + m[:, None] * N + offs_n[None, :],
                tl.full((BM, BN), 0, tl.float32),
                mask=keep[:, None] & nmask[None, :],
            )
        return
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    xp = X + m[:, None] * XS1 + offs_k[None, :] * XS2
    ap = A + w * AS0 + offs_n[None, :] * AS1 + offs_k[:, None] * AS2
    for k0 in range(0, K, BK):
        kmask = k0 + offs_k < K
        x = tl.load(xp, mask=keep[:, None] & kmask[None, :], other=0.0)
        a = tl.load(ap, mask=kmask[:, None] & nmask[None, :], other=0.0)
        acc = tl.dot(x, a, acc, input_precision="ieee")
        xp += BK * XS2
        ap += BK * AS2
    tl.store(
        OUT + m[:, None] * N + offs_n[None, :],
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
def _zero_fp32(OUT, META: tl.constexpr):
    COUNT: tl.constexpr = META[0]
    BZ: tl.constexpr = META[1]
    i = tl.program_id(0) * BZ + tl.arange(0, BZ)
    tl.store(OUT + i, tl.full((BZ,), 0, tl.float32), mask=i < COUNT)


@triton.jit
def _reduce_fp32(X, A, OUT, SEG, WIDX, RANKS, PERM, META: tl.constexpr):
    H: tl.constexpr = META[0]
    N: tl.constexpr = META[1]
    K: tl.constexpr = META[2]
    HAS_PERM: tl.constexpr = META[3]
    XS0: tl.constexpr = META[4]
    XS1: tl.constexpr = META[5]
    XS2: tl.constexpr = META[6]
    AS0: tl.constexpr = META[7]
    AS1: tl.constexpr = META[8]
    AS2: tl.constexpr = META[9]
    BK: tl.constexpr = META[10]
    GN: tl.constexpr = META[11]
    group = tl.program_id(1)
    groups: tl.constexpr = triton.cdiv(N, GN)
    b = group // groups
    n0 = group % groups * GN
    lo = tl.load(SEG + b) * H
    hi = tl.load(SEG + b + 1) * H
    if lo >= hi:
        return
    w = tl.load(WIDX + b)
    if tl.load(RANKS + w) == 0:
        return
    k = tl.arange(0, BK)
    m = lo + tl.program_id(0)
    if m >= hi:
        return
    pos = m // H
    h = m % H
    if HAS_PERM:
        row = tl.load(PERM + pos)
    else:
        row = pos
    x = tl.load(X + row * XS0 + h * XS1 + k * XS2, mask=k < K, other=0)
    for ni in tl.static_range(GN):
        n = n0 + ni
        if n < N:
            a = tl.load(A + w * AS0 + n * AS1 + k * AS2, mask=k < K, other=0)
            value = tl.sum(x * a, 0)
            tl.store(OUT + (row * H + h) * N + n, value)


@triton.jit
def _pad_seg_x(X, P, T, SEG, WIDX, RANKS, META: tl.constexpr):
    H: tl.constexpr = META[0]
    K: tl.constexpr = META[1]
    MP: tl.constexpr = META[2]
    KP: tl.constexpr = META[3]
    XS0: tl.constexpr = META[4]
    XS1: tl.constexpr = META[5]
    XS2: tl.constexpr = META[6]
    HAS_PERM: tl.constexpr = META[7]
    m = tl.program_id(0)
    b = tl.program_id(1)
    lo = tl.load(SEG + b)
    hi = tl.load(SEG + b + 1)
    if m >= (hi - lo) * H:
        return
    w = tl.load(WIDX + b)
    if tl.load(RANKS + w) == 0:
        return
    i = lo + m // H
    h = m % H
    if HAS_PERM:
        row = tl.load(P + i).to(tl.int32)
    else:
        row = i
    k = tl.arange(0, KP)
    v = tl.load(X + row * XS0 + h * XS1 + k * XS2, k < K, 0).to(tl.float32)
    tl.store(T + (b * MP + m) * KP + k, v)


@triton.jit
def _pad_weight(A, T, META: tl.constexpr):
    K: tl.constexpr = META[0]
    NP: tl.constexpr = META[1]
    KP: tl.constexpr = META[2]
    AS0: tl.constexpr = META[3]
    AS1: tl.constexpr = META[4]
    AS2: tl.constexpr = META[5]
    n = tl.program_id(0)
    w = tl.program_id(1)
    k = tl.arange(0, KP)
    v = tl.load(A + w * AS0 + n * AS1 + k * AS2, k < K, 0).to(tl.float32)
    tl.store(T + (w * NP + n) * KP + k, v)


@triton.jit
def _dense_dot(X, A, C, SEG, WIDX, RANKS, META: tl.constexpr):
    MP: tl.constexpr = META[0]
    NP: tl.constexpr = META[1]
    KP: tl.constexpr = META[2]
    TM: tl.constexpr = META[3]
    TN: tl.constexpr = META[4]
    H: tl.constexpr = META[5]
    BM: tl.constexpr = META[6]
    BN: tl.constexpr = META[7]
    BK: tl.constexpr = META[8]
    pid = tl.program_id(0)
    b = pid // (TM * TN)
    tile = pid // TN % TM
    lo = tl.load(SEG + b)
    hi = tl.load(SEG + b + 1)
    if hi > lo and tile * BM < (hi - lo) * H:
        w = tl.load(WIDX + b)
        if tl.load(RANKS + w) > 0:
            m = tile * BM + tl.arange(0, BM)
            n = pid % TN * BN + tl.arange(0, BN)
            k = tl.arange(0, BK)
            xp = X + (b * MP + m[:, None]) * KP + k[None, :]
            ap = A + (w * NP + n[None, :]) * KP + k[:, None]
            acc = tl.zeros((BM, BN), tl.float32)
            for start in range(0, KP, BK):
                xv = tl.load(xp + start)
                av = tl.load(ap + start)
                acc = tl.dot(xv, av, acc, input_precision="ieee")
            tl.store(C + (b * MP + m[:, None]) * NP + n[None, :], acc)


@triton.jit
def _scatter_dense(C, P, OUT, SEG, WIDX, RANKS, META: tl.constexpr):
    H: tl.constexpr = META[0]
    N: tl.constexpr = META[1]
    MP: tl.constexpr = META[2]
    NP: tl.constexpr = META[3]
    BN: tl.constexpr = META[4]
    HAS_PERM: tl.constexpr = META[5]
    m = tl.program_id(0)
    b = tl.program_id(1)
    lo = tl.load(SEG + b)
    hi = tl.load(SEG + b + 1)
    if m >= (hi - lo) * H:
        return
    w = tl.load(WIDX + b)
    if tl.load(RANKS + w) == 0:
        return
    i = lo + m // H
    h = m % H
    if HAS_PERM:
        row = tl.load(P + i).to(tl.int32)
    else:
        row = i
    n = tl.arange(0, BN)
    v = tl.load(C + (b * MP + m) * NP + n, n < N, 0)
    tl.store(OUT + (row * H + h) * N + n, v, n < N)


@triton.jit
def _pad_seg_token(X, P, T, SEG, WIDX, RANKS, META: tl.constexpr):
    H: tl.constexpr = META[0]
    K: tl.constexpr = META[1]
    MP: tl.constexpr = META[2]
    KP: tl.constexpr = META[3]
    X0: tl.constexpr = META[4]
    X1: tl.constexpr = META[5]
    X2: tl.constexpr = META[6]
    B: tl.constexpr = META[7]
    CHUNKS: tl.constexpr = META[8]
    HAS_PERM: tl.constexpr = META[9]
    p = tl.program_id(0)
    row_in_segment = p // CHUNKS
    b = tl.program_id(1)
    lo = tl.load(SEG + b)
    hi = tl.load(SEG + b + 1)
    if row_in_segment >= hi - lo:
        return
    w = tl.load(WIDX + b)
    if tl.load(RANKS + w) == 0:
        return
    logical_row = lo + row_in_segment
    if HAS_PERM:
        row = tl.load(P + logical_row).to(tl.int32)
    else:
        row = logical_row
    start = p % CHUNKS * B if CHUNKS > 1 else 0
    ix = start + tl.arange(0, B)
    h = ix // KP
    k = ix % KP
    keep = ix < H * KP
    value = tl.load(X + row * X0 + h * X1 + k * X2, keep & (k < K), 0).to(
        tl.float32
    )
    tl.store(T + (b * MP + row_in_segment * H) * KP + ix, value, keep)


@triton.jit
def _scatter_dense_token(C, P, OUT, SEG, WIDX, RANKS, META: tl.constexpr):
    H: tl.constexpr = META[0]
    N: tl.constexpr = META[1]
    MP: tl.constexpr = META[2]
    NP: tl.constexpr = META[3]
    B: tl.constexpr = META[4]
    CHUNKS: tl.constexpr = META[5]
    HAS_PERM: tl.constexpr = META[6]
    p = tl.program_id(0)
    row_in_segment = p // CHUNKS
    b = tl.program_id(1)
    lo = tl.load(SEG + b)
    hi = tl.load(SEG + b + 1)
    if row_in_segment >= hi - lo:
        return
    w = tl.load(WIDX + b)
    if tl.load(RANKS + w) == 0:
        return
    logical_row = lo + row_in_segment
    if HAS_PERM:
        row = tl.load(P + logical_row).to(tl.int32)
    else:
        row = logical_row
    start = p % CHUNKS * B if CHUNKS > 1 else 0
    ix = start + tl.arange(0, B)
    h = ix // N
    n = ix % N
    keep = ix < H * N
    v = tl.load(C + (b * MP + row_in_segment * H + h) * NP + n, keep, 0)
    tl.store(OUT + row * H * N + ix, v, keep)


def step_a_v(attn_output, A_buf, batch_info):
    fp32 = attn_output.dtype == torch.float32
    if attn_output.stride(-1) != 1:
        attn_output = attn_output.contiguous()
    if A_buf.stride(-1) != 1:
        A_buf = A_buf.contiguous()
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
    bk = min(256, kp)
    bm = 256
    bs = batch_info.bs
    ml = batch_info.max_len
    if not isinstance(ml, int) or ml <= 0 or ml > S:
        ml = S
    rows = perm.numel() if has_perm else S
    if (
        has_perm
        and rows == S
        and (S > 0)
        and (not fp32)
        and (H in (16, 32, 64))
        and (N in (16, 32))
        and (K == kp)
        and (K <= 512)
    ):
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
    if not fp32 and rows * H > 4096 and (N <= 32) and (N == bn) and (K == kp):
        bm = 512
    if fp32 and (K != kp or N % bn != 0):
        if bs == 0 or rows == 0 or A_buf.shape[0] == 0:
            out = attn_output.new_empty((S, H, N))
            _zero_fp32.run(
                out,
                (S * H * N, 128),
                grid=(max(1, (S * H * N + 128 - 1) // 128),),
                warmup=False,
                num_warps=4,
                num_stages=1,
            )
            return out
        mp = max(16, (ml * H + 16 - 1) // 16 * 16)
        npad = max(64, 0 if N == 0 else 1 << (N - 1).bit_length())
        xp = torch.zeros(
            (bs, mp, kp), dtype=torch.float32, device=attn_output.device
        )
        ap = torch.zeros(
            (A_buf.shape[0], npad, kp),
            dtype=torch.float32,
            device=attn_output.device,
        )
        mid = torch.empty(
            (bs, mp, npad), dtype=torch.float32, device=attn_output.device
        )
        out = torch.zeros(
            (S, H, N), dtype=attn_output.dtype, device=attn_output.device
        )
        if H * K > 0:
            pack_block = min(
                2048, 0 if H * kp == 0 else 1 << (H * kp - 1).bit_length()
            )
            pack_chunks = (H * kp + pack_block - 1) // pack_block
            _pad_seg_token.run(
                attn_output,
                perm if has_perm else attn_output,
                xp,
                seg,
                widx,
                ranks,
                (
                    H,
                    K,
                    mp,
                    kp,
                    xs0,
                    xs1,
                    xs2,
                    pack_block,
                    pack_chunks,
                    has_perm,
                ),
                grid=(ml * pack_chunks, bs),
                warmup=False,
                num_warps=4,
                num_stages=1,
            )
        _pad_weight.run(
            A_buf,
            ap,
            (K, npad, kp, *A_buf.stride()),
            grid=(N, A_buf.shape[0]),
            warmup=False,
            num_warps=4,
            num_stages=1,
        )
        _dense_dot.run(
            xp,
            ap,
            mid,
            seg,
            widx,
            ranks,
            (mp, npad, kp, mp // 16, npad // 64, H, 16, 64, min(128, kp)),
            grid=(bs * (mp // 16) * (npad // 64),),
            warmup=False,
            num_warps=4,
            num_stages=1,
        )
        scatter_block = min(
            2048, 0 if H * N == 0 else 1 << (H * N - 1).bit_length()
        )
        scatter_chunks = (H * N + scatter_block - 1) // scatter_block
        _scatter_dense_token.run(
            mid,
            perm if has_perm else attn_output,
            out,
            seg,
            widx,
            ranks,
            (H, N, mp, npad, scatter_block, scatter_chunks, has_perm),
            grid=(ml * scatter_chunks, bs),
            warmup=False,
            num_warps=4,
            num_stages=1,
        )
        return out
    if bs == 0 or rows == 0 or (not has_perm and xs0 == H * xs1):
        if not fp32 and rows * H <= 4096:
            bm = min(bm, 1 << max(4, (ml * H - 1).bit_length()))
        if bm == 512 and N <= 32:
            out = torch.empty_strided(
                (S, H, N),
                (H, 1, S * H),
                dtype=attn_output.dtype,
                device=attn_output.device,
            )
        else:
            out = attn_output.new_empty((S, H, N))
        if fp32:
            _zero_fp32.run(
                out,
                (S * H * N, 128),
                grid=(max(1, (S * H * N + 128 - 1) // 128),),
                warmup=False,
                num_warps=4,
                num_stages=1,
            )
            if bs == 0 or rows == 0:
                return out
        tiles = max(1, (ml * H + bm - 1) // bm)
        if bm == 512 and N <= 32:
            _flat_nmajor.run(
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
                    bs,
                    tiles,
                    not fp32,
                ),
                grid=(tiles * bs + (0 if fp32 else 1), nt),
                warmup=False,
                num_warps=4,
                num_stages=2,
            )
        else:
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
                    bs,
                    tiles,
                    not fp32,
                ),
                grid=(tiles * bs + (0 if fp32 else 1), nt),
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
    if fp32:
        _zero_fp32.run(
            mid,
            (rows * H * N, 128),
            grid=(max(1, (rows * H * N + 128 - 1) // 128),),
            warmup=False,
            num_warps=4,
            num_stages=1,
        )
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
            bs,
            tiles,
            not fp32,
        ),
        grid=(tiles * bs + (0 if fp32 else 1), nt),
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
