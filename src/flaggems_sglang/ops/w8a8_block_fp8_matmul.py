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
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied. See the License for the specific language governing
# permissions and limitations under the License.

import torch
import triton
import triton.language as tl


@triton.jit
def _bf16_bits(u):
    e = (u >> 3) & 15
    f = u & 7
    normal = ((e + 120) << 7) | (f << 4)
    sub = tl.where(
        f >= 4,
        (120 << 7) | ((f - 4) << 5),
        tl.where(
            f >= 2, (119 << 7) | ((f - 2) << 6), tl.where(f == 1, 118 << 7, 0)
        ),
    )
    bits = tl.where(e == 0, sub, normal) | ((u & 128) << 8)
    return tl.where((e == 15) & (f == 7), 0x7FC0, bits)


@triton.jit
def _decode(
    X,
    Y,
    PID,
    R: tl.constexpr,
    K: tl.constexpr,
    RP: tl.constexpr,
    KP: tl.constexpr,
    GK: tl.constexpr,
    BK: tl.constexpr,
    BLOCK: tl.constexpr,
    PACKED: tl.constexpr,
):
    p = PID * BLOCK + tl.arange(0, BLOCK // 2) * 2
    r = p // KP
    pk = p % KP
    k = pk // BK * GK + pk % BK
    valid = (r < R) & (k < K) & (pk % BK < GK)
    if PACKED:
        if R == RP and K == KP and GK == BK and R * K % BLOCK == 0:
            word = tl.load(X + p // 2).to(tl.uint32)
        else:
            word = tl.load(X + r * (K // 2) + k // 2, valid, other=0).to(
                tl.uint32
            )
        u0 = word & 255
        u1 = (word >> 8) & 255
    else:
        u0 = tl.load(X + r * K + k, valid, other=0).to(tl.uint32)
        u1 = tl.load(
            X + r * K + k + 1,
            (r < R) & (k + 1 < K) & (pk % BK + 1 < GK),
            other=0,
        ).to(tl.uint32)
    bits = _bf16_bits(u0) | (_bf16_bits(u1) << 16)
    if RP * KP % BLOCK == 0:
        tl.store(Y + p // 2, bits)
    else:
        tl.store(Y + p // 2, bits, p < RP * KP)


@triton.jit
def _decode_both(
    A,
    B,
    AD,
    BD,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    MP: tl.constexpr,
    NP: tl.constexpr,
    KP: tl.constexpr,
    GK: tl.constexpr,
    BK: tl.constexpr,
    BLOCK: tl.constexpr,
    PACKED: tl.constexpr,
):
    pid = tl.program_id(0)
    na = tl.cdiv(MP * KP, BLOCK)
    if pid < na:
        _decode(A, AD, pid, M, K, MP, KP, GK, BK, BLOCK, PACKED)
    else:
        _decode(
            B, BD + MP * KP // 2, pid - na, N, K, NP, KP, GK, BK, BLOCK, PACKED
        )


@triton.jit
def _matmul(
    A,
    B,
    As,
    Bs,
    C,
    M: tl.constexpr,
    N: tl.constexpr,
    KP: tl.constexpr,
    GN: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    AS0: tl.constexpr,
    AS1: tl.constexpr,
    BS0: tl.constexpr,
    BS1: tl.constexpr,
    IEEE: tl.constexpr,
):
    pid = tl.program_id(0)
    rm = pid // tl.cdiv(N, BN) * BM + tl.arange(0, BM)
    rn = pid % tl.cdiv(N, BN) * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    ap = A + rm[:, None] * KP + rk[None, :]
    bp = B + tl.cdiv(M, BM) * BM * KP + rn[None, :] * KP + rk[:, None]
    acc = tl.zeros((BM, BN), tl.float32)
    for i in range(KP // BK):
        if IEEE:
            av = tl.load(ap).to(tl.float32)
            bv = tl.load(bp).to(tl.float32)
            prod = tl.dot(av, bv, input_precision="ieee")
        else:
            av = tl.load(ap)
            bv = tl.load(bp)
            prod = tl.dot(av, bv, out_dtype=tl.float32)
        sa = tl.load(As + rm * AS0 + i * AS1, rm < M, other=0)
        sb = tl.load(Bs + (rn // GN) * BS0 + i * BS1, rn < N, other=0)
        acc = acc + prod * (sa[:, None] * sb[None, :])
        ap += BK
        bp += BK
    tl.store(
        C + rm[:, None] * N + rn[None, :],
        acc,
        (rm[:, None] < M) & (rn[None, :] < N),
    )


@triton.jit
def _split(
    A,
    B,
    As,
    Bs,
    P,
    M: tl.constexpr,
    N: tl.constexpr,
    MP: tl.constexpr,
    NP: tl.constexpr,
    KP: tl.constexpr,
    GN: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    AS0: tl.constexpr,
    AS1: tl.constexpr,
    BS0: tl.constexpr,
    BS1: tl.constexpr,
    SPLIT: tl.constexpr,
    IEEE: tl.constexpr,
):
    pid = tl.program_id(0)
    part = tl.program_id(1)
    rm = pid // tl.cdiv(N, BN) * BM + tl.arange(0, BM)
    rn = pid % tl.cdiv(N, BN) * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    ap = A + rm[:, None] * KP + rk[None, :]
    bp = B + MP * KP + rn[None, :] * KP + rk[:, None]
    nt = KP // BK
    begin = part * nt // SPLIT
    end = (part + 1) * nt // SPLIT
    acc = tl.zeros((BM, BN), tl.float32)
    for i in range(begin, end):
        if IEEE:
            av = tl.load(ap + i * BK).to(tl.float32)
            bv = tl.load(bp + i * BK).to(tl.float32)
            prod = tl.dot(av, bv, input_precision="ieee")
        else:
            av = tl.load(ap + i * BK)
            bv = tl.load(bp + i * BK)
            prod = tl.dot(av, bv, out_dtype=tl.float32)
        sa = tl.load(As + rm * AS0 + i * AS1, rm < M, other=0)
        sb = tl.load(Bs + (rn // GN) * BS0 + i * BS1, rn < N, other=0)
        acc = acc + prod * (sa[:, None] * sb[None, :])
    tl.store(
        P + part * MP * NP + rm[:, None] * NP + rn[None, :],
        acc,
        (rm[:, None] < M) & (rn[None, :] < N),
    )


@triton.jit
def _finish(
    P,
    C,
    M: tl.constexpr,
    N: tl.constexpr,
    MP: tl.constexpr,
    NP: tl.constexpr,
    SPLIT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    p = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    r = p // N
    n = p % N
    acc = tl.zeros((BLOCK,), tl.float32)
    for part in range(SPLIT):
        acc += tl.load(P + part * MP * NP + r * NP + n, p < M * N, other=0)
    tl.store(C + p, acc, p < M * N)


@triton.jit
def _value(u):
    return (
        _bf16_bits(u.to(tl.uint32)).to(tl.uint16).to(tl.bfloat16, bitcast=True)
    )


@triton.jit
def _fused_gemm(
    A,
    B,
    As,
    Bs,
    C,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    GN: tl.constexpr,
    GK: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    AS0: tl.constexpr,
    AS1: tl.constexpr,
    BS0: tl.constexpr,
    BS1: tl.constexpr,
    SPLIT: tl.constexpr,
    MP: tl.constexpr,
    NP: tl.constexpr,
):
    pid = tl.program_id(0)
    part = tl.program_id(1)
    rm = pid // tl.cdiv(N, BN) * BM + tl.arange(0, BM)
    rn = pid % tl.cdiv(N, BN) * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    nt = tl.cdiv(K, GK)
    acc = tl.zeros((BM, BN), tl.float32)
    for i in range(part * nt // SPLIT, (part + 1) * nt // SPLIT):
        kk = i * GK + rk
        av = _value(
            tl.load(
                A + rm[:, None] * K + kk[None, :],
                (rm[:, None] < M) & (rk[None, :] < GK) & (kk[None, :] < K),
                other=0,
            )
        )
        bv = _value(
            tl.load(
                B + rn[None, :] * K + kk[:, None],
                (rn[None, :] < N) & (rk[:, None] < GK) & (kk[:, None] < K),
                other=0,
            )
        )
        prod = tl.dot(av, bv, out_dtype=tl.float32)
        sa = tl.load(As + rm * AS0 + i * AS1, rm < M, other=0)
        sb = tl.load(Bs + (rn // GN) * BS0 + i * BS1, rn < N, other=0)
        scale = sa[:, None] * sb[None, :]
        acc = acc + prod * scale
    if SPLIT == 1:
        tl.store(
            C + rm[:, None] * N + rn[None, :],
            acc,
            (rm[:, None] < M) & (rn[None, :] < N),
        )
    else:
        tl.store(
            C + part * MP * NP + rm[:, None] * NP + rn[None, :],
            acc,
            (rm[:, None] < M) & (rn[None, :] < N),
        )


def _fused(A, B, As, Bs, block_size, output_dtype):
    m, k = A.shape
    n = B.shape[0]
    gn, gk = block_size
    bm = 16 if m <= 8 else 128 if m >= 4096 else 64 if m >= 512 else 32
    bn = 64
    bk = max(32, triton.next_power_of_2(gk))
    mp = triton.cdiv(m, bm) * bm
    np = triton.cdiv(n, bn) * bn
    tiles = triton.cdiv(m, bm) * triton.cdiv(n, bn)
    split = min(4, triton.cdiv(k, gk)) if tiles < 64 else 1
    out = torch.empty((m, n), dtype=output_dtype, device=A.device)
    dest = (
        torch.empty((split, mp, np), dtype=torch.float32, device=A.device)
        if split > 1
        else out
    )
    _fused_gemm[(tiles, split)](
        A.view(torch.uint8),
        B.view(torch.uint8),
        As,
        Bs,
        dest,
        m,
        n,
        k,
        gn,
        gk,
        bm,
        bn,
        bk,
        As.stride(0),
        As.stride(1),
        Bs.stride(0),
        Bs.stride(1),
        split,
        mp,
        np,
        num_warps=4,
        num_stages=1,
        enable_fp_fusion=False,
    )
    if split > 1:
        _finish[(triton.cdiv(m * n, 1024),)](
            dest, out, m, n, mp, np, split, 1024, enable_fp_fusion=False
        )
    return out


@triton.jit
def _gemv(
    A,
    B,
    As,
    Bs,
    P,
    N: tl.constexpr,
    K: tl.constexpr,
    GN: tl.constexpr,
    GK: tl.constexpr,
    BK: tl.constexpr,
    BN: tl.constexpr,
    AS1: tl.constexpr,
    BS0: tl.constexpr,
    BS1: tl.constexpr,
    SPLIT: tl.constexpr,
):
    rn = tl.program_id(0) * BN + tl.arange(0, BN)
    part = tl.program_id(1)
    rk = tl.arange(0, BK)
    nt = tl.cdiv(K, GK)
    acc = tl.zeros((BN,), tl.float32)
    for i in range(part * nt // SPLIT, (part + 1) * nt // SPLIT):
        kk = i * GK + rk
        av = _value(tl.load(A + kk, (kk < K) & (rk < GK), other=0)).to(
            tl.float32
        )
        bv = _value(
            tl.load(
                B + rn[:, None] * K + kk[None, :],
                (rn[:, None] < N) & (kk[None, :] < K) & (rk[None, :] < GK),
                other=0,
            )
        ).to(tl.float32)
        prod = tl.sum(av[None, :] * bv, axis=1)
        sa = tl.load(As + i * AS1)
        sb = tl.load(Bs + (rn // GN) * BS0 + i * BS1, rn < N, other=0)
        acc += prod * (sa * sb)
    tl.store(P + part * N + rn, acc, rn < N)


def _single_row(A, B, As, Bs, block_size, output_dtype):
    k = A.shape[1]
    n = B.shape[0]
    gn, gk = block_size
    split = min(8, triton.cdiv(k, gk))
    partial = torch.empty((split, n), dtype=torch.float32, device=A.device)
    out = torch.empty((1, n), dtype=output_dtype, device=A.device)
    _gemv[(triton.cdiv(n, 16), split)](
        A.view(torch.uint8),
        B.view(torch.uint8),
        As,
        Bs,
        partial,
        n,
        k,
        gn,
        gk,
        triton.next_power_of_2(gk),
        16,
        As.stride(1),
        Bs.stride(0),
        Bs.stride(1),
        split,
        num_warps=4,
        num_stages=1,
        enable_fp_fusion=False,
    )
    _finish[(triton.cdiv(n, 1024),)](
        partial, out, 1, n, 1, n, split, 1024, enable_fp_fusion=False
    )
    return out


def w8a8_block_fp8_matmul(A, B, As, Bs, block_size, output_dtype):
    m, k = A.shape
    if m == 1:
        return _single_row(A, B, As, Bs, block_size, output_dtype)
    if k <= 1024:
        return _fused(A, B, As, Bs, block_size, output_dtype)
    n = B.shape[0]
    gn, gk = block_size
    if m <= 8:
        bm, bn = 16, 64
    elif m >= 4096:
        bm, bn = 128, 64
    elif m >= 512:
        bm, bn = 128, 64
    else:
        bm, bn = 32, 64
    bk = max(32, triton.next_power_of_2(gk))
    mp = triton.cdiv(m, bm) * bm
    np = triton.cdiv(n, bn) * bn
    kp = triton.cdiv(k, gk) * bk
    ieee = False
    dt = torch.float32 if ieee else torch.bfloat16
    ad = torch.empty((mp + np, kp), dtype=dt, device=A.device)
    bd = ad
    out = torch.empty((m, n), dtype=output_dtype, device=A.device)
    packed = (
        k % 2 == 0
        and gk % 2 == 0
        and A.storage_offset() % 2 == 0
        and B.storage_offset() % 2 == 0
    )
    view_dtype = torch.int16 if packed else torch.uint8
    decoded_words = ad.view(torch.int32)
    _decode_both[(triton.cdiv(mp * kp, 1024) + triton.cdiv(np * kp, 1024),)](
        A.view(view_dtype),
        B.view(view_dtype),
        decoded_words,
        decoded_words,
        m,
        n,
        k,
        mp,
        np,
        kp,
        gk,
        bk,
        1024,
        packed,
    )
    tiles = triton.cdiv(m, bm) * triton.cdiv(n, bn)
    if tiles < 64:
        partial = torch.empty(
            (4, mp, np), dtype=torch.float32, device=A.device
        )
        _split[(tiles, 4)](
            ad,
            bd,
            As,
            Bs,
            partial,
            m,
            n,
            mp,
            np,
            kp,
            gn,
            bm,
            bn,
            bk,
            As.stride(0),
            As.stride(1),
            Bs.stride(0),
            Bs.stride(1),
            4,
            ieee,
            num_warps=4,
            num_stages=1,
            enable_fp_fusion=False,
        )
        _finish[(triton.cdiv(m * n, 1024),)](
            partial,
            out,
            m,
            n,
            mp,
            np,
            4,
            1024,
            enable_fp_fusion=False,
        )
    else:
        _matmul[(tiles,)](
            ad,
            bd,
            As,
            Bs,
            out,
            m,
            n,
            kp,
            gn,
            bm,
            bn,
            bk,
            As.stride(0),
            As.stride(1),
            Bs.stride(0),
            Bs.stride(1),
            ieee,
            num_warps=4,
            num_stages=1,
            enable_fp_fusion=False,
        )
    return out


__all__ = ["w8a8_block_fp8_matmul"]
