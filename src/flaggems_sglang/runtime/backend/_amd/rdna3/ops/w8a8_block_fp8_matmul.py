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

"""w8a8_block_fp8_matmul for RDNA3 (gfx1100).

With 128 x 128 scale blocks and K and N multiples of 128, A is decoded to
fp16 (exactly: e4m3 fits fp16) and each 128-wide K block is summed on the
matrix units on its own, then scaled by As * Bs and added with an fma, so
products and scales stay exact in float32.

- Up to 192 rows (384 for N <= 2048), B stays fp8 and is decoded in
  registers, four K values per 32-bit load, as the left operand of B @ A^T;
  K blocks are split across programs for small grids and the parts added in
  a second kernel.
- Above, B is decoded once too (rows padded by 32 elements) and the product
  runs in 128 x 128 tiles whose accumulator is kept as two 64-column halves
  sharing the A tile, so the per-block partial sum needs half a tile of
  registers.

One row against at most 2048 columns, and all other cases, follow the
generic op.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _e4m3_to_fp16(b32):
    lo = b32 & 0x7F
    e = lo >> 3
    den = ((lo << 7).to(tl.uint16)).to(tl.float16, bitcast=True) * tl.full(
        (1,), 256.0, tl.float16
    )
    nrm = (((lo << 7) + 0x2000).to(tl.uint16)).to(tl.float16, bitcast=True)
    y = tl.where(e == 0, den, nrm)
    isnan = lo == 0x7F
    y = tl.where(isnan, float("nan"), y.to(tl.float32)).to(tl.float16)
    return tl.where(b32 >= 0x80, -y, y)


@triton.jit
def _unpack4(w):
    b0 = _e4m3_to_fp16(w & 0xFF)
    b1 = _e4m3_to_fp16((w >> 8) & 0xFF)
    b2 = _e4m3_to_fp16((w >> 16) & 0xFF)
    b3 = _e4m3_to_fp16((w >> 24) & 0xFF)
    return tl.interleave(tl.interleave(b0, b2), tl.interleave(b1, b3))


@triton.jit
def _decode_rows(
    X, Y, R, pid, K: tl.constexpr, LD: tl.constexpr, BLOCK: tl.constexpr
):
    # X: fp8 [R, K] as uint32 words of four; Y: fp16 [R, LD]
    K4: tl.constexpr = K // 4
    p = pid * BLOCK + tl.arange(0, BLOCK)
    mask = p < R * K4
    y = _unpack4(tl.load(X + p, mask=mask, other=0))
    dst = (p // K4).to(tl.int64) * LD + p % K4 * 4
    tl.store(
        Y + dst[:, None] + tl.arange(0, 4)[None, :],
        tl.reshape(y, (BLOCK, 4)),
        mask=mask[:, None],
    )


@triton.jit
def _decode_both_kernel(
    XA,
    XB,
    YA,
    YB,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    LD: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    na = tl.cdiv(M * (K // 4), BLOCK)
    if pid < na:
        _decode_rows(XA, YA, M, pid, K, LD, BLOCK)
    else:
        _decode_rows(XB, YB, N, pid - na, K, LD, BLOCK)


@triton.jit
def _halves_kernel(
    A,
    B,
    As,
    Bs,
    C,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    LD: tl.constexpr,
    AS0,
    AS1,
    BS0,
    BS1,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    # A: fp16 [M, LD]; B: fp16 [N, LD]; one scale per 128 x 128 block of B,
    # one per row and 128 columns of A.
    HALF: tl.constexpr = 64
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n: tl.constexpr = N // 128
    in_group = GROUP_M * num_pid_n
    first_m = pid // in_group * GROUP_M
    group_m = tl.minimum(num_pid_m - first_m, GROUP_M)
    pid_m = first_m + pid % in_group % group_m
    pid_n = pid % in_group // group_m
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * 128 + tl.arange(0, HALF)
    rk = tl.arange(0, BLOCK_K)
    m_ok = rm < M
    a_ptrs = A + rm[:, None].to(tl.int64) * LD + rk[None, :]
    b0_ptrs = B + rn[None, :].to(tl.int64) * LD + rk[:, None]
    b1_ptrs = b0_ptrs + HALF * LD
    sa_ptrs = As + rm * AS0
    sb_ptr = Bs + pid_n * BS0
    acc0 = tl.zeros((BLOCK_M, HALF), dtype=tl.float32)
    acc1 = tl.zeros((BLOCK_M, HALF), dtype=tl.float32)
    for kb in range(K // 128):
        sa = tl.load(sa_ptrs + kb * AS1, mask=m_ok, other=0.0)
        scale = sa[:, None] * tl.load(sb_ptr + kb * BS1)
        part = tl.zeros((BLOCK_M, HALF), dtype=tl.float32)
        for kk in tl.static_range(128 // BLOCK_K):
            a = tl.load(a_ptrs + kk * BLOCK_K, mask=m_ok[:, None], other=0.0)
            part = tl.dot(a, tl.load(b0_ptrs + kk * BLOCK_K), part)
        acc0 = tl.fma(part, scale, acc0)
        part = tl.zeros((BLOCK_M, HALF), dtype=tl.float32)
        for kk in tl.static_range(128 // BLOCK_K):
            a = tl.load(a_ptrs + kk * BLOCK_K, mask=m_ok[:, None], other=0.0)
            part = tl.dot(a, tl.load(b1_ptrs + kk * BLOCK_K), part)
        acc1 = tl.fma(part, scale, acc1)
        a_ptrs += 128
        b0_ptrs += 128
        b1_ptrs += 128
    c_ptrs = C + rm[:, None].to(tl.int64) * N + rn[None, :]
    ty = C.dtype.element_ty
    tl.store(c_ptrs, acc0.to(ty), mask=m_ok[:, None])
    tl.store(c_ptrs + HALF, acc1.to(ty), mask=m_ok[:, None])


@triton.jit
def _decode_kernel(
    X, Y, R, K: tl.constexpr, LD: tl.constexpr, BLOCK: tl.constexpr
):
    _decode_rows(X, Y, R, tl.program_id(0), K, LD, BLOCK)


@triton.jit
def _swap_kernel(
    A,
    B,
    As,
    Bs,
    C,
    P,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    AS0,
    AS1,
    BS0,
    BS1,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SPLIT: tl.constexpr,
):
    # C^T tile [BLOCK_N, BLOCK_M] = B tile @ A^T tile. A: fp16 [M, K]; B: the
    # fp8 weight [N, K] as uint32 words of four K values. Split s takes
    # every SPLIT-th 128-wide K block from block s.
    pid_n = tl.program_id(0)
    split = tl.program_id(1)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rm = tl.program_id(2) * BLOCK_M + tl.arange(0, BLOCK_M)
    rk = tl.arange(0, BLOCK_K)
    m_ok = rm < M
    b_ptrs = (
        B
        + rn[:, None].to(tl.int64) * (K // 4)
        + tl.arange(0, BLOCK_K // 4)[None, :]
    )
    a_ptrs = A + rm[None, :].to(tl.int64) * K + rk[:, None]
    sb_ptr = Bs + (pid_n * BLOCK_N // 128) * BS0
    acc = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)
    for kb in range(split, K // 128, SPLIT):
        part = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)
        for kk in tl.static_range(128 // BLOCK_K):
            k = kb * 128 + kk * BLOCK_K
            b = _unpack4(tl.load(b_ptrs + k // 4))
            a = tl.load(a_ptrs + k, mask=m_ok[None, :], other=0.0)
            part = tl.dot(b, a, part)
        sa = tl.load(As + rm * AS0 + kb * AS1, mask=m_ok, other=0.0)
        acc = tl.fma(part, tl.load(sb_ptr + kb * BS1) * sa[None, :], acc)
    rows = rm[None, :].to(tl.int64)
    if SPLIT == 1:
        c_ptrs = C + rows * N + rn[:, None]
        tl.store(c_ptrs, acc.to(C.dtype.element_ty), mask=m_ok[None, :])
    else:
        p_ptrs = P + (split * M + rows) * N + rn[:, None]
        tl.store(p_ptrs, acc, mask=m_ok[None, :])


@triton.jit
def _sum_splits_kernel(P, C, total, SPLIT: tl.constexpr, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < total
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for split in tl.static_range(SPLIT):
        acc += tl.load(P + split * total + offs, mask=mask, other=0.0)
    tl.store(C + offs, acc.to(C.dtype.element_ty), mask=mask)


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


def _generic(A, B, As, Bs, block_size, output_dtype):
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


# Up to this many rows B is decoded in registers instead of once; narrow B
# leaves the 128 x 128 tiles too few programs for twice as long.
_SMALL_MAX_M = 192


def _fast_ok(A, B, block_size):
    k = A.shape[1]
    return (
        list(block_size) == [128, 128]
        and k % 128 == 0
        and B.shape[0] % 128 == 0
        and A.storage_offset() % 4 == 0
        and B.storage_offset() % 4 == 0
    )


def _small_config(m, n):
    """BLOCK_M, BLOCK_N, BLOCK_K, num_warps and splits of the K blocks."""
    if m <= 16:
        if n > 4096:
            return 16, 32, 128, 4, 2
        return 16, 64, 128, 4, max(1, min(8, 16384 // n))
    return 64, 32, 128, 8, 4 if n <= 2048 else 2


def _small(A, B, As, Bs, output_dtype):
    m, k = A.shape
    n = B.shape[0]
    bm, bn, bk, nw, splits = _small_config(m, n)
    splits = max(1, min(splits, k // 128))
    ad = torch.empty((m, k), dtype=torch.float16, device=A.device)
    _decode_kernel[(triton.cdiv(m * (k // 4), 1024),)](
        A.view(torch.uint8).view(torch.uint32),
        ad,
        m,
        k,
        k,
        BLOCK=1024,
        num_warps=4,
    )
    out = torch.empty((m, n), dtype=output_dtype, device=A.device)
    part = (
        torch.empty((splits, m, n), dtype=torch.float32, device=A.device)
        if splits > 1
        else out
    )
    _swap_kernel[(n // bn, splits, triton.cdiv(m, bm))](
        ad,
        B.view(torch.uint8).view(torch.uint32),
        As,
        Bs,
        out,
        part,
        m,
        n,
        k,
        As.stride(0),
        As.stride(1),
        Bs.stride(0),
        Bs.stride(1),
        BLOCK_M=bm,
        BLOCK_N=bn,
        BLOCK_K=bk,
        SPLIT=splits,
        num_warps=nw,
        num_stages=2,
    )
    if splits > 1:
        _sum_splits_kernel[(triton.cdiv(m * n, 1024),)](
            part, out, m * n, SPLIT=splits, BLOCK=1024, num_warps=4
        )
    return out


def _halves(A, B, As, Bs, output_dtype):
    m, k = A.shape
    n = B.shape[0]
    ld = k + 32
    ad = torch.empty((m, ld), dtype=torch.float16, device=A.device)
    bd = torch.empty((n, ld), dtype=torch.float16, device=A.device)
    _decode_both_kernel[
        (triton.cdiv(m * (k // 4), 1024) + triton.cdiv(n * (k // 4), 1024),)
    ](
        A.view(torch.uint8).view(torch.uint32),
        B.view(torch.uint8).view(torch.uint32),
        ad,
        bd,
        m,
        n,
        k,
        ld,
        BLOCK=1024,
        num_warps=4,
    )
    out = torch.empty((m, n), dtype=output_dtype, device=A.device)
    _halves_kernel[(triton.cdiv(m, 128) * (n // 128),)](
        ad,
        bd,
        As,
        Bs,
        out,
        m,
        n,
        k,
        ld,
        As.stride(0),
        As.stride(1),
        Bs.stride(0),
        Bs.stride(1),
        BLOCK_M=128,
        BLOCK_K=32,
        GROUP_M=8,
        num_warps=8,
        num_stages=2,
    )
    return out


def w8a8_block_fp8_matmul(A, B, As, Bs, block_size, output_dtype):
    # the generic kernels index A and B as dense rows
    A = A.contiguous()
    B = B.contiguous()
    m = A.shape[0]
    n = B.shape[0]
    if not _fast_ok(A, B, block_size) or (m == 1 and n <= 2048):
        return _generic(A, B, As, Bs, block_size, output_dtype)
    if m <= _SMALL_MAX_M or (n <= 2048 and m <= 2 * _SMALL_MAX_M):
        return _small(A, B, As, Bs, output_dtype)
    return _halves(A, B, As, Bs, output_dtype)


__all__ = ["w8a8_block_fp8_matmul"]
