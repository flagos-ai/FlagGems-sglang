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
def _decode(
    X,
    Y,
    R: tl.constexpr,
    K: tl.constexpr,
    RP: tl.constexpr,
    KP: tl.constexpr,
    GK: tl.constexpr,
    BK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    p = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    r = p // KP
    pk = p % KP
    k = pk // BK * GK + pk % BK
    valid = (r < R) & (k < K) & (pk % BK < GK)
    u = tl.load(X + r * K + k, valid, other=0).to(tl.uint32)
    e = (u >> 3) & 15
    f = u & 7
    bits = ((u & 128) << 24) | ((e + 120) << 23) | (f << 20)
    v = bits.to(tl.float32, bitcast=True)
    sub = f.to(tl.float32) * 0.001953125
    sub = tl.where((u & 128) != 0, -sub, sub)
    v = tl.where(e == 0, sub, v)
    v = tl.where((e == 15) & (f == 7), float("nan"), v)
    v = tl.where(valid, v, 0.0)
    tl.store(Y + p, v, p < RP * KP)


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
):
    pid = tl.program_id(0)
    rm = pid // tl.cdiv(N, BN) * BM + tl.arange(0, BM)
    rn = pid % tl.cdiv(N, BN) * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    ap = A + rm[:, None] * KP + rk[None, :]
    bp = B + rn[None, :] * KP + rk[:, None]
    acc = tl.zeros((BM, BN), tl.float32)
    for i in range(KP // BK):
        av = tl.load(ap)
        bv = tl.load(bp)
        prod = tl.dot(av, bv, out_dtype=tl.float32)
        sa = tl.load(As + rm * AS0 + i * AS1, rm < M, other=0)
        sb = tl.load(Bs + (rn // GN) * BS0 + i * BS1, rn < N, other=0)
        scale = sa[:, None] * sb[None, :]
        acc = acc + prod * scale
        ap += BK
        bp += BK
    tl.store(
        C + rm[:, None] * N + rn[None, :],
        acc,
        (rm[:, None] < M) & (rn[None, :] < N),
    )


def _small_path(A, B, As, Bs, block_size, output_dtype):
    m, k = A.shape
    n = B.shape[0]
    gn, gk = block_size
    bm, bn = 32, 64
    bk = max(32, triton.next_power_of_2(gk))
    mp = triton.cdiv(m, bm) * bm
    np_ = triton.cdiv(n, bn) * bn
    kp = triton.cdiv(k, gk) * bk
    ad = torch.empty((mp, kp), dtype=torch.bfloat16, device=A.device)
    bd = torch.empty((np_, kp), dtype=torch.bfloat16, device=B.device)
    out = torch.empty((m, n), dtype=output_dtype, device=A.device)
    _decode[(triton.cdiv(mp * kp, 1024),)](
        A.view(torch.uint8), ad, m, k, mp, kp, gk, bk, 1024
    )
    _decode[(triton.cdiv(np_ * kp, 1024),)](
        B.view(torch.uint8), bd, n, k, np_, kp, gk, bk, 1024
    )
    _matmul[(triton.cdiv(m, bm) * triton.cdiv(n, bn),)](
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
        num_warps=4,
        num_stages=2,
        enable_fp_fusion=False,
    )
    return out


@triton.jit
def _decode_e4m3fn_to_fp16(x):
    s = x >> 7 & 1
    e = x >> 3 & 15
    m = x & 7
    is_norm_f = (e >= 1).to(tl.float32)
    mantissa = is_norm_f + m.to(tl.float32) * 0.125
    exp_int = tl.maximum(e, 1) - 7
    pow2_bits = exp_int + 127 << 23
    pow2 = pow2_bits.to(tl.float32, bitcast=True)
    sign = 1.0 - 2.0 * s.to(tl.float32)
    return (sign * mantissa * pow2).to(tl.float16)


@triton.jit
def _convert_A_kernel(X, Y, numel, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < numel
    x = tl.load(X + offs, mask=mask, other=0)
    y = _decode_e4m3fn_to_fp16(x.to(tl.int32))
    tl.store(Y + offs, y, mask=mask)


@triton.jit
def _convert_B_transpose_kernel(
    B,
    Bt,
    N,
    K,
    stride_bn,
    stride_bk,
    stride_btk,
    stride_btn,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    mask_n = offs_n < N
    mask_k = offs_k < K
    b_ptrs = B + offs_n[:, None] * stride_bn + offs_k[None, :] * stride_bk
    b_raw = tl.load(b_ptrs, mask=mask_n[:, None] & mask_k[None, :], other=0)
    b = _decode_e4m3fn_to_fp16(b_raw.to(tl.int32))
    b_t = tl.trans(b)
    bt_ptrs = Bt + offs_k[:, None] * stride_btk + offs_n[None, :] * stride_btn
    tl.store(bt_ptrs, b_t, mask=mask_k[:, None] & mask_n[None, :])


@triton.jit
def _w8a8_gemm_kernel(
    A,
    Bt,
    As,
    Bs,
    C,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_btk,
    stride_btn,
    stride_asm,
    stride_ask,
    stride_bsn,
    stride_bsk,
    stride_cm,
    stride_cn,
    block_n: tl.constexpr,
    block_k: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
    EVEN_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + pid % group_size_m
    pid_n = pid % num_pid_in_group // group_size_m
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_m = tl.max_contiguous(tl.multiple_of(offs_m, BLOCK_M), BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_n = tl.max_contiguous(tl.multiple_of(offs_n, BLOCK_N), BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    k_tiles = tl.cdiv(K, block_k)
    for i in range(0, k_tiles):
        k_base = i * block_k
        if EVEN_M:
            a_s = tl.load(As + offs_m * stride_asm + i * stride_ask)
        else:
            a_s = tl.load(
                As + offs_m * stride_asm + i * stride_ask,
                mask=offs_m < M,
                other=0.0,
            )
        if block_n % BLOCK_N == 0:
            n_block = pid_n * BLOCK_N // block_n
            b_s = tl.load(Bs + n_block * stride_bsn + i * stride_bsk)
            scale = a_s[:, None] * b_s
        else:
            n_block = offs_n // block_n
            if EVEN_N:
                b_s = tl.load(Bs + n_block * stride_bsn + i * stride_bsk)
            else:
                b_s = tl.load(
                    Bs + n_block * stride_bsn + i * stride_bsk,
                    mask=offs_n < N,
                    other=0.0,
                )
            scale = a_s[:, None] * b_s[None, :]
        dot_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for kk in range(0, block_k, BLOCK_K):
            k_offs = k_base + kk + offs_k
            if EVEN_K:
                if EVEN_M:
                    a = tl.load(
                        A
                        + offs_m[:, None] * stride_am
                        + k_offs[None, :] * stride_ak
                    )
                else:
                    a = tl.load(
                        A
                        + offs_m[:, None] * stride_am
                        + k_offs[None, :] * stride_ak,
                        mask=offs_m[:, None] < M,
                        other=0.0,
                    )
                if EVEN_N:
                    b = tl.load(
                        Bt
                        + k_offs[:, None] * stride_btk
                        + offs_n[None, :] * stride_btn
                    )
                else:
                    b = tl.load(
                        Bt
                        + k_offs[:, None] * stride_btk
                        + offs_n[None, :] * stride_btn,
                        mask=offs_n[None, :] < N,
                        other=0.0,
                    )
            else:
                mask_k = (k_offs < K) & (k_offs < k_base + block_k)
                if EVEN_M:
                    a = tl.load(
                        A
                        + offs_m[:, None] * stride_am
                        + k_offs[None, :] * stride_ak,
                        mask=mask_k[None, :],
                        other=0.0,
                    )
                else:
                    a = tl.load(
                        A
                        + offs_m[:, None] * stride_am
                        + k_offs[None, :] * stride_ak,
                        mask=(offs_m[:, None] < M) & mask_k[None, :],
                        other=0.0,
                    )
                if EVEN_N:
                    b = tl.load(
                        Bt
                        + k_offs[:, None] * stride_btk
                        + offs_n[None, :] * stride_btn,
                        mask=mask_k[:, None],
                        other=0.0,
                    )
                else:
                    b = tl.load(
                        Bt
                        + k_offs[:, None] * stride_btk
                        + offs_n[None, :] * stride_btn,
                        mask=mask_k[:, None] & (offs_n[None, :] < N),
                        other=0.0,
                    )
            dot_acc += tl.dot(a, b, out_dtype=tl.float32)
        acc += dot_acc * scale
    c_ptrs = C + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    if EVEN_M and EVEN_N:
        tl.store(c_ptrs, acc)
    elif EVEN_M:
        tl.store(c_ptrs, acc, mask=offs_n[None, :] < N)
    elif EVEN_N:
        tl.store(c_ptrs, acc, mask=offs_m[:, None] < M)
    else:
        tl.store(
            c_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N)
        )


@triton.jit
def _w8a8_gemm_persistent_kernel(
    A,
    Bt,
    As,
    Bs,
    C,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_btk,
    stride_btn,
    stride_asm,
    stride_ask,
    stride_bsn,
    stride_bsk,
    stride_cm,
    stride_cn,
    block_n: tl.constexpr,
    block_k: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
    EVEN_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    NUM_SMS: tl.constexpr,
):
    start_pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid = num_pid_m * num_pid_n
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    for tile_id in range(start_pid, num_pid, NUM_SMS):
        group_id = tile_id // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + tile_id % group_size_m
        pid_n = tile_id % num_pid_in_group // group_size_m
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_m = tl.max_contiguous(tl.multiple_of(offs_m, BLOCK_M), BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_n = tl.max_contiguous(tl.multiple_of(offs_n, BLOCK_N), BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        k_tiles = tl.cdiv(K, block_k)
        for i in range(0, k_tiles):
            k_base = i * block_k
            if EVEN_M:
                a_s = tl.load(As + offs_m * stride_asm + i * stride_ask)
            else:
                a_s = tl.load(
                    As + offs_m * stride_asm + i * stride_ask,
                    mask=offs_m < M,
                    other=0.0,
                )
            if block_n % BLOCK_N == 0:
                n_block = pid_n * BLOCK_N // block_n
                b_s = tl.load(Bs + n_block * stride_bsn + i * stride_bsk)
                scale = a_s[:, None] * b_s
            else:
                n_block = offs_n // block_n
                if EVEN_N:
                    b_s = tl.load(Bs + n_block * stride_bsn + i * stride_bsk)
                else:
                    b_s = tl.load(
                        Bs + n_block * stride_bsn + i * stride_bsk,
                        mask=offs_n < N,
                        other=0.0,
                    )
                scale = a_s[:, None] * b_s[None, :]
            dot_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            for kk in range(0, block_k, BLOCK_K):
                k_offs = k_base + kk + offs_k
                if EVEN_K:
                    if EVEN_M:
                        a = tl.load(
                            A
                            + offs_m[:, None] * stride_am
                            + k_offs[None, :] * stride_ak
                        )
                    else:
                        a = tl.load(
                            A
                            + offs_m[:, None] * stride_am
                            + k_offs[None, :] * stride_ak,
                            mask=offs_m[:, None] < M,
                            other=0.0,
                        )
                    if EVEN_N:
                        b = tl.load(
                            Bt
                            + k_offs[:, None] * stride_btk
                            + offs_n[None, :] * stride_btn
                        )
                    else:
                        b = tl.load(
                            Bt
                            + k_offs[:, None] * stride_btk
                            + offs_n[None, :] * stride_btn,
                            mask=offs_n[None, :] < N,
                            other=0.0,
                        )
                else:
                    mask_k = (k_offs < K) & (k_offs < k_base + block_k)
                    if EVEN_M:
                        a = tl.load(
                            A
                            + offs_m[:, None] * stride_am
                            + k_offs[None, :] * stride_ak,
                            mask=mask_k[None, :],
                            other=0.0,
                        )
                    else:
                        a = tl.load(
                            A
                            + offs_m[:, None] * stride_am
                            + k_offs[None, :] * stride_ak,
                            mask=(offs_m[:, None] < M) & mask_k[None, :],
                            other=0.0,
                        )
                    if EVEN_N:
                        b = tl.load(
                            Bt
                            + k_offs[:, None] * stride_btk
                            + offs_n[None, :] * stride_btn,
                            mask=mask_k[:, None],
                            other=0.0,
                        )
                    else:
                        b = tl.load(
                            Bt
                            + k_offs[:, None] * stride_btk
                            + offs_n[None, :] * stride_btn,
                            mask=mask_k[:, None] & (offs_n[None, :] < N),
                            other=0.0,
                        )
                dot_acc += tl.dot(a, b, out_dtype=tl.float32)
            acc += dot_acc * scale
        c_ptrs = C + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
        if EVEN_M and EVEN_N:
            tl.store(c_ptrs, acc)
        elif EVEN_M:
            tl.store(c_ptrs, acc, mask=offs_n[None, :] < N)
        elif EVEN_N:
            tl.store(c_ptrs, acc, mask=offs_m[:, None] < M)
        else:
            tl.store(
                c_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N)
            )


@triton.jit
def _w8a8_gemm_splitk_kernel(
    A,
    Bt,
    As,
    Bs,
    C,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_btk,
    stride_btn,
    stride_asm,
    stride_ask,
    stride_bsn,
    stride_bsk,
    stride_cm,
    stride_cn,
    stride_csplit,
    block_n: tl.constexpr,
    block_k: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SPLIT_K: tl.constexpr,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
    EVEN_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(0)
    split_id = tl.program_id(1)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + pid % group_size_m
    pid_n = pid % num_pid_in_group // group_size_m
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_m = tl.max_contiguous(tl.multiple_of(offs_m, BLOCK_M), BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_n = tl.max_contiguous(tl.multiple_of(offs_n, BLOCK_N), BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    k_tiles = tl.cdiv(K, block_k)
    for i in range(split_id, k_tiles, SPLIT_K):
        k_base = i * block_k
        if EVEN_M:
            a_s = tl.load(As + offs_m * stride_asm + i * stride_ask)
        else:
            a_s = tl.load(
                As + offs_m * stride_asm + i * stride_ask,
                mask=offs_m < M,
                other=0.0,
            )
        if block_n % BLOCK_N == 0:
            n_block = pid_n * BLOCK_N // block_n
            b_s = tl.load(Bs + n_block * stride_bsn + i * stride_bsk)
            scale = a_s[:, None] * b_s
        else:
            n_block = offs_n // block_n
            if EVEN_N:
                b_s = tl.load(Bs + n_block * stride_bsn + i * stride_bsk)
            else:
                b_s = tl.load(
                    Bs + n_block * stride_bsn + i * stride_bsk,
                    mask=offs_n < N,
                    other=0.0,
                )
            scale = a_s[:, None] * b_s[None, :]
        dot_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for kk in range(0, block_k, BLOCK_K):
            k_offs = k_base + kk + offs_k
            if EVEN_K:
                if EVEN_M:
                    a = tl.load(
                        A
                        + offs_m[:, None] * stride_am
                        + k_offs[None, :] * stride_ak
                    )
                else:
                    a = tl.load(
                        A
                        + offs_m[:, None] * stride_am
                        + k_offs[None, :] * stride_ak,
                        mask=offs_m[:, None] < M,
                        other=0.0,
                    )
                if EVEN_N:
                    b = tl.load(
                        Bt
                        + k_offs[:, None] * stride_btk
                        + offs_n[None, :] * stride_btn
                    )
                else:
                    b = tl.load(
                        Bt
                        + k_offs[:, None] * stride_btk
                        + offs_n[None, :] * stride_btn,
                        mask=offs_n[None, :] < N,
                        other=0.0,
                    )
            else:
                mask_k = (k_offs < K) & (k_offs < k_base + block_k)
                if EVEN_M:
                    a = tl.load(
                        A
                        + offs_m[:, None] * stride_am
                        + k_offs[None, :] * stride_ak,
                        mask=mask_k[None, :],
                        other=0.0,
                    )
                else:
                    a = tl.load(
                        A
                        + offs_m[:, None] * stride_am
                        + k_offs[None, :] * stride_ak,
                        mask=(offs_m[:, None] < M) & mask_k[None, :],
                        other=0.0,
                    )
                if EVEN_N:
                    b = tl.load(
                        Bt
                        + k_offs[:, None] * stride_btk
                        + offs_n[None, :] * stride_btn,
                        mask=mask_k[:, None],
                        other=0.0,
                    )
                else:
                    b = tl.load(
                        Bt
                        + k_offs[:, None] * stride_btk
                        + offs_n[None, :] * stride_btn,
                        mask=mask_k[:, None] & (offs_n[None, :] < N),
                        other=0.0,
                    )
            dot_acc += tl.dot(a, b, out_dtype=tl.float32)
        acc += dot_acc * scale
    c_ptrs = (
        C
        + split_id * stride_csplit
        + offs_m[:, None] * stride_cm
        + offs_n[None, :] * stride_cn
    )
    if EVEN_M and EVEN_N:
        tl.store(c_ptrs, acc)
    elif EVEN_M:
        tl.store(c_ptrs, acc, mask=offs_n[None, :] < N)
    elif EVEN_N:
        tl.store(c_ptrs, acc, mask=offs_m[:, None] < M)
    else:
        tl.store(
            c_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N)
        )


@triton.jit
def _reduce_splitk_kernel(
    C_partial,
    C_out,
    numel,
    stride_csplit,
    SPLIT_K: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    OUT_DTYPE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < numel
    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    for i in tl.static_range(SPLIT_K):
        acc += tl.load(
            C_partial + i * stride_csplit + offs, mask=mask, other=0.0
        )
    tl.store(C_out + offs, acc.to(OUT_DTYPE), mask=mask)


def _convert_A(A):
    A_fp16 = torch.empty(
        (A.shape[0], A.shape[1]), dtype=torch.float16, device=A.device
    )
    numel = A.numel()
    grid = (triton.cdiv(numel, 1024),)
    _convert_A_kernel[grid](
        A.view(torch.uint8), A_fp16, numel, BLOCK_SIZE=1024, num_warps=4
    )
    return A_fp16


def _convert_Bt(B):
    N, K = B.shape
    Bt = torch.empty((K, N), dtype=torch.float16, device=B.device)
    grid = (triton.cdiv(N, 64), triton.cdiv(K, 64))
    _convert_B_transpose_kernel[grid](
        B.view(torch.uint8),
        Bt,
        N,
        K,
        B.stride(0),
        B.stride(1),
        Bt.stride(0),
        Bt.stride(1),
        64,
        64,
        num_warps=4,
    )
    return Bt


def _large_path(A, B, As, Bs, block_size, output_dtype):
    M, K = A.shape
    N = B.shape[0]
    block_n = int(block_size[0])
    block_k = int(block_size[1])
    A_fp16 = _convert_A(A)
    Bt_fp16 = _convert_Bt(B)
    if M <= 16:
        BLOCK_M = 16
        num_warps = 4
        num_stages = 2
    elif M <= 32:
        BLOCK_M = 32
        num_warps = 4
        num_stages = 2
    elif M < 256:
        BLOCK_M = 64
        num_warps = 4
        num_stages = 2
    else:
        BLOCK_M = 128
        num_warps = 8
        num_stages = 3
    BLOCK_N = 128
    BLOCK_K = 64
    if block_k < BLOCK_K:
        BLOCK_K = 1 << (block_k - 1).bit_length()
        if BLOCK_K < 16:
            BLOCK_K = 16
    EVEN_M = M % BLOCK_M == 0
    EVEN_N = N % BLOCK_N == 0
    EVEN_K = K % block_k == 0 and block_k % BLOCK_K == 0
    num_pid_m = triton.cdiv(M, BLOCK_M)
    num_pid_n = triton.cdiv(N, BLOCK_N)
    num_pid = num_pid_m * num_pid_n
    if M > 512:
        C = torch.empty((M, N), device=A.device, dtype=output_dtype)
        NUM_SMS = torch.cuda.get_device_properties(
            A.device
        ).multi_processor_count
        grid = (min(num_pid, NUM_SMS),)
        _w8a8_gemm_persistent_kernel[grid](
            A_fp16,
            Bt_fp16,
            As,
            Bs,
            C,
            M,
            N,
            K,
            A_fp16.stride(0),
            A_fp16.stride(1),
            Bt_fp16.stride(0),
            Bt_fp16.stride(1),
            As.stride(0),
            As.stride(1),
            Bs.stride(0),
            Bs.stride(1),
            C.stride(0),
            C.stride(1),
            block_n,
            block_k,
            BLOCK_M,
            BLOCK_N,
            BLOCK_K,
            EVEN_M,
            EVEN_N,
            EVEN_K,
            16,
            NUM_SMS,
            num_warps=num_warps,
            num_stages=num_stages,
        )
        return C
    else:
        SPLIT_K = 1
        if M <= 512 and num_pid < 128:
            k_tiles = triton.cdiv(K, block_k)
            SPLIT_K = min(8, max(1, 128 // num_pid), k_tiles)
            if SPLIT_K < 2:
                SPLIT_K = 1
        if SPLIT_K > 1:
            C_partial = torch.empty(
                (SPLIT_K, M, N), device=A.device, dtype=torch.float32
            )
            grid = (num_pid, SPLIT_K)
            _w8a8_gemm_splitk_kernel[grid](
                A_fp16,
                Bt_fp16,
                As,
                Bs,
                C_partial,
                M,
                N,
                K,
                A_fp16.stride(0),
                A_fp16.stride(1),
                Bt_fp16.stride(0),
                Bt_fp16.stride(1),
                As.stride(0),
                As.stride(1),
                Bs.stride(0),
                Bs.stride(1),
                N,
                1,
                M * N,
                block_n,
                block_k,
                BLOCK_M,
                BLOCK_N,
                BLOCK_K,
                SPLIT_K,
                EVEN_M,
                EVEN_N,
                EVEN_K,
                8,
                num_warps=num_warps,
                num_stages=num_stages,
            )
            C = torch.empty((M, N), device=A.device, dtype=output_dtype)
            numel = M * N
            grid_cast = (triton.cdiv(numel, 2048),)
            if output_dtype == torch.float16:
                out_tl = tl.float16
            elif output_dtype == torch.bfloat16:
                out_tl = tl.bfloat16
            else:
                out_tl = tl.float32
            _reduce_splitk_kernel[grid_cast](
                C_partial,
                C,
                numel,
                M * N,
                SPLIT_K=SPLIT_K,
                BLOCK_SIZE=2048,
                OUT_DTYPE=out_tl,
                num_warps=8,
            )
            return C
        else:
            C = torch.empty((M, N), device=A.device, dtype=output_dtype)
            grid = (num_pid,)
            _w8a8_gemm_kernel[grid](
                A_fp16,
                Bt_fp16,
                As,
                Bs,
                C,
                M,
                N,
                K,
                A_fp16.stride(0),
                A_fp16.stride(1),
                Bt_fp16.stride(0),
                Bt_fp16.stride(1),
                As.stride(0),
                As.stride(1),
                Bs.stride(0),
                Bs.stride(1),
                C.stride(0),
                C.stride(1),
                block_n,
                block_k,
                BLOCK_M,
                BLOCK_N,
                BLOCK_K,
                EVEN_M,
                EVEN_N,
                EVEN_K,
                8,
                num_warps=num_warps,
                num_stages=num_stages,
            )
            return C


def w8a8_block_fp8_matmul(A, B, As, Bs, block_size, output_dtype):
    if not A.is_contiguous():
        A = A.contiguous()
    if not B.is_contiguous():
        B = B.contiguous()
    m = A.shape[0]
    if m <= 64:
        return _small_path(A, B, As, Bs, block_size, output_dtype)
    return _large_path(A, B, As, Bs, block_size, output_dtype)


__all__ = ["w8a8_block_fp8_matmul"]
