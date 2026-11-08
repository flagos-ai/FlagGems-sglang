# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Ascend AWQ GEMM with explicit shape specializations and strided fallbacks.

Selected contiguous shapes use halfword FP32 dequantization and IEEE Cube
GEMM. The fused vector path uses group-local Split-K. Core counts come from
runtime device properties; output and scratch are allocated per invocation.
"""

import torch
import torch_npu
import triton
import triton.language as tl
from triton.runtime import driver


@triton.jit
def _unpack_constant(packed):
    # q0..q7 are logical AWQ columns, not consecutive physical nibbles.
    q0 = packed & 15
    q1 = (packed >> 16) & 15
    q2 = (packed >> 4) & 15
    q3 = (packed >> 20) & 15
    q4 = (packed >> 8) & 15
    q5 = (packed >> 24) & 15
    q6 = (packed >> 12) & 15
    q7 = (packed >> 28) & 15
    # Each intermediate still keeps a packed word's columns together.
    # [0,4] with [2,6] -> [0,2,4,6], then interleave even/odd.
    even = tl.interleave(tl.interleave(q0, q4), tl.interleave(q2, q6))
    odd = tl.interleave(tl.interleave(q1, q5), tl.interleave(q3, q7))
    return tl.interleave(even, odd).to(tl.float32)


@triton.jit
def _awq_sum_flat(
    P, OUT, SIZE: tl.constexpr, SPLIT: tl.constexpr, BLOCK: tl.constexpr
):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    acc = tl.zeros((BLOCK,), tl.float32)
    for part in range(SPLIT):
        acc += tl.load(P + part * SIZE + index, index < SIZE, 0)
    tl.store(OUT + index, acc, index < SIZE)


@triton.jit
def _awq_zero(OUT, SIZE: tl.constexpr, BLOCK: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(OUT + index, 0.0, index < SIZE)


def _device_cores():
    device = torch_npu.npu.current_device()
    props = driver.active.utils.get_device_properties(device)
    aic = props.get("num_aicore", 0) if isinstance(props, dict) else 0
    aiv = props.get("num_vectorcore", 0) if isinstance(props, dict) else 0
    aic = aic if isinstance(aic, int) and aic > 0 else 0
    aiv = aiv if isinstance(aiv, int) and aiv > 0 else 0
    return device, aic, aiv


def _program_count(total, cores):
    return min(total, cores) if cores > 0 else total


@triton.jit
def _awq_cube(
    A,
    Q,
    S,
    Z,
    P,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    G: tl.constexpr,
    AM: tl.constexpr,
    AK: tl.constexpr,
    QK: tl.constexpr,
    QN: tl.constexpr,
    SG: tl.constexpr,
    SN: tl.constexpr,
    ZG: tl.constexpr,
    ZN: tl.constexpr,
    SPLIT: tl.constexpr,
    METADATA_KEY: tl.constexpr,
    DEVICE: tl.constexpr,
    AIC: tl.constexpr,
    AIV: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    total: tl.constexpr = ((M + BM - 1) // BM) * ((N + BN - 1) // BN) * SPLIT
    programs = tl.num_programs(0)
    for work in range(tl.program_id(0), total, programs):
        tile = work // SPLIT
        part = work % SPLIT
        rm = (tile // tl.cdiv(N, BN)) * BM + tl.arange(0, BM)
        rn = (tile % tl.cdiv(N, BN)) * BN + tl.arange(0, BN)
        rp = (tile % tl.cdiv(N, BN)) * (BN // 8) + tl.arange(0, BN // 8)
        acc = tl.zeros((BM, BN), tl.float32)
        begin = part * (K // G) // SPLIT
        end = (part + 1) * (K // G) // SPLIT
        for group in range(begin, end):
            packed_zero = tl.load(Z + group * ZG + rp * ZN, rp < N // 8, 0)
            zero = _unpack_constant(packed_zero)
            scale = tl.load(S + group * SG + rn * SN, rn < N, 0).to(tl.float32)
            for block in range(tl.cdiv(G, BK)):
                local_k = block * BK + tl.arange(0, BK)
                rk = group * G + local_k
                a = tl.load(
                    A + rm[:, None] * AM + rk[None, :] * AK,
                    (rm[:, None] < M) & (local_k[None, :] < G),
                    0,
                ).to(tl.float32)
                packed = tl.load(
                    Q + rk[:, None] * QK + rp[None, :] * QN,
                    (local_k[:, None] < G) & (rp[None, :] < N // 8),
                    0,
                )
                q = _unpack_constant(packed)
                delta = tl.where(local_k[:, None] < G, q - zero[None, :], 0.0)
                b = delta * scale[None, :]
                acc = tl.dot(a, b, acc, input_precision="ieee")
        tl.store(
            P + part * M * N + rm[:, None] * N + rn[None, :],
            acc,
            (rm[:, None] < M) & (rn[None, :] < N),
        )


@triton.jit
def _awq_small(
    A,
    Q,
    S,
    Z,
    P,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    G: tl.constexpr,
    AM: tl.constexpr,
    AK: tl.constexpr,
    QK: tl.constexpr,
    QN: tl.constexpr,
    SG: tl.constexpr,
    SN: tl.constexpr,
    ZG: tl.constexpr,
    ZN: tl.constexpr,
    SPLIT: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    task = tl.program_id(0)
    tile = task // SPLIT
    part = task % SPLIT
    row = tile // tl.cdiv(N, BN)
    col = tile % tl.cdiv(N, BN) * BN + tl.arange(0, BN)
    packed_col = tile % tl.cdiv(N, BN) * (BN // 8) + tl.arange(0, BN // 8)
    lane = tl.arange(0, BK)
    begin = part * (K // G) // SPLIT
    end = (part + 1) * (K // G) // SPLIT
    acc = tl.zeros((BK, BN), tl.float32)
    for group in range(begin, end):
        z = _unpack_constant(
            tl.load(Z + group * ZG + packed_col * ZN, packed_col < N // 8, 0)
        )
        scale = tl.load(S + group * SG + col * SN, col < N, 0).to(tl.float32)
        for block in range(tl.cdiv(G, BK)):
            local = block * BK + lane
            k = group * G + local
            q = _unpack_constant(
                tl.load(
                    Q + k[:, None] * QK + packed_col[None, :] * QN,
                    (local[:, None] < G) & (packed_col[None, :] < N // 8),
                    0,
                    care_padding=False,
                )
            )
            a = tl.load(A + row * AM + k * AK, local < G, 0).to(tl.float32)
            w = tl.where(
                local[:, None] < G, (q - z[None, :]) * scale[None, :], 0.0
            )
            acc += a[:, None] * w
    value = tl.sum(acc, 0)
    tl.store(P + part * M * N + row * N + col, value, col < N)


@triton.jit
def _awq_blocked_work(
    A,
    Q,
    S,
    Z,
    P,
    work,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    G: tl.constexpr,
    AM: tl.constexpr,
    AK: tl.constexpr,
    QK: tl.constexpr,
    QN: tl.constexpr,
    SG: tl.constexpr,
    SN: tl.constexpr,
    ZG: tl.constexpr,
    ZN: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    SPLIT: tl.constexpr,
    CARE: tl.constexpr,
):
    tile = work // SPLIT
    part = work % SPLIT
    cols = tile % tl.cdiv(N, BN) * BN + tl.arange(0, BN)
    shift = (cols % 8 // 2 + cols % 2 * 4) * 4
    lane = tl.arange(0, BK)
    begin = part * (K // G) // SPLIT
    end = (part + 1) * (K // G) // SPLIT
    acc = tl.zeros((BN,), tl.float32)
    for group in range(begin, end):
        packed_zero = tl.load(Z + group * ZG + cols // 8 * ZN, cols < N, 0)
        zero = (packed_zero >> shift & 15).to(tl.float32)
        scale = tl.load(S + group * SG + cols * SN, cols < N, 0).to(tl.float32)
        group_acc = tl.zeros((BK, BN), tl.float32)
        for block in range(tl.cdiv(G, BK)):
            local = block * BK + lane
            kk = group * G + local
            packed = tl.load(
                Q + kk[:, None] * QK + cols[None, :] // 8 * QN,
                (local[:, None] < G) & (cols[None, :] < N),
                0,
                care_padding=CARE,
            )
            q = (packed >> shift[None, :] & 15).to(tl.float32)
            delta = tl.where(local[:, None] < G, q - zero[None, :], 0.0)
            row = tile // tl.cdiv(N, BN)
            a = tl.load(A + row * AM + kk * AK, local < G, 0).to(tl.float32)
            group_acc += a[:, None] * delta
        acc += tl.sum(group_acc, axis=0) * scale
    row = tile // tl.cdiv(N, BN)
    tl.store(P + part * M * N + row * N + cols, acc, cols < N)


@triton.jit
def _awq_blocked_fallback(
    A,
    Q,
    S,
    Z,
    P,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    G: tl.constexpr,
    AM: tl.constexpr,
    AK: tl.constexpr,
    QK: tl.constexpr,
    QN: tl.constexpr,
    SG: tl.constexpr,
    SN: tl.constexpr,
    ZG: tl.constexpr,
    ZN: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    SPLIT: tl.constexpr,
    PERSIST: tl.constexpr,
):
    if PERSIST:
        total: tl.constexpr = M * tl.cdiv(N, BN) * SPLIT
        for work in range(tl.program_id(0), total, tl.num_programs(0)):
            _awq_blocked_work(
                A,
                Q,
                S,
                Z,
                P,
                work,
                M,
                N,
                K,
                G,
                AM,
                AK,
                QK,
                QN,
                SG,
                SN,
                ZG,
                ZN,
                BN,
                BK,
                SPLIT,
                True,
            )
    else:
        _awq_blocked_work(
            A,
            Q,
            S,
            Z,
            P,
            tl.program_id(0),
            M,
            N,
            K,
            G,
            AM,
            AK,
            QK,
            QN,
            SG,
            SN,
            ZG,
            ZN,
            BN,
            BK,
            SPLIT,
            True,
        )


@triton.jit
def _awq_ts_matmul(
    A,
    W,
    OUT,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    AM: tl.constexpr,
    AK: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    tile = tl.program_id(0)
    rm = tile // tl.cdiv(N, BN) * BM + tl.arange(0, BM)
    rn = tile % tl.cdiv(N, BN) * BN + tl.arange(0, BN)
    kk = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), tl.float32)
    for block in range(tl.cdiv(K, BK)):
        rk = block * BK + kk
        a = tl.load(
            A + rm[:, None] * AM + rk[None, :] * AK,
            (rm[:, None] < M) & (rk[None, :] < K),
            0,
        ).to(tl.float32)
        w = tl.load(
            W + rk[:, None] * N + rn[None, :],
            (rk[:, None] < K) & (rn[None, :] < N),
            0,
        )
        acc = tl.dot(a, w, acc, input_precision="ieee")
    tl.store(
        OUT + rm[:, None] * N + rn[None, :],
        acc,
        (rm[:, None] < M) & (rn[None, :] < N),
    )


@triton.jit
def _awq_halfword_unpack(packed, logical):
    # Every halfword is at most 65535. FP32 preserves all source integers,
    # and multiplication by a power of two is exact. Since x is nonnegative,
    # FP32-to-INT32 truncation followed by &15 extracts the requested nibble.
    lo = (packed & 65535).to(tl.float32)
    hi = ((packed >> 16) & 65535).to(tl.float32)
    half = lo + (hi - lo) * (logical % 2).to(tl.float32)
    # FP32 exponent bits generate 1, 1/16, 1/256, 1/4096 for the four nibble pairs.
    coefficient = (1065353216 - (logical // 2) * 33554432).to(
        tl.float32, bitcast=True
    )
    x = half * coefficient
    return (x.to(tl.int32) & 15).to(tl.float32)


@triton.jit
def _awq_r4_pair_tile(
    Q,
    W,
    zero,
    scale,
    kstart,
    nstart,
    N: tl.constexpr,
    QK: tl.constexpr,
    QN: tl.constexpr,
    DK: tl.constexpr,
    DN: tl.constexpr,
):
    rk = kstart + tl.arange(0, DK)
    rn = nstart + tl.arange(0, DN)
    rp = nstart // 8 + tl.arange(0, DN // 8)
    logical = tl.arange(0, 8)
    packed = tl.load(Q + rk[:, None] * QK + rp[None, :] * QN)
    q = _awq_halfword_unpack(
        packed[:, :, None], logical[None, None, :]
    ).reshape((DK, DN))
    value = (q - zero[None, :]) * scale[None, :]
    tl.store(W + rk[:, None] * N + rn[None, :], value)


@triton.jit
def _awq_ts_dequant(
    Q,
    S,
    Z,
    W,
    N: tl.constexpr,
    K: tl.constexpr,
    G: tl.constexpr,
    QK: tl.constexpr,
    QN: tl.constexpr,
    SG: tl.constexpr,
    SN: tl.constexpr,
    ZG: tl.constexpr,
    ZN: tl.constexpr,
    DK: tl.constexpr,
    DN: tl.constexpr,
):
    # Strict route guarantees complete K groups and complete N tiles.
    # Reuse zero and scale while processing the two DK=64 halves of G=128.
    nt: tl.constexpr = N // DN
    total: tl.constexpr = K // G * nt
    for task in range(tl.program_id(0), total, tl.num_programs(0)):
        group = task // nt
        kstart = group * G
        nstart = task % nt * DN
        rn = nstart + tl.arange(0, DN)
        rp = nstart // 8 + tl.arange(0, DN // 8)
        logical = tl.arange(0, 8)
        scale = tl.load(S + group * SG + rn * SN)
        zp = tl.load(Z + group * ZG + rp * ZN)
        zero = _awq_halfword_unpack(zp[:, None], logical[None, :]).reshape(
            (DN,)
        )
        for inner in range(G // DK):
            _awq_r4_pair_tile(
                Q,
                W,
                zero,
                scale,
                kstart + inner * DK,
                nstart,
                N,
                QK,
                QN,
                DK,
                DN,
            )


@triton.jit
def _awq_m1_decode_fp32(packed, logical):
    # All halves, differences and power-of-two fractions are FP32 exact.
    lo = (packed & 65535).to(tl.float32)
    hi = ((packed >> 16) & 65535).to(tl.float32)
    half = lo + (hi - lo) * (logical % 2).to(tl.float32)
    # FP32 exponent bits generate 1, 1/16, 1/256, 1/4096 for the four nibble pairs.
    coefficient = (1065353216 - (logical // 2) * 33554432).to(
        tl.float32, bitcast=True
    )
    value = half * coefficient
    return tl.floor(value) - 16.0 * tl.floor(value * 0.0625)


@triton.jit
def _awq_m1_vector(
    A,
    Q,
    S,
    Z,
    OUT,
    N: tl.constexpr,
    K: tl.constexpr,
    G: tl.constexpr,
    AK: tl.constexpr,
    QK: tl.constexpr,
    QN: tl.constexpr,
    SG: tl.constexpr,
    SN: tl.constexpr,
    ZG: tl.constexpr,
    ZN: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    SPLIT: tl.constexpr,
):
    work = tl.program_id(0)
    part = work % SPLIT
    nt = work // SPLIT
    cols = nt * BN + tl.arange(0, BN)
    pc = nt * (BN // 8) + tl.arange(0, BN // 8)
    lane = tl.arange(0, BK)
    nibble = tl.arange(0, 8)
    begin = part * (K // G) // SPLIT
    end = (part + 1) * (K // G) // SPLIT
    acc = tl.zeros((BN,), tl.float32)
    for group in range(begin, end):
        scale = tl.load(S + group * SG + cols * SN).to(tl.float32)
        zp = tl.load(Z + group * ZG + pc * ZN)
        zero = tl.reshape(
            _awq_m1_decode_fp32(zp[:, None], nibble[None, :]), (BN,)
        )
        group_acc = tl.zeros((BK, BN), tl.float32)
        for block in range(G // BK):
            kk = group * G + block * BK + lane
            a = tl.load(A + kk * AK).to(tl.float32)
            packed = tl.load(Q + kk[:, None] * QK + pc[None, :] * QN)
            q = tl.reshape(
                _awq_m1_decode_fp32(packed[:, :, None], nibble[None, None, :]),
                (BK, BN),
            )
            group_acc += a[:, None] * (q - zero[None, :])
        acc += tl.sum(group_acc, axis=0) * scale
    tl.store(OUT + part * N + cols, acc)


def _awq_two_stage(input, qweight, scales, qzeros, split_k_iters):
    m, k = input.shape
    n = qweight.shape[1] * 8
    g = k // scales.shape[0]
    w = torch.empty((k, n), device=input.device, dtype=torch.float32)
    out = torch.empty((m, n), device=input.device, dtype=torch.float32)
    total = (k // 128) * (n // 64)
    props = driver.active.utils.get_device_properties(
        torch.npu.current_device()
    )
    vector_cores = (
        props.get("num_vectorcore", 0) if isinstance(props, dict) else 0
    )
    programs = (
        min(total, vector_cores)
        if isinstance(vector_cores, int) and vector_cores > 0
        else total
    )
    _awq_ts_dequant[(programs,)](
        qweight,
        scales,
        qzeros,
        w,
        n,
        k,
        g,
        *qweight.stride(),
        *scales.stride(),
        *qzeros.stride(),
        64,
        64,
        num_warps=4,
        num_stages=2,
        enable_ubuf_saving=False,
    )
    bm = 16 if m == 8 else (128 if m == 512 else 64)
    bn = (32 if m == 8 else 64) if m <= 64 and n == 1024 else 128
    _awq_ts_matmul[(triton.cdiv(m, bm) * triton.cdiv(n, bn),)](
        input,
        w,
        out,
        m,
        n,
        k,
        *input.stride(),
        bm,
        bn,
        128,
        num_warps=4,
        num_stages=1,
    )
    return out


@triton.jit
def _awq_short_contiguous(A, Q, S, Z, OUT, M: tl.constexpr):
    # Explicit contiguous shape guards establish the fixed dimensions below.
    N: tl.constexpr = 32 if M == 16 else (64 if M == 24 else 128)
    G: tl.constexpr = 32 if M == 24 else 128
    BK: tl.constexpr = 32 if M == 32 else 64
    _awq_small(
        A,
        Q,
        S,
        Z,
        OUT,
        M,
        N,
        128,
        G,
        128,
        1,
        N // 8,
        1,
        N,
        1,
        N // 8,
        1,
        1,
        N,
        BK,
    )


def awq_gemm(input, qweight, scales, qzeros, split_k_iters):
    m, k = input.shape
    n = qweight.shape[1] * 8
    groups = scales.shape[0]
    # Exact short-K metadata permits canonical contiguous strides, including
    # offset views. Other inputs continue through the general dispatch.
    if (
        k == 128
        and (
            (m == 16 and n == 32 and groups == 1 and split_k_iters == 1)
            or (m == 24 and n == 64 and groups == 4 and split_k_iters == 8)
            or (m == 32 and n == 128 and groups == 1 and split_k_iters == 1)
        )
        and qweight.shape == (128, n // 8)
        and scales.shape == (groups, n)
        and qzeros.shape == (groups, n // 8)
        and input.dtype == torch.float32
        and scales.dtype == torch.float32
        and qweight.dtype == torch.int32
        and qzeros.dtype == torch.int32
        and input.is_contiguous()
        and qweight.is_contiguous()
        and scales.is_contiguous()
        and qzeros.is_contiguous()
    ):
        out = torch.empty((m, n), device=input.device, dtype=torch.float32)
        _awq_short_contiguous.run(
            input,
            qweight,
            scales,
            qzeros,
            out,
            m,
            grid=(m,),
            warmup=False,
            num_warps=4,
            num_stages=1,
        )
        return out
    assert split_k_iters > 0 and qweight.shape[0] == k
    assert scales.shape[1] == n and qzeros.shape == (groups, n // 8)
    assert k == 0 or (groups > 0 and k % groups == 0)
    assert qweight.dtype == torch.int32 and qzeros.dtype == torch.int32
    assert scales.dtype == torch.float32
    # Fresh exact halfword dequantization, followed by a pure IEEE Cube GEMM.
    # Complete dequant tiles are enforced by this strict shape route.
    if (
        m in (8, 64, 512)
        and k == 4096
        and n in (1024, 4096)
        and groups == 32
        and split_k_iters == 8
        and input.dtype == torch.float32
        and input.is_contiguous()
        and qweight.is_contiguous()
        and scales.is_contiguous()
        and qzeros.is_contiguous()
    ):
        return _awq_two_stage(input, qweight, scales, qzeros, split_k_iters)
    out = torch.empty((m, n), device=input.device, dtype=scales.dtype)
    if m == 0 or n == 0:
        return out
    if k == 0:
        _awq_zero[((m * n + 255) // 256,)](
            out, m * n, 256, num_warps=4, num_stages=1
        )
        return out
    g = k // groups
    strides = (
        *input.stride(),
        *qweight.stride(),
        *scales.stride(),
        *qzeros.stride(),
    )
    contiguous = (
        input.is_contiguous()
        and qweight.is_contiguous()
        and scales.is_contiguous()
        and qzeros.is_contiguous()
    )
    if (
        k <= 128
        and g % 32 == 0
        and n < 128
        and n % 32 == 0
        and m * (n // 32) <= 65535
        and contiguous
    ):
        bk = 64
        _awq_small[(m * (n // 32),)](
            input,
            qweight,
            scales,
            qzeros,
            out,
            m,
            n,
            k,
            g,
            *strides,
            1,
            32,
            bk,
            num_warps=4,
            num_stages=1,
        )
        return out
    # Long-K M=1 uses AIV tiles with 32-byte packed-Q row segments.
    # Group-local FP32 reduction precedes FP32 scale; no half-input rounding.
    if (
        m == 1
        and k == 4096
        and g == 128
        and n in (1024, 4096)
        and split_k_iters == 8
        and contiguous
        and input.dtype == torch.float32
    ):
        split = 32 if n == 1024 else 8
        partial = torch.empty(
            (split, m, n), device=input.device, dtype=torch.float32
        )
        bk = 32 if n == 1024 else 64
        _awq_m1_vector[((n // 64) * split,)](
            input,
            qweight,
            scales,
            qzeros,
            partial,
            n,
            k,
            g,
            strides[1],
            strides[2],
            strides[3],
            strides[4],
            strides[5],
            strides[6],
            strides[7],
            64,
            bk,
            split,
            num_warps=4,
            num_stages=2,
            enable_ubuf_saving=False,
        )
        _awq_sum_flat[((m * n + 255) // 256,)](
            partial, out, m * n, split, 256, num_warps=4, num_stages=1
        )
        return out
    split = min(int(split_k_iters), groups)
    partial = (
        out
        if split == 1
        else torch.empty(
            (split, m, n), device=input.device, dtype=torch.float32
        )
    )
    if n % 128 == 0 and g % 32 == 0 and contiguous:
        device, aic, aiv = _device_cores()
        if k <= 128:
            bm, bn, bk = 32, 128, 32
        elif m >= 256:
            bm, bn, bk = 256, 64, 32
        elif n >= 4096:
            bm, bn, bk = (16 if m <= 16 else 64), 128, 32
        elif m >= 64 and g % 64 == 0:
            bm, bn, bk = 64, 64, 64
        else:
            bm, bn, bk = (
                (16 if m <= 16 else 32),
                64,
                (64 if g % 64 == 0 else 32),
            )
        total = ((m + bm - 1) // bm) * ((n + bn - 1) // bn) * split
        _awq_cube[(_program_count(total, aic),)](
            input,
            qweight,
            scales,
            qzeros,
            partial,
            m,
            n,
            k,
            g,
            *strides,
            split,
            0,
            device,
            aic,
            aiv,
            bm,
            bn,
            bk,
            num_warps=4,
            num_stages=1,
        )
    else:
        # Arbitrary strides/group tails use direct bit extraction and blocked K.
        # Persistent workers cap the grid when the logical work exceeds CANN's
        # launch bound; the worker count still comes from actual device cores.
        total = m * ((n + 31) // 32) * split
        persistent = total > 65535
        programs = total
        if persistent:
            device, aic, aiv = _device_cores()
            programs = min(_program_count(total, aiv), 65535)
        _awq_blocked_fallback[(programs,)](
            input,
            qweight,
            scales,
            qzeros,
            partial,
            m,
            n,
            k,
            g,
            *strides,
            32,
            32,
            split,
            persistent,
            num_warps=4,
            num_stages=1,
        )
    if split > 1:
        _awq_sum_flat[((m * n + 255) // 256,)](
            partial, out, m * n, split, 256, num_warps=4, num_stages=1
        )
    return out


def solution(input, qweight, scales, qzeros, split_k_iters):
    return awq_gemm(input, qweight, scales, qzeros, split_k_iters)


def reference(input, qweight, scales, qzeros, split_k_iters):
    return awq_gemm(input, qweight, scales, qzeros, split_k_iters)


__all__ = ["awq_gemm"]
