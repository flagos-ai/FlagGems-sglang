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

"""Iluvatar AWQ GEMM with explicit shape specializations and strided fallbacks.

Paths include fused vector/group products and two-stage FP32 dequantization
followed by GEMM. Packed INT4 decoding uses three-dimensional broadcasting
where selected. All arithmetic is Triton; output and scratch are freshly allocated.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _awq_vector_transposed(
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
    tile = tl.program_id(0)
    part = tl.program_id(1)
    row = tile // tl.cdiv(N, BN)
    rn = tile % tl.cdiv(N, BN) * BN + tl.arange(0, BN)
    rk = part * BK + tl.arange(0, BK)
    shift = ((rn % 8) // 2 + (rn % 2) * 4) * 4
    acc = tl.zeros((BN, BK), tl.float32)
    for step in range(tl.cdiv(K, BK * SPLIT)):
        a = tl.load(A + row * AM + rk * AK, rk < K, 0).to(tl.float32)
        q = tl.load(
            Q + rk[None, :] * QK + (rn[:, None] // 8) * QN,
            (rn[:, None] < N) & (rk[None, :] < K),
            0,
        )
        if G % BK == 0:
            group = (part * BK + step * BK * SPLIT) // G
            z = tl.load(
                Z + group * ZG + (rn // 8) * ZN, (rn < N) & (group < K // G), 0
            )
            scale = tl.load(
                S + group * SG + rn * SN, (rn < N) & (group < K // G), 0
            ).to(tl.float32)
            zero = ((z >> shift) & 15).to(tl.float32)
            w = (
                ((q >> shift[:, None]) & 15).to(tl.float32) - zero[:, None]
            ) * scale[:, None]
        else:
            group = rk // G
            z = tl.load(
                Z + group[None, :] * ZG + (rn[:, None] // 8) * ZN,
                (rn[:, None] < N) & (rk[None, :] < K),
                0,
            )
            scale = tl.load(
                S + group[None, :] * SG + rn[:, None] * SN,
                (rn[:, None] < N) & (rk[None, :] < K),
                0,
            ).to(tl.float32)
            w = (
                ((q >> shift[:, None]) & 15).to(tl.float32)
                - ((z >> shift[:, None]) & 15).to(tl.float32)
            ) * scale
        acc += a[None, :] * w
        rk += BK * SPLIT
    tl.store(P + part * M * N + row * N + rn, tl.sum(acc, 1), rn < N)


@triton.jit
def _awq_sum(
    P,
    OUT,
    M: tl.constexpr,
    N: tl.constexpr,
    SPLIT: tl.constexpr,
    RS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    x = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    s = tl.arange(0, RS)
    v = tl.load(
        P + s[:, None] * (M * N) + x[None, :],
        (s[:, None] < SPLIT) & (x[None, :] < M * N),
        0,
    )
    tl.store(OUT + x, tl.sum(v, 0), x < M * N)


@triton.jit
def _awq_zero(OUT, SIZE: tl.constexpr):
    x = tl.program_id(0) * 256 + tl.arange(0, 256)
    tl.store(OUT + x, 0.0, x < SIZE)


@triton.jit
def _awq_kernel(
    A,
    Q,
    S,
    Z,
    P,
    OUT,
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
    REQUESTED: tl.constexpr,
    DEVICE: tl.constexpr,
    ARCH: tl.constexpr,
    MODE: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    SPLIT: tl.constexpr,
    TRANSPOSE: tl.constexpr,
    VECTOR: tl.constexpr,
):
    tile = tl.program_id(0)
    part = tl.program_id(1)
    rm = tile // tl.cdiv(N, BN) * BM + tl.arange(0, BM)
    rn = tile % tl.cdiv(N, BN) * BN + tl.arange(0, BN)
    rp = tile % tl.cdiv(N, BN) * (BN // 8) + tl.arange(0, BN // 8)
    shift = ((rn % 8) // 2 + (rn % 2) * 4) * 4
    if VECTOR:
        acc = tl.zeros((BK, BN), tl.float32)
    elif TRANSPOSE:
        acc = tl.zeros((BN, BM), tl.float32)
    else:
        acc = tl.zeros((BM, BN), tl.float32)
    gb = part * (K // G) // SPLIT
    ge = (part + 1) * (K // G) // SPLIT
    for group in range(gb, ge):
        z = tl.load(Z + group * ZG + rp * ZN, rp < N // 8, 0)
        z = tl.interleave(z, z)
        z = tl.interleave(z, z)
        z = tl.interleave(z, z)
        zero = ((z >> shift) & 15).to(tl.float32)
        scale = tl.load(S + group * SG + rn * SN, rn < N, 0).to(tl.float32)
        if MODE == "fp16x2" and not VECTOR:
            if TRANSPOSE:
                group_acc = tl.zeros((BN, BM), tl.float32)
            else:
                group_acc = tl.zeros((BM, BN), tl.float32)
        for block in range(tl.cdiv(G, BK)):
            local_k = block * BK + tl.arange(0, BK)
            rk = group * G + local_k
            q = tl.load(
                Q + rk[:, None] * QK + rp[None, :] * QN,
                (local_k[:, None] < G) & (rp[None, :] < N // 8),
                0,
            )
            q = tl.interleave(q, q)
            q = tl.interleave(q, q)
            q = tl.interleave(q, q)
            delta = ((q >> shift[None, :]) & 15).to(tl.float32) - zero[None, :]
            if MODE == "fp16x2" and not VECTOR:
                b = tl.where(local_k[:, None] < G, delta, 0.0)
            else:
                b = tl.where(local_k[:, None] < G, delta * scale[None, :], 0.0)
            if VECTOR:
                a = tl.load(
                    A + (tile // tl.cdiv(N, BN)) * AM + rk * AK, local_k < G, 0
                ).to(tl.float32)
                acc += a[:, None] * b
            else:
                a = tl.load(
                    A + rm[:, None] * AM + rk[None, :] * AK,
                    (rm[:, None] < M) & (local_k[None, :] < G),
                    0,
                ).to(tl.float32)
                if MODE == "fp16x2":
                    ah = a.to(tl.float16)
                    al = (a - ah.to(tl.float32)).to(tl.float16)
                    bh = b.to(tl.float16)
                    if TRANSPOSE:
                        group_acc = tl.dot(
                            tl.trans(bh), tl.trans(al), group_acc
                        )
                        group_acc = tl.dot(
                            tl.trans(bh), tl.trans(ah), group_acc
                        )
                    else:
                        group_acc = tl.dot(al, bh, group_acc)
                        group_acc = tl.dot(ah, bh, group_acc)
                else:
                    if TRANSPOSE:
                        acc = tl.dot(
                            tl.trans(b), tl.trans(a), acc, input_precision=MODE
                        )
                    else:
                        acc = tl.dot(a, b, acc, input_precision=MODE)
        if MODE == "fp16x2" and not VECTOR:
            if TRANSPOSE:
                acc += group_acc * scale[:, None]
            else:
                acc += group_acc * scale[None, :]
    if VECTOR:
        value = tl.sum(acc, 0)
        off = (tile // tl.cdiv(N, BN)) * N + rn
        mask = rn < N
    else:
        if TRANSPOSE:
            value = tl.trans(acc)
        else:
            value = acc
        off = rm[:, None] * N + rn[None, :]
        mask = (rm[:, None] < M) & (rn[None, :] < N)
    if SPLIT == 1:
        tl.store(OUT + off, value, mask)
    else:
        tl.store(P + part * M * N + off, value, mask)


@triton.jit
def _awq_control_gemv(
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
    tile = tl.program_id(0)
    part = tl.program_id(1)
    row = tile // tl.cdiv(N, BN)
    rn = (tile % tl.cdiv(N, BN)) * BN + tl.arange(0, BN)
    rp = (tile % tl.cdiv(N, BN)) * (BN // 8) + tl.arange(0, BN // 8)
    rk = part * BK + tl.arange(0, BK)
    shift = ((rn % 8) // 2 + (rn % 2) * 4) * 4
    acc = tl.zeros((BK, BN), tl.float32)
    for _ in range(tl.cdiv(K, BK * SPLIT)):
        a = tl.load(A + row * AM + rk * AK, rk < K, 0).to(tl.float32)
        packed = tl.load(
            Q + rk[:, None] * QK + rp[None, :] * QN,
            (rk[:, None] < K) & (rp[None, :] < N // 8),
            0,
        )
        packed = tl.interleave(packed, packed)
        packed = tl.interleave(packed, packed)
        packed = tl.interleave(packed, packed)
        group = rk // G
        zero = tl.load(
            Z + group[:, None] * ZG + rp[None, :] * ZN,
            (rk[:, None] < K) & (rp[None, :] < N // 8),
            0,
        )
        zero = tl.interleave(zero, zero)
        zero = tl.interleave(zero, zero)
        zero = tl.interleave(zero, zero)
        scale = tl.load(
            S + group[:, None] * SG + rn[None, :] * SN,
            (rk[:, None] < K) & (rn[None, :] < N),
            0,
        ).to(tl.float32)
        weight = ((packed >> shift[None, :]) & 15).to(tl.float32)
        zero = ((zero >> shift[None, :]) & 15).to(tl.float32)
        b = (weight - zero) * scale
        acc += a[:, None] * b
        rk += BK * SPLIT
    result = tl.sum(acc, axis=0)
    tl.store(P + part * M * N + row * N + rn, result, rn < N)


@triton.jit
def _awq_group_dot(
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
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    SPLIT: tl.constexpr,
):
    tile = tl.program_id(0)
    part = tl.program_id(1)
    rm = tile // tl.cdiv(N, BN) * BM + tl.arange(0, BM)
    rn = tile % tl.cdiv(N, BN) * BN + tl.arange(0, BN)
    rp = tile % tl.cdiv(N, BN) * (BN // 8) + tl.arange(0, BN // 8)
    shift = ((rn % 8) // 2 + (rn % 2) * 4) * 4
    acc = tl.zeros((BM, BN), tl.float32)
    for group in range(
        part * (K // G) // SPLIT, (part + 1) * (K // G) // SPLIT
    ):
        z = tl.load(Z + group * ZG + rp * ZN, rp < N // 8, 0)
        z = tl.interleave(z, z)
        z = tl.interleave(z, z)
        z = tl.interleave(z, z)
        zero = ((z >> shift) & 15).to(tl.float32)
        scale = tl.load(S + group * SG + rn * SN, rn < N, 0).to(tl.float32)
        group_acc = tl.zeros((BM, BN), tl.float32)
        for block in range(tl.cdiv(G, BK)):
            local = block * BK + tl.arange(0, BK)
            rk = group * G + local
            q = tl.load(
                Q + rk[:, None] * QK + rp[None, :] * QN,
                (local[:, None] < G) & (rp[None, :] < N // 8),
                0,
            )
            q = tl.interleave(q, q)
            q = tl.interleave(q, q)
            q = tl.interleave(q, q)
            delta = ((q >> shift[None, :]) & 15).to(tl.float32) - zero[None, :]
            b = tl.where(local[:, None] < G, delta, 0.0).to(tl.float16)
            a = tl.load(
                A + rm[:, None] * AM + rk[None, :] * AK,
                (rm[:, None] < M) & (local[None, :] < G),
                0,
            ).to(tl.float32)
            ah = a.to(tl.float16)
            al = (a - ah.to(tl.float32)).to(tl.float16)
            group_acc = tl.dot(al, b, group_acc)
            group_acc = tl.dot(ah, b, group_acc)
        acc += group_acc * scale[None, :]
    tl.store(
        P + part * M * N + rm[:, None] * N + rn[None, :],
        acc,
        (rm[:, None] < M) & (rn[None, :] < N),
    )


def _select(m, n, k, g):
    if m == 16 and n == 32 and k == 128 and g == 128:
        return ("gemv_t", "ieee", 1, 16, 128, 1, 2, 1, 0)
    if m == 24 and n == 64 and k == 128 and g == 32:
        return ("gemv_t", "ieee", 1, 8, 128, 1, 4, 1, 0)
    if m == 32 and n == 128 and k == 128 and g == 128:
        return ("gemv_t", "ieee", 1, 16, 128, 1, 4, 1, 0)
    if m == 1 and n == 1024 and k == 4096 and g == 128:
        return ("gemv_t", "ieee", 1, 32, 128, 4, 2, 1, 0)
    if m == 1 and n == 4096 and k == 4096 and g == 128:
        return ("gemv_t", "ieee", 1, 64, 128, 4, 4, 1, 0)
    if m == 8 and n == 1024 and k == 4096 and g == 128:
        return ("matrix", "fp16x2", 16, 128, 128, 16, 4, 1, 0)
    if m == 8 and n == 4096 and k == 4096 and g == 128:
        return ("matrix", "fp16x2", 16, 128, 128, 16, 4, 1, 0)
    if m == 64 and n == 1024 and k == 4096 and g == 128:
        return ("groupdot", "fp16x2", 64, 64, 64, 4, 4, 1, 0)
    if m == 64 and n == 4096 and k == 4096 and g == 128:
        return ("groupdot", "fp16x2", 64, 128, 64, 4, 4, 1, 0)
    if m == 512 and n == 1024 and k == 4096 and g == 128:
        return ("groupdot", "fp16x2", 128, 128, 64, 1, 8, 1, 0)
    if m == 512 and n == 4096 and k == 4096 and g == 128:
        return ("groupdot", "fp16x2", 128, 128, 64, 1, 8, 1, 0)
    return None


def _awq_original(input, qweight, scales, qzeros, split_k_iters):
    m, k = input.shape
    n = qweight.shape[1] * 8
    groups = scales.shape[0]
    assert split_k_iters > 0 and qweight.shape[0] == k and scales.shape[1] == n
    assert qzeros.shape == (groups, n // 8)
    assert k == 0 or (groups > 0 and k % groups == 0)
    out = torch.empty((m, n), device=input.device, dtype=scales.dtype)
    if m == 0 or n == 0:
        return out
    if k == 0:
        _awq_zero[((m * n + 255) // 256,)](
            out, m * n, num_warps=4, num_stages=1
        )
        return out
    g = k // groups
    config = (
        _select(m, n, k, g)
        if input.dtype == torch.float32 and scales.dtype == torch.float32
        else None
    )
    if config is None:
        split = min(split_k_iters, (k + 63) // 64)
        config = ("gemv", "ieee", 1, 32, 64, split, 4, 1, 0)
    family, mode, bm, bn, bk, split, warps, stages, sme = config
    partial = (
        out
        if split == 1
        else torch.empty(
            (split, m, n), device=input.device, dtype=torch.float32
        )
    )
    meta = (
        m,
        n,
        k,
        g,
        *input.stride(),
        *qweight.stride(),
        *scales.stride(),
        *qzeros.stride(),
    )
    if family == "gemv":
        _awq_control_gemv[(m * ((n + bn - 1) // bn), split)](
            input,
            qweight,
            scales,
            qzeros,
            partial,
            *meta,
            split,
            bn,
            bk,
            num_warps=warps,
            num_stages=stages,
            use_sme=sme,
        )
    elif family == "gemv_t":
        _awq_vector_transposed[(m * ((n + bn - 1) // bn), split)](
            input,
            qweight,
            scales,
            qzeros,
            partial,
            *meta,
            split,
            bn,
            bk,
            num_warps=warps,
            num_stages=stages,
            use_sme=sme,
        )
    elif family == "groupdot":
        _awq_group_dot[(((m + bm - 1) // bm) * ((n + bn - 1) // bn), split)](
            input,
            qweight,
            scales,
            qzeros,
            partial,
            *meta,
            bm,
            bn,
            bk,
            split,
            num_warps=warps,
            num_stages=stages,
            use_sme=sme,
        )
    elif family == "matrix":
        _awq_kernel[(((m + bm - 1) // bm) * ((n + bn - 1) // bn), split)](
            input,
            qweight,
            scales,
            qzeros,
            partial,
            out,
            *meta,
            split_k_iters,
            0,
            71,
            mode,
            bm,
            bn,
            bk,
            split,
            False,
            False,
            num_warps=warps,
            num_stages=stages,
            use_sme=sme,
        )
    if split > 1:
        _awq_sum[((m * n + 255) // 256,)](
            partial,
            out,
            m,
            n,
            split,
            triton.next_power_of_2(split),
            256,
            num_warps=4,
            num_stages=1,
        )
    return out


@triton.jit
def _r5_unpack(packed):
    logical = tl.arange(0, 8)
    shift = (logical // 2 + (logical % 2) * 4) * 4
    if len(packed.shape) == 1:
        value = ((packed[:, None] >> shift[None, :]) & 15).to(tl.float32)
        return value.reshape((packed.shape[0] * 8,))
    else:
        value = ((packed[:, :, None] >> shift[None, None, :]) & 15).to(
            tl.float32
        )
        return value.reshape((packed.shape[0], packed.shape[1] * 8))


@triton.jit
def _r5_group_dot(
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
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    SPLIT: tl.constexpr,
):
    tile = tl.program_id(0)
    part = tl.program_id(1)
    rm = tile // tl.cdiv(N, BN) * BM + tl.arange(0, BM)
    rn = tile % tl.cdiv(N, BN) * BN + tl.arange(0, BN)
    rp = tile % tl.cdiv(N, BN) * (BN // 8) + tl.arange(0, BN // 8)
    _ = ((rn % 8) // 2 + (rn % 2) * 4) * 4
    acc = tl.zeros((BM, BN), tl.float32)
    for group in range(
        part * (K // G) // SPLIT, (part + 1) * (K // G) // SPLIT
    ):
        z = tl.load(Z + group * ZG + rp * ZN, rp < N // 8, 0)
        zero = _r5_unpack(z)
        scale = tl.load(S + group * SG + rn * SN, rn < N, 0).to(tl.float32)
        group_acc = tl.zeros((BM, BN), tl.float32)
        for block in range(tl.cdiv(G, BK)):
            local = block * BK + tl.arange(0, BK)
            rk = group * G + local
            q = tl.load(
                Q + rk[:, None] * QK + rp[None, :] * QN,
                (local[:, None] < G) & (rp[None, :] < N // 8),
                0,
            )
            delta = _r5_unpack(q) - zero[None, :]
            b = tl.where(local[:, None] < G, delta, 0.0).to(tl.float16)
            a = tl.load(
                A + rm[:, None] * AM + rk[None, :] * AK,
                (rm[:, None] < M) & (local[None, :] < G),
                0,
            ).to(tl.float32)
            ah = a.to(tl.float16)
            al = (a - ah.to(tl.float32)).to(tl.float16)
            group_acc = tl.dot(al, b, group_acc)
            group_acc = tl.dot(ah, b, group_acc)
        acc += group_acc * scale[None, :]
    tl.store(
        P + part * M * N + rm[:, None] * N + rn[None, :],
        acc,
        (rm[:, None] < M) & (rn[None, :] < N),
    )


@triton.jit
def _r5_dequant(
    Q,
    S,
    Z,
    W,
    N: tl.constexpr,
    K: tl.constexpr,
    G: tl.constexpr,
    DK: tl.constexpr,
    DN: tl.constexpr,
):
    task = tl.program_id(0)
    nt = task % (N // DN)
    kt = task // (N // DN)
    kk = kt * DK + tl.arange(0, DK)
    rn = nt * DN + tl.arange(0, DN)
    rp = nt * (DN // 8) + tl.arange(0, DN // 8)
    zp = tl.load(Z + (kt * DK // G) * (N // 8) + rp)
    zero = _r5_unpack(zp)
    scale = tl.load(S + (kt * DK // G) * N + rn)
    q = _r5_unpack(tl.load(Q + kk[:, None] * (N // 8) + rp[None, :]))
    w = (q - zero[None, :]) * scale[None, :]
    tl.store(W + kk[:, None] * N + rn[None, :], w)


@triton.jit
def _r5_dense(
    A,
    W,
    OUT,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    t = tl.program_id(0)
    rm = t // tl.cdiv(N, BN) * BM + tl.arange(0, BM)
    rn = t % tl.cdiv(N, BN) * BN + tl.arange(0, BN)
    lane = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), tl.float32)
    for block in range(K // BK):
        kk = block * BK + lane
        a = tl.load(A + rm[:, None] * K + kk[None, :], rm[:, None] < M, 0).to(
            tl.float32
        )
        w = tl.load(W + kk[:, None] * N + rn[None, :], rn[None, :] < N, 0)
        acc = tl.dot(a, w, acc, input_precision="ieee")
    tl.store(
        OUT + rm[:, None] * N + rn[None, :],
        acc,
        (rm[:, None] < M) & (rn[None, :] < N),
    )


@triton.jit
def _r5_dequant_11(Q, S, Z, W):
    _r5_dequant(Q, S, Z, W, 4096, 4096, 128, 32, 128)


@triton.jit
def _r5_dense_11(A, W, OUT):
    _r5_dense(A, W, OUT, 512, 4096, 4096, 128, 128, 64)


@triton.jit
def _r5_case_0(A, Q, S, Z, P, OUT):
    _awq_vector_transposed(
        A, Q, S, Z, P, 16, 32, 128, 128, 128, 1, 4, 1, 32, 1, 4, 1, 1, 16, 128
    )


@triton.jit
def _r5_case_1(A, Q, S, Z, P, OUT):
    _awq_vector_transposed(
        A, Q, S, Z, P, 24, 64, 128, 32, 128, 1, 8, 1, 64, 1, 8, 1, 1, 8, 128
    )


@triton.jit
def _r5_case_2(A, Q, S, Z, P, OUT):
    _awq_vector_transposed(
        A,
        Q,
        S,
        Z,
        P,
        32,
        128,
        128,
        128,
        128,
        1,
        16,
        1,
        128,
        1,
        16,
        1,
        1,
        16,
        128,
    )


@triton.jit
def _r5_case_3(A, Q, S, Z, P, OUT):
    _awq_vector_transposed(
        A,
        Q,
        S,
        Z,
        P,
        1,
        1024,
        4096,
        128,
        4096,
        1,
        128,
        1,
        1024,
        1,
        128,
        1,
        4,
        32,
        128,
    )


@triton.jit
def _r5_sum_3(P, OUT):
    _awq_sum(P, OUT, 1, 1024, 4, 4, 256)


@triton.jit
def _r5_case_4(A, Q, S, Z, P, OUT):
    _awq_vector_transposed(
        A,
        Q,
        S,
        Z,
        P,
        1,
        4096,
        4096,
        128,
        4096,
        1,
        512,
        1,
        4096,
        1,
        512,
        1,
        4,
        64,
        128,
    )


@triton.jit
def _r5_sum_4(P, OUT):
    _awq_sum(P, OUT, 1, 4096, 4, 4, 256)


@triton.jit
def _r5_case_5(A, Q, S, Z, P, OUT):
    _awq_kernel(
        A,
        Q,
        S,
        Z,
        P,
        OUT,
        8,
        1024,
        4096,
        128,
        4096,
        1,
        128,
        1,
        1024,
        1,
        128,
        1,
        8,
        0,
        71,
        "fp16x2",
        16,
        128,
        128,
        16,
        False,
        False,
    )


@triton.jit
def _r5_sum_5(P, OUT):
    _awq_sum(P, OUT, 8, 1024, 16, 16, 256)


@triton.jit
def _r5_case_6(A, Q, S, Z, P, OUT):
    _awq_kernel(
        A,
        Q,
        S,
        Z,
        P,
        OUT,
        8,
        4096,
        4096,
        128,
        4096,
        1,
        512,
        1,
        4096,
        1,
        512,
        1,
        8,
        0,
        71,
        "fp16x2",
        16,
        128,
        128,
        16,
        False,
        False,
    )


@triton.jit
def _r5_sum_6(P, OUT):
    _awq_sum(P, OUT, 8, 4096, 16, 16, 256)


@triton.jit
def _r5_case_7(A, Q, S, Z, P, OUT):
    _awq_group_dot(
        A,
        Q,
        S,
        Z,
        P,
        64,
        1024,
        4096,
        128,
        4096,
        1,
        128,
        1,
        1024,
        1,
        128,
        1,
        64,
        64,
        64,
        4,
    )


@triton.jit
def _r5_sum_7(P, OUT):
    _awq_sum(P, OUT, 64, 1024, 4, 4, 256)


@triton.jit
def _r5_case_8(A, Q, S, Z, P, OUT):
    _awq_group_dot(
        A,
        Q,
        S,
        Z,
        P,
        64,
        4096,
        4096,
        128,
        4096,
        1,
        512,
        1,
        4096,
        1,
        512,
        1,
        64,
        128,
        64,
        4,
    )


@triton.jit
def _r5_sum_8(P, OUT):
    _awq_sum(P, OUT, 64, 4096, 4, 4, 256)


@triton.jit
def _r5_case_9(A, Q, S, Z, P, OUT):
    _r5_group_dot(
        A,
        Q,
        S,
        Z,
        P,
        512,
        1024,
        4096,
        128,
        4096,
        1,
        128,
        1,
        1024,
        1,
        128,
        1,
        128,
        128,
        64,
        1,
    )


def awq_gemm(input, qweight, scales, qzeros, split_k_iters):
    m, k = input.shape
    n = qweight.shape[1] * 8
    groups = scales.shape[0]
    if (
        input.dtype != torch.float32
        or scales.dtype != torch.float32
        or qweight.dtype != torch.int32
        or qzeros.dtype != torch.int32
        or qweight.shape != (k, n // 8)
        or scales.shape != (groups, n)
        or qzeros.shape != (groups, n // 8)
        or not input.is_contiguous()
        or not qweight.is_contiguous()
        or not scales.is_contiguous()
        or not qzeros.is_contiguous()
    ):
        return _awq_original(input, qweight, scales, qzeros, split_k_iters)
    if m == 16 and n == 32 and k == 128 and groups == 1 and split_k_iters == 1:
        out = torch.empty((16, 32), device=input.device, dtype=torch.float32)
        partial = out
        _r5_case_0.run(
            input,
            qweight,
            scales,
            qzeros,
            partial,
            out,
            grid=(32, 1),
            warmup=False,
            num_warps=2,
            num_stages=1,
            use_sme=0,
        )
        return out
    if m == 24 and n == 64 and k == 128 and groups == 4 and split_k_iters == 8:
        out = torch.empty((24, 64), device=input.device, dtype=torch.float32)
        partial = out
        _r5_case_1.run(
            input,
            qweight,
            scales,
            qzeros,
            partial,
            out,
            grid=(192, 1),
            warmup=False,
            num_warps=4,
            num_stages=1,
            use_sme=0,
        )
        return out
    if (
        m == 32
        and n == 128
        and k == 128
        and groups == 1
        and split_k_iters == 1
    ):
        out = torch.empty((32, 128), device=input.device, dtype=torch.float32)
        partial = out
        _r5_case_2.run(
            input,
            qweight,
            scales,
            qzeros,
            partial,
            out,
            grid=(256, 1),
            warmup=False,
            num_warps=4,
            num_stages=1,
            use_sme=0,
        )
        return out
    if (
        m == 1
        and n == 1024
        and k == 4096
        and groups == 32
        and split_k_iters == 8
    ):
        out = torch.empty((1, 1024), device=input.device, dtype=torch.float32)
        partial = torch.empty(
            (4, 1, 1024), device=input.device, dtype=torch.float32
        )
        _r5_case_3.run(
            input,
            qweight,
            scales,
            qzeros,
            partial,
            out,
            grid=(32, 4),
            warmup=False,
            num_warps=2,
            num_stages=1,
            use_sme=0,
        )
        _r5_sum_3.run(
            partial, out, grid=(4,), warmup=False, num_warps=4, num_stages=1
        )
        return out
    if (
        m == 1
        and n == 4096
        and k == 4096
        and groups == 32
        and split_k_iters == 8
    ):
        out = torch.empty((1, 4096), device=input.device, dtype=torch.float32)
        partial = torch.empty(
            (4, 1, 4096), device=input.device, dtype=torch.float32
        )
        _r5_case_4.run(
            input,
            qweight,
            scales,
            qzeros,
            partial,
            out,
            grid=(64, 4),
            warmup=False,
            num_warps=4,
            num_stages=1,
            use_sme=0,
        )
        _r5_sum_4.run(
            partial, out, grid=(16,), warmup=False, num_warps=4, num_stages=1
        )
        return out
    if (
        m == 8
        and n == 1024
        and k == 4096
        and groups == 32
        and split_k_iters == 8
    ):
        out = torch.empty((8, 1024), device=input.device, dtype=torch.float32)
        partial = torch.empty(
            (16, 8, 1024), device=input.device, dtype=torch.float32
        )
        _r5_case_5.run(
            input,
            qweight,
            scales,
            qzeros,
            partial,
            out,
            grid=(8, 16),
            warmup=False,
            num_warps=4,
            num_stages=1,
            use_sme=0,
        )
        _r5_sum_5.run(
            partial, out, grid=(32,), warmup=False, num_warps=4, num_stages=1
        )
        return out
    if (
        m == 8
        and n == 4096
        and k == 4096
        and groups == 32
        and split_k_iters == 8
    ):
        out = torch.empty((8, 4096), device=input.device, dtype=torch.float32)
        partial = torch.empty(
            (16, 8, 4096), device=input.device, dtype=torch.float32
        )
        _r5_case_6.run(
            input,
            qweight,
            scales,
            qzeros,
            partial,
            out,
            grid=(32, 16),
            warmup=False,
            num_warps=4,
            num_stages=1,
            use_sme=0,
        )
        _r5_sum_6.run(
            partial, out, grid=(128,), warmup=False, num_warps=4, num_stages=1
        )
        return out
    if (
        m == 64
        and n == 1024
        and k == 4096
        and groups == 32
        and split_k_iters == 8
    ):
        out = torch.empty((64, 1024), device=input.device, dtype=torch.float32)
        partial = torch.empty(
            (4, 64, 1024), device=input.device, dtype=torch.float32
        )
        _r5_case_7.run(
            input,
            qweight,
            scales,
            qzeros,
            partial,
            out,
            grid=(16, 4),
            warmup=False,
            num_warps=4,
            num_stages=1,
            use_sme=0,
        )
        _r5_sum_7.run(
            partial, out, grid=(256,), warmup=False, num_warps=4, num_stages=1
        )
        return out
    if (
        m == 64
        and n == 4096
        and k == 4096
        and groups == 32
        and split_k_iters == 8
    ):
        out = torch.empty((64, 4096), device=input.device, dtype=torch.float32)
        partial = torch.empty(
            (4, 64, 4096), device=input.device, dtype=torch.float32
        )
        _r5_case_8.run(
            input,
            qweight,
            scales,
            qzeros,
            partial,
            out,
            grid=(32, 4),
            warmup=False,
            num_warps=4,
            num_stages=1,
            use_sme=0,
        )
        _r5_sum_8.run(
            partial, out, grid=(1024,), warmup=False, num_warps=4, num_stages=1
        )
        return out
    if (
        m == 512
        and n == 1024
        and k == 4096
        and groups == 32
        and split_k_iters == 8
    ):
        out = torch.empty(
            (512, 1024), device=input.device, dtype=torch.float32
        )
        partial = out
        _r5_case_9.run(
            input,
            qweight,
            scales,
            qzeros,
            partial,
            out,
            grid=(32, 1),
            warmup=False,
            num_warps=8,
            num_stages=1,
            use_sme=0,
        )
        return out
    if (
        m == 512
        and n == 4096
        and k == 4096
        and groups == 32
        and split_k_iters == 8
    ):
        out = torch.empty(
            (512, 4096), device=input.device, dtype=torch.float32
        )
        weight = torch.empty(
            (4096, 4096), device=input.device, dtype=torch.float32
        )
        _r5_dequant_11.run(
            qweight,
            scales,
            qzeros,
            weight,
            grid=(4096,),
            warmup=False,
            num_warps=4,
            num_stages=1,
            use_sme=0,
        )
        _r5_dense_11.run(
            input,
            weight,
            out,
            grid=(128,),
            warmup=False,
            num_warps=8,
            num_stages=1,
            use_sme=0,
        )
        return out
    return _awq_original(input, qweight, scales, qzeros, split_k_iters)


__all__ = ["awq_gemm"]
