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

"""MetaX AWQ INT4 GEMM with explicit shape specializations and FP32 reduction.

Logical columns map to packed nibbles [0,4,1,5,2,6,3,7]. Zero points are used
as stored, without a +1 offset. Output has scales.dtype. Fused kernels use
split partials or write output directly, according to the selected shape route.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _awq_matmul(
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
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    PRECISION: tl.constexpr,
):
    tile = tl.program_id(0)
    part = tl.program_id(1)
    rm = (tile // tl.cdiv(N, BN)) * BM + tl.arange(0, BM)
    rn = (tile % tl.cdiv(N, BN)) * BN + tl.arange(0, BN)
    rp = (tile % tl.cdiv(N, BN)) * (BN // 8) + tl.arange(0, BN // 8)
    rk = part * BK + tl.arange(0, BK)
    shift = ((rn % 8) // 2 + (rn % 2) * 4) * 4
    acc = tl.zeros((BM, BN), tl.float32)
    for _ in range(tl.cdiv(K, BK * SPLIT)):
        a = tl.load(
            A + rm[:, None] * AM + rk[None, :] * AK,
            (rm[:, None] < M) & (rk[None, :] < K),
            0,
        ).to(tl.float32)
        packed = tl.load(
            Q + rk[:, None] * QK + rp[None, :] * QN,
            (rk[:, None] < K) & (rp[None, :] < N // 8),
            0,
        )
        packed = tl.interleave(packed, packed)
        packed = tl.interleave(packed, packed)
        packed = tl.interleave(packed, packed)
        if G % BK == 0:
            group = (part * BK + _ * BK * SPLIT) // G
            zero = tl.load(
                Z + group * ZG + rp * ZN, (group < K // G) & (rp < N // 8), 0
            )
            zero = tl.interleave(zero, zero)
            zero = tl.interleave(zero, zero)
            zero = tl.interleave(zero, zero)
            scale = tl.load(
                S + group * SG + rn * SN, (group < K // G) & (rn < N), 0
            ).to(tl.float32)
            weight = ((packed >> shift[None, :]) & 15).to(tl.float32)
            zero = ((zero >> shift) & 15).to(tl.float32)
            b = (weight - zero[None, :]) * scale[None, :]
        else:
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
        acc = tl.dot(a, b, acc, input_precision=PRECISION)
        rk += BK * SPLIT
    tl.store(
        P + part * M * N + rm[:, None] * N + rn[None, :],
        acc,
        (rm[:, None] < M) & (rn[None, :] < N),
    )


@triton.jit
def _awq_sum(
    P,
    OUT,
    SIZE: tl.constexpr,
    SPLIT: tl.constexpr,
    RS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    part = tl.arange(0, RS)
    value = tl.load(
        P + part[:, None] * SIZE + index[None, :],
        (part[:, None] < SPLIT) & (index[None, :] < SIZE),
        0,
    )
    result = tl.sum(value, axis=0)
    tl.store(OUT + index, result, index < SIZE)


@triton.jit
def _awq_zero(OUT, SIZE: tl.constexpr, BLOCK: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(OUT + index, 0.0, index < SIZE)


def _baseline_awq(input, qweight, scales, qzeros, split_k_iters):
    """Return input @ dequantize_awq(qweight, scales, qzeros).

    Supports strided inputs, all positive split counts and group boundaries
    inside a K tile.  Each split writes its own partial result; a Triton
    reduction combines partials without atomics or a PyTorch math fallback.
    """
    m, k = input.shape
    n = qweight.shape[1] * 8
    groups = scales.shape[0]
    assert split_k_iters > 0
    assert qweight.shape[0] == k
    assert scales.shape[1] == n
    assert qzeros.shape == (groups, n // 8)
    assert k == 0 or (groups > 0 and k % groups == 0)
    output = torch.empty((m, n), device=input.device, dtype=scales.dtype)
    if m == 0 or n == 0:
        return output
    if k == 0:
        _awq_zero[(((m * n + 255) // 256),)](output, m * n, 256)
        return output
    group_size = k // groups
    # Native FP32 dot semantics; NVIDIA-only tf32x3 is not assumed.
    bm, bn, bk = (16 if m <= 16 else 32), 64, 32
    split = split_k_iters
    partial = (
        output
        if split == 1
        else torch.empty(
            (split, m, n), device=input.device, dtype=torch.float32
        )
    )
    _awq_matmul[(((m + bm - 1) // bm) * ((n + bn - 1) // bn), split)](
        input,
        qweight,
        scales,
        qzeros,
        partial,
        m,
        n,
        k,
        group_size,
        *input.stride(),
        *qweight.stride(),
        *scales.stride(),
        *qzeros.stride(),
        split,
        bm,
        bn,
        bk,
        "ieee",
        num_warps=4,
        num_stages=2,
    )
    if split != 1:
        _awq_sum[(((m * n + 255) // 256),)](
            partial,
            output,
            m * n,
            split,
            (1 << (split - 1).bit_length()),
            256,
            num_warps=4,
        )
    return output


def solution(input, qweight, scales, qzeros, split_k_iters):
    """Backward-compatible alias for the operator entry point."""
    return awq_gemm(input, qweight, scales, qzeros, split_k_iters)


def reference(input, qweight, scales, qzeros, split_k_iters):
    """Compatibility entry point with the exact requested argument names."""
    return awq_gemm(input, qweight, scales, qzeros, split_k_iters)


@triton.jit
def _group_sum(
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
def _awq_grouped(
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
def _tiny_simt(
    A,
    Q,
    S,
    Z,
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
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    task = tl.program_id(0)
    row = task // tl.cdiv(N, BN)
    n = task % tl.cdiv(N, BN) * BN + tl.arange(0, BN)
    k = tl.arange(0, BK)
    group = k // G
    shift = ((n % 8) // 2 + (n % 2) * 4) * 4
    mask = (k[:, None] < K) & (n[None, :] < N)
    q = tl.load(Q + k[:, None] * QK + (n[None, :] // 8) * QN, mask, 0)
    z = tl.load(Z + group[:, None] * ZG + (n[None, :] // 8) * ZN, mask, 0)
    s = tl.load(S + group[:, None] * SG + n[None, :] * SN, mask, 0).to(
        tl.float32
    )
    w = (
        ((q >> shift[None, :]) & 15).to(tl.float32)
        - ((z >> shift[None, :]) & 15).to(tl.float32)
    ) * s
    a = tl.load(A + row * AM + k * AK, k < K, 0).to(tl.float32)
    value = tl.sum(a[:, None] * w, axis=0)
    tl.store(OUT + row * N + n, value, n < N)


def _config(m, n, k, g, requested):
    if (m, n, k, g, requested) == (1, 1024, 4096, 128, 8):
        return (1, 32, 128, 8, False, True, "ieee", 4, 1)
    if (m, n, k, g, requested) == (1, 4096, 4096, 128, 8):
        return (1, 32, 64, 16, False, True, "ieee", 4, 1)
    if (m, n, k, g, requested) == (8, 1024, 4096, 128, 8):
        return (16, 64, 32, 8, True, False, "fp16x2", 4, 1)
    if (m, n, k, g, requested) == (8, 4096, 4096, 128, 8):
        return (16, 64, 32, 8, True, False, "ieee", 4, 1)
    if (m, n, k, g, requested) == (64, 4096, 4096, 128, 8):
        return (64, 64, 64, 8, False, False, "fp16x2", 4, 1)
    if (m, n, k, g, requested) == (512, 1024, 4096, 128, 8):
        return (64, 64, 64, 8, False, False, "fp16x2", 4, 1)
    if (m, n, k, g, requested) == (512, 4096, 4096, 128, 8):
        return (64, 64, 64, 4, False, False, "fp16x2", 4, 1)
    return None


def _r4_dispatch(input, qweight, scales, qzeros, split_k_iters):
    m, k = input.shape
    n = qweight.shape[1] * 8
    groups = scales.shape[0]
    g = k // groups if groups else 0
    compatible = (
        input.dtype == torch.float32
        and scales.dtype == torch.float32
        and input.is_contiguous()
        and qweight.is_contiguous()
        and scales.is_contiguous()
        and qzeros.is_contiguous()
    )
    if compatible and 0 < m <= 32 and 0 < n <= 128 and 0 < k <= 128 and g > 0:
        assert split_k_iters > 0 and k % groups == 0
        assert (
            qweight.shape[0] == k
            and scales.shape[1] == n
            and qzeros.shape == (groups, n // 8)
        )
        output = torch.empty((m, n), device=input.device, dtype=scales.dtype)
        _tiny_simt[(m * ((n + 7) // 8),)](
            input,
            qweight,
            scales,
            qzeros,
            output,
            m,
            n,
            k,
            g,
            *input.stride(),
            *qweight.stride(),
            *scales.stride(),
            *qzeros.stride(),
            8,
            1 << (k - 1).bit_length(),
            num_warps=2,
            num_stages=1,
        )
        return output
    cfg = _config(m, n, k, g, split_k_iters)
    if cfg is None or not compatible:
        return _baseline_awq(input, qweight, scales, qzeros, split_k_iters)
    assert (
        split_k_iters > 0
        and qweight.shape[0] == k
        and scales.shape[1] == n
        and qzeros.shape == (groups, n // 8)
    )
    bm, bn, bk, split, transpose, vector, mode, warps, stages = cfg
    output = torch.empty((m, n), device=input.device, dtype=scales.dtype)
    partial = (
        output
        if split == 1
        else torch.empty(
            (split, m, n), device=input.device, dtype=torch.float32
        )
    )
    _awq_grouped[(((m + bm - 1) // bm) * ((n + bn - 1) // bn), split)](
        input,
        qweight,
        scales,
        qzeros,
        partial,
        output,
        m,
        n,
        k,
        g,
        *input.stride(),
        *qweight.stride(),
        *scales.stride(),
        *qzeros.stride(),
        mode,
        bm,
        bn,
        bk,
        split,
        transpose,
        vector,
        num_warps=warps,
        num_stages=stages,
    )
    if split > 1:
        _group_sum[((m * n + 255) // 256,)](
            partial,
            output,
            m,
            n,
            split,
            1 << (split - 1).bit_length(),
            256,
            num_warps=4,
        )
    return output


# Immediate shift extraction keeps packed INT32 lanes in registers.
# Dequantized weights are not rounded to FP16 in this path.
@triton.jit
def _r5_unpack_constant(p):
    x0 = p & 15
    x1 = (p >> 16) & 15
    x2 = (p >> 4) & 15
    x3 = (p >> 20) & 15
    x4 = (p >> 8) & 15
    x5 = (p >> 24) & 15
    x6 = (p >> 12) & 15
    x7 = (p >> 28) & 15
    q = tl.interleave(
        tl.interleave(tl.interleave(x0, x4), tl.interleave(x2, x6)),
        tl.interleave(tl.interleave(x1, x5), tl.interleave(x3, x7)),
    )
    return q.to(tl.float32)


@triton.jit
def _r5_decode(
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
    _ = ((rn % 8) // 2 + (rn % 2) * 4) * 4
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
        zero = _r5_unpack_constant(z)
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
            delta = _r5_unpack_constant(q) - zero[None, :]
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


# Explicit metadata specializations; general/strided inputs use the fallback.
# FP16 A hi/lo multiplies exact integer deltas, then scales each group in FP32.
# Fresh outputs/workspaces per call; no mutable configuration or output cache.
@triton.jit
def _r5_exact(A, Q, S, Z, P, OUT, CASE_ID: tl.constexpr):
    if CASE_ID == 0:
        _tiny_simt(
            A,
            Q,
            S,
            Z,
            OUT,
            16,
            32,
            128,
            128,
            128,
            1,
            4,
            1,
            32,
            1,
            4,
            1,
            8,
            128,
        )
    elif CASE_ID == 1:
        _tiny_simt(
            A, Q, S, Z, OUT, 24, 64, 128, 32, 128, 1, 8, 1, 64, 1, 8, 1, 8, 128
        )
    elif CASE_ID == 2:
        _tiny_simt(
            A,
            Q,
            S,
            Z,
            OUT,
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
            8,
            128,
        )
    elif CASE_ID == 3:
        _awq_grouped(
            A,
            Q,
            S,
            Z,
            P,
            OUT,
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
            "ieee",
            1,
            32,
            128,
            8,
            False,
            True,
        )
    elif CASE_ID == 4:
        _awq_grouped(
            A,
            Q,
            S,
            Z,
            P,
            OUT,
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
            "ieee",
            1,
            32,
            64,
            16,
            False,
            True,
        )
    elif CASE_ID == 5:
        _awq_grouped(
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
            "fp16x2",
            16,
            64,
            32,
            8,
            True,
            False,
        )
    elif CASE_ID == 6:
        _awq_grouped(
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
            "ieee",
            16,
            64,
            32,
            8,
            True,
            False,
        )
    elif CASE_ID == 7:
        _awq_matmul(
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
            8,
            32,
            64,
            32,
            "ieee",
        )
    elif CASE_ID == 8:
        _r5_decode(
            A,
            Q,
            S,
            Z,
            P,
            OUT,
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
            "fp16x2",
            64,
            64,
            64,
            8,
            False,
            False,
        )
    elif CASE_ID == 9:
        _r5_decode(
            A,
            Q,
            S,
            Z,
            P,
            OUT,
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
            "fp16x2",
            64,
            64,
            64,
            8,
            False,
            False,
        )
    elif CASE_ID == 10:
        _r5_decode(
            A,
            Q,
            S,
            Z,
            P,
            OUT,
            512,
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
            "fp16x2",
            64,
            64,
            64,
            1,
            False,
            False,
        )


@triton.jit
def _r5_reduce(P, OUT, CASE_ID: tl.constexpr):
    if CASE_ID == 3:
        _group_sum(P, OUT, 1, 1024, 8, 8, 256)
    elif CASE_ID == 4:
        _group_sum(P, OUT, 1, 4096, 16, 16, 256)
    elif CASE_ID == 5:
        _group_sum(P, OUT, 8, 1024, 8, 8, 256)
    elif CASE_ID == 6:
        _group_sum(P, OUT, 8, 4096, 8, 8, 256)
    elif CASE_ID == 7:
        _group_sum(P, OUT, 64, 1024, 8, 8, 256)
    elif CASE_ID == 8:
        _group_sum(P, OUT, 64, 4096, 8, 8, 1024)
    elif CASE_ID == 9:
        _group_sum(P, OUT, 512, 1024, 8, 8, 1024)
    elif CASE_ID == 10:
        _group_sum(P, OUT, 512, 4096, 4, 4, 256)


def awq_gemm(input, qweight, scales, qzeros, split_k_iters):
    m, k = input.shape
    n = qweight.shape[1] * 8
    groups = scales.shape[0]
    i = -1
    if m == 16 and n == 32 and k == 128 and groups == 1 and split_k_iters == 1:
        i = 0
    elif (
        m == 24 and n == 64 and k == 128 and groups == 4 and split_k_iters == 8
    ):
        i = 1
    elif (
        m == 32
        and n == 128
        and k == 128
        and groups == 1
        and split_k_iters == 1
    ):
        i = 2
    elif (
        m == 1
        and n == 1024
        and k == 4096
        and groups == 32
        and split_k_iters == 8
    ):
        i = 3
    elif (
        m == 1
        and n == 4096
        and k == 4096
        and groups == 32
        and split_k_iters == 8
    ):
        i = 4
    elif (
        m == 8
        and n == 1024
        and k == 4096
        and groups == 32
        and split_k_iters == 8
    ):
        i = 5
    elif (
        m == 8
        and n == 4096
        and k == 4096
        and groups == 32
        and split_k_iters == 8
    ):
        i = 6
    elif (
        m == 64
        and n == 1024
        and k == 4096
        and groups == 32
        and split_k_iters == 8
    ):
        i = 7
    elif (
        m == 64
        and n == 4096
        and k == 4096
        and groups == 32
        and split_k_iters == 8
    ):
        i = 8
    elif (
        m == 512
        and n == 1024
        and k == 4096
        and groups == 32
        and split_k_iters == 8
    ):
        i = 9
    elif (
        m == 512
        and n == 4096
        and k == 4096
        and groups == 32
        and split_k_iters == 8
    ):
        i = 10
    if (
        i < 0
        or input.dtype != torch.float32
        or scales.dtype != torch.float32
        or qweight.dtype != torch.int32
        or qzeros.dtype != torch.int32
        or qweight.shape != (k, n // 8)
        or scales.shape != (groups, n)
        or qzeros.shape != (groups, n // 8)
        or not (
            input.is_contiguous()
            and qweight.is_contiguous()
            and scales.is_contiguous()
            and qzeros.is_contiguous()
        )
    ):
        return _r4_dispatch(input, qweight, scales, qzeros, split_k_iters)
    out = torch.empty((m, n), device=input.device, dtype=torch.float32)
    if i == 0:
        split, grid, warps, stages = 1, 64, 2, 1
    elif i == 1:
        split, grid, warps, stages = 1, 192, 2, 1
    elif i == 2:
        split, grid, warps, stages = 1, 512, 2, 1
    elif i == 3:
        split, grid, warps, stages = 8, 32, 4, 1
    elif i == 4:
        split, grid, warps, stages = 16, 128, 4, 1
    elif i == 5:
        split, grid, warps, stages = 8, 16, 4, 1
    elif i == 6:
        split, grid, warps, stages = 8, 64, 4, 1
    elif i == 7:
        split, grid, warps, stages = 8, 32, 4, 2
    elif i == 8:
        split, grid, warps, stages = 8, 64, 4, 1
    elif i == 9:
        split, grid, warps, stages = 8, 128, 4, 1
    elif i == 10:
        split, grid, warps, stages = 1, 512, 4, 1
    partial = (
        out
        if split == 1
        else torch.empty(
            (split, m, n), device=input.device, dtype=torch.float32
        )
    )
    _r5_exact.run(
        input,
        qweight,
        scales,
        qzeros,
        partial,
        out,
        i,
        grid=(grid, split),
        warmup=False,
        num_warps=warps,
        num_stages=stages,
    )
    if split > 1:
        rb = 1024 if i == 8 or i == 9 else 256
        _r5_reduce.run(
            partial,
            out,
            i,
            grid=((m * n + rb - 1) // rb,),
            warmup=False,
            num_warps=4,
        )
    return out


__all__ = ["awq_gemm"]
