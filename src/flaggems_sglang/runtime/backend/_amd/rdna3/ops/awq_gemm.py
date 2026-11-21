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

"""awq_gemm for RDNA3 (gfx1100).

A single activation row takes an fp32 multiply-add kernel. More rows take a
matrix-core kernel: fp16 and bf16 inputs dequantize each weight tile in
registers to (q - z) * scale in the input dtype, rounded once as in the
reference, and take a single dot per K block. fp32 inputs keep fp32
accuracy: q - z is exact in fp16, the activations are split into high and
low fp16 halves, and each group's two dots are scaled in fp32.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _nibbles(w, shift):
    w = tl.interleave(w, w)
    w = tl.interleave(w, w)
    w = tl.interleave(w, w)
    return (w >> shift) & 15


@triton.jit
def _mm(
    A,
    Q,
    S,
    Z,
    P,
    M,
    SPLIT,
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
    HILO: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    # Split part j covers groups [j * GROUPS // SPLIT, (j + 1) * GROUPS //
    # SPLIT) and writes P[j]; with SPLIT == 1, P is the output.
    tile_n = tl.program_id(0) % tl.cdiv(N, BN)
    part = tl.program_id(1)
    rm = tl.program_id(0) // tl.cdiv(N, BN) * BM + tl.arange(0, BM)
    rn = tile_n * BN + tl.arange(0, BN)
    rp = tile_n * (BN // 8) + tl.arange(0, BN // 8)
    shift = ((rn % 8) // 2 + (rn % 2) * 4) * 4
    m_ok = rm < M
    n_ok = rn < N
    p_ok = rp < N // 8
    acc = tl.zeros((BM, BN), tl.float32)
    GROUPS: tl.constexpr = K // G
    for group in range(part * GROUPS // SPLIT, (part + 1) * GROUPS // SPLIT):
        zero = _nibbles(tl.load(Z + group * ZG + rp * ZN, p_ok, 0), shift)
        scale = tl.load(S + group * SG + rn * SN, n_ok, 0)
        if HILO:
            group_acc = tl.zeros((BM, BN), tl.float32)
        for block in range(tl.cdiv(G, BK)):
            local = block * BK + tl.arange(0, BK)
            rk = group * G + local
            k_ok = local < G
            q = tl.load(
                Q + rk[:, None] * QK + rp[None, :] * QN,
                k_ok[:, None] & p_ok[None, :],
                0,
            )
            delta = _nibbles(q, shift[None, :]) - zero[None, :]
            a = tl.load(
                A + rm[:, None] * AM + rk[None, :] * AK,
                m_ok[:, None] & k_ok[None, :],
                0,
            )
            if HILO:
                b = delta.to(tl.float16)
                a = a.to(tl.float32)
                ah = a.to(tl.float16)
                al = (a - ah.to(tl.float32)).to(tl.float16)
                group_acc = tl.dot(ah, b, group_acc)
                group_acc = tl.dot(al, b, group_acc)
            elif a.dtype == tl.bfloat16:
                # A bf16 multiply here does not round to nearest.
                b = delta.to(tl.float32) * scale.to(tl.float32)[None, :]
                acc = tl.dot(a, b.to(tl.bfloat16), acc)
            else:
                acc = tl.dot(a, delta.to(a.dtype) * scale[None, :], acc)
        if HILO:
            acc += group_acc * scale.to(tl.float32)[None, :]
    tl.store(
        P + part * M * N + rm[:, None] * N + rn[None, :],
        acc,
        m_ok[:, None] & n_ok[None, :],
    )


@triton.jit
def _gemv(
    A,
    Q,
    S,
    Z,
    P,
    SPLIT,
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
):
    # One activation row; the dequantized weight is rounded to the scales
    # dtype as in the reference, then multiplied and summed in fp32.
    tile_n = tl.program_id(0)
    part = tl.program_id(1)
    rn = tile_n * BN + tl.arange(0, BN)
    rp = tile_n * (BN // 8) + tl.arange(0, BN // 8)
    shift = ((rn % 8) // 2 + (rn % 2) * 4) * 4
    n_ok = rn < N
    p_ok = rp < N // 8
    acc = tl.zeros((BK, BN), tl.float32)
    GROUPS: tl.constexpr = K // G
    for group in range(part * GROUPS // SPLIT, (part + 1) * GROUPS // SPLIT):
        zero = _nibbles(tl.load(Z + group * ZG + rp * ZN, p_ok, 0), shift)
        scale = tl.load(S + group * SG + rn * SN, n_ok, 0)
        for block in range(tl.cdiv(G, BK)):
            local = block * BK + tl.arange(0, BK)
            rk = group * G + local
            k_ok = local < G
            q = tl.load(
                Q + rk[:, None] * QK + rp[None, :] * QN,
                k_ok[:, None] & p_ok[None, :],
                0,
            )
            delta = _nibbles(q, shift[None, :]) - zero[None, :]
            if scale.dtype == tl.float16:
                w = (delta.to(tl.float16) * scale[None, :]).to(tl.float32)
            else:
                w = delta.to(tl.float32) * scale.to(tl.float32)[None, :]
                if scale.dtype == tl.bfloat16:
                    # Round to nearest even at bf16 precision, kept as fp32.
                    bits = w.to(tl.uint32, bitcast=True)
                    bits = (bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000
                    w = bits.to(tl.float32, bitcast=True)
            a = tl.load(A + rk * AK, k_ok, 0).to(tl.float32)
            acc += a[:, None] * w
    tl.store(P + part * N + rn, tl.sum(acc, 0), n_ok)


@triton.jit
def _sum(P, OUT, SIZE, SPLIT: tl.constexpr, BLOCK: tl.constexpr):
    x = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    acc = tl.zeros((BLOCK,), tl.float32)
    for part in tl.static_range(SPLIT):
        acc += tl.load(P + part * SIZE + x, x < SIZE, 0)
    tl.store(OUT + x, acc, x < SIZE)


def _gemv_config(g, groups):
    """BN, BK, num_warps, num_stages and splits of the one-row kernel."""
    return 32, _block_k(64, g), 1, 1, min(16, groups)


def _block_k(bk, g):
    while bk > 16 and g % bk:
        bk //= 2
    return bk


def _config(m, n, g, groups, hilo):
    """BM, BN, BK, num_warps, num_stages and splits of the matrix kernel."""
    if hilo:
        if triton.cdiv(m, 16) * triton.cdiv(n, 32) <= 64:
            return 16, 32, _block_k(64, g), 2, 1, min(16, groups)
        if m > 64:
            return 32, 32, _block_k(32, g), 1, 1, min(8, groups)
        return 16, 32, _block_k(32, g), 1, 2, min(8, groups)
    if m <= 32:
        bm = 16 if m <= 16 else 32
        if triton.cdiv(m, bm) * triton.cdiv(n, 32) <= 16:
            return bm, 32, _block_k(128, g), bm // 8, 2, min(16, groups)
        if n <= 2048:
            return bm, 32, _block_k(64, g), 2, 1, min(16, groups)
        return bm, 32, _block_k(32, g), 1, 2, min(8, groups)
    if m <= 64:
        return 64, 32, _block_k(64, g), 2, 1, min(8, groups)
    bm = 128 if n >= 4096 else 64
    split = 512 // (triton.cdiv(m, bm) * triton.cdiv(n, 64))
    return bm, 64, _block_k(32, g), 4, 2, max(1, min(16, groups, split))


def awq_gemm(input, qweight, scales, qzeros, split_k_iters):
    m, k = input.shape
    n = qweight.shape[1] * 8
    groups = scales.shape[0]
    assert split_k_iters > 0
    assert qweight.shape[0] == k and scales.shape[1] == n
    assert qzeros.shape == (groups, n // 8)
    assert k == 0 or (groups > 0 and k % groups == 0)
    output = torch.empty((m, n), device=input.device, dtype=scales.dtype)
    if m == 0 or n == 0:
        return output
    if k == 0:
        return output.zero_()
    g = k // groups
    if m == 1:
        bn, bk, warps, stages, split = _gemv_config(g, groups)
        partial = (
            output
            if split == 1
            else torch.empty(
                (split, 1, n), device=input.device, dtype=torch.float32
            )
        )
        _gemv[(triton.cdiv(n, bn), split)](
            input,
            qweight,
            scales,
            qzeros,
            partial,
            split,
            n,
            k,
            g,
            input.stride(1),
            *qweight.stride(),
            *scales.stride(),
            *qzeros.stride(),
            bn,
            bk,
            num_warps=warps,
            num_stages=stages,
        )
    else:
        hilo = not (
            input.dtype in (torch.float16, torch.bfloat16)
            and scales.dtype == input.dtype
        )
        bm, bn, bk, warps, stages, split = _config(m, n, g, groups, hilo)
        partial = (
            output
            if split == 1
            else torch.empty(
                (split, m, n), device=input.device, dtype=torch.float32
            )
        )
        _mm[(triton.cdiv(m, bm) * triton.cdiv(n, bn), split)](
            input,
            qweight,
            scales,
            qzeros,
            partial,
            m,
            split,
            n,
            k,
            g,
            *input.stride(),
            *qweight.stride(),
            *scales.stride(),
            *qzeros.stride(),
            hilo,
            bm,
            bn,
            bk,
            num_warps=warps,
            num_stages=stages,
        )
    if split > 1:
        _sum[(triton.cdiv(m * n, 1024),)](
            partial, output, m * n, split, 1024, num_warps=4
        )
    return output


__all__ = ["awq_gemm"]
