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

import torch
import triton
import triton.language as tl


@triton.jit
def _full_gemv(
    A,
    B,
    C,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    A0: tl.constexpr,
    A1: tl.constexpr,
    B0: tl.constexpr,
    B1: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    row = tl.program_id(1)
    n = tl.program_id(0) * BN + tl.arange(0, BN)
    k = tl.arange(0, BK)
    acc = tl.zeros((BN, BK), tl.float32)
    for start in range(tl.cdiv(K, BK)):
        kk = start * BK + k
        a = tl.load(A + row * A0 + kk * A1, kk < K, 0).to(tl.float32)
        b = tl.load(
            B + n[:, None] * B1 + kk[None, :] * B0,
            (n[:, None] < N) & (kk[None, :] < K),
            0,
            cache_modifier=".cg",
        ).to(tl.float32)
        acc += a[None, :] * b
    tl.store(C + row * N + n, tl.sum(acc, 1), n < N)


@triton.jit
def _cta_aligned(
    A,
    B,
    C,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    A0: tl.constexpr,
    A1: tl.constexpr,
    B0: tl.constexpr,
    B1: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    GROUPS: tl.constexpr,
):
    groups = tl.arange(0, GROUPS)
    m = tl.arange(0, 16)
    rows = m % M
    n = tl.program_id(0) * BN + tl.arange(0, BN)
    k = groups[:, None] * BK + tl.arange(0, BK)[None, :]
    acc = tl.zeros((GROUPS, 16, BN), tl.float32)
    for start in range(tl.cdiv(K, BK * GROUPS)):
        kk = k + start * BK * GROUPS
        ap = A + rows[None, :, None] * A0 + kk[:, None, :] * A1
        bp = B + kk[:, :, None] * B0 + n[None, None, :] * B1
        if K % (BK * GROUPS) == 0:
            a = tl.load(ap)
            if N % BN == 0:
                b = tl.load(bp, cache_modifier=".cg")
            else:
                b = tl.load(bp, n[None, None, :] < N, 0, cache_modifier=".cg")
        else:
            a = tl.load(ap, kk[:, None, :] < K, 0)
            b = tl.load(
                bp,
                (kk[:, :, None] < K) & (n[None, None, :] < N),
                0,
                cache_modifier=".cg",
            )
        acc = tl.dot(a, b, acc, input_precision="ieee")
    total = tl.sum(acc, 0)
    tl.store(
        C + m[:, None] * N + n[None, :],
        total,
        (m[:, None] < M) & (n[None, :] < N),
    )


@triton.jit
def _transposed(
    A,
    B,
    C,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    A0: tl.constexpr,
    A1: tl.constexpr,
    B0: tl.constexpr,
    B1: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    GROUPS: tl.constexpr,
):
    groups = tl.arange(0, GROUPS)
    m = tl.arange(0, 16)
    rows = m % M
    n = tl.program_id(0) * BN + tl.arange(0, BN)
    k = groups[:, None] * BK + tl.arange(0, BK)[None, :]
    acc = tl.zeros((GROUPS, BN, 16), tl.float32)
    for start in range(tl.cdiv(K, BK * GROUPS)):
        kk = k + start * BK * GROUPS
        ap = A + rows[None, :, None] * A0 + kk[:, None, :] * A1
        bp = B + kk[:, :, None] * B0 + n[None, None, :] * B1
        if K % (BK * GROUPS) == 0:
            a = tl.load(ap)
            if N % BN == 0:
                b = tl.load(bp, cache_modifier=".cg")
            else:
                b = tl.load(bp, n[None, None, :] < N, 0, cache_modifier=".cg")
        else:
            a = tl.load(ap, kk[:, None, :] < K, 0)
            b = tl.load(
                bp,
                (kk[:, :, None] < K) & (n[None, None, :] < N),
                0,
                cache_modifier=".cg",
            )
        acc = tl.dot(
            tl.trans(b, 0, 2, 1),
            tl.trans(a, 0, 2, 1),
            acc,
            input_precision="ieee",
        )
    total = tl.trans(tl.sum(acc, 0))
    tl.store(
        C + m[:, None] * N + n[None, :],
        total,
        (m[:, None] < M) & (n[None, :] < N),
    )


@triton.jit
def _split_k(
    A,
    B,
    P,
    M,
    N,
    K,
    A0,
    A1,
    B0,
    B1,
    SPLITS: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    EVEN_K: tl.constexpr,
):
    num_n = tl.cdiv(N, BN)
    pid_n = tl.program_id(0) % num_n
    part = tl.program_id(0) // num_n
    m = tl.program_id(1) * BM + tl.arange(0, BM)
    n = pid_n * BN + tl.arange(0, BN)
    k = tl.arange(0, BK)
    m_mask = m[:, None] < M
    n_mask = n[None, :] < N
    acc = tl.zeros((BM, BN), tl.float32)
    for start in range(part * BK, K, SPLITS * BK):
        kk = start + k
        ap = A + m[:, None] * A0 + kk[None, :] * A1
        bp = B + kk[:, None] * B0 + n[None, :] * B1
        if EVEN_K:
            a = tl.load(ap, m_mask, 0)
            b = tl.load(bp, n_mask, 0)
        else:
            a = tl.load(ap, m_mask & (kk[None, :] < K), 0)
            b = tl.load(bp, (kk[:, None] < K) & n_mask, 0)
        acc = tl.dot(a, b, acc, input_precision="ieee")
    tl.store(
        P + (part * M + m[:, None]) * N + n[None, :], acc, m_mask & n_mask
    )


@triton.jit
def _reduce(P, C, TOTAL, SPLITS: tl.constexpr, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < TOTAL
    acc = tl.load(P + offs, mask, 0.0)
    for part in tl.static_range(1, SPLITS):
        acc += tl.load(P + part * TOTAL + offs, mask, 0.0)
    tl.store(C + offs, acc, mask)


@triton.jit
def _shared_b(
    A,
    B,
    C,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    A0: tl.constexpr,
    A1: tl.constexpr,
    B0: tl.constexpr,
    B1: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    m = tl.arange(0, BM)
    n = tl.program_id(0) * BN + tl.arange(0, BN)
    k = tl.arange(0, BK)
    acc = tl.zeros((BM, BN, BK), tl.float32)
    for start in range(tl.cdiv(K, BK)):
        kk = start * BK + k
        a = tl.load(
            A + m[:, None] * A0 + kk[None, :] * A1,
            (m[:, None] < M) & (kk[None, :] < K),
            0,
        ).to(tl.float32)
        b = tl.load(
            B + n[:, None] * B1 + kk[None, :] * B0,
            (n[:, None] < N) & (kk[None, :] < K),
            0,
            cache_modifier=".cg",
        ).to(tl.float32)
        acc += a[:, None, :] * b[None, :, :]
    result = tl.sum(acc, 2)
    tl.store(
        C + m[:, None] * N + n[None, :],
        result,
        (m[:, None] < M) & (n[None, :] < N),
    )


def dsv3_fused_a_gemm(mat_a, mat_b):
    m, k = mat_a.shape
    n = mat_b.shape[1]
    out = torch.empty((m, n), dtype=mat_a.dtype, device=mat_a.device)
    strides = (*mat_a.stride(), *mat_b.stride())
    if m == 1:
        _full_gemv.run(
            mat_a,
            mat_b,
            out,
            m,
            n,
            k,
            *strides,
            1,
            triton.next_power_of_2(k),
            grid=(n, m),
            warmup=False,
            num_warps=8,
            num_stages=1,
        )
    elif m == 2:
        _shared_b.run(
            mat_a,
            mat_b,
            out,
            m,
            n,
            k,
            *strides,
            2,
            2,
            1024,
            grid=(triton.cdiv(n, 2),),
            warmup=False,
            num_warps=8,
            num_stages=1,
        )
    elif k > 4096 and n <= 3072:
        # gfx1100: one program per 16 columns leaves 132 programs for the
        # 2112-wide weight, each streaming all of K. Splitting K four ways
        # (interleaved 256-wide blocks, so concurrent programs share one
        # K window) is 1.1-1.3x for M = 3-16; contiguous splits whose
        # starts sat an even number of blocks apart ran up to 2.5x slower.
        # From N = 4096 on the column programs alone fill the GPU and the
        # split is slower; the prefetch stage only pays up to N = 2112.
        splits = 4
        partial = torch.empty(
            (splits, m, n), dtype=torch.float32, device=mat_a.device
        )
        _split_k[(triton.cdiv(n, 64) * splits, triton.cdiv(m, 16))](
            mat_a,
            mat_b,
            partial,
            m,
            n,
            k,
            *strides,
            SPLITS=splits,
            BM=16,
            BN=64,
            BK=256,
            EVEN_K=k % 256 == 0,
            num_warps=4,
            num_stages=2 if n <= 2112 else 1,
        )
        _reduce[(triton.cdiv(m * n, 256),)](
            partial, out, m * n, SPLITS=splits, BLOCK=256, num_warps=1
        )
    elif m >= 8 and k > 4096:
        _transposed.run(
            mat_a,
            mat_b,
            out,
            m,
            n,
            k,
            *strides,
            16,
            128,
            4,
            grid=(triton.cdiv(n, 16),),
            warmup=False,
            num_warps=4,
            num_stages=2,
        )
    else:
        groups, warps, stages = (4, 4, 2) if k > 4096 else (8, 8, 1)
        _cta_aligned.run(
            mat_a,
            mat_b,
            out,
            m,
            n,
            k,
            *strides,
            16,
            128,
            groups,
            grid=(triton.cdiv(n, 16),),
            warmup=False,
            num_warps=warps,
            num_stages=stages,
        )
    return out


__all__ = ["dsv3_fused_a_gemm"]
