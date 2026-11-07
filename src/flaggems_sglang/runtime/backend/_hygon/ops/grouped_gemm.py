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


@triton.jit(do_not_specialize=["OFFS"])
def _gemm(
    A,
    B,
    C,
    OFFS,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    E: tl.constexpr,
    EP: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    EDGE: tl.constexpr,
):
    pid = tl.program_id(1)
    ei = tl.arange(0, EP)
    exp = ei < E
    lo = tl.load(OFFS + tl.minimum(ei, E), exp, 0)
    hi = tl.load(OFFS + tl.minimum(ei + 1, E), exp, 0)
    lo = tl.where(ei == E, tl.load(OFFS + E), tl.where(ei == E + 1, 0, lo))
    hi = tl.where(ei == E, M, tl.where(ei == E + 1, tl.load(OFFS), hi))
    nb = tl.where(ei <= E + 1, (tl.maximum(hi - lo, 0) + BM - 1) // BM, 0)
    end = tl.cumsum(nb, 0)
    seg = tl.sum((end <= pid).to(tl.int32), 0)
    if seg <= E + 1:
        at = ei == seg
        r = (
            tl.sum(tl.where(at, lo - (end - nb) * BM, 0), 0)
            + pid * BM
            + tl.arange(0, BM)
        )
        rows = r < tl.sum(tl.where(at, hi, 0), 0)
        n = tl.program_id(0) * BN + tl.arange(0, BN)
        acc = tl.zeros((BM, BN), tl.float32)
        if seg < E:
            k = tl.arange(0, BK)
            if M >= 2048 and M // E <= 256:
                ar = tl.minimum(r, M - 1)
            else:
                ar = r
            ap = A + ar[:, None] * K + k[None, :]
            bp = B + (seg * N + n[None, :]) * K + k[:, None]
            if EDGE:
                cols = n < N
                for i in range(tl.cdiv(K, BK)):
                    kk = i * BK + k < K
                    av = tl.load(ap, rows[:, None] & kk[None, :], 0)
                    bv = tl.load(bp, cols[None, :] & kk[:, None], 0)
                    acc = tl.dot(av, bv, acc)
                    ap += BK
                    bp += BK
            else:
                for i in range(K // BK):
                    if M >= 2048 and M // E <= 256:
                        av = tl.load(ap)
                    else:
                        av = tl.load(ap, rows[:, None], 0)
                    acc = tl.dot(av, tl.load(bp), acc)
                    ap += BK
                    bp += BK
        mask = rows[:, None]
        if EDGE:
            mask = mask & (n[None, :] < N)
        tl.store(C + r[:, None] * N + n[None, :], acc, mask)


def grouped_gemm(
    a,
    b,
    num_experts,
    num_tokens_per_expert,
    expert_token_offs,
    expert_block_offs,
    expert_block_schedule,
):
    m, k = a.shape
    n = b.shape[1]
    if m <= 256 and k >= 1024 and 8 < m // num_experts <= 32:
        bn = max(16, min(64, triton.next_power_of_2(n)))
        out = torch.empty((m, n), device=a.device, dtype=a.dtype)
        _gemm[(triton.cdiv(n, bn), triton.cdiv(m, 16) + num_experts + 2)](
            a,
            b,
            out,
            expert_token_offs,
            m,
            n,
            k,
            num_experts,
            triton.next_power_of_2(num_experts + 2),
            16,
            bn,
            128,
            n % bn != 0 or k % 128 != 0,
            num_warps=4,
            num_stages=2,
        )
        return out
    if m <= 512 and k <= 512:
        bn, bk = 8, 256
        out = torch.empty((m, n), dtype=a.dtype, device=a.device)
        _gemv[(triton.cdiv(n, bn), m)](
            a,
            b,
            out,
            expert_token_offs,
            m,
            n,
            k,
            num_experts,
            triton.next_power_of_2(num_experts),
            bn,
            bk,
            num_warps=4,
            num_stages=1,
        )
        return out
    out = torch.empty((m, n), device=a.device, dtype=a.dtype)
    per = m // num_experts
    if k >= 1024 and not (m >= 2048 and per <= 256):
        if per <= 8:
            bm, bn, bk, w = 32, 128, 64, 4
        elif per <= 256:
            bm, bn, bk, w = 64, 128, 64, 8
        else:
            bm, bn, bk, w = 128, 128, 32, 8
        bn = max(16, min(bn, triton.next_power_of_2(n)))
        _dual[(triton.cdiv(n, bn), triton.cdiv(m, bm) + num_experts + 2)](
            a,
            b,
            out,
            expert_token_offs,
            m,
            n,
            k,
            num_experts,
            triton.next_power_of_2(num_experts + 2),
            bm,
            bn,
            bk,
            2,
            n % bn != 0 or k % bk != 0,
            num_warps=w,
            num_stages=2,
        )
        return out
    if per <= 64:
        bm, bn, bk, w, s = 32, 64, 128, 4, 2
    elif per <= 256:
        bm, bn, bk, w, s = 64, 128, 64, 8, 2
    else:
        bm, bn, bk, w, s = 128, 128, 32, 8, 2
    bn = max(16, min(bn, triton.next_power_of_2(n)))
    bk = max(16, min(bk, triton.next_power_of_2(k)))
    while (bm + bn) * bk > 16384:
        if bk > 32:
            bk //= 2
        else:
            bn //= 2
    _gemm[(triton.cdiv(n, bn), triton.cdiv(m, bm) + num_experts + 2)](
        a,
        b,
        out,
        expert_token_offs,
        m,
        n,
        k,
        num_experts,
        triton.next_power_of_2(num_experts + 2),
        bm,
        bn,
        bk,
        n % bn != 0 or k % bk != 0,
        num_warps=w,
        num_stages=s,
    )
    return out


@triton.jit(do_not_specialize=["OFFS"])
def _gemv(
    A,
    B,
    C,
    OFFS,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    E: tl.constexpr,
    EP: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    r = tl.program_id(1)
    ei = tl.arange(0, EP)
    lo = tl.load(OFFS + ei, ei < E, 0)
    hi = tl.load(OFFS + ei + 1, ei < E, 0)
    at = (ei < E) & (r >= lo) & (r < hi)
    valid = tl.sum(at.to(tl.int32), 0) != 0
    e = tl.sum(tl.where(at, ei, 0), 0)
    ns = tl.program_id(0) * BN + tl.arange(0, BN)
    ks = tl.arange(0, BK)
    accum = tl.zeros((BN, BK), tl.float32)
    if valid:
        for start in range(tl.cdiv(K, BK)):
            k = start * BK + ks
            av = tl.load(A + r * K + k, k < K, 0).to(tl.float32)
            bv = tl.load(
                B + (e * N + ns[:, None]) * K + k[None, :],
                (ns < N)[:, None] & (k < K)[None, :],
                0,
            ).to(tl.float32)
            accum += bv * av[None, :]
    out = tl.sum(accum, 1)
    tl.store(C + r * N + ns, out, ns < N)


@triton.jit(do_not_specialize=["OFFS"])
def _dual(
    A,
    B,
    C,
    OFFS,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    E: tl.constexpr,
    EP: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    FACTOR: tl.constexpr,
    EDGE: tl.constexpr,
):
    pid = tl.program_id(1)
    ei = tl.arange(0, EP)
    exp = ei < E
    lo = tl.load(OFFS + tl.minimum(ei, E), exp, 0)
    hi = tl.load(OFFS + tl.minimum(ei + 1, E), exp, 0)
    lo = tl.where(ei == E, tl.load(OFFS + E), tl.where(ei == E + 1, 0, lo))
    hi = tl.where(ei == E, M, tl.where(ei == E + 1, tl.load(OFFS), hi))
    nb = tl.where(ei <= E + 1, (tl.maximum(hi - lo, 0) + BM - 1) // BM, 0)
    end = tl.cumsum(nb, 0)
    seg = tl.sum((end <= pid).to(tl.int32), 0)
    if seg <= E + 1:
        at = ei == seg
        r = (
            tl.sum(tl.where(at, lo - (end - nb) * BM, 0), 0)
            + pid * BM
            + tl.arange(0, BM)
        )
        rows = r < tl.sum(tl.where(at, hi, 0), 0)
        n = tl.program_id(0) * BN + tl.arange(0, BN)
        acc = tl.zeros((BM, BN), tl.float32)
        if seg < E:
            k = tl.arange(0, BK)
            ap = A + r[:, None] * K + k[None, :]
            bp = B + (seg * N + n[None, :]) * K + k[:, None]
            if EDGE:
                cols = n < N
                for i in range(tl.cdiv(K, BK)):
                    kk = i * BK + k < K
                    av = tl.load(ap, rows[:, None] & kk[None, :], 0)
                    bv = tl.load(bp, cols[None, :] & kk[:, None], 0)
                    acc = tl.dot(av, bv, acc)
                    ap += BK
                    bp += BK
            else:
                if K % (BK * FACTOR) == 0:
                    acc0 = tl.zeros((BM, BN), tl.float32)
                    acc1 = tl.zeros((BM, BN), tl.float32)
                    acc2 = tl.zeros((BM, BN), tl.float32)
                    acc3 = tl.zeros((BM, BN), tl.float32)
                    for i in range(K // (BK * FACTOR)):
                        if FACTOR > 0:
                            av0 = tl.load(ap, rows[:, None], 0)
                            bv0 = tl.load(bp)
                            acc0 = tl.dot(av0, bv0, acc0)
                            ap += BK
                            bp += BK
                        if FACTOR > 1:
                            av1 = tl.load(ap, rows[:, None], 0)
                            bv1 = tl.load(bp)
                            acc1 = tl.dot(av1, bv1, acc1)
                            ap += BK
                            bp += BK
                        if FACTOR > 2:
                            av2 = tl.load(ap, rows[:, None], 0)
                            bv2 = tl.load(bp)
                            acc2 = tl.dot(av2, bv2, acc2)
                            ap += BK
                            bp += BK
                        if FACTOR > 3:
                            av3 = tl.load(ap, rows[:, None], 0)
                            bv3 = tl.load(bp)
                            acc3 = tl.dot(av3, bv3, acc3)
                            ap += BK
                            bp += BK
                    acc = (acc0 + acc1) + (acc2 + acc3)
                else:
                    for i in range(K // BK):
                        acc = tl.dot(
                            tl.load(ap, rows[:, None], 0), tl.load(bp), acc
                        )
                        ap += BK
                        bp += BK
        mask = rows[:, None]
        if EDGE:
            mask = mask & (n[None, :] < N)
        tl.store(C + r[:, None] * N + n[None, :], acc, mask)


__all__ = ["grouped_gemm"]
