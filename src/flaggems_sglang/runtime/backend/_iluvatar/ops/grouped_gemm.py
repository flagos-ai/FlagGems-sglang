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


@triton.jit(do_not_specialize=["A", "C", "OFFS"])
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
    pid = tl.program_id(0)
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
        n = tl.program_id(1) * BN + tl.arange(0, BN)
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
    if m >= 2048 and m // num_experts >= 64:
        bt = torch.empty(b.shape, dtype=b.dtype, device=b.device)
        _transpose_weights[
            num_experts, triton.cdiv(n, 32), triton.cdiv(k, 128)
        ](b, bt, n, k, 32, 128, num_warps=4)
        out = torch.empty((m, n), dtype=a.dtype, device=a.device)
        _packed[triton.cdiv(m, 64) + num_experts + 2, triton.cdiv(n, 256)](
            a,
            bt,
            out,
            expert_token_offs,
            m,
            n,
            k,
            num_experts,
            triton.next_power_of_2(num_experts + 2),
            64,
            256,
            32,
            n % 256 != 0 or k % 32 != 0,
            num_warps=4,
            num_stages=3,
        )
        return out
    if m <= 512 and k >= 1024:
        return _split_run(
            (
                a,
                b,
                num_experts,
                num_tokens_per_expert,
                expert_token_offs,
                expert_block_offs,
                expert_block_schedule,
            ),
            (16, 128, 64, 4, 4),
        )
    out = torch.empty((m, n), device=a.device, dtype=a.dtype)
    per = m // num_experts
    if per <= 64 and m <= 512:
        bm, bn, bk, w = (16, 128, 128, 4)
    elif per <= 64:
        bm, bn, bk, w = (64, 128, 64, 8)
    else:
        bm, bn, bk, w = (64, 256, 64, 4)
    bn = max(16, min(bn, triton.next_power_of_2(n)))
    bk = max(16, min(bk, triton.next_power_of_2(k)))
    while (bm + bn) * bk > 20480:
        if bk > 32:
            bk //= 2
        else:
            bn //= 2
    _gemm[triton.cdiv(m, bm) + num_experts + 2, triton.cdiv(n, bn)](
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
        num_stages=3,
    )
    return out


@triton.jit(do_not_specialize=["A", "P", "OFFS"])
def _partial(A, B, P, OFFS, META: tl.constexpr):
    M: tl.constexpr = META[0]
    N: tl.constexpr = META[1]
    K: tl.constexpr = META[2]
    E: tl.constexpr = META[3]
    EP: tl.constexpr = META[4]
    BM: tl.constexpr = META[5]
    BN: tl.constexpr = META[6]
    BK: tl.constexpr = META[7]
    SK: tl.constexpr = META[8]
    ei = tl.arange(0, EP)
    lo = tl.load(OFFS + tl.minimum(ei, E), ei < E, 0)
    hi = tl.load(OFFS + tl.minimum(ei + 1, E), ei < E, 0)
    lo = tl.where(ei == E, tl.load(OFFS + E), tl.where(ei == E + 1, 0, lo))
    hi = tl.where(ei == E, M, tl.where(ei == E + 1, tl.load(OFFS), hi))
    nb = tl.where(ei <= E + 1, tl.cdiv(tl.maximum(hi - lo, 0), BM), 0)
    end = tl.cumsum(nb, 0)
    NC: tl.constexpr = (N + BN - 1) // BN
    total = tl.sum(nb, 0)
    for task in range(tl.program_id(0), total * NC * SK, tl.num_programs(0)):
        split = task % SK
        column = task // SK % NC
        pid = task // (SK * NC)
        seg = tl.sum((end <= pid).to(tl.int32), 0)
        if seg <= E + 1:
            at = ei == seg
            r = (
                tl.sum(tl.where(at, lo - (end - nb) * BM, 0), 0)
                + pid * BM
                + tl.arange(0, BM)
            )
            rows = r < tl.sum(tl.where(at, hi, 0), 0)
            n = column * BN + tl.arange(0, BN)
            acc = tl.zeros((BM, BN), tl.float32)
            if seg < E:
                k = split * BK + tl.arange(0, BK)
                for i in range(tl.cdiv(K, BK * SK)):
                    av = tl.load(
                        A + r[:, None] * K + k[None, :],
                        rows[:, None] & (k < K)[None, :],
                        0,
                    )
                    bv = tl.load(
                        B + (seg * N + n[None, :]) * K + k[:, None],
                        (n < N)[None, :] & (k < K)[:, None],
                        0,
                    )
                    acc = tl.dot(av, bv, acc)
                    k += BK * SK
            tl.store(
                P + split * M * N + r[:, None] * N + n[None, :],
                acc,
                rows[:, None] & (n < N)[None, :],
            )


@triton.jit(do_not_specialize=["P", "C"])
def _finish(P, C, META: tl.constexpr):
    SIZE: tl.constexpr = META[0]
    SK: tl.constexpr = META[1]
    BLOCK: tl.constexpr = META[2]
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    acc = tl.full((BLOCK,), 0, tl.float32)
    for s in tl.static_range(SK):
        acc += tl.load(P + s * SIZE + i, i < SIZE, 0)
    tl.store(C + i, acc, i < SIZE)


def _split_run(args, cfg):
    a, b, e, counts, offs, boffs, sched = args
    m, k = a.shape
    n = b.shape[1]
    bm, bn, bk, sk, warps = cfg
    parts = torch.empty((sk, m, n), device=a.device, dtype=torch.float32)
    out = torch.empty((m, n), device=a.device, dtype=a.dtype)
    if e < 16:
        _partial.run(
            a,
            b,
            parts,
            offs,
            (m, n, k, e, triton.next_power_of_2(e + 2), bm, bn, bk, sk),
            num_warps=warps,
            num_stages=2,
            grid=(
                min(
                    128, sk * triton.cdiv(n, bn) * (triton.cdiv(m, bm) + e + 2)
                ),
            ),
            warmup=False,
        )
    else:
        _partial_standard.run(
            a,
            b,
            parts,
            offs,
            (m, n, k, e, triton.next_power_of_2(e + 2), bm, bn, bk, sk),
            num_warps=warps,
            num_stages=2,
            grid=(sk, triton.cdiv(n, bn), triton.cdiv(m, bm) + e + 2),
            warmup=False,
        )
    _finish.run(
        parts,
        out,
        (m * n, sk, 512),
        grid=(triton.cdiv(m * n, 512),),
        warmup=False,
    )
    return out


@triton.jit(do_not_specialize=["A", "C", "OFFS"])
def _packed(
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
    pid = tl.program_id(0)
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
        n = tl.program_id(1) * BN + tl.arange(0, BN)
        acc = tl.zeros((BM, BN), tl.float32)
        if seg < E:
            k = tl.arange(0, BK)
            ap = A + r[:, None] * K + k[None, :]
            bp = B + seg * N * K + n[None, :] + k[:, None] * N
            if EDGE:
                cols = n < N
                for i in range(tl.cdiv(K, BK)):
                    kk = i * BK + k < K
                    av = tl.load(ap, rows[:, None] & kk[None, :], 0)
                    bv = tl.load(bp, cols[None, :] & kk[:, None], 0)
                    acc = tl.dot(av, bv, acc)
                    ap += BK
                    bp += BK * N
            else:
                for i in range(K // BK):
                    acc = tl.dot(
                        tl.load(ap, rows[:, None], 0), tl.load(bp), acc
                    )
                    ap += BK
                    bp += BK * N
        mask = rows[:, None]
        if EDGE:
            mask = mask & (n[None, :] < N)
        tl.store(C + r[:, None] * N + n[None, :], acc, mask)


@triton.jit
def _transpose_weights(
    B, BT, N: tl.constexpr, K: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr
):
    e = tl.program_id(0)
    n = tl.program_id(1) * BN + tl.arange(0, BN)
    k = tl.program_id(2) * BK + tl.arange(0, BK)
    v = tl.load(
        B + (e * N + n[:, None]) * K + k[None, :],
        (n < N)[:, None] & (k < K)[None, :],
        0,
    )
    tl.store(
        BT + (e * K + k[None, :]) * N + n[:, None],
        v,
        (n < N)[:, None] & (k < K)[None, :],
    )


@triton.jit(do_not_specialize=["A", "P", "OFFS"])
def _partial_standard(A, B, P, OFFS, META: tl.constexpr):
    M: tl.constexpr = META[0]
    N: tl.constexpr = META[1]
    K: tl.constexpr = META[2]
    E: tl.constexpr = META[3]
    EP: tl.constexpr = META[4]
    BM: tl.constexpr = META[5]
    BN: tl.constexpr = META[6]
    BK: tl.constexpr = META[7]
    SK: tl.constexpr = META[8]
    pid = tl.program_id(2)
    split = tl.program_id(0)
    ei = tl.arange(0, EP)
    lo = tl.load(OFFS + tl.minimum(ei, E), ei < E, 0)
    hi = tl.load(OFFS + tl.minimum(ei + 1, E), ei < E, 0)
    lo = tl.where(ei == E, tl.load(OFFS + E), tl.where(ei == E + 1, 0, lo))
    hi = tl.where(ei == E, M, tl.where(ei == E + 1, tl.load(OFFS), hi))
    nb = tl.where(ei <= E + 1, tl.cdiv(tl.maximum(hi - lo, 0), BM), 0)
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
        n = tl.program_id(1) * BN + tl.arange(0, BN)
        acc = tl.zeros((BM, BN), tl.float32)
        if seg < E:
            k = split * BK + tl.arange(0, BK)
            for i in range(tl.cdiv(K, BK * SK)):
                av = tl.load(
                    A + r[:, None] * K + k[None, :],
                    rows[:, None] & (k < K)[None, :],
                    0,
                )
                bv = tl.load(
                    B + (seg * N + n[None, :]) * K + k[:, None],
                    (n < N)[None, :] & (k < K)[:, None],
                    0,
                )
                acc = tl.dot(av, bv, acc)
                k += BK * SK
        tl.store(
            P + split * M * N + r[:, None] * N + n[None, :],
            acc,
            rows[:, None] & (n < N)[None, :],
        )


__all__ = ["grouped_gemm"]
