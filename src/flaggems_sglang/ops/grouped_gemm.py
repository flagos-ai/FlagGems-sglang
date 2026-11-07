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


@triton.jit
def _gemm(
    A,
    B,
    C,
    OFFS,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    E,
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
    nb = tl.where(exp, tl.cdiv(tl.maximum(hi - lo, 0), BM), 0)
    end = tl.cumsum(nb, 0)
    at = (pid >= end - nb) & (pid < end) & exp
    total = tl.sum(nb, 0)
    head = tl.load(OFFS)
    tail = tl.load(OFFS + E)
    tail_blocks = tl.cdiv(M - tail, BM)
    head_blocks = tl.cdiv(head, BM)
    tail_task = (pid >= total) & (pid < total + tail_blocks)
    head_task = (pid >= total + tail_blocks) & (
        pid < total + tail_blocks + head_blocks
    )
    seg = tl.where(
        pid < total,
        tl.sum(tl.where(at, ei, 0), 0),
        tl.where(tail_task, E, tl.where(head_task, E + 1, E + 2)),
    )
    start = tl.sum(tl.where(at, lo + (pid - end + nb) * BM, 0), 0)
    start = tl.where(
        tail_task,
        tail + (pid - total) * BM,
        tl.where(head_task, (pid - total - tail_blocks) * BM, start),
    )
    stop = tl.sum(tl.where(at, hi, 0), 0)
    stop = tl.where(tail_task, M, tl.where(head_task, head, stop))
    if seg <= E + 1:
        r = start + tl.arange(0, BM)
        rows = r < stop
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
    if torch.version.hip is not None and 512 < m < 2048 and k >= 1024:
        bm = 64 if m // num_experts <= 64 else 128
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
            (bm, 128, 32, 4, 8),
        )
    if torch.version.hip is not None and m >= 2048:
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
            (128, 128, 32, 4, 8),
        )
    if (
        torch.version.hip is not None
        and m <= 512
        and k >= 1024
        and k % 2 == 0
        and 8 < m // num_experts <= 32
    ):
        out = torch.empty((m, n), dtype=a.dtype, device=a.device)
        _bf16dot61[(triton.cdiv(m, 8) + num_experts + 2, triton.cdiv(n, 4))](
            a,
            b,
            out,
            expert_token_offs,
            m,
            n,
            k,
            num_experts,
            triton.next_power_of_2(num_experts + 2),
            8,
            4,
            512,
            ((a.data_ptr() | b.data_ptr()) & 3) != 0,
            num_warps=4,
            num_stages=1,
        )
        return out
    if (
        torch.version.hip is not None
        and m <= 512
        and k >= 1024
        and 8 < m // num_experts <= 32
    ):
        out = torch.empty((m, n), dtype=a.dtype, device=a.device)
        _gemv_multi[(triton.cdiv(m, 4) + num_experts + 2, triton.cdiv(n, 16))](
            a,
            b,
            out,
            expert_token_offs,
            m,
            n,
            k,
            num_experts,
            triton.next_power_of_2(num_experts + 2),
            4,
            16,
            256,
            num_warps=4,
            num_stages=1,
        )
        return out
    if (
        torch.version.hip is not None
        and m <= 512
        and (k <= 512 or (m // num_experts <= 8 and k >= 1024))
    ):
        bn = 16 if k <= 512 else 8
        bk = 256 if k <= 512 else 1024
        out = torch.empty((m, n), dtype=a.dtype, device=a.device)
        _gemv[(m, triton.cdiv(n, bn))](
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
    if (
        torch.version.hip is not None
        and m >= 512
        and m // num_experts <= 64
        and k >= 2048
    ):
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
            (32, 128, 64, 2, 4),
        )
    out = torch.empty((m, n), device=a.device, dtype=a.dtype)
    per = m // num_experts
    if torch.version.hip is None and m < 2048 and k >= 1024:
        bm = 16 if per <= 12 else 32 if per <= 64 else 64
        bn = 128 if n >= 512 else 64 if m * n >= 1 << 16 else 32
        bn = max(16, min(bn, triton.next_power_of_2(n)))
        bk = 128
        _gemm[(triton.cdiv(m, bm) + num_experts + 2, triton.cdiv(n, bn))](
            a,
            b,
            out,
            expert_token_offs,
            m,
            n,
            k,
            num_experts,
            triton.next_power_of_2(num_experts),
            bm,
            bn,
            bk,
            n % bn != 0 or k % bk != 0,
            num_warps=4,
            num_stages=3,
        )
        return out
    if torch.version.hip is None:
        bm = 16 if per <= 12 else 32 if per <= 64 else 64
        bn = 128 if n >= 512 else 64 if m * n >= 1 << 16 else 32
        bk = 128 if k >= 1024 and per <= 64 else 64
        w, s = 4, 3
    elif m >= 4096:
        bm, bn, bk, w, s = 128, 128, 32, 8, 2
    elif per >= 65:
        bm, bn, bk, w, s = 64, 128, 64, 4, 3
    elif m <= 200 and k >= 2048 and per >= 16:
        bm, bn, bk, w, s = 32, 64, 128, 4, 3
    elif k >= 2048 or m > 200:
        bm, bn, bk, w, s = 32, 128, 64, 4, 3
    else:
        bm, bn, bk, w, s = 64, 64, 128, 4, 2
    bn = max(16, min(bn, triton.next_power_of_2(n)))
    bk = max(16, min(bk, triton.next_power_of_2(k)))
    while (bm + bn) * bk > 16384:
        if bk > 32:
            bk //= 2
        else:
            bn //= 2
    _gemm[(triton.cdiv(m, bm) + num_experts + 2, triton.cdiv(n, bn))](
        a,
        b,
        out,
        expert_token_offs,
        m,
        n,
        k,
        num_experts,
        triton.next_power_of_2(num_experts),
        bm,
        bn,
        bk,
        n % bn != 0 or k % bk != 0,
        num_warps=w,
        num_stages=s,
    )
    return out


@triton.jit
def _partial(
    A,
    B,
    P,
    OFFS,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    E: tl.constexpr,
    EP: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    SK: tl.constexpr,
):
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


@triton.jit
def _finish(P, C, SIZE: tl.constexpr, SK: tl.constexpr, BLOCK: tl.constexpr):
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
    if 512 < m < 2048 and k >= 1024:
        _partial_persistent[
            (min(128, sk * triton.cdiv(n, bn) * (triton.cdiv(m, bm) + e + 2)),)
        ](
            a,
            b,
            parts,
            offs,
            m,
            n,
            k,
            e,
            triton.next_power_of_2(e + 2),
            bm,
            bn,
            bk,
            sk,
            num_warps=warps,
            num_stages=2,
        )
    else:
        _partial[(sk, triton.cdiv(n, bn), triton.cdiv(m, bm) + e + 2)](
            a,
            b,
            parts,
            offs,
            m,
            n,
            k,
            e,
            triton.next_power_of_2(e + 2),
            bm,
            bn,
            bk,
            sk,
            num_warps=warps,
            num_stages=2,
        )
    _finish[(triton.cdiv(m * n, 512),)](parts, out, m * n, sk, 512)
    return out


@triton.jit
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
    r = tl.program_id(0)
    ei = tl.arange(0, EP)
    lo = tl.load(OFFS + ei, ei < E, 0)
    hi = tl.load(OFFS + ei + 1, ei < E, 0)
    at = (ei < E) & (r >= lo) & (r < hi)
    valid = tl.sum(at.to(tl.int32), 0) != 0
    e = tl.sum(tl.where(at, ei, 0), 0)
    ns = tl.program_id(1) * BN + tl.arange(0, BN)
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


@triton.jit
def _gemv_multi(
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
):
    pid = tl.program_id(0)
    ei = tl.arange(0, EP)
    lo = tl.load(OFFS + tl.minimum(ei, E), ei < E, 0)
    hi = tl.load(OFFS + tl.minimum(ei + 1, E), ei < E, 0)
    lo = tl.where(ei == E, tl.load(OFFS + E), tl.where(ei == E + 1, 0, lo))
    hi = tl.where(ei == E, M, tl.where(ei == E + 1, tl.load(OFFS), hi))
    nb = tl.where(ei <= E + 1, tl.cdiv(tl.maximum(hi - lo, 0), BM), 0)
    end = tl.cumsum(nb, 0)
    at = (pid >= end - nb) & (pid < end) & (ei <= E + 1)
    valid = tl.sum(at.to(tl.int32), 0) != 0
    e = tl.sum(tl.where(at, ei, 0), 0)
    if valid:
        r = tl.sum(tl.where(at, lo + (pid - end + nb) * BM, 0), 0) + tl.arange(
            0, BM
        )
        stop = tl.sum(tl.where(at, hi, 0), 0)
        ns = tl.program_id(1) * BN + tl.arange(0, BN)
        ks = tl.arange(0, BK)
        acc = tl.zeros((BM, BN, BK), tl.float32)
        if e < E:
            for i in range(tl.cdiv(K, BK)):
                k = i * BK + ks
                av = tl.load(
                    A + r[:, None] * K + k[None, :],
                    (r < stop)[:, None] & (k < K)[None, :],
                    0,
                ).to(tl.float32)
                bv = tl.load(
                    B + (e * N + ns[:, None]) * K + k[None, :],
                    (ns < N)[:, None] & (k < K)[None, :],
                    0,
                ).to(tl.float32)
                acc += av[:, None, :] * bv[None, :, :]
        value = tl.sum(acc, 2)
        tl.store(
            C + r[:, None] * N + ns[None, :],
            value,
            (r < stop)[:, None] & (ns < N)[None, :],
        )


@triton.jit
def _bf16dot61(
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
    PACK: tl.constexpr,
):
    pid = tl.program_id(0)
    ei = tl.arange(0, EP)
    lo = tl.load(OFFS + tl.minimum(ei, E), ei < E, 0)
    hi = tl.load(OFFS + tl.minimum(ei + 1, E), ei < E, 0)
    lo = tl.where(ei == E, tl.load(OFFS + E), tl.where(ei == E + 1, 0, lo))
    hi = tl.where(ei == E, M, tl.where(ei == E + 1, tl.load(OFFS), hi))
    nb = tl.where(ei <= E + 1, tl.cdiv(tl.maximum(hi - lo, 0), BM), 0)
    end = tl.cumsum(nb, 0)
    at = (pid >= end - nb) & (pid < end) & (ei <= E + 1)
    valid = tl.sum(at.to(tl.int32), 0) != 0
    e = tl.sum(tl.where(at, ei, 0), 0)
    if valid:
        r = tl.sum(tl.where(at, lo + (pid - end + nb) * BM, 0), 0) + tl.arange(
            0, BM
        )
        stop = tl.sum(tl.where(at, hi, 0), 0)
        ns = tl.program_id(1) * BN + tl.arange(0, BN)
        ks = tl.arange(0, BK // 2)
        acc = tl.zeros((BM, BN, BK // 2), tl.float32)
        if e < E:
            for i in range(tl.cdiv(K, BK)):
                k = i * (BK // 2) + ks
                if PACK:
                    ap = A + r[:, None] * K + 2 * k[None, :]
                    bp = B + (e * N + ns[:, None]) * K + 2 * k[None, :]
                    am = (r < stop)[:, None] & (k < K // 2)[None, :]
                    bm = (ns < N)[:, None] & (k < K // 2)[None, :]
                    av0 = (
                        tl.load(ap, am, 0)
                        .to(tl.uint16, bitcast=True)
                        .to(tl.uint32)
                    )
                    av1 = (
                        tl.load(ap + 1, am, 0)
                        .to(tl.uint16, bitcast=True)
                        .to(tl.uint32)
                    )
                    bv0 = (
                        tl.load(bp, bm, 0)
                        .to(tl.uint16, bitcast=True)
                        .to(tl.uint32)
                    )
                    bv1 = (
                        tl.load(bp + 1, bm, 0)
                        .to(tl.uint16, bitcast=True)
                        .to(tl.uint32)
                    )
                    av = av0 | (av1 << 16)
                    bv = bv0 | (bv1 << 16)
                else:
                    av = tl.load(
                        A.to(tl.pointer_type(tl.int32))
                        + r[:, None] * (K // 2)
                        + k[None, :],
                        (r < stop)[:, None] & (k < K // 2)[None, :],
                        0,
                    )
                    bv = tl.load(
                        B.to(tl.pointer_type(tl.int32))
                        + (e * N + ns[:, None]) * (K // 2)
                        + k[None, :],
                        (ns < N)[:, None] & (k < K // 2)[None, :],
                        0,
                    )
                acc = tl.inline_asm_elementwise(
                    "v_dot2_f32_bf16 $0, $1, $2, $3",
                    constraints="=v,v,v,0",
                    args=[av[:, None, :], bv[None, :, :], acc],
                    dtype=tl.float32,
                    is_pure=True,
                    pack=1,
                )
        value = tl.sum(acc, 2)
        tl.store(
            C + r[:, None] * N + ns[None, :],
            value,
            (r < stop)[:, None] & (ns < N)[None, :],
        )


@triton.jit
def _partial_persistent(
    A,
    B,
    P,
    OFFS,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    E: tl.constexpr,
    EP: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    SK: tl.constexpr,
):
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


__all__ = ["grouped_gemm"]
