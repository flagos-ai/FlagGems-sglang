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
def _pad(
    X, Y, R: tl.constexpr, K: tl.constexpr, RP: tl.constexpr, KP: tl.constexpr
):
    for block in range(
        tl.program_id(0), tl.cdiv(RP * KP, 2048), tl.num_programs(0)
    ):
        i = block * 2048 + tl.arange(0, 2048)
        r = i // KP
        c = i % KP
        v = tl.load(
            X + tl.minimum(r, R - 1) * K + tl.minimum(c, K - 1),
            i < RP * KP,
            other=0,
        )
        tl.store(Y + i, tl.where((r < R) & (c < K), v, 0), i < RP * KP)


@triton.jit
def _one(
    X,
    W,
    O,
    OFFS,
    SCHED,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    NP: tl.constexpr,
    E: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    SPLIT: tl.constexpr,
    NS: tl.constexpr,
):
    NC: tl.constexpr = (N + BN - 1) // BN
    for task in range(
        tl.program_id(0), (NS * SPLIT + 2) * NC, tl.num_programs(0)
    ):
        pid = task // NC
        col = task % NC
        if pid < NS * SPLIT:
            raw = tl.load(SCHED + pid // SPLIT)
            if raw >= 0:
                data = tl.maximum(raw, 0)
                e = data & 65535
                lo = tl.load(OFFS + e)
                hi = tl.load(OFFS + e + 1)
                base = lo + (data >> 16) * BM * SPLIT + pid % SPLIT * BM
                mi = tl.minimum(base, M - BM) + tl.arange(0, BM)
                ni = tl.minimum(col * BN, NP - BN) + tl.arange(0, BN)
                ki = tl.arange(0, BK)
                acc = tl.zeros((BM, BN), tl.float32)
                xp = X + mi[:, None] * K + ki[None, :]
                wp = W + (e * NP + ni[None, :]) * K + ki[:, None]
                for start in range(K // BK):
                    acc = tl.dot(
                        tl.load(xp), tl.load(wp), acc, input_precision="ieee"
                    )
                    xp += BK
                    wp += BK
                tl.store(
                    O + mi[:, None] * N + ni[None, :],
                    acc,
                    (raw >= 0)
                    & (mi >= base)[:, None]
                    & (mi >= lo)[:, None]
                    & (mi < hi)[:, None]
                    & (ni >= col * BN)[None, :]
                    & (ni < N)[None, :],
                )
        else:
            lo = tl.where(pid == NS * SPLIT, tl.load(OFFS + E), 0)
            hi = tl.where(pid == NS * SPLIT, M, tl.load(OFFS))
            ni = col * BN + tl.arange(0, BN)
            for start in range(lo, hi, BM):
                mi = start + tl.arange(0, BM)
                tl.store(
                    O + mi[:, None] * N + ni[None, :],
                    0.0,
                    (mi < hi)[:, None] & (ni < N)[None, :],
                )


@triton.jit
def _gemm(
    X,
    W,
    O,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    NP: tl.constexpr,
    E: tl.constexpr,
    LO: tl.constexpr,
    HI: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    mi = tl.minimum(LO + tl.program_id(0) * BM, M - BM) + tl.arange(0, BM)
    ni = tl.minimum(tl.program_id(1) * BN, NP - BN) + tl.arange(0, BN)
    ki = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), tl.float32)
    xp = X + mi[:, None] * K + ki[None, :]
    wp = W + (E * NP + ni[None, :]) * K + ki[:, None]
    for start in range(K // BK):
        acc = tl.dot(tl.load(xp), tl.load(wp), acc, input_precision="ieee")
        xp += BK
        wp += BK
    tl.store(
        O + mi[:, None] * N + ni[None, :],
        acc,
        (mi >= LO)[:, None] & (mi < HI)[:, None] & (ni < N)[None, :],
    )


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
    if m == 0:
        return torch.empty((m, n), device=a.device, dtype=a.dtype)
    if (
        64 <= m <= 512
        and m // num_experts <= 32
        and (n % 256 == 0)
        and (k % 256 == 0)
        and (k <= 512 or num_experts < 16)
    ):
        out = torch.empty((m, n), device=a.device, dtype=a.dtype)
        if k <= 512:
            _direct[min(12, (num_experts + 2) * (n // 256)),](
                a,
                b,
                out,
                expert_token_offs,
                m,
                n,
                k,
                num_experts,
                64,
                256,
                256,
                num_warps=4,
                num_stages=1,
            )
        else:
            _pair[min(12, (triton.cdiv(num_experts, 2) + 2) * (n // 256)),](
                a,
                b,
                out,
                expert_token_offs,
                m,
                n,
                k,
                num_experts,
                32,
                256,
                256,
                num_warps=4,
                num_stages=1,
            )
        return out
    if (
        64 <= m <= 1024
        and k >= 1024
        and (num_experts >= 16)
        and (m // num_experts <= 32)
        and (n % 256 == 0)
        and (k % 256 == 0)
    ):
        out = torch.empty((m, n), device=a.device, dtype=a.dtype)
        if m // num_experts <= 8:
            _tiny[min(12, ((num_experts + 1) // 2 + 2) * (n // 256)),](
                a,
                b,
                out,
                expert_token_offs,
                m,
                n,
                k,
                num_experts,
                16,
                256,
                256,
                num_warps=4,
                num_stages=1,
            )
        else:
            _pair[min(12, (triton.cdiv(num_experts, 2) + 2) * (n // 256)),](
                a,
                b,
                out,
                expert_token_offs,
                m,
                n,
                k,
                num_experts,
                64,
                256,
                256,
                num_warps=4,
                num_stages=1,
            )
        return out
    bm, bn = (128, 256)
    bk = (
        256
        if k % 256 == 0
        else 64 if k % 64 == 0 else 32 if k % 32 == 0 else 0
    )
    if bk and m >= bm and (n >= bn) and (n % bn == 0):
        out = torch.empty((m, n), device=a.device, dtype=a.dtype)
        ns = expert_block_schedule.numel()
        _one[min(12, (ns * (128 // bm) + 2) * triton.cdiv(n, bn)),](
            a,
            b,
            out,
            expert_token_offs,
            expert_block_schedule,
            m,
            n,
            k,
            n,
            num_experts,
            bm,
            bn,
            bk,
            128 // bm,
            ns,
            num_warps=4,
            num_stages=1,
        )
        return out
    out = torch.zeros((m, n), device=a.device, dtype=a.dtype)
    bk = 256 if k >= 256 else 32
    kp = triton.cdiv(k, bk) * bk
    mp = triton.cdiv(m, bm) * bm + bm
    np_ = triton.cdiv(n, bn) * bn
    src = torch.empty((mp, kp), dtype=a.dtype, device=a.device)
    _pad[12,](a, src, m, k, mp, kp)
    wsrc = torch.empty((num_experts, np_, kp), dtype=b.dtype, device=b.device)
    for e in range(num_experts):
        _pad[12,](b[e], wsrc[e], n, k, np_, kp)
    offs = expert_token_offs.tolist()
    for e in range(num_experts):
        lo, hi = (offs[e], offs[e + 1])
        if hi > lo:
            _gemm[triton.cdiv(hi - lo, bm), triton.cdiv(n, bn)](
                src,
                wsrc,
                out,
                mp,
                n,
                kp,
                np_,
                e,
                lo,
                hi,
                bm,
                bn,
                bk,
                num_warps=4,
                num_stages=1,
            )
    return out


@triton.jit
def _pair(
    A,
    B,
    C,
    OFFS,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    E: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    NC: tl.constexpr = N // BN
    for task in range(
        tl.program_id(0), (tl.cdiv(E, 2) + 2) * NC, tl.num_programs(0)
    ):
        pair = task // NC
        col = task % NC
        nn = col * BN + tl.arange(0, BN)
        if pair < tl.cdiv(E, 2):
            e0 = pair * 2
            e1 = tl.minimum(e0 + 1, E - 1)
            lo0 = tl.load(OFFS + e0)
            hi0 = tl.load(OFFS + e0 + 1)
            lo1 = tl.load(OFFS + tl.minimum(e0 + 1, E))
            hi1 = tl.load(OFFS + tl.minimum(e0 + 2, E))
            for block in range(tl.cdiv(tl.maximum(hi0 - lo0, hi1 - lo1), BM)):
                base0 = lo0 + block * BM
                base1 = lo1 + block * BM
                r0 = tl.minimum(base0, M - BM) + tl.arange(0, BM)
                r1 = tl.minimum(base1, M - BM) + tl.arange(0, BM)
                kk = tl.arange(0, BK)
                acc0 = tl.zeros((BM, BN), tl.float32)
                acc1 = tl.zeros((BM, BN), tl.float32)
                for pos in range(K // BK):
                    k = pos * BK + kk
                    a0 = tl.load(A + r0[:, None] * K + k[None, :])
                    b0 = tl.load(B + (e0 * N + nn[None, :]) * K + k[:, None])
                    a1 = tl.load(A + r1[:, None] * K + k[None, :])
                    b1 = tl.load(B + (e1 * N + nn[None, :]) * K + k[:, None])
                    acc0 = tl.dot(a0, b0, acc0, input_precision="ieee")
                    acc1 = tl.dot(a1, b1, acc1, input_precision="ieee")
                tl.store(
                    C + r0[:, None] * N + nn[None, :],
                    acc0,
                    ((r0 >= base0) & (r0 < hi0))[:, None],
                )
                tl.store(
                    C + r1[:, None] * N + nn[None, :],
                    acc1,
                    ((r1 >= base1) & (r1 < hi1))[:, None],
                )
        else:
            gap_lo = tl.where(pair == tl.cdiv(E, 2), tl.load(OFFS + E), 0)
            gap_hi = tl.where(pair == tl.cdiv(E, 2), M, tl.load(OFFS))
            for start in range(gap_lo, gap_hi, BM):
                rows = start + tl.arange(0, BM)
                tl.store(
                    C + rows[:, None] * N + nn[None, :],
                    0.0,
                    (rows < gap_hi)[:, None],
                )


@triton.jit
def _tiny(
    A,
    B,
    C,
    OFFS,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    E: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    NC: tl.constexpr = N // BN
    GROUPS: tl.constexpr = (E + 1) // 2
    for task in range(tl.program_id(0), (GROUPS + 2) * NC, tl.num_programs(0)):
        pair = task // NC
        col = task % NC
        nn = col * BN + tl.arange(0, BN)
        if pair < GROUPS:
            e0 = tl.minimum(pair * 2 + 0, E - 1)
            lo0 = tl.load(OFFS + tl.minimum(pair * 2 + 0, E))
            hi0 = tl.load(OFFS + tl.minimum(pair * 2 + 1, E))
            e1 = tl.minimum(pair * 2 + 1, E - 1)
            lo1 = tl.load(OFFS + tl.minimum(pair * 2 + 1, E))
            hi1 = tl.load(OFFS + tl.minimum(pair * 2 + 2, E))
            for block in range(tl.cdiv(tl.maximum(hi0 - lo0, hi1 - lo1), BM)):
                base0 = lo0 + block * BM
                r0 = tl.minimum(base0, M - BM) + tl.arange(0, BM)
                acc0 = tl.zeros((BM, BN), tl.float32)
                base1 = lo1 + block * BM
                r1 = tl.minimum(base1, M - BM) + tl.arange(0, BM)
                acc1 = tl.zeros((BM, BN), tl.float32)
                kk = tl.arange(0, BK)
                for pos in range(K // BK):
                    k = pos * BK + kk
                    a0 = tl.load(A + r0[:, None] * K + k[None, :])
                    b0 = tl.load(B + (e0 * N + nn[None, :]) * K + k[:, None])
                    a1 = tl.load(A + r1[:, None] * K + k[None, :])
                    b1 = tl.load(B + (e1 * N + nn[None, :]) * K + k[:, None])
                    acc0 = tl.dot(a0, b0, acc0, input_precision="ieee")
                    acc1 = tl.dot(a1, b1, acc1, input_precision="ieee")
                tl.store(
                    C + r0[:, None] * N + nn[None, :],
                    acc0,
                    ((r0 >= base0) & (r0 < hi0))[:, None],
                )
                tl.store(
                    C + r1[:, None] * N + nn[None, :],
                    acc1,
                    ((r1 >= base1) & (r1 < hi1))[:, None],
                )
        else:
            gap_lo = tl.where(pair == GROUPS, tl.load(OFFS + E), 0)
            gap_hi = tl.where(pair == GROUPS, M, tl.load(OFFS))
            for start in range(gap_lo, gap_hi, BM):
                rows = start + tl.arange(0, BM)
                tl.store(
                    C + rows[:, None] * N + nn[None, :],
                    0.0,
                    (rows < gap_hi)[:, None],
                )


@triton.jit
def _direct(
    A,
    B,
    C,
    OFFS,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    E: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    NC: tl.constexpr = N // BN
    GROUPS: tl.constexpr = (E + 0) // 1
    for task in range(tl.program_id(0), (GROUPS + 2) * NC, tl.num_programs(0)):
        pair = task // NC
        col = task % NC
        nn = col * BN + tl.arange(0, BN)
        if pair < GROUPS:
            e0 = tl.minimum(pair * 1 + 0, E - 1)
            lo0 = tl.load(OFFS + tl.minimum(pair * 1 + 0, E))
            hi0 = tl.load(OFFS + tl.minimum(pair * 1 + 1, E))
            for block in range(tl.cdiv(hi0 - lo0, BM)):
                base0 = lo0 + block * BM
                r0 = tl.minimum(base0, M - BM) + tl.arange(0, BM)
                acc0 = tl.zeros((BM, BN), tl.float32)
                kk = tl.arange(0, BK)
                for pos in range(K // BK):
                    k = pos * BK + kk
                    a0 = tl.load(A + r0[:, None] * K + k[None, :])
                    b0 = tl.load(B + (e0 * N + nn[None, :]) * K + k[:, None])
                    acc0 = tl.dot(a0, b0, acc0, input_precision="ieee")
                tl.store(
                    C + r0[:, None] * N + nn[None, :],
                    acc0,
                    ((r0 >= base0) & (r0 < hi0))[:, None],
                )
        else:
            gap_lo = tl.where(pair == GROUPS, tl.load(OFFS + E), 0)
            gap_hi = tl.where(pair == GROUPS, M, tl.load(OFFS))
            for start in range(gap_lo, gap_hi, BM):
                rows = start + tl.arange(0, BM)
                tl.store(
                    C + rows[:, None] * N + nn[None, :],
                    0.0,
                    (rows < gap_hi)[:, None],
                )


__all__ = ["grouped_gemm"]
