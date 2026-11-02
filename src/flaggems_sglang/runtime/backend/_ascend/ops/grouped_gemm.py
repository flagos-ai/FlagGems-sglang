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


@triton.jit(do_not_specialize=["A", "B", "C", "OFFS", "SCHED"])
def _gemm(A, B, C, OFFS, SCHED, META: tl.constexpr):
    A = A.to(tl.pointer_type(tl.bfloat16))
    B = B.to(tl.pointer_type(tl.bfloat16))
    C = C.to(tl.pointer_type(tl.bfloat16))
    OFFS = OFFS.to(tl.pointer_type(tl.int32))
    SCHED = SCHED.to(tl.pointer_type(tl.int32))
    M: tl.constexpr = META[0]
    E: tl.constexpr = META[1]
    NS: tl.constexpr = META[2]
    N: tl.constexpr = META[3]
    K: tl.constexpr = META[4]
    BM: tl.constexpr = 128
    BN: tl.constexpr = 256
    BK: tl.constexpr = 256
    EDGE: tl.constexpr = N % BN != 0 or K % BK != 0
    pid = tl.program_id(0)
    n = tl.program_id(1) * BN + tl.arange(0, BN)
    if pid < NS:
        data = tl.load(SCHED + pid)
        if data >= 0:
            e = data & 65535
            lo = tl.load(OFFS + e)
            hi = tl.load(OFFS + e + 1)
            start = lo + (data >> 16) * 128
            if hi - start <= 32:
                _rows(A, B, C, e, start, hi, n, N, K, 32, BN, BK, EDGE)
            elif hi - start <= 64:
                _rows(A, B, C, e, start, hi, n, N, K, 64, BN, BK, EDGE)
            else:
                _rows(A, B, C, e, start, hi, n, N, K, 128, BN, BK, EDGE)
    else:
        head = tl.load(OFFS)
        tail = tl.load(OFFS + E)
        lo = tl.where(pid == NS, tail, 0)
        hi = tl.where(pid == NS, M, head)
        for start in range(lo, hi, BM):
            r = start + tl.arange(0, BM)
            tl.store(
                C + r[:, None] * N + n[None, :],
                0.0,
                (r < hi)[:, None] & (n < N)[None, :],
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
    out = torch.empty((m, n), device=a.device, dtype=a.dtype)
    ns = expert_block_schedule.numel()
    _gemm.run(
        a.data_ptr(),
        b.data_ptr(),
        out.data_ptr(),
        expert_token_offs.data_ptr(),
        expert_block_schedule.data_ptr(),
        (m, num_experts, ns, n, k),
        num_warps=4,
        num_stages=3,
        grid=(ns + 2, (n + 256 - 1) // 256),
        warmup=False,
    )
    return out


@triton.jit
def _rows(
    A,
    B,
    C,
    e,
    start,
    hi,
    n,
    N: tl.constexpr,
    K: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    EDGE: tl.constexpr,
):
    r = start + tl.arange(0, BM)
    ok = r < hi
    k = tl.arange(0, BK)
    ap = A + r[:, None] * K + k[None, :]
    bp = B + (e * N + n[None, :]) * K + k[:, None]
    acc = tl.zeros((BM, BN), tl.float32)
    if EDGE:
        cols = n < N
        for i in range(tl.cdiv(K, BK)):
            kk = i * BK + k < K
            acc = tl.dot(
                tl.load(ap, ok[:, None] & kk[None, :], 0),
                tl.load(bp, cols[None, :] & kk[:, None], 0),
                acc,
            )
            ap += BK
            bp += BK
        tl.store(
            C + r[:, None] * N + n[None, :], acc, ok[:, None] & cols[None, :]
        )
    else:
        for i in range(K // BK):
            acc = tl.dot(tl.load(ap, ok[:, None], 0), tl.load(bp), acc)
            ap += BK
            bp += BK
        tl.store(C + r[:, None] * N + n[None, :], acc, ok[:, None])


__all__ = ["grouped_gemm"]
