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

"""grouped_gemm for RDNA3 (gfx1100).

One program per (row block of an expert, column block), as in the generic
kernel, without its split-K paths: for most shapes from 512 tokens the
generic op splits K up to four ways and writes fp32 partial sums that a
second kernel reads back and adds. The column blocks of a row block are
launched next to each other so they share its rows of A in the cache, and
the tile follows the shape (see _config). Each K block is its own dot
summed in fp32, and small-K shapes store through a split / join round trip
for 16-byte writes. Below 512 tokens (and at 512 with few rows per expert
and N up to 512), and for non-contiguous or non-16-bit inputs, the generic
op runs.
"""

import torch
import triton
import triton.language as tl

from flaggems_sglang.ops.grouped_gemm import grouped_gemm as _generic


@triton.jit
def _grouped_kernel(
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
    SPLIT_STORE: tl.constexpr,
):
    # Tasks: the row blocks of each expert in order, then the rows after the
    # last expert, then the rows before the first; those two are zeroed.
    pid = tl.program_id(1)
    pid_n = tl.program_id(0)
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
        n = pid_n * BN + tl.arange(0, BN)
        cols = n < N
        acc = tl.zeros((BM, BN), tl.float32)
        if seg < E:
            k = tl.arange(0, BK)
            ap = A + r[:, None].to(tl.int64) * K + k[None, :]
            bp = B + (seg.to(tl.int64) * N + n[None, :]) * K + k[:, None]
            ones = tl.full((BM, BN), 1.0, tl.float32)
            for i in range(tl.cdiv(K, BK)):
                kk = i * BK + k < K
                av = tl.load(ap, rows[:, None] & kk[None, :], 0)
                bv = tl.load(bp, cols[None, :] & kk[:, None], 0)
                # A dot accumulating into acc truncates the fp32 addition.
                acc = tl.fma(tl.dot(av, bv), ones, acc)
                ap += BK
                bp += BK
        if SPLIT_STORE:
            # Without this round trip the store keeps the WMMA layout and
            # writes two bytes a lane.
            low, high = tl.split(tl.reshape(acc, (BM, BN // 2, 2)))
            acc = tl.reshape(tl.join(low, high), (BM, BN))
        tl.store(
            C + r[:, None].to(tl.int64) * N + n[None, :],
            acc.to(C.dtype.element_ty),
            rows[:, None] & cols[None, :],
        )


def _config(m, n, k, e):
    """BM, BN, BK, num_warps, num_stages.

    None keeps the generic op, whose GEMV paths win below 512 tokens, and
    at 512 tokens with up to 8 rows per expert and N up to 512. Fitted on
    gfx1100 over 512-8192 tokens, 8-128 experts and (N, K) of
    (512, 2048), (1536, 2048) and (2048, 768).
    """
    per = m / e
    if m < 512 or (m == 512 and per <= 8 and n <= 512):
        return None
    if per <= 32 and n <= 512 and e <= 32:
        return 16, 32, 256, 2, 2
    if per <= 16 or (per <= 32 and n <= 512):
        return 16, 32, 128, 2, 3
    if per <= 128 or m <= 2048:
        return 32, 64, 256, 8, 1
    if k >= 1536:
        return 64, 32, 64, 2, 2
    return 64, 128, 64, 4, 2


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
    cfg = _config(m, n, k, num_experts) if num_experts else None
    if (
        cfg is None
        or a.dtype not in (torch.float16, torch.bfloat16)
        or b.dtype != a.dtype
        or not (a.is_contiguous() and b.is_contiguous())
    ):
        return _generic(
            a,
            b,
            num_experts,
            num_tokens_per_expert,
            expert_token_offs,
            expert_block_offs,
            expert_block_schedule,
        )
    bm, bn, bk, warps, stages = cfg
    c = torch.empty((m, n), device=a.device, dtype=a.dtype)
    tasks = triton.cdiv(m, bm) + num_experts + 2
    _grouped_kernel[(triton.cdiv(n, bn), tasks)](
        a,
        b,
        c,
        expert_token_offs,
        m,
        n,
        k,
        num_experts,
        triton.next_power_of_2(num_experts),
        bm,
        bn,
        bk,
        k <= 1024 and bn % 2 == 0,
        num_warps=warps,
        num_stages=stages,
    )
    return c


__all__ = ["grouped_gemm"]
