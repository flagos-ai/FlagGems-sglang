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
def _fill(Y, PAD: tl.constexpr, B: tl.constexpr):
    i = tl.program_id(0) * B + tl.arange(0, B)
    tl.store(Y + i, tl.full((B,), PAD, tl.int32))


@triton.jit
def _rank(
    X,
    CNT,
    DST,
    N: tl.constexpr,
    K: tl.constexpr,
    S0: tl.constexpr,
    S1: tl.constexpr,
    NT: tl.constexpr,
    B: tl.constexpr,
    EXACT: tl.constexpr,
    R: tl.constexpr,
):
    e = tl.program_id(0)
    r = tl.arange(0, B)
    total = tl.full((), 0, tl.int32)
    for t in range(NT):
        i = t * B + r
        if EXACT:
            v = tl.load(X + i // K * S0 + i % K * S1)
            m = v == e
        else:
            safe = tl.minimum(i, N - 1)
            v = tl.load(X + safe // K * S0 + safe % K * S1)
            m = (i < N) & (v == e)
        ranks = tl.cumsum(m.to(tl.int32), 0)
        tl.store(DST + e * NT * B + i, total + ranks - 1)
        total += tl.sum(m.to(tl.int32), 0)
    tl.store(CNT + e, total)


@triton.jit
def _offs(
    CNT, OFF, NPOST, R: tl.constexpr, BS: tl.constexpr, EP: tl.constexpr
):
    x = tl.arange(0, EP)
    c = tl.load(CNT + tl.minimum(x, R - 1))
    c = tl.where(x < R, c, 0)
    a = tl.cdiv(c, BS) * BS
    inc = tl.cumsum(a, 0)
    tl.store(OFF + x, inc - a)
    tl.store(NPOST, tl.sum(a, 0))


@triton.jit
def _eids(
    OFF,
    SRC,
    EID,
    NPOST,
    R: tl.constexpr,
    BS: tl.constexpr,
    EP: tl.constexpr,
    BIG: tl.constexpr,
):
    b = tl.program_id(0)
    x = tl.arange(0, EP)
    o = tl.load(OFF + tl.minimum(x, R - 1))
    o = tl.where(x < R, o, BIG)
    p = b * BS
    fresh = tl.sum((o <= p).to(tl.int32), 0) - 1
    keep = tl.load(SRC + b)
    tl.store(EID + b, tl.where(p < tl.load(NPOST), fresh, keep))


@triton.jit
def _place(
    X,
    DST,
    OFF,
    Y,
    N: tl.constexpr,
    K: tl.constexpr,
    S0: tl.constexpr,
    S1: tl.constexpr,
    R: tl.constexpr,
    DUMP: tl.constexpr,
    B: tl.constexpr,
    EXACT: tl.constexpr,
    P: tl.constexpr,
):
    pid = tl.program_id(0)
    i = pid * B + tl.arange(0, B)
    if EXACT:
        v = tl.load(X + i // K * S0 + i % K * S1)
        ok = (v >= 0) & (v < R)
        rk = tl.load(DST + tl.minimum(tl.maximum(v, 0), R - 1) * P + i)
    else:
        safe = tl.minimum(i, N - 1)
        v = tl.load(X + safe // K * S0 + safe % K * S1)
        ok = (i < N) & (v >= 0) & (v < R)
        rk = tl.load(DST + tl.minimum(tl.maximum(v, 0), R - 1) * P + safe)
    base = tl.load(OFF + tl.minimum(tl.maximum(v, 0), R - 1))
    tl.store(Y + tl.where(ok, base + rk, DUMP + i), i)


def _block_for(n, cap):
    b = cap
    while b > 1 and n % b:
        b //= 2
    return b


def moe_align_block_size(
    topk_ids,
    num_experts,
    block_size,
    sorted_token_ids,
    expert_ids,
    num_tokens_post_pad,
    cumsum_buffer,
    pad_sorted_token_ids,
):
    device = topk_ids.device
    n = topk_ids.numel()
    k = topk_ids.shape[1]
    s0 = topk_ids.stride(0)
    s1 = topk_ids.stride(1)
    r = num_experts - 1
    ln = sorted_token_ids.numel()
    nblocks = expert_ids.numel()
    cap = 16384 if n > 1024 else 2048
    b = _block_for(n, cap)
    if b < 64:
        b = triton.next_power_of_2(n) if n < cap else cap
    b = max(b, 256)
    exact = 1 if n >= b and n % b == 0 else 0
    nt = (n + b - 1) // b
    bf = _block_for(ln, 2048)
    if bf < 64:
        bf = triton.next_power_of_2(ln) if ln < 2048 else 2048
    lp = (ln + bf - 1) // bf * bf
    ep = triton.next_power_of_2(r)
    out = torch.empty(
        (lp + nt * b,), dtype=sorted_token_ids.dtype, device=device
    )
    eids = torch.empty_like(expert_ids)
    npost = torch.empty_like(num_tokens_post_pad)
    dst = torch.empty((r * nt * b,), dtype=torch.int32, device=device)
    cnt = torch.empty((r,), dtype=torch.int32, device=device)
    off = torch.empty((max(ep, 256),), dtype=torch.int32, device=device)
    _fill[lp // bf,](out, n, bf)
    _rank[r,](topk_ids, cnt, dst, n, k, s0, s1, nt, b, exact, r)
    _offs[1,](cnt, off, npost, r, block_size, max(ep, 256))
    _eids[nblocks,](off, expert_ids, eids, npost, r, block_size, ep, 1 << 30)
    _place[nt,](topk_ids, dst, off, out, n, k, s0, s1, r, lp, b, exact, nt * b)
    return (out[:ln], eids, npost)


__all__ = ["moe_align_block_size"]
