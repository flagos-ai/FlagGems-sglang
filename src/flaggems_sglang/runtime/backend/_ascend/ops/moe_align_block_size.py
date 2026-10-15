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
def _tiny_pairs(
    X,
    OUT,
    EI,
    EO,
    NP,
    N: tl.constexpr,
    K: tl.constexpr,
    S0: tl.constexpr,
    S1: tl.constexpr,
    R: tl.constexpr,
    BS: tl.constexpr,
    NS: tl.constexpr,
    EN: tl.constexpr,
    BN: tl.constexpr,
    BO: tl.constexpr,
    NN: tl.constexpr,
    PN: tl.constexpr,
):
    pid = tl.program_id(0)
    i = tl.arange(0, BN)
    v = tl.load(X + i // K * S0 + i % K * S1, i < N, other=R)
    valid = (i < N) & (v >= 0) & (v < R)
    equal = (v[:, None] == v[None, :]) & valid[None, :]
    count = tl.sum(equal.to(tl.int32), 1)
    rank = tl.sum((equal & (i[None, :] < i[:, None])).to(tl.int32), 1)
    first = valid & (rank == 0)
    length = tl.cdiv(count, BS) * BS
    padded = tl.where(first, length, 0)
    start = tl.sum(tl.where(v[None, :] < v[:, None], padded[None, :], 0), 1)
    target = tl.where(valid, start + rank, -1)
    j = pid * BO + tl.arange(0, BO)
    pair = tl.max(tl.where(j[:, None] == target[None, :], i[None, :], -1), 1)
    value = tl.where(pair >= 0, pair, N)
    tl.store(OUT + j, value, j < NS)
    if pid * BO < EN:
        position = j * BS
        match = (
            (position[:, None] >= start[None, :])
            & (position[:, None] < start[None, :] + length[None, :])
            & valid[None, :]
        )
        expert = tl.max(tl.where(match, v[None, :], -1), 1)
        old = tl.load(EI + j, j < EN, other=0)
        tl.store(EO + j, tl.where(expert >= 0, expert, old), j < EN)
    if pid == 0:
        q = tl.arange(0, PN)
        tl.store(NP + q, tl.sum(padded, 0), q < NN)


@triton.jit
def _count_init(
    X,
    C,
    OUT,
    EI,
    EO,
    N: tl.constexpr,
    K: tl.constexpr,
    S0: tl.constexpr,
    S1: tl.constexpr,
    NE: tl.constexpr,
    NS: tl.constexpr,
    EN: tl.constexpr,
    B: tl.constexpr,
):
    expert = tl.program_id(0)
    r = tl.arange(0, B)
    total = tl.full((), 0, tl.int32)
    for start in range(0, N, B):
        i = start + r
        safe = tl.minimum(i, N - 1)
        v = tl.load(X + safe // K * S0 + safe % K * S1)
        total += tl.sum(((i < N) & (v == expert)).to(tl.int32), 0)
    tl.store(C + expert, total)
    for start in range(expert * B, tl.maximum(NS, EN), NE * B):
        i = start + r
        tl.store(OUT + i, N, i < NS)
        old = tl.load(EI + i, i < EN, other=0)
        tl.store(EO + i, old, i < EN)


@triton.jit
def _prefix_place(
    X,
    C,
    OUT,
    EO,
    NP,
    N: tl.constexpr,
    K: tl.constexpr,
    S0: tl.constexpr,
    S1: tl.constexpr,
    NE: tl.constexpr,
    BS: tl.constexpr,
    NS: tl.constexpr,
    EN: tl.constexpr,
    EP: tl.constexpr,
    B: tl.constexpr,
    NN: tl.constexpr,
    PN: tl.constexpr,
):
    expert = tl.program_id(0)
    ex = tl.arange(0, EP)
    count = tl.load(C + tl.minimum(ex, NE - 1))
    aligned = tl.where(ex < NE - 1, tl.cdiv(count, BS) * BS, 0)
    begin = tl.sum(tl.where(ex < expert, aligned, 0), 0)
    length = tl.sum(tl.where(ex == expert, aligned, 0), 0)
    r = tl.arange(0, B)
    if expert < NE - 1:
        for start in range(begin // BS, (begin + length) // BS, B):
            j = start + r
            tl.store(EO + j, expert, (j < (begin + length) // BS) & (j < EN))
        consumed = tl.full((), 0, tl.int32)
        for start in range(0, N, B):
            i = start + r
            safe = tl.minimum(i, N - 1)
            v = tl.load(X + safe // K * S0 + safe % K * S1)
            hit = (i < N) & (v == expert)
            rank = tl.cumsum(hit.to(tl.int32), 0)
            pos = begin + consumed + rank - 1
            tl.store(OUT + pos, i, hit & (pos < NS))
            consumed += tl.sum(hit.to(tl.int32), 0)
    if expert == 0:
        q = tl.arange(0, PN)
        tl.store(NP + q, tl.sum(aligned, 0), q < NN)


@triton.jit
def _init(
    OUT,
    EI,
    EO,
    H,
    N: tl.constexpr,
    NS: tl.constexpr,
    EN: tl.constexpr,
    HS: tl.constexpr,
    B: tl.constexpr,
):
    pid = tl.program_id(0)
    i = pid * B + tl.arange(0, B)
    tl.store(OUT + i, N, i < NS)
    old = tl.load(EI + i, i < EN, other=0)
    tl.store(EO + i, old, i < EN)
    tl.store(H + i, 0, i < HS)


@triton.jit
def _pair_count(
    X,
    H,
    RK,
    N: tl.constexpr,
    K: tl.constexpr,
    S0: tl.constexpr,
    S1: tl.constexpr,
    R: tl.constexpr,
    EP: tl.constexpr,
    PB: tl.constexpr,
    CONTIG: tl.constexpr,
):
    c = tl.program_id(0)
    i = c * PB + tl.arange(0, PB)
    if CONTIG:
        v = tl.load(X + i, i < N, other=R)
    else:
        v = tl.load(X + i // K * S0 + i % K * S1, i < N, other=R)
    valid = (i < N) & (v >= 0) & (v < R)
    equal = (v[:, None] == v[None, :]) & valid[None, :]
    rank = tl.sum(
        (equal & (tl.arange(0, PB)[None, :] < tl.arange(0, PB)[:, None])).to(
            tl.int32
        ),
        1,
    )
    first = valid & (rank == 0)
    count = tl.sum(equal.to(tl.int32), 1)
    tl.store(H + c * EP + v, count, first)
    tl.store(RK + i, rank, i < N)


@triton.jit
def _coltot(H, P, C, CH: tl.constexpr, EP: tl.constexpr, CB: tl.constexpr):
    e = tl.program_id(0)
    running = tl.zeros((), tl.int32)
    for c0 in range(0, CH, CB):
        c = c0 + tl.arange(0, CB)
        h = tl.load(H + c * EP + e, c < CH, other=0)
        p = tl.cumsum(h, 0) - h + running
        tl.store(P + c * EP + e, p, c < CH)
        running += tl.sum(h, 0)
    tl.store(C + e, running)


@triton.jit
def _begin(
    C,
    OFF,
    EI,
    EO,
    NP,
    R: tl.constexpr,
    BS: tl.constexpr,
    EP: tl.constexpr,
    EN: tl.constexpr,
    NN: tl.constexpr,
    PN: tl.constexpr,
    BC: tl.constexpr,
):
    e = tl.arange(0, EP)
    cnt = tl.load(C + e, e < R, other=0)
    aligned = tl.where(e < R, tl.cdiv(cnt, BS) * BS, 0)
    end = tl.cumsum(aligned, 0)
    begin = end - aligned
    tl.store(OFF + e, begin, e <= R)
    total = tl.sum(aligned, 0)
    q = tl.arange(0, PN)
    tl.store(NP + q, total, q < NN)
    for bstart in range(0, EN, BC):
        b = bstart + tl.arange(0, BC)
        pos = b * BS
        expert = tl.sum((begin[None, :] <= pos[:, None]).to(tl.int32), 1) - 1
        live = (b < EN) & (pos < total)
        tl.store(EO + b, expert, live)


@triton.jit
def _pair_place(
    X,
    RK,
    OFF,
    P,
    OUT,
    N: tl.constexpr,
    K: tl.constexpr,
    S0: tl.constexpr,
    S1: tl.constexpr,
    R: tl.constexpr,
    NS: tl.constexpr,
    EP: tl.constexpr,
    PB: tl.constexpr,
    CONTIG: tl.constexpr,
):
    c = tl.program_id(0)
    i = c * PB + tl.arange(0, PB)
    if CONTIG:
        v = tl.load(X + i, i < N, other=R)
    else:
        v = tl.load(X + i // K * S0 + i % K * S1, i < N, other=R)
    valid = (i < N) & (v >= 0) & (v < R)
    rank = tl.load(RK + i, i < N, other=0)
    vs = tl.minimum(tl.maximum(v, 0), R - 1)
    base = tl.load(OFF + vs, valid, other=0) + tl.load(
        P + c * EP + vs, valid, other=0
    )
    target = base + rank
    tl.store(OUT + target, i, valid & (target < NS))


@triton.jit
def _pair_count_t(
    X,
    H,
    RK,
    N: tl.constexpr,
    K: tl.constexpr,
    S0: tl.constexpr,
    S1: tl.constexpr,
    R: tl.constexpr,
    CH: tl.constexpr,
    PB: tl.constexpr,
    CONTIG: tl.constexpr,
):
    c = tl.program_id(0)
    i = c * PB + tl.arange(0, PB)
    if CONTIG:
        v = tl.load(X + i, i < N, other=R)
    else:
        v = tl.load(X + i // K * S0 + i % K * S1, i < N, other=R)
    valid = (i < N) & (v >= 0) & (v < R)
    equal = (v[:, None] == v[None, :]) & valid[None, :]
    rank = tl.sum(
        (equal & (tl.arange(0, PB)[None, :] < tl.arange(0, PB)[:, None])).to(
            tl.int32
        ),
        1,
    )
    first = valid & (rank == 0)
    count = tl.sum(equal.to(tl.int32), 1)
    tl.store(H + v * CH + c, count, first)
    tl.store(RK + i, rank, i < N)


@triton.jit
def _coltot_t(H, P, C, CH: tl.constexpr, CB: tl.constexpr):
    e = tl.program_id(0)
    running = tl.zeros((), tl.int32)
    for c0 in range(0, CH, CB):
        c = c0 + tl.arange(0, CB)
        h = tl.load(H + e * CH + c, c < CH, other=0)
        p = tl.cumsum(h, 0) - h + running
        tl.store(P + e * CH + c, p, c < CH)
        running += tl.sum(h, 0)
    tl.store(C + e, running)


@triton.jit
def _bake(P, OFF, Q, R: tl.constexpr, CH: tl.constexpr, CB: tl.constexpr):
    e = tl.program_id(0)
    if e < R:
        base = tl.load(OFF + e)
        for c0 in range(0, CH, CB):
            c = c0 + tl.arange(0, CB)
            p = tl.load(P + e * CH + c, c < CH, other=0)
            tl.store(Q + e * CH + c, base + p, c < CH)


@triton.jit
def _pair_place_t(
    X,
    RK,
    Q,
    OUT,
    N: tl.constexpr,
    K: tl.constexpr,
    S0: tl.constexpr,
    S1: tl.constexpr,
    R: tl.constexpr,
    NS: tl.constexpr,
    CH: tl.constexpr,
    PB: tl.constexpr,
    CONTIG: tl.constexpr,
):
    c = tl.program_id(0)
    i = c * PB + tl.arange(0, PB)
    if CONTIG:
        v = tl.load(X + i, i < N, other=R)
    else:
        v = tl.load(X + i // K * S0 + i % K * S1, i < N, other=R)
    valid = (i < N) & (v >= 0) & (v < R)
    rank = tl.load(RK + i, i < N, other=0)
    vs = tl.minimum(tl.maximum(v, 0), R - 1)
    base = tl.load(Q + vs * CH + c, valid, other=0)
    target = base + rank
    tl.store(OUT + target, i, valid & (target < NS))


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
    n = topk_ids.numel()
    k = max(topk_ids.shape[1], 1)
    s0, s1 = topk_ids.stride()
    ne = num_experts
    ns = sorted_token_ids.numel()
    en = expert_ids.numel()
    s = torch.empty_like(sorted_token_ids)
    e = torch.empty_like(expert_ids)
    np = torch.empty_like(num_tokens_post_pad)
    if 0 < n <= 64:
        _tiny_pairs[max(1, triton.cdiv(max(ns, en), 128)),](
            topk_ids,
            s,
            expert_ids,
            e,
            np,
            n,
            k,
            s0,
            s1,
            ne - 1,
            block_size,
            ns,
            en,
            triton.next_power_of_2(n),
            128,
            np.numel(),
            triton.next_power_of_2(max(np.numel(), 1)),
            num_warps=4,
        )
        return (s, e, np)
    if n <= 1024 and 1 <= ne <= 1024:
        counts = torch.empty((ne,), device=topk_ids.device, dtype=torch.int32)
        _count_init[ne,](
            topk_ids,
            counts,
            s,
            expert_ids,
            e,
            n,
            k,
            s0,
            s1,
            ne,
            ns,
            en,
            256,
            num_warps=4,
        )
        _prefix_place[max(ne - 1, 1),](
            topk_ids,
            counts,
            s,
            e,
            np,
            n,
            k,
            s0,
            s1,
            ne,
            block_size,
            ns,
            en,
            triton.next_power_of_2(ne),
            256,
            np.numel(),
            triton.next_power_of_2(max(np.numel(), 1)),
            num_warps=4,
        )
        return (s, e, np)
    contig = 1 if (s0 == k and s1 == 1) else 0
    routed = ne - 1
    ep = triton.next_power_of_2(max(routed, 1))
    pb = 128
    ch = triton.cdiv(n, pb)
    hist = torch.empty((ch * ep,), dtype=torch.int32, device=topk_ids.device)
    pre = torch.empty((ch * ep,), dtype=torch.int32, device=topk_ids.device)
    rk = torch.empty((max(n, 1),), dtype=torch.int32, device=topk_ids.device)
    cnt = torch.empty((ep,), dtype=torch.int32, device=topk_ids.device)
    off = torch.empty((ne + 1,), dtype=torch.int32, device=topk_ids.device)
    size = max(ns, en, ch * ep)
    if size:
        _init[triton.cdiv(size, 8192),](
            s, expert_ids, e, hist, n, ns, en, ch * ep, 8192, num_warps=4
        )
    if n and ch >= 1024:
        qtab = torch.empty(
            (ch * ep,), dtype=torch.int32, device=topk_ids.device
        )
        _pair_count_t[ch,](
            topk_ids,
            hist,
            rk,
            n,
            k,
            s0,
            s1,
            routed,
            ch,
            pb,
            contig,
            num_warps=4,
        )
        _coltot_t[ep,](hist, pre, cnt, ch, 256, num_warps=4)
        _begin[1,](
            cnt,
            off,
            expert_ids,
            e,
            np,
            routed,
            block_size,
            ep,
            en,
            np.numel(),
            triton.next_power_of_2(max(np.numel(), 1)),
            32,
            num_warps=4,
        )
        _bake[ep,](pre, off, qtab, routed, ch, 256, num_warps=4)
        _pair_place_t[ch,](
            topk_ids,
            rk,
            qtab,
            s,
            n,
            k,
            s0,
            s1,
            routed,
            ns,
            ch,
            pb,
            contig,
            num_warps=4,
        )
        return (s, e, np)
    if n:
        _pair_count[ch,](
            topk_ids,
            hist,
            rk,
            n,
            k,
            s0,
            s1,
            routed,
            ep,
            pb,
            contig,
            num_warps=4,
        )
    _coltot[ep,](hist, pre, cnt, ch, ep, 64, num_warps=4)
    _begin[1,](
        cnt,
        off,
        expert_ids,
        e,
        np,
        routed,
        block_size,
        ep,
        en,
        np.numel(),
        triton.next_power_of_2(max(np.numel(), 1)),
        32,
        num_warps=4,
    )
    if n:
        _pair_place[ch,](
            topk_ids,
            rk,
            off,
            pre,
            s,
            n,
            k,
            s0,
            s1,
            routed,
            ns,
            ep,
            pb,
            contig,
            num_warps=4,
        )
    return (s, e, np)


__all__ = ["moe_align_block_size"]
