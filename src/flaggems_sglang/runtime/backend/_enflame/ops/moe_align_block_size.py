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
def _tiny_onehot(
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
    EP: tl.constexpr,
    BO: tl.constexpr,
    NN: tl.constexpr,
    PN: tl.constexpr,
):
    pid = tl.program_id(0)
    i = tl.arange(0, BN)
    v = tl.load(X + i // K * S0 + i % K * S1, i < N, other=R)
    valid = (i < N) & (v >= 0) & (v < R)
    before = (i[None, :] < i[:, None]) & (i[None, :] < N)
    equal = (v[None, :] == v[:, None]) & valid[None, :]
    rank = tl.sum((before & equal).to(tl.int32), 1)
    count = tl.sum(equal.to(tl.int32), 1)
    first = valid & (rank == 0)
    padded = tl.where(first, tl.cdiv(count, BS) * BS, 0)
    start = tl.sum(tl.where(v[None, :] < v[:, None], padded[None, :], 0), 1)
    length = tl.cdiv(count, BS) * BS
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
    OUT,
    EI,
    EO,
    H,
    RK,
    N: tl.constexpr,
    K: tl.constexpr,
    S0: tl.constexpr,
    S1: tl.constexpr,
    R: tl.constexpr,
    NS: tl.constexpr,
    EN: tl.constexpr,
    EP: tl.constexpr,
    CH: tl.constexpr,
    PB: tl.constexpr,
    B: tl.constexpr,
    CONTIG: tl.constexpr,
):
    pid = tl.program_id(0)
    i = pid * B + tl.arange(0, B)
    tl.store(OUT + i, N, i < NS)
    old = tl.load(EI + i, i < EN, other=0)
    tl.store(EO + i, old, i < EN)
    if pid < CH:
        j = pid * PB + tl.arange(0, PB)
        if CONTIG:
            v = tl.load(X + j, j < N, other=R)
        else:
            v = tl.load(X + j // K * S0 + j % K * S1, j < N, other=R)
        valid = (j < N) & (v >= 0) & (v < R)
        equal = (v[:, None] == v[None, :]) & valid[None, :]
        rank = tl.sum(
            (
                equal & (tl.arange(0, PB)[None, :] < tl.arange(0, PB)[:, None])
            ).to(tl.int32),
            1,
        )
        tl.store(RK + j, rank, j < N)
        e = tl.arange(0, EP)
        row = tl.sum(
            ((v[:, None] == e[None, :]) & valid[:, None]).to(tl.int32), 0
        )
        tl.store(H + pid * EP + e, row)


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
    CH: tl.constexpr,
    PB: tl.constexpr,
    DBL: tl.constexpr,
    CONTIG: tl.constexpr,
):
    pid = tl.program_id(0)
    for j in range(DBL):
        c = pid * DBL + j
        if c < CH:
            i = c * PB + tl.arange(0, PB)
            if CONTIG:
                v = tl.load(X + i, i < N, other=R)
            else:
                v = tl.load(X + i // K * S0 + i % K * S1, i < N, other=R)
            valid = (i < N) & (v >= 0) & (v < R)
            equal = (v[:, None] == v[None, :]) & valid[None, :]
            rank = tl.sum(
                (
                    equal
                    & (tl.arange(0, PB)[None, :] < tl.arange(0, PB)[:, None])
                ).to(tl.int32),
                1,
            )
            first = valid & (rank == 0)
            count = tl.sum(equal.to(tl.int32), 1)
            tl.store(H + c * EP + v, count, first)
            tl.store(RK + i, rank, i < N)


@triton.jit
def _prefix_fused(
    H,
    Q,
    EI,
    EO,
    NP,
    R: tl.constexpr,
    BS: tl.constexpr,
    EP: tl.constexpr,
    CH: tl.constexpr,
    EN: tl.constexpr,
    NN: tl.constexpr,
    PN: tl.constexpr,
    CB: tl.constexpr,
    EB: tl.constexpr,
):
    e = tl.arange(0, EP)
    cc = tl.arange(0, CB)
    total = tl.zeros((EP,), tl.int32)
    for c0 in range(0, CH, CB):
        tile = tl.load(
            H + (c0 + cc)[:, None] * EP + e[None, :],
            (c0 + cc)[:, None] < CH,
            other=0,
        )
        total += tl.sum(tile, 0)
    aligned = tl.where(e < R, tl.cdiv(total, BS) * BS, 0)
    end = tl.cumsum(aligned, 0)
    begin = end - aligned
    npost = tl.sum(aligned, 0)
    q = tl.arange(0, PN)
    tl.store(NP + q, npost, q < NN)
    for bstart in range(0, EN, EB):
        b = bstart + tl.arange(0, EB)
        pos = b * BS
        expert = tl.sum((begin[None, :] <= pos[:, None]).to(tl.int32), 1) - 1
        live = (b < EN) & (pos < npost)
        tl.store(EO + b, expert, live)
    running = tl.zeros((EP,), tl.int32)
    for c0 in range(0, CH, CB):
        tile = tl.load(
            H + (c0 + cc)[:, None] * EP + e[None, :],
            (c0 + cc)[:, None] < CH,
            other=0,
        )
        tpre = tl.cumsum(tile, 0) - tile
        tl.store(
            Q + (c0 + cc)[:, None] * EP + e[None, :],
            begin[None, :] + running[None, :] + tpre,
            (c0 + cc)[:, None] < CH,
        )
        running += tl.sum(tile, 0)


@triton.jit
def _pair_place(
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
    EP: tl.constexpr,
    CH: tl.constexpr,
    PB: tl.constexpr,
    DBL: tl.constexpr,
    CONTIG: tl.constexpr,
):
    pid = tl.program_id(0)
    for j in range(DBL):
        c = pid * DBL + j
        if c < CH:
            i = c * PB + tl.arange(0, PB)
            if CONTIG:
                v = tl.load(X + i, i < N, other=R)
            else:
                v = tl.load(X + i // K * S0 + i % K * S1, i < N, other=R)
            valid = (i < N) & (v >= 0) & (v < R)
            rank = tl.load(RK + i, i < N, other=0)
            vs = tl.minimum(tl.maximum(v, 0), R - 1)
            base = tl.load(Q + c * EP + vs, valid, other=0)
            target = base + rank
            tl.store(OUT + target, i, valid & (target < NS))


@triton.jit
def _mid_place(
    X,
    RK,
    H,
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
    EP: tl.constexpr,
    CH: tl.constexpr,
    PB: tl.constexpr,
    NN: tl.constexpr,
    PN: tl.constexpr,
    CB: tl.constexpr,
    EB: tl.constexpr,
    CONTIG: tl.constexpr,
):
    pid = tl.program_id(0)
    e = tl.arange(0, EP)
    cc = tl.arange(0, CB)
    h = tl.load(H + cc[:, None] * EP + e[None, :], cc[:, None] < CH, other=0)
    count = tl.sum(h, 0)
    aligned = tl.where(e < R, tl.cdiv(count, BS) * BS, 0)
    endv = tl.cumsum(aligned, 0)
    begin = endv - aligned
    total = tl.sum(aligned, 0)
    if pid < CH:
        i = pid * PB + tl.arange(0, PB)
        if CONTIG:
            v = tl.load(X + i, i < N, other=R)
        else:
            v = tl.load(X + i // K * S0 + i % K * S1, i < N, other=R)
        valid = (i < N) & (v >= 0) & (v < R)
        rank = tl.load(RK + i, i < N, other=0)
        vs = tl.minimum(tl.maximum(v, 0), R - 1)
        pre = tl.sum(tl.where(cc[:, None] < pid, h, 0), 0)
        base = begin + pre
        sel = e[None, :] == vs[:, None]
        base_i = tl.sum(tl.where(sel, base[None, :], 0), 1)
        target = base_i + rank
        tl.store(OUT + target, i, valid & (target < NS))
    j = pid * EB + tl.arange(0, EB)
    if pid * EB < EN:
        pos = j * BS
        expert = tl.sum((begin[None, :] <= pos[:, None]).to(tl.int32), 1) - 1
        live = (j < EN) & (pos < total)
        old = tl.load(EI + j, j < EN, other=0)
        tl.store(EO + j, tl.where(live, expert, old), j < EN)
    if pid == 0:
        q = tl.arange(0, PN)
        tl.store(NP + q, total, q < NN)


@triton.jit
def _tiny_prepare(
    X,
    OUT,
    SRC_EI,
    EI,
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
    BE: tl.constexpr,
    NN: tl.constexpr,
    PN: tl.constexpr,
    FB: tl.constexpr,
):
    pid = tl.program_id(0)
    j = pid * FB + tl.arange(0, FB)
    tl.store(OUT + j, N, j < NS)
    i = tl.arange(0, BN)
    v = tl.load(X + i // K * S0 + i % K * S1, i < N, other=R)
    valid = (i < N) & (v >= 0) & (v < R)
    equal = (v[:, None] == v[None, :]) & valid[None, :]
    rank = tl.sum((equal & (i[None, :] < i[:, None])).to(tl.int32), 1)
    first = valid & (rank == 0)
    group = tl.sum(
        ((v[None, :] < v[:, None]) & first[None, :]).to(tl.int32), 1
    )
    target = tl.where(valid, group * BS + rank, -1)
    mine = valid & (target >= pid * FB) & (target < pid * FB + FB)
    tl.store(OUT + target, i, mine & (target < NS))
    if pid == 0:
        groups = tl.sum(first.to(tl.int32), 0)
        e = tl.arange(0, BE)
        expert = tl.max(
            tl.where(
                first[None, :] & (group[None, :] == e[:, None]), v[None, :], -1
            ),
            1,
        )
        old = tl.load(SRC_EI + e, e < EN, other=0)
        tl.store(EI + e, tl.where(e < groups, expert, old), e < EN)
        q = tl.arange(0, PN)
        tl.store(NP + q, groups * BS, q < NN)


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
    if 16 < n <= 128 and n <= block_size and en <= 128 and ns <= 8192:
        _tiny_prepare[max(1, triton.cdiv(ns, 8192)),](
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
            triton.next_power_of_2(max(n, 1)),
            triton.next_power_of_2(max(en, 1)),
            np.numel(),
            triton.next_power_of_2(max(np.numel(), 1)),
            8192,
            num_warps=8,
        )
        return (s, e, np)
    if 0 < n <= 16:
        bo = 128 if ne <= 128 else 256
        _tiny_onehot[max(1, triton.cdiv(max(ns, en), bo)),](
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
            triton.next_power_of_2(ne),
            bo,
            np.numel(),
            triton.next_power_of_2(max(np.numel(), 1)),
            num_warps=4 if ne <= 128 else 1,
        )
        return (s, e, np)
    contig = 1 if (s0 == k and s1 == 1) else 0
    routed = ne - 1
    if 128 < n <= 1024 and num_experts <= 1024 and ns <= 65536:
        ep = triton.next_power_of_2(max(routed, 1))
        pb = min(128, max(64, triton.next_power_of_2(triton.cdiv(n, 4))))
        ch = triton.cdiv(n, pb)
        hist = torch.empty(
            (ch * ep,), dtype=torch.int32, device=topk_ids.device
        )
        rk = torch.empty(
            (max(n, 1),), dtype=torch.int32, device=topk_ids.device
        )
        size = max(ns, en)
        _count_init[max(triton.cdiv(size, 8192), ch, 1),](
            topk_ids,
            s,
            expert_ids,
            e,
            hist,
            rk,
            n,
            k,
            s0,
            s1,
            routed,
            ns,
            en,
            ep,
            ch,
            pb,
            8192,
            contig,
            num_warps=4,
        )
        grid = max(ch, triton.cdiv(en, 64), 1)
        _mid_place[grid,](
            topk_ids,
            rk,
            hist,
            s,
            expert_ids,
            e,
            np,
            n,
            k,
            s0,
            s1,
            routed,
            block_size,
            ns,
            en,
            ep,
            ch,
            pb,
            np.numel(),
            triton.next_power_of_2(max(np.numel(), 1)),
            triton.next_power_of_2(ch),
            64,
            contig,
            num_warps=4,
        )
        return (s, e, np)
    ep = triton.next_power_of_2(max(routed, 1))
    gch = triton.cdiv(max(n, 1), 128)
    if n and gch <= 32 and gch * ep <= 8192:
        pb = 256
        ch = triton.cdiv(n, pb)
        hist = torch.empty(
            (ch * ep,), dtype=torch.int32, device=topk_ids.device
        )
        rk = torch.empty(
            (max(n, 1),), dtype=torch.int32, device=topk_ids.device
        )
        _count_init[max(triton.cdiv(max(ns, en), 8192), ch, 1),](
            topk_ids,
            s,
            expert_ids,
            e,
            hist,
            rk,
            n,
            k,
            s0,
            s1,
            routed,
            ns,
            en,
            ep,
            ch,
            pb,
            8192,
            contig,
            num_warps=1,
        )
        grid = max(ch, triton.cdiv(en, 64), 1)
        _mid_place[grid,](
            topk_ids,
            rk,
            hist,
            s,
            expert_ids,
            e,
            np,
            n,
            k,
            s0,
            s1,
            routed,
            block_size,
            ns,
            en,
            ep,
            ch,
            pb,
            np.numel(),
            triton.next_power_of_2(max(np.numel(), 1)),
            triton.next_power_of_2(ch),
            64,
            contig,
            num_warps=1,
        )
        return (s, e, np)
    pb = 128
    ch = triton.cdiv(n, pb)
    hist = torch.empty((ch * ep,), dtype=torch.int32, device=topk_ids.device)
    qtab = torch.empty((ch * ep,), dtype=torch.int32, device=topk_ids.device)
    rk = torch.empty((max(n, 1),), dtype=torch.int32, device=topk_ids.device)
    dbl = 4 if ch >= 512 else (2 if ch >= 64 else 1)
    chg = triton.cdiv(ch, dbl)
    size = max(ns, en, ch * ep)
    if size:
        _init[triton.cdiv(size, 8192),](
            s, expert_ids, e, hist, n, ns, en, ch * ep, 8192, num_warps=4
        )
    if n:
        _pair_count[chg,](
            topk_ids,
            hist,
            rk,
            n,
            k,
            s0,
            s1,
            routed,
            ep,
            ch,
            pb,
            dbl,
            contig,
            num_warps=4,
        )
    _prefix_fused[1,](
        hist,
        qtab,
        expert_ids,
        e,
        np,
        routed,
        block_size,
        ep,
        ch,
        en,
        np.numel(),
        triton.next_power_of_2(max(np.numel(), 1)),
        32,
        32,
        num_warps=4,
    )
    if n:
        _pair_place[chg,](
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
            ep,
            ch,
            pb,
            dbl,
            contig,
            num_warps=4,
        )
    return (s, e, np)


__all__ = ["moe_align_block_size"]
