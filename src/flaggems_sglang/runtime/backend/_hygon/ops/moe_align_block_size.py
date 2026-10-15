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

_PLAN_MAX = 200000


@triton.jit
def _sorted_direct(
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
    BP: tl.constexpr,
    NN: tl.constexpr,
    PN: tl.constexpr,
):
    pid = tl.program_id(0)
    i = tl.arange(0, BN)
    v = tl.load(X + i // K * S0 + i % K * S1, i < N, other=R)
    bucket = tl.where((v >= 0) & (v < R), v, R)
    ordered = tl.sort(bucket * BN + i, descending=False)
    expert = ordered // BN
    original = ordered % BN
    valid = (expert < R) & (original < N)
    previous = tl.gather(expert, tl.maximum(i - 1, 0), 0)
    first = ((i == 0) | (expert != previous)) & valid
    group = tl.cumsum(first.to(tl.int32), 0)
    match = valid & (group == pid + 1)
    begin = tl.min(tl.where(match, i, BN), 0)
    count = tl.sum(match.to(tl.int32), 0)
    selected = tl.min(tl.where(match, expert, R), 0)
    groups = tl.sum(first.to(tl.int32), 0)
    lane = tl.arange(0, BP)
    index = tl.minimum(begin + lane, BN - 1)
    pair = tl.gather(original, index, 0)
    j = pid * BS + lane
    tl.store(OUT + j, tl.where(lane < count, pair, N), (lane < BS) & (j < NS))
    if pid < EN:
        old = tl.load(EI + pid)
        tl.store(EO + pid, tl.where(pid < groups, selected, old))
    if pid == 0:
        q = tl.arange(0, PN)
        tl.store(NP + q, groups * BS, q < NN)


@triton.jit
def _input_owner(
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
    BP: tl.constexpr,
    NN: tl.constexpr,
    PN: tl.constexpr,
):
    pid = tl.program_id(0)
    safe = tl.minimum(pid, N - 1)
    expert = tl.load(X + safe // K * S0 + safe % K * S1)
    i = tl.arange(0, BN)
    v = tl.load(X + i // K * S0 + i % K * S1, i < N, other=R)
    good = (i < N) & (v >= 0) & (v < R)
    hist = tl.histogram(tl.where(good, v, 0), EP)
    ex = tl.arange(0, EP)
    drop = tl.sum((~good).to(tl.int32), 0)
    hist = hist - tl.where(ex == 0, drop, 0)
    aligned = tl.where(ex < R, tl.cdiv(hist, BS) * BS, 0)
    begin = tl.sum(tl.where(ex < expert, aligned, 0), 0)
    total = tl.sum(aligned, 0)
    j = tl.arange(0, BP)
    earlier = tl.sum(((i < pid) & (v == expert)).to(tl.int32), 0)
    owner = (pid < N) & (expert >= 0) & (expert < R) & (earlier == 0)
    if owner:
        hit = (i < N) & (v == expert)
        rank = tl.cumsum(hit.to(tl.int32), 0)
        count = tl.sum(hit.to(tl.int32), 0)
        end = tl.cdiv(count, BS) * BS
        target = begin + rank - 1
        tl.store(OUT + target, i, hit & (target < NS))
        pad = count + j
        tl.store(OUT + begin + pad, N, (pad < end) & (begin + pad < NS))
        block = begin // BS + j
        tl.store(EO + block, expert, (j < end // BS) & (block < EN))
    if pid == N:
        for start in range(total, NS, BP):
            idx = start + j
            tl.store(OUT + idx, N, idx < NS)
        for start in range(total // BS, EN, BP):
            idx = start + j
            old = tl.load(EI + idx, idx < EN, other=0)
            tl.store(EO + idx, old, idx < EN)
        q = tl.arange(0, PN)
        tl.store(NP + q, total, q < NN)


@triton.jit
def _small(
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
    BP: tl.constexpr,
    NN: tl.constexpr,
    PN: tl.constexpr,
):
    expert = tl.program_id(0)
    i = tl.arange(0, BN)
    v = tl.load(X + i // K * S0 + i % K * S1, i < N, other=R)
    good = (i < N) & (v >= 0) & (v < R)
    hist = tl.histogram(tl.where(good, v, 0), EP)
    ex = tl.arange(0, EP)
    drop = tl.sum((~good).to(tl.int32), 0)
    hist = hist - tl.where(ex == 0, drop, 0)
    aligned = tl.where(ex < R, tl.cdiv(hist, BS) * BS, 0)
    begin = tl.sum(tl.where(ex < expert, aligned, 0), 0)
    total = tl.sum(aligned, 0)
    j = tl.arange(0, BP)
    if expert < R:
        hit = (i < N) & (v == expert)
        rank = tl.cumsum(hit.to(tl.int32), 0)
        count = tl.sum(hit.to(tl.int32), 0)
        end = tl.cdiv(count, BS) * BS
        target = begin + rank - 1
        tl.store(OUT + target, i, hit & (target < NS))
        pad = count + j
        tl.store(OUT + begin + pad, N, (pad < end) & (begin + pad < NS))
        block = begin // BS + j
        tl.store(EO + block, expert, (j < end // BS) & (block < EN))
    else:
        for start in range(total, NS, BP):
            idx = start + j
            tl.store(OUT + idx, N, idx < NS)
        for start in range(total // BS, EN, BP):
            idx = start + j
            old = tl.load(EI + idx, idx < EN, other=0)
            tl.store(EO + idx, old, idx < EN)
    if expert == 0:
        q = tl.arange(0, PN)
        tl.store(NP + q, total, q < NN)


@triton.jit
def _plan(
    X,
    P,
    NP,
    N: tl.constexpr,
    K: tl.constexpr,
    S0: tl.constexpr,
    S1: tl.constexpr,
    R: tl.constexpr,
    BS: tl.constexpr,
    EP: tl.constexpr,
    PS: tl.constexpr,
    PB: tl.constexpr,
    NN: tl.constexpr,
    PN: tl.constexpr,
):
    e = tl.arange(0, EP)
    acc = tl.full((EP,), 0, tl.int32)
    for s in range(0, N, PB):
        j = s + tl.arange(0, PB)
        v = tl.load(X + j // K * S0 + j % K * S1, j < N, other=-1)
        good = (j < N) & (v >= 0) & (v < R)
        acc += tl.histogram(tl.where(good, v, 0), EP)
        acc -= tl.where(e == 0, tl.sum((~good).to(tl.int32), 0), 0)
    count = tl.where(e < R, acc, 0)
    padded = tl.cdiv(count, BS) * BS
    begin = tl.cumsum(padded, 0) - padded
    total = tl.sum(padded, 0)
    tl.store(P + e, begin, e < R)
    tl.store(P + R, total)
    tl.store(P + PS + e, begin, e < R)
    tl.store(P + 2 * PS + e, begin + count, e < R)
    q = tl.arange(0, PN)
    tl.store(NP + q, total, q < NN)


@triton.jit
def _place(
    X,
    P,
    OUT,
    EI,
    EO,
    N: tl.constexpr,
    K: tl.constexpr,
    S0: tl.constexpr,
    S1: tl.constexpr,
    R: tl.constexpr,
    BS: tl.constexpr,
    NS: tl.constexpr,
    EN: tl.constexpr,
    PS: tl.constexpr,
    B: tl.constexpr,
    FB: tl.constexpr,
    G: tl.constexpr,
    GB: tl.constexpr,
):
    pid = tl.program_id(0)
    r = tl.arange(0, B)
    f = tl.arange(0, FB)
    total = tl.load(P + R)
    if pid * B < N:
        i = pid * B + r
        raw = tl.load(X + i // K * S0 + i % K * S1, i < N, other=-1)
        valid = (i < N) & (raw >= 0) & (raw < R)
        e = tl.where(valid, raw, 0)
        slot = tl.atomic_add(P + PS + e, 1, valid, sem="relaxed")
        tl.store(OUT + slot, i, valid & (slot < NS))
    gl = tl.arange(0, GB)
    for p in range(pid, R, G):
        head = tl.load(P + p)
        gap = tl.load(P + 2 * PS + p)
        stop = tl.load(P + p + 1)
        for s in range(gap, stop, GB):
            idx = s + gl
            tl.store(OUT + idx, N, (idx < stop) & (idx < NS))
        for s in range(head // BS, stop // BS, GB):
            idx = s + gl
            tl.store(EO + idx, p, (idx < stop // BS) & (idx < EN))
    for s in range(pid * FB, NS, G * FB):
        idx = s + f
        tl.store(OUT + idx, N, (idx >= total) & (idx < NS))
    for s in range(pid * FB, EN, G * FB):
        idx = s + f
        keep = (idx >= total // BS) & (idx < EN)
        old = tl.load(EI + idx, keep, other=0)
        tl.store(EO + idx, old, keep)


@triton.jit
def _init_hist(
    X,
    S,
    EI,
    EO,
    H,
    N: tl.constexpr,
    K: tl.constexpr,
    S0: tl.constexpr,
    S1: tl.constexpr,
    NS: tl.constexpr,
    EN: tl.constexpr,
    R: tl.constexpr,
    EP: tl.constexpr,
    CH: tl.constexpr,
    HB: tl.constexpr,
    B: tl.constexpr,
):
    pid = tl.program_id(0)
    i = pid * B + tl.arange(0, B)
    tl.store(S + i, N, i < NS)
    old = tl.load(EI + i, i < EN, other=0)
    tl.store(EO + i, old, i < EN)
    if pid < CH:
        e = tl.arange(0, EP)
        acc = tl.full((EP,), 0, tl.int32)
        for start in range(0, HB, B):
            j = pid * HB + start + tl.arange(0, B)
            v = tl.load(X + j // K * S0 + j % K * S1, j < N, other=-1)
            good = (j < N) & (v >= 0) & (v < R)
            acc += tl.histogram(tl.where(good, v, 0), EP)
            acc -= tl.where(e == 0, tl.sum((~good).to(tl.int32), 0), 0)
        tl.store(H + pid * EP + e, acc)


@triton.jit
def _hist_prefix(
    H,
    OFF,
    NP,
    R: tl.constexpr,
    BS: tl.constexpr,
    EP: tl.constexpr,
    CH: tl.constexpr,
    NN: tl.constexpr,
    BN: tl.constexpr,
    BC: tl.constexpr,
):
    e = tl.arange(0, EP)
    q = tl.arange(0, BC)
    count = tl.full((EP,), 0, tl.int32)
    for start in range(0, CH, BC):
        chunk = start + q
        mask = chunk[:, None] < CH
        h = tl.load(H + chunk[:, None] * EP + e[None, :], mask, other=0)
        within = tl.cumsum(h, 0) - h
        tl.store(
            H + chunk[:, None] * EP + e[None, :], count[None, :] + within, mask
        )
        count += tl.sum(h, 0)
    count = tl.where(e < R, count, 0)
    padded = tl.cdiv(count, BS) * BS
    end = tl.cumsum(padded, 0)
    begin = end - padded
    tl.store(OFF + e, begin, e < R)
    tl.store(OFF + R, tl.sum(padded, 0))
    j = tl.arange(0, BN)
    tl.store(NP + j, tl.sum(padded, 0), j < NN)


@triton.jit
def _scatter(
    X,
    H,
    OFF,
    OUT,
    EI,
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
    HB: tl.constexpr,
    B: tl.constexpr,
):
    pid = tl.program_id(0)
    r = tl.arange(0, B)
    if pid * B < N:
        i = pid * B + r
        raw = tl.load(X + i // K * S0 + i % K * S1, i < N, other=-1)
        valid = (i < N) & (raw >= 0) & (raw < R)
        e = tl.where(valid, raw, 0)
        local = tl.atomic_add(
            H + pid * B // HB * EP + e, 1, valid, sem="relaxed"
        )
        slot = tl.load(OFF + e, valid, other=0) + local
        tl.store(OUT + slot, i, valid & (slot < NS))
    if pid < R:
        begin = tl.load(OFF + pid) // BS
        end = tl.load(OFF + pid + 1) // BS
        for start in range(begin, end, B):
            j = start + r
            tl.store(EI + j, pid, (j < end) & (j < EN))


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
    routed = num_experts - 1
    ns = sorted_token_ids.numel()
    en = expert_ids.numel()
    out = torch.empty_like(sorted_token_ids)
    eids = torch.empty_like(expert_ids)
    npost = torch.empty_like(num_tokens_post_pad)
    if (
        0 < n <= min(64, block_size)
        and block_size <= 1024
        and (num_experts <= 1024)
        and (n * block_size < 1 << 30)
    ):
        _sorted_direct[max(triton.cdiv(ns, block_size), en, 1),](
            topk_ids,
            out,
            expert_ids,
            eids,
            npost,
            n,
            k,
            s0,
            s1,
            routed,
            block_size,
            ns,
            en,
            triton.next_power_of_2(n),
            triton.next_power_of_2(block_size),
            npost.numel(),
            triton.next_power_of_2(max(npost.numel(), 1)),
            num_warps=1,
        )
        return (out, eids, npost)
    if 0 < n < routed and n <= 1024 and (num_experts <= 1024):
        _input_owner[n + 1,](
            topk_ids,
            out,
            expert_ids,
            eids,
            npost,
            n,
            k,
            s0,
            s1,
            routed,
            block_size,
            ns,
            en,
            triton.next_power_of_2(n),
            triton.next_power_of_2(max(routed, 1)),
            triton.next_power_of_2(n + block_size - 1),
            npost.numel(),
            triton.next_power_of_2(max(npost.numel(), 1)),
            num_warps=4,
        )
        return (out, eids, npost)
    if 0 < n <= 1024 and num_experts <= 1024:
        _small[num_experts,](
            topk_ids,
            out,
            expert_ids,
            eids,
            npost,
            n,
            k,
            s0,
            s1,
            routed,
            block_size,
            ns,
            en,
            triton.next_power_of_2(n),
            triton.next_power_of_2(max(routed, 1)),
            triton.next_power_of_2(n + block_size - 1),
            npost.numel(),
            triton.next_power_of_2(max(npost.numel(), 1)),
            num_warps=4,
        )
        return (out, eids, npost)
    if 1024 < n <= _PLAN_MAX and num_experts <= 1024:
        ps = num_experts
        plan = torch.empty((3, ps), dtype=torch.int32, device=topk_ids.device)
        _plan[1,](
            topk_ids,
            plan,
            npost,
            n,
            k,
            s0,
            s1,
            routed,
            block_size,
            triton.next_power_of_2(max(routed, 1)),
            ps,
            min(4096, triton.next_power_of_2(n)),
            npost.numel(),
            triton.next_power_of_2(max(npost.numel(), 1)),
            num_warps=8,
        )
        g = max(triton.cdiv(n, 256), routed, 1)
        _place[g,](
            topk_ids,
            plan,
            out,
            expert_ids,
            eids,
            n,
            k,
            s0,
            s1,
            routed,
            block_size,
            ns,
            en,
            ps,
            256,
            1024,
            g,
            triton.next_power_of_2(block_size),
            num_warps=4,
        )
        return (out, eids, npost)
    offsets = torch.empty(
        (num_experts,), dtype=torch.int32, device=topk_ids.device
    )
    ep = triton.next_power_of_2(max(routed, 1))
    hb = max(1024, triton.next_power_of_2(triton.cdiv(max(n, 1), 32)))
    chunks = triton.cdiv(n, hb)
    histograms = torch.empty(
        (chunks, ep), dtype=torch.int32, device=topk_ids.device
    )
    size = max(ns, en, n)
    if size:
        _init_hist[triton.cdiv(size, 1024),](
            topk_ids,
            out,
            expert_ids,
            eids,
            histograms,
            n,
            k,
            s0,
            s1,
            ns,
            en,
            routed,
            ep,
            chunks,
            hb,
            1024,
            num_warps=4,
        )
    _hist_prefix[1,](
        histograms,
        offsets,
        npost,
        routed,
        block_size,
        ep,
        chunks,
        npost.numel(),
        triton.next_power_of_2(max(npost.numel(), 1)),
        min(64, triton.next_power_of_2(max(chunks, 1))),
        num_warps=8 if chunks > 16 else 4,
    )
    grid = max(triton.cdiv(n, 256), routed)
    if grid:
        _scatter[grid,](
            topk_ids,
            histograms,
            offsets,
            out,
            eids,
            n,
            k,
            s0,
            s1,
            routed,
            block_size,
            ns,
            en,
            ep,
            chunks,
            hb,
            256,
            num_warps=4,
        )
    return (out, eids, npost)


__all__ = ["moe_align_block_size"]
