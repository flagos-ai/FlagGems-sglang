# Copyright 2026, The FlagOS Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
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

# GCU pipeline:
#   * `moe_align_block_size` with at most 64 routed payloads and at most
#     1024 routed experts takes a single-program `_fused_small` launch:
#     atomic cursor rank + in-register prefix + full output fill, saving the
#     two inter-kernel sync points that dominate tiny shapes.
#   * Larger inputs run the three-kernel pipeline:
#   1. `_count_rank_init`: per-chunk histogram + within-chunk rank via one
#      relaxed-atomic pass (old value == rank), plus the global N-fill of
#      `sorted_token_ids` and the `expert_ids` copy.
#   2. `_prefix_offsets`: chunked expert prefix (2D cumsum along chunks) plus
#      the padded begin/end table; rewrites the histogram in place as
#      `begin[e] + cross-chunk offset` so the scatter needs a single gather.
#   3. `_place_scatter`: rank scatter into the expert segments and the
#      `expert_ids` block fill.
# Intra-expert order is nondeterministic (atomic cursor), which the op
# contract allows (per-expert multiset comparison).


@triton.jit
def _fused_small(
    X,
    OUT,
    EI,
    EO,
    NP,
    CNT,
    N: tl.constexpr,
    K: tl.constexpr,
    S0: tl.constexpr,
    S1: tl.constexpr,
    R: tl.constexpr,
    BS: tl.constexpr,
    LOGBS: tl.constexpr,
    POW2BS: tl.constexpr,
    NS: tl.constexpr,
    EN: tl.constexpr,
    BN: tl.constexpr,
    EP: tl.constexpr,
    FB: tl.constexpr,
    NN: tl.constexpr,
    PN: tl.constexpr,
    KMAX: tl.constexpr,
):
    e = tl.arange(0, EP)
    tl.store(CNT + e, 0)
    tl.debug_barrier()
    i = tl.arange(0, BN)
    v = tl.load(X + i // K * S0 + i % K * S1, i < N, other=R)
    valid = (i < N) & (v >= 0) & (v < R)
    old = tl.atomic_add(CNT + v, 1, valid, sem="relaxed")
    tl.debug_barrier()
    cnt_v = tl.load(CNT + e)
    if POW2BS:
        padded = ((cnt_v + (BS - 1)) >> LOGBS) << LOGBS
    else:
        padded = tl.cdiv(cnt_v, BS) * BS
    end = tl.cumsum(padded, 0)
    begin = end - padded
    grand = tl.sum(padded, 0)
    tl.store(CNT + e, begin)
    tl.debug_barrier()
    for f0 in range(0, NS, FB):
        f = f0 + tl.arange(0, FB)
        tl.store(OUT + f, N, f < NS)
    for g0 in range(0, EN, FB):
        g = g0 + tl.arange(0, FB)
        old_e = tl.load(EI + g, g < EN, other=0)
        tl.store(EO + g, old_e, g < EN)
    tl.debug_barrier()
    base = tl.load(CNT + v, valid, other=0)
    slot = base + old
    tl.store(OUT + slot, i, valid & (slot < NS))
    if POW2BS:
        hb = begin >> LOGBS
        nb = (end - begin) >> LOGBS
    else:
        hb = begin // BS
        nb = (end - begin) // BS
    for m in range(0, KMAX):
        tl.store(EO + hb + m, e, (m < nb) & (e < R) & (hb + m < EN))
    q = tl.arange(0, PN)
    tl.store(NP + q, grand, q < NN)


@triton.jit
def _count_rank_init(
    X,
    OUT,
    EI,
    EO,
    CNT,
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
    C: tl.constexpr,
    FB: tl.constexpr,
    CONTIG: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid < CH:
        e = tl.arange(0, EP)
        tl.store(CNT + pid * EP + e, 0)
        tl.debug_barrier()
        i = pid * C + tl.arange(0, C)
        if CONTIG:
            v = tl.load(X + i, i < N, other=R)
        else:
            v = tl.load(X + i // K * S0 + i % K * S1, i < N, other=R)
        valid = (i < N) & (v >= 0) & (v < R)
        old = tl.atomic_add(CNT + pid * EP + v, 1, valid, sem="relaxed")
        tl.store(RK + i, old, valid)
    f = pid * FB + tl.arange(0, FB)
    tl.store(OUT + f, N, f < NS)
    old_e = tl.load(EI + f, f < EN, other=0)
    tl.store(EO + f, old_e, f < EN)


@triton.jit
def _prefix_offsets(
    CNT,
    EO,
    NP,
    R: tl.constexpr,
    BS: tl.constexpr,
    EP: tl.constexpr,
    CH: tl.constexpr,
    BC: tl.constexpr,
    EN: tl.constexpr,
    NN: tl.constexpr,
    PN: tl.constexpr,
    EO_FILL: tl.constexpr,
    KMAX: tl.constexpr,
    OFF_BASE: tl.constexpr,
):
    e = tl.arange(0, EP)
    if CH == 1:
        cnt0 = tl.load(CNT + e)
        total = tl.where(e < R, cnt0, 0)
        padded = tl.cdiv(total, BS) * BS
        end = tl.cumsum(padded, 0)
        begin = end - padded
        grand = tl.sum(padded, 0)
        tl.store(CNT + e, begin)
    else:
        q = tl.arange(0, BC)
        running = tl.zeros((EP,), tl.int32)
        for start in range(0, CH, BC):
            chunk = start + q
            mask = chunk[:, None] < CH
            h = tl.load(CNT + chunk[:, None] * EP + e[None, :], mask, other=0)
            within = tl.cumsum(h, 0) - h
            tl.store(
                CNT + chunk[:, None] * EP + e[None, :],
                running[None, :] + within,
                mask,
            )
            running += tl.sum(h, 0)
        total = tl.where(e < R, running, 0)
        padded = tl.cdiv(total, BS) * BS
        end = tl.cumsum(padded, 0)
        begin = end - padded
        grand = tl.sum(padded, 0)
    tl.store(CNT + OFF_BASE + e, begin, e < R)
    tl.store(CNT + OFF_BASE + R, grand)
    qn = tl.arange(0, PN)
    tl.store(NP + qn, grand, qn < NN)
    if EO_FILL:
        hb = begin // BS
        nb = (end - begin) // BS
        for m in range(0, KMAX):
            tl.store(EO + hb + m, e, (m < nb) & (e < R) & (hb + m < EN))


@triton.jit
def _place_scatter(
    X,
    CNT,
    RK,
    OUT,
    EO,
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
    C: tl.constexpr,
    OFF_BASE: tl.constexpr,
    B: tl.constexpr,
    CONTIG: tl.constexpr,
    EO_IN_K3: tl.constexpr,
    FOLDED: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid < CH:
        i = pid * C + tl.arange(0, C)
        if CONTIG:
            v = tl.load(X + i, i < N, other=R)
        else:
            v = tl.load(X + i // K * S0 + i % K * S1, i < N, other=R)
        valid = (i < N) & (v >= 0) & (v < R)
        rk = tl.load(RK + i, valid, other=0)
        if FOLDED:
            base = tl.load(CNT + pid * EP + v, valid, other=0)
        else:
            off_v = tl.load(CNT + OFF_BASE + v, valid, other=0)
            col_v = tl.load(CNT + pid * EP + v, valid, other=0)
            base = off_v + col_v
        slot = base + rk
        tl.store(OUT + slot, i, valid & (slot < NS))
    if EO_IN_K3 and pid < R:
        b0 = tl.load(CNT + OFF_BASE + pid) // BS
        b1 = tl.load(CNT + OFF_BASE + pid + 1) // BS
        r = tl.arange(0, B)
        for s in range(b0, b1, B):
            j = s + r
            tl.store(EO + j, pid, (j < b1) & (j < EN))


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
    routed = ne - 1
    if 0 < n <= 64 and routed <= 1024 and ns <= 32768:
        ep = triton.next_power_of_2(max(routed, 1))
        pow2bs = 1 if (block_size & (block_size - 1)) == 0 else 0
        logbs = (block_size.bit_length() - 1) if pow2bs else 0
        bn = triton.next_power_of_2(n)
        kmax = triton.cdiv(n, block_size) + 1
        cnt = torch.empty((ep,), dtype=torch.int32, device=topk_ids.device)
        _fused_small[1,](
            topk_ids,
            s,
            expert_ids,
            e,
            np,
            cnt,
            n,
            k,
            s0,
            s1,
            routed,
            block_size,
            logbs,
            pow2bs,
            ns,
            en,
            bn,
            ep,
            min(2048, triton.next_power_of_2(max(ns, 1))),
            np.numel(),
            triton.next_power_of_2(max(np.numel(), 1)),
            kmax,
            num_warps=max(1, min(4, bn // 16)),
        )
        return (s, e, np)
    if routed <= 8192:
        ep = triton.next_power_of_2(max(routed, 1))
        if n <= 4096:
            c = min(128, max(8, triton.next_power_of_2(max(n, 1))))
        elif n <= 32768:
            c = 256
        else:
            c = 1024
        ch = triton.cdiv(n, c)
        off_base = ch * ep
        cnt = torch.empty(
            (off_base + routed + 2,), dtype=torch.int32, device=topk_ids.device
        )
        rk = torch.empty(
            (max(n, 1),), dtype=torch.int32, device=topk_ids.device
        )
        contig = 1 if (s0 == k and s1 == 1) else 0
        grid1 = max(ch, triton.cdiv(ns, 8192), triton.cdiv(en, 8192), 1)
        _count_rank_init[grid1,](
            topk_ids,
            s,
            expert_ids,
            e,
            cnt,
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
            c,
            8192,
            contig,
            num_warps=4,
        )
        kmax = triton.cdiv(n, block_size) + 1 if n else 1
        eo_in_k2 = kmax <= 2
        bc = max(1, min(65536 // (ep * 4), 64 if ch > 32 else 32))
        _prefix_offsets[1,](
            cnt,
            e,
            np,
            routed,
            block_size,
            ep,
            ch,
            bc,
            en,
            np.numel(),
            triton.next_power_of_2(max(np.numel(), 1)),
            1 if eo_in_k2 else 0,
            kmax if eo_in_k2 else 1,
            off_base,
            num_warps=8,
        )
        grid3 = max(ch, routed) if not eo_in_k2 else max(ch, 1)
        _place_scatter[grid3,](
            topk_ids,
            cnt,
            rk,
            s,
            e,
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
            c,
            off_base,
            128,
            contig,
            0 if eo_in_k2 else 1,
            1 if ch == 1 else 0,
            num_warps=4,
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
            routed,
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
    ep = triton.next_power_of_2(max(routed, 1))
    pb = 128
    ch = triton.cdiv(n, pb)
    hist = torch.empty((ch * ep,), dtype=torch.int32, device=topk_ids.device)
    qtab = torch.empty((ch * ep,), dtype=torch.int32, device=topk_ids.device)
    rk = torch.empty((max(n, 1),), dtype=torch.int32, device=topk_ids.device)
    dbl = 4 if ch >= 512 else (2 if ch >= 64 else 1)
    chg = triton.cdiv(ch, dbl)
    size = max(ns, en, ch * ep)
    contig = 1 if (s0 == k and s1 == 1) else 0
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
