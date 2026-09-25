# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Triton implementation of moe/fused_moe_dispatch_index.

Builds the permutation indices for masked/DeepGEMM-style grouped GEMM MoE
"""

import torch
import triton
import triton.language as tl

# Single-program path: one [E, BLOCK] one-hot tile must hold the whole
# flattened input (tile ceiling on this backend's local memory).
_SMALL_MAX = 512
# Single-launch rescanning path bound: beyond this the O(nc^2) base
# re-derivation redundancy outgrows the saved launch overhead.
_MID_MAX = 2048
# Elements per chunk for the single-launch and two-launch paths.
_MID_CHUNK = 512
_LARGE_CHUNK_SMALL = 1024  # 2048 < n <= 16384
_LARGE_CHUNK = 2048  # n > 16384
# Histogram-tile rows handled per iteration inside the base
# re-derivation loops.
_CH_SCAN = 16


@triton.jit
def _small_kernel(
    ids_ptr,  # [n] int32 expert ids, -1 = padding
    out_ptr,  # [n] int32 destination rows
    masked_m_ptr,  # [num_experts] int32
    n,
    m_max,
    num_experts,
    E_PAD: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offs = tl.arange(0, BLOCK)
    mask = offs < n
    # Tail lanes load -1, which never matches an expert column — the
    # one-hot predicate needs no separate bounds term.
    e = tl.load(ids_ptr + offs, mask=mask, other=-1)
    ev = tl.arange(0, E_PAD)
    # [E, BLOCK] one-hot; cumsum along the last (contiguous) axis gives
    # each slot's running per-expert rank. This orientation measures
    # faster than the transposed [BLOCK, E] tile on this backend.
    onehot = (e[None, :] == ev[:, None]).to(tl.int32)
    cs = tl.cumsum(onehot, axis=1)
    # coef[ev] = ev*m_max - 1; summing (cs + coef) * onehot along experts
    # yields e*m_max + rank - 1 + 1 - 1 per matched slot directly.
    coef = ev * m_max - 1
    dst = tl.sum((cs + coef[:, None]) * onehot, axis=0)
    dst = tl.where(e >= 0, dst, 0)
    tl.store(out_ptr + offs, dst, mask=mask)
    tl.store(masked_m_ptr + ev, tl.sum(onehot, axis=1), mask=ev < num_experts)


@triton.jit
def _single_launch_kernel(
    ids_ptr,  # [n] int32 expert ids, -1 = padding
    out_ptr,  # [n] int32 destination rows
    masked_m_ptr,  # [num_experts] int32
    n,
    m_max,
    num_experts,
    E_PAD: tl.constexpr,
    CHUNK: tl.constexpr,
    SUB: tl.constexpr,
):
    p = tl.program_id(0)
    NP = tl.num_programs(0)
    ev = tl.arange(0, E_PAD)
    # Exclusive per-expert base for this chunk: rescan every earlier
    # chunk. At nc <= 4 the redundancy is cheaper than a second launch.
    hist = tl.zeros([E_PAD], dtype=tl.int32)
    for c in range(0, p):
        for s in range(0, CHUNK // SUB):
            offs = c * CHUNK + s * SUB + tl.arange(0, SUB)
            m = offs < n
            ids = tl.load(ids_ptr + offs, mask=m, other=-1)
            oh = (ids[None, :] == ev[:, None]).to(tl.int32)
            hist += tl.sum(oh, axis=1)
    base = hist
    # Rank the chunk in SUB-blocks, carrying the running per-expert rank
    # across blocks; the tile is built once per block and reused for the
    # carry, the cumsum and the fused reduction.
    run = tl.zeros([E_PAD], dtype=tl.int32)
    for s in range(0, CHUNK // SUB):
        offs = p * CHUNK + s * SUB + tl.arange(0, SUB)
        m = offs < n
        ids = tl.load(ids_ptr + offs, mask=m, other=-1)
        oh_t = (ids[None, :] == ev[:, None]).to(tl.int32)  # [E, SUB]
        cs = tl.cumsum(oh_t, axis=1)
        coef = ev * m_max + base + run - 1
        dst = tl.sum((cs + coef[:, None]) * oh_t, axis=0)
        tl.store(out_ptr + offs, dst, mask=m)
        run += tl.sum(oh_t, axis=1)
    # The last program has seen every element (its own chunk + all
    # earlier ones), so it publishes masked_m.
    if p == NP - 1:
        tl.store(masked_m_ptr + ev, hist + run, mask=ev < num_experts)


@triton.jit
def _count_trim_kernel(
    ids_ptr,  # [n] int32 expert ids, -1 = padding
    cnt_ptr,  # [num_chunks, E_PAD] int32; row p = chunk p's expert histogram
    n,
    num_experts,
    E_PAD: tl.constexpr,
    BLOCK: tl.constexpr,
    SUB: tl.constexpr,
):
    p = tl.program_id(0)
    ev = tl.arange(0, E_PAD)
    cnt = tl.zeros([E_PAD], dtype=tl.int32)
    for s in range(0, BLOCK // SUB):
        offs = p * BLOCK + s * SUB + tl.arange(0, SUB)
        mask = offs < n
        e = tl.load(ids_ptr + offs, mask=mask, other=-1)
        onehot = (e[:, None] == ev[None, :]).to(tl.int32)
        cnt += tl.sum(onehot, axis=0)
    # row store: chunk p's histogram lands in contiguous row p
    # ([num_chunks, E] layout — one contiguous [E] vector per chunk in
    # the rank kernel's base re-derivation).
    tl.store(cnt_ptr + p * E_PAD + ev, cnt, mask=ev < num_experts)


@triton.jit
def _rank_fused_kernel(
    ids_ptr,  # [n] int32 expert ids, -1 = padding
    cnt_ptr,  # [num_chunks, E_PAD] int32; row p = chunk p's expert histogram
    out_ptr,  # [n] int32 destination rows
    masked_m_ptr,  # [num_experts] int32
    n,
    m_max,
    num_experts,
    E_PAD: tl.constexpr,
    BLOCK: tl.constexpr,
    SUB: tl.constexpr,
    CH: tl.constexpr,
):
    p = tl.program_id(0)
    NP = tl.num_programs(0)
    ev = tl.arange(0, E_PAD)
    # Every program derives the full histogram total `tot` (scanning ALL
    # rows) and its exclusive per-expert base as `tot - tail(rows > p)`.
    # This uniform work schedules into the same wave as the ranking
    # instead of leaving a serialized full-histogram scan on program 0
    # to run after the last wave (the v8 scheme); every program stores
    # the same masked_m total — benign duplicate stores keep the
    # publish race-free with no atomics.
    tot = tl.zeros([E_PAD], dtype=tl.int32)
    for t in range(0, tl.cdiv(NP, CH)):
        cv = t * CH + tl.arange(0, CH)
        hv = tl.load(
            cnt_ptr + cv[:, None] * E_PAD + ev[None, :],
            mask=(cv[:, None] < NP) & (ev[None, :] < num_experts),
            other=0,
        )
        tot += tl.sum(hv, axis=0)
    tl.store(masked_m_ptr + ev, tot, mask=ev < num_experts)
    tail = tl.zeros([E_PAD], dtype=tl.int32)
    for t in range(0, tl.cdiv(NP - p, CH)):
        cv = p + t * CH + tl.arange(0, CH)
        hv = tl.load(
            cnt_ptr + cv[:, None] * E_PAD + ev[None, :],
            mask=(cv[:, None] < NP) & (ev[None, :] < num_experts),
            other=0,
        )
        tail += tl.sum(hv, axis=0)
    base = tot - tail

    # Rank the chunk in SUB-blocks, carrying the per-expert base across
    # blocks. The one-hot tile is built once per block and reused for
    # both the carry and the fused rank reduction; the cumsum runs along
    # the tile's last (contiguous) axis.
    for s in range(0, BLOCK // SUB):
        offs = p * BLOCK + s * SUB + tl.arange(0, SUB)
        mask = offs < n
        e = tl.load(ids_ptr + offs, mask=mask, other=-1)
        onehot_t = (e[None, :] == ev[:, None]).to(tl.int32)  # [E, SUB]
        cs = tl.cumsum(onehot_t, axis=1)
        # coef[ev] = ev*m_max + base[ev] - 1: summing (cs + coef)*onehot
        # along experts yields the destination row directly (padding
        # slots contribute via no expert column and are zeroed below).
        coef = ev * m_max + base - 1
        dst = tl.sum((cs + coef[:, None]) * onehot_t, axis=0)
        dst = tl.where(e >= 0, dst, 0)
        tl.store(out_ptr + offs, dst, mask=mask)
        base += tl.sum(onehot_t, axis=1)


def fused_moe_dispatch_index(topk_ids, num_local_experts, m_max):
    flat = topk_ids.reshape(-1)
    if flat.dtype != torch.int32:
        flat = flat.to(torch.int32)
    ids = flat.contiguous()
    n = ids.numel()
    device = ids.device
    num_local_experts = int(num_local_experts)
    masked_m = torch.empty(num_local_experts, dtype=torch.int32, device=device)
    src2dst = torch.empty(n, dtype=torch.int32, device=device)
    if n == 0 or num_local_experts == 0:
        return masked_m.zero_(), src2dst

    e_pad = max(triton.next_power_of_2(num_local_experts), 1)
    m_max = int(m_max)

    if n <= _SMALL_MAX:
        # Small path: a single [E, BLOCK] one-hot tile spanning the whole
        # input in one single-program launch.
        block = max(triton.next_power_of_2(n), 16)
        _small_kernel[(1,)](
            ids,
            src2dst,
            masked_m,
            n,
            m_max,
            num_local_experts,
            E_PAD=e_pad,
            BLOCK=block,
            num_warps=1,
        )
        return masked_m, src2dst

    if n <= _MID_MAX:
        # Single launch: each chunk re-derives its base by rescanning
        # earlier chunks; at <= 4 chunks that beats a second launch.
        chunk = _MID_CHUNK
        num_chunks = triton.cdiv(n, chunk)
        _single_launch_kernel[(num_chunks,)](
            ids,
            src2dst,
            masked_m,
            n,
            m_max,
            num_local_experts,
            E_PAD=e_pad,
            CHUNK=chunk,
            SUB=chunk,
            num_warps=1,
            num_stages=1,
        )
        return masked_m, src2dst

    if n <= 16384:
        chunk = _LARGE_CHUNK_SMALL
    else:
        chunk = _LARGE_CHUNK
    num_chunks = triton.cdiv(n, chunk)
    counts = torch.empty((num_chunks, e_pad), dtype=torch.int32, device=device)
    _count_trim_kernel[(num_chunks,)](
        ids,
        counts,
        n,
        num_local_experts,
        E_PAD=e_pad,
        BLOCK=chunk,
        SUB=chunk,
        num_warps=1,
    )
    _rank_fused_kernel[(num_chunks,)](
        ids,
        counts,
        src2dst,
        masked_m,
        n,
        m_max,
        num_local_experts,
        E_PAD=e_pad,
        BLOCK=chunk,
        SUB=chunk,
        CH=_CH_SCAN,
        num_warps=1,
    )
    return masked_m, src2dst


__all__ = ["fused_moe_dispatch_index"]
