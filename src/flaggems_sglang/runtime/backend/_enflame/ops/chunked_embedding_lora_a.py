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
def _gather_presence_tiled(
    Ids,
    Weights,
    Widx,
    Ranks,
    Presence,
    Out,
    S: tl.constexpr,
    R: tl.constexpr,
    B: tl.constexpr,
    L: tl.constexpr,
    V: tl.constexpr,
    SI: tl.constexpr,
    SW: tl.constexpr,
    SR: tl.constexpr,
    WL: tl.constexpr,
    WR: tl.constexpr,
    WV: tl.constexpr,
    WIDE: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    si = tl.full((), SI, tl.int64) if WIDE else SI
    sr = tl.full((), SR, tl.int64) if WIDE else SR
    sw = tl.full((), SW, tl.int64) if WIDE else SW
    wl = tl.full((), WL, tl.int64) if WIDE else WL
    wr = tl.full((), WR, tl.int64) if WIDE else WR
    wv = tl.full((), WV, tl.int64) if WIDE else WV
    pid = tl.program_id(0).to(tl.int64 if WIDE else tl.int32)
    rank_blocks = tl.cdiv(R, BLOCK_R)
    rows = pid // rank_blocks * BLOCK_S + tl.arange(0, BLOCK_S)
    ks = pid % rank_blocks * BLOCK_R + tl.arange(0, BLOCK_R)
    valid_rows = rows < S
    valid = valid_rows[:, None] & (ks[None, :] < R)
    if B == 0:
        value = tl.full((BLOCK_S, BLOCK_R), 0, tl.float32)
    else:
        adapter = tl.full(
            (BLOCK_S, BLOCK_R), 0, tl.int64 if WIDE else tl.int32
        )
        found = tl.full((BLOCK_S, BLOCK_R), False, tl.int1)
        for b in range(B):
            present = tl.load(Presence + b * S + rows, valid_rows, 0)
            wi = tl.load(Widx + b * sw).to(tl.int64 if WIDE else tl.int32)
            wi = tl.where(wi < 0, wi + L, wi)
            active_rank = tl.load(Ranks + wi * sr).to(
                tl.int64 if WIDE else tl.int32
            )
            selected = (present[:, None] != 0) & (ks[None, :] < active_rank)
            adapter = tl.where(selected, wi, adapter)
            found |= selected
        token = tl.load(Ids + rows * si, valid_rows, 0).to(
            tl.int64 if WIDE else tl.int32
        )
        token = tl.where(token < 0, token + V, token)
        value = tl.load(
            Weights + adapter * wl + ks[None, :] * wr + token[:, None] * wv,
            valid & found,
            0.0,
        )
    tl.store(Out + rows[:, None] * R + ks[None, :], value, valid)


def _index_tensor(value, bound):
    return (
        value.to(torch.int32)
        if value.dtype == torch.int64 and bound <= 2147483647
        else value
    )


def chunked_embedding_lora_a(input_ids, weights, batch_info, vocab_size):
    s = input_ids.shape[0]
    loras, rank, vocab = weights.shape
    out = torch.empty((s, rank), dtype=weights.dtype, device=weights.device)
    if s == 0 or rank == 0:
        return out
    ids = _index_tensor(input_ids, vocab)
    seg = _index_tensor(batch_info.seg_indptr, s)
    widx = _index_tensor(batch_info.weight_indices, loras)
    ranks = _index_tensor(batch_info.lora_ranks, rank)
    perm = _index_tensor(batch_info.permutation, s)
    bs = int(batch_info.bs)
    wide = (
        max(
            s * rank,
            bs * s + 1,
            s,
            rank,
            loras,
            vocab,
            bs,
            sum(
                (
                    (d - 1) * st
                    for d, st in zip(weights.shape, weights.stride())
                )
            ),
            *(v.numel() * v.stride(0) for v in (ids, seg, widx, ranks, perm)),
        )
        + 1024
        > 2147483647
    )
    if wide and max(s, rank, loras, vocab, bs) + 1024 <= 2147483647:
        return _wide_rebased(ids, weights, seg, widx, ranks, perm, bs, out)
    if not wide and s <= 64:
        _inverse_small[triton.cdiv(s * rank, 64),](
            ids,
            weights,
            seg,
            widx,
            ranks,
            perm,
            out,
            s,
            rank,
            bs,
            loras,
            vocab,
            ids.stride(0),
            seg.stride(0),
            widx.stride(0),
            ranks.stride(0),
            perm.stride(0),
            *weights.stride(),
            False,
            triton.next_power_of_2(s) if bs else 1,
            64,
            num_warps=4,
            num_stages=1,
        )
        return out
    if not wide and 64 < s <= 16384 and (0 < bs <= 32):
        proof = torch.empty((), dtype=torch.int32, device=weights.device)
        _prove_identity_chunked[1,](
            seg,
            perm,
            proof,
            s,
            bs,
            seg.stride(0),
            perm.stride(0),
            False,
            2048,
            num_warps=4,
            num_stages=1,
        )
        _validated_gather_chunked[
            triton.cdiv(s, 256)
            * triton.cdiv(rank, min(64, triton.next_power_of_2(rank))),
        ](
            ids,
            weights,
            seg,
            widx,
            ranks,
            perm,
            out,
            proof,
            s,
            rank,
            bs,
            loras,
            vocab,
            ids.stride(0),
            seg.stride(0),
            widx.stride(0),
            ranks.stride(0),
            perm.stride(0),
            *weights.stride(),
            False,
            2048,
            256,
            min(64, triton.next_power_of_2(rank)),
            num_warps=4,
            num_stages=1,
        )
        return out
    presence = torch.empty((bs * s,), dtype=torch.int32, device=weights.device)
    if bs > 0:
        proof = torch.empty((), dtype=torch.int32, device=weights.device)
        _prove_unique_identity[1,](
            seg,
            perm,
            proof,
            s,
            bs,
            seg.stride(0),
            perm.stride(0),
            wide,
            128,
            num_warps=4,
            num_stages=1,
        )
        _fill_unique_presence[bs * s,](
            seg,
            perm,
            proof,
            presence,
            s,
            seg.stride(0),
            perm.stride(0),
            wide,
            128,
            num_warps=4,
            num_stages=1,
        )
    grid = (triton.cdiv(s, 128) * triton.cdiv(rank, 32),)
    _gather_presence_tiled[grid](
        ids,
        weights,
        widx,
        ranks,
        presence,
        out,
        s,
        rank,
        bs,
        loras,
        vocab,
        ids.stride(0),
        widx.stride(0),
        ranks.stride(0),
        *weights.stride(),
        wide,
        128,
        32,
        num_warps=4,
        num_stages=1,
    )
    return out


@triton.jit
def _zero_rebased(
    Out,
    M: tl.constexpr,
    K: tl.constexpr,
    SO: tl.constexpr,
    BLOCK: tl.constexpr,
):
    x = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row = x // K
    k = x % K
    tl.store(Out + row * SO + k, 0.0, x < M * K)


@triton.jit
def _ordered_rebased(
    Ids,
    Weights,
    Start,
    End,
    Widx,
    Rank,
    Perm,
    Out,
    S: tl.constexpr,
    R: tl.constexpr,
    L: tl.constexpr,
    V: tl.constexpr,
    ADAPTER: tl.constexpr,
    P0: tl.constexpr,
    PC: tl.constexpr,
    ROW0: tl.constexpr,
    ROWC: tl.constexpr,
    K0: tl.constexpr,
    KC: tl.constexpr,
    V0: tl.constexpr,
    VC: tl.constexpr,
    SI: tl.constexpr,
    SP: tl.constexpr,
    WR: tl.constexpr,
    WV: tl.constexpr,
    SO: tl.constexpr,
    BP: tl.constexpr,
    BK: tl.constexpr,
):
    local_row = tl.program_id(0) // tl.cdiv(KC, BK)
    k = tl.program_id(0) % tl.cdiv(KC, BK) * BK + tl.arange(0, BK)
    start = tl.load(Start).to(tl.int32)
    end = tl.load(End).to(tl.int32)
    wi = tl.load(Widx).to(tl.int32)
    wi = tl.where(wi < 0, wi + L, wi)
    rank = tl.load(Rank).to(tl.int32)
    member = tl.full((), 0, tl.int32)
    for chunk in range(tl.cdiv(PC, BP)):
        p = chunk * BP + tl.arange(0, BP)
        row = tl.load(Perm + tl.minimum(p, PC - 1) * SP).to(tl.int32)
        row = tl.where(row < 0, row + S, row)
        covered = (p < PC) & (p + P0 >= start) & (p + P0 < end)
        member |= tl.max((covered & (row == local_row + ROW0)).to(tl.int32), 0)
    token = tl.load(Ids + local_row * SI).to(tl.int32)
    token = tl.where(token < 0, token + V, token)
    local_token = token - V0
    active_row = (
        (member != 0)
        & (wi == ADAPTER)
        & (local_token >= 0)
        & (local_token < VC)
    )
    safe_token = tl.minimum(tl.maximum(local_token, 0), VC - 1)
    safe_k = tl.minimum(k, KC - 1)
    active = active_row & (k < KC) & (k + K0 < rank)
    value = tl.load(Weights + safe_k * WR + safe_token * WV, active, 0.0)
    tl.store(Out + local_row * SO + safe_k, value, active)


def _tile_weight_ranges(rank, vocab, rank_stride, vocab_stride, limit):
    pending = [(0, rank, 0, vocab)]
    while pending:
        k0, k1, v0, v1 = pending.pop()
        kr = (k1 - k0 - 1) * rank_stride
        vr = (v1 - v0 - 1) * vocab_stride
        if kr + vr <= limit:
            yield (k0, k1, v0, v1)
        elif kr >= vr and k1 - k0 > 1:
            mid = (k0 + k1) // 2
            pending.extend(((k0, mid, v0, v1), (mid, k1, v0, v1)))
        else:
            mid = (v0 + v1) // 2
            pending.extend(((k0, k1, v0, mid), (k0, k1, mid, v1)))


def _wide_rebased(ids, weights, seg, widx, ranks, perm, bs, out):
    s = ids.shape[0]
    loras, rank, vocab = weights.shape
    limit = 2147482623
    row_step = max(1, min(s, limit // max(rank, ids.stride(0), 1)))
    perm_step = max(1, min(s, limit // max(perm.stride(0), 1)))
    for row0 in range(0, s, row_step):
        rowc = min(row_step, s - row0)
        target = out.narrow(0, row0, rowc)
        _zero_rebased[triton.cdiv(rowc * rank, 128),](
            target,
            rowc,
            rank,
            rank if rowc > 1 else 0,
            128,
            num_warps=4,
            num_stages=1,
        )
    for b in range(bs):
        start = seg.narrow(0, b, 1)
        end = seg.narrow(0, b + 1, 1)
        index = widx.narrow(0, b, 1)
        for lora in range(loras):
            active_rank = ranks.narrow(0, lora, 1)
            for k0, k1, v0, v1 in _tile_weight_ranges(
                rank, vocab, weights.stride(1), weights.stride(2), limit
            ):
                wc = (
                    weights.narrow(0, lora, 1)
                    .narrow(1, k0, k1 - k0)
                    .narrow(2, v0, v1 - v0)
                )
                for row0 in range(0, s, row_step):
                    rowc = min(row_step, s - row0)
                    ic = ids.narrow(0, row0, rowc)
                    oc = out.narrow(0, row0, rowc).narrow(1, k0, k1 - k0)
                    for p0 in range(0, s, perm_step):
                        pc = min(perm_step, s - p0)
                        pv = perm.narrow(0, p0, pc)
                        _ordered_rebased[rowc * triton.cdiv(k1 - k0, 32),](
                            ic,
                            wc,
                            start,
                            end,
                            index,
                            active_rank,
                            pv,
                            oc,
                            s,
                            rank,
                            loras,
                            vocab,
                            lora,
                            p0,
                            pc,
                            row0,
                            rowc,
                            k0,
                            k1 - k0,
                            v0,
                            v1 - v0,
                            ids.stride(0) if rowc > 1 else 0,
                            perm.stride(0) if pc > 1 else 0,
                            weights.stride(1) if k1 - k0 > 1 else 0,
                            weights.stride(2) if v1 - v0 > 1 else 0,
                            rank if rowc > 1 else 0,
                            128,
                            32,
                            num_warps=4,
                            num_stages=1,
                        )
    return out


@triton.jit
def _or_membership(a, b):
    return a | b


@triton.jit
def _validated_gather_chunked(
    Ids,
    Weights,
    Seg,
    Widx,
    Ranks,
    Perm,
    Out,
    Proof,
    S: tl.constexpr,
    R: tl.constexpr,
    B: tl.constexpr,
    L: tl.constexpr,
    V: tl.constexpr,
    SI: tl.constexpr,
    SS: tl.constexpr,
    SW: tl.constexpr,
    SR: tl.constexpr,
    SP: tl.constexpr,
    WL: tl.constexpr,
    WR: tl.constexpr,
    WV: tl.constexpr,
    WIDE: tl.constexpr,
    CHUNK: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    si = tl.full((), SI, tl.int64) if WIDE else SI
    sp = tl.full((), SP, tl.int64) if WIDE else SP
    sr = tl.full((), SR, tl.int64) if WIDE else SR
    ss = tl.full((), SS, tl.int64) if WIDE else SS
    sw = tl.full((), SW, tl.int64) if WIDE else SW
    wl = tl.full((), WL, tl.int64) if WIDE else WL
    wr = tl.full((), WR, tl.int64) if WIDE else WR
    wv = tl.full((), WV, tl.int64) if WIDE else WV
    base = (
        tl.program_id(0).to(tl.int64 if WIDE else tl.int32)
        // tl.cdiv(R, BLOCK_R)
        * BLOCK_S
    )
    ks = tl.program_id(0).to(tl.int64 if WIDE else tl.int32) % tl.cdiv(
        R, BLOCK_R
    ) * BLOCK_R + tl.arange(0, BLOCK_R)
    first = tl.load(Seg)
    last = tl.load(Seg + B * ss)
    mismatch = tl.load(Proof)
    if mismatch == 0:
        rows = base + tl.arange(0, BLOCK_S)
        owner = tl.full((), 0, tl.int32)
        for b in range(1, B):
            owner += (base >= tl.load(Seg + b * ss)).to(tl.int32)
        end = tl.load(Seg + (owner + 1) * ss)
        if (
            (base >= first)
            & (base + BLOCK_S <= last)
            & (base + BLOCK_S <= end)
        ):
            uniform_wi = tl.load(Widx + owner * sw).to(
                tl.int64 if WIDE else tl.int32
            )
            uniform_wi = tl.where(uniform_wi < 0, uniform_wi + L, uniform_wi)
            uniform_rank = tl.load(Ranks + uniform_wi * sr)
            uniform_token = tl.load(Ids + rows * si).to(
                tl.int64 if WIDE else tl.int32
            )
            uniform_token = tl.where(
                uniform_token < 0, uniform_token + V, uniform_token
            )
            uniform_mask = (ks[None, :] < uniform_rank) & (ks[None, :] < R)
            uniform_value = tl.load(
                Weights
                + uniform_wi * wl
                + ks[None, :] * wr
                + uniform_token[:, None] * wv,
                uniform_mask,
                0.0,
            )
            tl.store(
                Out + rows[:, None] * R + ks[None, :],
                uniform_value,
                ks[None, :] < R,
            )
        else:
            present = (rows < S) & (rows >= first) & (rows < last)
            batch = tl.full((BLOCK_S,), 0, tl.int32)
            for b in range(1, B):
                batch += (rows >= tl.load(Seg + b * ss)).to(tl.int32)
            wi = tl.load(Widx + batch * sw, present, 0).to(
                tl.int64 if WIDE else tl.int32
            )
            wi = tl.where(wi < 0, wi + L, wi)
            active_rank = tl.load(Ranks + wi * sr, present, 0)
            token = tl.load(Ids + rows * si, present, 0).to(
                tl.int64 if WIDE else tl.int32
            )
            token = tl.where(token < 0, token + V, token)
            active = (
                present[:, None]
                & (ks[None, :] < active_rank[:, None])
                & (ks[None, :] < R)
            )
            value = tl.load(
                Weights
                + wi[:, None] * wl
                + ks[None, :] * wr
                + token[:, None] * wv,
                active,
                0.0,
            )
            tl.store(
                Out + rows[:, None] * R + ks[None, :],
                value,
                (rows[:, None] < S) & (ks[None, :] < R),
            )
    else:
        for j in range(BLOCK_S):
            row = base + j
            if row < S:
                members = tl.full((), 0, tl.uint32)
                for chunk in range(tl.cdiv(S, CHUNK)):
                    pos = chunk * CHUNK + tl.arange(0, CHUNK)
                    covered = (pos < S) & (pos >= first) & (pos < last)
                    mapped = tl.load(Perm + pos * sp, covered, 0).to(
                        tl.int64 if WIDE else tl.int32
                    )
                    mapped = tl.where(mapped < 0, mapped + S, mapped)
                    batches = tl.full((CHUNK,), 0, tl.int32)
                    for b in range(1, B):
                        batches += (pos >= tl.load(Seg + b * ss)).to(tl.int32)
                    bits = tl.full((CHUNK,), 1, tl.uint32) << batches.to(
                        tl.uint32
                    )
                    members |= tl.reduce(
                        tl.where(covered & (mapped == row), bits, 0),
                        0,
                        _or_membership,
                    )
                adapter = tl.full(
                    (BLOCK_R,), 0, tl.int64 if WIDE else tl.int32
                )
                found = tl.full((BLOCK_R,), False, tl.int1)
                for b in range(B):
                    wi = tl.load(Widx + b * sw).to(
                        tl.int64 if WIDE else tl.int32
                    )
                    wi = tl.where(wi < 0, wi + L, wi)
                    active_rank = tl.load(Ranks + wi * sr)
                    selected = (members >> b & 1 != 0) & (ks < active_rank)
                    adapter = tl.where(selected, wi, adapter)
                    found |= selected
                token = tl.load(Ids + row * si).to(
                    tl.int64 if WIDE else tl.int32
                )
                token = tl.where(token < 0, token + V, token)
                value = tl.load(
                    Weights + adapter * wl + ks * wr + token * wv,
                    found & (ks < R),
                    0.0,
                )
                tl.store(Out + row * R + ks, value, ks < R)


@triton.jit
def _prove_identity_chunked(
    Seg,
    Perm,
    Proof,
    S: tl.constexpr,
    B: tl.constexpr,
    SS: tl.constexpr,
    SP: tl.constexpr,
    WIDE: tl.constexpr,
    CHUNK: tl.constexpr,
):
    ss = tl.full((), SS, tl.int64) if WIDE else SS
    sp = tl.full((), SP, tl.int64) if WIDE else SP
    first = tl.load(Seg)
    last = tl.load(Seg + B * ss)
    mismatch = tl.full((), 0, tl.int32)
    for chunk in range(tl.cdiv(S, CHUNK)):
        pos = chunk * CHUNK + tl.arange(0, CHUNK)
        covered = (pos < S) & (pos >= first) & (pos < last)
        mapped = tl.load(Perm + pos * sp, covered, 0).to(
            tl.int64 if WIDE else tl.int32
        )
        mapped = tl.where(mapped < 0, mapped + S, mapped)
        mismatch += tl.sum((covered & (mapped != pos)).to(tl.int32), 0)
    tl.store(Proof, mismatch)


@triton.jit
def _prove_unique_identity(
    Seg,
    Perm,
    Proof,
    S: tl.constexpr,
    B: tl.constexpr,
    SS: tl.constexpr,
    SP: tl.constexpr,
    WIDE: tl.constexpr,
    CHUNK: tl.constexpr,
):
    ss = tl.full((), SS, tl.int64) if WIDE else SS
    sp = tl.full((), SP, tl.int64) if WIDE else SP
    first = tl.load(Seg)
    last = tl.load(Seg + B * ss)
    mismatch = tl.full((), 0, tl.int32)
    for chunk in range(tl.cdiv(S, CHUNK)):
        pos = chunk.to(tl.int64 if WIDE else tl.int32) * CHUNK + tl.arange(
            0, CHUNK
        )
        safe_pos = tl.minimum(pos, S - 1)
        mapped = tl.load(Perm + safe_pos * sp).to(
            tl.int64 if WIDE else tl.int32
        )
        mapped = tl.where(mapped < 0, mapped + S, mapped)
        covered = (pos < S) & (pos >= first) & (pos < last)
        mismatch |= tl.max((covered & (mapped != pos)).to(tl.int32), 0)
    tl.store(Proof, mismatch)


@triton.jit
def _fill_unique_presence(
    Seg,
    Perm,
    Proof,
    Presence,
    S: tl.constexpr,
    SS: tl.constexpr,
    SP: tl.constexpr,
    WIDE: tl.constexpr,
    CHUNK: tl.constexpr,
):
    ss = tl.full((), SS, tl.int64) if WIDE else SS
    sp = tl.full((), SP, tl.int64) if WIDE else SP
    b = tl.program_id(0).to(tl.int64 if WIDE else tl.int32) // S
    row = tl.program_id(0).to(tl.int64 if WIDE else tl.int32) % S
    start = tl.load(Seg + b * ss)
    end = tl.load(Seg + (b + 1) * ss)
    mismatch = tl.load(Proof)
    found = ((mismatch == 0) & (row >= start) & (row < end)).to(tl.int32)
    steps = tl.where(mismatch == 0, 0, tl.cdiv(S, CHUNK))
    for chunk in range(steps):
        pos = chunk.to(tl.int64 if WIDE else tl.int32) * CHUNK + tl.arange(
            0, CHUNK
        )
        safe_pos = tl.minimum(pos, S - 1)
        mapped = tl.load(Perm + safe_pos * sp).to(
            tl.int64 if WIDE else tl.int32
        )
        mapped = tl.where(mapped < 0, mapped + S, mapped)
        covered = (pos < S) & (pos >= start) & (pos < end)
        found |= tl.max((covered & (mapped == row)).to(tl.int32), 0)
    tl.store(Presence + b * S + row, found)


@triton.jit
def _inverse_small(
    Ids,
    Weights,
    Seg,
    Widx,
    Ranks,
    Perm,
    Out,
    S: tl.constexpr,
    R: tl.constexpr,
    B: tl.constexpr,
    L: tl.constexpr,
    V: tl.constexpr,
    SI: tl.constexpr,
    SS: tl.constexpr,
    SW: tl.constexpr,
    SR: tl.constexpr,
    SP: tl.constexpr,
    WL: tl.constexpr,
    WR: tl.constexpr,
    WV: tl.constexpr,
    WIDE: tl.constexpr,
    NP: tl.constexpr,
    BLOCK: tl.constexpr,
):
    si = tl.full((), SI, tl.int64) if WIDE else SI
    sp = tl.full((), SP, tl.int64) if WIDE else SP
    sr = tl.full((), SR, tl.int64) if WIDE else SR
    ss = tl.full((), SS, tl.int64) if WIDE else SS
    sw = tl.full((), SW, tl.int64) if WIDE else SW
    wl = tl.full((), WL, tl.int64) if WIDE else WL
    wr = tl.full((), WR, tl.int64) if WIDE else WR
    wv = tl.full((), WV, tl.int64) if WIDE else WV
    offsets = tl.program_id(0).to(
        tl.int64 if WIDE else tl.int32
    ) * BLOCK + tl.arange(0, BLOCK)
    row = offsets // R
    k = offsets % R
    valid = offsets < S * R
    if B == 0:
        value = tl.full((BLOCK,), 0, tl.float32)
    else:
        pos = tl.arange(0, NP)
        first = tl.load(Seg)
        last = tl.load(Seg + B * ss)
        covered = (pos < S) & (pos >= first) & (pos < last)
        batch = tl.full((NP,), 0, tl.int64 if WIDE else tl.int32)
        for b in range(1, B):
            boundary = tl.load(Seg + b * ss)
            batch += (pos >= boundary).to(tl.int64 if WIDE else tl.int32)
        mapped = tl.load(Perm + pos * sp, covered, 0).to(
            tl.int64 if WIDE else tl.int32
        )
        mapped = tl.where(mapped < 0, mapped + S, mapped)
        wi = tl.load(Widx + batch * sw, covered, 0).to(
            tl.int64 if WIDE else tl.int32
        )
        wi = tl.where(wi < 0, wi + L, wi)
        rank = tl.load(Ranks + wi * sr, covered, 0)
        match = (
            covered[None, :]
            & (mapped[None, :] == row[:, None])
            & (k[:, None] < rank[None, :])
        )
        owner = tl.max(tl.where(match, batch[None, :], -1), 1)
        active = valid & (owner >= 0)
        adapter = tl.load(Widx + owner * sw, active, 0).to(
            tl.int64 if WIDE else tl.int32
        )
        adapter = tl.where(adapter < 0, adapter + L, adapter)
        token = tl.load(Ids + row * si, active, 0).to(
            tl.int64 if WIDE else tl.int32
        )
        token = tl.where(token < 0, token + V, token)
        value = tl.load(
            Weights + adapter * wl + k * wr + token * wv, active, 0.0
        )
    tl.store(Out + offsets, value, valid)


__all__ = ["chunked_embedding_lora_a"]
