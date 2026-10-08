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
def _init_owner(
    Owner, N: tl.constexpr, WIDE: tl.constexpr, BLOCK: tl.constexpr
):
    offsets = tl.program_id(0).to(
        tl.int64 if WIDE else tl.int32
    ) * BLOCK + tl.arange(0, BLOCK)
    tl.store(Owner + offsets, -1, offsets < N)


@triton.jit
def _assign_owner(
    Seg,
    Widx,
    Ranks,
    Perm,
    Owner,
    S: tl.constexpr,
    R: tl.constexpr,
    B: tl.constexpr,
    L: tl.constexpr,
    SS: tl.constexpr,
    SI: tl.constexpr,
    SR: tl.constexpr,
    SP: tl.constexpr,
    WIDE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    si = tl.full((), SI, tl.int64) if WIDE else SI
    sp = tl.full((), SP, tl.int64) if WIDE else SP
    sr = tl.full((), SR, tl.int64) if WIDE else SR
    ss = tl.full((), SS, tl.int64) if WIDE else SS
    offsets = tl.program_id(0).to(
        tl.int64 if WIDE else tl.int32
    ) * BLOCK + tl.arange(0, BLOCK)
    pos = offsets // R
    rank_pos = offsets % R
    first = tl.load(Seg)
    last = tl.load(Seg + B * ss)
    valid = (pos < S) & (pos >= first) & (pos < last)
    batch = tl.full((BLOCK,), 0, Owner.dtype.element_ty)
    for b in range(1, B):
        boundary = tl.load(Seg + b * ss)
        batch += (pos >= boundary).to(Owner.dtype.element_ty)
    wi = tl.load(Widx + batch * si, valid, 0).to(
        tl.int64 if WIDE else tl.int32
    )
    wi = tl.where(wi < 0, wi + L, wi)
    active_rank = tl.load(Ranks + wi * sr, valid, 0).to(
        tl.int64 if WIDE else tl.int32
    )
    active = valid & (rank_pos < active_rank)
    row = tl.load(Perm + pos * sp, active, 0).to(
        tl.int64 if WIDE else tl.int32
    )
    row = tl.where(row < 0, row + S, row)
    tl.atomic_max(Owner + row * R + rank_pos, batch, active, sem="relaxed")


@triton.jit
def _gather_owned(
    Ids,
    Weights,
    Widx,
    Owner,
    Out,
    S: tl.constexpr,
    R: tl.constexpr,
    V: tl.constexpr,
    L: tl.constexpr,
    SI: tl.constexpr,
    SW: tl.constexpr,
    WL: tl.constexpr,
    WR: tl.constexpr,
    WV: tl.constexpr,
    WIDE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    si = tl.full((), SI, tl.int64) if WIDE else SI
    sw = tl.full((), SW, tl.int64) if WIDE else SW
    wl = tl.full((), WL, tl.int64) if WIDE else WL
    wr = tl.full((), WR, tl.int64) if WIDE else WR
    wv = tl.full((), WV, tl.int64) if WIDE else WV
    offsets = tl.program_id(0).to(
        tl.int64 if WIDE else tl.int32
    ) * BLOCK + tl.arange(0, BLOCK)
    valid = offsets < S * R
    row = offsets // R
    rank_pos = offsets % R
    batch = tl.load(Owner + offsets, valid, -1)
    active = valid & (batch >= 0)
    wi = tl.load(Widx + batch * sw, active, 0).to(
        tl.int64 if WIDE else tl.int32
    )
    wi = tl.where(wi < 0, wi + L, wi)
    token = tl.load(Ids + row * si, active, 0).to(
        tl.int64 if WIDE else tl.int32
    )
    token = tl.where(token < 0, token + V, token)
    value = tl.load(
        Weights + wi * wl + rank_pos * wr + token * wv, active, 0.0
    )
    tl.store(Out + offsets, value, valid)


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


def chunked_embedding_lora_a(input_ids, weights, batch_info, vocab_size):
    s = input_ids.shape[0]
    loras, rank, vocab = weights.shape
    out = torch.empty((s, rank), dtype=weights.dtype, device=weights.device)
    if s == 0 or rank == 0:
        return out
    ids = input_ids
    seg = batch_info.seg_indptr
    widx = batch_info.weight_indices
    ranks = batch_info.lora_ranks
    perm = batch_info.permutation
    bs = int(batch_info.bs)
    si = ids.stride(0)
    ss = seg.stride(0)
    sw = widx.stride(0)
    sr = ranks.stride(0)
    sp = perm.stride(0)
    wl, wr, wv = weights.stride()
    wide = (
        max(
            s * rank,
            s,
            rank,
            loras,
            vocab,
            bs,
            (loras - 1) * wl + (rank - 1) * wr + (vocab - 1) * wv,
            s * si,
            (bs + 1) * ss,
            bs * sw,
            loras * sr,
            s * sp,
        )
        + 1024
        > 2147483647
    )
    grid = (triton.cdiv(s * rank, 128),)
    if s <= 64 or bs == 0:
        _inverse_small[grid](
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
            si,
            ss,
            sw,
            sr,
            sp,
            *(wl, wr, wv),
            wide,
            triton.next_power_of_2(s) if bs else 1,
            128,
            num_warps=4,
            num_stages=1,
        )
    elif s <= 16384 and bs <= 32:
        _validated_gather[triton.cdiv(s, 64), triton.cdiv(rank, 32)](
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
            si,
            ss,
            sw,
            sr,
            sp,
            *(wl, wr, wv),
            wide,
            triton.next_power_of_2(s),
            64,
            32,
            num_warps=4,
            num_stages=1,
        )
    else:
        owner = torch.empty(
            (s, rank),
            dtype=torch.int64 if bs > 2147483647 else torch.int32,
            device=weights.device,
        )
        _init_owner[grid](
            owner, s * rank, wide, 128, num_warps=4, num_stages=1
        )
        _assign_owner[grid](
            seg,
            widx,
            ranks,
            perm,
            owner,
            s,
            rank,
            bs,
            loras,
            ss,
            sw,
            sr,
            sp,
            wide,
            128,
            num_warps=4,
            num_stages=1,
        )
        _gather_owned[grid](
            ids,
            weights,
            widx,
            owner,
            out,
            s,
            rank,
            vocab,
            loras,
            si,
            sw,
            *(wl, wr, wv),
            wide,
            128,
            num_warps=4,
            num_stages=1,
        )
    return out


@triton.jit
def _or_membership(a, b):
    return a | b


@triton.jit
def _validated_gather(
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
    base = tl.program_id(0).to(tl.int64 if WIDE else tl.int32) * BLOCK_S
    ks = tl.program_id(1) * BLOCK_R + tl.arange(0, BLOCK_R)
    pos = tl.arange(0, NP)
    first = tl.load(Seg)
    last = tl.load(Seg + B * ss)
    covered = (pos < S) & (pos >= first) & (pos < last)
    mapped = tl.load(Perm + pos * sp, covered, 0).to(
        tl.int64 if WIDE else tl.int32
    )
    mapped = tl.where(mapped < 0, mapped + S, mapped)
    mismatch = tl.sum((covered & (mapped != pos)).to(tl.int32), 0)
    if mismatch == 0:
        rows = base + tl.arange(0, BLOCK_S)
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
        batches = tl.full((NP,), 0, tl.int32)
        for b in range(1, B):
            batches += (pos >= tl.load(Seg + b * ss)).to(tl.int32)
        bits = tl.full((NP,), 1, tl.uint32) << batches.to(tl.uint32)
        for j in range(BLOCK_S):
            row = base + j
            if row < S:
                members = tl.reduce(
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


__all__ = ["chunked_embedding_lora_a"]
