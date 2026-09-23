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
            *(
                ids.numel() * si,
                seg.numel() * ss,
                widx.numel() * sw,
                ranks.numel() * sr,
                perm.numel() * sp,
            ),
        )
        + 1024
        > 2147483647
    )
    grid = ((s * rank + 128 - 1) // 128,)
    if s <= 64 or bs == 0:
        if 0 < bs <= 32:
            _small_row_membership[(s + 4 - 1) // 4, (rank + 32 - 1) // 32](
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
                1 << (s - 1).bit_length(),
                4,
                32,
                num_warps=4,
                num_stages=1,
            )
        else:
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
                1 << (s - 1).bit_length() if bs else 1,
                128,
                num_warps=4,
                num_stages=1,
            )
    elif s <= 16384 and bs <= 32:
        if rank >= 64:
            block_s = 128
            gather_kernel = (
                _validated_gather_complete_words
                if weights.dtype in (torch.float16, torch.bfloat16)
                and wv == 1
                and (wr % 2 == 0)
                and (wl % 2 == 0)
                and (vocab % 2 == 0)
                and (weights.data_ptr() % 4 == 0)
                else _validated_gather_complete_stream
            )
            gather_kernel.run(
                (ids, weights, seg, widx, ranks, perm, out),
                (
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
                    wl,
                    wr,
                    wv,
                    wide,
                    1 << (s - 1).bit_length(),
                    block_s,
                    32,
                ),
                num_warps=4,
                num_stages=1,
                grid=((s + block_s - 1) // block_s,),
                warmup=False,
            )
        else:
            _validated_gather[(s + 64 - 1) // 64, (rank + 32 - 1) // 32](
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
                1 << (s - 1).bit_length(),
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
    local_pos = base + tl.arange(0, BLOCK_S)
    first = tl.load(Seg)
    last = tl.load(Seg + B * ss)
    local_covered = local_pos < S
    local_mapped = tl.load(Perm + local_pos * sp, local_covered, 0).to(
        tl.int64 if WIDE else tl.int32
    )
    local_mapped = tl.where(local_mapped < 0, local_mapped + S, local_mapped)
    mismatch = tl.sum(
        (local_covered & (local_mapped != local_pos)).to(tl.int32), 0
    )
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
        pos = tl.arange(0, NP)
        first = tl.load(Seg)
        last = tl.load(Seg + B * ss)
        covered = (pos < S) & (pos >= first) & (pos < last)
        mapped = tl.load(Perm + pos * sp, covered, 0).to(
            tl.int64 if WIDE else tl.int32
        )
        mapped = tl.where(mapped < 0, mapped + S, mapped)
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


@triton.jit
def _small_row_membership(
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
    BS: tl.constexpr,
    BR: tl.constexpr,
):
    si = tl.full((), SI, tl.int64) if WIDE else SI
    sp = tl.full((), SP, tl.int64) if WIDE else SP
    sr = tl.full((), SR, tl.int64) if WIDE else SR
    ss = tl.full((), SS, tl.int64) if WIDE else SS
    sw = tl.full((), SW, tl.int64) if WIDE else SW
    wl = tl.full((), WL, tl.int64) if WIDE else WL
    wr = tl.full((), WR, tl.int64) if WIDE else WR
    wv = tl.full((), WV, tl.int64) if WIDE else WV
    rows = tl.program_id(0).to(
        tl.int64 if WIDE else tl.int32
    ) * BS + tl.arange(0, BS)
    ks = tl.program_id(1).to(tl.int64 if WIDE else tl.int32) * BR + tl.arange(
        0, BR
    )
    valid = (rows[:, None] < S) & (ks[None, :] < R)
    if B == 0:
        value = tl.full((BS, BR), 0, tl.float32)
    else:
        pos = tl.arange(0, NP)
        first = tl.load(Seg)
        last = tl.load(Seg + B * ss)
        covered = (pos < S) & (pos >= first) & (pos < last)
        batch = tl.full((NP,), 0, tl.int32)
        for b in range(1, B):
            batch += (pos >= tl.load(Seg + b * ss)).to(tl.int32)
        mapped = tl.load(Perm + pos * sp, covered, 0).to(
            tl.int64 if WIDE else tl.int32
        )
        mapped = tl.where(mapped < 0, mapped + S, mapped)
        bits = tl.full((NP,), 1, tl.uint32) << batch.to(tl.uint32)
        members = tl.reduce(
            tl.where(
                covered[None, :] & (mapped[None, :] == rows[:, None]),
                bits[None, :],
                0,
            ),
            1,
            _or_membership,
        )
        adapter = tl.full((BS, BR), 0, tl.int64 if WIDE else tl.int32)
        found = tl.full((BS, BR), False, tl.int1)
        for b in range(B):
            wi = tl.load(Widx + b * sw).to(tl.int64 if WIDE else tl.int32)
            wi = tl.where(wi < 0, wi + L, wi)
            active_rank = tl.load(Ranks + wi * sr)
            selected = (members[:, None] >> b & 1 != 0) & (
                ks[None, :] < active_rank
            )
            adapter = tl.where(selected, wi, adapter)
            found |= selected
        token = tl.load(Ids + rows * si, rows < S, 0).to(
            tl.int64 if WIDE else tl.int32
        )
        token = tl.where(token < 0, token + V, token)
        value = tl.load(
            Weights + adapter * wl + ks[None, :] * wr + token[:, None] * wv,
            valid & found,
            0.0,
        )
    tl.store(Out + rows[:, None] * R + ks[None, :], value, valid)


@triton.jit
def _validated_gather_complete_stream(PTRS, META: tl.constexpr):
    Ids = PTRS[0]
    Weights = PTRS[1]
    Seg = PTRS[2]
    Widx = PTRS[3]
    Ranks = PTRS[4]
    Perm = PTRS[5]
    Out = PTRS[6]
    S: tl.constexpr = META[0]
    R: tl.constexpr = META[1]
    B: tl.constexpr = META[2]
    L: tl.constexpr = META[3]
    V: tl.constexpr = META[4]
    SI: tl.constexpr = META[5]
    SS: tl.constexpr = META[6]
    SW: tl.constexpr = META[7]
    SR: tl.constexpr = META[8]
    SP: tl.constexpr = META[9]
    WL: tl.constexpr = META[10]
    WR: tl.constexpr = META[11]
    WV: tl.constexpr = META[12]
    WIDE: tl.constexpr = META[13]
    BLOCK_S: tl.constexpr = META[15]
    BLOCK_R: tl.constexpr = META[16]
    si = tl.full((), SI, tl.int64) if WIDE else SI
    sp = tl.full((), SP, tl.int64) if WIDE else SP
    sr = tl.full((), SR, tl.int64) if WIDE else SR
    ss = tl.full((), SS, tl.int64) if WIDE else SS
    sw = tl.full((), SW, tl.int64) if WIDE else SW
    wl = tl.full((), WL, tl.int64) if WIDE else WL
    wr = tl.full((), WR, tl.int64) if WIDE else WR
    wv = tl.full((), WV, tl.int64) if WIDE else WV
    base = tl.program_id(0).to(tl.int64 if WIDE else tl.int32) * BLOCK_S
    first = tl.load(Seg)
    last = tl.load(Seg + B * ss)
    local_pos = base + tl.arange(0, BLOCK_S)
    local_covered = local_pos < S
    local_mapped = tl.load(Perm + local_pos * sp, local_covered, 0).to(
        tl.int64 if WIDE else tl.int32
    )
    local_mapped = tl.where(local_mapped < 0, local_mapped + S, local_mapped)
    mismatch = tl.sum(
        (local_covered & (local_mapped != local_pos)).to(tl.int32), 0
    )
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
        for rank_start in range(0, R, BLOCK_R):
            ks = rank_start + tl.arange(0, BLOCK_R)
            active = (
                present[:, None]
                & (ks[None, :] < active_rank[:, None])
                & (ks[None, :] < R)
                & ((token[:, None] >= 0) & (token[:, None] < V))
            )
            value = tl.load(
                Weights
                + wi[:, None] * wl
                + ks[None, :] * wr
                + token[:, None] * wv,
                active,
                0.0,
                cache_modifier=".cg",
            )
            tl.store(
                Out + rows[:, None] * R + ks[None, :],
                value,
                (rows[:, None] < S) & (ks[None, :] < R),
            )
    else:
        for j in range(BLOCK_S // 4):
            inv_rows = base + j * 4 + tl.arange(0, 4)
            inv_valid = inv_rows < S
            inv_members = tl.full((4,), 0, tl.uint32)
            for chunk_start in range(0, S, 256):
                pos = chunk_start + tl.arange(0, 256)
                covered = (pos < S) & (pos >= first) & (pos < last)
                mapped = tl.load(Perm + pos * sp, covered, 0).to(
                    tl.int64 if WIDE else tl.int32
                )
                mapped = tl.where(mapped < 0, mapped + S, mapped)
                batches = tl.full((256,), 0, tl.int32)
                for b in range(1, B):
                    batches += (pos >= tl.load(Seg + b * ss)).to(tl.int32)
                bits = tl.full((256,), 1, tl.uint32) << batches.to(tl.uint32)
                inv_members |= tl.reduce(
                    tl.where(
                        covered[None, :]
                        & (mapped[None, :] == inv_rows[:, None]),
                        bits[None, :],
                        0,
                    ),
                    1,
                    _or_membership,
                )
            for rank_start in range(0, R, BLOCK_R):
                ks = rank_start + tl.arange(0, BLOCK_R)
                inv_adapter = tl.full(
                    (4, BLOCK_R), 0, tl.int64 if WIDE else tl.int32
                )
                inv_found = tl.full((4, BLOCK_R), False, tl.int1)
                for b in range(B):
                    inv_wi = tl.load(Widx + b * sw).to(
                        tl.int64 if WIDE else tl.int32
                    )
                    inv_wi = tl.where(inv_wi < 0, inv_wi + L, inv_wi)
                    inv_rank = tl.load(Ranks + inv_wi * sr)
                    inv_selected = (inv_members[:, None] >> b & 1 != 0) & (
                        ks[None, :] < inv_rank
                    )
                    inv_adapter = tl.where(inv_selected, inv_wi, inv_adapter)
                    inv_found |= inv_selected
                inv_token = tl.load(Ids + inv_rows * si, inv_valid, 0).to(
                    tl.int64 if WIDE else tl.int32
                )
                inv_token = tl.where(inv_token < 0, inv_token + V, inv_token)
                inv_value = tl.load(
                    Weights
                    + inv_adapter * wl
                    + ks[None, :] * wr
                    + inv_token[:, None] * wv,
                    inv_found & inv_valid[:, None] & (ks[None, :] < R),
                    0.0,
                    cache_modifier=".cg",
                )
                tl.store(
                    Out + inv_rows[:, None] * R + ks[None, :],
                    inv_value,
                    inv_valid[:, None] & (ks[None, :] < R),
                )


@triton.jit
def _weight_word(Weights, offsets, mask, CACHE: tl.constexpr):
    word_pointer = Weights.to(tl.pointer_type(tl.int32))
    raw = tl.load(
        word_pointer + offsets // 2, mask, 0, cache_modifier=CACHE
    ).to(tl.uint32)
    lane = raw >> offsets % 2 * 16 & 65535
    return lane.to(tl.uint16).to(Weights.dtype.element_ty, bitcast=True)


@triton.jit
def _validated_gather_complete_words(PTRS, META: tl.constexpr):
    Ids = PTRS[0]
    Weights = PTRS[1]
    Seg = PTRS[2]
    Widx = PTRS[3]
    Ranks = PTRS[4]
    Perm = PTRS[5]
    Out = PTRS[6]
    S: tl.constexpr = META[0]
    R: tl.constexpr = META[1]
    B: tl.constexpr = META[2]
    L: tl.constexpr = META[3]
    V: tl.constexpr = META[4]
    SI: tl.constexpr = META[5]
    SS: tl.constexpr = META[6]
    SW: tl.constexpr = META[7]
    SR: tl.constexpr = META[8]
    SP: tl.constexpr = META[9]
    WL: tl.constexpr = META[10]
    WR: tl.constexpr = META[11]
    WV: tl.constexpr = META[12]
    WIDE: tl.constexpr = META[13]
    BLOCK_S: tl.constexpr = META[15]
    BLOCK_R: tl.constexpr = META[16]
    si = tl.full((), SI, tl.int64) if WIDE else SI
    sp = tl.full((), SP, tl.int64) if WIDE else SP
    sr = tl.full((), SR, tl.int64) if WIDE else SR
    ss = tl.full((), SS, tl.int64) if WIDE else SS
    sw = tl.full((), SW, tl.int64) if WIDE else SW
    wl = tl.full((), WL, tl.int64) if WIDE else WL
    wr = tl.full((), WR, tl.int64) if WIDE else WR
    wv = tl.full((), WV, tl.int64) if WIDE else WV
    base = tl.program_id(0).to(tl.int64 if WIDE else tl.int32) * BLOCK_S
    first = tl.load(Seg)
    last = tl.load(Seg + B * ss)
    local_pos = base + tl.arange(0, BLOCK_S)
    local_covered = local_pos < S
    local_mapped = tl.load(Perm + local_pos * sp, local_covered, 0).to(
        tl.int64 if WIDE else tl.int32
    )
    local_mapped = tl.where(local_mapped < 0, local_mapped + S, local_mapped)
    mismatch = tl.sum(
        (local_covered & (local_mapped != local_pos)).to(tl.int32), 0
    )
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
        for rank_start in range(0, R, BLOCK_R):
            ks = rank_start + tl.arange(0, BLOCK_R)
            active = (
                present[:, None]
                & (ks[None, :] < active_rank[:, None])
                & (ks[None, :] < R)
                & ((token[:, None] >= 0) & (token[:, None] < V))
            )
            value = _weight_word(
                Weights,
                0 + wi[:, None] * wl + ks[None, :] * wr + token[:, None] * wv,
                active,
                ".cg",
            )
            tl.store(
                Out + rows[:, None] * R + ks[None, :],
                value,
                (rows[:, None] < S) & (ks[None, :] < R),
            )
    else:
        for j in range(BLOCK_S // 4):
            inv_rows = base + j * 4 + tl.arange(0, 4)
            inv_valid = inv_rows < S
            inv_members = tl.full((4,), 0, tl.uint32)
            for chunk_start in range(0, S, 256):
                pos = chunk_start + tl.arange(0, 256)
                covered = (pos < S) & (pos >= first) & (pos < last)
                mapped = tl.load(Perm + pos * sp, covered, 0).to(
                    tl.int64 if WIDE else tl.int32
                )
                mapped = tl.where(mapped < 0, mapped + S, mapped)
                batches = tl.full((256,), 0, tl.int32)
                for b in range(1, B):
                    batches += (pos >= tl.load(Seg + b * ss)).to(tl.int32)
                bits = tl.full((256,), 1, tl.uint32) << batches.to(tl.uint32)
                inv_members |= tl.reduce(
                    tl.where(
                        covered[None, :]
                        & (mapped[None, :] == inv_rows[:, None]),
                        bits[None, :],
                        0,
                    ),
                    1,
                    _or_membership,
                )
            for rank_start in range(0, R, BLOCK_R):
                ks = rank_start + tl.arange(0, BLOCK_R)
                inv_adapter = tl.full(
                    (4, BLOCK_R), 0, tl.int64 if WIDE else tl.int32
                )
                inv_found = tl.full((4, BLOCK_R), False, tl.int1)
                for b in range(B):
                    inv_wi = tl.load(Widx + b * sw).to(
                        tl.int64 if WIDE else tl.int32
                    )
                    inv_wi = tl.where(inv_wi < 0, inv_wi + L, inv_wi)
                    inv_rank = tl.load(Ranks + inv_wi * sr)
                    inv_selected = (inv_members[:, None] >> b & 1 != 0) & (
                        ks[None, :] < inv_rank
                    )
                    inv_adapter = tl.where(inv_selected, inv_wi, inv_adapter)
                    inv_found |= inv_selected
                inv_token = tl.load(Ids + inv_rows * si, inv_valid, 0).to(
                    tl.int64 if WIDE else tl.int32
                )
                inv_token = tl.where(inv_token < 0, inv_token + V, inv_token)
                inv_value = _weight_word(
                    Weights,
                    0
                    + inv_adapter * wl
                    + ks[None, :] * wr
                    + inv_token[:, None] * wv,
                    inv_found & inv_valid[:, None] & (ks[None, :] < R),
                    ".cg",
                )
                tl.store(
                    Out + inv_rows[:, None] * R + ks[None, :],
                    inv_value,
                    inv_valid[:, None] & (ks[None, :] < R),
                )


__all__ = ["chunked_embedding_lora_a"]
