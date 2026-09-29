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


@triton.jit(do_not_specialize=["Owner"])
def _init_owner(Owner, META: tl.constexpr):
    N: tl.constexpr = META[0]
    WIDE: tl.constexpr = META[1]
    BLOCK: tl.constexpr = META[2]
    offsets = tl.program_id(0).to(
        tl.int64 if WIDE else tl.int32
    ) * BLOCK + tl.arange(0, BLOCK)
    tl.store(Owner + offsets, -1, offsets < N)


@triton.jit(do_not_specialize=["Seg", "Widx", "Ranks", "Perm", "Owner"])
def _assign_owner(Seg, Widx, Ranks, Perm, Owner, META: tl.constexpr):
    S: tl.constexpr = META[0]
    R: tl.constexpr = META[1]
    B: tl.constexpr = META[2]
    L: tl.constexpr = META[3]
    SS: tl.constexpr = META[4]
    SI: tl.constexpr = META[5]
    SR: tl.constexpr = META[6]
    SP: tl.constexpr = META[7]
    WIDE: tl.constexpr = META[8]
    BLOCK: tl.constexpr = META[9]
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


@triton.jit(do_not_specialize=["Ids", "Weights", "Widx", "Owner", "Out"])
def _gather_owned(Ids, Weights, Widx, Owner, Out, META: tl.constexpr):
    S: tl.constexpr = META[0]
    R: tl.constexpr = META[1]
    V: tl.constexpr = META[2]
    L: tl.constexpr = META[3]
    SI: tl.constexpr = META[4]
    SW: tl.constexpr = META[5]
    WL: tl.constexpr = META[6]
    WR: tl.constexpr = META[7]
    WV: tl.constexpr = META[8]
    WIDE: tl.constexpr = META[9]
    BLOCK: tl.constexpr = META[10]
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


@triton.jit(
    do_not_specialize=["Ids", "Weights", "Seg", "Widx", "Ranks", "Perm", "Out"]
)
def _inverse_small(
    Ids, Weights, Seg, Widx, Ranks, Perm, Out, META: tl.constexpr
):
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
    NP: tl.constexpr = META[14]
    BLOCK: tl.constexpr = META[15]
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
        mapped = tl.load(Perm + tl.minimum(pos, S - 1) * sp, covered, 0).to(
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
        adapter = tl.load(Widx + tl.maximum(owner, 0) * sw, active, 0).to(
            tl.int64 if WIDE else tl.int32
        )
        adapter = tl.where(adapter < 0, adapter + L, adapter)
        token = tl.load(Ids + tl.minimum(row, S - 1) * si, active, 0).to(
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
    out = weights.new_empty((s, rank), dtype=weights.dtype)
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
            loras * rank * vocab,
        )
        + 1024
        > 2147483647
    )
    grid = ((s * rank + 128 - 1) // 128,)
    if s <= 64 or bs == 0:
        small_block = (
            min(32, 1024 // (1 << (s - 1).bit_length())) if bs else 32
        )
        _inverse_small.run(
            ids,
            weights,
            seg,
            widx,
            ranks,
            perm,
            out,
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
                1 << (s - 1).bit_length() if bs else 1,
                small_block,
            ),
            num_warps=4,
            num_stages=1,
            grid=((s * rank + small_block - 1) // small_block,),
            warmup=False,
        )
    elif (
        s <= 16384
        and bs <= 32
        and (si == 1)
        and (ss == 1)
        and (sw == 1)
        and (sr == 1)
        and (sp == 1)
        and (wv == 1)
        and (wr == vocab)
        and (wl == rank * vocab)
    ) and (rank % 32 == 0 and vocab % 32 == 0):
        gather_weights = weights
        gather_strides = (wl, wr, wv)
        proof_offset = 0
        proof_count = min(40, triton.cdiv(s, 128))
        proof = weights
        direct_rows = 64 if rank <= 32 else 128
        _validated_gather_chunked.run(
            ids,
            gather_weights,
            seg,
            widx,
            ranks,
            perm,
            out,
            proof,
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
                gather_strides[0],
                gather_strides[1],
                gather_strides[2],
                wide,
                128,
                direct_rows,
                16,
                proof_offset,
                proof_count,
            ),
            num_warps=1,
            num_stages=2,
            grid=(40,),
            warmup=False,
        )
    else:
        owner = weights.new_empty(
            (s, rank), dtype=torch.int64 if bs > 2147483647 else torch.int32
        )
        _init_owner.run(
            owner,
            (s * rank, wide, 128),
            num_warps=4,
            num_stages=1,
            grid=grid,
            warmup=False,
        )
        _assign_owner.run(
            seg,
            widx,
            ranks,
            perm,
            owner,
            (s, rank, bs, loras, ss, sw, sr, sp, wide, 128),
            num_warps=4,
            num_stages=1,
            grid=grid,
            warmup=False,
        )
        _gather_owned.run(
            ids,
            weights,
            widx,
            owner,
            out,
            (s, rank, vocab, loras, si, sw, wl, wr, wv, wide, 128),
            num_warps=4,
            num_stages=1,
            grid=grid,
            warmup=False,
        )
    return out


@triton.jit
def _or_membership(a, b):
    return a | b


@triton.jit(
    do_not_specialize=[
        "Ids",
        "Weights",
        "Seg",
        "Widx",
        "Ranks",
        "Perm",
        "Out",
        "Proof",
    ]
)
def _validated_gather_chunked(
    Ids, Weights, Seg, Widx, Ranks, Perm, Out, Proof, META: tl.constexpr
):
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
    CHUNK: tl.constexpr = META[14]
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
    first = tl.load(Seg)
    last = tl.load(Seg + B * ss)
    for tile in range(
        tl.program_id(0),
        tl.cdiv(S, BLOCK_S) * tl.cdiv(R, BLOCK_R),
        tl.num_programs(0),
    ):
        row_program = tile // tl.cdiv(R, BLOCK_R)
        rank_program = tile % tl.cdiv(R, BLOCK_R)
        base = row_program.to(tl.int64 if WIDE else tl.int32) * BLOCK_S
        ks = rank_program * BLOCK_R + tl.arange(0, BLOCK_R)
        local_pos = base + tl.arange(0, BLOCK_S)
        local_mapped = tl.load(Perm + tl.minimum(local_pos, S - 1) * sp).to(
            tl.int64 if WIDE else tl.int32
        )
        local_mapped = tl.where(
            local_mapped < 0, local_mapped + S, local_mapped
        )
        mismatch = tl.sum(
            ((local_pos < S) & (local_mapped != local_pos)).to(tl.int32), 0
        )
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
                uniform_wi = tl.where(
                    uniform_wi < 0, uniform_wi + L, uniform_wi
                )
                uniform_rank = tl.load(Ranks + uniform_wi * sr)
                uniform_token = tl.load(Ids + rows * si).to(
                    tl.int64 if WIDE else tl.int32
                )
                uniform_token = tl.where(
                    uniform_token < 0, uniform_token + V, uniform_token
                )
                if (uniform_rank >= (rank_program + 1) * BLOCK_R) & (
                    (rank_program + 1) * BLOCK_R <= R
                ):
                    uniform_value = tl.load(
                        Weights
                        + uniform_wi * wl
                        + ks[None, :] * wr
                        + uniform_token[:, None] * wv
                    )
                    tl.store(
                        Out + rows[:, None] * R + ks[None, :], uniform_value
                    )
                else:
                    for sub in range(0, BLOCK_S, 32):
                        partial_rows = base + sub + tl.arange(0, 32)
                        partial_token = tl.load(Ids + partial_rows * si).to(
                            tl.int64 if WIDE else tl.int32
                        )
                        partial_token = tl.where(
                            partial_token < 0, partial_token + V, partial_token
                        )
                        partial_mask = (ks[None, :] < uniform_rank) & (
                            ks[None, :] < R
                        )
                        partial_offset = (
                            uniform_wi * wl
                            + ks[None, :] * wr
                            + partial_token[:, None] * wv
                        )
                        partial_loaded = tl.load(
                            Weights + tl.where(partial_mask, partial_offset, 0)
                        )
                        partial_integer: tl.constexpr = (
                            tl.int32
                            if Weights.dtype.element_ty == tl.float32
                            else tl.int16
                        )
                        partial_bits = partial_loaded.to(
                            partial_integer, bitcast=True
                        )
                        partial_value = tl.where(
                            partial_mask,
                            partial_bits,
                            tl.full((), 0, partial_integer),
                        ).to(Weights.dtype.element_ty, bitcast=True)
                        tl.store(
                            Out + partial_rows[:, None] * R + ks[None, :],
                            partial_value,
                            ks[None, :] < R,
                        )
            else:
                for j in range(BLOCK_S):
                    row = base + j
                    if row < S:
                        members = tl.full((), 0, tl.uint32)
                        for chunk in range(tl.cdiv(S, CHUNK)):
                            pos = chunk * CHUNK + tl.arange(0, CHUNK)
                            covered = (pos < S) & (pos >= first) & (pos < last)
                            mapped = tl.load(
                                Perm + tl.minimum(pos, S - 1) * sp
                            ).to(tl.int64 if WIDE else tl.int32)
                            mapped = tl.where(mapped < 0, mapped + S, mapped)
                            batches = tl.full((CHUNK,), 0, tl.int32)
                            for b in range(1, B):
                                batches += (pos >= tl.load(Seg + b * ss)).to(
                                    tl.int32
                                )
                            bits = tl.full(
                                (CHUNK,), 1, tl.uint32
                            ) << batches.to(tl.uint32)
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
                            selected = (members >> b & 1 != 0) & (
                                ks < active_rank
                            )
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
        else:
            for j in range(BLOCK_S):
                row = base + j
                if row < S:
                    members = tl.full((), 0, tl.uint32)
                    for chunk in range(tl.cdiv(S, CHUNK)):
                        pos = chunk * CHUNK + tl.arange(0, CHUNK)
                        covered = (pos < S) & (pos >= first) & (pos < last)
                        mapped = tl.load(
                            Perm + tl.minimum(pos, S - 1) * sp
                        ).to(tl.int64 if WIDE else tl.int32)
                        mapped = tl.where(mapped < 0, mapped + S, mapped)
                        batches = tl.full((CHUNK,), 0, tl.int32)
                        for b in range(1, B):
                            batches += (pos >= tl.load(Seg + b * ss)).to(
                                tl.int32
                            )
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


@triton.jit(do_not_specialize=["Seg", "Perm", "Proof"])
def _prove_identity_chunked(Seg, Perm, Proof, META: tl.constexpr):
    S: tl.constexpr = META[0]
    B: tl.constexpr = META[1]
    SS: tl.constexpr = META[2]
    SP: tl.constexpr = META[3]
    WIDE: tl.constexpr = META[4]
    CHUNK: tl.constexpr = META[5]
    ss = tl.full((), SS, tl.int64) if WIDE else SS
    sp = tl.full((), SP, tl.int64) if WIDE else SP
    first = tl.load(Seg)
    last = tl.load(Seg + B * ss)
    mismatch = tl.full((), 0, tl.int32)
    for chunk in range(
        tl.program_id(0), tl.cdiv(S, CHUNK), tl.num_programs(0)
    ):
        pos = chunk * CHUNK + tl.arange(0, CHUNK)
        covered = (pos < S) & (pos >= first) & (pos < last)
        mapped = tl.load(Perm + tl.minimum(pos, S - 1) * sp).to(
            tl.int64 if WIDE else tl.int32
        )
        mapped = tl.where(mapped < 0, mapped + S, mapped)
        mismatch += tl.sum((covered & (mapped != pos)).to(tl.int32), 0)
    tl.store(Proof + tl.program_id(0), mismatch)


@triton.jit(do_not_specialize=["Weights", "Packed", "Seg", "Perm", "Proof"])
def _repack_weights(Weights, Packed, Seg, Perm, Proof, META: tl.constexpr):
    L: tl.constexpr = META[0]
    R: tl.constexpr = META[1]
    V: tl.constexpr = META[2]
    WL: tl.constexpr = META[3]
    WR: tl.constexpr = META[4]
    WV: tl.constexpr = META[5]
    WIDE: tl.constexpr = META[6]
    BR: tl.constexpr = META[7]
    BV: tl.constexpr = META[8]
    S: tl.constexpr = META[9]
    B: tl.constexpr = META[10]
    SS: tl.constexpr = META[11]
    SP: tl.constexpr = META[12]
    CHUNK: tl.constexpr = META[13]
    wl = tl.full((), WL, tl.int64) if WIDE else WL
    wr = tl.full((), WR, tl.int64) if WIDE else WR
    wv = tl.full((), WV, tl.int64) if WIDE else WV
    nr = tl.cdiv(R, BR)
    nv = tl.cdiv(V, BV)
    for tile in range(
        tl.program_id(0).to(tl.int64 if WIDE else tl.int32),
        L * nr * nv,
        2 * tl.num_programs(0),
    ):
        adapter = tile // (nr * nv)
        rank = tile // nv % nr * BR + tl.arange(0, BR)
        token = tile % nv * BV + tl.arange(0, BV)
        valid = (rank[:, None] < R) & (token[None, :] < V)
        value = tl.load(
            Weights + adapter * wl + rank[:, None] * wr + token[None, :] * wv,
            valid,
            0.0,
        )
        tile_second = tile + tl.num_programs(0)
        if tile_second < L * nr * nv:
            adapter_second = tile_second // (nr * nv)
            rank_second = tile_second // nv % nr * BR + tl.arange(0, BR)
            token_second = tile_second % nv * BV + tl.arange(0, BV)
            valid_second = (rank_second[:, None] < R) & (
                token_second[None, :] < V
            )
            value_second = tl.load(
                Weights
                + adapter_second * wl
                + rank_second[:, None] * wr
                + token_second[None, :] * wv,
                valid_second,
                0.0,
            )
            tl.store(
                Packed + (adapter * V + token[None, :]) * R + rank[:, None],
                value,
                valid,
            )
            tl.store(
                Packed
                + (adapter_second * V + token_second[None, :]) * R
                + rank_second[:, None],
                value_second,
                valid_second,
            )
        else:
            tl.store(
                Packed + (adapter * V + token[None, :]) * R + rank[:, None],
                value,
                valid,
            )
    ss = tl.full((), SS, tl.int64) if WIDE else SS
    sp = tl.full((), SP, tl.int64) if WIDE else SP
    first = tl.load(Seg)
    last = tl.load(Seg + B * ss)
    mismatch = tl.full((), 0, tl.int32)
    for chunk in range(
        tl.program_id(0), tl.cdiv(S, CHUNK), tl.num_programs(0)
    ):
        pos = chunk * CHUNK + tl.arange(0, CHUNK)
        covered = (pos < S) & (pos >= first) & (pos < last)
        mapped = tl.load(Perm + tl.minimum(pos, S - 1) * sp).to(
            tl.int64 if WIDE else tl.int32
        )
        mapped = tl.where(mapped < 0, mapped + S, mapped)
        mismatch += tl.sum((covered & (mapped != pos)).to(tl.int32), 0)
    tl.store(Proof + L * R * V + tl.program_id(0), mismatch != 0)


__all__ = ["chunked_embedding_lora_a"]
