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
def _base_gather_presence(
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
    BLOCK: tl.constexpr,
):
    si = tl.full((), SI, tl.int64) if WIDE else SI
    sr = tl.full((), SR, tl.int64) if WIDE else SR
    sw = tl.full((), SW, tl.int64) if WIDE else SW
    wl = tl.full((), WL, tl.int64) if WIDE else WL
    wr = tl.full((), WR, tl.int64) if WIDE else WR
    wv = tl.full((), WV, tl.int64) if WIDE else WV
    offsets = tl.program_id(0).to(
        tl.int64 if WIDE else tl.int32
    ) * BLOCK + tl.arange(0, BLOCK)
    valid = offsets < S * R
    if B == 0:
        value = tl.full((BLOCK,), 0, tl.float32)
    else:
        safe_offsets = tl.minimum(offsets, S * R - 1)
        row = safe_offsets // R
        rank_pos = safe_offsets % R
        adapter = tl.full((BLOCK,), 0, tl.int64 if WIDE else tl.int32)
        found = tl.full((BLOCK,), False, tl.int1)
        for b in range(B):
            present = tl.load(
                Presence + b.to(tl.int64 if WIDE else tl.int32) * S + row
            )
            wi = tl.load(Widx + b * sw).to(tl.int64 if WIDE else tl.int32)
            wi = tl.where(wi < 0, wi + L, wi)
            active_rank = tl.load(Ranks + wi * sr).to(
                tl.int64 if WIDE else tl.int32
            )
            selected = (present != 0) & (rank_pos < active_rank)
            adapter = tl.where(selected, wi, adapter)
            found |= selected
        token = tl.load(Ids + row * si).to(tl.int64 if WIDE else tl.int32)
        token = tl.where(token < 0, token + V, token)
        token = tl.where(found, token, 0)
        loaded = tl.load(Weights + adapter * wl + rank_pos * wr + token * wv)
        value = tl.where(found, loaded, 0.0)
    tl.store(Out + offsets, value, valid)


def _base_index_tensor(value, bound):
    return (
        value.to(torch.int32)
        if value.dtype == torch.int64 and bound <= 2147483647
        else value
    )


@triton.jit
def _base_prove_unique_identity(
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
def _base_fill_fused_presence(
    Seg,
    Perm,
    Proof,
    Presence,
    S: tl.constexpr,
    SS: tl.constexpr,
    SP: tl.constexpr,
    WIDE: tl.constexpr,
    CHUNK: tl.constexpr,
    ROWS: tl.constexpr,
    PARTS: tl.constexpr,
    ID_BLOCK: tl.constexpr,
):
    ss = tl.full((), SS, tl.int64) if WIDE else SS
    sp = tl.full((), SP, tl.int64) if WIDE else SP
    pid = tl.program_id(0).to(tl.int64 if WIDE else tl.int32)
    b = pid // PARTS
    part = pid % PARTS
    start = tl.load(Seg + b * ss)
    end = tl.load(Seg + (b + 1) * ss)
    mismatch = tl.load(Proof)
    count: tl.constexpr = (S + PARTS - 1) // PARTS
    lower = part * count
    upper = tl.minimum(lower + count, S)
    for identity_chunk in range(
        tl.where(mismatch == 0, tl.cdiv(count, ID_BLOCK), 0)
    ):
        identity_row = (
            lower
            + identity_chunk.to(tl.int64 if WIDE else tl.int32) * ID_BLOCK
            + tl.arange(0, ID_BLOCK)
        )
        identity_present = ((identity_row >= start) & (identity_row < end)).to(
            tl.int32
        )
        tl.store(
            Presence + b * S + identity_row,
            identity_present,
            identity_row < upper,
        )
    chunks = tl.cdiv(S, CHUNK)
    found = tl.full((), 0, tl.int32)
    iterations: tl.constexpr = (
        (S + PARTS - 1) // PARTS * ((S + CHUNK - 1) // CHUNK)
    )
    remaining = (
        tl.full((), S, tl.int64 if iterations > 2147483647 else tl.int32)
        - part
    )
    limit = tl.cdiv(remaining, PARTS) * chunks
    for step in range(tl.where(mismatch == 0, 0, limit)):
        tile = step // chunks
        chunk = step % chunks
        row = tile.to(tl.int64 if WIDE else tl.int32) * PARTS + part
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
        last_chunk = chunk == chunks - 1
        tl.store(Presence + b * S + row, found, last_chunk)
        found = tl.where(last_chunk, 0, found)


def _basechunked_embedding_lora_a(input_ids, weights, batch_info, vocab_size):
    s = input_ids.shape[0]
    loras, rank, vocab = weights.shape
    out = torch.empty((s, rank), dtype=weights.dtype, device=weights.device)
    if s == 0 or rank == 0:
        return out
    ids = _base_index_tensor(input_ids, vocab)
    seg = _base_index_tensor(batch_info.seg_indptr, s)
    widx = _base_index_tensor(batch_info.weight_indices, loras)
    ranks = _base_index_tensor(batch_info.lora_ranks, rank)
    perm = _base_index_tensor(batch_info.permutation, s)
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
    presence = torch.empty((bs * s,), dtype=torch.int32, device=weights.device)
    if bs > 0:
        proof = torch.empty((), dtype=torch.int32, device=weights.device)
        _base_prove_unique_identity[1,](
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
        parts = triton.cdiv(s, 128)
        _base_fill_fused_presence[bs * parts,](
            seg,
            perm,
            proof,
            presence,
            s,
            seg.stride(0),
            perm.stride(0),
            wide,
            128,
            1,
            parts,
            min(128, triton.next_power_of_2(triton.cdiv(s, parts))),
            num_warps=4,
            num_stages=1,
        )
    _base_gather_presence[
        triton.cdiv(
            s * rank,
            128 if weights.dtype == torch.float16 or s * rank < 1024 else 1024,
        ),
    ](
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
        128 if weights.dtype == torch.float16 or s * rank < 1024 else 1024,
        num_warps=4,
        num_stages=1,
    )
    return out


@triton.jit
def _direct_gather_presence(
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
    BLOCK: tl.constexpr,
    Seg,
    Proof,
    SS: tl.constexpr,
):
    ss = tl.full((), SS, tl.int64) if WIDE else SS
    si = tl.full((), SI, tl.int64) if WIDE else SI
    sr = tl.full((), SR, tl.int64) if WIDE else SR
    sw = tl.full((), SW, tl.int64) if WIDE else SW
    wl = tl.full((), WL, tl.int64) if WIDE else WL
    wr = tl.full((), WR, tl.int64) if WIDE else WR
    wv = tl.full((), WV, tl.int64) if WIDE else WV
    offsets = tl.program_id(0).to(
        tl.int64 if WIDE else tl.int32
    ) * BLOCK + tl.arange(0, BLOCK)
    valid = offsets < S * R
    if B == 0:
        value = tl.full((BLOCK,), 0, tl.float32)
    else:
        safe_offsets = tl.minimum(offsets, S * R - 1)
        row = safe_offsets // R
        rank_pos = safe_offsets % R
        if tl.load(Proof) == 0:
            identity_first = tl.load(Seg)
            identity_last = tl.load(Seg + B * ss)
            identity_present = (row >= identity_first) & (row < identity_last)
            identity_batch = tl.full((BLOCK,), 0, tl.int32)
            for b in range(1, B):
                identity_batch += (row >= tl.load(Seg + b * ss)).to(tl.int32)
            identity_batch = tl.where(identity_present, identity_batch, 0)
            adapter = tl.load(
                Widx + identity_batch.to(tl.int64 if WIDE else tl.int32) * sw
            ).to(tl.int64 if WIDE else tl.int32)
            adapter = tl.where(adapter < 0, adapter + L, adapter)
            identity_rank = tl.load(Ranks + adapter * sr).to(
                tl.int64 if WIDE else tl.int32
            )
            found = identity_present & (rank_pos < identity_rank)
            adapter = tl.where(found, adapter, 0)
        else:
            adapter = tl.full((BLOCK,), 0, tl.int64 if WIDE else tl.int32)
            found = tl.full((BLOCK,), False, tl.int1)
            for b in range(B):
                present = tl.load(
                    Presence + b.to(tl.int64 if WIDE else tl.int32) * S + row
                )
                wi = tl.load(Widx + b * sw).to(tl.int64 if WIDE else tl.int32)
                wi = tl.where(wi < 0, wi + L, wi)
                active_rank = tl.load(Ranks + wi * sr).to(
                    tl.int64 if WIDE else tl.int32
                )
                selected = (present != 0) & (rank_pos < active_rank)
                adapter = tl.where(selected, wi, adapter)
                found |= selected
        token = tl.load(Ids + row * si).to(tl.int64 if WIDE else tl.int32)
        token = tl.where(token < 0, token + V, token)
        token = tl.where(found, token, 0)
        loaded = tl.load(Weights + adapter * wl + rank_pos * wr + token * wv)
        value = tl.where(found, loaded, 0.0)
    tl.store(Out + offsets, value, valid)


def _direct_index_tensor(value, bound):
    return (
        value.to(torch.int32)
        if value.dtype == torch.int64 and bound <= 2147483647
        else value
    )


@triton.jit
def _direct_prove_unique_identity(
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
def _direct_fill_fused_presence(
    Seg,
    Perm,
    Proof,
    Presence,
    S: tl.constexpr,
    SS: tl.constexpr,
    SP: tl.constexpr,
    WIDE: tl.constexpr,
    CHUNK: tl.constexpr,
    ROWS: tl.constexpr,
    PARTS: tl.constexpr,
):
    ss = tl.full((), SS, tl.int64) if WIDE else SS
    sp = tl.full((), SP, tl.int64) if WIDE else SP
    pid = tl.program_id(0).to(tl.int64 if WIDE else tl.int32)
    b = pid // PARTS
    part = pid % PARTS
    start = tl.load(Seg + b * ss)
    end = tl.load(Seg + (b + 1) * ss)
    mismatch = tl.load(Proof)
    chunks = tl.cdiv(S, CHUNK)
    found = tl.full((), 0, tl.int32)
    iterations: tl.constexpr = (
        (S + PARTS - 1) // PARTS * ((S + CHUNK - 1) // CHUNK)
    )
    remaining = (
        tl.full((), S, tl.int64 if iterations > 2147483647 else tl.int32)
        - part
    )
    limit = tl.cdiv(remaining, PARTS) * chunks
    for step in range(tl.where(mismatch == 0, 0, limit)):
        tile = step // chunks
        chunk = step % chunks
        row = tile.to(tl.int64 if WIDE else tl.int32) * PARTS + part
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
        last_chunk = chunk == chunks - 1
        tl.store(Presence + b * S + row, found, last_chunk)
        found = tl.where(last_chunk, 0, found)


def _directchunked_embedding_lora_a(
    input_ids, weights, batch_info, vocab_size
):
    s = input_ids.shape[0]
    loras, rank, vocab = weights.shape
    out = torch.empty((s, rank), dtype=weights.dtype, device=weights.device)
    if s == 0 or rank == 0:
        return out
    ids = _direct_index_tensor(input_ids, vocab)
    seg = _direct_index_tensor(batch_info.seg_indptr, s)
    widx = _direct_index_tensor(batch_info.weight_indices, loras)
    ranks = _direct_index_tensor(batch_info.lora_ranks, rank)
    perm = _direct_index_tensor(batch_info.permutation, s)
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
    presence = torch.empty((bs * s,), dtype=torch.int32, device=weights.device)
    proof = presence
    if bs > 0:
        proof = torch.empty((), dtype=torch.int32, device=weights.device)
        _direct_prove_unique_identity[1,](
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
        parts = triton.cdiv(s, 128)
        _direct_fill_fused_presence[bs * parts,](
            seg,
            perm,
            proof,
            presence,
            s,
            seg.stride(0),
            perm.stride(0),
            wide,
            128,
            1,
            parts,
            num_warps=4,
            num_stages=1,
        )
    _direct_gather_presence[
        triton.cdiv(
            s * rank,
            128 if weights.dtype == torch.float16 or s * rank < 1024 else 1024,
        ),
    ](
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
        128 if weights.dtype == torch.float16 or s * rank < 1024 else 1024,
        seg,
        proof,
        seg.stride(0),
        num_warps=4,
        num_stages=1,
    )
    return out


def chunked_embedding_lora_a(input_ids, weights, batch_info, vocab_size):
    if weights.shape[1] <= 32:
        return _directchunked_embedding_lora_a(
            input_ids, weights, batch_info, vocab_size
        )
    return _basechunked_embedding_lora_a(
        input_ids, weights, batch_info, vocab_size
    )


__all__ = ["chunked_embedding_lora_a"]
