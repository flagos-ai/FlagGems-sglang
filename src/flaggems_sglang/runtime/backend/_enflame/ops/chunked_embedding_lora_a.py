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

import os

import torch
import triton
import triton.language as tl


@triton.jit
def _cela_owner(
    Perm,
    Seg,
    Owner,
    S,
    SS,
    SP,
    BLOCK: tl.constexpr,
):
    b = tl.program_id(0)
    start = tl.load(Seg + b * SS)
    end = tl.load(Seg + (b + 1) * SS)
    pos = start + tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    valid = (pos < end) & (pos < S)
    row = tl.load(Perm + pos * SP, valid, 0)
    row = tl.where(row < 0, row + S, row)
    tl.store(Owner + row, tl.zeros((BLOCK,), tl.int32) + b, valid)


@triton.jit
def _cela_gather(
    Ids,
    Weights,
    Widx,
    Ranks,
    Owner,
    Out,
    S,
    L,
    V,
    B,
    SI,
    SW,
    SR,
    WL,
    WR,
    WV,
    R: tl.constexpr,
    WIDE: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    wl = tl.full((), WL, tl.int64) if WIDE else WL
    wr = tl.full((), WR, tl.int64) if WIDE else WR
    wv = tl.full((), WV, tl.int64) if WIDE else WV
    rows = tl.program_id(0) * BLOCK_S + tl.arange(0, BLOCK_S)
    kb = tl.program_id(1) * BLOCK_K
    rc = tl.minimum(rows, S - 1)
    owner = tl.load(Owner + rc)
    so = tl.minimum(tl.maximum(owner, 0), B - 1)
    wi = tl.load(Widx + so * SW)
    wi = tl.where(wi < 0, wi + L, wi)
    rk = tl.minimum(tl.load(Ranks + wi * SR), R)
    tok = tl.load(Ids + rc * SI)
    tok = tl.where(tok < 0, tok + V, tok)
    base = wi * wl + tok * wv
    rcw = rc.to(tl.int64) if WIDE else rc
    act = owner >= 0
    rmask = rows < S
    for k in range(kb, kb + BLOCK_K):
        v = tl.load(Weights + base + k * wr, act & (k < rk), 0.0)
        tl.store(Out + rcw * R + k, v, rmask & (k < R))


_I64_TRUE8 = None


def _int64_true8(device):
    """Detect the on-device int64 storage layout (cached per process).

    torch_gcu stores int64 either as a true 8-byte little-endian layout (when
    TORCH_GCU_ENABLE_INT64_AND_UINT64 is on) or as 4-byte packed values (when
    off). The raw-int32 reinterpretation below differs between the two, so
    probe the device once and cache the verdict.
    """
    global _I64_TRUE8
    if _I64_TRUE8 is None:
        try:
            raw = (
                torch.arange(2, dtype=torch.int64, device=device)
                .view(torch.int32)
                .tolist()
            )
            # True 8-byte little-endian reads [0, 0, 1, 0]; packed reads
            # [0, 1, ...].
            _I64_TRUE8 = bool(len(raw) >= 4 and raw[1] == 0 and raw[2] == 1)
        except Exception:
            _I64_TRUE8 = os.environ.get(
                "TORCH_GCU_ENABLE_INT64_AND_UINT64", ""
            ) not in ("", "0")
    return _I64_TRUE8


def _index_tensor(value, bound):
    """Zero-copy int32 view of an int64 index tensor -> (tensor, stride scale).

    Values fit in int32 because ``bound`` (vocab, segment count, ...) is
    below 2**31, so the low word equals the value under either layout. This
    avoids a per-call device-side int64->int32 conversion kernel; the stride
    scale is 2 (int32 words per logical int64 step) for the true 8-byte
    layout, 1 for the packed layout.
    """
    if value.dtype != torch.int64 or bound > 2147483647:
        return value, 1
    try:
        raw = value.view(torch.int32)
    except (RuntimeError, ValueError):
        return value.to(torch.int32), 1
    if _int64_true8(value.device):
        return raw, 2 * value.stride(0)
    return raw[: value.numel()], value.stride(0)


def chunked_embedding_lora_a(input_ids, weights, batch_info, vocab_size):
    s = input_ids.shape[0]
    loras, rank, vocab = weights.shape
    out = torch.empty((s, rank), dtype=weights.dtype, device=weights.device)
    if s == 0 or rank == 0:
        return out
    bs = int(batch_info.bs)
    if bs <= 0:
        out.zero_()
        return out
    ids, ids_scale = _index_tensor(input_ids, vocab)
    seg, seg_scale = _index_tensor(batch_info.seg_indptr, s)
    widx, widx_scale = _index_tensor(batch_info.weight_indices, loras)
    ranks, ranks_scale = _index_tensor(batch_info.lora_ranks, rank)
    perm = batch_info.permutation
    if perm is None:
        perm = torch.arange(s, dtype=torch.int32, device=weights.device)
        perm_scale = 1
    else:
        perm, perm_scale = _index_tensor(perm, s)
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
            *(v.numel() * v.stride(0) for v in (ids, seg, widx, ranks, perm)),
        )
        + 1024
        > 2147483647
    )
    owner = torch.empty((s,), dtype=torch.int32, device=weights.device)
    max_len = int(getattr(batch_info, "max_len", 0)) or s
    _cela_owner[(bs, triton.cdiv(max_len, 512))](
        perm,
        seg,
        owner,
        s,
        seg.stride(0) * seg_scale,
        perm.stride(0) * perm_scale,
        512,
        num_warps=4,
        num_stages=1,
    )
    block_s = min(256 if s <= 4096 else 512, max(4, triton.next_power_of_2(s)))
    block_k = 8 if rank <= 8 else 16
    _cela_gather[(triton.cdiv(s, block_s), triton.cdiv(rank, block_k))](
        ids,
        weights,
        widx,
        ranks,
        owner,
        out,
        s,
        loras,
        vocab,
        bs,
        ids.stride(0) * ids_scale,
        widx.stride(0) * widx_scale,
        ranks.stride(0) * ranks_scale,
        wl,
        wr,
        wv,
        R=rank,
        WIDE=wide,
        BLOCK_S=block_s,
        BLOCK_K=block_k,
        num_warps=4,
        num_stages=1,
    )
    return out


__all__ = ["chunked_embedding_lora_a"]
