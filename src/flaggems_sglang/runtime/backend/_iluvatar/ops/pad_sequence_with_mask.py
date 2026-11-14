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
def _rank_tile(X, Offsets, L, Y, A, CFG: tl.constexpr, P, MODE: tl.constexpr):
    M: tl.constexpr = CFG[1]
    H: tl.constexpr = CFG[2]
    T: tl.constexpr = CFG[3]
    XS: tl.constexpr = CFG[4]
    XH: tl.constexpr = CFG[5]
    OS: tl.constexpr = CFG[6]
    LS: tl.constexpr = CFG[7]
    BM: tl.constexpr = CFG[8]
    BH: tl.constexpr = CFG[9]
    NH: tl.constexpr = CFG[10]
    b = P // NH
    base = P % NH * BH
    off = tl.load(Offsets + b * OS).to(tl.int64)
    length = tl.load(L + b * LS)
    if MODE >= 20:
        U: tl.constexpr = min(BM, MODE - 20)
        s = (
            tl.arange(0, BM // U)[None, :, None, None] * U
            + tl.arange(0, U)[None, None, :, None]
        )
        h = (
            base
            + tl.arange(0, BH // 2)[:, None, None, None] * 2
            + tl.arange(0, 2)[None, None, None, :]
        )
    else:
        V: tl.constexpr = min(BH, 1 if MODE == 3 else 2 if MODE == 1 else 4)
        s = tl.arange(0, BM)[None, :, None]
        h = (
            base
            + tl.arange(0, BH // V)[:, None, None] * V
            + tl.arange(0, V)[None, None, :]
        )
    valid = s < length
    row = tl.minimum(tl.maximum(off + s, 0), T - 1)
    x = tl.load(X + row * XS + h * XH, (s < M) & valid & (h < H), other=0)
    tl.store(Y + (b * M + s) * H + h, x, (s < M) & (h < H))
    if P % NH == 0:
        sm = tl.arange(0, BM)
        tl.store(A + b * M + sm, sm < length, sm < M)


@triton.jit(do_not_specialize=("Offsets", "L"))
def _rank_compact(X, Offsets, L, Y, A, CFG: tl.constexpr, MODE: tl.constexpr):
    if CFG[0] * CFG[10] <= CFG[11]:
        _rank_tile(X, Offsets, L, Y, A, CFG, tl.program_id(0), MODE)
    else:
        for p in range(tl.program_id(0), CFG[0] * CFG[10], CFG[11]):
            _rank_tile(X, Offsets, L, Y, A, CFG, p, MODE)


def _ordinary397(input_emb, offsets, lengths, max_len):
    total, hidden = input_emb.shape
    b = offsets.shape[0]
    bh = max(2, min(1 << (hidden - 1).bit_length(), 128))
    nh = (hidden + bh - 1) // bh
    np = min(b * nh, 65535)
    output = input_emb.new_empty((b, max_len, hidden))
    mask = input_emb.new_empty((b * max_len,), dtype=torch.bool)
    cfg = (
        b,
        max_len,
        hidden,
        total,
        input_emb.stride(0),
        input_emb.stride(1),
        offsets.stride(0),
        lengths.stride(0),
        1 << (max_len - 1).bit_length(),
        bh,
        nh,
        np,
    )
    _rank_compact.run(
        input_emb,
        offsets,
        lengths,
        output,
        mask,
        cfg,
        24,
        num_warps=8,
        grid=(np,),
        warmup=False,
    )
    return (b, output, mask)


@triton.jit(do_not_specialize=("Offsets", "L"))
def _split396(
    X,
    Offsets,
    L,
    Y,
    A,
    CFG: tl.constexpr,
    ZH: tl.constexpr,
    UNIT: tl.constexpr,
):
    if UNIT == 2:
        X = X.to(tl.pointer_type(tl.uint32))
        Y = Y.to(tl.pointer_type(tl.uint32))
    B: tl.constexpr = CFG[0]
    M: tl.constexpr = CFG[1]
    H: tl.constexpr = CFG[2]
    T: tl.constexpr = CFG[3]
    XS: tl.constexpr = CFG[4]
    XH: tl.constexpr = CFG[5]
    OS: tl.constexpr = CFG[6]
    LS: tl.constexpr = CFG[7]
    BM: tl.constexpr = CFG[8]
    BH: tl.constexpr = CFG[9]
    NH: tl.constexpr = CFG[10]
    NP: tl.constexpr = CFG[11]
    NZ: tl.constexpr = (H + ZH - 1) // ZH
    for work in range(tl.program_id(0), B * (NH + NZ), NP):
        if work < B * NH:
            b = work // NH
            base = work % NH * BH
            U: tl.constexpr = min(BM, 4)
            s = (
                tl.arange(0, BM // U)[None, :, None, None] * U
                + tl.arange(0, U)[None, None, :, None]
            )
            h = (
                base
                + tl.arange(0, BH // 2)[:, None, None, None] * 2
                + tl.arange(0, 2)[None, None, None, :]
            )
            off = tl.load(Offsets + b * OS).to(tl.int64)
            length = tl.load(L + b * LS)
            row = tl.minimum(tl.maximum(off + s, 0), T - 1)
            pred = (s < M) & (s < length) & (h < H)
            value = tl.load(X + row * XS + h * XH, pred, other=0)
            tl.store(Y + (b * M + s) * H + h, value, pred)
        else:
            p = work - B * NH
            zb = p // NZ
            zh = p % NZ * ZH + tl.arange(0, ZH)
            zs = tl.arange(0, BM)
            zl = tl.load(L + zb * LS)
            zp = (zs[:, None] < M) & (zs[:, None] >= zl) & (zh[None, :] < H)
            tl.store(Y + (zb * M + zs[:, None]) * H + zh[None, :], 0, zp)
            if p % NZ == 0:
                tl.store(A + zb * M + zs, zs < zl, zs < M)


def pad_sequence_with_mask(input_emb, offsets, lengths, max_len):
    total, hidden = input_emb.shape
    b = offsets.shape[0]
    if b * max_len * hidden < 16777216 or max_len > 128 or hidden < 128:
        return _ordinary397(input_emb, offsets, lengths, max_len)
    bh = 128
    nh = (hidden + bh - 1) // bh
    np = min(b * (nh + nh), 65535)
    output = input_emb.new_empty((b, max_len, hidden))
    mask = input_emb.new_empty((b * max_len,), dtype=torch.bool)
    cfg = (
        b,
        max_len,
        hidden,
        total,
        input_emb.stride(0),
        input_emb.stride(1),
        offsets.stride(0),
        lengths.stride(0),
        1 << (max_len - 1).bit_length(),
        bh,
        nh,
        np,
    )
    _split396.run(
        input_emb,
        offsets,
        lengths,
        output,
        mask,
        cfg,
        bh,
        1,
        num_warps=8,
        grid=(np,),
        warmup=False,
    )
    return (b, output, mask)


__all__ = ["pad_sequence_with_mask"]
