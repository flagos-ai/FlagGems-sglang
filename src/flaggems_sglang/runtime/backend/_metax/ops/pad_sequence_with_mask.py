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
def _pad(
    X,
    Offsets,
    L,
    Y,
    A,
    B: tl.constexpr,
    M: tl.constexpr,
    H: tl.constexpr,
    T: tl.constexpr,
    XS: tl.constexpr,
    XH: tl.constexpr,
    OS: tl.constexpr,
    LS: tl.constexpr,
    BM: tl.constexpr,
    BH: tl.constexpr,
    NH: tl.constexpr,
    NP: tl.constexpr,
):
    for p in range(tl.program_id(0), B * NH, NP):
        b = p // NH
        h = p % NH * BH + tl.arange(0, BH)
        s = tl.arange(0, BM)
        off = tl.load(Offsets + b * OS).to(tl.int64)
        length = tl.load(L + b * LS)
        valid = s < length
        row = tl.minimum(tl.maximum(off + s, 0), T - 1)
        x = tl.load(
            X + row[:, None] * XS + h[None, :] * XH,
            (s[:, None] < M) & valid[:, None] & (h[None, :] < H),
            other=0,
        )
        tl.store(
            Y + (b * M + s[:, None]) * H + h[None, :],
            x,
            (s[:, None] < M) & (h[None, :] < H),
        )
        if p % NH == 0:
            tl.store(A + b * M + s, valid, s < M)


@triton.jit(do_not_specialize=("Offsets", "L"))
def _compact(X, Offsets, L, Y, A, CFG: tl.constexpr):
    _pad(
        X,
        Offsets,
        L,
        Y,
        A,
        CFG[0],
        CFG[1],
        CFG[2],
        CFG[3],
        CFG[4],
        CFG[5],
        CFG[6],
        CFG[7],
        CFG[8],
        CFG[9],
        CFG[10],
        CFG[11],
    )


def _ordinary242(input_emb, offsets, lengths, max_len):
    total, hidden = input_emb.shape
    xs, xh = input_emb.stride()
    b = offsets.shape[0]
    os = offsets.stride(0)
    ls = lengths.stride(0)
    output = input_emb.new_empty((b, max_len, hidden))
    attn_mask = input_emb.new_empty((b * max_len,), dtype=torch.bool)
    bh = min(1 << (hidden - 1).bit_length(), 128)
    warps = 4
    nh = (hidden + bh - 1) // bh
    np = min(b * nh, 65535)
    _compact.run(
        input_emb,
        offsets,
        lengths,
        output,
        attn_mask,
        (
            b,
            max_len,
            hidden,
            total,
            xs,
            xh,
            os,
            ls,
            1 << (max_len - 1).bit_length(),
            bh,
            nh,
            np,
        ),
        num_warps=warps,
        grid=(np,),
        warmup=False,
    )
    return (b, output, attn_mask)


@triton.jit(do_not_specialize=("Offsets", "L"))
def _split241(
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
    elif UNIT == 4:
        X = X.to(tl.pointer_type(tl.uint64))
        Y = Y.to(tl.pointer_type(tl.uint64))
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
            b = work % B
            h = work // B * BH + tl.arange(0, BH)
            s = tl.arange(0, BM)
            off = tl.load(Offsets + b * OS).to(tl.int64)
            length = tl.load(L + b * LS)
            row = tl.minimum(tl.maximum(off + s, 0), T - 1)
            pred = (s[:, None] < M) & (s[:, None] < length) & (h[None, :] < H)
            value = tl.load(
                X + row[:, None] * XS + h[None, :] * XH, pred, other=0
            )
            tl.store(Y + (b * M + s[:, None]) * H + h[None, :], value, pred)
        else:
            p = work - B * NH
            b = p % B
            z_h = p // B * ZH + tl.arange(0, ZH)
            s = tl.arange(0, BM)
            length = tl.load(L + b * LS)
            z_pred = (
                (s[:, None] < M) & (s[:, None] >= length) & (z_h[None, :] < H)
            )
            tl.store(Y + (b * M + s[:, None]) * H + z_h[None, :], 0, z_pred)
            if p // B == 0:
                tl.store(A + b * M + s, s < length, s < M)


def pad_sequence_with_mask(input_emb, offsets, lengths, max_len):
    total, hidden = input_emb.shape
    b = offsets.shape[0]
    eligible = (
        b * max_len * hidden >= 33554432
        and input_emb.element_size() == 2
        and (input_emb.stride(1) == 1)
        and (hidden % 4 == 0)
        and (input_emb.stride(0) % 4 == 0)
        and (input_emb.data_ptr() % 8 == 0)
    )
    if not eligible:
        return _ordinary242(input_emb, offsets, lengths, max_len)
    output = input_emb.new_empty((b, max_len, hidden))
    attn_mask = input_emb.new_empty((b * max_len,), dtype=torch.bool)
    h = hidden // 2
    bh = min(1 << (h - 1).bit_length(), 128)
    zh = min(1 << (h - 1).bit_length(), 256)
    nh = (h + bh - 1) // bh
    np = min(b * (nh + (h + zh - 1) // zh), 65535)
    cfg = (
        b,
        max_len,
        h,
        total,
        input_emb.stride(0) // 2,
        1,
        offsets.stride(0),
        lengths.stride(0),
        1 << (max_len - 1).bit_length(),
        bh,
        nh,
        np,
    )
    _split241.run(
        input_emb,
        offsets,
        lengths,
        output,
        attn_mask,
        cfg,
        zh,
        2,
        num_warps=4,
        grid=(np,),
        warmup=False,
    )
    return (b, output, attn_mask)


__all__ = ["pad_sequence_with_mask"]
