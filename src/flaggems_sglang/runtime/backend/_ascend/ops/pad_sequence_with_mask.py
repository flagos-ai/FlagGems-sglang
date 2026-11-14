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
def _tile(
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
    P: tl.constexpr,
):
    b = P // NH
    h = P % NH * BH + tl.arange(0, BH)
    s = tl.arange(0, BM)
    off = tl.load(Offsets + b * OS).to(tl.int64)
    length = tl.load(L + b * LS)
    valid = s < length
    row = tl.minimum(tl.maximum(off + s, 0), T - 1)
    if BH == 1:
        column = P % NH
        x = tl.load(X + row * XS + column * XH, (s < M) & valid, other=0)
        tl.store(
            (
                Y + (b * H + column) * M + s
                if BM >= 512
                else Y + (b * M + s) * H + column
            ),
            x,
            s < M,
        )
    else:
        if XH == 1:
            x = tl.load(
                X + row[:, None] * XS + h[None, :],
                (s[:, None] < M) & valid[:, None] & (h[None, :] < H),
                other=0,
            )
        else:
            BP: tl.constexpr = triton.next_power_of_2((BH - 1) * XH + 1)
            physical = P % NH * BH * XH + tl.arange(0, BP)
            block = tl.load(
                X + row[:, None] * XS + physical[None, :],
                (s[:, None] < M)
                & valid[:, None]
                & (physical[None, :] <= (H - 1) * XH),
                other=0,
            )
            indices = tl.broadcast_to(
                (tl.arange(0, BH) * XH)[None, :], (BM, BH)
            )
            if block.dtype == tl.int16:
                words = block.to(tl.int32)
                finite = (words & 65535 | 1065353216).to(
                    tl.float32, bitcast=True
                )
                selected = tl.gather(finite, indices, 1).to(
                    tl.int32, bitcast=True
                )
                x = selected.to(tl.int16)
            else:
                x = tl.gather(block, indices, 1)
        tl.store(
            Y + (b * M + s[:, None]) * H + h[None, :],
            x,
            (s[:, None] < M) & (h[None, :] < H),
        )
    if P % NH == 0:
        tl.store(A + b * M + s, valid, s < M)


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
    if B * NH <= NP:
        _tile(
            X,
            Offsets,
            L,
            Y,
            A,
            B,
            M,
            H,
            T,
            XS,
            XH,
            OS,
            LS,
            BM,
            BH,
            NH,
            tl.program_id(0),
        )
    else:
        for p in range(tl.program_id(0), B * NH, NP):
            _tile(
                X, Offsets, L, Y, A, B, M, H, T, XS, XH, OS, LS, BM, BH, NH, p
            )


@triton.jit(do_not_specialize=("X", "Offsets", "L", "Y", "A"))
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


def pad_sequence_with_mask(input_emb, offsets, lengths, max_len):
    total, hidden = input_emb.shape
    xs, xh = input_emb.stride()
    b = offsets.shape[0]
    output = (
        torch.empty_strided(
            (b, max_len, hidden),
            (hidden * max_len, 1, max_len),
            dtype=input_emb.dtype,
            device=input_emb.device,
        )
        if max_len > 256
        else input_emb.new_empty((b, max_len, hidden))
    )
    attn_mask = input_emb.new_empty((b * max_len,), dtype=torch.bool)
    bm = 1 << (max_len - 1).bit_length()
    budget = 8192 if xh != 1 or input_emb.dtype == torch.float32 else 16384
    bh = (
        1
        if bm >= 512
        else min(
            1 << (hidden - 1).bit_length(),
            256,
            max(1, budget // bm // (1 << (max(1, xh) - 1).bit_length())),
        )
    )
    nh = (hidden + bh - 1) // bh
    np = min(b * nh, 320)
    transport = xh != 1 and input_emb.dtype in (torch.float16, torch.bfloat16)
    source = input_emb.view(torch.int16) if transport else input_emb
    destination = output.view(torch.int16) if transport else output
    _compact.run(
        source,
        offsets,
        lengths,
        destination,
        attn_mask,
        (
            b,
            max_len,
            hidden,
            total,
            xs,
            xh,
            offsets.stride(0),
            lengths.stride(0),
            bm,
            bh,
            nh,
            np,
        ),
        num_warps=4,
        grid=(np,),
        warmup=False,
    )
    return (b, output, attn_mask)


__all__ = ["pad_sequence_with_mask"]
