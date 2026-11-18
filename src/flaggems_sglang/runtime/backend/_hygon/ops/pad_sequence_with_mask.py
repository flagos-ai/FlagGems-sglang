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
    if (
        T > 0
        and T + BM < 2147483648
        and (XS >= 0)
        and (XH >= 0)
        and ((T - 1) * XS + (H - 1) * XH < 1073741824)
    ):
        bounded = tl.minimum(tl.maximum(off, -BM), T - 1).to(tl.int32)
        row = tl.minimum(tl.maximum(bounded + s, 0), T - 1)
    else:
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


@triton.jit(do_not_specialize=("Offsets", "L"))
def _packed(X, Offsets, L, Y, A, CFG: tl.constexpr):
    X = X.to(tl.pointer_type(tl.uint32))
    Y = Y.to(tl.pointer_type(tl.uint32))
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


@triton.jit
def _short_tile(X, Offsets, L, Y, A, CFG: tl.constexpr, P):
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
    valid = s < length
    if (
        T > 0
        and T + BM < 2147483648
        and (XS >= 0)
        and (XH >= 0)
        and ((T - 1) * XS + (H - 1) * XH < 1073741824)
    ):
        bounded = tl.minimum(tl.maximum(off, -BM), T - 1).to(tl.int32)
        row = tl.minimum(tl.maximum(bounded + s, 0), T - 1)
    else:
        row = tl.minimum(tl.maximum(off + s, 0), T - 1)
    x = tl.load(X + row * XS + h * XH, (s < M) & valid & (h < H), other=0)
    tl.store(Y + (b * M + s) * H + h, x, (s < M) & (h < H))
    if P % NH == 0:
        sm = tl.arange(0, BM)
        tl.store(A + b * M + sm, sm < length, sm < M)


@triton.jit(do_not_specialize=("Offsets", "L"))
def _short(X, Offsets, L, Y, A, CFG: tl.constexpr):
    if CFG[0] * CFG[10] <= CFG[11]:
        _short_tile(X, Offsets, L, Y, A, CFG, tl.program_id(0))
    else:
        for p in range(tl.program_id(0), CFG[0] * CFG[10], CFG[11]):
            _short_tile(X, Offsets, L, Y, A, CFG, p)


@triton.jit
def _stream_store(base, byte, bits, H: tl.constexpr):
    return tl.inline_asm_elementwise(
        (
            f"s_mov_b32 s32, $4\n"
            f"s_mov_b32 s33, $12\n"
            f"s_mov_b32 s34, 2147483646\n"
            f"s_mov_b32 s35, 0x27000\n"
            f"s_mov_b32 s36, 0\n"
            f"buffer_store_dword $28, $20, s[32:35], s36 offen slc\n"
            f"s_add_u32 s36, s36, {H * 2}\n"
            f"buffer_store_dword $29, $20, s[32:35], s36 offen slc\n"
            f"s_add_u32 s36, s36, {H * 2}\n"
            f"buffer_store_dword $30, $20, s[32:35], s36 offen slc\n"
            f"s_add_u32 s36, s36, {H * 2}\n"
            f"buffer_store_dword $31, $20, s[32:35], s36 offen slc\n"
            f"v_mov_b32 $0, 0\n"
            f"v_mov_b32 $1, 0\n"
            f"v_mov_b32 $2, 0\n"
            f"v_mov_b32 $3, 0"
        ),
        constraints=(
            "=&v,=&v,=&v,=&v,s,s,s,s,s,s,s,s,s,s,s,s,s,s,s,s,v,v,v,v,v,v,v,v,"
            "v,v,v,v,~{s32},~{s33},~{s34},~{s35},~{memory},~{s36},~{scc}"
        ),
        args=[
            base.to(tl.uint32),
            (base >> 32).to(tl.uint32) & 65535,
            byte,
            bits,
        ],
        dtype=tl.uint16,
        is_pure=False,
        pack=8,
    )


@triton.jit
def _stream_tile(X, Offsets, L, Y, A, CFG: tl.constexpr, P):
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
    length = tl.load(L + b * LS)
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
    valid = s < length
    off = tl.load(Offsets + b * OS).to(tl.int64)
    if (
        T > 0
        and T + BM < 2147483648
        and (XS >= 0)
        and (XH >= 0)
        and ((T - 1) * XS + (H - 1) * XH < 1073741824)
    ):
        bounded = tl.minimum(tl.maximum(off, -BM), T - 1).to(tl.int32)
        row = tl.minimum(tl.maximum(bounded + s, 0), T - 1)
    else:
        row = tl.minimum(tl.maximum(off + s, 0), T - 1)
    x = tl.load(X + row * XS + h * XH, (s < M) & valid & (h < H), other=0)
    destination = (b * M + s // 4 * 4) * H + h
    outpred = (s < M) & (h < H)
    byte = tl.where(outpred, destination * 2, -2147483648).to(tl.uint32)
    bits = x.to(tl.uint16, bitcast=True)
    _stream_store(Y.to(tl.uint64), byte, bits, H)
    if P % NH == 0:
        sm = tl.arange(0, BM)
        tl.store(A + b * M + sm, sm < length, sm < M)


@triton.jit(do_not_specialize=("Offsets", "L"))
def _stream(X, Offsets, L, Y, A, CFG: tl.constexpr):
    if CFG[0] * CFG[10] <= CFG[11]:
        _stream_tile(X, Offsets, L, Y, A, CFG, tl.program_id(0))
    else:
        for p in range(tl.program_id(0), CFG[0] * CFG[10], CFG[11]):
            _stream_tile(X, Offsets, L, Y, A, CFG, p)


def pad_sequence_with_mask(input_emb, offsets, lengths, max_len):
    total, hidden = input_emb.shape
    xs, xh = input_emb.stride()
    b = offsets.shape[0]
    os = offsets.stride(0)
    ls = lengths.stride(0)
    output = input_emb.new_empty((b, max_len, hidden))
    attn_mask = input_emb.new_empty((b * max_len,), dtype=torch.bool)
    n = b * max_len * hidden
    stream = (
        max_len == 64
        and hidden % 128 == 0
        and (xh == 1)
        and (xs >= 0)
        and (input_emb.element_size() == 2)
        and (4194304 <= n)
        and (n * 2 < 2147483646)
    ) and (xs % 2 == 0 and input_emb.data_ptr() % 4 == 0)
    packed = (
        64 < max_len <= 128
        and max_len & max_len - 1 == 0
        and (hidden >= 128)
        and (hidden % 128 == 0)
        and (input_emb.element_size() == 2)
        and (xh == 1)
        and (xs % 2 == 0)
        and (input_emb.data_ptr() % 4 == 0)
    )
    unit = 2 if packed else 1
    logical_h = hidden // unit
    bh = min(
        1 << (logical_h - 1).bit_length(),
        (
            64
            if packed
            else (256 if max_len <= 32 else 128) if hidden >= 128 else 64
        ),
    )
    warps = 8 if packed or hidden < 128 else 16
    nh = (logical_h + bh - 1) // bh
    np = min(b * nh, 65535)
    cfg = (
        b,
        max_len,
        logical_h,
        total,
        xs // unit,
        xh,
        os,
        ls,
        1 << (max_len - 1).bit_length(),
        bh,
        nh,
        np,
    )
    if stream:
        _stream.run(
            input_emb,
            offsets,
            lengths,
            output,
            attn_mask,
            cfg,
            num_warps=warps,
            grid=(np,),
            warmup=False,
        )
    elif packed:
        _packed.run(
            input_emb,
            offsets,
            lengths,
            output,
            attn_mask,
            cfg,
            num_warps=warps,
            grid=(np,),
            warmup=False,
        )
    elif (
        32 < max_len <= 64
        and hidden >= 128
        and (b * nh >= 4096)
        and (xh == 1)
        and (input_emb.element_size() == 2)
    ):
        _short.run(
            input_emb,
            offsets,
            lengths,
            output,
            attn_mask,
            cfg,
            num_warps=warps,
            grid=(np,),
            warmup=False,
        )
    else:
        _compact.run(
            input_emb,
            offsets,
            lengths,
            output,
            attn_mask,
            cfg,
            num_warps=warps,
            grid=(np,),
            warmup=False,
        )
    return (b, output, attn_mask)


__all__ = ["pad_sequence_with_mask"]
