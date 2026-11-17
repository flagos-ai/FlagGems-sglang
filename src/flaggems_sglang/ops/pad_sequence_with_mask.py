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
    LC: tl.constexpr,
    SC: tl.constexpr,
):
    b = P // NH
    h = P % NH * BH + tl.arange(0, BH)
    s = tl.arange(0, BM)
    off = tl.load(Offsets + b * OS).to(tl.int64)
    length = tl.load(L + b * LS)
    valid = s < length
    row = tl.minimum(tl.maximum(off + s, 0), T - 1)
    x = tl.load(
        X + row[:, None] * XS + h[None, :] * XH,
        (s[:, None] < M) & valid[:, None] & (h[None, :] < H),
        other=0,
        cache_modifier=LC,
    )
    tl.store(
        Y + (b * M + s[:, None]) * H + h[None, :],
        x,
        (s[:, None] < M) & (h[None, :] < H),
        cache_modifier=SC,
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
    LC: tl.constexpr,
    SC: tl.constexpr,
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
            LC,
            SC,
        )
    else:
        for p in range(tl.program_id(0), B * NH, NP):
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
                p,
                LC,
                SC,
            )


@triton.jit(do_not_specialize=("Offsets", "L"))
def _compact(
    X, Offsets, L, Y, A, CFG: tl.constexpr, LC: tl.constexpr, SC: tl.constexpr
):
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
        LC,
        SC,
    )


@triton.jit(do_not_specialize=("Offsets", "L"))
def _zerowidth120(
    X,
    Offsets,
    L,
    Y,
    A,
    CFG: tl.constexpr,
    MODE: tl.constexpr,
    ZW: tl.constexpr,
    POLICY: tl.constexpr,
):
    B: tl.constexpr = CFG[0]
    M: tl.constexpr = CFG[1]
    H: tl.constexpr = CFG[2]
    T: tl.constexpr = CFG[3]
    XS: tl.constexpr = CFG[4]
    OS: tl.constexpr = CFG[6]
    LS: tl.constexpr = CFG[7]
    BM: tl.constexpr = CFG[8]
    BH: tl.constexpr = CFG[9]
    NH: tl.constexpr = CFG[10]
    NP: tl.constexpr = CFG[11]
    X = X.to(tl.pointer_type(tl.uint32))
    Y = Y.to(tl.pointer_type(tl.uint32))
    ZH: tl.constexpr = ZW
    NZ: tl.constexpr = (H + ZH - 1) // ZH
    NT: tl.constexpr = B * (NH + NZ)
    for work in range(tl.program_id(0), NT, NP):
        if MODE > 0:
            is_data = work < B * NH
            p = tl.where(is_data, work, work - B * NH)
        else:
            is_data = work >= B * NZ
            p = tl.where(is_data, work - B * NZ, work)
        if is_data:
            b = p // NH
            hc = tl.arange(0, BH)
            sr = tl.arange(0, BM)
            h = tl.broadcast_to(hc[None, :], (BM, BH))
            s = tl.broadcast_to(sr[:, None], (BM, BH))
            h = p % NH * BH + h
            off = tl.load(Offsets + b * OS).to(tl.int64)
            length = tl.load(L + b * LS)
            row = tl.minimum(tl.maximum(off + s, 0), T - 1)
            pred = (s < M) & (s < length) & (h < H)
            ptr = tl.inline_asm_elementwise(
                "mad.wide.u32 $0, $1, 16, $2;",
                constraints="=l,r,l",
                args=[
                    row.to(tl.uint32) * XS + h.to(tl.uint32),
                    X.to(tl.uint64),
                ],
                dtype=tl.uint64,
                is_pure=True,
                pack=1,
            )
            dst = Y + ((b * M + s) * H + h) * 4
            outpred = (s < M) & (h < H)
            asm: tl.constexpr = (
                "{ .reg .pred p; "
                + (
                    ".reg .b64 c; createpolicy.fractional.L2::"
                    + ("evict_first" if POLICY == 5 else "evict_last")
                    + ".b64 c,1.0; "
                    if POLICY >= 3
                    else ""
                )
                + "setp.ne.u32 p,$5,0; mov.u32 $0,0; mov.u32 $1,0; mov.u32 $2,0; mov.u32 $3,0; @p ld.global."
                + ("cs" if POLICY == 6 else "cg")
                + (".L2::cache_hint" if POLICY >= 3 else "")
                + ".v4.b32 {$0,$1,$2,$3},[$4]"
                + (",c" if POLICY >= 3 else "")
                + "; }"
            )
            x0, x1, x2, x3 = tl.inline_asm_elementwise(
                asm,
                constraints="=r,=r,=r,=r,l,r",
                args=[ptr, pred.to(tl.int32)],
                dtype=(tl.uint32, tl.uint32, tl.uint32, tl.uint32),
                is_pure=False,
                pack=1,
            )
            storepred = pred if 4 > 0 else outpred
            sasm: tl.constexpr = (
                "{ .reg .pred p; "
                + (
                    ".reg .b64 c; createpolicy.fractional.L2::evict_first.b64 c,1.0; "
                    if POLICY == 2 or POLICY >= 4
                    else ""
                )
                + "setp.ne.u32 p,$6,0; @p st.global.cs"
                + (".L2::cache_hint" if POLICY == 2 or POLICY >= 4 else "")
                + ".v4.b32 [$1],{$2,$3,$4,$5}"
                + (",c" if POLICY == 2 or POLICY >= 4 else "")
                + "; mov.u32 $0,0; }"
            )
            tl.inline_asm_elementwise(
                sasm,
                constraints="=r,l,r,r,r,r,r",
                args=[dst, x0, x1, x2, x3, storepred.to(tl.int32)],
                dtype=tl.uint32,
                is_pure=False,
                pack=1,
            )
        else:
            z_b = p // NZ
            z_h = p % NZ * ZH + tl.arange(0, ZH)
            z_s = tl.arange(0, BM)
            z_length = tl.load(L + z_b * LS)
            z_zero = (
                (z_s[:, None] < M)
                & (z_s[:, None] >= z_length)
                & (z_h[None, :] < H)
            )
            z_dst = Y + ((z_b * M + z_s[:, None]) * H + z_h[None, :]) * 4
            tl.inline_asm_elementwise(
                (
                    (
                        "{ .reg .pred z; .reg .b64 c; "
                        "createpolicy.fractional.L2::evict_first.b64 c,1.0; setp.ne.u32 z,"
                        "$2,0; @z st.global.cs.L2::cache_hint.v4.b32 [$1],{0,0,0,0},c; "
                        "mov.u32 $0,0; }"
                    )
                    if POLICY > 0
                    else "{ .reg .pred z; setp.ne.u32 z,$2,0; @z st.global.cs.v4.b32 [$1],{0,0,0,0}; mov.u32 $0,0; }"
                ),
                constraints="=r,l,r",
                args=[z_dst, z_zero.to(tl.int32)],
                dtype=tl.uint32,
                is_pure=False,
                pack=1,
            )
            if p % NZ == 0:
                tl.store(A + z_b * M + z_s, z_s < z_length, z_s < M)


def pad_sequence_with_mask(input_emb, offsets, lengths, max_len):
    lc = ".cg" if input_emb.numel() >= 16000000 else ""
    sc = ""
    b = offsets.shape[0]
    hidden = input_emb.shape[1]
    output = input_emb.new_empty((b, max_len, hidden))
    attn_mask = input_emb.new_empty((b * max_len,), dtype=torch.bool)
    n = b * max_len * hidden
    if torch.version.hip is None and max_len == 64 and (n >= 8000000):
        unit = 16 // input_emb.element_size()
        total = input_emb.shape[0]
        xs, xh = input_emb.stride()
        if (
            xh == 1
            and xs > 0
            and (xs % unit == 0)
            and (hidden % unit == 0)
            and (input_emb.data_ptr() % 16 == 0)
            and (((total - 1) * xs + hidden) // unit < 4294967296)
            and (b * max_len * (hidden // unit) * 4 < 2147483648)
        ):
            vh = hidden // unit
            bh = min(triton.next_power_of_2(vh), 16 if n < 16000000 else 32)
            nh = triton.cdiv(vh, bh)
            np = min(b * (nh + triton.cdiv(vh, 4)), 65535)
            cfg = (
                b,
                max_len,
                vh,
                total,
                xs // unit,
                1,
                offsets.stride(0),
                lengths.stride(0),
                triton.next_power_of_2(max_len),
                bh,
                nh,
                np,
            )
            _zerowidth120.run(
                input_emb,
                offsets,
                lengths,
                output,
                attn_mask,
                cfg,
                1,
                4,
                5 if n >= 33554432 else 0,
                num_warps=8 if n < 16000000 else 4,
                grid=(np,),
                warmup=False,
            )
            return (b, output, attn_mask)
    bm = triton.next_power_of_2(max_len)
    bh = min(
        triton.next_power_of_2(hidden),
        32 if b * max_len * hidden >= 33554432 else 64,
    )
    if 16000000 <= b * max_len * hidden < 33554432:
        bh = min(triton.next_power_of_2(hidden), 128)
    nh = triton.cdiv(hidden, bh)
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
            input_emb.shape[0],
            input_emb.stride(0),
            input_emb.stride(1),
            offsets.stride(0),
            lengths.stride(0),
            bm,
            bh,
            nh,
            np,
        ),
        lc,
        sc,
        num_warps=(
            32
            if 16000000 <= b * max_len * hidden < 33554432
            else 16 if bh == 32 else 4
        ),
        grid=(np,),
        warmup=False,
    )
    return (b, output, attn_mask)


__all__ = ["pad_sequence_with_mask"]
