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


@triton.jit(do_not_specialize=("X", "Offsets", "L", "Y", "A"))
def _schedule619(
    X,
    Offsets,
    L,
    Y,
    A,
    B: tl.constexpr,
    M: tl.constexpr,
    H: tl.constexpr,
    T: tl.constexpr,
    OS: tl.constexpr,
    LS: tl.constexpr,
    BM: tl.constexpr,
    BH: tl.constexpr,
    NP: tl.constexpr,
    STYLE: tl.constexpr,
    MBH: tl.constexpr,
):
    X = X.to(tl.pointer_type(tl.uint32))
    Y = Y.to(tl.pointer_type(tl.uint32))
    A = A.to(tl.pointer_type(tl.uint64))
    pid = tl.program_id(0)
    mask_program = pid == NP
    s = tl.arange(0, BM)[:, None]
    h = tl.arange(0, BH)[None, :]
    NH: tl.constexpr = triton.cdiv(H, BH)
    if STYLE:
        batch_count: tl.constexpr = triton.cdiv(B, NP)
        first = tl.where(mask_program, 0, pid * batch_count)
        last = tl.where(
            mask_program, 1, tl.minimum((pid + 1) * batch_count, B)
        )
        step: tl.constexpr = 1
    else:
        first = tl.where(mask_program, 0, pid)
        last = tl.where(mask_program, NP, B * NH)
        step: tl.constexpr = NP
    if M <= 64:
        preload = tl.minimum(first // NH, B - 1)
        piped_off = tl.load(Offsets + preload * OS).to(tl.int64)
        if B * M * H >= 16777216:
            piped_length = tl.load(L + preload * LS)
    for p in range(first, last, step):
        if mask_program:
            index = s * MBH + tl.arange(0, MBH)[None, :]
            mask_row = index * 8
            mask_batch = tl.minimum(mask_row // M, B - 1)
            mask_length = tl.load(L + mask_batch * LS)
            bounded = tl.minimum(tl.maximum(mask_length, 0), M)
            count = bounded - mask_row % M
            shift = tl.where(
                count >= 8, 0, tl.where(count > 0, (8 - count) * 8, 56)
            )
            word = tl.full(
                (BM, MBH), 72340172838076673, tl.uint64
            ) >> shift.to(tl.uint32)
            word = word * (count > 0).to(tl.uint64)
            tl.store(A + index, word, offset_state_policy="continuous")
        else:
            b = p // NH
            cols = p % NH * BH + h
            if M <= 64:
                off = piped_off
                if B * M * H >= 16777216:
                    length = piped_length
                else:
                    length = tl.load(L + b * LS)
                next_batch = tl.minimum((p + step) // NH, B - 1)
                piped_off = tl.load(Offsets + next_batch * OS).to(tl.int64)
                if B * M * H >= 16777216:
                    piped_length = tl.load(L + next_batch * LS)
            else:
                off = tl.load(Offsets + b * OS).to(tl.int64)
                length = tl.load(L + b * LS)
            row = tl.minimum(tl.maximum(off + s, 0), T - 1)
            x = tl.load(
                X + row * H + cols, offset_state_policy="locally_continuous"
            )
            if B >= 128 or H <= 1024:
                x = x & tl.full((), 0, tl.uint32) - (s < length).to(tl.uint32)
            else:
                x = tl.where(s < length, x, 0)
            tl.store(
                Y + (b * M + s) * H + cols, x, offset_state_policy="continuous"
            )


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
    NP: tl.constexpr,
    KIND: tl.constexpr,
    GROUP: tl.constexpr,
    MASK: tl.constexpr,
    PHASE: tl.constexpr,
    ASSIGN: tl.constexpr,
    PREFETCH: tl.constexpr,
):
    A = A.to(tl.pointer_type(tl.uint8))
    if KIND == 2:
        X = X.to(tl.pointer_type(tl.uint32))
        Y = Y.to(tl.pointer_type(tl.uint32))
    elif KIND == 3:
        X = X.to(tl.pointer_type(tl.uint64))
        Y = Y.to(tl.pointer_type(tl.uint64))
    s = tl.arange(0, BM)
    h = tl.arange(0, BH)
    count: tl.constexpr = triton.cdiv(B, NP)
    first = tl.program_id(0) * count if ASSIGN else tl.program_id(0)
    last = tl.minimum(first + count, B) if ASSIGN else B
    step: tl.constexpr = 1 if ASSIGN else NP
    if PREFETCH:
        preload = tl.minimum(first, B - 1)
        piped_off = tl.load(Offsets + preload * OS).to(tl.int64)
        if PREFETCH == 2:
            piped_length = tl.load(L + preload * LS)
    for b in range(first, last, step):
        if PREFETCH:
            off = piped_off
            next_b = tl.minimum(b + step, last - 1)
            piped_off = tl.load(Offsets + next_b * OS).to(tl.int64)
            if PREFETCH == 2:
                length = piped_length
                piped_length = tl.load(L + next_b * LS)
            else:
                length = tl.load(L + b * LS)
        else:
            off = tl.load(Offsets + b * OS).to(tl.int64)
            length = tl.load(L + b * LS)
        valid = s < length
        row = tl.minimum(tl.maximum(off + s, 0), T - 1)
        if XS == H and XH == 1 and (H == BH) and (M == BM):
            x = tl.load(
                X + row[:, None] * XS + h[None, :] * XH,
                offset_state_policy="locally_continuous",
            )
            x = tl.where(valid[:, None], x, 0)
            tl.store(
                Y + (b * M + s[:, None]) * H + h[None, :],
                x,
                offset_state_policy="continuous",
            )
        else:
            x = tl.load(
                X + row[:, None] * XS + tl.minimum(h[None, :], H - 1) * XH
            )
            x = tl.where(valid[:, None], x, 0)
            if M == BM and H == BH:
                tl.store(Y + (b * M + s[:, None]) * H + h[None, :], x)
            else:
                tl.store(
                    Y + (b * M + s[:, None]) * H + h[None, :],
                    x,
                    (s[:, None] < M) & (h[None, :] < H),
                )
        if MASK == 0:
            tl.store(A + b * M + s, valid.to(tl.uint8), s < M)
    if PHASE:
        A64 = A.to(tl.pointer_type(tl.uint64))
        for b in range(first, last, step):
            r = tl.arange(0, GROUP * 8) * 8
            lens = tl.load(L + b * LS)
            bounded210 = tl.where(lens < 0, 0, tl.where(lens > M, M, lens))
            count210 = bounded210 - r
            shift210 = tl.where(
                count210 >= 8,
                0,
                tl.where(count210 > 0, (8 - count210) * 8, 56),
            )
            value = tl.full(
                (GROUP * 8,), 72340172838076673, tl.uint64
            ) >> shift210.to(tl.uint32)
            value = value * (count210 > 0).to(tl.uint64)
            tl.store(
                A64 + (b * M + r) // 8,
                value,
                r < M,
                offset_state_policy="continuous",
            )
    elif MASK == 0:
        pass
    elif MASK == 1:
        for q in range(tl.program_id(0), triton.cdiv(B * M, GROUP * 64), NP):
            r = (
                q * (GROUP * 64)
                + tl.arange(0, GROUP)[:, None] * 64
                + tl.arange(0, 64)[None, :]
            )
            lengths = tl.load(L + tl.minimum(r // M, B - 1) * LS)
            tl.store(A + r, (r % M < lengths).to(tl.uint8), r < B * M)
    else:
        A64 = A.to(tl.pointer_type(tl.uint64))
        for q in range(tl.program_id(0), triton.cdiv(B * M, GROUP * 64), NP):
            r = (q * (GROUP * 8) + tl.arange(0, GROUP * 8)) * 8
            lengths = tl.load(L + tl.minimum(r // M, B - 1) * LS)
            bounded210 = tl.where(
                lengths < 0, 0, tl.where(lengths > M, M, lengths)
            )
            count210 = bounded210 - r % M
            shift210 = tl.where(
                count210 >= 8,
                0,
                tl.where(count210 > 0, (8 - count210) * 8, 56),
            )
            value = tl.full(
                (GROUP * 8,), 72340172838076673, tl.uint64
            ) >> shift210.to(tl.uint32)
            value = value * (count210 > 0).to(tl.uint64)
            tl.store(
                A64 + r // 8,
                value,
                r < B * M,
                offset_state_policy="continuous",
            )


@triton.jit(do_not_specialize=("X", "Offsets", "L", "Y", "A"))
def _all(X, Offsets, L, Y, A, CFG: tl.constexpr):
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
        CFG[12],
        CFG[13],
        CFG[14],
        CFG[15],
        CFG[16],
    )


def pad_sequence_with_mask(input_emb, offsets, lengths, max_len):
    total, hidden = input_emb.shape
    xs, xh = input_emb.stride()
    b = offsets.shape[0]
    output = input_emb.new_empty((b, max_len, hidden))
    unit = 1
    kind = 0
    if input_emb.dtype in (torch.bfloat16, torch.float16) and xh == 1:
        origin = input_emb.storage_offset()
        if hidden % 4 == 0 and xs % 4 == 0 and (origin % 4 == 0):
            unit = 4
            kind = 3
        elif hidden % 2 == 0 and xs % 2 == 0 and (origin % 2 == 0):
            unit = 2
            kind = 2
    small = (
        kind == 3
        and 1 < b <= 12
        and (max_len == 64)
        and (hidden >= 1024)
        and (xs == hidden)
        and (hidden & hidden - 1 == 0)
        and (b * max_len * hidden < 4194304)
    )
    if small:
        unit = 2
        kind = 2
    h = hidden // unit
    xs = xs // unit
    bm = 1 << (max_len - 1).bit_length()
    bh = 1 << (h - 1).bit_length()
    np = min(b, 12)
    group = min(bm, 8) if kind else 1
    mask_mode = 0
    if (
        (kind == 3 or small)
        and xs == h
        and (bm == max_len)
        and (bh == h)
        and (max_len % 8 == 0)
    ):
        group = min(bm, 16)
        mask_mode = 2
    phase = mask_mode == 2 and b >= 16 and (64 <= max_len <= group * 8)
    assign = b >= 128 or (b >= 64 and bm * bh >= 65536)
    if (
        mask_mode == 2
        and h >= 128
        and (64 <= max_len <= 128)
        and (4194304 <= b * max_len * hidden)
        and (bm * bh <= 65536)
        and (b * max_len <= bm * bh * 8)
    ):
        mask_width = 1 << ((b + 7) // 8 - 1).bit_length()
        mask_storage = input_emb.new_empty(
            (bm * mask_width * 8,), dtype=torch.bool
        )
        attn_mask = mask_storage[: b * max_len]
        np = min(b, 12)
        _schedule619.run(
            input_emb,
            offsets,
            lengths,
            output,
            attn_mask,
            b,
            max_len,
            h * 2,
            total,
            offsets.stride(0),
            lengths.stride(0),
            bm,
            bh * 2,
            np,
            1,
            mask_width,
            num_warps=8,
            groups_per_cluster=min(bm, 64 if h <= 512 else 32),
            isCloseMemoryAsync=False,
            unroll_num=16,
            grid=(np + 1,),
            warmup=False,
        )
    else:
        attn_mask = input_emb.new_empty((b * max_len,), dtype=torch.bool)
        _all.run(
            input_emb,
            offsets,
            lengths,
            output,
            attn_mask,
            (
                b,
                max_len,
                h,
                total,
                xs,
                xh,
                offsets.stride(0),
                lengths.stride(0),
                bm,
                bh,
                np,
                kind,
                group,
                mask_mode,
                phase,
                assign,
                2 if b >= 128 else int(assign),
            ),
            num_warps=4,
            groups_per_cluster=group,
            isCloseMemoryAsync=False,
            unroll_num=16 if small else 8,
            grid=(np,),
            warmup=False,
        )
    return (b, output, attn_mask)


__all__ = ["pad_sequence_with_mask"]
