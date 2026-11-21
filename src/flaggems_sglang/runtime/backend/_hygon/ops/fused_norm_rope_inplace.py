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
from triton.language import float32 as fp32


@triton.jit
def _fp32_pair_bits(a, b):
    lo = a.to(tl.uint32, bitcast=True).to(tl.uint64)
    hi = b.to(tl.uint32, bitcast=True).to(tl.uint64)
    return lo | hi << 32


@triton.jit
def _affine_pair_scalar(x0, x1, i0, i1, w0, w1):
    x = _fp32_pair_bits(x0, x1)
    inv = _fp32_pair_bits(i0, i1)
    weight = _fp32_pair_bits(w0, w1)
    normalized = tl.inline_asm_elementwise(
        "v_pk_mul_f32 $0, $1, $2;",
        constraints="=v,v,v",
        args=[x, inv],
        dtype=tl.uint64,
        is_pure=True,
        pack=1,
    )
    affine = tl.inline_asm_elementwise(
        "v_pk_mul_f32 $0, $1, $2;",
        constraints="=v,v,v",
        args=[normalized, weight],
        dtype=tl.uint64,
        is_pure=True,
        pack=1,
    )
    first = affine.to(tl.uint32).to(tl.float32, bitcast=True)
    second = (affine >> 32).to(tl.uint32).to(tl.float32, bitcast=True)
    return (first, second)


@triton.jit
def _prefix_affine(x, inv, weight):
    if x.numel >= 512:
        result = tl.map_elementwise(
            _affine_pair_scalar, x, inv, weight, pack=2
        )
        return result[0]
    else:
        return x * inv * weight


def _requires_wide(
    kv, weight, freqs_cis, positions, packed=False, freq64=False
):
    m, d = kv.shape
    divisor = 2 if packed else 1
    xspan = max(0, m - 1) * (kv.stride(0) // divisor) + max(
        0, d // divisor - 1
    )
    ospan = max(0, m * d // divisor - 1)
    wspan = (
        max(0, d - 1) * weight.stride(0) // divisor
        if weight is not None
        else 0
    )
    fspan = max(0, freqs_cis.shape[0] - 1) * freqs_cis.stride(0) + max(
        0, freqs_cis.shape[1] - 1
    ) * freqs_cis.stride(1)
    fspan = fspan if freq64 else 2 * fspan + 1
    pspan = max(0, m - 1) * positions.stride(0) if positions is not None else 0
    return max(xspan, ospan, wspan, fspan, pspan) > 2147483647


@triton.jit
def _wide_mul(index, stride, WIDE: tl.constexpr):
    if WIDE:
        return index.to(tl.int64) * stride
    else:
        return index * stride


@triton.jit
def _segmented(X, W, F, P, Out, META: tl.constexpr):
    if Out is None:
        Out = X
    if len(META) == 8:
        M: tl.constexpr = META[0]
        D: tl.constexpr = META[1]
        R: tl.constexpr = META[2]
        EPS: tl.constexpr = META[3]
        HW: tl.constexpr = META[4]
        HP: tl.constexpr = META[5]
        WIDE: tl.constexpr = META[6]
        NF: tl.constexpr = META[7]
        SX: tl.constexpr = D
        SW: tl.constexpr = 1 if HW else 0
        SF0: tl.constexpr = R
        SF1: tl.constexpr = 2
        SP: tl.constexpr = 1 if HP else 0
        BS: tl.constexpr = triton.next_power_of_2(max(1, D - R))
        BT: tl.constexpr = triton.next_power_of_2(max(2, R))
        BR: tl.constexpr = 4 if M < 512 else 16
    else:
        M: tl.constexpr = META[0]
        D: tl.constexpr = META[1]
        R: tl.constexpr = META[2]
        SX: tl.constexpr = META[3]
        SW: tl.constexpr = META[4]
        SF0: tl.constexpr = META[5]
        SF1: tl.constexpr = META[6]
        SP: tl.constexpr = META[7]
        EPS: tl.constexpr = META[8]
        HW: tl.constexpr = META[9]
        HP: tl.constexpr = META[10]
        BS: tl.constexpr = META[11]
        BT: tl.constexpr = META[12]
        BR: tl.constexpr = META[13]
        WIDE: tl.constexpr = META[14]
        NF: tl.constexpr = META[15]
    for block in range(tl.program_id(0), tl.cdiv(M, BR), tl.num_programs(0)):
        rows = block * BR + tl.arange(0, BR)
        ss = tl.full((BR,), 0, tl.float32)
        if D > R:
            prefix = tl.arange(0, BS)
            xp = tl.load(
                X + _wide_mul(rows[:, None], SX, WIDE) + prefix[None, :],
                (rows[:, None] < M) & (prefix[None, :] < D - R),
                0,
            ).to(tl.float32)
            ss += tl.sum(xp * xp, 1)
        if R > 0:
            tail = tl.arange(0, BT)
            xt = tl.load(
                X + _wide_mul(rows[:, None], SX, WIDE) + D - R + tail[None, :],
                (rows[:, None] < M) & (tail[None, :] < R),
                0,
            ).to(tl.float32)
            ss += tl.sum(xt * xt, 1)
            if BR == 16:
                pos = rows
                if HP:
                    pos = tl.load(P + _wide_mul(rows, SP, WIDE), rows < M, 0)
                    pos = tl.where(pos < 0, pos + NF, pos)
                ft = tl.load(
                    F
                    + _wide_mul(pos[:, None], SF0, WIDE)
                    + _wide_mul(tail[None, :] // 2, SF1, WIDE)
                    + tail[None, :] % 2,
                    (rows[:, None] < M) & (tail[None, :] < R),
                    0,
                )
                fr, fi = tl.split(tl.reshape(ft, (BR, BT // 2, 2)))
        inv = tl.rsqrt(ss / D + EPS)
        if D > R:
            if HW and X.dtype.element_ty != tl.float16:
                wp = tl.load(
                    W + _wide_mul(prefix, SW, WIDE), prefix < D - R, 0
                ).to(tl.float32)
                yp = _prefix_affine(xp, inv[:, None], wp[None, :])
            else:
                yp = xp * inv[:, None]
                if HW:
                    wp = tl.load(
                        W + _wide_mul(prefix, SW, WIDE), prefix < D - R, 0
                    ).to(tl.float32)
                    yp *= wp[None, :]
            tl.store(
                Out + _wide_mul(rows[:, None], D, WIDE) + prefix[None, :],
                yp,
                (rows[:, None] < M) & (prefix[None, :] < D - R),
            )
        if R > 0:
            yt = xt * inv[:, None]
            if HW:
                wt = tl.load(
                    W + _wide_mul(D - R + tail, SW, WIDE), tail < R, 0
                ).to(tl.float32)
                yt *= wt[None, :]
            yr, yi = tl.split(tl.reshape(yt, (BR, BT // 2, 2)))
            if BR != 16:
                pos = rows
                if HP:
                    pos = tl.load(P + _wide_mul(rows, SP, WIDE), rows < M, 0)
                    pos = tl.where(pos < 0, pos + NF, pos)
                ft = tl.load(
                    F
                    + _wide_mul(pos[:, None], SF0, WIDE)
                    + _wide_mul(tail[None, :] // 2, SF1, WIDE)
                    + tail[None, :] % 2,
                    (rows[:, None] < M) & (tail[None, :] < R),
                    0,
                )
                fr, fi = tl.split(tl.reshape(ft, (BR, BT // 2, 2)))
            yr_out = yr * fr - yi * fi
            yi_out = yr * fi + yi * fr
            y = tl.reshape(tl.join(yr_out, yi_out), (BR, BT))
            tl.store(
                Out
                + _wide_mul(rows[:, None], D, WIDE)
                + D
                - R
                + tail[None, :],
                y,
                (rows[:, None] < M) & (tail[None, :] < R),
            )


@triton.jit
def _segmented_once(X, W, F, P, Out, META: tl.constexpr):
    if Out is None:
        Out = X
    if len(META) == 8:
        M: tl.constexpr = META[0]
        D: tl.constexpr = META[1]
        R: tl.constexpr = META[2]
        EPS: tl.constexpr = META[3]
        HW: tl.constexpr = META[4]
        HP: tl.constexpr = META[5]
        WIDE: tl.constexpr = META[6]
        NF: tl.constexpr = META[7]
        SX: tl.constexpr = D
        SW: tl.constexpr = 1 if HW else 0
        SF0: tl.constexpr = R
        SF1: tl.constexpr = 2
        SP: tl.constexpr = 1 if HP else 0
        BS: tl.constexpr = triton.next_power_of_2(max(1, D - R))
        BT: tl.constexpr = triton.next_power_of_2(max(2, R))
        BR: tl.constexpr = 4 if M < 512 else 16
    else:
        M: tl.constexpr = META[0]
        D: tl.constexpr = META[1]
        R: tl.constexpr = META[2]
        SX: tl.constexpr = META[3]
        SW: tl.constexpr = META[4]
        SF0: tl.constexpr = META[5]
        SF1: tl.constexpr = META[6]
        SP: tl.constexpr = META[7]
        EPS: tl.constexpr = META[8]
        HW: tl.constexpr = META[9]
        HP: tl.constexpr = META[10]
        BS: tl.constexpr = META[11]
        BT: tl.constexpr = META[12]
        BR: tl.constexpr = META[13]
        WIDE: tl.constexpr = META[14]
        NF: tl.constexpr = META[15]
    block = tl.program_id(0)
    rows = block * BR + tl.arange(0, BR)
    ss = tl.full((BR,), 0, tl.float32)
    if D > R:
        prefix = tl.arange(0, BS)
        xp = tl.load(
            X + _wide_mul(rows[:, None], SX, WIDE) + prefix[None, :],
            (rows[:, None] < M) & (prefix[None, :] < D - R),
            0,
        ).to(tl.float32)
        ss += tl.sum(xp * xp, 1)
    if R > 0:
        tail = tl.arange(0, BT)
        xt = tl.load(
            X + _wide_mul(rows[:, None], SX, WIDE) + D - R + tail[None, :],
            (rows[:, None] < M) & (tail[None, :] < R),
            0,
        ).to(tl.float32)
        ss += tl.sum(xt * xt, 1)
        if BR == 16:
            pos = rows
            if HP:
                pos = tl.load(P + _wide_mul(rows, SP, WIDE), rows < M, 0)
                pos = tl.where(pos < 0, pos + NF, pos)
            ft = tl.load(
                F
                + _wide_mul(pos[:, None], SF0, WIDE)
                + _wide_mul(tail[None, :] // 2, SF1, WIDE)
                + tail[None, :] % 2,
                (rows[:, None] < M) & (tail[None, :] < R),
                0,
            )
            fr, fi = tl.split(tl.reshape(ft, (BR, BT // 2, 2)))
    inv = tl.rsqrt(ss / D + EPS)
    if D > R:
        if HW and X.dtype.element_ty != tl.float16:
            wp = tl.load(
                W + _wide_mul(prefix, SW, WIDE), prefix < D - R, 0
            ).to(tl.float32)
            yp = _prefix_affine(xp, inv[:, None], wp[None, :])
        else:
            yp = xp * inv[:, None]
            if HW:
                wp = tl.load(
                    W + _wide_mul(prefix, SW, WIDE), prefix < D - R, 0
                ).to(tl.float32)
                yp *= wp[None, :]
        tl.store(
            Out + _wide_mul(rows[:, None], D, WIDE) + prefix[None, :],
            yp,
            (rows[:, None] < M) & (prefix[None, :] < D - R),
        )
    if R > 0:
        yt = xt * inv[:, None]
        if HW:
            wt = tl.load(
                W + _wide_mul(D - R + tail, SW, WIDE), tail < R, 0
            ).to(tl.float32)
            yt *= wt[None, :]
        yr, yi = tl.split(tl.reshape(yt, (BR, BT // 2, 2)))
        if BR != 16:
            pos = rows
            if HP:
                pos = tl.load(P + _wide_mul(rows, SP, WIDE), rows < M, 0)
                pos = tl.where(pos < 0, pos + NF, pos)
            ft = tl.load(
                F
                + _wide_mul(pos[:, None], SF0, WIDE)
                + _wide_mul(tail[None, :] // 2, SF1, WIDE)
                + tail[None, :] % 2,
                (rows[:, None] < M) & (tail[None, :] < R),
                0,
            )
            fr, fi = tl.split(tl.reshape(ft, (BR, BT // 2, 2)))
        yr_out = yr * fr - yi * fi
        yi_out = yr * fi + yi * fr
        y = tl.reshape(tl.join(yr_out, yi_out), (BR, BT))
        tl.store(
            Out + _wide_mul(rows[:, None], D, WIDE) + D - R + tail[None, :],
            y,
            (rows[:, None] < M) & (tail[None, :] < R),
        )


@triton.jit
def _segmented_frequency(X, W, F: "*fp32", P, Out, META: tl.constexpr):
    _segmented(X, W, F, P, Out, META)


@triton.jit
def _segmented_once_frequency(X, W, F: "*fp32", P, Out, META: tl.constexpr):
    _segmented_once(X, W, F, P, Out, META)


def fused_norm_rope_inplace(kv, weight, eps, freqs_cis, positions):
    m, d = kv.shape
    nf, half_r = freqs_cis.shape
    r = 2 * half_r
    if m == 0 or d == 0:
        return torch.empty_like(kv, memory_format=torch.contiguous_format)
    storage = kv.untyped_storage().data_ptr()
    can_inplace = (
        kv.is_contiguous()
        and (weight is None or weight.untyped_storage().data_ptr() != storage)
        and (
            freqs_cis is None
            or freqs_cis.untyped_storage().data_ptr() != storage
        )
        and (
            positions is None
            or positions.untyped_storage().data_ptr() != storage
        )
    )
    out = (
        kv
        if can_inplace
        else torch.empty_like(kv, memory_format=torch.contiguous_format)
    )
    sx = kv.stride(0)
    hw = weight is not None
    hp = positions is not None
    sw = weight.stride(0) if hw else 0
    sp = positions.stride(0) if hp else 0
    sf, sf1 = freqs_cis.stride()
    dense = (
        sx == d
        and sf == half_r
        and (sf1 == 1)
        and (not hw or sw == 1)
        and (not hp or sp == 1)
    )
    if dense and nf > 0 and half_r > 0:
        WIDE = max(m * d, 2 * nf * half_r) > 2147483648
    else:
        xspan = (m - 1) * sx + (d - 1)
        ospan = m * d - 1
        wspan = (d - 1) * sw
        fspan = 2 * (max(0, nf - 1) * sf + max(0, half_r - 1) * sf1) + 1
        pspan = (m - 1) * sp
        WIDE = max(xspan, ospan, wspan, fspan, pspan) > 2147483647
    f = torch.view_as_real(freqs_cis)
    br = 4 if m < 512 else 16
    blocks = (m + br - 1) // br
    kernel = (
        (
            _segmented_once_frequency
            if blocks <= 32768
            else _segmented_frequency
        )
        if freqs_cis.dtype == torch.complex64
        else _segmented_once if blocks <= 32768 else _segmented
    )
    meta = (
        (m, d, r, eps, hw, hp, WIDE, nf)
        if dense
        else (
            m,
            d,
            r,
            sx,
            sw,
            2 * sf,
            2 * sf1,
            sp,
            eps,
            hw,
            hp,
            1 << (max(1, d - r) - 1).bit_length(),
            1 << (max(2, r) - 1).bit_length(),
            br,
            WIDE,
            nf,
        )
    )
    kernel.run(
        kv,
        weight,
        f,
        positions,
        None if can_inplace else out,
        meta,
        num_warps=4,
        enable_fp_fusion=False,
        grid=(min(blocks, 32768),),
        warmup=False,
    )
    return out


__all__ = ["fused_norm_rope_inplace"]
