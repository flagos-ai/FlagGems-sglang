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
from triton.language import bfloat16 as bf16
from triton.language import float32 as fp32


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
def _fused(X, W, F, P, Out, META: tl.constexpr):
    if len(META) == 8:
        M: tl.constexpr = META[0]
        D: tl.constexpr = META[1]
        R: tl.constexpr = META[2]
        EPS: tl.constexpr = META[3]
        HAS_W: tl.constexpr = META[4]
        HAS_P: tl.constexpr = META[5]
        WIDE: tl.constexpr = META[6]
        NF: tl.constexpr = META[7]
        SX: tl.constexpr = D
        SW: tl.constexpr = 1 if HAS_W else 0
        SF0: tl.constexpr = R
        SF1: tl.constexpr = 2
        SP: tl.constexpr = 1 if HAS_P else 0
        B: tl.constexpr = triton.next_power_of_2(D)
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
        HAS_W: tl.constexpr = META[9]
        HAS_P: tl.constexpr = META[10]
        B: tl.constexpr = META[11]
        WIDE: tl.constexpr = META[12]
        NF: tl.constexpr = META[13]
    c = tl.arange(0, B)
    for row in range(tl.program_id(0), M, tl.num_programs(0)):
        x = tl.load(X + _wide_mul(row, SX, WIDE) + c, c < D, 0).to(tl.float32)
        inv = tl.rsqrt(tl.sum(x * x, 0) / D + EPS)
        y = x * inv
        if HAS_W:
            w = tl.load(W + _wide_mul(c, SW, WIDE), c < D, 0).to(tl.float32)
            y = y * w
        if R > 0:
            j = c - (D - R)
            if D % 2 == 0 and B <= 1024:
                z = tl.reshape(tl.flip(tl.reshape(y, (B // 2, 2)), 1), (B,))
            else:
                partner = D - R + (j ^ 1)
                z = (
                    tl.load(
                        X + _wide_mul(row, SX, WIDE) + partner,
                        (j >= 0) & (c < D),
                        0,
                    ).to(tl.float32)
                    * inv
                )
                if HAS_W:
                    wp = tl.load(
                        W + _wide_mul(partner, SW, WIDE), (j >= 0) & (c < D), 0
                    ).to(tl.float32)
                    z = z * wp
            pos = row
            if HAS_P:
                pos = tl.load(P + _wide_mul(row, SP, WIDE))
                pos = tl.where(pos < 0, pos + NF, pos)
            fr = tl.load(
                F + _wide_mul(pos, SF0, WIDE) + _wide_mul(j // 2, SF1, WIDE),
                (j >= 0) & (c < D),
                0,
            )
            fi = tl.load(
                F
                + _wide_mul(pos, SF0, WIDE)
                + _wide_mul(j // 2, SF1, WIDE)
                + 1,
                (j >= 0) & (c < D),
                0,
            )
            signed = tl.where(j % 2 == 0, -z, z)
            rotated = y * fr + signed * fi
            y = tl.where(j >= 0, rotated, y)
        tl.store(Out + _wide_mul(row, D, WIDE) + c, y, c < D)


@triton.jit
def _fused_pair(X, W, F, P, Out, META: tl.constexpr):
    if len(META) == 8:
        M: tl.constexpr = META[0]
        D: tl.constexpr = META[1]
        R: tl.constexpr = META[2]
        EPS: tl.constexpr = META[3]
        HAS_W: tl.constexpr = META[4]
        HAS_P: tl.constexpr = META[5]
        WIDE: tl.constexpr = META[6]
        NF: tl.constexpr = META[7]
        SX: tl.constexpr = D
        SW: tl.constexpr = 1 if HAS_W else 0
        SF0: tl.constexpr = R
        SF1: tl.constexpr = 2
        SP: tl.constexpr = 1 if HAS_P else 0
        B: tl.constexpr = triton.next_power_of_2(D)
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
        HAS_W: tl.constexpr = META[9]
        HAS_P: tl.constexpr = META[10]
        B: tl.constexpr = META[11]
        WIDE: tl.constexpr = META[12]
        NF: tl.constexpr = META[13]
    c = tl.arange(0, B)
    for row in range(tl.program_id(0), M, tl.num_programs(0)):
        x = tl.load(
            X + _wide_mul(row, SX, WIDE) + c, c < D, 0, cache_modifier=".cg"
        ).to(tl.float32)
        inv = tl.rsqrt(tl.sum(x * x, 0) / D + EPS)
        y = x * inv
        if HAS_W:
            w = tl.load(W + _wide_mul(c, SW, WIDE), c < D, 0).to(tl.float32)
            y = y * w
        if R > 0 and D % 2 == 0 and (B <= 1024):
            pair = tl.arange(0, B // 2)
            yr, yi = tl.split(tl.reshape(y, (B // 2, 2)))
            jpair = pair - (D - R) // 2
            pos = row
            if HAS_P:
                pos = tl.load(P + _wide_mul(row, SP, WIDE))
                pos = tl.where(pos < 0, pos + NF, pos)
            fr = tl.load(
                F + _wide_mul(pos, SF0, WIDE) + _wide_mul(jpair, SF1, WIDE),
                (jpair >= 0) & (pair < D // 2),
                0,
            )
            fi = tl.load(
                F
                + _wide_mul(pos, SF0, WIDE)
                + _wide_mul(jpair, SF1, WIDE)
                + 1,
                (jpair >= 0) & (pair < D // 2),
                0,
            )
            rr = yr * fr - yi * fi
            ri = yr * fi + yi * fr
            rr = tl.where(jpair >= 0, rr, yr)
            ri = tl.where(jpair >= 0, ri, yi)
            y = tl.reshape(tl.join(rr, ri), (B,))
        elif R > 0:
            j = c - (D - R)
            if D % 2 == 0 and B <= 1024:
                z = tl.reshape(tl.flip(tl.reshape(y, (B // 2, 2)), 1), (B,))
            else:
                partner = D - R + (j ^ 1)
                z = (
                    tl.load(
                        X + _wide_mul(row, SX, WIDE) + partner,
                        (j >= 0) & (c < D),
                        0,
                    ).to(tl.float32)
                    * inv
                )
                if HAS_W:
                    wp = tl.load(
                        W + _wide_mul(partner, SW, WIDE), (j >= 0) & (c < D), 0
                    ).to(tl.float32)
                    z = z * wp
            pos = row
            if HAS_P:
                pos = tl.load(P + _wide_mul(row, SP, WIDE))
                pos = tl.where(pos < 0, pos + NF, pos)
            fr = tl.load(
                F + _wide_mul(pos, SF0, WIDE) + _wide_mul(j // 2, SF1, WIDE),
                (j >= 0) & (c < D),
                0,
            )
            fi = tl.load(
                F
                + _wide_mul(pos, SF0, WIDE)
                + _wide_mul(j // 2, SF1, WIDE)
                + 1,
                (j >= 0) & (c < D),
                0,
            )
            signed = tl.where(j % 2 == 0, -z, z)
            rotated = y * fr + signed * fi
            y = tl.where(j >= 0, rotated, y)
        tl.store(
            Out + _wide_mul(row, D, WIDE) + c, y, c < D, cache_modifier=".cs"
        )


@triton.jit
def _fused_medium_xcg(X, W, F, P, Out, META: tl.constexpr):
    if len(META) == 8:
        M: tl.constexpr = META[0]
        D: tl.constexpr = META[1]
        R: tl.constexpr = META[2]
        EPS: tl.constexpr = META[3]
        HAS_W: tl.constexpr = META[4]
        HAS_P: tl.constexpr = META[5]
        WIDE: tl.constexpr = META[6]
        NF: tl.constexpr = META[7]
        SX: tl.constexpr = D
        SW: tl.constexpr = 1 if HAS_W else 0
        SF0: tl.constexpr = R
        SF1: tl.constexpr = 2
        SP: tl.constexpr = 1 if HAS_P else 0
        B: tl.constexpr = triton.next_power_of_2(D)
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
        HAS_W: tl.constexpr = META[9]
        HAS_P: tl.constexpr = META[10]
        B: tl.constexpr = META[11]
        WIDE: tl.constexpr = META[12]
        NF: tl.constexpr = META[13]
    c = tl.arange(0, B)
    for row in range(tl.program_id(0), M, tl.num_programs(0)):
        x = tl.load(
            X + _wide_mul(row, SX, WIDE) + c, c < D, 0, cache_modifier=".cg"
        ).to(tl.float32)
        inv = tl.rsqrt(tl.sum(x * x, 0) / D + EPS)
        y = x * inv
        if HAS_W:
            w = tl.load(W + _wide_mul(c, SW, WIDE), c < D, 0).to(tl.float32)
            y = y * w
        if R > 0:
            j = c - (D - R)
            if D % 2 == 0 and B <= 1024:
                z = tl.reshape(tl.flip(tl.reshape(y, (B // 2, 2)), 1), (B,))
            else:
                partner = D - R + (j ^ 1)
                z = (
                    tl.load(
                        X + _wide_mul(row, SX, WIDE) + partner,
                        (j >= 0) & (c < D),
                        0,
                    ).to(tl.float32)
                    * inv
                )
                if HAS_W:
                    wp = tl.load(
                        W + _wide_mul(partner, SW, WIDE), (j >= 0) & (c < D), 0
                    ).to(tl.float32)
                    z = z * wp
            pos = row
            if HAS_P:
                pos = tl.load(P + _wide_mul(row, SP, WIDE))
                pos = tl.where(pos < 0, pos + NF, pos)
            fr = tl.load(
                F + _wide_mul(pos, SF0, WIDE) + _wide_mul(j // 2, SF1, WIDE),
                (j >= 0) & (c < D),
                0,
            )
            fi = tl.load(
                F
                + _wide_mul(pos, SF0, WIDE)
                + _wide_mul(j // 2, SF1, WIDE)
                + 1,
                (j >= 0) & (c < D),
                0,
            )
            signed = tl.where(j % 2 == 0, -z, z)
            rotated = y * fr + signed * fi
            y = tl.where(j >= 0, rotated, y)
        tl.store(Out + _wide_mul(row, D, WIDE) + c, y, c < D)


@triton.jit
def _fused_fpointer(X, W, F: "*fp32", P, Out, META: tl.constexpr):
    _fused(X, W, F, P, Out, META)


@triton.jit
def _fused_xo_fpointer(
    X: "*bf16", W, F: "*fp32", P, Out: "*bf16", META: tl.constexpr
):
    _fused(X, W, F, P, Out, META)


@triton.jit
def _fused_xwo_fpointer(
    X: "*bf16", W: "*bf16", F: "*fp32", P, Out: "*bf16", META: tl.constexpr
):
    _fused(X, W, F, P, Out, META)


@triton.jit
def _fused_pair_fpointer(X, W, F: "*fp32", P, Out, META: tl.constexpr):
    _fused_pair(X, W, F, P, Out, META)


@triton.jit
def _fused_medium_xcg_fpointer(X, W, F: "*fp32", P, Out, META: tl.constexpr):
    _fused_medium_xcg(X, W, F, P, Out, META)


def fused_norm_rope_inplace(kv, weight, eps, freqs_cis, positions):
    m, d = kv.shape
    nf, half_r = freqs_cis.shape
    r = 2 * half_r
    dtype = kv.dtype
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
    out = kv if can_inplace else kv.new_empty((m, d))
    if m == 0 or d == 0:
        return out
    sx = kv.stride(0)
    has_w = weight is not None
    has_p = positions is not None
    sw = weight.stride(0) if has_w else 0
    sp = positions.stride(0) if has_p else 0
    sf, sf1 = freqs_cis.stride()
    is_hip = torch.version.hip is not None
    if is_hip:
        xspan = max(0, m - 1) * sx + max(0, d - 1)
        ospan = max(0, m * d - 1)
        wspan = max(0, d - 1) * sw
        fspan = 2 * (max(0, nf - 1) * sf + max(0, half_r - 1) * sf1) + 1
        pspan = max(0, m - 1) * sp
    else:
        xspan = (m - 1) * sx + (d - 1)
        ospan = m * d - 1
        wspan = (d - 1) * sw
        fspan = 2 * (max(0, nf - 1) * sf + max(0, half_r - 1) * sf1) + 1
        pspan = (m - 1) * sp
    wide = max(xspan, ospan, wspan, fspan, pspan) > 2147483647
    f = torch.view_as_real(freqs_cis)
    kernel = (
        _fused_pair
        if is_hip
        and m >= 8192
        and (d == 576)
        and (r == 64)
        and (kv.dtype == torch.bfloat16)
        else _fused
    )
    if (
        is_hip
        and 512 <= m < 8192
        and (d == 576)
        and (r == 64)
        and (kv.dtype == torch.bfloat16)
    ):
        kernel = _fused_medium_xcg
    if freqs_cis.dtype == torch.complex64:
        kernel = (
            _fused_pair_fpointer
            if kernel is _fused_pair
            else (
                _fused_medium_xcg_fpointer
                if kernel is _fused_medium_xcg
                else _fused_fpointer
            )
        )
    meta = (
        (m, d, r, eps, has_w, has_p, wide, nf)
        if not is_hip
        and (
            sx == d
            and sf == half_r
            and (sf1 == 1)
            and (not has_w or sw == 1)
            and (not has_p or sp == 1)
        )
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
            has_w,
            has_p,
            1 << (d - 1).bit_length(),
            wide,
            nf,
        )
    )
    x, o, w = kv, out, weight
    if not is_hip and dtype == torch.bfloat16 and kernel is _fused_fpointer:
        x = kv
        o = out
        kernel = _fused_xo_fpointer
        if weight is not None and weight.dtype == torch.bfloat16:
            w = weight
            kernel = _fused_xwo_fpointer
    kernel.run(
        x,
        w,
        f,
        positions,
        o,
        meta,
        num_warps=1,
        enable_fp_fusion=False,
        grid=(min(m, 32768),),
        warmup=False,
    )
    return out


__all__ = ["fused_norm_rope_inplace"]
