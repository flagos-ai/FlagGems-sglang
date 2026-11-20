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
    if Out is None:
        Out = X
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
def _one_row(X, W, F, P, Out, META: tl.constexpr):
    if Out is None:
        Out = X
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
    row = tl.program_id(0)
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
            F + _wide_mul(pos, SF0, WIDE) + _wide_mul(j // 2, SF1, WIDE) + 1,
            (j >= 0) & (c < D),
            0,
        )
        signed = tl.where(j % 2 == 0, -z, z)
        rotated = y * fr + signed * fi
        y = tl.where(j >= 0, rotated, y)
    tl.store(Out + _wide_mul(row, D, WIDE) + c, y, c < D)


@triton.jit
def _parts3_row(X, W, F, P, Out, row, META: tl.constexpr):
    if Out is None:
        Out = X
    D: tl.constexpr = META[1]
    SX: tl.constexpr = META[3]
    SW: tl.constexpr = META[4]
    SF0: tl.constexpr = META[5]
    SF1: tl.constexpr = META[6]
    SP: tl.constexpr = META[7]
    EPS: tl.constexpr = META[8]
    HAS_W: tl.constexpr = META[9]
    HAS_P: tl.constexpr = META[10]
    WIDE: tl.constexpr = META[12]
    NF: tl.constexpr = META[13]
    c = tl.arange(0, 256)
    prefix = tl.arange(0, 512)
    xb = X + _wide_mul(row, SX, WIDE)
    xp = tl.load(xb + prefix)
    x0, x1 = tl.split(tl.reshape(xp, (256, 2)))
    x0 = x0.to(tl.float32)
    x1 = x1.to(tl.float32)
    x2 = tl.load(xb + c + 512, c < 64, 0).to(tl.float32)
    ss = x0 * x0 + x1 * x1 + x2 * x2
    inv = tl.rsqrt(tl.sum(ss, 0) / 576 + EPS)
    y0 = x0 * inv
    y1 = x1 * inv
    y2 = x2 * inv
    if HAS_W:
        wp = tl.load(W + _wide_mul(prefix, SW, WIDE)).to(tl.float32)
        w0, w1 = tl.split(tl.reshape(wp, (256, 2)))
        w2 = tl.load(W + _wide_mul(c + 512, SW, WIDE), c < 64, 0).to(
            tl.float32
        )
        y0 = y0 * w0
        y1 = y1 * w1
        y2 = y2 * w2
    partner = tl.reshape(tl.flip(tl.reshape(y2, (128, 2)), 1), (256,))
    pos = row
    if HAS_P:
        pos = tl.load(P + _wide_mul(row, SP, WIDE))
        pos = tl.where(pos < 0, pos + NF, pos)
    fb = F + _wide_mul(pos, SF0, WIDE)
    fc = _wide_mul(c // 2, SF1 - 2, WIDE)
    component = tl.load(fb + fc + c, c < 64, 0)
    other_component = tl.reshape(
        tl.flip(tl.reshape(component, (128, 2)), 1), (256,)
    )
    fr = tl.where(c % 2 == 0, component, other_component)
    fi = tl.where(c % 2 == 0, other_component, component)
    signed = tl.where(c % 2 == 0, -partner, partner)
    y2 = y2 * fr + signed * fi
    ob = Out + _wide_mul(row, D, WIDE)
    yp = tl.reshape(tl.join(y0, y1), (512,))
    tl.store(ob + prefix, yp)
    tl.store(ob + c + 512, y2, c < 64)


@triton.jit
def _parts3_loop(X, W, F, P, Out, META: tl.constexpr):
    M: tl.constexpr = META[0]
    for row in range(tl.program_id(0), M, tl.num_programs(0)):
        _parts3_row(X, W, F, P, Out, row, META)


@triton.jit
def _parts3_once(X, W, F, P, Out, META: tl.constexpr):
    _parts3_row(X, W, F, P, Out, tl.program_id(0), META)


@triton.jit
def _fused_fpointer(X, W, F: "*fp32", P, Out, META: tl.constexpr):
    _fused(X, W, F, P, Out, META)


@triton.jit
def _one_row_fpointer(X, W, F: "*fp32", P, Out, META: tl.constexpr):
    _one_row(X, W, F, P, Out, META)


@triton.jit
def _parts3_loop_fpointer(X, W, F: "*fp32", P, Out, META: tl.constexpr):
    _parts3_loop(X, W, F, P, Out, META)


@triton.jit
def _parts3_once_fpointer(X, W, F: "*fp32", P, Out, META: tl.constexpr):
    _parts3_once(X, W, F, P, Out, META)


def fused_norm_rope_inplace(kv, weight, eps, freqs_cis, positions):
    m, d = kv.shape
    nf, half_r = freqs_cis.shape
    r = 2 * half_r
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
    if m == 0 or d == 0:
        return out
    sx = kv.stride(0)
    has_w = weight is not None
    has_p = positions is not None
    sw = weight.stride(0) if has_w else 0
    sp = positions.stride(0) if has_p else 0
    sf, sf1 = freqs_cis.stride()
    xspan = (m - 1) * sx + (d - 1)
    ospan = m * d - 1
    wspan = (d - 1) * sw
    fspan = 2 * (max(0, nf - 1) * sf + max(0, half_r - 1) * sf1) + 1
    pspan = (m - 1) * sp
    wide = max(xspan, ospan, wspan, fspan, pspan) > 2147483647
    f = torch.view_as_real(freqs_cis)
    b = 1 << (d - 1).bit_length()
    kernel = (
        (_parts3_once if m <= 32768 else _parts3_loop)
        if kv.dtype == torch.bfloat16 and d == 576 and (r == 64)
        else _one_row if 1024 <= m <= 32768 else _fused
    )
    if freqs_cis.dtype == torch.complex64:
        kernel = (
            _parts3_once_fpointer
            if kernel is _parts3_once
            else (
                _parts3_loop_fpointer
                if kernel is _parts3_loop
                else (
                    _one_row_fpointer
                    if kernel is _one_row
                    else _fused_fpointer
                )
            )
        )
    kernel.run(
        kv,
        weight,
        f,
        positions,
        None if can_inplace else out,
        (m, d, r, sx, sw, 2 * sf, 2 * sf1, sp, eps, has_w, has_p, b, wide, nf),
        num_warps=4,
        enable_fp_fusion=False,
        grid=(min(m, 32768),),
        warmup=False,
    )
    return out


__all__ = ["fused_norm_rope_inplace"]
