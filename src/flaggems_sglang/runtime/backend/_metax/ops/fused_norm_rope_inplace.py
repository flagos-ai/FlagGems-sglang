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
import triton.language.core as core
from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl
from triton.language import float32 as fp32


@core.extern
def _gl_hardware_rsq(x, _semantic=None):
    return core.extern_elementwise(
        "",
        "",
        [x],
        {(core.float32,): ("llvm.mxc.rsq.f32", core.float32)},
        is_pure=True,
        _semantic=_semantic,
    )


@g.jit
def _gl_nonnegative_rsqrt(x, EPS: gl.constexpr):
    if EPS >= 7.888609052210118e-31:
        return 1.0 * _gl_hardware_rsq(1.0 * x)
    else:
        return gl.rsqrt(x)


@g.jit
def _gl_wide_mul(index, stride, WIDE: gl.constexpr):
    if WIDE:
        return index.to(gl.int64) * stride
    else:
        return index * stride


@g.jit
def _gl_partial_row_fold(xp, xt, BR: gl.constexpr, BS: gl.constexpr):
    PAIR: gl.constexpr = gl.BlockedLayout(
        [1, 1, 1], [1, 2, 32], [4, 1, 1], [2, 1, 0]
    )
    PARTIAL: gl.constexpr = gl.BlockedLayout([1, 1], [2, 32], [4, 1], [1, 0])
    prefix_local = gl.sum(gl.reshape(xp * xp, (BR, BS // 256, 32, 8)), 3)
    prefix_partials = gl.sum(prefix_local, 1)
    prefix_even, prefix_odd = gl.split(
        gl.permute(gl.reshape(prefix_partials, (2, 4, 32)), (1, 2, 0))
    )
    prefix_even = gl.convert_layout(
        prefix_even, gl.SliceLayout(1, PAIR), assert_trivial=True
    )
    prefix_odd = gl.convert_layout(
        prefix_odd, gl.SliceLayout(1, PAIR), assert_trivial=True
    )
    pair_index = gl.arange(
        0, 2, layout=gl.SliceLayout(0, gl.SliceLayout(2, PAIR))
    )
    prefix_selected = gl.where(
        pair_index[None, :, None] == 0,
        prefix_even[:, None, :],
        prefix_odd[:, None, :],
    )
    prefix_selected = gl.convert_layout(
        gl.reshape(prefix_selected, (BR, 32)), PARTIAL, assert_trivial=True
    )
    tail_partials = gl.sum(gl.reshape(xt * xt, (BR, 32, 2)), 2)
    tail_partials = gl.convert_layout(
        tail_partials, PARTIAL, assert_trivial=True
    )
    return gl.sum(prefix_selected + tail_partials, 1)


@g.jit
def _gl_segment_row(X, W, F, P, Out, block, META: gl.constexpr):
    M: gl.constexpr = META[0]
    D: gl.constexpr = META[1]
    R: gl.constexpr = META[2]
    SX: gl.constexpr = META[3]
    SW: gl.constexpr = META[4]
    SF0: gl.constexpr = META[5]
    SF1: gl.constexpr = META[6]
    SP: gl.constexpr = META[7]
    EPS: gl.constexpr = META[8]
    HW: gl.constexpr = META[9]
    HP: gl.constexpr = META[10]
    BS: gl.constexpr = META[11]
    BT: gl.constexpr = META[12]
    BR: gl.constexpr = META[13]
    WIDE: gl.constexpr = META[14]
    NF: gl.constexpr = META[15]
    JOIN_PARTIALS: gl.constexpr = (
        BR == 8 and BS >= 256 and (BT == 64) and (R > 0)
    )
    EARLY_W: gl.constexpr = (
        HW and (not HP) and (SX == D) and (SW == 1) and (M >= 16384)
    )
    PREFIX: gl.constexpr = gl.BlockedLayout([1, 8], [1, 64], [4, 1], [1, 0])
    TAIL: gl.constexpr = gl.BlockedLayout([1, 2], [2, 32], [4, 1], [1, 0])
    WEIGHT: gl.constexpr = gl.BlockedLayout([2], [64], [4], [0])
    rows = block * BR + gl.arange(0, BR, layout=gl.SliceLayout(1, TAIL))
    ss = gl.full((BR,), 0, gl.float32, layout=gl.SliceLayout(1, TAIL))
    if D > R:
        prefix = gl.arange(0, BS, layout=gl.SliceLayout(0, PREFIX))
        prefix_logical = gl.arange(0, BR, layout=gl.SliceLayout(1, PREFIX))
        prefix_actual = prefix_logical % 4 * (BR // 4) + prefix_logical // 4
        prefix_rows = block * BR + prefix_actual
        xp = gl.load(
            X + _gl_wide_mul(prefix_rows[:, None], SX, WIDE) + prefix[None, :],
            (prefix_rows[:, None] < M) & (prefix[None, :] < D - R),
            0,
        ).to(gl.float32)
        if HW and EARLY_W:
            weight_prefix = gl.arange(0, BS, layout=WEIGHT)
            wp_bits = gl.load(
                W + _gl_wide_mul(weight_prefix, SW, WIDE),
                weight_prefix < D - R,
                0,
            )
        if not JOIN_PARTIALS:
            prefix_sum = gl.sum(xp * xp, 1)
            prefix_sum_ordered = gl.reshape(
                gl.permute(gl.reshape(prefix_sum, (BR // 4, 4)), (1, 0)), (BR,)
            )
            ss += gl.convert_layout(
                prefix_sum_ordered, gl.SliceLayout(1, TAIL)
            )
    if R > 0:
        tail = gl.arange(0, BT, layout=gl.SliceLayout(0, TAIL))
        xt = gl.load(
            X + _gl_wide_mul(rows[:, None], SX, WIDE) + D - R + tail[None, :],
            (rows[:, None] < M) & (tail[None, :] < R),
            0,
        ).to(gl.float32)
        if JOIN_PARTIALS:
            ss += gl.convert_layout(
                _gl_partial_row_fold(xp, xt, BR, BS),
                gl.SliceLayout(1, TAIL),
                assert_trivial=True,
            )
        else:
            ss += gl.sum(xt * xt, 1)
    inv = _gl_nonnegative_rsqrt(ss / D + EPS, EPS)
    if D > R:
        inverse_ordered = gl.reshape(
            gl.permute(gl.reshape(inv, (4, BR // 4)), (1, 0)), (BR,)
        )
        prefix_inv = gl.convert_layout(
            inverse_ordered, gl.SliceLayout(1, PREFIX)
        )
        yp = xp * prefix_inv[:, None]
        if HW:
            if not EARLY_W:
                weight_prefix = gl.arange(0, BS, layout=WEIGHT)
                wp_bits = gl.load(
                    W + _gl_wide_mul(weight_prefix, SW, WIDE),
                    weight_prefix < D - R,
                    0,
                )
            wp = gl.convert_layout(wp_bits, gl.SliceLayout(0, PREFIX)).to(
                gl.float32
            )
            yp *= wp[None, :]
        gl.store(
            Out
            + _gl_wide_mul(prefix_rows[:, None], D, WIDE)
            + prefix[None, :],
            yp,
            (prefix_rows[:, None] < M) & (prefix[None, :] < D - R),
        )
    if R > 0:
        yt = xt * inv[:, None]
        if HW:
            wt = gl.load(
                W + _gl_wide_mul(D - R + tail, SW, WIDE), tail < R, 0
            ).to(gl.float32)
            yt *= wt[None, :]
        yr, yi = gl.split(gl.reshape(yt, (BR, BT // 2, 2)))
        pos = rows
        if HP:
            pos = gl.load(P + _gl_wide_mul(rows, SP, WIDE), rows < M, 0)
            pos = gl.where(pos < 0, pos + NF, pos)
        ft = gl.load(
            F
            + _gl_wide_mul(pos[:, None], SF0, WIDE)
            + _gl_wide_mul(tail[None, :] // 2, SF1, WIDE)
            + tail[None, :] % 2,
            (rows[:, None] < M) & (tail[None, :] < R),
            0,
        )
        fr, fi = gl.split(gl.reshape(ft, (BR, BT // 2, 2)))
        yr_out = gl.fma(yr, fr, -0.0) - gl.fma(yi, fi, -0.0)
        yi_out = gl.fma(yr, fi, -0.0) + gl.fma(yi, fr, -0.0)
        y = gl.reshape(gl.join(yr_out, yi_out), (BR, BT))
        gl.store(
            Out + _gl_wide_mul(rows[:, None], D, WIDE) + D - R + tail[None, :],
            y,
            (rows[:, None] < M) & (tail[None, :] < R),
        )


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
def _fused(
    X,
    W,
    F,
    P,
    O,
    M: tl.constexpr,
    D: tl.constexpr,
    R: tl.constexpr,
    SX: tl.constexpr,
    SW: tl.constexpr,
    SF0: tl.constexpr,
    SF1: tl.constexpr,
    SP: tl.constexpr,
    EPS: tl.constexpr,
    HAS_W: tl.constexpr,
    HAS_P: tl.constexpr,
    B: tl.constexpr,
    WIDE: tl.constexpr,
    NF: tl.constexpr,
):
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
            rotated = tl.fma(y, fr, -0.0) + tl.fma(signed, fi, -0.0)
            y = tl.where(j >= 0, rotated, y)
        tl.store(O + _wide_mul(row, D, WIDE) + c, y, c < D)


@triton.jit
def _fused_once(X, W, F, P, Out, META: tl.constexpr):
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
        partner = D - R + (j ^ 1)
        z = (
            tl.load(
                X + _wide_mul(row, SX, WIDE) + partner, (j >= 0) & (c < D), 0
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
        rotated = tl.fma(y, fr, -0.0) + tl.fma(signed, fi, -0.0)
        y = tl.where(j >= 0, rotated, y)
    tl.store(Out + _wide_mul(row, D, WIDE) + c, y, c < D)


@g.jit
def _segmented(X, W, F, P, Out, META: gl.constexpr):
    M: gl.constexpr = META[0]
    BR: gl.constexpr = META[13]
    for block in range(
        gl.program_id(0), triton.cdiv(M, BR), gl.num_programs(0)
    ):
        _gl_segment_row(X, W, F, P, Out, block, META)


@g.jit
def _segmented_once(X, W, F, P, Out, META: gl.constexpr):
    _gl_segment_row(X, W, F, P, Out, gl.program_id(0), META)


@triton.jit
def _small_parts3(X, W, F, P, Out, META: tl.constexpr):
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
    row = tl.program_id(0)
    c = tl.arange(0, 256)
    xb = X + _wide_mul(row, SX, WIDE)
    x0 = tl.load(xb + c).to(tl.float32)
    x1 = tl.load(xb + c + 256).to(tl.float32)
    x2 = tl.load(xb + c + 512, c < 64, 0).to(tl.float32)
    ss = x0 * x0 + x1 * x1 + x2 * x2
    inv = tl.rsqrt(tl.sum(ss, 0) / 576 + EPS)
    y0 = x0 * inv
    y1 = x1 * inv
    y2 = x2 * inv
    if HAS_W:
        w0 = tl.load(W + _wide_mul(c, SW, WIDE)).to(tl.float32)
        w1 = tl.load(W + _wide_mul(c + 256, SW, WIDE)).to(tl.float32)
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
    fc = _wide_mul(c // 2, SF1, WIDE)
    fr = tl.load(fb + fc, c < 64, 0)
    fi = tl.load(fb + fc + 1, c < 64, 0)
    signed = tl.where(c % 2 == 0, -partner, partner)
    y2 = tl.fma(y2, fr, -0.0) + tl.fma(signed, fi, -0.0)
    ob = Out + _wide_mul(row, D, WIDE)
    tl.store(ob + c, y0)
    tl.store(ob + c + 256, y1)
    tl.store(ob + c + 512, y2, c < 64)


@triton.jit
def _fused_once_frequency(X, W, F: "*fp32", P, Out, META: tl.constexpr):
    _fused_once(X, W, F, P, Out, META)


@triton.jit
def _small_parts3_frequency(X, W, F: "*fp32", P, Out, META: tl.constexpr):
    _small_parts3(X, W, F, P, Out, META)


@g.jit
def _segmented_frequency(X, W, F: "*fp32", P, Out, META: gl.constexpr):
    _segmented(X, W, F, P, Out, META)


@g.jit
def _segmented_once_frequency(X, W, F: "*fp32", P, Out, META: gl.constexpr):
    _segmented_once(X, W, F, P, Out, META)


@triton.jit
def _fused_once_inplace(X, W, F, P, META: tl.constexpr):
    _fused_once(X, W, F, P, X, META)


@triton.jit
def _fused_once_frequency_inplace(X, W, F: "*fp32", P, META: tl.constexpr):
    _fused_once_frequency(X, W, F, P, X, META)


@triton.jit
def _small_parts3_inplace(X, W, F, P, META: tl.constexpr):
    _small_parts3(X, W, F, P, X, META)


@triton.jit
def _small_parts3_frequency_inplace(X, W, F: "*fp32", P, META: tl.constexpr):
    _small_parts3_frequency(X, W, F, P, X, META)


@g.jit
def _segmented_inplace(X, W, F, P, META: gl.constexpr):
    _segmented(X, W, F, P, X, META)


@g.jit
def _segmented_frequency_inplace(X, W, F: "*fp32", P, META: gl.constexpr):
    _segmented_frequency(X, W, F, P, X, META)


@g.jit
def _segmented_once_inplace(X, W, F, P, META: gl.constexpr):
    _segmented_once(X, W, F, P, X, META)


@g.jit
def _segmented_once_frequency_inplace(X, W, F: "*fp32", P, META: gl.constexpr):
    _segmented_once_frequency(X, W, F, P, X, META)


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
    xspan = (m - 1) * sx + (d - 1)
    ospan = m * d - 1
    wspan = (d - 1) * sw
    fspan = (
        2
        * (
            (nf - 1 if nf > 0 else 0) * sf
            + (half_r - 1 if half_r > 0 else 0) * sf1
        )
        + 1
    )
    pspan = (m - 1) * sp
    WIDE = (
        xspan > 2147483647
        or ospan > 2147483647
        or wspan > 2147483647
        or (fspan > 2147483647)
        or (pspan > 2147483647)
    )
    f = torch.view_as_real(freqs_cis)
    if m < 16:
        small_kernel = (
            (
                (
                    _small_parts3_frequency_inplace
                    if kv.dtype == torch.bfloat16 and d == 576 and (r == 64)
                    else _fused_once_frequency_inplace
                )
                if freqs_cis.dtype == torch.complex64
                else (
                    _small_parts3_inplace
                    if kv.dtype == torch.bfloat16 and d == 576 and (r == 64)
                    else _fused_once_inplace
                )
            )
            if can_inplace
            else (
                (
                    _small_parts3_frequency
                    if kv.dtype == torch.bfloat16 and d == 576 and (r == 64)
                    else _fused_once_frequency
                )
                if freqs_cis.dtype == torch.complex64
                else (
                    _small_parts3
                    if kv.dtype == torch.bfloat16 and d == 576 and (r == 64)
                    else _fused_once
                )
            )
        )
        if can_inplace:
            small_kernel.run(
                kv,
                weight,
                f,
                positions,
                (
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
                    1 << (d - 1).bit_length(),
                    WIDE,
                    nf,
                ),
                num_warps=4,
                enable_fp_fusion=False,
                grid=(m if m < 32768 else 32768,),
                warmup=False,
            )
        else:
            small_kernel.run(
                kv,
                weight,
                f,
                positions,
                out,
                (
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
                    1 << (d - 1).bit_length(),
                    WIDE,
                    nf,
                ),
                num_warps=4,
                enable_fp_fusion=False,
                grid=(m if m < 32768 else 32768,),
                warmup=False,
            )
    else:
        br = 8
        blocks = (m + br - 1) // br
        kernel = (
            (
                (
                    _segmented_once_frequency_inplace
                    if blocks <= 32768
                    else _segmented_frequency_inplace
                )
                if freqs_cis.dtype == torch.complex64
                else (
                    _segmented_once_inplace
                    if blocks <= 32768
                    else _segmented_inplace
                )
            )
            if can_inplace
            else (
                (
                    _segmented_once_frequency
                    if blocks <= 32768
                    else _segmented_frequency
                )
                if freqs_cis.dtype == torch.complex64
                else _segmented_once if blocks <= 32768 else _segmented
            )
        )
        if can_inplace:
            kernel.run(
                kv,
                weight,
                f,
                positions,
                (
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
                    1 << ((d - r if d > r else 1) - 1).bit_length(),
                    1 << ((r if r > 2 else 2) - 1).bit_length(),
                    br,
                    WIDE,
                    nf,
                ),
                num_warps=4,
                enable_fp_fusion=False,
                grid=(blocks if blocks < 32768 else 32768,),
                warmup=False,
            )
        else:
            kernel.run(
                kv,
                weight,
                f,
                positions,
                out,
                (
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
                    1 << ((d - r if d > r else 1) - 1).bit_length(),
                    1 << ((r if r > 2 else 2) - 1).bit_length(),
                    br,
                    WIDE,
                    nf,
                ),
                num_warps=4,
                enable_fp_fusion=False,
                grid=(blocks if blocks < 32768 else 32768,),
                warmup=False,
            )
    return out


__all__ = ["fused_norm_rope_inplace"]
