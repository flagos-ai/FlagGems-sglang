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
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.language.extra import libdevice


@triton.jit
def _tile0010(X, G, B, Out, P: tl.constexpr):
    N: tl.constexpr = P[0]
    C: tl.constexpr = P[1]
    GS: tl.constexpr = P[2]
    BS: tl.constexpr = P[3]
    SCALE: tl.constexpr = P[4]
    EPS: tl.constexpr = P[5]
    BIAS: tl.constexpr = P[6]
    BC: tl.constexpr = P[7]
    BR: tl.constexpr = P[8]
    row = tl.program_id(0) * BR + tl.arange(0, BR)
    col = tl.arange(0, BC)
    if N * C + BR * BC < 2147483648:
        offset = row[:, None] * C + col[None, :]
    else:
        offset = row[:, None].to(tl.int64) * C + col[None, :]
    if C == BC and N % BR == 0:
        x = tl.load(X + offset).to(tl.float32)
    else:
        x = tl.load(X + offset, (row[:, None] < N) & (col[None, :] < C), 0).to(
            tl.float32
        )
    g = tl.load(G + col * GS, col < C, 0).to(tl.float32)
    total = tl.sum(x * x, 1)
    if EPS < 1e-18:
        norm = tl.maximum(libdevice.sqrt(total), EPS)
        y = tl.div_rn(x, norm[:, None]) * SCALE * g[None, :]
    elif EPS >= 1e-30:
        inv = tl.minimum(_hardware_rsq41910(total), 1.0 / EPS)
        y = x * inv[:, None] * SCALE * g[None, :]
    else:
        norm = tl.maximum(tl.sqrt(total), EPS)
        y = x / norm[:, None] * SCALE * g[None, :]
    if BIAS:
        bias = tl.load(B + col * BS, col < C, 0).to(tl.float32)
        y = y + bias[None, :]
    if C >= 512 and N >= 4096:
        y = y * tl.exp2(-tl.log2(1.0 + tl.exp2(-y * 1.4426950408889634)))
    else:
        y = y / (1.0 + tl.exp(-y))
    if C == BC and N % BR == 0:
        tl.store(Out + offset, y)
    else:
        tl.store(Out + offset, y, (row[:, None] < N) & (col[None, :] < C))


@triton.jit
def _tile_inplace0010(X, G, B, P: tl.constexpr):
    N: tl.constexpr = P[0]
    C: tl.constexpr = P[1]
    GS: tl.constexpr = P[2]
    BS: tl.constexpr = P[3]
    SCALE: tl.constexpr = P[4]
    EPS: tl.constexpr = P[5]
    BIAS: tl.constexpr = P[6]
    BC: tl.constexpr = P[7]
    BR: tl.constexpr = P[8]
    row = tl.program_id(0) * BR + tl.arange(0, BR)
    col = tl.arange(0, BC)
    if N * C + BR * BC < 2147483648:
        offset = row[:, None] * C + col[None, :]
    else:
        offset = row[:, None].to(tl.int64) * C + col[None, :]
    if C == BC and N % BR == 0:
        x = tl.load(X + offset).to(tl.float32)
    else:
        x = tl.load(X + offset, (row[:, None] < N) & (col[None, :] < C), 0).to(
            tl.float32
        )
    g = tl.load(G + col * GS, col < C, 0).to(tl.float32)
    total = tl.sum(x * x, 1)
    if EPS < 1e-18:
        norm = tl.maximum(libdevice.sqrt(total), EPS)
        y = tl.div_rn(x, norm[:, None]) * SCALE * g[None, :]
    elif EPS >= 1e-30:
        inv = tl.minimum(_hardware_rsq41910(total), 1.0 / EPS)
        y = x * inv[:, None] * SCALE * g[None, :]
    else:
        norm = tl.maximum(tl.sqrt(total), EPS)
        y = x / norm[:, None] * SCALE * g[None, :]
    if BIAS:
        bias = tl.load(B + col * BS, col < C, 0).to(tl.float32)
        y = y + bias[None, :]
    if C >= 512 and N >= 4096:
        y = y * tl.exp2(-tl.log2(1.0 + tl.exp2(-y * 1.4426950408889634)))
    else:
        y = y / (1.0 + tl.exp(-y))
    if C == BC and N % BR == 0:
        tl.store(X + offset, y)
    else:
        tl.store(X + offset, y, (row[:, None] < N) & (col[None, :] < C))


@triton.jit
def _halves010(X, G, B, Out, P: tl.constexpr):
    N: tl.constexpr = P[0]
    C: tl.constexpr = P[1]
    GS: tl.constexpr = P[2]
    BS: tl.constexpr = P[3]
    SCALE: tl.constexpr = P[4]
    EPS: tl.constexpr = P[5]
    BIAS: tl.constexpr = P[6]
    BC: tl.constexpr = P[7]
    BR: tl.constexpr = P[8]
    INPLACE: tl.constexpr = P[9]
    row = tl.program_id(0) * BR + tl.arange(0, BR)
    col = tl.arange(0, BC // 2)
    if N * C + BR * BC < 2147483648:
        offset = row[:, None] * C + col[None, :]
    else:
        offset = row[:, None].to(tl.int64) * C + col[None, :]
    if N % BR == 0:
        x0 = tl.load(X + offset).to(tl.float32)
        x1 = tl.load(X + offset + BC // 2).to(tl.float32)
    else:
        x0 = tl.load(X + offset, row[:, None] < N, 0).to(tl.float32)
        x1 = tl.load(X + offset + BC // 2, row[:, None] < N, 0).to(tl.float32)
    total = tl.sum(x0 * x0 + x1 * x1, 1)
    if EPS < 1e-18:
        norm = tl.maximum(libdevice.sqrt(total), EPS)
        v0 = tl.div_rn(x0, norm[:, None]) * SCALE
        v1 = tl.div_rn(x1, norm[:, None]) * SCALE
    else:
        inv = tl.minimum(_hardware_rsq41910(total), 1.0 / EPS)
        v0 = x0 * inv[:, None] * SCALE
        v1 = x1 * inv[:, None] * SCALE
    g0 = tl.load(G + col * GS).to(tl.float32)
    g1 = tl.load(G + (col + BC // 2) * GS).to(tl.float32)
    y0 = v0 * g0[None, :]
    y1 = v1 * g1[None, :]
    if BIAS:
        b0 = tl.load(B + col * BS).to(tl.float32)
        b1 = tl.load(B + (col + BC // 2) * BS).to(tl.float32)
        y0 = y0 + b0[None, :]
        y1 = y1 + b1[None, :]
    y0 = y0 / (1.0 + tl.exp(-y0))
    y1 = y1 / (1.0 + tl.exp(-y1))
    if INPLACE:
        destination = X
    else:
        destination = Out
    if N % BR == 0:
        tl.store(destination + offset, y0)
        tl.store(destination + offset + BC // 2, y1)
    else:
        tl.store(destination + offset, y0, row[:, None] < N)
        tl.store(destination + offset + BC // 2, y1, row[:, None] < N)


@gluon.jit
def _gl35810(X, G, B, Out, P: gl.constexpr):
    N: gl.constexpr = P[0]
    C: gl.constexpr = P[1]
    GS: gl.constexpr = P[2]
    BS: gl.constexpr = P[3]
    SCALE: gl.constexpr = P[4]
    EPS: gl.constexpr = P[5]
    BIAS: gl.constexpr = P[6]
    BR: gl.constexpr = P[7]
    NW: gl.constexpr = 1
    TR: gl.constexpr = BR
    V: gl.constexpr = 4 if C == 128 else 8
    layout: gl.constexpr = gl.BlockedLayout(
        [1, V], [TR, 64 // TR], [NW, 1], [1, 0]
    )
    col = gl.arange(0, C, layout=gl.SliceLayout(0, layout))
    TILES: gl.constexpr = 2 if C >= 512 and N >= 16384 else 1
    pref_row0 = (gl.program_id(0) * TILES + 0) * BR + gl.arange(
        0, BR, layout=gl.SliceLayout(1, layout)
    )
    if N * C + TILES * BR * C < 2147483648:
        pref_offset0 = pref_row0[:, None] * C + col[None, :]
    else:
        pref_offset0 = pref_row0[:, None].to(gl.int64) * C + col[None, :]
    if N % (BR * TILES) == 0:
        pref_value0 = gl.load(X + pref_offset0)
    else:
        pref_value0 = gl.load(X + pref_offset0, pref_row0[:, None] < N, 0)
    if TILES == 2:
        pref_row1 = (gl.program_id(0) * TILES + 1) * BR + gl.arange(
            0, BR, layout=gl.SliceLayout(1, layout)
        )
        if N * C + TILES * BR * C < 2147483648:
            pref_offset1 = pref_row1[:, None] * C + col[None, :]
        else:
            pref_offset1 = pref_row1[:, None].to(gl.int64) * C + col[None, :]
        if N % (BR * TILES) == 0:
            pref_value1 = gl.load(X + pref_offset1)
        else:
            pref_value1 = gl.load(X + pref_offset1, pref_row1[:, None] < N, 0)
    gamma = gl.load(G + col * GS).to(gl.float32)
    if TILES == 2:
        row0 = (gl.program_id(0) * TILES + 0) * BR + gl.arange(
            0, BR, layout=gl.SliceLayout(1, layout)
        )
        if N * C + TILES * BR * C < 2147483648:
            offset0 = row0[:, None] * C + col[None, :]
        else:
            offset0 = row0[:, None].to(gl.int64) * C + col[None, :]
        values0 = pref_value0.to(gl.float32)
        inv0 = gl.minimum(
            _hardware_rsq41910(gl.sum(values0 * values0, 1)), 1.0 / EPS
        )
        if TILES == 2 and (
            SCALE >= 0.5
            and SCALE <= 32.0
            or (SCALE <= -0.5 and SCALE >= -32.0)
        ):
            gain = inv0 * SCALE
            values0 = values0 * gain[:, None]
        else:
            values0 = values0 * inv0[:, None] * SCALE
        if BIAS:
            values0 = gl.fma(
                values0,
                gamma[None, :],
                gl.load(B + col * BS).to(gl.float32)[None, :],
            )
        else:
            values0 = values0 * gamma[None, :]
        denominator_values0 = 1.0 + gl.exp(-values0)
        row1 = (gl.program_id(0) * TILES + 1) * BR + gl.arange(
            0, BR, layout=gl.SliceLayout(1, layout)
        )
        if N * C + TILES * BR * C < 2147483648:
            offset1 = row1[:, None] * C + col[None, :]
        else:
            offset1 = row1[:, None].to(gl.int64) * C + col[None, :]
        values1 = pref_value1.to(gl.float32)
        inv1 = gl.minimum(
            _hardware_rsq41910(gl.sum(values1 * values1, 1)), 1.0 / EPS
        )
        if TILES == 2 and (
            SCALE >= 0.5
            and SCALE <= 32.0
            or (SCALE <= -0.5 and SCALE >= -32.0)
        ):
            gain = inv1 * SCALE
            values1 = values1 * gain[:, None]
        else:
            values1 = values1 * inv1[:, None] * SCALE
        if BIAS:
            values1 = gl.fma(
                values1,
                gamma[None, :],
                gl.load(B + col * BS).to(gl.float32)[None, :],
            )
        else:
            values1 = values1 * gamma[None, :]
        denominator_values1 = 1.0 + gl.exp(-values1)
        reciprocal_values0 = _native_rcp40010(denominator_values0)
        values0 = values0 * reciprocal_values0
        if N % (BR * TILES) == 0:
            gl.store(Out + offset0, _output551(values0, Out))
        else:
            gl.store(
                Out + offset0, _output551(values0, Out), row0[:, None] < N
            )
        reciprocal_values1 = _native_rcp40010(denominator_values1)
        values1 = values1 * reciprocal_values1
        if N % (BR * TILES) == 0:
            gl.store(Out + offset1, _output551(values1, Out))
        else:
            gl.store(
                Out + offset1, _output551(values1, Out), row1[:, None] < N
            )
    else:
        for tile in gl.static_range(0, TILES):
            row = (gl.program_id(0) * TILES + tile) * BR + gl.arange(
                0, BR, layout=gl.SliceLayout(1, layout)
            )
            if N * C + TILES * BR * C < 2147483648:
                offset = row[:, None] * C + col[None, :]
            else:
                offset = row[:, None].to(gl.int64) * C + col[None, :]
            if tile == 0:
                values = pref_value0.to(gl.float32)
            else:
                values = pref_value1.to(gl.float32)
            inv = gl.minimum(
                _hardware_rsq41910(gl.sum(values * values, 1)), 1.0 / EPS
            )
            if TILES == 2 and (
                SCALE >= 0.5
                and SCALE <= 32.0
                or (SCALE <= -0.5 and SCALE >= -32.0)
            ):
                gain = inv * SCALE
                values = values * gain[:, None]
            else:
                values = values * inv[:, None] * SCALE
            if BIAS:
                values = gl.fma(
                    values,
                    gamma[None, :],
                    gl.load(B + col * BS).to(gl.float32)[None, :],
                )
            else:
                values = values * gamma[None, :]
            denominator_values = gl.fma(
                gl.exp(-values), 5.960464477539063e-08, 5.960464477539063e-08
            )
            reciprocal_values = _native_rcp40010(denominator_values)
            values = values * (reciprocal_values * 5.960464477539063e-08)
            if N % (BR * TILES) == 0:
                gl.store(Out + offset, _output551(values, Out))
            else:
                gl.store(
                    Out + offset, _output551(values, Out), row[:, None] < N
                )


@tl.core.extern
def _hardware_rsq41910(x, _semantic=None):
    return tl.core.extern_elementwise(
        "",
        "",
        [x],
        {(tl.float32,): ("llvm.mxc.rsq.f32", tl.float32)},
        is_pure=True,
        _semantic=_semantic,
    )


@tl.core.extern
def _native_rcp40010(arg0, _semantic=None):
    return tl.core.extern_elementwise(
        "",
        "",
        [arg0],
        {(tl.float32,): ("llvm.mxc.rcp.f32", tl.float32)},
        is_pure=True,
        _semantic=_semantic,
    )


@gluon.jit
def _gl35811(X, G, B, Out, P: gl.constexpr):
    N: gl.constexpr = P[0]
    C: gl.constexpr = P[1]
    GS: gl.constexpr = P[2]
    BS: gl.constexpr = P[3]
    SCALE: gl.constexpr = P[4]
    EPS: gl.constexpr = P[5]
    BIAS: gl.constexpr = P[6]
    BR: gl.constexpr = P[7]
    NW: gl.constexpr = 1
    TR: gl.constexpr = BR
    V: gl.constexpr = 8
    layout: gl.constexpr = gl.BlockedLayout(
        [1, V], [TR, 64 // TR], [NW, 1], [1, 0]
    )
    col = gl.arange(0, C, layout=gl.SliceLayout(0, layout))
    gamma = gl.load(G + col * GS).to(gl.float32)
    if BIAS:
        bias = gl.load(B + col * BS).to(gl.float32)
    row0 = (gl.program_id(0) * 2 + 0) * BR + gl.arange(
        0, BR, layout=gl.SliceLayout(1, layout)
    )
    if N * C + BR * C < 2147483648:
        offset0 = row0[:, None] * C + col[None, :]
    else:
        offset0 = row0[:, None].to(gl.int64) * C + col[None, :]
    if N % (BR * 2) == 0:
        values0 = gl.load(X + offset0, cache_modifier=".ca")
    else:
        values0 = gl.load(
            X + offset0, row0[:, None] < N, 0, cache_modifier=".ca"
        )
    row1 = (gl.program_id(0) * 2 + 1) * BR + gl.arange(
        0, BR, layout=gl.SliceLayout(1, layout)
    )
    if N * C + BR * C < 2147483648:
        offset1 = row1[:, None] * C + col[None, :]
    else:
        offset1 = row1[:, None].to(gl.int64) * C + col[None, :]
    if N % (BR * 2) == 0:
        values1 = gl.load(X + offset1, cache_modifier=".ca")
    else:
        values1 = gl.load(
            X + offset1, row1[:, None] < N, 0, cache_modifier=".ca"
        )
    values0 = values0.to(gl.float32)
    inv0 = gl.minimum(
        _hardware_rsq41910(gl.sum(values0 * values0, 1)), 1.0 / EPS
    )
    if SCALE >= 0.5 and SCALE <= 32.0 or (SCALE <= -0.5 and SCALE >= -32.0):
        gain0 = inv0 * SCALE
        values0 = values0 * gain0[:, None]
    else:
        values0 = values0 * inv0[:, None] * SCALE
    if BIAS:
        values0 = gl.fma(values0, gamma[None, :], bias[None, :])
    else:
        values0 = values0 * gamma[None, :]
    denominator_values0 = gl.fma(
        gl.exp(-values0), 5.960464477539063e-08, 5.960464477539063e-08
    )
    reciprocal_values0 = _native_rcp40010(denominator_values0)
    values0 = values0 * (reciprocal_values0 * 5.960464477539063e-08)
    if N % (BR * 2) == 0:
        gl.store(Out + offset0, _output551(values0, Out), cache_modifier=".cs")
    else:
        gl.store(
            Out + offset0,
            _output551(values0, Out),
            row0[:, None] < N,
            cache_modifier=".cs",
        )
    values1 = values1.to(gl.float32)
    inv1 = gl.minimum(
        _hardware_rsq41910(gl.sum(values1 * values1, 1)), 1.0 / EPS
    )
    if SCALE >= 0.5 and SCALE <= 32.0 or (SCALE <= -0.5 and SCALE >= -32.0):
        gain1 = inv1 * SCALE
        values1 = values1 * gain1[:, None]
    else:
        values1 = values1 * inv1[:, None] * SCALE
    if BIAS:
        values1 = gl.fma(values1, gamma[None, :], bias[None, :])
    else:
        values1 = values1 * gamma[None, :]
    denominator_values1 = gl.fma(
        gl.exp(-values1), 5.960464477539063e-08, 5.960464477539063e-08
    )
    reciprocal_values1 = _native_rcp40010(denominator_values1)
    values1 = values1 * (reciprocal_values1 * 5.960464477539063e-08)
    if N % (BR * 2) == 0:
        gl.store(Out + offset1, _output551(values1, Out), cache_modifier=".cs")
    else:
        gl.store(
            Out + offset1,
            _output551(values1, Out),
            row1[:, None] < N,
            cache_modifier=".cs",
        )


def wan_rmsnorm_silu(x, gamma, bias, rms_scale, eps):
    c = x.shape[1]
    n = x.numel() // c
    dtype = (
        x.dtype
        if x.dtype == gamma.dtype
        else torch.promote_types(x.dtype, gamma.dtype)
    )
    if (
        dtype == x.dtype
        and (not torch._C._overlaps(x, gamma))
        and (bias is None or not torch._C._overlaps(x, bias))
    ):
        out = x
    else:
        out = torch.empty_strided(
            x.shape, x.stride(), dtype=dtype, device=x.device
        )
    if not n:
        return out
    b = gamma if bias is None else bias
    scale = c**0.5 if rms_scale is None else rms_scale
    gs = gamma.stride(0)
    bs = gs if bias is None else b.stride(0)
    if eps >= 1e-18 and n >= 256 and (c >= 128) and (c & c - 1 == 0):
        if c == 128 and n >= 16384:
            br = 4
            params = (n, c, gs, bs, scale, eps, bias is not None, br)
            if bias is None:
                if out is x:
                    _gl35811_nbip602.run(
                        x,
                        gamma,
                        params,
                        grid=(triton.cdiv(n, br * 2),),
                        warmup=False,
                        num_warps=1,
                        enable_fp_fusion=False,
                    )
                else:
                    _gl35811_nb602.run(
                        x,
                        gamma,
                        out,
                        params,
                        grid=(triton.cdiv(n, br * 2),),
                        warmup=False,
                        num_warps=1,
                        enable_fp_fusion=False,
                    )
            elif out is x:
                _gl35811_ip602.run(
                    x,
                    gamma,
                    b,
                    params,
                    grid=(triton.cdiv(n, br * 2),),
                    warmup=False,
                    num_warps=1,
                    enable_fp_fusion=False,
                )
            else:
                _gl35811.run(
                    x,
                    gamma,
                    b,
                    out,
                    params,
                    grid=(triton.cdiv(n, br * 2),),
                    warmup=False,
                    num_warps=1,
                    enable_fp_fusion=False,
                )
        else:
            br = (
                4
                if c == 128 or (c == 256 and n >= 4096)
                else 1 if n < 4096 else 2
            )
            params = (n, c, gs, bs, scale, eps, bias is not None, br)
            if bias is None:
                if out is x:
                    _gl35810_nbip602.run(
                        x,
                        gamma,
                        params,
                        grid=(
                            triton.cdiv(
                                n, br * (2 if c >= 512 and n >= 16384 else 1)
                            ),
                        ),
                        warmup=False,
                        num_warps=1,
                        enable_fp_fusion=False,
                    )
                else:
                    _gl35810_nb602.run(
                        x,
                        gamma,
                        out,
                        params,
                        grid=(
                            triton.cdiv(
                                n, br * (2 if c >= 512 and n >= 16384 else 1)
                            ),
                        ),
                        warmup=False,
                        num_warps=1,
                        enable_fp_fusion=False,
                    )
            elif out is x:
                _gl35810_ip602.run(
                    x,
                    gamma,
                    b,
                    params,
                    grid=(
                        triton.cdiv(
                            n, br * (2 if c >= 512 and n >= 16384 else 1)
                        ),
                    ),
                    warmup=False,
                    num_warps=1,
                    enable_fp_fusion=False,
                )
            else:
                _gl35810.run(
                    x,
                    gamma,
                    b,
                    out,
                    params,
                    grid=(
                        triton.cdiv(
                            n, br * (2 if c >= 512 and n >= 16384 else 1)
                        ),
                    ),
                    warmup=False,
                    num_warps=1,
                    enable_fp_fusion=False,
                )
    elif c >= 64 and c <= 128 and (c & c - 1 == 0) and (n >= 4096):
        br = 4
        params = (n, c, gs, bs, scale, eps, bias is not None, c, br, out is x)
        _halves010.run(
            x,
            gamma,
            b,
            out,
            params,
            grid=(triton.cdiv(n, br),),
            warmup=False,
            num_warps=1,
            enable_fp_fusion=False,
        )
    else:
        bc = triton.next_power_of_2(c)
        br = max(1, min(2, 4096 // bc))
        params = (n, c, gs, bs, scale, eps, bias is not None, bc, br)
        if out is x:
            _tile_inplace0010.run(
                x,
                gamma,
                b,
                params,
                grid=(triton.cdiv(n, br),),
                warmup=False,
                num_warps=1,
                enable_fp_fusion=False,
            )
        else:
            _tile0010.run(
                x,
                gamma,
                b,
                out,
                params,
                grid=(triton.cdiv(n, br),),
                warmup=False,
                num_warps=1,
                enable_fp_fusion=False,
            )
    return out


@gluon.jit
def _output551(value, pointer):
    if pointer.dtype.element_ty == gl.bfloat16:
        value = gl.where(value >= 3.39617752923046e38, float("inf"), value)
        return value.to(gl.bfloat16, fp_downcast_rounding="rtz")
    else:
        return value


@gluon.jit
def _gl35810_ip602(X, G, B, P: gl.constexpr):
    _gl35810(X, G, B, X, P)


@gluon.jit
def _gl35810_nb602(X, G, Out, P: gl.constexpr):
    _gl35810(X, G, G, Out, P)


@gluon.jit
def _gl35810_nbip602(X, G, P: gl.constexpr):
    _gl35810(X, G, G, X, P)


@gluon.jit
def _gl35811_ip602(X, G, B, P: gl.constexpr):
    _gl35811(X, G, B, X, P)


@gluon.jit
def _gl35811_nb602(X, G, Out, P: gl.constexpr):
    _gl35811(X, G, G, Out, P)


@gluon.jit
def _gl35811_nbip602(X, G, P: gl.constexpr):
    _gl35811(X, G, G, X, P)


__all__ = ["wan_rmsnorm_silu"]
