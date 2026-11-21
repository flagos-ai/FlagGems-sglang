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
def _voxel(X, G, B, Out, P: tl.constexpr):
    START: tl.constexpr = P[0]
    N: tl.constexpr = P[1]
    C: tl.constexpr = P[2]
    GS: tl.constexpr = P[3]
    BS: tl.constexpr = P[4]
    SCALE: tl.constexpr = P[5]
    EPS: tl.constexpr = P[6]
    BIAS: tl.constexpr = P[7]
    BC: tl.constexpr = P[8]
    BR: tl.constexpr = P[9]
    row = START + tl.program_id(0) * BR + tl.arange(0, BR)
    col = tl.arange(0, BC)
    if N * C + BR * BC < 2147483648:
        offset = row[:, None] * C + col[None, :]
    else:
        offset = row[:, None].to(tl.int64) * C + col[None, :]
    if C == BC and N % BR == 0:
        x = tl.load(X + offset, cache_modifier=".cg").to(tl.float32)
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
        inv = tl.minimum(tl.rsqrt(total), 1.0 / EPS)
        y = x * inv[:, None] * SCALE * g[None, :]
    else:
        norm = tl.maximum(tl.sqrt(total), EPS)
        y = x / norm[:, None] * SCALE * g[None, :]
    if BIAS:
        bias = tl.load(B + col * BS, col < C, 0).to(tl.float32)
        y = y + bias[None, :]
    y = _silu852(y, P[10])
    if C == BC and N % BR == 0:
        tl.store(Out + offset, y, cache_modifier=".cs")
    else:
        tl.store(
            Out + offset,
            y,
            (row[:, None] < N) & (col[None, :] < C),
            cache_modifier=".cs",
        )


def wan_rmsnorm_silu(x, gamma, bias, rms_scale, eps):
    if torch.version.hip is None:
        c = x.size(1)
        n = x.numel() // c
        rows = 4 if c == 128 and n >= 16384 else 1
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
        b = gamma if bias is None else bias
        gs873 = gamma.stride(0)
        bs873 = gs873 if bias is None else b.stride(0)
        if c == 128 and n < 4096 and (bias is not None) and (eps >= 1e-18):
            p948 = (
                n,
                gs873,
                bs873,
                c**0.5 if rms_scale is None else rms_scale,
                eps,
            )
            if out is x:
                _small948_ip.run(
                    x,
                    gamma,
                    b,
                    p948,
                    grid=((n + 3) // 4,),
                    warmup=False,
                    num_warps=4,
                    enable_fp_fusion=True,
                    enable_reflect_ftz=False,
                )
            else:
                _small948.run(
                    x,
                    gamma,
                    b,
                    out,
                    p948,
                    grid=((n + 3) // 4,),
                    warmup=False,
                    num_warps=4,
                    enable_fp_fusion=True,
                    enable_reflect_ftz=False,
                )
            return out
        if (
            c in (128, 256, 512)
            and n >= 4096
            and (eps >= 1e-18)
            and (c != 512 or n < 16384)
        ):
            params = (
                n,
                c,
                gs873,
                bs873,
                c**0.5 if rms_scale is None else rms_scale,
                eps,
                bias is not None,
                8,
                32,
                4,
            )
            if out is x:
                if bias is None:
                    _row876_inplace.run(
                        x,
                        gamma,
                        (
                            n,
                            c,
                            gs873,
                            c**0.5 if rms_scale is None else rms_scale,
                            eps,
                            True,
                        ),
                        grid=((n + 8 - 1) // 8,),
                        warmup=False,
                        num_warps=4,
                        enable_fp_fusion=True,
                        enable_reflect_ftz=False,
                    )
                else:
                    _row850_inplace.run(
                        x,
                        gamma,
                        b,
                        params,
                        grid=((n + 8 - 1) // 8,),
                        warmup=False,
                        num_warps=4,
                        enable_fp_fusion=True,
                        enable_reflect_ftz=False,
                    )
            elif bias is None:
                _row876_output.run(
                    x,
                    gamma,
                    out,
                    (
                        n,
                        c,
                        gs873,
                        c**0.5 if rms_scale is None else rms_scale,
                        eps,
                        True,
                    ),
                    grid=((n + 8 - 1) // 8,),
                    warmup=False,
                    num_warps=4,
                    enable_fp_fusion=True,
                    enable_reflect_ftz=False,
                )
            else:
                _row843.run(
                    x,
                    gamma,
                    b,
                    out,
                    params,
                    grid=((n + 8 - 1) // 8,),
                    warmup=False,
                    num_warps=4,
                    enable_fp_fusion=True,
                    enable_reflect_ftz=False,
                )
            return out
        if n:
            if out is x:
                if bias is None:
                    _voxel876_inplace.run(
                        x,
                        gamma,
                        (
                            n,
                            c,
                            gs873,
                            c**0.5 if rms_scale is None else rms_scale,
                            eps,
                            True,
                        ),
                        grid=((n + rows - 1) // rows,),
                        warmup=False,
                        num_warps=2 if c >= 512 and n < 4096 else 1,
                        enable_fp_fusion=True,
                        enable_reflect_ftz=False,
                    )
                else:
                    _bias951_ip.run(
                        x,
                        gamma,
                        b,
                        (
                            n,
                            c,
                            gs873,
                            bs873,
                            c**0.5 if rms_scale is None else rms_scale,
                            eps,
                        ),
                        grid=((n + rows - 1) // rows,),
                        warmup=False,
                        num_warps=2 if c >= 512 and n < 4096 else 1,
                        enable_fp_fusion=True,
                        enable_reflect_ftz=False,
                    )
            elif bias is None:
                _voxel876_output.run(
                    x,
                    gamma,
                    out,
                    (
                        n,
                        c,
                        gs873,
                        c**0.5 if rms_scale is None else rms_scale,
                        eps,
                        True,
                    ),
                    grid=((n + rows - 1) // rows,),
                    warmup=False,
                    num_warps=2 if c >= 512 and n < 4096 else 1,
                    enable_fp_fusion=True,
                    enable_reflect_ftz=False,
                )
            else:
                _bias951_out.run(
                    x,
                    gamma,
                    b,
                    out,
                    (
                        n,
                        c,
                        gs873,
                        bs873,
                        c**0.5 if rms_scale is None else rms_scale,
                        eps,
                    ),
                    grid=((n + rows - 1) // rows,),
                    warmup=False,
                    num_warps=2 if c >= 512 and n < 4096 else 1,
                    enable_fp_fusion=True,
                    enable_reflect_ftz=False,
                )
    else:
        c = x.size(1)
        n = x.numel() // c
        rows = 4 if c == 128 and n >= 16384 else 1
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
        b = gamma if bias is None else bias
        gs873 = gamma.stride(0)
        bs873 = gs873 if bias is None else b.stride(0)
        if n:
            if out is x:
                if bias is None:
                    _voxel876_inplace.run(
                        x,
                        gamma,
                        (
                            n,
                            c,
                            gs873,
                            c**0.5 if rms_scale is None else rms_scale,
                            eps,
                            False,
                        ),
                        grid=((n + rows - 1) // rows,),
                        warmup=False,
                        num_warps=2 if c >= 512 and n < 4096 else 1,
                        enable_fp_fusion=True,
                        allow_flush_denorm=False,
                    )
                else:
                    _voxel_inplace.run(
                        x,
                        gamma,
                        b,
                        (
                            0,
                            n,
                            c,
                            gs873,
                            bs873,
                            c**0.5 if rms_scale is None else rms_scale,
                            eps,
                            bias is not None,
                            triton.next_power_of_2(c),
                            rows,
                            False,
                        ),
                        grid=((n + rows - 1) // rows,),
                        warmup=False,
                        num_warps=2 if c >= 512 and n < 4096 else 1,
                        enable_fp_fusion=True,
                        allow_flush_denorm=False,
                    )
            elif bias is None:
                _voxel876_output.run(
                    x,
                    gamma,
                    out,
                    (
                        n,
                        c,
                        gs873,
                        c**0.5 if rms_scale is None else rms_scale,
                        eps,
                        False,
                    ),
                    grid=((n + rows - 1) // rows,),
                    warmup=False,
                    num_warps=2 if c >= 512 and n < 4096 else 1,
                    enable_fp_fusion=True,
                    allow_flush_denorm=False,
                )
            else:
                _voxel.run(
                    x,
                    gamma,
                    b,
                    out,
                    (
                        0,
                        n,
                        c,
                        gs873,
                        bs873,
                        c**0.5 if rms_scale is None else rms_scale,
                        eps,
                        bias is not None,
                        triton.next_power_of_2(c),
                        rows,
                        False,
                    ),
                    grid=((n + rows - 1) // rows,),
                    warmup=False,
                    num_warps=2 if c >= 512 and n < 4096 else 1,
                    enable_fp_fusion=True,
                    allow_flush_denorm=False,
                )
    return out


@triton.jit
def _voxel_inplace(X, G, B, P: tl.constexpr):
    START: tl.constexpr = P[0]
    N: tl.constexpr = P[1]
    C: tl.constexpr = P[2]
    GS: tl.constexpr = P[3]
    BS: tl.constexpr = P[4]
    SCALE: tl.constexpr = P[5]
    EPS: tl.constexpr = P[6]
    BIAS: tl.constexpr = P[7]
    BC: tl.constexpr = P[8]
    BR: tl.constexpr = P[9]
    row = START + tl.program_id(0) * BR + tl.arange(0, BR)
    col = tl.arange(0, BC)
    if N * C + BR * BC < 2147483648:
        offset = row[:, None] * C + col[None, :]
    else:
        offset = row[:, None].to(tl.int64) * C + col[None, :]
    if C == BC and N % BR == 0:
        x = tl.load(X + offset, cache_modifier=".cg").to(tl.float32)
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
        inv = tl.minimum(tl.rsqrt(total), 1.0 / EPS)
        y = x * inv[:, None] * SCALE * g[None, :]
    else:
        norm = tl.maximum(tl.sqrt(total), EPS)
        y = x / norm[:, None] * SCALE * g[None, :]
    if BIAS:
        bias = tl.load(B + col * BS, col < C, 0).to(tl.float32)
        y = y + bias[None, :]
    y = _silu852(y, P[10])
    if C == BC and N % BR == 0:
        tl.store(X + offset, y, cache_modifier=".cs")
    else:
        tl.store(
            X + offset,
            y,
            (row[:, None] < N) & (col[None, :] < C),
            cache_modifier=".cs",
        )


@triton.jit
def _voxel_inplace_nb851(X, G, P: tl.constexpr):
    _voxel_inplace(X, G, G, P)


@triton.jit
def _voxel_nb851(X, G, Out, P: tl.constexpr):
    _voxel(X, G, G, Out, P)


@triton.jit
def _silu852(value, CUDA: tl.constexpr):
    if CUDA:
        tangent = tl.inline_asm_elementwise(
            "tanh.approx.f32 $0, $1;",
            constraints="=f,f",
            args=[value * 0.5],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )
        return value * 0.5 * (1.0 + tangent)
    else:
        return value / (1.0 + tl.exp(-value))


@gluon.jit
def _row843(X, G, B, Out, P: gl.constexpr):
    N: gl.constexpr = P[0]
    C: gl.constexpr = P[1]
    GS: gl.constexpr = P[2]
    BS: gl.constexpr = P[3]
    SCALE: gl.constexpr = P[4]
    EPS: gl.constexpr = P[5]
    BIAS: gl.constexpr = P[6]
    BR: gl.constexpr = P[7]
    WARP: gl.constexpr = P[8]
    V: gl.constexpr = P[9]
    LR: gl.constexpr = 2 if C == 128 else 1
    layout: gl.constexpr = gl.BlockedLayout(
        [1, V], [LR, WARP // LR], [4, 1], [1, 0]
    )
    row = gl.program_id(0) * BR + gl.arange(
        0, BR, layout=gl.SliceLayout(1, layout)
    )
    col = gl.arange(0, C, layout=gl.SliceLayout(0, layout))
    if N * C + BR * C < 2147483648:
        offset = row[:, None] * C + col[None, :]
    else:
        offset = row[:, None].to(gl.int64) * C + col[None, :]
    if N % BR == 0:
        value = gl.load(X + offset, cache_modifier=".cg").to(gl.float32)
    else:
        value = gl.load(X + offset, row[:, None] < N, 0).to(gl.float32)
    gamma = gl.load(G + col * GS).to(gl.float32)
    inverse = gl.minimum(gl.rsqrt(gl.sum(value * value, 1)), 1.0 / EPS)
    value = value * inverse[:, None] * SCALE * gamma[None, :]
    if BIAS:
        value = value + gl.load(B + col * BS).to(gl.float32)[None, :]
    value = _silu856(value)
    if N % BR == 0:
        gl.store(Out + offset, value, cache_modifier=".cs")
    else:
        gl.store(Out + offset, value, row[:, None] < N, cache_modifier=".cs")


@gluon.jit
def _row850_inplace(X, G, B, P: gl.constexpr):
    N: gl.constexpr = P[0]
    C: gl.constexpr = P[1]
    GS: gl.constexpr = P[2]
    BS: gl.constexpr = P[3]
    SCALE: gl.constexpr = P[4]
    EPS: gl.constexpr = P[5]
    BIAS: gl.constexpr = P[6]
    BR: gl.constexpr = P[7]
    WARP: gl.constexpr = P[8]
    V: gl.constexpr = P[9]
    LR: gl.constexpr = 2 if C == 128 else 1
    layout: gl.constexpr = gl.BlockedLayout(
        [1, V], [LR, WARP // LR], [4, 1], [1, 0]
    )
    row = gl.program_id(0) * BR + gl.arange(
        0, BR, layout=gl.SliceLayout(1, layout)
    )
    col = gl.arange(0, C, layout=gl.SliceLayout(0, layout))
    if N * C + BR * C < 2147483648:
        offset = row[:, None] * C + col[None, :]
    else:
        offset = row[:, None].to(gl.int64) * C + col[None, :]
    if N % BR == 0:
        value = gl.load(X + offset, cache_modifier=".cg").to(gl.float32)
    else:
        value = gl.load(X + offset, row[:, None] < N, 0).to(gl.float32)
    gamma = gl.load(G + col * GS).to(gl.float32)
    inverse = gl.minimum(gl.rsqrt(gl.sum(value * value, 1)), 1.0 / EPS)
    value = value * inverse[:, None] * SCALE * gamma[None, :]
    if BIAS:
        value = value + gl.load(B + col * BS).to(gl.float32)[None, :]
    value = _silu856(value)
    if N % BR == 0:
        gl.store(X + offset, value, cache_modifier=".cs")
    else:
        gl.store(X + offset, value, row[:, None] < N, cache_modifier=".cs")


@gluon.jit
def _silu856(value):
    tangent = gl.inline_asm_elementwise(
        "tanh.approx.f32 $0, $1;",
        constraints="=f,f",
        args=[value * 0.5],
        dtype=gl.float32,
        is_pure=True,
        pack=1,
    )
    return value * 0.5 * (1.0 + tangent)


@triton.jit
def _voxel876_inplace(X, G, Q: tl.constexpr):
    P: tl.constexpr = (
        0,
        Q[0],
        Q[1],
        Q[2],
        Q[2],
        Q[3],
        Q[4],
        False,
        triton.next_power_of_2(Q[1]),
        4 if Q[1] == 128 and Q[0] >= 16384 else 1,
        Q[5],
    )
    _voxel_inplace(X, G, G, P)


@triton.jit
def _voxel876_output(X, G, Out, Q: tl.constexpr):
    P: tl.constexpr = (
        0,
        Q[0],
        Q[1],
        Q[2],
        Q[2],
        Q[3],
        Q[4],
        False,
        triton.next_power_of_2(Q[1]),
        4 if Q[1] == 128 and Q[0] >= 16384 else 1,
        Q[5],
    )
    _voxel(X, G, G, Out, P)


@gluon.jit
def _row876_inplace(X, G, Q: gl.constexpr):
    P: gl.constexpr = (Q[0], Q[1], Q[2], Q[2], Q[3], Q[4], False, 8, 32, 4)
    _row850_inplace(X, G, G, P)


@gluon.jit
def _row876_output(X, G, Out, Q: gl.constexpr):
    P: gl.constexpr = (Q[0], Q[1], Q[2], Q[2], Q[3], Q[4], False, 8, 32, 4)
    _row843(X, G, G, Out, P)


@gluon.jit
def _small948(X, G, B, Out, P: gl.constexpr):
    N: gl.constexpr = P[0]
    C: gl.constexpr = 128
    GS: gl.constexpr = P[1]
    BS: gl.constexpr = P[2]
    SCALE: gl.constexpr = P[3]
    EPS: gl.constexpr = P[4]
    BIAS: gl.constexpr = True
    BR: gl.constexpr = 4
    WARP: gl.constexpr = 32
    V: gl.constexpr = 4
    LR: gl.constexpr = 1
    layout: gl.constexpr = gl.BlockedLayout(
        [1, V], [LR, WARP // LR], [4, 1], [1, 0]
    )
    row = gl.program_id(0) * BR + gl.arange(
        0, BR, layout=gl.SliceLayout(1, layout)
    )
    col = gl.arange(0, C, layout=gl.SliceLayout(0, layout))
    if N * C + BR * C < 2147483648:
        offset = row[:, None] * C + col[None, :]
    else:
        offset = row[:, None].to(gl.int64) * C + col[None, :]
    if N % BR == 0:
        value = gl.load(X + offset, cache_modifier=".cg").to(gl.float32)
    else:
        value = gl.load(X + offset, row[:, None] < N, 0).to(gl.float32)
    gamma = gl.load(G + col * GS).to(gl.float32)
    inverse = gl.minimum(gl.rsqrt(gl.sum(value * value, 1)), 1.0 / EPS)
    value = value * inverse[:, None] * SCALE * gamma[None, :]
    if BIAS:
        value = value + gl.load(B + col * BS).to(gl.float32)[None, :]
    value = _silu856(value)
    if N % BR == 0:
        gl.store(Out + offset, value, cache_modifier=".cs")
    else:
        gl.store(Out + offset, value, row[:, None] < N, cache_modifier=".cs")


@gluon.jit
def _small948_ip(X, G, B, P: gl.constexpr):
    N: gl.constexpr = P[0]
    C: gl.constexpr = 128
    GS: gl.constexpr = P[1]
    BS: gl.constexpr = P[2]
    SCALE: gl.constexpr = P[3]
    EPS: gl.constexpr = P[4]
    BIAS: gl.constexpr = True
    BR: gl.constexpr = 4
    WARP: gl.constexpr = 32
    V: gl.constexpr = 4
    LR: gl.constexpr = 1
    layout: gl.constexpr = gl.BlockedLayout(
        [1, V], [LR, WARP // LR], [4, 1], [1, 0]
    )
    row = gl.program_id(0) * BR + gl.arange(
        0, BR, layout=gl.SliceLayout(1, layout)
    )
    col = gl.arange(0, C, layout=gl.SliceLayout(0, layout))
    if N * C + BR * C < 2147483648:
        offset = row[:, None] * C + col[None, :]
    else:
        offset = row[:, None].to(gl.int64) * C + col[None, :]
    if N % BR == 0:
        value = gl.load(X + offset, cache_modifier=".cg").to(gl.float32)
    else:
        value = gl.load(X + offset, row[:, None] < N, 0).to(gl.float32)
    gamma = gl.load(G + col * GS).to(gl.float32)
    inverse = gl.minimum(gl.rsqrt(gl.sum(value * value, 1)), 1.0 / EPS)
    value = value * inverse[:, None] * SCALE * gamma[None, :]
    if BIAS:
        value = value + gl.load(B + col * BS).to(gl.float32)[None, :]
    value = _silu856(value)
    if N % BR == 0:
        gl.store(X + offset, value, cache_modifier=".cs")
    else:
        gl.store(X + offset, value, row[:, None] < N, cache_modifier=".cs")


@triton.jit
def _bias951_ip(X, G, B, Q: tl.constexpr):
    P: tl.constexpr = (
        0,
        Q[0],
        Q[1],
        Q[2],
        Q[3],
        Q[4],
        Q[5],
        True,
        triton.next_power_of_2(Q[1]),
        4 if Q[1] == 128 and Q[0] >= 16384 else 1,
        True,
    )
    _voxel_inplace(X, G, B, P)


@triton.jit
def _bias951_out(X, G, B, Out, Q: tl.constexpr):
    P: tl.constexpr = (
        0,
        Q[0],
        Q[1],
        Q[2],
        Q[3],
        Q[4],
        Q[5],
        True,
        triton.next_power_of_2(Q[1]),
        4 if Q[1] == 128 and Q[0] >= 16384 else 1,
        True,
    )
    _voxel(X, G, B, Out, P)


__all__ = ["wan_rmsnorm_silu"]
