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
from triton.language.extra import libdevice


@triton.jit
def _tile10(X, G, B, Out, P: tl.constexpr):
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
        inv = tl.minimum(tl.rsqrt(total), 1.0 / EPS)
        y = x * inv[:, None] * SCALE * g[None, :]
    else:
        norm = tl.maximum(tl.sqrt(total), EPS)
        y = x / norm[:, None] * SCALE * g[None, :]
    if BIAS:
        bias = tl.load(B + col * BS, col < C, 0).to(tl.float32)
        y = y + bias[None, :]
    denominator = 1.0 + tl.exp(-y)
    reciprocal_root = tl.rsqrt(denominator)
    y = y * reciprocal_root * reciprocal_root
    if C == BC and N % BR == 0:
        tl.store(Out + offset, y)
    else:
        tl.store(Out + offset, y, (row[:, None] < N) & (col[None, :] < C))


@triton.jit
def _tile_inplace10(X, G, B, P: tl.constexpr):
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
        inv = tl.minimum(tl.rsqrt(total), 1.0 / EPS)
        y = x * inv[:, None] * SCALE * g[None, :]
    else:
        norm = tl.maximum(tl.sqrt(total), EPS)
        y = x / norm[:, None] * SCALE * g[None, :]
    if BIAS:
        bias = tl.load(B + col * BS, col < C, 0).to(tl.float32)
        y = y + bias[None, :]
    denominator = 1.0 + tl.exp(-y)
    reciprocal_root = tl.rsqrt(denominator)
    y = y * reciprocal_root * reciprocal_root
    if C == BC and N % BR == 0:
        tl.store(X + offset, y)
    else:
        tl.store(X + offset, y, (row[:, None] < N) & (col[None, :] < C))


def _output10(x, gamma, bias):
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
        return x
    return torch.empty_strided(
        x.shape, x.stride(), dtype=dtype, device=x.device
    )


def _entry10(x, gamma, bias, rms_scale, eps):
    c = x.shape[1]
    n = x.numel() // c
    out = _output10(x, gamma, bias)
    b = gamma if bias is None else bias
    bc = triton.next_power_of_2(c)
    br = 1 if bc >= 512 else 4 if n >= 4096 else 2
    if n:
        warps = 2 if bc >= 512 or n >= 4096 else 1
        if out is x:
            if bias is None:
                _tile_inplace10_short953_0.run(
                    x,
                    gamma,
                    (
                        n,
                        c,
                        gamma.stride(0),
                        c**0.5 if rms_scale is None else rms_scale,
                        eps,
                    ),
                    grid=(triton.cdiv(n, br),),
                    warmup=False,
                    num_warps=warps,
                    enable_fp_fusion=False,
                )
            else:
                _tile_inplace10_short953_1.run(
                    x,
                    gamma,
                    b,
                    (
                        n,
                        c,
                        gamma.stride(0),
                        b.stride(0),
                        c**0.5 if rms_scale is None else rms_scale,
                        eps,
                    ),
                    grid=(triton.cdiv(n, br),),
                    warmup=False,
                    num_warps=warps,
                    enable_fp_fusion=False,
                )
        elif bias is None:
            _tile10_short953_0.run(
                x,
                gamma,
                out,
                (
                    n,
                    c,
                    gamma.stride(0),
                    c**0.5 if rms_scale is None else rms_scale,
                    eps,
                ),
                grid=(triton.cdiv(n, br),),
                warmup=False,
                num_warps=warps,
                enable_fp_fusion=False,
            )
        else:
            _tile10_short953_1.run(
                x,
                gamma,
                b,
                out,
                (
                    n,
                    c,
                    gamma.stride(0),
                    b.stride(0),
                    c**0.5 if rms_scale is None else rms_scale,
                    eps,
                ),
                grid=(triton.cdiv(n, br),),
                warmup=False,
                num_warps=warps,
                enable_fp_fusion=False,
            )
    return out


@triton.jit
def _tile11(X, G, B, Out, P: tl.constexpr):
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
    g = tl.load(
        G + tl.broadcast_to(col[None, :] * GS, (BR, BC)), col[None, :] < C, 0
    ).to(tl.float32)
    total = tl.sum(x * x, 1)
    if EPS < 1e-18:
        norm = tl.maximum(libdevice.sqrt(total), EPS)
        y = tl.div_rn(x, norm[:, None]) * SCALE * g
    elif EPS >= 1e-30:
        inv = tl.minimum(tl.rsqrt(total), 1.0 / EPS)
        y = x * inv[:, None] * SCALE * g
    else:
        norm = tl.maximum(tl.sqrt(total), EPS)
        y = x / norm[:, None] * SCALE * g
    if BIAS:
        bias = tl.load(
            B + tl.broadcast_to(col[None, :] * BS, (BR, BC)),
            col[None, :] < C,
            0,
        ).to(tl.float32)
        y = y + bias
    denominator = 1.0 + tl.exp(-y)
    reciprocal_root = tl.rsqrt(denominator)
    y = y * reciprocal_root * reciprocal_root
    if C == BC and N % BR == 0:
        tl.store(Out + offset, y)
    else:
        tl.store(Out + offset, y, (row[:, None] < N) & (col[None, :] < C))


@triton.jit
def _tile_inplace11(X, G, B, P: tl.constexpr):
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
    g = tl.load(
        G + tl.broadcast_to(col[None, :] * GS, (BR, BC)), col[None, :] < C, 0
    ).to(tl.float32)
    total = tl.sum(x * x, 1)
    if EPS < 1e-18:
        norm = tl.maximum(libdevice.sqrt(total), EPS)
        y = tl.div_rn(x, norm[:, None]) * SCALE * g
    elif EPS >= 1e-30:
        inv = tl.minimum(tl.rsqrt(total), 1.0 / EPS)
        y = x * inv[:, None] * SCALE * g
    else:
        norm = tl.maximum(tl.sqrt(total), EPS)
        y = x / norm[:, None] * SCALE * g
    if BIAS:
        bias = tl.load(
            B + tl.broadcast_to(col[None, :] * BS, (BR, BC)),
            col[None, :] < C,
            0,
        ).to(tl.float32)
        y = y + bias
    denominator = 1.0 + tl.exp(-y)
    reciprocal_root = tl.rsqrt(denominator)
    y = y * reciprocal_root * reciprocal_root
    if C == BC and N % BR == 0:
        tl.store(X + offset, y)
    else:
        tl.store(X + offset, y, (row[:, None] < N) & (col[None, :] < C))


def _entry11(x, gamma, bias, rms_scale, eps):
    c = x.shape[1]
    n = x.numel() // c
    out = _output10(x, gamma, bias)
    b = gamma if bias is None else bias
    bc = triton.next_power_of_2(c)
    br = 1 if bc >= 512 else 4 if n >= 4096 else 2
    if n:
        warps = 2 if bc >= 512 or n >= 4096 else 1
        if out is x:
            if bias is None:
                _tile_inplace11_short953_0.run(
                    x,
                    gamma,
                    (
                        n,
                        c,
                        gamma.stride(0),
                        c**0.5 if rms_scale is None else rms_scale,
                        eps,
                    ),
                    grid=(triton.cdiv(n, br),),
                    warmup=False,
                    num_warps=warps,
                    enable_fp_fusion=False,
                )
            else:
                _tile_inplace11_short953_1.run(
                    x,
                    gamma,
                    b,
                    (
                        n,
                        c,
                        gamma.stride(0),
                        b.stride(0),
                        c**0.5 if rms_scale is None else rms_scale,
                        eps,
                    ),
                    grid=(triton.cdiv(n, br),),
                    warmup=False,
                    num_warps=warps,
                    enable_fp_fusion=False,
                )
        elif bias is None:
            _tile11_short953_0.run(
                x,
                gamma,
                out,
                (
                    n,
                    c,
                    gamma.stride(0),
                    c**0.5 if rms_scale is None else rms_scale,
                    eps,
                ),
                grid=(triton.cdiv(n, br),),
                warmup=False,
                num_warps=warps,
                enable_fp_fusion=False,
            )
        else:
            _tile11_short953_1.run(
                x,
                gamma,
                b,
                out,
                (
                    n,
                    c,
                    gamma.stride(0),
                    b.stride(0),
                    c**0.5 if rms_scale is None else rms_scale,
                    eps,
                ),
                grid=(triton.cdiv(n, br),),
                warmup=False,
                num_warps=warps,
                enable_fp_fusion=False,
            )
    return out


def wan_rmsnorm_silu(x, gamma, bias, rms_scale, eps):
    if x.shape[1] == 128:
        return _entry11(x, gamma, bias, rms_scale, eps)
    return _entry10(x, gamma, bias, rms_scale, eps)


@triton.jit
def _tile10_short953_0(X, G, Out, Q: tl.constexpr):
    P: tl.constexpr = (
        Q[0],
        Q[1],
        Q[2],
        Q[3] if False else Q[2],
        Q[3],
        Q[4],
        False,
        triton.next_power_of_2(Q[1]),
        1 if triton.next_power_of_2(Q[1]) >= 512 else 4 if Q[0] >= 4096 else 2,
    )
    _tile10(X, G, G, Out, P)


@triton.jit
def _tile10_short953_1(X, G, B, Out, Q: tl.constexpr):
    P: tl.constexpr = (
        Q[0],
        Q[1],
        Q[2],
        Q[3] if True else Q[2],
        Q[4],
        Q[5],
        True,
        triton.next_power_of_2(Q[1]),
        1 if triton.next_power_of_2(Q[1]) >= 512 else 4 if Q[0] >= 4096 else 2,
    )
    _tile10(X, G, B, Out, P)


@triton.jit
def _tile_inplace10_short953_0(X, G, Q: tl.constexpr):
    P: tl.constexpr = (
        Q[0],
        Q[1],
        Q[2],
        Q[3] if False else Q[2],
        Q[3],
        Q[4],
        False,
        triton.next_power_of_2(Q[1]),
        1 if triton.next_power_of_2(Q[1]) >= 512 else 4 if Q[0] >= 4096 else 2,
    )
    _tile_inplace10(X, G, G, P)


@triton.jit
def _tile_inplace10_short953_1(X, G, B, Q: tl.constexpr):
    P: tl.constexpr = (
        Q[0],
        Q[1],
        Q[2],
        Q[3] if True else Q[2],
        Q[4],
        Q[5],
        True,
        triton.next_power_of_2(Q[1]),
        1 if triton.next_power_of_2(Q[1]) >= 512 else 4 if Q[0] >= 4096 else 2,
    )
    _tile_inplace10(X, G, B, P)


@triton.jit
def _tile11_short953_0(X, G, Out, Q: tl.constexpr):
    P: tl.constexpr = (
        Q[0],
        Q[1],
        Q[2],
        Q[3] if False else Q[2],
        Q[3],
        Q[4],
        False,
        triton.next_power_of_2(Q[1]),
        1 if triton.next_power_of_2(Q[1]) >= 512 else 4 if Q[0] >= 4096 else 2,
    )
    _tile11(X, G, G, Out, P)


@triton.jit
def _tile11_short953_1(X, G, B, Out, Q: tl.constexpr):
    P: tl.constexpr = (
        Q[0],
        Q[1],
        Q[2],
        Q[3] if True else Q[2],
        Q[4],
        Q[5],
        True,
        triton.next_power_of_2(Q[1]),
        1 if triton.next_power_of_2(Q[1]) >= 512 else 4 if Q[0] >= 4096 else 2,
    )
    _tile11(X, G, B, Out, P)


@triton.jit
def _tile_inplace11_short953_0(X, G, Q: tl.constexpr):
    P: tl.constexpr = (
        Q[0],
        Q[1],
        Q[2],
        Q[3] if False else Q[2],
        Q[3],
        Q[4],
        False,
        triton.next_power_of_2(Q[1]),
        1 if triton.next_power_of_2(Q[1]) >= 512 else 4 if Q[0] >= 4096 else 2,
    )
    _tile_inplace11(X, G, G, P)


@triton.jit
def _tile_inplace11_short953_1(X, G, B, Q: tl.constexpr):
    P: tl.constexpr = (
        Q[0],
        Q[1],
        Q[2],
        Q[3] if True else Q[2],
        Q[4],
        Q[5],
        True,
        triton.next_power_of_2(Q[1]),
        1 if triton.next_power_of_2(Q[1]) >= 512 else 4 if Q[0] >= 4096 else 2,
    )
    _tile_inplace11(X, G, B, P)


__all__ = ["wan_rmsnorm_silu"]
