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
def _multi(
    X,
    W,
    B,
    Y,
    S: tl.constexpr,
    D: tl.constexpr,
    G: tl.constexpr,
    ROWS: tl.constexpr,
    BG: tl.constexpr,
    DP: tl.constexpr,
    SP: tl.constexpr,
    E: tl.constexpr,
):
    K = D * S
    g = tl.program_id(0) * BG + tl.arange(0, BG)
    d0 = tl.arange(0, DP)
    s0 = tl.arange(0, SP)
    off = g[:, None, None] * K + d0[None, :, None] * S + s0[None, None, :]
    ch = (g % G)[:, None] * D + d0[None, :]
    if ROWS % BG == 0 and D == DP and S == SP:
        shift = tl.load(X + g * K).to(tl.float32)[:, None, None]
        v = tl.load(X + off).to(tl.float32) - shift
        w = tl.load(W + ch).to(tl.float32)[:, :, None]
        b = tl.load(B + ch).to(tl.float32)[:, :, None]
        mu = tl.sum(tl.sum(v, 2), 1) / K
        rstd = tl.rsqrt(
            tl.maximum(tl.sum(tl.sum(v * v, 2), 1) / K - mu * mu, 0.0) + E
        )
        scale = rstd[:, None, None] * w
        y = v * scale + (b - mu[:, None, None] * scale)
        tl.store(Y + off, y * tl.sigmoid(y))
    else:
        live = g < ROWS
        dok = live[:, None] & (d0 < D)[None, :]
        m = dok[:, :, None] & (s0 < S)[None, None, :]
        shift = tl.load(X + tl.minimum(g, ROWS - 1) * K).to(tl.float32)[
            :, None, None
        ]
        v = tl.where(m, tl.load(X + off, m, 0.0).to(tl.float32) - shift, 0.0)
        w = tl.load(W + ch, dok, 0.0).to(tl.float32)[:, :, None]
        b = tl.load(B + ch, dok, 0.0).to(tl.float32)[:, :, None]
        mu = tl.sum(tl.sum(v, 2), 1) / K
        rstd = tl.rsqrt(
            tl.maximum(tl.sum(tl.sum(v * v, 2), 1) / K - mu * mu, 0.0) + E
        )
        scale = rstd[:, None, None] * w
        y = v * scale + (b - mu[:, None, None] * scale)
        tl.store(Y + off, y * tl.sigmoid(y), m)


@triton.jit
def _rows(
    X,
    W,
    B,
    Y,
    S: tl.constexpr,
    G: tl.constexpr,
    ROWS: tl.constexpr,
    BG: tl.constexpr,
    SP: tl.constexpr,
    E: tl.constexpr,
):
    g = tl.program_id(0) * BG + tl.arange(0, BG)
    s0 = tl.arange(0, SP)
    off = g[:, None] * S + s0[None, :]
    ch = g % G
    if ROWS % BG == 0 and S == SP:
        shift = tl.load(X + g * S).to(tl.float32)[:, None]
        v = tl.load(X + off).to(tl.float32) - shift
        w = tl.load(W + ch).to(tl.float32)[:, None]
        b = tl.load(B + ch).to(tl.float32)[:, None]
        mu = tl.sum(v, 1) / S
        rstd = tl.rsqrt(tl.maximum(tl.sum(v * v, 1) / S - mu * mu, 0.0) + E)
        scale = rstd[:, None] * w
        y = v * scale + (b - mu[:, None] * scale)
        tl.store(Y + off, y * tl.sigmoid(y))
    else:
        live = g < ROWS
        m = live[:, None] & (s0 < S)[None, :]
        shift = tl.load(X + tl.minimum(g, ROWS - 1) * S).to(tl.float32)[
            :, None
        ]
        v = tl.where(m, tl.load(X + off, m, 0.0).to(tl.float32) - shift, 0.0)
        w = tl.load(W + ch, live, 0.0).to(tl.float32)[:, None]
        b = tl.load(B + ch, live, 0.0).to(tl.float32)[:, None]
        mu = tl.sum(v, 1) / S
        rstd = tl.rsqrt(tl.maximum(tl.sum(v * v, 1) / S - mu * mu, 0.0) + E)
        scale = rstd[:, None] * w
        y = v * scale + (b - mu[:, None] * scale)
        tl.store(Y + off, y * tl.sigmoid(y), m)


@triton.jit
def _fused(
    X,
    W,
    B,
    Y,
    S: tl.constexpr,
    D: tl.constexpr,
    G: tl.constexpr,
    BD: tl.constexpr,
    BS: tl.constexpr,
    E: tl.constexpr,
):
    K = D * S
    DC = (D + BD - 1) // BD
    SC = (S + BS - 1) // BS
    base = tl.program_id(0) * K
    cbase = tl.program_id(0) % G * D
    d0 = tl.arange(0, BD)
    s0 = tl.arange(0, BS)
    shift = tl.load(X + base).to(tl.float32)
    total = tl.zeros((), tl.float32)
    totalq = tl.zeros((), tl.float32)
    for t in range(DC * SC):
        d = t // SC * BD + d0
        s = t % SC * BS + s0
        x_tile = tl.make_block_ptr(
            X + base,
            (D, S),
            (S, 1),
            (t // SC * BD, t % SC * BS),
            (BD, BS),
            (1, 0),
        )
        if D % BD == 0 and S % BS == 0:
            v = (
                tl.load(
                    x_tile, boundary_check=(0, 1), padding_option="zero"
                ).to(tl.float32)
                - shift
            )
        else:
            m = (d[:, None] < D) & (s[None, :] < S)
            v = tl.where(
                m,
                tl.load(
                    x_tile, boundary_check=(0, 1), padding_option="zero"
                ).to(tl.float32)
                - shift,
                0.0,
            )
        total += tl.sum(tl.sum(v, 1), 0)
        totalq += tl.sum(tl.sum(v * v, 1), 0)
    mu = total / K
    rstd = tl.rsqrt(tl.maximum(totalq / K - mu * mu, 0.0) + E)
    mean = shift + mu
    for t in range(DC * SC):
        d = t // SC * BD + d0
        s = t % SC * BS + s0
        x_tile = tl.make_block_ptr(
            X + base,
            (D, S),
            (S, 1),
            (t // SC * BD, t % SC * BS),
            (BD, BS),
            (1, 0),
        )
        y_tile = tl.make_block_ptr(
            Y + base,
            (D, S),
            (S, 1),
            (t // SC * BD, t % SC * BS),
            (BD, BS),
            (1, 0),
        )
        if D % BD == 0 and S % BS == 0:
            w = tl.load(W + cbase + d).to(tl.float32)[:, None]
            b = tl.load(B + cbase + d).to(tl.float32)[:, None]
            scale = rstd * w
            y = tl.load(
                x_tile, boundary_check=(0, 1), padding_option="zero"
            ).to(tl.float32) * scale + (b - mean * scale)
            tl.store(
                y_tile,
                (y * tl.sigmoid(y)).to(Y.dtype.element_ty),
                boundary_check=(0, 1),
            )
        else:
            valid = d < D
            m = valid[:, None] & (s[None, :] < S)
            w = tl.load(W + cbase + d, valid, 0.0).to(tl.float32)[:, None]
            b = tl.load(B + cbase + d, valid, 0.0).to(tl.float32)[:, None]
            scale = rstd * w
            y = tl.load(
                x_tile, boundary_check=(0, 1), padding_option="zero"
            ).to(tl.float32) * scale + (b - mean * scale)
            tl.store(
                y_tile,
                (y * tl.sigmoid(y)).to(Y.dtype.element_ty),
                boundary_check=(0, 1),
            )


@triton.jit
def _centered(
    X,
    W,
    B,
    Y,
    S: tl.constexpr,
    D: tl.constexpr,
    G: tl.constexpr,
    DP: tl.constexpr,
    SP: tl.constexpr,
    E: tl.constexpr,
):
    row = tl.program_id(0)
    d = tl.arange(0, DP)
    s = tl.arange(0, SP)
    valid = (d[:, None] < D) & (s[None, :] < S)
    off = row * D * S + d[:, None] * S + s[None, :]
    i = tl.arange(0, DP * SP)
    flat = tl.load(X + row * D * S + i).to(tl.float32)
    mean = tl.sum(flat, 0) / (D * S)
    flat = flat - mean
    var = tl.sum(flat * flat, 0) / (D * S)
    v = tl.reshape(flat, (DP, SP))
    r = tl.rsqrt(var + E)
    ch = row % G * D + d
    w = tl.load(W + ch, d < D, 0).to(tl.float32)
    b = tl.load(B + ch, d < D, 0).to(tl.float32)
    y = v * (r * w)[:, None] + b[:, None]
    tl.store(Y + off, y * tl.sigmoid(y), valid)


@triton.jit
def _copy_strided(
    X,
    Y,
    SHAPE: tl.constexpr,
    STRIDES: tl.constexpr,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    remainder = off
    source = tl.full((BLOCK,), 0, tl.int64)
    for axis in tl.static_range(len(SHAPE) - 1, -1, -1):
        source += (remainder % SHAPE[axis]) * STRIDES[axis]
        remainder = remainder // SHAPE[axis]
    value = tl.load(X + source, off < N, 0)
    tl.store(Y + off, value, off < N)


def _contiguous(x):
    if x.is_contiguous():
        return x
    out = torch.empty(x.shape, dtype=x.dtype, device=x.device)
    n = x.numel()
    if n:
        _copy_strided[(triton.cdiv(n, 1024),)](
            x, out, tuple(x.shape), x.stride(), n, 1024
        )
    return out


def _pow2(value):
    return 1 << (value - 1).bit_length()


def group_norm_silu(x, weight, bias, num_groups, eps):
    x = _contiguous(x)
    weight = _contiguous(weight)
    bias = _contiguous(bias)
    shape = x.shape
    channels = shape[1]
    spatial = x.numel() // (shape[0] * channels)
    depth = channels // num_groups
    rows = shape[0] * num_groups
    deep = _pow2(depth)
    span = _pow2(spatial)
    out = torch.empty_like(x)
    batch = 4096 // (deep * span)
    if batch > 1 and depth == 1:
        batch = min(_pow2(rows), batch)
        _rows[(-(-rows // batch),)](
            x,
            weight,
            bias,
            out,
            spatial,
            num_groups,
            rows,
            batch,
            span,
            eps,
            num_warps=1,
        )
    elif batch > 1:
        batch = min(_pow2(rows), batch)
        _multi[(-(-rows // batch),)](
            x,
            weight,
            bias,
            out,
            spatial,
            depth,
            num_groups,
            rows,
            batch,
            deep,
            span,
            eps,
            num_warps=1,
        )
    elif depth == deep and spatial == span and deep * span <= 16384:
        _centered[(rows,)](
            x,
            weight,
            bias,
            out,
            spatial,
            depth,
            num_groups,
            deep,
            span,
            eps,
            num_warps=1,
        )
    else:
        cols = min(span, 4096)
        _fused[(rows,)](
            x,
            weight,
            bias,
            out,
            spatial,
            depth,
            num_groups,
            min(deep, 4096 // cols),
            cols,
            eps,
            num_warps=1,
        )
    return out


__all__ = ["group_norm_silu"]
