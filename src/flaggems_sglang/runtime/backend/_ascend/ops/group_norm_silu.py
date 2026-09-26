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


@triton.jit(
    do_not_specialize=["X", "W", "B", "Y"],
    do_not_specialize_on_alignment=["X", "W", "B", "Y"],
)
def _fused(X, W, B, Y, META: tl.constexpr):
    S: tl.constexpr = META[0]
    D: tl.constexpr = META[1]
    G: tl.constexpr = META[2]
    R: tl.constexpr = META[8]
    BD: tl.constexpr = META[4]
    BS: tl.constexpr = META[5]
    E: tl.constexpr = META[6]
    K = D * S
    DC: tl.constexpr = (D + BD - 1) // BD
    SC: tl.constexpr = (S + BS - 1) // BS
    RD: tl.constexpr = R if SC < R else 1
    RS: tl.constexpr = R // RD
    d0 = tl.arange(0, BD)
    s0 = tl.arange(0, BS)
    row = tl.program_id(0) // R
    part = tl.program_id(0) % R
    base = row * K
    cbase = row % G * D
    shift = tl.load(X + base).to(tl.float32)
    total = tl.zeros((), tl.float32)
    totalq = tl.zeros((), tl.float32)
    for t in tl.static_range(DC * SC):
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
    for offset_depth in tl.static_range((DC + RD - 1) // RD):
        d_tile = part // RS * ((DC + RD - 1) // RD) + offset_depth
        d = d_tile * BD + d0
        if D % BD == 0 and DC % RD == 0:
            w = tl.load(W + cbase + d).to(tl.float32)
            b = tl.load(B + cbase + d).to(tl.float32)
        else:
            valid = d < D
            w = tl.load(W + cbase + d, valid, 0.0).to(tl.float32)
            b = tl.load(B + cbase + d, valid, 0.0).to(tl.float32)
        for offset_tile in tl.static_range((SC + RS - 1) // RS):
            s_tile = part % RS * ((SC + RS - 1) // RS) + offset_tile
            s = s_tile * BS + s0
            x_tile = tl.make_block_ptr(
                X + base,
                (D, S),
                (S, 1),
                (d_tile * BD, s_tile * BS),
                (BD, BS),
                (1, 0),
            )
            y_tile = tl.make_block_ptr(
                Y + base,
                (D, S),
                (S, 1),
                (d_tile * BD, s_tile * BS),
                (BD, BS),
                (1, 0),
            )
            if (
                D % BD == 0
                and DC % RD == 0
                and (S % BS == 0)
                and (SC % RS == 0)
            ):
                y = (
                    tl.load(
                        x_tile, boundary_check=(0, 1), padding_option="zero"
                    ).to(tl.float32)
                    - mean
                ) * rstd * w[:, None] + b[:, None]
                tl.store(
                    y_tile,
                    (y / (1.0 + tl.exp(-y))).to(Y.dtype.element_ty),
                    boundary_check=(0, 1),
                )
            else:
                valid = d < D
                m = valid[:, None] & (s[None, :] < S)
                y = (
                    tl.load(
                        x_tile, boundary_check=(0, 1), padding_option="zero"
                    ).to(tl.float32)
                    - mean
                ) * rstd * w[:, None] + b[:, None]
                tl.store(
                    y_tile,
                    (y / (1.0 + tl.exp(-y))).to(Y.dtype.element_ty),
                    boundary_check=(0, 1),
                )


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


def _plan(shape, num_groups, eps):
    channels = shape[1]
    spatial = 1
    for extent in shape[2:]:
        spatial = spatial * extent
    depth = channels // num_groups
    rows = shape[0] * num_groups
    span = min(_pow2(spatial), 4096)
    replicas = (
        4
        if spatial > 12288 and rows <= 16
        else (2 if spatial > 4096 and rows <= 32 else 1)
    )
    programs = rows * replicas
    meta = (
        spatial,
        depth,
        num_groups,
        rows,
        min(_pow2(depth), 4096 // span),
        span,
        eps,
        programs,
        replicas,
    )
    return programs, meta


def group_norm_silu(x, weight, bias, num_groups, eps):
    x = _contiguous(x)
    weight = _contiguous(weight)
    bias = _contiguous(bias)
    out = torch.empty_like(x)
    programs, meta = _plan(tuple(x.shape), num_groups, eps)
    _fused[(programs,)](x, weight, bias, out, meta)
    return out


__all__ = ["group_norm_silu"]
