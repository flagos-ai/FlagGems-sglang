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


@triton.jit
def _offset(row, SH: tl.constexpr, ST: tl.constexpr):
    base = tl.full((), 0, tl.int64)
    for j in tl.static_range(len(SH) - 1, -1, -1):
        base += (row % SH[j]).to(tl.int64) * ST[j]
        row //= SH[j]
    return base


@triton.jit
def _row(
    X,
    W,
    S,
    Y,
    D: tl.constexpr,
    REP: tl.constexpr,
    XSH: tl.constexpr,
    XST: tl.constexpr,
    SSH: tl.constexpr,
    SST: tl.constexpr,
    EPS: tl.constexpr,
    B: tl.constexpr,
):
    row = tl.program_id(0)
    c = tl.arange(0, B)
    x = tl.load(X + _offset(row, XSH, XST) + c * XST[-1], c < D, other=0).to(
        tl.float32
    )
    w = tl.load(W + c, c < D, other=0).to(tl.float32)
    s = tl.load(
        S + _offset(row // REP, SSH, SST) + c * SST[-1], c < D, other=0
    ).to(tl.float32)
    inv = tl.rsqrt(tl.sum(x * x, 0) / D + EPS)
    n = (x * inv).to(tl.bfloat16).to(tl.float32)
    z = (n * w).to(tl.bfloat16).to(tl.float32)
    tl.store(Y + row * D + c, z * s, c < D)


@triton.jit
def _lean(X, W, S, Y, P: tl.constexpr):
    D: tl.constexpr = P[0]
    REP: tl.constexpr = P[1]
    EPS: tl.constexpr = P[2]
    B: tl.constexpr = P[3]
    row = tl.program_id(0)
    col = tl.arange(0, B)
    x = tl.load(X + row * D + col, col < D, other=0).to(tl.float32)
    w = tl.load(W + col, col < D, other=0).to(tl.float32)
    s = tl.load(S + row // REP * D + col, col < D, other=0).to(tl.float32)
    inv = tl.rsqrt(tl.sum(x * x, 0) / D + EPS)
    n = (x * inv).to(tl.bfloat16).to(tl.float32)
    z = (n * w).to(tl.bfloat16).to(tl.float32)
    tl.store(Y + row * D + col, z * s, col < D)


@triton.jit
def _parts2(X, W, S, Y, P: tl.constexpr):
    D: tl.constexpr = P[0]
    REP: tl.constexpr = P[1]
    EPS: tl.constexpr = P[2]
    B: tl.constexpr = P[3]
    row = tl.program_id(0)
    c = tl.arange(0, B // 2)
    c0 = c + 0 * (B // 2)
    x0 = tl.load(X + row * D + c0, c0 < D, other=0).to(tl.float32)
    c1 = c + 1 * (B // 2)
    x1 = tl.load(X + row * D + c1, c1 < D, other=0).to(tl.float32)
    inv = tl.rsqrt(tl.sum(x0 * x0 + x1 * x1, 0) / D + EPS)
    w0 = tl.load(W + c0, c0 < D, other=0).to(tl.float32)
    s0 = tl.load(S + row // REP * D + c0, c0 < D, other=0).to(tl.float32)
    n0 = (x0 * inv).to(tl.bfloat16).to(tl.float32)
    z0 = (n0 * w0).to(tl.bfloat16).to(tl.float32)
    tl.store(Y + row * D + c0, z0 * s0, c0 < D)
    w1 = tl.load(W + c1, c1 < D, other=0).to(tl.float32)
    s1 = tl.load(S + row // REP * D + c1, c1 < D, other=0).to(tl.float32)
    n1 = (x1 * inv).to(tl.bfloat16).to(tl.float32)
    z1 = (n1 * w1).to(tl.bfloat16).to(tl.float32)
    tl.store(Y + row * D + c1, z1 * s1, c1 < D)


def _flat_rows(t):
    if t.ndim <= 2:
        return True
    row_stride = None
    rows = 1
    for size, stride in zip(reversed(t.shape[:-1]), reversed(t.stride()[:-1])):
        if size == 1:
            continue
        if row_stride is None:
            row_stride = stride
        elif stride != row_stride * rows:
            return False
        rows *= size
    return True


def rmsnorm_scale(x, weight, scale, eps):
    xshape = x.shape
    sshape = scale.shape
    if not xshape or not sshape:
        return None
    if not x.dtype == weight.dtype == scale.dtype == torch.bfloat16:
        return None
    if not x.device == weight.device == scale.device:
        return None
    d = xshape[-1]
    if (
        not 0 < d <= 8192
        or weight.shape != (d,)
        or (not weight.is_contiguous())
        or (sshape[-1] != d)
    ):
        return None
    m, sm = (x.numel() // d, scale.numel() // d)
    if sm == 0 or m % sm != 0:
        return None
    xc, sc = (x.is_contiguous(), scale.is_contiguous())
    if not xc and (not _flat_rows(x)) or (not sc and (not _flat_rows(scale))):
        return None
    y = (
        torch.empty_like(x)
        if xc
        else torch.empty_like(x, memory_format=torch.contiguous_format)
    )
    if m:
        b = 1 << max(6, (d - 1).bit_length())
        if xc and sc and (d == 2048 and m >= 8192 or d == 4096 and m >= 1024):
            nw = 2 if d == 2048 else 4
            _gluon_row.run(
                x,
                weight,
                scale,
                y,
                d,
                m // sm,
                float(eps),
                b,
                8,
                nw,
                grid=(m,),
                warmup=False,
                num_warps=nw,
                enable_fp_fusion=False,
            )
        elif xc and sc and (m >= 1024) and (d in (2048, 4096)):
            _parts2.run(
                x,
                weight,
                scale,
                y,
                (d, m // sm, float(eps), b),
                grid=(m,),
                warmup=False,
                num_warps=2,
                enable_fp_fusion=False,
            )
        elif xc and sc:
            _lean.run(
                x,
                weight,
                scale,
                y,
                (d, m // sm, float(eps), b),
                grid=(m,),
                warmup=False,
                num_warps=8 if m < 1024 and 256 < d <= 4096 else 4,
                enable_fp_fusion=False,
            )
        else:
            _row[m,](
                x,
                weight,
                scale,
                y,
                d,
                m // sm,
                tuple(x.shape[:-1]),
                tuple(x.stride()),
                tuple(scale.shape[:-1]),
                tuple(scale.stride()),
                float(eps),
                b,
                num_warps=4 if b <= 2048 else 8,
                num_stages=1,
                enable_fp_fusion=False,
            )
    return y


@gluon.jit
def _gluon_row(
    X,
    W,
    S,
    Y,
    D: gl.constexpr,
    REP: gl.constexpr,
    EPS: gl.constexpr,
    B: gl.constexpr,
    V: gl.constexpr,
    NW: gl.constexpr,
):
    row = gl.program_id(0)
    c = gl.arange(0, B, layout=gl.BlockedLayout([V], [64], [NW], [0]))
    x = gl.load(X + row * D + c, c < D, other=0).to(gl.float32)
    inv = gl.rsqrt(gl.sum(x * x, 0) / D + EPS)
    w = gl.load(W + c, c < D, other=0).to(gl.float32)
    s = gl.load(S + row // REP * D + c, c < D, other=0).to(gl.float32)
    n = (x * inv).to(gl.bfloat16).to(gl.float32)
    z = (n * w).to(gl.bfloat16).to(gl.float32)
    gl.store(Y + row * D + c, z * s, c < D)


__all__ = ["rmsnorm_scale"]
