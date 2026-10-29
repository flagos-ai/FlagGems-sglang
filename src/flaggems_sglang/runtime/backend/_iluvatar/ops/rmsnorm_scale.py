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
def _parts4(X, W, S, Y, P: tl.constexpr):
    D: tl.constexpr = P[0]
    REP: tl.constexpr = P[1]
    EPS: tl.constexpr = P[2]
    B: tl.constexpr = P[3]
    row = tl.program_id(0)
    c = tl.arange(0, B // 4)
    c0 = c + 0 * (B // 4)
    x0 = tl.load(X + row * D + c0, c0 < D, other=0).to(tl.float32)
    c1 = c + 1 * (B // 4)
    x1 = tl.load(X + row * D + c1, c1 < D, other=0).to(tl.float32)
    c2 = c + 2 * (B // 4)
    x2 = tl.load(X + row * D + c2, c2 < D, other=0).to(tl.float32)
    c3 = c + 3 * (B // 4)
    x3 = tl.load(X + row * D + c3, c3 < D, other=0).to(tl.float32)
    inv = tl.rsqrt(tl.sum(x0 * x0 + x1 * x1 + x2 * x2 + x3 * x3, 0) / D + EPS)
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
    w2 = tl.load(W + c2, c2 < D, other=0).to(tl.float32)
    s2 = tl.load(S + row // REP * D + c2, c2 < D, other=0).to(tl.float32)
    n2 = (x2 * inv).to(tl.bfloat16).to(tl.float32)
    z2 = (n2 * w2).to(tl.bfloat16).to(tl.float32)
    tl.store(Y + row * D + c2, z2 * s2, c2 < D)
    w3 = tl.load(W + c3, c3 < D, other=0).to(tl.float32)
    s3 = tl.load(S + row // REP * D + c3, c3 < D, other=0).to(tl.float32)
    n3 = (x3 * inv).to(tl.bfloat16).to(tl.float32)
    z3 = (n3 * w3).to(tl.bfloat16).to(tl.float32)
    tl.store(Y + row * D + c3, z3 * s3, c3 < D)


@triton.jit
def _parts8(X, W, S, Y, P: tl.constexpr):
    D: tl.constexpr = P[0]
    REP: tl.constexpr = P[1]
    EPS: tl.constexpr = P[2]
    B: tl.constexpr = P[3]
    row = tl.program_id(0)
    c = tl.arange(0, B // 8)
    c0 = c + 0 * (B // 8)
    x0 = tl.load(X + row * D + c0, c0 < D, other=0).to(tl.float32)
    c1 = c + 1 * (B // 8)
    x1 = tl.load(X + row * D + c1, c1 < D, other=0).to(tl.float32)
    c2 = c + 2 * (B // 8)
    x2 = tl.load(X + row * D + c2, c2 < D, other=0).to(tl.float32)
    c3 = c + 3 * (B // 8)
    x3 = tl.load(X + row * D + c3, c3 < D, other=0).to(tl.float32)
    c4 = c + 4 * (B // 8)
    x4 = tl.load(X + row * D + c4, c4 < D, other=0).to(tl.float32)
    c5 = c + 5 * (B // 8)
    x5 = tl.load(X + row * D + c5, c5 < D, other=0).to(tl.float32)
    c6 = c + 6 * (B // 8)
    x6 = tl.load(X + row * D + c6, c6 < D, other=0).to(tl.float32)
    c7 = c + 7 * (B // 8)
    x7 = tl.load(X + row * D + c7, c7 < D, other=0).to(tl.float32)
    inv = tl.rsqrt(
        tl.sum(
            x0 * x0
            + x1 * x1
            + x2 * x2
            + x3 * x3
            + x4 * x4
            + x5 * x5
            + x6 * x6
            + x7 * x7,
            0,
        )
        / D
        + EPS
    )
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
    w2 = tl.load(W + c2, c2 < D, other=0).to(tl.float32)
    s2 = tl.load(S + row // REP * D + c2, c2 < D, other=0).to(tl.float32)
    n2 = (x2 * inv).to(tl.bfloat16).to(tl.float32)
    z2 = (n2 * w2).to(tl.bfloat16).to(tl.float32)
    tl.store(Y + row * D + c2, z2 * s2, c2 < D)
    w3 = tl.load(W + c3, c3 < D, other=0).to(tl.float32)
    s3 = tl.load(S + row // REP * D + c3, c3 < D, other=0).to(tl.float32)
    n3 = (x3 * inv).to(tl.bfloat16).to(tl.float32)
    z3 = (n3 * w3).to(tl.bfloat16).to(tl.float32)
    tl.store(Y + row * D + c3, z3 * s3, c3 < D)
    w4 = tl.load(W + c4, c4 < D, other=0).to(tl.float32)
    s4 = tl.load(S + row // REP * D + c4, c4 < D, other=0).to(tl.float32)
    n4 = (x4 * inv).to(tl.bfloat16).to(tl.float32)
    z4 = (n4 * w4).to(tl.bfloat16).to(tl.float32)
    tl.store(Y + row * D + c4, z4 * s4, c4 < D)
    w5 = tl.load(W + c5, c5 < D, other=0).to(tl.float32)
    s5 = tl.load(S + row // REP * D + c5, c5 < D, other=0).to(tl.float32)
    n5 = (x5 * inv).to(tl.bfloat16).to(tl.float32)
    z5 = (n5 * w5).to(tl.bfloat16).to(tl.float32)
    tl.store(Y + row * D + c5, z5 * s5, c5 < D)
    w6 = tl.load(W + c6, c6 < D, other=0).to(tl.float32)
    s6 = tl.load(S + row // REP * D + c6, c6 < D, other=0).to(tl.float32)
    n6 = (x6 * inv).to(tl.bfloat16).to(tl.float32)
    z6 = (n6 * w6).to(tl.bfloat16).to(tl.float32)
    tl.store(Y + row * D + c6, z6 * s6, c6 < D)
    w7 = tl.load(W + c7, c7 < D, other=0).to(tl.float32)
    s7 = tl.load(S + row // REP * D + c7, c7 < D, other=0).to(tl.float32)
    n7 = (x7 * inv).to(tl.bfloat16).to(tl.float32)
    z7 = (n7 * w7).to(tl.bfloat16).to(tl.float32)
    tl.store(Y + row * D + c7, z7 * s7, c7 < D)


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


@triton.jit
def _lean(X, W, S, Y, PARAMS: tl.constexpr):
    D: tl.constexpr = PARAMS[0]
    REP: tl.constexpr = PARAMS[1]
    EPS: tl.constexpr = PARAMS[2]
    B: tl.constexpr = PARAMS[3]
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
def _bf_lean(X, W, S, Y, PARAMS: tl.constexpr):
    D: tl.constexpr = PARAMS[0]
    REP: tl.constexpr = PARAMS[1]
    EPS: tl.constexpr = PARAMS[2]
    B: tl.constexpr = PARAMS[3]
    row = tl.program_id(0)
    col = tl.arange(0, B)
    x = tl.load(X + row * D + col, col < D, other=0).to(tl.float32)
    w = tl.load(W + col, col < D, other=0)
    s = tl.load(S + row // REP * D + col, col < D, other=0)
    inv = tl.rsqrt(tl.sum(x * x, 0) / D + EPS)
    n = (x * inv).to(tl.bfloat16)
    z = (n * w).to(tl.bfloat16)
    tl.store(Y + row * D + col, z * s, col < D)


@triton.jit
def _exact_parts3(X, W, S, Y, P: tl.constexpr):
    D: tl.constexpr = P[0]
    REP: tl.constexpr = P[1]
    EPS: tl.constexpr = P[2]
    B: tl.constexpr = P[3]
    row = tl.program_id(0)
    c = tl.arange(0, B // 3)
    c0 = c + 0 * (B // 3)
    x0 = tl.load(X + row * D + c0, c0 < D, other=0).to(tl.float32)
    c1 = c + 1 * (B // 3)
    x1 = tl.load(X + row * D + c1, c1 < D, other=0).to(tl.float32)
    c2 = c + 2 * (B // 3)
    x2 = tl.load(X + row * D + c2, c2 < D, other=0).to(tl.float32)
    inv = tl.rsqrt(tl.sum(x0 * x0 + x1 * x1 + x2 * x2, 0) / D + EPS)
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
    w2 = tl.load(W + c2, c2 < D, other=0).to(tl.float32)
    s2 = tl.load(S + row // REP * D + c2, c2 < D, other=0).to(tl.float32)
    n2 = (x2 * inv).to(tl.bfloat16).to(tl.float32)
    z2 = (n2 * w2).to(tl.bfloat16).to(tl.float32)
    tl.store(Y + row * D + c2, z2 * s2, c2 < D)


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
    y = torch.empty_like(x, memory_format=torch.contiguous_format)
    if m:
        b = 1 << max(6, (d - 1).bit_length())
        if xc and sc and (d == 1536) and (32 <= m < 1024):
            _exact_parts3.run(
                x,
                weight,
                scale,
                y,
                (d, m // sm, float(eps), d),
                grid=(m,),
                warmup=False,
                num_warps=4,
                enable_fp_fusion=False,
            )
        elif xc and sc and (m >= 1024) and (d in (2048, 4096)):
            fn = _parts8 if d == 4096 else _parts4
            fn.run(
                x,
                weight,
                scale,
                y,
                (d, m // sm, float(eps), b),
                grid=(m,),
                warmup=False,
                num_warps=4,
                enable_fp_fusion=False,
            )
        elif xc and sc:
            (_bf_lean if m < 32 and d == 4096 else _lean).run(
                x,
                weight,
                scale,
                y,
                (d, m // sm, float(eps), b),
                grid=(m,),
                warmup=False,
                num_warps=(
                    1
                    if d <= 256 or (d >= 4096 and m >= 8192)
                    else 8 if m < 32 or d <= 2048 else 4
                ),
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


__all__ = ["rmsnorm_scale"]
