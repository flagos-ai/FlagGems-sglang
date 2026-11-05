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
def _lean(X, W, S, Y, P: tl.constexpr):
    D: tl.constexpr = P[0]
    REP: tl.constexpr = P[1]
    EPS: tl.constexpr = P[2]
    B: tl.constexpr = P[3]
    CACHE: tl.constexpr = P[4]
    STORE: tl.constexpr = P[5]
    WC: tl.constexpr = P[6]
    row = tl.program_id(0)
    col = tl.arange(0, B)
    x = tl.load(X + row * D + col, col < D, other=0, cache_modifier=CACHE).to(
        tl.float32
    )
    w = tl.load(W + col, col < D, other=0, cache_modifier=WC).to(tl.float32)
    s = tl.load(
        S + row // REP * D + col, col < D, other=0, cache_modifier=CACHE
    ).to(tl.float32)
    inv = tl.rsqrt(tl.sum(x * x, 0) / D + EPS)
    n = (x * inv).to(tl.bfloat16).to(tl.float32)
    z = (n * w).to(tl.bfloat16).to(tl.float32)
    tl.store(Y + row * D + col, z * s, col < D, cache_modifier=STORE)


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
    m, sm = (xshape.numel() // d, sshape.numel() // d)
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
        if xc and sc:
            xp, wp, sp, yp = (
                x.data_ptr(),
                weight.data_ptr(),
                scale.data_ptr(),
                y.data_ptr(),
            )
            input_alignment = xp | wp | sp
            aligned = (input_alignment | yp) & 15 == 0
            if not aligned:
                xp, wp, sp, yp = (x, weight, scale, y)
        else:
            aligned = False
            xp, wp, sp, yp = (x, weight, scale, y)
        b = 1 << max(6, (d - 1).bit_length())
        if xc and sc:
            bf = (
                torch.version.hip is not None
                and d == 2048
                and (1024 <= m < 8192)
            )
            (
                (
                    _address_hint_lean_delayed_scale
                    if aligned
                    else _lean_delayed_scale
                )
                if torch.version.hip is not None and d == 4096 and (m >= 1024)
                else (
                    (_address_hint_bf_lean if aligned else _bf_lean)
                    if bf
                    else _address_hint_lean if aligned else _lean
                )
            ).run(
                xp,
                wp,
                sp,
                yp,
                (
                    d,
                    m // sm,
                    float(eps),
                    b,
                    (
                        ".cg"
                        if torch.version.hip is not None or m >= 1024
                        else ""
                    ),
                    (
                        ".cs"
                        if torch.version.hip is not None and m >= 1024
                        else ""
                    ),
                    ".cg" if torch.version.hip is not None else "",
                ),
                grid=(m,),
                warmup=False,
                num_warps=(
                    2
                    if bf
                    else (
                        4
                        if torch.version.hip is None
                        and d <= 2048
                        and (m >= 1024)
                        else 8
                    )
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


@triton.jit
def _bf_lean(X, W, S, Y, P: tl.constexpr):
    D: tl.constexpr = P[0]
    REP: tl.constexpr = P[1]
    EPS: tl.constexpr = P[2]
    B: tl.constexpr = P[3]
    CACHE: tl.constexpr = P[4]
    STORE: tl.constexpr = P[5]
    WC: tl.constexpr = P[6]
    row = tl.program_id(0)
    col = tl.arange(0, B)
    x = tl.load(X + row * D + col, col < D, other=0, cache_modifier=CACHE).to(
        tl.float32
    )
    w = tl.load(W + col, col < D, other=0, cache_modifier=WC)
    s = tl.load(
        S + row // REP * D + col, col < D, other=0, cache_modifier=CACHE
    )
    inv = tl.rsqrt(tl.sum(x * x, 0) / D + EPS)
    n = (x * inv).to(tl.bfloat16)
    z = (n * w).to(tl.bfloat16)
    tl.store(Y + row * D + col, z * s, col < D, cache_modifier=STORE)


@triton.jit
def _lean_delayed_scale(X, W, S, Y, P: tl.constexpr):
    D: tl.constexpr = P[0]
    REP: tl.constexpr = P[1]
    EPS: tl.constexpr = P[2]
    B: tl.constexpr = P[3]
    CACHE: tl.constexpr = P[4]
    STORE: tl.constexpr = P[5]
    WC: tl.constexpr = P[6]
    row = tl.program_id(0)
    col = tl.arange(0, B)
    x = tl.load(X + row * D + col, col < D, other=0, cache_modifier=CACHE).to(
        tl.float32
    )
    w = tl.load(W + col, col < D, other=0, cache_modifier=WC).to(tl.float32)
    inv = tl.rsqrt(tl.sum(x * x, 0) / D + EPS)
    s = tl.load(
        S + row // REP * D + col, col < D, other=0, cache_modifier=CACHE
    ).to(tl.float32)
    n = (x * inv).to(tl.bfloat16).to(tl.float32)
    z = (n * w).to(tl.bfloat16).to(tl.float32)
    tl.store(Y + row * D + col, z * s, col < D, cache_modifier=STORE)


@triton.jit(do_not_specialize=["X", "W", "S", "Y"])
def _address_hint_bf_lean(X, W, S, Y, P: tl.constexpr):
    X = X.to(tl.pointer_type(tl.bfloat16))
    W = W.to(tl.pointer_type(tl.bfloat16))
    S = S.to(tl.pointer_type(tl.bfloat16))
    Y = Y.to(tl.pointer_type(tl.bfloat16))
    X = tl.multiple_of(X, 16)
    W = tl.multiple_of(W, 16)
    S = tl.multiple_of(S, 16)
    Y = tl.multiple_of(Y, 16)
    D: tl.constexpr = P[0]
    REP: tl.constexpr = P[1]
    EPS: tl.constexpr = P[2]
    B: tl.constexpr = P[3]
    CACHE: tl.constexpr = P[4]
    STORE: tl.constexpr = P[5]
    WC: tl.constexpr = P[6]
    row = tl.program_id(0)
    col = tl.arange(0, B)
    x = tl.load(X + row * D + col, col < D, other=0, cache_modifier=CACHE).to(
        tl.float32
    )
    w = tl.load(W + col, col < D, other=0, cache_modifier=WC)
    s = tl.load(
        S + row // REP * D + col, col < D, other=0, cache_modifier=CACHE
    )
    inv = tl.rsqrt(tl.sum(x * x, 0) / D + EPS)
    n = (x * inv).to(tl.bfloat16)
    z = (n * w).to(tl.bfloat16)
    tl.store(Y + row * D + col, z * s, col < D, cache_modifier=STORE)


@triton.jit(do_not_specialize=["X", "W", "S", "Y"])
def _address_hint_lean(X, W, S, Y, P: tl.constexpr):
    X = X.to(tl.pointer_type(tl.bfloat16))
    W = W.to(tl.pointer_type(tl.bfloat16))
    S = S.to(tl.pointer_type(tl.bfloat16))
    Y = Y.to(tl.pointer_type(tl.bfloat16))
    X = tl.multiple_of(X, 16)
    W = tl.multiple_of(W, 16)
    S = tl.multiple_of(S, 16)
    Y = tl.multiple_of(Y, 16)
    D: tl.constexpr = P[0]
    REP: tl.constexpr = P[1]
    EPS: tl.constexpr = P[2]
    B: tl.constexpr = P[3]
    CACHE: tl.constexpr = P[4]
    STORE: tl.constexpr = P[5]
    WC: tl.constexpr = P[6]
    row = tl.program_id(0)
    col = tl.arange(0, B)
    x = tl.load(X + row * D + col, col < D, other=0, cache_modifier=CACHE).to(
        tl.float32
    )
    w = tl.load(W + col, col < D, other=0, cache_modifier=WC).to(tl.float32)
    s = tl.load(
        S + row // REP * D + col, col < D, other=0, cache_modifier=CACHE
    ).to(tl.float32)
    inv = tl.rsqrt(tl.sum(x * x, 0) / D + EPS)
    n = (x * inv).to(tl.bfloat16).to(tl.float32)
    z = (n * w).to(tl.bfloat16).to(tl.float32)
    tl.store(Y + row * D + col, z * s, col < D, cache_modifier=STORE)


@triton.jit(do_not_specialize=["X", "W", "S", "Y"])
def _address_hint_lean_delayed_scale(X, W, S, Y, P: tl.constexpr):
    X = X.to(tl.pointer_type(tl.bfloat16))
    W = W.to(tl.pointer_type(tl.bfloat16))
    S = S.to(tl.pointer_type(tl.bfloat16))
    Y = Y.to(tl.pointer_type(tl.bfloat16))
    X = tl.multiple_of(X, 16)
    W = tl.multiple_of(W, 16)
    S = tl.multiple_of(S, 16)
    Y = tl.multiple_of(Y, 16)
    D: tl.constexpr = P[0]
    REP: tl.constexpr = P[1]
    EPS: tl.constexpr = P[2]
    B: tl.constexpr = P[3]
    CACHE: tl.constexpr = P[4]
    STORE: tl.constexpr = P[5]
    WC: tl.constexpr = P[6]
    row = tl.program_id(0)
    col = tl.arange(0, B)
    x = tl.load(X + row * D + col, col < D, other=0, cache_modifier=CACHE).to(
        tl.float32
    )
    w = tl.load(W + col, col < D, other=0, cache_modifier=WC).to(tl.float32)
    inv = tl.rsqrt(tl.sum(x * x, 0) / D + EPS)
    s = tl.load(
        S + row // REP * D + col, col < D, other=0, cache_modifier=CACHE
    ).to(tl.float32)
    n = (x * inv).to(tl.bfloat16).to(tl.float32)
    z = (n * w).to(tl.bfloat16).to(tl.float32)
    tl.store(Y + row * D + col, z * s, col < D, cache_modifier=STORE)


__all__ = ["rmsnorm_scale"]
