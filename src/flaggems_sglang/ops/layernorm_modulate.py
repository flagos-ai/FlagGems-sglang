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
def _norm(
    X,
    A,
    H,
    Y,
    S: tl.constexpr,
    D: tl.constexpr,
    X0: tl.constexpr,
    X1: tl.constexpr,
    X2: tl.constexpr,
    A0: tl.constexpr,
    A1: tl.constexpr,
    H0: tl.constexpr,
    H1: tl.constexpr,
    EPS: tl.constexpr,
    C: tl.constexpr,
    SHIFT: tl.constexpr,
):
    row = tl.program_id(0)
    b = row // S
    s = row % S
    col = tl.arange(0, C)
    x = tl.load(X + b * X0 + s * X1 + col * X2, col < D, 0).to(tl.float32)
    if SHIFT:
        anchor = tl.load(X + b * X0 + s * X1).to(tl.float32)
        x = tl.where(col < D, x - anchor, 0.0)
    mean = tl.sum(x, 0) / D
    z = tl.where(col < D, x - mean, 0.0)
    var = tl.sum(z * z, 0) / D
    a = tl.load(A + b * A0 + col * A1, col < D, 0).to(tl.float32)
    h = tl.load(H + b * H0 + col * H1, col < D, 0).to(tl.float32)
    y = z * tl.rsqrt(var + EPS)
    y = y * (1.0 + a) + h
    tl.store(Y + row * D + col, y, col < D)


def _current(x, scale, shift, eps):
    b, s, d = x.shape
    y = torch.empty((b, s, d), device=x.device, dtype=x.dtype)
    c = triton.next_power_of_2(d)
    kernel = _norm
    if d == 2240 and x.dtype == torch.bfloat16:
        if b * s >= 16384:
            kernel = _stream_large
        elif b * s >= 4096:
            kernel = _stream_medium
    kernel[b * s,](
        x,
        scale,
        shift,
        y,
        s,
        d,
        *x.stride(),
        *scale.stride(),
        *shift.stride(),
        eps,
        c,
        x.dtype == torch.float32,
        num_warps=4,
        enable_fp_fusion=False,
    )
    return y


@triton.jit
def _stream_large(
    X,
    A,
    H,
    Y,
    S: tl.constexpr,
    D: tl.constexpr,
    X0: tl.constexpr,
    X1: tl.constexpr,
    X2: tl.constexpr,
    A0: tl.constexpr,
    A1: tl.constexpr,
    H0: tl.constexpr,
    H1: tl.constexpr,
    EPS: tl.constexpr,
    C: tl.constexpr,
    SHIFT: tl.constexpr,
):
    row = tl.program_id(0)
    b = row // S
    s = row % S
    col = tl.arange(0, C)
    x = tl.load(
        X + b * X0 + s * X1 + col * X2, col < D, 0, cache_modifier=".cg"
    ).to(tl.float32)
    if SHIFT:
        anchor = tl.load(X + b * X0 + s * X1, cache_modifier=".cg").to(
            tl.float32
        )
        x = tl.where(col < D, x - anchor, 0.0)
    mean = tl.sum(x, 0) / D
    z = tl.where(col < D, x - mean, 0.0)
    var = tl.sum(z * z, 0) / D
    a = tl.load(A + b * A0 + col * A1, col < D, 0).to(tl.float32)
    h = tl.load(H + b * H0 + col * H1, col < D, 0).to(tl.float32)
    y = z * tl.rsqrt(var + EPS)
    y = y * (1.0 + a) + h
    tl.store(Y + row * D + col, y, col < D, cache_modifier=".cs")


@triton.jit
def _stream_medium(
    X,
    A,
    H,
    Y,
    S: tl.constexpr,
    D: tl.constexpr,
    X0: tl.constexpr,
    X1: tl.constexpr,
    X2: tl.constexpr,
    A0: tl.constexpr,
    A1: tl.constexpr,
    H0: tl.constexpr,
    H1: tl.constexpr,
    EPS: tl.constexpr,
    C: tl.constexpr,
    SHIFT: tl.constexpr,
):
    row = tl.program_id(0)
    b = row // S
    s = row % S
    col = tl.arange(0, C)
    x = tl.load(
        X + b * X0 + s * X1 + col * X2, col < D, 0, cache_modifier=".cg"
    ).to(tl.float32)
    if SHIFT:
        anchor = tl.load(X + b * X0 + s * X1, cache_modifier=".cg").to(
            tl.float32
        )
        x = tl.where(col < D, x - anchor, 0.0)
    mean = tl.sum(x, 0) / D
    z = tl.where(col < D, x - mean, 0.0)
    var = tl.sum(z * z, 0) / D
    a = tl.load(A + b * A0 + col * A1, col < D, 0).to(tl.float32)
    h = tl.load(H + b * H0 + col * H1, col < D, 0).to(tl.float32)
    y = z * tl.rsqrt(var + EPS)
    y = y * (1.0 + a) + h
    tl.store(Y + row * D + col, y, col < D)


@triton.jit
def _flat_norm(
    X,
    A,
    H,
    Y,
    S: tl.constexpr,
    D: tl.constexpr,
    EPS: tl.constexpr,
    C: tl.constexpr,
    SHIFT: tl.constexpr,
):
    row = tl.program_id(0)
    b = row // S
    col = tl.arange(0, C)
    x = tl.load(X + row * D + col, col < D, 0).to(tl.float32)
    if SHIFT:
        anchor = tl.load(X + row * D).to(tl.float32)
        x = tl.where(col < D, x - anchor, 0.0)
    mean = tl.sum(x, 0) / D
    z = tl.where(col < D, x - mean, 0.0)
    var = tl.sum(z * z, 0) / D
    a = tl.load(A + b * D + col, col < D, 0).to(tl.float32)
    h = tl.load(H + b * D + col, col < D, 0).to(tl.float32)
    y = z * tl.rsqrt(var + EPS)
    y = y * (1.0 + a) + h
    tl.store(Y + row * D + col, y, col < D)


@triton.jit
def _flat_stream_large(
    X,
    A,
    H,
    Y,
    S: tl.constexpr,
    D: tl.constexpr,
    EPS: tl.constexpr,
    C: tl.constexpr,
    SHIFT: tl.constexpr,
):
    row = tl.program_id(0)
    b = row // S
    col = tl.arange(0, C)
    x = tl.load(X + row * D + col, col < D, 0, cache_modifier=".cg").to(
        tl.float32
    )
    if SHIFT:
        anchor = tl.load(X + row * D, cache_modifier=".cg").to(tl.float32)
        x = tl.where(col < D, x - anchor, 0.0)
    mean = tl.sum(x, 0) / D
    z = tl.where(col < D, x - mean, 0.0)
    var = tl.sum(z * z, 0) / D
    a = tl.load(A + b * D + col, col < D, 0).to(tl.float32)
    h = tl.load(H + b * D + col, col < D, 0).to(tl.float32)
    y = z * tl.rsqrt(var + EPS)
    y = y * (1.0 + a) + h
    tl.store(Y + row * D + col, y, col < D, cache_modifier=".cs")


@triton.jit
def _flat_stream_medium(
    X,
    A,
    H,
    Y,
    S: tl.constexpr,
    D: tl.constexpr,
    EPS: tl.constexpr,
    C: tl.constexpr,
    SHIFT: tl.constexpr,
):
    row = tl.program_id(0)
    b = row // S
    col = tl.arange(0, C)
    x = tl.load(X + row * D + col, col < D, 0, cache_modifier=".cg").to(
        tl.float32
    )
    if SHIFT:
        anchor = tl.load(X + row * D, cache_modifier=".cg").to(tl.float32)
        x = tl.where(col < D, x - anchor, 0.0)
    mean = tl.sum(x, 0) / D
    z = tl.where(col < D, x - mean, 0.0)
    var = tl.sum(z * z, 0) / D
    a = tl.load(A + b * D + col, col < D, 0).to(tl.float32)
    h = tl.load(H + b * D + col, col < D, 0).to(tl.float32)
    y = z * tl.rsqrt(var + EPS)
    y = y * (1.0 + a) + h
    tl.store(Y + row * D + col, y, col < D)


def layernorm_modulate(x, scale, shift, eps):
    if not (
        x.is_contiguous() and scale.is_contiguous() and shift.is_contiguous()
    ):
        return _current(x, scale, shift, eps)
    b, s, d = x.shape
    y = torch.empty_like(x)
    c = triton.next_power_of_2(d)
    kernel = _flat_norm
    warps = 4
    rows = b * s
    if d == 2240 and x.dtype == torch.bfloat16:
        if rows >= 16384:
            kernel = _flat_stream_large
            warps = 8
        elif rows >= 8192:
            kernel = _flat_stream_medium
            warps = 2
        elif rows >= 1024:
            kernel = _flat_stream_medium
    kernel[rows,](
        x,
        scale,
        shift,
        y,
        s,
        d,
        eps,
        c,
        x.dtype == torch.float32,
        num_warps=warps,
        enable_fp_fusion=False,
    )
    return y


__all__ = ["layernorm_modulate"]
