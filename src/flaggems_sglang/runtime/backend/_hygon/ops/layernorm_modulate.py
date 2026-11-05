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
):
    row = tl.program_id(0)
    b = row // S
    s = row % S
    col = tl.arange(0, C)
    x = tl.load(X + b * X0 + s * X1 + col * X2, col < D, 0).to(tl.float32)
    mean = tl.sum(x, 0) / D
    z = tl.where(col < D, x - mean, 0.0)
    var = tl.sum(z * z, 0) / D
    a = tl.load(A + b * A0 + col * A1, col < D, 0).to(tl.float32)
    h = tl.load(H + b * H0 + col * H1, col < D, 0).to(tl.float32)
    y = z * tl.rsqrt(var + EPS)
    y = y * (1.0 + a) + h
    tl.store(Y + row * D + col, y, col < D)


@triton.jit
def _split(
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
    C0: tl.constexpr,
    C1: tl.constexpr,
):
    row = tl.program_id(0)
    batch = row // S
    seq = row % S
    c0 = tl.arange(0, C0)
    c1 = C0 + tl.arange(0, C1)
    x0 = tl.load(X + batch * X0 + seq * X1 + c0 * X2).to(tl.float32)
    x1 = tl.load(X + batch * X0 + seq * X1 + c1 * X2, c1 < D, 0).to(tl.float32)
    mean = (tl.sum(x0, 0) + tl.sum(x1, 0)) / D
    z0 = x0 - mean
    z1 = tl.where(c1 < D, x1 - mean, 0.0)
    var = (tl.sum(z0 * z0, 0) + tl.sum(z1 * z1, 0)) / D
    inv = tl.rsqrt(var + EPS)
    a0 = tl.load(A + batch * A0 + c0 * A1).to(tl.float32)
    a1 = tl.load(A + batch * A0 + c1 * A1, c1 < D, 0).to(tl.float32)
    h0 = tl.load(H + batch * H0 + c0 * H1).to(tl.float32)
    h1 = tl.load(H + batch * H0 + c1 * H1, c1 < D, 0).to(tl.float32)
    y0 = z0 * inv * (1.0 + a0) + h0
    y1 = z1 * inv * (1.0 + a1) + h1
    tl.store(Y + row * D + c0, y0)
    tl.store(Y + row * D + c1, y1, c1 < D)


def _modulate_lowp(x, scale, shift, eps):
    b, s, d = x.shape
    y = torch.empty((b, s, d), device=x.device, dtype=x.dtype)
    c = triton.next_power_of_2(d)
    args = (
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
    )
    if d != c and b * s >= 4096:
        c0 = c // 2
        c1 = triton.next_power_of_2(d - c0)
        _split[(b * s,)](*args, c0, c1, num_warps=1, enable_fp_fusion=False)
    else:
        _norm[(b * s,)](
            *args,
            c,
            num_warps=1 if b * s >= 1024 else 4,
            enable_fp_fusion=False,
        )
    return y


@triton.jit
def _centered_fp32(
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


def layernorm_modulate(x, scale, shift, eps):
    if x.dtype == torch.float32:
        b, s, d = x.shape
        y = torch.empty((b, s, d), device=x.device, dtype=x.dtype)
        c = triton.next_power_of_2(d)
        _centered_fp32[(b * s,)](
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
            True,
            num_warps=4,
            enable_fp_fusion=False,
        )
        return y
    return _modulate_lowp(x, scale, shift, eps)


__all__ = ["layernorm_modulate"]
