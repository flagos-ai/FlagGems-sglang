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
def _grouped_compact207(
    X,
    A,
    H,
    Y,
    S: tl.constexpr,
    D: tl.constexpr,
    EPS: tl.constexpr,
    C: tl.constexpr,
    R: tl.constexpr,
    BLOCKS: tl.constexpr,
    PROGRAMS: tl.constexpr,
    PERSISTENT: tl.constexpr,
):
    batch = tl.program_id(1)
    col = tl.arange(0, C)
    a = tl.load(A + batch * D + col * 1, col < D, 0).to(tl.float32)
    h = tl.load(H + batch * D + col * 1, col < D, 0).to(tl.float32)
    start = tl.program_id(0)
    end = BLOCKS if PERSISTENT else start + 1
    for block in range(start, end, PROGRAMS):
        row = block * R + tl.arange(0, R)
        mask = (row[:, None] < S) & (col[None, :] < D)
        x = tl.load(
            X + batch * (S * D) + row[:, None] * D + col[None, :] * 1, mask, 0
        ).to(tl.float32)
        mean = tl.sum(x, 1) / D
        z = tl.where(col[None, :] < D, x - mean[:, None], 0.0)
        var = tl.sum(z * z, 1) / D
        y = z * tl.rsqrt(var[:, None] + EPS)
        y = y * (1.0 + a[None, :]) + h[None, :]
        tl.store(Y + (batch * S + row[:, None]) * D + col[None, :], y, mask)


@triton.jit
def _split_grouped_compact207(
    X,
    A,
    H,
    Y,
    S: tl.constexpr,
    D: tl.constexpr,
    EPS: tl.constexpr,
    C0: tl.constexpr,
    C1: tl.constexpr,
    R: tl.constexpr,
    BLOCKS: tl.constexpr,
    PROGRAMS: tl.constexpr,
    PERSISTENT: tl.constexpr,
):
    batch = tl.program_id(1)
    c0 = tl.arange(0, C0)
    c1 = C0 + tl.arange(0, C1)
    a0 = tl.load(A + batch * D + c0 * 1).to(tl.float32)
    a1 = tl.load(A + batch * D + c1 * 1, c1 < D, 0).to(tl.float32)
    h0 = tl.load(H + batch * D + c0 * 1).to(tl.float32)
    h1 = tl.load(H + batch * D + c1 * 1, c1 < D, 0).to(tl.float32)
    start = tl.program_id(0)
    end = BLOCKS if PERSISTENT else start + 1
    for block in range(start, end, PROGRAMS):
        row = block * R + tl.arange(0, R)
        x0 = tl.load(
            X + batch * (S * D) + row[:, None] * D + c0[None, :] * 1,
            row[:, None] < S,
            0,
        ).to(tl.float32)
        mask1 = (row[:, None] < S) & (c1[None, :] < D)
        x1 = tl.load(
            X + batch * (S * D) + row[:, None] * D + c1[None, :] * 1, mask1, 0
        ).to(tl.float32)
        mean = (tl.sum(x0, 1) + tl.sum(x1, 1)) / D
        z0 = x0 - mean[:, None]
        z1 = tl.where(c1[None, :] < D, x1 - mean[:, None], 0.0)
        var = (tl.sum(z0 * z0, 1) + tl.sum(z1 * z1, 1)) / D
        inv = tl.rsqrt(var[:, None] + EPS)
        y0 = z0 * inv * (1.0 + a0[None, :]) + h0[None, :]
        y1 = z1 * inv * (1.0 + a1[None, :]) + h1[None, :]
        tl.store(
            Y + (batch * S + row[:, None]) * D + c0[None, :],
            y0,
            row[:, None] < S,
        )
        tl.store(Y + (batch * S + row[:, None]) * D + c1[None, :], y1, mask1)


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


def _launch_compact207(x, scale, shift, eps):
    x = x.contiguous()
    scale = scale.contiguous()
    shift = shift.contiguous()
    b, s, d = x.shape
    y = torch.empty((b, s, d), device=x.device, dtype=x.dtype)
    c = triton.next_power_of_2(d)
    split = d != c and c <= 4096
    r = 4 if split else min(2, max(1, 8192 // c))
    blocks = triton.cdiv(s, r)
    programs = min(blocks, max(1, 40 // b))
    grid = (programs, b, 1)
    common = (x, scale, shift, y, s, d, eps)
    if split:
        c0 = c // 2
        c1 = triton.next_power_of_2(d - c0)
        args = (*common, c0, c1, r, blocks, programs, True)
        kernel = _split_grouped_compact207
    else:
        args = (*common, c, r, blocks, programs, True)
        kernel = _grouped_compact207
    kernel.run(
        *args,
        grid=grid,
        warmup=False,
        num_warps=4,
        num_stages=1,
        enable_fp_fusion=False,
    )
    return y


def _general(x, scale, shift, eps):
    if x.dtype == torch.float32:
        b, s, d = x.shape
        y = torch.empty((b, s, d), device=x.device, dtype=x.dtype)
        c = triton.next_power_of_2(d)
        _centered_fp32[b * s,](
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
    return _launch_compact207(x, scale, shift, eps)


@triton.jit
def _split2240_214(
    X, A, H, Y, S: tl.constexpr, EPS: tl.constexpr, PROGRAMS: tl.constexpr
):
    batch = tl.program_id(1)
    c0 = tl.arange(0, 2048)
    c1 = 2048 + tl.arange(0, 256)
    a0 = tl.load(A + batch * 2240 + c0 * 1).to(tl.float32)
    a1 = tl.load(A + batch * 2240 + c1 * 1, c1 < 2240, 0).to(tl.float32)
    h0 = tl.load(H + batch * 2240 + c0 * 1).to(tl.float32)
    h1 = tl.load(H + batch * 2240 + c1 * 1, c1 < 2240, 0).to(tl.float32)
    start = tl.program_id(0)
    end = tl.cdiv(S, 4) if True else start + 1
    for block in range(start, end, PROGRAMS):
        row = block * 4 + tl.arange(0, 4)
        x0 = tl.load(
            X + batch * (S * 2240) + row[:, None] * 2240 + c0[None, :] * 1,
            row[:, None] < S,
            0,
        ).to(tl.float32)
        mask1 = (row[:, None] < S) & (c1[None, :] < 2240)
        x1 = tl.load(
            X + batch * (S * 2240) + row[:, None] * 2240 + c1[None, :] * 1,
            mask1,
            0,
        ).to(tl.float32)
        mean = (tl.sum(x0, 1) + tl.sum(x1, 1)) / 2240
        z0 = x0 - mean[:, None]
        z1 = tl.where(c1[None, :] < 2240, x1 - mean[:, None], 0.0)
        var = (tl.sum(z0 * z0, 1) + tl.sum(z1 * z1, 1)) / 2240
        inv = tl.rsqrt(var[:, None] + EPS)
        y0 = z0 * inv * (1.0 + a0[None, :]) + h0[None, :]
        y1 = z1 * inv * (1.0 + a1[None, :]) + h1[None, :]
        tl.store(
            Y + (batch * S + row[:, None]) * 2240 + c0[None, :],
            y0,
            row[:, None] < S,
        )
        tl.store(
            Y + (batch * S + row[:, None]) * 2240 + c1[None, :], y1, mask1
        )


@triton.jit
def _r8u2_356(X, A, H, Y, S: tl.constexpr, EPS: tl.constexpr, P: tl.constexpr):
    batch = tl.program_id(1)
    c0 = 0 + tl.arange(0, 2048)
    c1 = 2048 + tl.arange(0, 128)
    c2 = 2176 + tl.arange(0, 64)
    for block in tl.range(
        tl.program_id(0), tl.cdiv(S, 8), P, loop_unroll_factor=2
    ):
        rows = block * 8 + tl.arange(0, 8)
        base = batch * S * 2240 + rows[:, None] * 2240
        if S % 8 == 0:
            x0 = tl.load(X + base + c0[None, :]).to(tl.float32)
        else:
            x0 = tl.load(X + base + c0[None, :], rows[:, None] < S, 0).to(
                tl.float32
            )
        if S % 8 == 0:
            x1 = tl.load(X + base + c1[None, :]).to(tl.float32)
        else:
            x1 = tl.load(X + base + c1[None, :], rows[:, None] < S, 0).to(
                tl.float32
            )
        if S % 8 == 0:
            x2 = tl.load(X + base + c2[None, :]).to(tl.float32)
        else:
            x2 = tl.load(X + base + c2[None, :], rows[:, None] < S, 0).to(
                tl.float32
            )
        mean = (tl.sum(x0, 1) + tl.sum(x1, 1) + tl.sum(x2, 1)) * (1.0 / 2240)
        z0 = x0 - mean[:, None]
        z1 = x1 - mean[:, None]
        z2 = x2 - mean[:, None]
        var = (
            tl.sum(z0 * z0, 1) + tl.sum(z1 * z1, 1) + tl.sum(z2 * z2, 1)
        ) * (1.0 / 2240)
        inv = tl.rsqrt(var + EPS)
        a0 = tl.load(A + batch * 2240 + c0).to(tl.float32)
        h0 = tl.load(H + batch * 2240 + c0).to(tl.float32)
        y0 = z0 * inv[:, None] * (1.0 + a0[None, :]) + h0[None, :]
        if S % 8 == 0:
            tl.store(Y + base + c0[None, :], y0)
        else:
            tl.store(Y + base + c0[None, :], y0, rows[:, None] < S)
        a1 = tl.load(A + batch * 2240 + c1).to(tl.float32)
        h1 = tl.load(H + batch * 2240 + c1).to(tl.float32)
        y1 = z1 * inv[:, None] * (1.0 + a1[None, :]) + h1[None, :]
        if S % 8 == 0:
            tl.store(Y + base + c1[None, :], y1)
        else:
            tl.store(Y + base + c1[None, :], y1, rows[:, None] < S)
        a2 = tl.load(A + batch * 2240 + c2).to(tl.float32)
        h2 = tl.load(H + batch * 2240 + c2).to(tl.float32)
        y2 = z2 * inv[:, None] * (1.0 + a2[None, :]) + h2[None, :]
        if S % 8 == 0:
            tl.store(Y + base + c2[None, :], y2)
        else:
            tl.store(Y + base + c2[None, :], y2, rows[:, None] < S)


@triton.jit
def _tailu3_371(
    X, A, H, Y, S: tl.constexpr, EPS: tl.constexpr, P: tl.constexpr
):
    batch = tl.program_id(1)
    c0 = 0 + tl.arange(0, 2048)
    c1 = 2048 + tl.arange(0, 64)
    c2 = 2112 + tl.arange(0, 64)
    c3 = 2176 + tl.arange(0, 64)
    for block in tl.range(
        tl.program_id(0), tl.cdiv(S, 8), P, loop_unroll_factor=3
    ):
        rows = block * 8 + tl.arange(0, 8)
        base = batch * S * 2240 + rows[:, None] * 2240
        if S % 8 == 0:
            x0 = tl.load(X + base + c0[None, :]).to(tl.float32)
        else:
            x0 = tl.load(X + base + c0[None, :], rows[:, None] < S, 0).to(
                tl.float32
            )
        if S % 8 == 0:
            x1 = tl.load(X + base + c1[None, :]).to(tl.float32)
        else:
            x1 = tl.load(X + base + c1[None, :], rows[:, None] < S, 0).to(
                tl.float32
            )
        if S % 8 == 0:
            x2 = tl.load(X + base + c2[None, :]).to(tl.float32)
        else:
            x2 = tl.load(X + base + c2[None, :], rows[:, None] < S, 0).to(
                tl.float32
            )
        if S % 8 == 0:
            x3 = tl.load(X + base + c3[None, :]).to(tl.float32)
        else:
            x3 = tl.load(X + base + c3[None, :], rows[:, None] < S, 0).to(
                tl.float32
            )
        mean = (tl.sum(x0, 1) + tl.sum(x1 + x2 + x3, 1)) * (1.0 / 2240)
        z0 = x0 - mean[:, None]
        z1 = x1 - mean[:, None]
        z2 = x2 - mean[:, None]
        z3 = x3 - mean[:, None]
        var = (tl.sum(z0 * z0, 1) + tl.sum(z1 * z1 + z2 * z2 + z3 * z3, 1)) * (
            1.0 / 2240
        )
        inv = tl.rsqrt(var + EPS)
        a0 = tl.load(A + batch * 2240 + c0).to(tl.float32)
        h0 = tl.load(H + batch * 2240 + c0).to(tl.float32)
        y0 = z0 * inv[:, None] * (1 + a0[None, :]) + h0[None, :]
        if S % 8 == 0:
            tl.store(Y + base + c0[None, :], y0)
        else:
            tl.store(Y + base + c0[None, :], y0, rows[:, None] < S)
        a1 = tl.load(A + batch * 2240 + c1).to(tl.float32)
        h1 = tl.load(H + batch * 2240 + c1).to(tl.float32)
        y1 = z1 * inv[:, None] * (1 + a1[None, :]) + h1[None, :]
        if S % 8 == 0:
            tl.store(Y + base + c1[None, :], y1)
        else:
            tl.store(Y + base + c1[None, :], y1, rows[:, None] < S)
        a2 = tl.load(A + batch * 2240 + c2).to(tl.float32)
        h2 = tl.load(H + batch * 2240 + c2).to(tl.float32)
        y2 = z2 * inv[:, None] * (1 + a2[None, :]) + h2[None, :]
        if S % 8 == 0:
            tl.store(Y + base + c2[None, :], y2)
        else:
            tl.store(Y + base + c2[None, :], y2, rows[:, None] < S)
        a3 = tl.load(A + batch * 2240 + c3).to(tl.float32)
        h3 = tl.load(H + batch * 2240 + c3).to(tl.float32)
        y3 = z3 * inv[:, None] * (1 + a3[None, :]) + h3[None, :]
        if S % 8 == 0:
            tl.store(Y + base + c3[None, :], y3)
        else:
            tl.store(Y + base + c3[None, :], y3, rows[:, None] < S)


def layernorm_modulate(x, scale, shift, eps):
    if x.shape[-1] != 2240 or x.dtype == torch.float32:
        return _general(x, scale, shift, eps)
    if not x.is_contiguous():
        x = x.contiguous()
    if not scale.is_contiguous():
        scale = scale.contiguous()
    if not shift.is_contiguous():
        shift = shift.contiguous()
    b, s, d = x.shape
    y = torch.empty_like(x)
    if x.dtype == torch.bfloat16 and s >= 1024:
        programs = min(triton.cdiv(s, 8), max(1, 40 // b))
        (_tailu3_371 if b * s >= 16384 else _r8u2_356).run(
            x,
            scale,
            shift,
            y,
            s,
            eps,
            programs,
            num_warps=4,
            num_stages=1,
            enable_fp_fusion=False,
            grid=(programs, b),
            warmup=False,
        )
        return y
    programs = min(triton.cdiv(s, 4), max(1, 40 // b))
    _split2240_214.run(
        x,
        scale,
        shift,
        y,
        s,
        eps,
        programs,
        num_warps=4,
        num_stages=1,
        enable_fp_fusion=False,
        grid=(programs, b, 1),
        warmup=False,
    )
    return y


__all__ = ["layernorm_modulate"]
