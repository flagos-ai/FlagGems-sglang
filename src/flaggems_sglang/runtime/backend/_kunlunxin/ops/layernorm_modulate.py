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
def _grouped(
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
    R: tl.constexpr,
):
    batch = tl.program_id(1)
    col = tl.arange(0, C)
    a = tl.load(A + batch * A0 + col * A1, col < D, 0).to(tl.float32)
    h = tl.load(H + batch * H0 + col * H1, col < D, 0).to(tl.float32)
    block = tl.program_id(0)
    row = block * R + tl.arange(0, R)
    mask = (row[:, None] < S) & (col[None, :] < D)
    x = tl.load(
        X + batch * X0 + row[:, None] * X1 + col[None, :] * X2, mask, 0
    ).to(tl.float32)
    mean = tl.sum(x, 1) / D
    z = tl.where(col[None, :] < D, x - mean[:, None], 0.0)
    var = tl.sum(z * z, 1) / D
    y = z * tl.rsqrt(var[:, None] + EPS)
    y = y * (1.0 + a[None, :]) + h[None, :]
    tl.store(Y + (batch * S + row[:, None]) * D + col[None, :], y, mask)


@triton.jit
def _stats(
    X,
    T,
    S: tl.constexpr,
    D: tl.constexpr,
    N: tl.constexpr,
    X0: tl.constexpr,
    X1: tl.constexpr,
    X2: tl.constexpr,
    EPS: tl.constexpr,
    C: tl.constexpr,
    R: tl.constexpr,
):
    b = tl.program_id(1)
    row = tl.program_id(0) * R + tl.arange(0, R)
    col = tl.arange(0, C)
    mask = (row[:, None] < S) & (col[None, :] < D)
    x = tl.load(
        X + b * X0 + row[:, None] * X1 + col[None, :] * X2, mask, 0
    ).to(tl.float32)
    mean = tl.sum(x, 1) / D
    z = tl.where(col[None, :] < D, x - mean[:, None], 0.0)
    var = tl.sum(z * z, 1) / D
    inv = tl.rsqrt(var + EPS)
    tl.store(T + b * S + row, mean, row < S)
    tl.store(T + N + b * S + row, inv, row < S)


def _modulate_lowp(x, scale, shift, eps):
    x = x.contiguous()
    scale = scale.contiguous()
    shift = shift.contiguous()
    b, s, d = x.shape
    y = torch.empty((b, s, d), device=x.device, dtype=x.dtype)
    c = triton.next_power_of_2(d)
    if s < 256:
        _grouped[triton.cdiv(s, 4), b](
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
            4,
            num_warps=4,
            num_stages=1,
            enable_fp_fusion=False,
        )
    else:
        temp = torch.empty((2, b * s), device=x.device, dtype=torch.float32)
        rows = (
            32
            if b == 1 and s == 1024 and (d == 2240)
            else min(16, max(2, 65536 // c))
        )
        if (
            x.dtype == torch.bfloat16
            and d == 2240
            and (s in (1024, 4096))
            and (b in (1, 2, 8))
        ):
            reload_rows = (
                256
                if b == 2 and s == 4096
                else (
                    128
                    if b == 1 and s == 4096 or (b == 8 and s == 1024)
                    else 64
                )
            )
            _stats_narrow327[b * s // reload_rows,](
                x,
                temp,
                s,
                d,
                b * s,
                eps,
                reload_rows,
                1,
                True,
                8,
                b,
                num_warps=4,
                num_stages=1,
                enable_fp_fusion=False,
                buffer_size_limit=4096,
                isCloseMemoryAsync=False,
            )
        else:
            _stats_g8[triton.cdiv(s, rows), b](
                x,
                temp,
                s,
                d,
                b * s,
                *x.stride(),
                eps,
                c,
                rows,
                num_warps=4,
                num_stages=1,
                enable_fp_fusion=False,
                buffer_size_limit=min(c, 4096),
            )
        blocks = triton.cdiv(s * d, 4096)
        repeats = (
            64
            if b == 8
            and s == 4096
            and (d == 2240)
            and (x.dtype == torch.bfloat16)
            else (
                64
                if b == 1 and s % 4096 == 0
                else (
                    32
                    if s % 4096 == 0
                    else 16 if s % 1024 == 0 else 8 if s % 512 == 0 else 4
                )
            )
        )
        if (
            s * d % 4096 == 0
            and blocks % repeats == 0
            and (blocks // repeats * 4096 % d == 0)
        ):
            if (
                b == 1
                and s == 1024
                and (d == 2240)
                and (x.dtype == torch.bfloat16)
            ):
                _apply_order8[blocks // repeats, b](
                    x,
                    scale,
                    shift,
                    temp,
                    y,
                    s,
                    d,
                    b * s,
                    4096,
                    repeats,
                    7,
                    num_warps=4,
                    num_stages=1,
                    enable_fp_fusion=False,
                    isCloseMemoryAsync=False,
                )
            else:
                apply = (
                    (_prefetch332 if b == 1 else _prefetch233)
                    if b in (1, 8)
                    and s == 4096
                    and (d == 2240)
                    and (x.dtype == torch.bfloat16)
                    else (
                        (
                            _apply_unroll8_affine343
                            if (b, s) == (2, 1024)
                            else _apply_unroll8
                        )
                        if b in (1, 2, 8)
                        and s in (1024, 4096)
                        and (d == 2240)
                        and (x.dtype == torch.bfloat16)
                        else _apply_reuse
                    )
                )
                apply[blocks // repeats, b](
                    x,
                    scale,
                    shift,
                    temp,
                    y,
                    s,
                    d,
                    b * s,
                    4096,
                    repeats,
                    num_warps=4,
                    num_stages=1,
                    enable_fp_fusion=False,
                    isCloseMemoryAsync=not (
                        b in (1, 2, 8)
                        and s in (1024, 4096)
                        and (d == 2240)
                        and (x.dtype == torch.bfloat16)
                    ),
                )
        else:
            _apply_contiguous[blocks, b](
                x,
                scale,
                shift,
                temp,
                y,
                s,
                d,
                b * s,
                4096,
                num_warps=4,
                num_stages=1,
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
    if (
        x.dtype == torch.bfloat16
        and x.shape[-1] == 2240
        and (0 < x.shape[1] < 256)
        and (0 < x.shape[0] <= 65535)
    ):
        b, s, d = x.shape
        out = torch.empty((b, s, d), dtype=x.dtype, device=x.device)
        groups = min(s, 12 if b == 1 else 6, 65535 // b)
        _reuse239[b * groups,](
            x,
            scale,
            shift,
            out,
            s,
            d,
            groups,
            *x.stride(),
            *scale.stride(),
            *shift.stride(),
            eps,
            max(64, triton.next_power_of_2(d)),
            num_warps=1,
            enable_fp_fusion=False,
        )
        return out
    if x.dtype == torch.float32 or x.shape[-1] != 2240:
        x = x.contiguous()
        scale = scale.contiguous()
        shift = shift.contiguous()
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
            x.dtype == torch.float32,
            num_warps=4,
            enable_fp_fusion=False,
        )
        return y
    return _modulate_lowp(x, scale, shift, eps)


@triton.jit
def _apply_contiguous(
    X,
    A,
    H,
    T,
    Y,
    S: tl.constexpr,
    D: tl.constexpr,
    N: tl.constexpr,
    C: tl.constexpr,
):
    batch = tl.program_id(1)
    block = tl.program_id(0)
    blocks = tl.cdiv(S * D, C)
    if blocks % 16 == 0:
        block = block % 16 * (blocks // 16) + block // 16
    index = block * C + tl.arange(0, C)
    row = index // D
    col = index % D
    if S * D % C == 0:
        mask = True
    else:
        mask = index < S * D
    x = tl.load(X + batch * S * D + index, mask, 0).to(tl.float32)
    mean = tl.load(T + batch * S + row, mask, 0)
    inv = tl.load(T + N + batch * S + row, mask, 0)
    a = tl.load(A + batch * D + col, mask, 0).to(tl.float32)
    h = tl.load(H + batch * D + col, mask, 0).to(tl.float32)
    y = (x - mean) * inv * (1.0 + a) + h
    tl.store(Y + batch * S * D + index, y, mask)


@triton.jit
def _stats_g8(
    X,
    T,
    S: tl.constexpr,
    D: tl.constexpr,
    N: tl.constexpr,
    X0: tl.constexpr,
    X1: tl.constexpr,
    X2: tl.constexpr,
    EPS: tl.constexpr,
    C: tl.constexpr,
    R: tl.constexpr,
):
    b = tl.program_id(1)
    block = tl.program_id(0)
    blocks = tl.cdiv(S, R)
    if blocks % 8 == 0:
        block = block % 8 * (blocks // 8) + block // 8
    row = block * R + tl.arange(0, R)
    col = tl.arange(0, C)
    mask = (row[:, None] < S) & (col[None, :] < D)
    x = tl.load(
        X + b * X0 + row[:, None] * X1 + col[None, :] * X2, mask, 0
    ).to(tl.float32)
    mean = tl.sum(x, 1) / D
    z = tl.where(col[None, :] < D, x - mean[:, None], 0.0)
    var = tl.sum(z * z, 1) / D
    inv = tl.rsqrt(var + EPS)
    tl.store(T + b * S + row, mean, row < S)
    tl.store(T + N + b * S + row, inv, row < S)


@triton.jit
def _apply_reuse(
    X,
    A,
    H,
    T,
    Y,
    S: tl.constexpr,
    D: tl.constexpr,
    N: tl.constexpr,
    C: tl.constexpr,
    REPEATS: tl.constexpr,
):
    batch = tl.program_id(1)
    blocks = S * D // C
    programs = blocks // REPEATS
    block = tl.program_id(0)
    if programs % 16 == 0:
        block = block % 16 * (programs // 16) + block // 16
    base = block * C + tl.arange(0, C)
    col = base % D
    base_row = base // D
    a = tl.load(A + batch * D + col).to(tl.float32)
    h = tl.load(H + batch * D + col).to(tl.float32)
    factor = 1.0 + a
    for step in range(REPEATS):
        index = base + step * programs * C
        row = base_row + step * (programs * C // D)
        x = tl.load(X + batch * S * D + index).to(tl.float32)
        mean = tl.load(T + batch * S + row)
        inv = tl.load(T + N + batch * S + row)
        y = (x - mean) * inv * factor + h
        tl.store(Y + batch * S * D + index, y)


@triton.jit
def _apply_unroll2(
    X,
    A,
    H,
    T,
    Y,
    S: tl.constexpr,
    D: tl.constexpr,
    N: tl.constexpr,
    C: tl.constexpr,
    REPEATS: tl.constexpr,
):
    batch = tl.program_id(1)
    blocks = S * D // C
    programs = blocks // REPEATS
    block = tl.program_id(0)
    if programs % 16 == 0:
        block = block % 16 * (programs // 16) + block // 16
    base = block * C + tl.arange(0, C)
    col = base % D
    base_row = base // D
    a = tl.load(A + batch * D + col).to(tl.float32)
    h = tl.load(H + batch * D + col).to(tl.float32)
    factor = 1.0 + a
    for pair in range(REPEATS // 2):
        index0 = base + 2 * pair * programs * C
        row0 = base_row + 2 * pair * (programs * C // D)
        index1 = base + (2 * pair + 1) * programs * C
        row1 = base_row + (2 * pair + 1) * (programs * C // D)
        x0 = tl.load(X + batch * S * D + index0).to(tl.float32)
        mean0 = tl.load(T + batch * S + row0)
        inv0 = tl.load(T + N + batch * S + row0)
        y0 = (x0 - mean0) * inv0 * factor + h
        tl.store(Y + batch * S * D + index0, y0)
        x1 = tl.load(X + batch * S * D + index1).to(tl.float32)
        mean1 = tl.load(T + batch * S + row1)
        inv1 = tl.load(T + N + batch * S + row1)
        y1 = (x1 - mean1) * inv1 * factor + h
        tl.store(Y + batch * S * D + index1, y1)


@triton.jit
def _apply_order2(
    X,
    A,
    H,
    T,
    Y,
    S: tl.constexpr,
    D: tl.constexpr,
    N: tl.constexpr,
    C: tl.constexpr,
    REPEATS: tl.constexpr,
    G: tl.constexpr,
):
    batch = tl.program_id(1)
    blocks = S * D // C
    programs = blocks // REPEATS
    block = tl.program_id(0)
    block = block % G * (programs // G) + block // G
    base = block * C + tl.arange(0, C)
    col = base % D
    base_row = base // D
    a = tl.load(A + batch * D + col).to(tl.float32)
    h = tl.load(H + batch * D + col).to(tl.float32)
    factor = 1.0 + a
    for pair in range(REPEATS // 2):
        index0 = base + 2 * pair * programs * C
        row0 = base_row + 2 * pair * (programs * C // D)
        index1 = base + (2 * pair + 1) * programs * C
        row1 = base_row + (2 * pair + 1) * (programs * C // D)
        x0 = tl.load(X + batch * S * D + index0).to(tl.float32)
        mean0 = tl.load(T + batch * S + row0)
        inv0 = tl.load(T + N + batch * S + row0)
        y0 = (x0 - mean0) * inv0 * factor + h
        tl.store(Y + batch * S * D + index0, y0)
        x1 = tl.load(X + batch * S * D + index1).to(tl.float32)
        mean1 = tl.load(T + batch * S + row1)
        inv1 = tl.load(T + N + batch * S + row1)
        y1 = (x1 - mean1) * inv1 * factor + h
        tl.store(Y + batch * S * D + index1, y1)


@triton.jit
def _apply_unroll8(
    X,
    A,
    H,
    T,
    Y,
    S: tl.constexpr,
    D: tl.constexpr,
    N: tl.constexpr,
    C: tl.constexpr,
    REPEATS: tl.constexpr,
):
    batch = tl.program_id(1)
    blocks = S * D // C
    programs = blocks // REPEATS
    block = tl.program_id(0)
    if programs % 16 == 0:
        block = block % 16 * (programs // 16) + block // 16
    base = block * C + tl.arange(0, C)
    col = base % D
    base_row = base // D
    a = tl.load(A + batch * D + col).to(tl.float32)
    h = tl.load(H + batch * D + col).to(tl.float32)
    factor = 1.0 + a
    for pair in range(REPEATS // 8):
        index0 = base + (8 * pair + 0) * programs * C
        row0 = base_row + (8 * pair + 0) * (programs * C // D)
        index1 = base + (8 * pair + 1) * programs * C
        row1 = base_row + (8 * pair + 1) * (programs * C // D)
        index2 = base + (8 * pair + 2) * programs * C
        row2 = base_row + (8 * pair + 2) * (programs * C // D)
        index3 = base + (8 * pair + 3) * programs * C
        row3 = base_row + (8 * pair + 3) * (programs * C // D)
        index4 = base + (8 * pair + 4) * programs * C
        row4 = base_row + (8 * pair + 4) * (programs * C // D)
        index5 = base + (8 * pair + 5) * programs * C
        row5 = base_row + (8 * pair + 5) * (programs * C // D)
        index6 = base + (8 * pair + 6) * programs * C
        row6 = base_row + (8 * pair + 6) * (programs * C // D)
        index7 = base + (8 * pair + 7) * programs * C
        row7 = base_row + (8 * pair + 7) * (programs * C // D)
        x0 = tl.load(X + batch * S * D + index0).to(tl.float32)
        mean0 = tl.load(T + batch * S + row0)
        inv0 = tl.load(T + N + batch * S + row0)
        y0 = (x0 - mean0) * inv0 * factor + h
        tl.store(Y + batch * S * D + index0, y0)
        x1 = tl.load(X + batch * S * D + index1).to(tl.float32)
        mean1 = tl.load(T + batch * S + row1)
        inv1 = tl.load(T + N + batch * S + row1)
        y1 = (x1 - mean1) * inv1 * factor + h
        tl.store(Y + batch * S * D + index1, y1)
        x2 = tl.load(X + batch * S * D + index2).to(tl.float32)
        mean2 = tl.load(T + batch * S + row2)
        inv2 = tl.load(T + N + batch * S + row2)
        y2 = (x2 - mean2) * inv2 * factor + h
        tl.store(Y + batch * S * D + index2, y2)
        x3 = tl.load(X + batch * S * D + index3).to(tl.float32)
        mean3 = tl.load(T + batch * S + row3)
        inv3 = tl.load(T + N + batch * S + row3)
        y3 = (x3 - mean3) * inv3 * factor + h
        tl.store(Y + batch * S * D + index3, y3)
        x4 = tl.load(X + batch * S * D + index4).to(tl.float32)
        mean4 = tl.load(T + batch * S + row4)
        inv4 = tl.load(T + N + batch * S + row4)
        y4 = (x4 - mean4) * inv4 * factor + h
        tl.store(Y + batch * S * D + index4, y4)
        x5 = tl.load(X + batch * S * D + index5).to(tl.float32)
        mean5 = tl.load(T + batch * S + row5)
        inv5 = tl.load(T + N + batch * S + row5)
        y5 = (x5 - mean5) * inv5 * factor + h
        tl.store(Y + batch * S * D + index5, y5)
        x6 = tl.load(X + batch * S * D + index6).to(tl.float32)
        mean6 = tl.load(T + batch * S + row6)
        inv6 = tl.load(T + N + batch * S + row6)
        y6 = (x6 - mean6) * inv6 * factor + h
        tl.store(Y + batch * S * D + index6, y6)
        x7 = tl.load(X + batch * S * D + index7).to(tl.float32)
        mean7 = tl.load(T + batch * S + row7)
        inv7 = tl.load(T + N + batch * S + row7)
        y7 = (x7 - mean7) * inv7 * factor + h
        tl.store(Y + batch * S * D + index7, y7)


@triton.jit
def _apply_order8(
    X,
    A,
    H,
    T,
    Y,
    S: tl.constexpr,
    D: tl.constexpr,
    N: tl.constexpr,
    C: tl.constexpr,
    REPEATS: tl.constexpr,
    G: tl.constexpr,
):
    batch = tl.program_id(1)
    blocks = S * D // C
    programs = blocks // REPEATS
    block = tl.program_id(0)
    block = block % G * (programs // G) + block // G
    base = block * C + tl.arange(0, C)
    col = base % D
    base_row = base // D
    a = tl.load(A + batch * D + col).to(tl.float32)
    h = tl.load(H + batch * D + col).to(tl.float32)
    factor = 1.0 + a
    for pair in range(REPEATS // 8):
        index0 = base + (8 * pair + 0) * programs * C
        row0 = base_row + (8 * pair + 0) * (programs * C // D)
        index1 = base + (8 * pair + 1) * programs * C
        row1 = base_row + (8 * pair + 1) * (programs * C // D)
        index2 = base + (8 * pair + 2) * programs * C
        row2 = base_row + (8 * pair + 2) * (programs * C // D)
        index3 = base + (8 * pair + 3) * programs * C
        row3 = base_row + (8 * pair + 3) * (programs * C // D)
        index4 = base + (8 * pair + 4) * programs * C
        row4 = base_row + (8 * pair + 4) * (programs * C // D)
        index5 = base + (8 * pair + 5) * programs * C
        row5 = base_row + (8 * pair + 5) * (programs * C // D)
        index6 = base + (8 * pair + 6) * programs * C
        row6 = base_row + (8 * pair + 6) * (programs * C // D)
        index7 = base + (8 * pair + 7) * programs * C
        row7 = base_row + (8 * pair + 7) * (programs * C // D)
        x0 = tl.load(X + batch * S * D + index0).to(tl.float32)
        mean0 = tl.load(T + batch * S + row0)
        inv0 = tl.load(T + N + batch * S + row0)
        y0 = (x0 - mean0) * inv0 * factor + h
        tl.store(Y + batch * S * D + index0, y0)
        x1 = tl.load(X + batch * S * D + index1).to(tl.float32)
        mean1 = tl.load(T + batch * S + row1)
        inv1 = tl.load(T + N + batch * S + row1)
        y1 = (x1 - mean1) * inv1 * factor + h
        tl.store(Y + batch * S * D + index1, y1)
        x2 = tl.load(X + batch * S * D + index2).to(tl.float32)
        mean2 = tl.load(T + batch * S + row2)
        inv2 = tl.load(T + N + batch * S + row2)
        y2 = (x2 - mean2) * inv2 * factor + h
        tl.store(Y + batch * S * D + index2, y2)
        x3 = tl.load(X + batch * S * D + index3).to(tl.float32)
        mean3 = tl.load(T + batch * S + row3)
        inv3 = tl.load(T + N + batch * S + row3)
        y3 = (x3 - mean3) * inv3 * factor + h
        tl.store(Y + batch * S * D + index3, y3)
        x4 = tl.load(X + batch * S * D + index4).to(tl.float32)
        mean4 = tl.load(T + batch * S + row4)
        inv4 = tl.load(T + N + batch * S + row4)
        y4 = (x4 - mean4) * inv4 * factor + h
        tl.store(Y + batch * S * D + index4, y4)
        x5 = tl.load(X + batch * S * D + index5).to(tl.float32)
        mean5 = tl.load(T + batch * S + row5)
        inv5 = tl.load(T + N + batch * S + row5)
        y5 = (x5 - mean5) * inv5 * factor + h
        tl.store(Y + batch * S * D + index5, y5)
        x6 = tl.load(X + batch * S * D + index6).to(tl.float32)
        mean6 = tl.load(T + batch * S + row6)
        inv6 = tl.load(T + N + batch * S + row6)
        y6 = (x6 - mean6) * inv6 * factor + h
        tl.store(Y + batch * S * D + index6, y6)
        x7 = tl.load(X + batch * S * D + index7).to(tl.float32)
        mean7 = tl.load(T + batch * S + row7)
        inv7 = tl.load(T + N + batch * S + row7)
        y7 = (x7 - mean7) * inv7 * factor + h
        tl.store(Y + batch * S * D + index7, y7)


@triton.jit
def _prefetch2(
    X,
    A,
    H,
    T,
    Y,
    S: tl.constexpr,
    D: tl.constexpr,
    N: tl.constexpr,
    C: tl.constexpr,
    REPEATS: tl.constexpr,
):
    batch = tl.program_id(1)
    blocks = S * D // C
    programs = blocks // REPEATS
    block = tl.program_id(0)
    if programs % 16 == 0:
        block = block % 16 * (programs // 16) + block // 16
    base = block * C + tl.arange(0, C)
    col = base % D
    base_row = base // D
    a = tl.load(A + batch * D + col).to(tl.float32)
    h = tl.load(H + batch * D + col).to(tl.float32)
    factor = 1.0 + a
    for pair in range(REPEATS // 8):
        index0 = base + (8 * pair + 0) * programs * C
        row0 = base_row + (8 * pair + 0) * (programs * C // D)
        index1 = base + (8 * pair + 1) * programs * C
        row1 = base_row + (8 * pair + 1) * (programs * C // D)
        index2 = base + (8 * pair + 2) * programs * C
        row2 = base_row + (8 * pair + 2) * (programs * C // D)
        index3 = base + (8 * pair + 3) * programs * C
        row3 = base_row + (8 * pair + 3) * (programs * C // D)
        index4 = base + (8 * pair + 4) * programs * C
        row4 = base_row + (8 * pair + 4) * (programs * C // D)
        index5 = base + (8 * pair + 5) * programs * C
        row5 = base_row + (8 * pair + 5) * (programs * C // D)
        index6 = base + (8 * pair + 6) * programs * C
        row6 = base_row + (8 * pair + 6) * (programs * C // D)
        index7 = base + (8 * pair + 7) * programs * C
        row7 = base_row + (8 * pair + 7) * (programs * C // D)
        x0 = tl.load(X + batch * S * D + index0).to(tl.float32)
        mean0 = tl.load(T + batch * S + row0)
        inv0 = tl.load(T + N + batch * S + row0)
        x1 = tl.load(X + batch * S * D + index1).to(tl.float32)
        mean1 = tl.load(T + batch * S + row1)
        inv1 = tl.load(T + N + batch * S + row1)
        y0 = (x0 - mean0) * inv0 * factor + h
        tl.store(Y + batch * S * D + index0, y0)
        y1 = (x1 - mean1) * inv1 * factor + h
        tl.store(Y + batch * S * D + index1, y1)
        x2 = tl.load(X + batch * S * D + index2).to(tl.float32)
        mean2 = tl.load(T + batch * S + row2)
        inv2 = tl.load(T + N + batch * S + row2)
        x3 = tl.load(X + batch * S * D + index3).to(tl.float32)
        mean3 = tl.load(T + batch * S + row3)
        inv3 = tl.load(T + N + batch * S + row3)
        y2 = (x2 - mean2) * inv2 * factor + h
        tl.store(Y + batch * S * D + index2, y2)
        y3 = (x3 - mean3) * inv3 * factor + h
        tl.store(Y + batch * S * D + index3, y3)
        x4 = tl.load(X + batch * S * D + index4).to(tl.float32)
        mean4 = tl.load(T + batch * S + row4)
        inv4 = tl.load(T + N + batch * S + row4)
        x5 = tl.load(X + batch * S * D + index5).to(tl.float32)
        mean5 = tl.load(T + batch * S + row5)
        inv5 = tl.load(T + N + batch * S + row5)
        y4 = (x4 - mean4) * inv4 * factor + h
        tl.store(Y + batch * S * D + index4, y4)
        y5 = (x5 - mean5) * inv5 * factor + h
        tl.store(Y + batch * S * D + index5, y5)
        x6 = tl.load(X + batch * S * D + index6).to(tl.float32)
        mean6 = tl.load(T + batch * S + row6)
        inv6 = tl.load(T + N + batch * S + row6)
        x7 = tl.load(X + batch * S * D + index7).to(tl.float32)
        mean7 = tl.load(T + batch * S + row7)
        inv7 = tl.load(T + N + batch * S + row7)
        y6 = (x6 - mean6) * inv6 * factor + h
        tl.store(Y + batch * S * D + index6, y6)
        y7 = (x7 - mean7) * inv7 * factor + h
        tl.store(Y + batch * S * D + index7, y7)


@triton.jit
def _prefetch233(
    X,
    A,
    H,
    T,
    Y,
    S: tl.constexpr,
    D: tl.constexpr,
    N: tl.constexpr,
    C: tl.constexpr,
    REPEATS: tl.constexpr,
):
    batch = tl.program_id(1)
    blocks = S * D // C
    programs = blocks // REPEATS
    block = tl.program_id(0)
    if programs % 16 == 0:
        block = block % 16 * (programs // 16) + block // 16
    base = block * C + tl.arange(0, C)
    col = base % D
    base_row = base // D
    a = tl.load(A + batch * D + col).to(tl.float32)
    h = tl.load(H + batch * D + col).to(tl.float32)
    factor = 1.0 + a
    for pair in range(REPEATS // 8):
        index0 = base + (8 * pair + 0) * programs * C
        row0 = base_row + (8 * pair + 0) * (programs * C // D)
        index1 = base + (8 * pair + 1) * programs * C
        row1 = base_row + (8 * pair + 1) * (programs * C // D)
        index2 = base + (8 * pair + 2) * programs * C
        row2 = base_row + (8 * pair + 2) * (programs * C // D)
        index3 = base + (8 * pair + 3) * programs * C
        row3 = base_row + (8 * pair + 3) * (programs * C // D)
        index4 = base + (8 * pair + 4) * programs * C
        row4 = base_row + (8 * pair + 4) * (programs * C // D)
        index5 = base + (8 * pair + 5) * programs * C
        row5 = base_row + (8 * pair + 5) * (programs * C // D)
        index6 = base + (8 * pair + 6) * programs * C
        row6 = base_row + (8 * pair + 6) * (programs * C // D)
        index7 = base + (8 * pair + 7) * programs * C
        row7 = base_row + (8 * pair + 7) * (programs * C // D)
        x0 = tl.load(X + batch * S * D + index0).to(tl.float32)
        mean0 = tl.load(T + batch * S + row0)
        inv0 = tl.load(T + N + batch * S + row0)
        x1 = tl.load(X + batch * S * D + index1).to(tl.float32)
        mean1 = tl.load(T + batch * S + row1)
        inv1 = tl.load(T + N + batch * S + row1)
        y0 = (x0 - mean0) * inv0 * factor + h
        tl.store(Y + batch * S * D + index0, y0)
        y1 = (x1 - mean1) * inv1 * factor + h
        tl.store(Y + batch * S * D + index1, y1)
        x2 = tl.load(X + batch * S * D + index2).to(tl.float32)
        mean2 = tl.load(T + batch * S + row2)
        inv2 = tl.load(T + N + batch * S + row2)
        x3 = tl.load(X + batch * S * D + index3).to(tl.float32)
        mean3 = tl.load(T + batch * S + row3)
        inv3 = tl.load(T + N + batch * S + row3)
        x4 = tl.load(X + batch * S * D + index4).to(tl.float32)
        mean4 = tl.load(T + batch * S + row4)
        inv4 = tl.load(T + N + batch * S + row4)
        y2 = (x2 - mean2) * inv2 * factor + h
        tl.store(Y + batch * S * D + index2, y2)
        y3 = (x3 - mean3) * inv3 * factor + h
        tl.store(Y + batch * S * D + index3, y3)
        y4 = (x4 - mean4) * inv4 * factor + h
        tl.store(Y + batch * S * D + index4, y4)
        x5 = tl.load(X + batch * S * D + index5).to(tl.float32)
        mean5 = tl.load(T + batch * S + row5)
        inv5 = tl.load(T + N + batch * S + row5)
        x6 = tl.load(X + batch * S * D + index6).to(tl.float32)
        mean6 = tl.load(T + batch * S + row6)
        inv6 = tl.load(T + N + batch * S + row6)
        x7 = tl.load(X + batch * S * D + index7).to(tl.float32)
        mean7 = tl.load(T + batch * S + row7)
        inv7 = tl.load(T + N + batch * S + row7)
        y5 = (x5 - mean5) * inv5 * factor + h
        tl.store(Y + batch * S * D + index5, y5)
        y6 = (x6 - mean6) * inv6 * factor + h
        tl.store(Y + batch * S * D + index6, y6)
        y7 = (x7 - mean7) * inv7 * factor + h
        tl.store(Y + batch * S * D + index7, y7)


@triton.jit
def _prefetch332(
    X,
    A,
    H,
    T,
    Y,
    S: tl.constexpr,
    D: tl.constexpr,
    N: tl.constexpr,
    C: tl.constexpr,
    REPEATS: tl.constexpr,
):
    batch = tl.program_id(1)
    blocks = S * D // C
    programs = blocks // REPEATS
    block = tl.program_id(0)
    if programs % 16 == 0:
        block = block % 16 * (programs // 16) + block // 16
    base = block * C + tl.arange(0, C)
    col = base % D
    base_row = base // D
    a = tl.load(A + batch * D + col).to(tl.float32)
    h = tl.load(H + batch * D + col).to(tl.float32)
    factor = 1.0 + a
    for pair in range(REPEATS // 8):
        index0 = base + (8 * pair + 0) * programs * C
        row0 = base_row + (8 * pair + 0) * (programs * C // D)
        index1 = base + (8 * pair + 1) * programs * C
        row1 = base_row + (8 * pair + 1) * (programs * C // D)
        index2 = base + (8 * pair + 2) * programs * C
        row2 = base_row + (8 * pair + 2) * (programs * C // D)
        index3 = base + (8 * pair + 3) * programs * C
        row3 = base_row + (8 * pair + 3) * (programs * C // D)
        index4 = base + (8 * pair + 4) * programs * C
        row4 = base_row + (8 * pair + 4) * (programs * C // D)
        index5 = base + (8 * pair + 5) * programs * C
        row5 = base_row + (8 * pair + 5) * (programs * C // D)
        index6 = base + (8 * pair + 6) * programs * C
        row6 = base_row + (8 * pair + 6) * (programs * C // D)
        index7 = base + (8 * pair + 7) * programs * C
        row7 = base_row + (8 * pair + 7) * (programs * C // D)
        x0 = tl.load(X + batch * S * D + index0).to(tl.float32)
        mean0 = tl.load(T + batch * S + row0)
        inv0 = tl.load(T + N + batch * S + row0)
        x1 = tl.load(X + batch * S * D + index1).to(tl.float32)
        mean1 = tl.load(T + batch * S + row1)
        inv1 = tl.load(T + N + batch * S + row1)
        x2 = tl.load(X + batch * S * D + index2).to(tl.float32)
        mean2 = tl.load(T + batch * S + row2)
        inv2 = tl.load(T + N + batch * S + row2)
        y0 = (x0 - mean0) * inv0 * factor + h
        tl.store(Y + batch * S * D + index0, y0)
        y1 = (x1 - mean1) * inv1 * factor + h
        tl.store(Y + batch * S * D + index1, y1)
        y2 = (x2 - mean2) * inv2 * factor + h
        tl.store(Y + batch * S * D + index2, y2)
        x3 = tl.load(X + batch * S * D + index3).to(tl.float32)
        mean3 = tl.load(T + batch * S + row3)
        inv3 = tl.load(T + N + batch * S + row3)
        x4 = tl.load(X + batch * S * D + index4).to(tl.float32)
        mean4 = tl.load(T + batch * S + row4)
        inv4 = tl.load(T + N + batch * S + row4)
        x5 = tl.load(X + batch * S * D + index5).to(tl.float32)
        mean5 = tl.load(T + batch * S + row5)
        inv5 = tl.load(T + N + batch * S + row5)
        y3 = (x3 - mean3) * inv3 * factor + h
        tl.store(Y + batch * S * D + index3, y3)
        y4 = (x4 - mean4) * inv4 * factor + h
        tl.store(Y + batch * S * D + index4, y4)
        y5 = (x5 - mean5) * inv5 * factor + h
        tl.store(Y + batch * S * D + index5, y5)
        x6 = tl.load(X + batch * S * D + index6).to(tl.float32)
        mean6 = tl.load(T + batch * S + row6)
        inv6 = tl.load(T + N + batch * S + row6)
        x7 = tl.load(X + batch * S * D + index7).to(tl.float32)
        mean7 = tl.load(T + batch * S + row7)
        inv7 = tl.load(T + N + batch * S + row7)
        y6 = (x6 - mean6) * inv6 * factor + h
        tl.store(Y + batch * S * D + index6, y6)
        y7 = (x7 - mean7) * inv7 * factor + h
        tl.store(Y + batch * S * D + index7, y7)


@triton.jit
def _reuse239(
    X,
    A,
    T,
    Y,
    S: tl.constexpr,
    D: tl.constexpr,
    GROUPS: tl.constexpr,
    X0: tl.constexpr,
    X1: tl.constexpr,
    X2: tl.constexpr,
    A0: tl.constexpr,
    A1: tl.constexpr,
    T0: tl.constexpr,
    T1: tl.constexpr,
    EPS: tl.constexpr,
    BD: tl.constexpr,
):
    batch = tl.program_id(0) // GROUPS
    start = tl.program_id(0) % GROUPS
    col = tl.arange(0, BD)
    scale = tl.load(A + batch * A0 + col * A1, col < D, 0).to(tl.float32)
    shift = tl.load(T + batch * T0 + col * T1, col < D, 0).to(tl.float32)
    gain = 1.0 + scale
    for seq in range(start, S, GROUPS):
        x = tl.load(X + batch * X0 + seq * X1 + col * X2, col < D, 0).to(
            tl.float32
        )
        mean = tl.sum(x, 0) / D
        centered = tl.where(col < D, x - mean, 0.0)
        var = tl.sum(centered * centered, 0) / D
        normalized = centered * tl.rsqrt(var + EPS)
        out = normalized * gain + shift
        tl.store(Y + (batch * S + seq) * D + col, out, col < D)


@triton.jit
def _stats_overlap259(
    X,
    T,
    S: tl.constexpr,
    D: tl.constexpr,
    N: tl.constexpr,
    EPS: tl.constexpr,
    R: tl.constexpr,
):
    batch = tl.program_id(1)
    block = tl.program_id(0)
    blocks = S // R
    if blocks % 8 == 0:
        block = block % 8 * (blocks // 8) + block // 8
    row = block * R + tl.arange(0, R)
    col = tl.arange(0, 1024)
    base = X + (batch * S + row[:, None]) * D + col[None, :]
    x0 = tl.load(base).to(tl.float32)
    x1 = tl.load(base + 1024).to(tl.float32)
    x2 = tl.load(base + D - 1024).to(tl.float32)
    tail = col[None, :] >= 3072 - D
    part2 = tl.where(tail, x2, 0.0)
    mean = tl.sum(x0 + x1 + part2, 1) / D
    z0 = x0 - mean[:, None]
    z1 = x1 - mean[:, None]
    z2 = tl.where(tail, x2 - mean[:, None], 0.0)
    var = tl.sum(z0 * z0 + z1 * z1 + z2 * z2, 1) / D
    inv = tl.rsqrt(var + EPS)
    tl.store(T + batch * S + row, mean)
    tl.store(T + N + batch * S + row, inv)


@triton.jit
def _stats_separate273(
    X,
    T,
    S: tl.constexpr,
    D: tl.constexpr,
    N: tl.constexpr,
    EPS: tl.constexpr,
    R: tl.constexpr,
    MODE: tl.constexpr,
    FLAT: tl.constexpr,
    G: tl.constexpr,
    B: tl.constexpr,
):
    blocks = S // R
    if FLAT:
        batch = tl.program_id(0) // blocks
        block = tl.program_id(0) % blocks
    else:
        batch = tl.program_id(1)
        block = tl.program_id(0)
    if blocks % G == 0:
        block = block % G * (blocks // G) + block // G
    row = block * R + tl.arange(0, R)
    col = tl.arange(0, 1024)
    base = X + (batch * S + row[:, None]) * D + col[None, :]
    x0 = tl.load(base).to(tl.float32)
    x1 = tl.load(base + 1024).to(tl.float32)
    x2 = tl.load(base + D - 1024).to(tl.float32)
    tail = col[None, :] >= 3072 - D
    part2 = tl.where(tail, x2, 0.0)
    if MODE == 1:
        mean = tl.sum(x0 + x1 + part2, 1) / D
    else:
        mean = (tl.sum(x0, 1) + tl.sum(x1, 1) + tl.sum(part2, 1)) / D
    z0 = x0 - mean[:, None]
    v0 = tl.sum(z0 * z0, 1)
    z1 = x1 - mean[:, None]
    v1 = tl.sum(z1 * z1, 1)
    z2 = tl.where(tail, x2 - mean[:, None], 0.0)
    v2 = tl.sum(z2 * z2, 1)
    var = (v0 + v1 + v2) / D
    inv = tl.rsqrt(var + EPS)
    tl.store(T + batch * S + row, mean)
    tl.store(T + N + batch * S + row, inv)


@triton.jit
def _stats_reload286(
    X,
    T,
    S: tl.constexpr,
    D: tl.constexpr,
    N: tl.constexpr,
    EPS: tl.constexpr,
    R: tl.constexpr,
    MODE: tl.constexpr,
    FLAT: tl.constexpr,
    G: tl.constexpr,
    B: tl.constexpr,
):
    blocks = S // R
    batch = tl.program_id(0) // blocks
    block = tl.program_id(0) % blocks
    if blocks % G == 0:
        block = block % G * (blocks // G) + block // G
    row = block * R + tl.arange(0, R)
    col = tl.arange(0, 1024)
    base = X + (batch * S + row[:, None]) * D + col[None, :]
    tail = col[None, :] >= 3072 - D
    if MODE == 1:
        x0 = tl.load(base).to(tl.float32)
        x1 = tl.load(base + 1024).to(tl.float32)
        m0 = tl.sum(x0, 1)
        m1 = tl.sum(x1, 1)
    else:
        m0 = tl.sum(tl.load(base).to(tl.float32), 1)
        m1 = tl.sum(tl.load(base + 1024).to(tl.float32), 1)
    m2 = tl.sum(
        tl.where(tail, tl.load(base + D - 1024).to(tl.float32), 0.0), 1
    )
    mean = (m0 + m1 + m2) / D
    if MODE == 1:
        z0 = x0 - mean[:, None]
        v0 = tl.sum(z0 * z0, 1)
        z1 = x1 - mean[:, None]
        v1 = tl.sum(z1 * z1, 1)
    else:
        z0 = tl.load(base, volatile=True).to(tl.float32) - mean[:, None]
        v0 = tl.sum(z0 * z0, 1)
        z1 = tl.load(base + 1024, volatile=True).to(tl.float32) - mean[:, None]
        v1 = tl.sum(z1 * z1, 1)
    z2 = tl.where(
        tail,
        tl.load(base + D - 1024, volatile=True).to(tl.float32) - mean[:, None],
        0.0,
    )
    v2 = tl.sum(z2 * z2, 1)
    var = (v0 + v1 + v2) / D
    inv = tl.rsqrt(var + EPS)
    tl.store(T + batch * S + row, mean)
    tl.store(T + N + batch * S + row, inv)


@triton.jit
def _stats_fold312(
    X,
    T,
    S: tl.constexpr,
    D: tl.constexpr,
    N: tl.constexpr,
    EPS: tl.constexpr,
    R: tl.constexpr,
    MODE: tl.constexpr,
    FLAT: tl.constexpr,
    G: tl.constexpr,
    B: tl.constexpr,
):
    blocks = S // R
    batch = tl.program_id(0) // blocks
    block = tl.program_id(0) % blocks
    if blocks % G == 0:
        block = block % G * (blocks // G) + block // G
    row = block * R + tl.arange(0, R)
    col = tl.arange(0, 1024)
    base = X + (batch * S + row[:, None]) * D + col[None, :]
    tail = col[None, :] >= 3072 - D
    if MODE == 1 or MODE == 3:
        m = tl.load(base).to(tl.float32) + tl.load(base + 1024).to(tl.float32)
        m = m + tl.where(tail, tl.load(base + D - 1024).to(tl.float32), 0.0)
        mean = tl.sum(m, 1) / D
    else:
        m0 = tl.sum(tl.load(base).to(tl.float32), 1)
        m1 = tl.sum(tl.load(base + 1024).to(tl.float32), 1)
        m2 = tl.sum(
            tl.where(tail, tl.load(base + D - 1024).to(tl.float32), 0.0), 1
        )
        mean = (m0 + m1 + m2) / D
    z0 = tl.load(base, volatile=True).to(tl.float32) - mean[:, None]
    if MODE == 2 or MODE == 3:
        v = z0 * z0
        z1 = tl.load(base + 1024, volatile=True).to(tl.float32) - mean[:, None]
        v = v + z1 * z1
        z2 = tl.where(
            tail,
            tl.load(base + D - 1024, volatile=True).to(tl.float32)
            - mean[:, None],
            0.0,
        )
        v = v + z2 * z2
        var = tl.sum(v, 1) / D
    else:
        v0 = tl.sum(z0 * z0, 1)
        z1 = tl.load(base + 1024, volatile=True).to(tl.float32) - mean[:, None]
        v1 = tl.sum(z1 * z1, 1)
        z2 = tl.where(
            tail,
            tl.load(base + D - 1024, volatile=True).to(tl.float32)
            - mean[:, None],
            0.0,
        )
        v2 = tl.sum(z2 * z2, 1)
        var = (v0 + v1 + v2) / D
    tl.store(T + batch * S + row, mean)
    tl.store(T + N + batch * S + row, tl.rsqrt(var + EPS))


@triton.jit
def _stats_narrow323(
    X,
    T,
    S: tl.constexpr,
    D: tl.constexpr,
    N: tl.constexpr,
    EPS: tl.constexpr,
    R: tl.constexpr,
    MODE: tl.constexpr,
    FLAT: tl.constexpr,
    G: tl.constexpr,
    B: tl.constexpr,
):
    blocks = S // R
    batch = tl.program_id(0) // blocks
    block = tl.program_id(0) % blocks
    if blocks % G == 0:
        block = block % G * (blocks // G) + block // G
    row = block * R + tl.arange(0, R)
    col = tl.arange(0, 512)
    base = X + (batch * S + row[:, None]) * D + col[None, :]
    tail = col[None, :] >= 320
    m0 = tl.sum(tl.load(base + 0).to(tl.float32), 1)
    m1 = tl.sum(tl.load(base + 512).to(tl.float32), 1)
    m2 = tl.sum(tl.load(base + 1024).to(tl.float32), 1)
    m3 = tl.sum(tl.load(base + 1536).to(tl.float32), 1)
    mt = tl.sum(tl.where(tail, tl.load(base + D - 512).to(tl.float32), 0.0), 1)
    mean = (m0 + m1 + m2 + m3 + mt) / D
    z = tl.load(base + 0, volatile=True).to(tl.float32) - mean[:, None]
    v = z * z
    z = tl.load(base + 512, volatile=True).to(tl.float32) - mean[:, None]
    v = v + z * z
    z = tl.load(base + 1024, volatile=True).to(tl.float32) - mean[:, None]
    v = v + z * z
    z = tl.load(base + 1536, volatile=True).to(tl.float32) - mean[:, None]
    v = v + z * z
    z = tl.where(
        tail,
        tl.load(base + D - 512, volatile=True).to(tl.float32) - mean[:, None],
        0.0,
    )
    v = v + z * z
    var = tl.sum(v, 1) / D
    tl.store(T + batch * S + row, mean)
    tl.store(T + N + batch * S + row, tl.rsqrt(var + EPS))


@triton.jit
def _stats_narrow327(
    X,
    T,
    S: tl.constexpr,
    D: tl.constexpr,
    N: tl.constexpr,
    EPS: tl.constexpr,
    R: tl.constexpr,
    MODE: tl.constexpr,
    FLAT: tl.constexpr,
    G: tl.constexpr,
    B: tl.constexpr,
):
    blocks = S // R
    batch = tl.program_id(0) // blocks
    block = tl.program_id(0) % blocks
    if blocks % G == 0:
        block = block % G * (blocks // G) + block // G
    row = block * R + tl.arange(0, R)
    col = tl.arange(0, 256)
    base = X + (batch * S + row[:, None]) * D + col[None, :]
    tail = col[None, :] >= 64
    m0 = tl.sum(tl.load(base + 0).to(tl.float32), 1)
    m1 = tl.sum(tl.load(base + 256).to(tl.float32), 1)
    m2 = tl.sum(tl.load(base + 512).to(tl.float32), 1)
    m3 = tl.sum(tl.load(base + 768).to(tl.float32), 1)
    m4 = tl.sum(tl.load(base + 1024).to(tl.float32), 1)
    m5 = tl.sum(tl.load(base + 1280).to(tl.float32), 1)
    m6 = tl.sum(tl.load(base + 1536).to(tl.float32), 1)
    m7 = tl.sum(tl.load(base + 1792).to(tl.float32), 1)
    mt = tl.sum(tl.where(tail, tl.load(base + D - 256).to(tl.float32), 0.0), 1)
    mean = (m0 + m1 + m2 + m3 + m4 + m5 + m6 + m7 + mt) / D
    z = tl.load(base + 0, volatile=True).to(tl.float32) - mean[:, None]
    v = z * z
    z = tl.load(base + 256, volatile=True).to(tl.float32) - mean[:, None]
    v = v + z * z
    z = tl.load(base + 512, volatile=True).to(tl.float32) - mean[:, None]
    v = v + z * z
    z = tl.load(base + 768, volatile=True).to(tl.float32) - mean[:, None]
    v = v + z * z
    z = tl.load(base + 1024, volatile=True).to(tl.float32) - mean[:, None]
    v = v + z * z
    z = tl.load(base + 1280, volatile=True).to(tl.float32) - mean[:, None]
    v = v + z * z
    z = tl.load(base + 1536, volatile=True).to(tl.float32) - mean[:, None]
    v = v + z * z
    z = tl.load(base + 1792, volatile=True).to(tl.float32) - mean[:, None]
    v = v + z * z
    z = tl.where(
        tail,
        tl.load(base + D - 256, volatile=True).to(tl.float32) - mean[:, None],
        0.0,
    )
    v = v + z * z
    var = tl.sum(v, 1) / D
    tl.store(T + batch * S + row, mean)
    tl.store(T + N + batch * S + row, tl.rsqrt(var + EPS))


@triton.jit
def _apply_unroll8_affine343(
    X,
    A,
    H,
    T,
    Y,
    S: tl.constexpr,
    D: tl.constexpr,
    N: tl.constexpr,
    C: tl.constexpr,
    REPEATS: tl.constexpr,
):
    batch = tl.program_id(1)
    blocks = S * D // C
    programs = blocks // REPEATS
    block = tl.program_id(0)
    if programs % 16 == 0:
        block = block % 16 * (programs // 16) + block // 16
    base = block * C + tl.arange(0, C)
    col = base % D
    base_row = base // D
    a = tl.load(A + batch * D + col).to(tl.float32)
    h = tl.load(H + batch * D + col).to(tl.float32)
    factor = 1.0 + a
    for pair in range(REPEATS // 8):
        iteration_index = 8 * pair * programs * C
        iteration_row = 8 * pair * (programs * C // D)
        index0 = base + (iteration_index + 0 * programs * C)
        row0 = base_row + (iteration_row + 0 * (programs * C // D))
        index1 = base + (iteration_index + 1 * programs * C)
        row1 = base_row + (iteration_row + 1 * (programs * C // D))
        index2 = base + (iteration_index + 2 * programs * C)
        row2 = base_row + (iteration_row + 2 * (programs * C // D))
        index3 = base + (iteration_index + 3 * programs * C)
        row3 = base_row + (iteration_row + 3 * (programs * C // D))
        index4 = base + (iteration_index + 4 * programs * C)
        row4 = base_row + (iteration_row + 4 * (programs * C // D))
        index5 = base + (iteration_index + 5 * programs * C)
        row5 = base_row + (iteration_row + 5 * (programs * C // D))
        index6 = base + (iteration_index + 6 * programs * C)
        row6 = base_row + (iteration_row + 6 * (programs * C // D))
        index7 = base + (iteration_index + 7 * programs * C)
        row7 = base_row + (iteration_row + 7 * (programs * C // D))
        x0 = tl.load(X + batch * S * D + index0).to(tl.float32)
        mean0 = tl.load(T + batch * S + row0)
        inv0 = tl.load(T + N + batch * S + row0)
        y0 = (x0 - mean0) * inv0 * factor + h
        tl.store(Y + batch * S * D + index0, y0)
        x1 = tl.load(X + batch * S * D + index1).to(tl.float32)
        mean1 = tl.load(T + batch * S + row1)
        inv1 = tl.load(T + N + batch * S + row1)
        y1 = (x1 - mean1) * inv1 * factor + h
        tl.store(Y + batch * S * D + index1, y1)
        x2 = tl.load(X + batch * S * D + index2).to(tl.float32)
        mean2 = tl.load(T + batch * S + row2)
        inv2 = tl.load(T + N + batch * S + row2)
        y2 = (x2 - mean2) * inv2 * factor + h
        tl.store(Y + batch * S * D + index2, y2)
        x3 = tl.load(X + batch * S * D + index3).to(tl.float32)
        mean3 = tl.load(T + batch * S + row3)
        inv3 = tl.load(T + N + batch * S + row3)
        y3 = (x3 - mean3) * inv3 * factor + h
        tl.store(Y + batch * S * D + index3, y3)
        x4 = tl.load(X + batch * S * D + index4).to(tl.float32)
        mean4 = tl.load(T + batch * S + row4)
        inv4 = tl.load(T + N + batch * S + row4)
        y4 = (x4 - mean4) * inv4 * factor + h
        tl.store(Y + batch * S * D + index4, y4)
        x5 = tl.load(X + batch * S * D + index5).to(tl.float32)
        mean5 = tl.load(T + batch * S + row5)
        inv5 = tl.load(T + N + batch * S + row5)
        y5 = (x5 - mean5) * inv5 * factor + h
        tl.store(Y + batch * S * D + index5, y5)
        x6 = tl.load(X + batch * S * D + index6).to(tl.float32)
        mean6 = tl.load(T + batch * S + row6)
        inv6 = tl.load(T + N + batch * S + row6)
        y6 = (x6 - mean6) * inv6 * factor + h
        tl.store(Y + batch * S * D + index6, y6)
        x7 = tl.load(X + batch * S * D + index7).to(tl.float32)
        mean7 = tl.load(T + batch * S + row7)
        inv7 = tl.load(T + N + batch * S + row7)
        y7 = (x7 - mean7) * inv7 * factor + h
        tl.store(Y + batch * S * D + index7, y7)


__all__ = ["layernorm_modulate"]
