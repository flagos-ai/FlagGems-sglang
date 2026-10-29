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
def _kernel(
    X,
    Y,
    N: tl.constexpr,
    S: tl.constexpr,
    R: tl.constexpr,
    SCALE: tl.constexpr,
):
    row = tl.program_id(0)
    offset = row * N
    idx = tl.arange(0, N // R)
    j = idx // S * (R * S) + idx % S
    if R == 1:
        v0_0 = tl.load(X + offset + (j + 0 * S)).to(tl.float32)
        tl.store(Y + row * N + j + 0 * S, v0_0 * SCALE, mask=idx < N // R)
    elif R == 2:
        v0_0 = tl.load(X + offset + (j + 0 * S)).to(tl.float32)
        v0_1 = tl.load(X + offset + (j + 1 * S)).to(tl.float32)
        v1_0 = v0_0 + v0_1
        v1_1 = v0_0 - v0_1
        tl.store(Y + row * N + j + 0 * S, v1_0 * SCALE, mask=idx < N // R)
        tl.store(Y + row * N + j + 1 * S, v1_1 * SCALE, mask=idx < N // R)
    elif R == 4:
        v0_0 = tl.load(X + offset + (j + 0 * S)).to(tl.float32)
        v0_1 = tl.load(X + offset + (j + 1 * S)).to(tl.float32)
        v0_2 = tl.load(X + offset + (j + 2 * S)).to(tl.float32)
        v0_3 = tl.load(X + offset + (j + 3 * S)).to(tl.float32)
        v1_0 = v0_0 + v0_1
        v1_1 = v0_0 - v0_1
        v1_2 = v0_2 + v0_3
        v1_3 = v0_2 - v0_3
        v2_0 = v1_0 + v1_2
        v2_1 = v1_1 + v1_3
        v2_2 = v1_0 - v1_2
        v2_3 = v1_1 - v1_3
        tl.store(Y + row * N + j + 0 * S, v2_0 * SCALE, mask=idx < N // R)
        tl.store(Y + row * N + j + 1 * S, v2_1 * SCALE, mask=idx < N // R)
        tl.store(Y + row * N + j + 2 * S, v2_2 * SCALE, mask=idx < N // R)
        tl.store(Y + row * N + j + 3 * S, v2_3 * SCALE, mask=idx < N // R)
    elif R == 8:
        v0_0 = tl.load(X + offset + (j + 0 * S)).to(tl.float32)
        v0_1 = tl.load(X + offset + (j + 1 * S)).to(tl.float32)
        v0_2 = tl.load(X + offset + (j + 2 * S)).to(tl.float32)
        v0_3 = tl.load(X + offset + (j + 3 * S)).to(tl.float32)
        v0_4 = tl.load(X + offset + (j + 4 * S)).to(tl.float32)
        v0_5 = tl.load(X + offset + (j + 5 * S)).to(tl.float32)
        v0_6 = tl.load(X + offset + (j + 6 * S)).to(tl.float32)
        v0_7 = tl.load(X + offset + (j + 7 * S)).to(tl.float32)
        v1_0 = v0_0 + v0_1
        v1_1 = v0_0 - v0_1
        v1_2 = v0_2 + v0_3
        v1_3 = v0_2 - v0_3
        v1_4 = v0_4 + v0_5
        v1_5 = v0_4 - v0_5
        v1_6 = v0_6 + v0_7
        v1_7 = v0_6 - v0_7
        v2_0 = v1_0 + v1_2
        v2_1 = v1_1 + v1_3
        v2_2 = v1_0 - v1_2
        v2_3 = v1_1 - v1_3
        v2_4 = v1_4 + v1_6
        v2_5 = v1_5 + v1_7
        v2_6 = v1_4 - v1_6
        v2_7 = v1_5 - v1_7
        v3_0 = v2_0 + v2_4
        v3_1 = v2_1 + v2_5
        v3_2 = v2_2 + v2_6
        v3_3 = v2_3 + v2_7
        v3_4 = v2_0 - v2_4
        v3_5 = v2_1 - v2_5
        v3_6 = v2_2 - v2_6
        v3_7 = v2_3 - v2_7
        tl.store(Y + row * N + j + 0 * S, v3_0 * SCALE, mask=idx < N // R)
        tl.store(Y + row * N + j + 1 * S, v3_1 * SCALE, mask=idx < N // R)
        tl.store(Y + row * N + j + 2 * S, v3_2 * SCALE, mask=idx < N // R)
        tl.store(Y + row * N + j + 3 * S, v3_3 * SCALE, mask=idx < N // R)
        tl.store(Y + row * N + j + 4 * S, v3_4 * SCALE, mask=idx < N // R)
        tl.store(Y + row * N + j + 5 * S, v3_5 * SCALE, mask=idx < N // R)
        tl.store(Y + row * N + j + 6 * S, v3_6 * SCALE, mask=idx < N // R)
        tl.store(Y + row * N + j + 7 * S, v3_7 * SCALE, mask=idx < N // R)
    elif R == 16:
        v0_0 = tl.load(X + offset + (j + 0 * S)).to(tl.float32)
        v0_1 = tl.load(X + offset + (j + 1 * S)).to(tl.float32)
        v0_2 = tl.load(X + offset + (j + 2 * S)).to(tl.float32)
        v0_3 = tl.load(X + offset + (j + 3 * S)).to(tl.float32)
        v0_4 = tl.load(X + offset + (j + 4 * S)).to(tl.float32)
        v0_5 = tl.load(X + offset + (j + 5 * S)).to(tl.float32)
        v0_6 = tl.load(X + offset + (j + 6 * S)).to(tl.float32)
        v0_7 = tl.load(X + offset + (j + 7 * S)).to(tl.float32)
        v0_8 = tl.load(X + offset + (j + 8 * S)).to(tl.float32)
        v0_9 = tl.load(X + offset + (j + 9 * S)).to(tl.float32)
        v0_10 = tl.load(X + offset + (j + 10 * S)).to(tl.float32)
        v0_11 = tl.load(X + offset + (j + 11 * S)).to(tl.float32)
        v0_12 = tl.load(X + offset + (j + 12 * S)).to(tl.float32)
        v0_13 = tl.load(X + offset + (j + 13 * S)).to(tl.float32)
        v0_14 = tl.load(X + offset + (j + 14 * S)).to(tl.float32)
        v0_15 = tl.load(X + offset + (j + 15 * S)).to(tl.float32)
        v1_0 = v0_0 + v0_1
        v1_1 = v0_0 - v0_1
        v1_2 = v0_2 + v0_3
        v1_3 = v0_2 - v0_3
        v1_4 = v0_4 + v0_5
        v1_5 = v0_4 - v0_5
        v1_6 = v0_6 + v0_7
        v1_7 = v0_6 - v0_7
        v1_8 = v0_8 + v0_9
        v1_9 = v0_8 - v0_9
        v1_10 = v0_10 + v0_11
        v1_11 = v0_10 - v0_11
        v1_12 = v0_12 + v0_13
        v1_13 = v0_12 - v0_13
        v1_14 = v0_14 + v0_15
        v1_15 = v0_14 - v0_15
        v2_0 = v1_0 + v1_2
        v2_1 = v1_1 + v1_3
        v2_2 = v1_0 - v1_2
        v2_3 = v1_1 - v1_3
        v2_4 = v1_4 + v1_6
        v2_5 = v1_5 + v1_7
        v2_6 = v1_4 - v1_6
        v2_7 = v1_5 - v1_7
        v2_8 = v1_8 + v1_10
        v2_9 = v1_9 + v1_11
        v2_10 = v1_8 - v1_10
        v2_11 = v1_9 - v1_11
        v2_12 = v1_12 + v1_14
        v2_13 = v1_13 + v1_15
        v2_14 = v1_12 - v1_14
        v2_15 = v1_13 - v1_15
        v3_0 = v2_0 + v2_4
        v3_1 = v2_1 + v2_5
        v3_2 = v2_2 + v2_6
        v3_3 = v2_3 + v2_7
        v3_4 = v2_0 - v2_4
        v3_5 = v2_1 - v2_5
        v3_6 = v2_2 - v2_6
        v3_7 = v2_3 - v2_7
        v3_8 = v2_8 + v2_12
        v3_9 = v2_9 + v2_13
        v3_10 = v2_10 + v2_14
        v3_11 = v2_11 + v2_15
        v3_12 = v2_8 - v2_12
        v3_13 = v2_9 - v2_13
        v3_14 = v2_10 - v2_14
        v3_15 = v2_11 - v2_15
        v4_0 = v3_0 + v3_8
        v4_1 = v3_1 + v3_9
        v4_2 = v3_2 + v3_10
        v4_3 = v3_3 + v3_11
        v4_4 = v3_4 + v3_12
        v4_5 = v3_5 + v3_13
        v4_6 = v3_6 + v3_14
        v4_7 = v3_7 + v3_15
        v4_8 = v3_0 - v3_8
        v4_9 = v3_1 - v3_9
        v4_10 = v3_2 - v3_10
        v4_11 = v3_3 - v3_11
        v4_12 = v3_4 - v3_12
        v4_13 = v3_5 - v3_13
        v4_14 = v3_6 - v3_14
        v4_15 = v3_7 - v3_15
        tl.store(Y + row * N + j + 0 * S, v4_0 * SCALE, mask=idx < N // R)
        tl.store(Y + row * N + j + 1 * S, v4_1 * SCALE, mask=idx < N // R)
        tl.store(Y + row * N + j + 2 * S, v4_2 * SCALE, mask=idx < N // R)
        tl.store(Y + row * N + j + 3 * S, v4_3 * SCALE, mask=idx < N // R)
        tl.store(Y + row * N + j + 4 * S, v4_4 * SCALE, mask=idx < N // R)
        tl.store(Y + row * N + j + 5 * S, v4_5 * SCALE, mask=idx < N // R)
        tl.store(Y + row * N + j + 6 * S, v4_6 * SCALE, mask=idx < N // R)
        tl.store(Y + row * N + j + 7 * S, v4_7 * SCALE, mask=idx < N // R)
        tl.store(Y + row * N + j + 8 * S, v4_8 * SCALE, mask=idx < N // R)
        tl.store(Y + row * N + j + 9 * S, v4_9 * SCALE, mask=idx < N // R)
        tl.store(Y + row * N + j + 10 * S, v4_10 * SCALE, mask=idx < N // R)
        tl.store(Y + row * N + j + 11 * S, v4_11 * SCALE, mask=idx < N // R)
        tl.store(Y + row * N + j + 12 * S, v4_12 * SCALE, mask=idx < N // R)
        tl.store(Y + row * N + j + 13 * S, v4_13 * SCALE, mask=idx < N // R)
        tl.store(Y + row * N + j + 14 * S, v4_14 * SCALE, mask=idx < N // R)
        tl.store(Y + row * N + j + 15 * S, v4_15 * SCALE, mask=idx < N // R)


def _butterfly(x, scale):
    if not x.is_contiguous():
        x = x.contiguous()
    out = torch.empty_like(x)
    if x.numel() == 0:
        return out
    n = x.shape[-1]
    rows = x.numel() // n
    packed_rows = min(rows & -rows, max(1, 8192 // n))
    block = n * packed_rows
    programs = rows // packed_rows
    stages = n.bit_length() - 1
    if stages == 0:
        _kernel.run(
            x,
            out,
            n,
            1,
            1,
            scale,
            grid=(rows,),
            warmup=False,
            num_warps=1,
            num_stages=1,
        )
        return out
    a = torch.empty((rows, n), device=x.device, dtype=torch.float32)
    current = x
    for stage in range(0, stages, 4):
        radix = 1 << min(4, stages - stage)
        last = stage + 4 >= stages
        target = out if last else a
        _kernel.run(
            current,
            target,
            block,
            1 << stage,
            radix,
            scale if last else 1.0,
            grid=(programs,),
            warmup=False,
            num_warps=1,
            num_stages=1,
            enable_fp_fusion=False,
        )
        current = target
    return out


@triton.jit
def _round_final(v, DT: tl.constexpr):
    if DT == tl.bfloat16:
        bits = v.to(tl.uint32, bitcast=True)
        rounded = ((bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000).to(
            tl.float32, bitcast=True
        )
        return tl.where((bits & 0x7F800000) == 0x7F800000, v, rounded)
    return v


@triton.jit
def _finalize(X, Y, TOTAL: tl.constexpr, B: tl.constexpr, SCALE: tl.constexpr):
    i = tl.program_id(0) * B + tl.arange(0, B)
    value = tl.load(X + i, mask=i < TOTAL, other=0.0)
    tl.store(
        Y + i, _round_final(value * SCALE, Y.dtype.element_ty), mask=i < TOTAL
    )


def _finalize_call(result, x, scale):
    out = torch.empty_like(x)
    total = x.numel()
    block = min(8192, total & -total)
    _finalize.run(
        result,
        out,
        total,
        block,
        scale,
        grid=(total // block,),
        warmup=False,
        num_warps=1,
        num_stages=1,
        enable_fp_fusion=False,
    )
    return out


@triton.jit
def _factor_prepare(
    X,
    F,
    HA,
    HG,
    M: tl.constexpr,
    N: tl.constexpr,
    A: tl.constexpr,
    G: tl.constexpr,
    B: tl.constexpr,
    D: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid < M * N // B:
        offsets = pid * B + tl.arange(0, B)
        values = tl.load(X + offsets).to(tl.float32)
        tl.store(F + offsets, values)
    elif pid == M * N // B:
        a_offsets = tl.arange(0, A * A)
        a_bits = a_offsets // A & a_offsets % A
        a_bits = a_bits ^ (a_bits >> 8)
        a_bits = a_bits ^ (a_bits >> 4)
        a_bits = a_bits ^ (a_bits >> 2)
        a_bits = a_bits ^ (a_bits >> 1)
        tl.store(
            HA + a_offsets,
            tl.where((a_bits & 1) == 0, 1.0, -1.0).to(tl.float32),
        )
    else:
        g_offsets = tl.arange(0, G * G)
        g_bits = g_offsets // G & g_offsets % G
        g_bits = g_bits & (D - 1)
        g_bits = g_bits ^ (g_bits >> 8)
        g_bits = g_bits ^ (g_bits >> 4)
        g_bits = g_bits ^ (g_bits >> 2)
        g_bits = g_bits ^ (g_bits >> 1)
        g_signs = 1.0 - 2.0 * (g_bits & 1).to(tl.float32)
        g_separate = (g_offsets // G) ^ (g_offsets % G)
        for bit in tl.static_range((G // D).bit_length() - 1):
            g_signs = g_signs * (
                1.0
                - ((g_separate >> (D.bit_length() - 1 + bit)) & 1).to(
                    tl.float32
                )
            )
        tl.store(HG + g_offsets, g_signs)


@triton.jit
def _factor_fused(
    F, HA, HG, Y, A: tl.constexpr, G: tl.constexpr, SCALE: tl.constexpr
):
    row = tl.program_id(0)
    outer = tl.arange(0, G)
    inner = tl.arange(0, A)
    values = tl.load(F + row * (G * A) + outer[:, None] * A + inner[None, :])
    low = tl.load(HA + inner[:, None] * A + inner[None, :])
    high = tl.load(HG + outer[:, None] * G + outer[None, :])
    first = tl.dot(values, low, input_precision="ieee", out_dtype=tl.float32)
    result = tl.dot(high, first, input_precision="ieee", out_dtype=tl.float32)
    tl.store(
        Y + row * (G * A) + outer[:, None] * A + inner[None, :], result * SCALE
    )


def _call(x, scale, packed):
    n = x.shape[-1]
    rows = x.numel() // n
    if n not in (128, 256, 512, 1024, 2048) or rows == 0 or rows % packed:
        return _butterfly(x, scale)
    if not x.is_contiguous():
        x = x.contiguous()
    a = 16 if n <= 256 else 32
    g = packed * (n // a)
    full = torch.empty((rows, n), device=x.device, dtype=torch.float32)
    ha = torch.empty((a, a), device=x.device, dtype=torch.float32)
    hg = torch.empty((g, g), device=x.device, dtype=torch.float32)
    out = torch.empty((rows, n), device=x.device, dtype=torch.float32)
    cap = (
        8192
        if n == 2048 and rows >= 512 or n == 512 and rows >= 4096
        else 1024
    )
    block = min(cap, (rows * n) & -(rows * n))
    _factor_prepare.run(
        x,
        full,
        ha,
        hg,
        rows,
        n,
        a,
        g,
        block,
        n // a,
        grid=(rows * n // block + 2,),
        warmup=False,
        num_warps=4,
        num_stages=1,
    )
    _factor_fused.run(
        full,
        ha,
        hg,
        out,
        a,
        g,
        1.0,
        grid=(rows * n // (a * g),),
        warmup=False,
        num_warps=4,
        num_stages=1,
        enable_fp_fusion=False,
    )
    return _finalize_call(out, x, scale)


def _fallback(x, scale):
    n = x.shape[-1]
    rows = x.numel() // n
    if n == 128 and (rows < 2 or rows % 2):
        return _butterfly(x, scale)
    packed = min(16, rows & -rows) if n == 128 else 1
    return _call(x, scale, packed)


@triton.jit
def _dense128_prepare(X, F, H, M: tl.constexpr, B: tl.constexpr):
    pid = tl.program_id(0)
    lane = tl.arange(0, B)
    if pid < M * 128 // B:
        offsets = pid * B + lane
        tl.store(F + offsets, tl.load(X + offsets).to(tl.float32))
    else:
        offsets = (pid - M * 128 // B) * B + lane
        bits = (offsets // 128) & (offsets % 128)
        bits = bits ^ (bits >> 4)
        bits = bits ^ (bits >> 2)
        bits = bits ^ (bits >> 1)
        signs = 1.0 - 2.0 * (bits & 1).to(tl.float32)
        tl.store(H + offsets, signs)


@triton.jit
def _dense128(F, H, Y, BR: tl.constexpr, SCALE: tl.constexpr):
    rows = tl.program_id(0) * BR + tl.arange(0, BR)
    columns = tl.arange(0, 128)
    lhs = tl.load(F + rows[:, None] * 128 + columns[None, :])
    rhs = tl.load(H + columns[:, None] * 128 + columns[None, :])
    result = tl.dot(lhs, rhs, input_precision="ieee", out_dtype=tl.float32)
    tl.store(Y + rows[:, None] * 128 + columns[None, :], result * SCALE)


def _dense128_call(x, scale, block_rows):
    if not x.is_contiguous():
        x = x.contiguous()
    rows = x.numel() // 128
    full = torch.empty((rows, 128), device=x.device, dtype=torch.float32)
    h = torch.empty((128, 128), device=x.device, dtype=torch.float32)
    out = torch.empty((rows, 128), device=x.device, dtype=torch.float32)
    cap = 4096 if rows >= 4096 else 1024
    block = min(cap, (rows * 128) & -(rows * 128))
    _dense128_prepare.run(
        x,
        full,
        h,
        rows,
        block,
        grid=(rows * 128 // block + 16384 // block,),
        warmup=False,
        num_warps=4,
        num_stages=1,
    )
    _dense128.run(
        full,
        h,
        out,
        block_rows,
        1.0,
        grid=(rows // block_rows,),
        warmup=False,
        num_warps=4,
        num_stages=1,
        enable_fp_fusion=False,
    )
    return _finalize_call(out, x, scale)


def hadamard_transform(x, scale):
    n = x.shape[-1]
    rows = x.numel() // n
    if n == 128 and rows >= 64 and rows % 64 == 0:
        return _dense128_call(x, scale, 64)
    if n == 512 and rows >= 4096 and rows % 8 == 0:
        return _call(x, scale, 8)
    if n == 512 and rows >= 512 and rows % 4 == 0:
        return _call(x, scale, 4)
    if n == 2048 and rows >= 4096 and rows % 2 == 0:
        return _call(x, scale, 2)
    return _fallback(x, scale)


__all__ = ["hadamard_transform"]
