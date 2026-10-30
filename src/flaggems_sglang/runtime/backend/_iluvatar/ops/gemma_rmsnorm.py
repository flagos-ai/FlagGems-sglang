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
def _permuted_chunked(
    X,
    W,
    Y,
    N: tl.constexpr,
    E: tl.constexpr,
    M: tl.constexpr,
    S: tl.constexpr,
    SH: tl.constexpr,
    ST: tl.constexpr,
):
    r = tl.program_id(0).to(tl.int64) * 16 + tl.arange(0, 16)
    c = tl.arange(0, 256).to(tl.int64)
    destination = tl.full((16,), 0, tl.int64)
    factor = 1
    for axis in tl.static_range(len(SH) - 2, -1, -1):
        if SH[axis] > 1:
            destination += r // ST[axis] % SH[axis] * factor
        factor *= SH[axis]
    total = tl.full((16,), 0, tl.float32)
    for start in range(0, N, 256):
        col = start + c
        x = tl.load(
            X + col[:, None] * S + r[None, :], r[None, :] < M, other=0
        ).to(tl.float32)
        total += tl.sum(x * x, 0)
    inv = tl.rsqrt(total / N + E)
    for start in range(0, N, 256):
        col = N - 256 - start + c
        x = tl.load(
            X + col[:, None] * S + r[None, :], r[None, :] < M, other=0
        ).to(tl.float32)
        w = tl.load(W + col).to(tl.float32)
        tl.store(
            Y + destination[None, :] * N + col[:, None],
            x * inv[None, :] * (1 + w[:, None]),
            r[None, :] < M,
        )


@triton.jit
def _permuted_transposed(
    X,
    W,
    Y,
    N: tl.constexpr,
    E: tl.constexpr,
    M: tl.constexpr,
    S: tl.constexpr,
    R: tl.constexpr,
    SH: tl.constexpr,
    ST: tl.constexpr,
):
    r = tl.program_id(0).to(tl.int64) * R + tl.arange(0, R)
    destination = tl.full((R,), 0, tl.int64)
    factor = 1
    for axis in tl.static_range(len(SH) - 2, -1, -1):
        if SH[axis] > 1:
            destination += r // ST[axis] % SH[axis] * factor
        factor *= SH[axis]
    c = tl.arange(0, N).to(tl.int64)
    x = tl.load(X + c[:, None] * S + r[None, :], r[None, :] < M, other=0).to(
        tl.float32
    )
    w = tl.load(W + c).to(tl.float32)
    inv = tl.rsqrt(tl.sum(x * x, 0) / N + E)
    y = x * inv[None, :] * (1.0 + w[:, None])
    tl.store(Y + destination[None, :] * N + c[:, None], y, r[None, :] < M)


@triton.jit
def _transposed(
    X,
    W,
    Y,
    N: tl.constexpr,
    E: tl.constexpr,
    M: tl.constexpr,
    S: tl.constexpr,
    R: tl.constexpr,
):
    r = tl.program_id(0).to(tl.int64) * R + tl.arange(0, R)
    c = tl.arange(0, N).to(tl.int64)
    x = tl.load(X + c[:, None] * S + r[None, :], r[None, :] < M, other=0).to(
        tl.float32
    )
    w = tl.load(W + c).to(tl.float32)
    inv = tl.rsqrt(tl.sum(x * x, 0) / N + E)
    y = x * inv[None, :] * (1.0 + w[:, None])
    tl.store(Y + r[None, :] * N + c[:, None], y, r[None, :] < M)


@triton.jit
def _rowfragment(
    X, W, Y, N: tl.constexpr, E: tl.constexpr, R: tl.constexpr, S: tl.constexpr
):
    row = tl.program_id(0).to(tl.int64) * R + tl.arange(0, R)
    c = tl.arange(0, 128)
    acc = tl.full((R, 128), 0, tl.float32)
    if N > 0:
        x0 = tl.load(X + row[:, None] * S + c[None, :] + 0)
        f0 = x0.to(tl.float32)
        acc = acc + f0 * f0
    if N > 128:
        x1 = tl.load(X + row[:, None] * S + c[None, :] + 128)
        f1 = x1.to(tl.float32)
        acc = acc + f1 * f1
    if N > 256:
        x2 = tl.load(X + row[:, None] * S + c[None, :] + 256)
        f2 = x2.to(tl.float32)
        acc = acc + f2 * f2
    if N > 384:
        x3 = tl.load(X + row[:, None] * S + c[None, :] + 384)
        f3 = x3.to(tl.float32)
        acc = acc + f3 * f3
    if N > 512:
        x4 = tl.load(X + row[:, None] * S + c[None, :] + 512)
        f4 = x4.to(tl.float32)
        acc = acc + f4 * f4
    if N > 640:
        x5 = tl.load(X + row[:, None] * S + c[None, :] + 640)
        f5 = x5.to(tl.float32)
        acc = acc + f5 * f5
    if N > 768:
        x6 = tl.load(X + row[:, None] * S + c[None, :] + 768)
        f6 = x6.to(tl.float32)
        acc = acc + f6 * f6
    if N > 896:
        x7 = tl.load(X + row[:, None] * S + c[None, :] + 896)
        f7 = x7.to(tl.float32)
        acc = acc + f7 * f7
    if N > 1024:
        x8 = tl.load(X + row[:, None] * S + c[None, :] + 1024)
        f8 = x8.to(tl.float32)
        acc = acc + f8 * f8
    if N > 1152:
        x9 = tl.load(X + row[:, None] * S + c[None, :] + 1152)
        f9 = x9.to(tl.float32)
        acc = acc + f9 * f9
    if N > 1280:
        x10 = tl.load(X + row[:, None] * S + c[None, :] + 1280)
        f10 = x10.to(tl.float32)
        acc = acc + f10 * f10
    if N > 1408:
        x11 = tl.load(X + row[:, None] * S + c[None, :] + 1408)
        f11 = x11.to(tl.float32)
        acc = acc + f11 * f11
    if N > 1536:
        x12 = tl.load(X + row[:, None] * S + c[None, :] + 1536)
        f12 = x12.to(tl.float32)
        acc = acc + f12 * f12
    if N > 1664:
        x13 = tl.load(X + row[:, None] * S + c[None, :] + 1664)
        f13 = x13.to(tl.float32)
        acc = acc + f13 * f13
    if N > 1792:
        x14 = tl.load(X + row[:, None] * S + c[None, :] + 1792)
        f14 = x14.to(tl.float32)
        acc = acc + f14 * f14
    if N > 1920:
        x15 = tl.load(X + row[:, None] * S + c[None, :] + 1920)
        f15 = x15.to(tl.float32)
        acc = acc + f15 * f15
    inv = tl.rsqrt(tl.sum(acc, 1) / N + E)
    if N > 0:
        w0 = tl.load(W + tl.broadcast_to(c[None, :] + 0, (R, 128))).to(
            tl.float32
        )
        y0 = x0.to(tl.float32) * inv[:, None] * (1.0 + w0)
        tl.store(Y + row[:, None] * N + c[None, :] + 0, y0)
    if N > 128:
        w1 = tl.load(W + tl.broadcast_to(c[None, :] + 128, (R, 128))).to(
            tl.float32
        )
        y1 = x1.to(tl.float32) * inv[:, None] * (1.0 + w1)
        tl.store(Y + row[:, None] * N + c[None, :] + 128, y1)
    if N > 256:
        w2 = tl.load(W + tl.broadcast_to(c[None, :] + 256, (R, 128))).to(
            tl.float32
        )
        y2 = x2.to(tl.float32) * inv[:, None] * (1.0 + w2)
        tl.store(Y + row[:, None] * N + c[None, :] + 256, y2)
    if N > 384:
        w3 = tl.load(W + tl.broadcast_to(c[None, :] + 384, (R, 128))).to(
            tl.float32
        )
        y3 = x3.to(tl.float32) * inv[:, None] * (1.0 + w3)
        tl.store(Y + row[:, None] * N + c[None, :] + 384, y3)
    if N > 512:
        w4 = tl.load(W + tl.broadcast_to(c[None, :] + 512, (R, 128))).to(
            tl.float32
        )
        y4 = x4.to(tl.float32) * inv[:, None] * (1.0 + w4)
        tl.store(Y + row[:, None] * N + c[None, :] + 512, y4)
    if N > 640:
        w5 = tl.load(W + tl.broadcast_to(c[None, :] + 640, (R, 128))).to(
            tl.float32
        )
        y5 = x5.to(tl.float32) * inv[:, None] * (1.0 + w5)
        tl.store(Y + row[:, None] * N + c[None, :] + 640, y5)
    if N > 768:
        w6 = tl.load(W + tl.broadcast_to(c[None, :] + 768, (R, 128))).to(
            tl.float32
        )
        y6 = x6.to(tl.float32) * inv[:, None] * (1.0 + w6)
        tl.store(Y + row[:, None] * N + c[None, :] + 768, y6)
    if N > 896:
        w7 = tl.load(W + tl.broadcast_to(c[None, :] + 896, (R, 128))).to(
            tl.float32
        )
        y7 = x7.to(tl.float32) * inv[:, None] * (1.0 + w7)
        tl.store(Y + row[:, None] * N + c[None, :] + 896, y7)
    if N > 1024:
        w8 = tl.load(W + tl.broadcast_to(c[None, :] + 1024, (R, 128))).to(
            tl.float32
        )
        y8 = x8.to(tl.float32) * inv[:, None] * (1.0 + w8)
        tl.store(Y + row[:, None] * N + c[None, :] + 1024, y8)
    if N > 1152:
        w9 = tl.load(W + tl.broadcast_to(c[None, :] + 1152, (R, 128))).to(
            tl.float32
        )
        y9 = x9.to(tl.float32) * inv[:, None] * (1.0 + w9)
        tl.store(Y + row[:, None] * N + c[None, :] + 1152, y9)
    if N > 1280:
        w10 = tl.load(W + tl.broadcast_to(c[None, :] + 1280, (R, 128))).to(
            tl.float32
        )
        y10 = x10.to(tl.float32) * inv[:, None] * (1.0 + w10)
        tl.store(Y + row[:, None] * N + c[None, :] + 1280, y10)
    if N > 1408:
        w11 = tl.load(W + tl.broadcast_to(c[None, :] + 1408, (R, 128))).to(
            tl.float32
        )
        y11 = x11.to(tl.float32) * inv[:, None] * (1.0 + w11)
        tl.store(Y + row[:, None] * N + c[None, :] + 1408, y11)
    if N > 1536:
        w12 = tl.load(W + tl.broadcast_to(c[None, :] + 1536, (R, 128))).to(
            tl.float32
        )
        y12 = x12.to(tl.float32) * inv[:, None] * (1.0 + w12)
        tl.store(Y + row[:, None] * N + c[None, :] + 1536, y12)
    if N > 1664:
        w13 = tl.load(W + tl.broadcast_to(c[None, :] + 1664, (R, 128))).to(
            tl.float32
        )
        y13 = x13.to(tl.float32) * inv[:, None] * (1.0 + w13)
        tl.store(Y + row[:, None] * N + c[None, :] + 1664, y13)
    if N > 1792:
        w14 = tl.load(W + tl.broadcast_to(c[None, :] + 1792, (R, 128))).to(
            tl.float32
        )
        y14 = x14.to(tl.float32) * inv[:, None] * (1.0 + w14)
        tl.store(Y + row[:, None] * N + c[None, :] + 1792, y14)
    if N > 1920:
        w15 = tl.load(W + tl.broadcast_to(c[None, :] + 1920, (R, 128))).to(
            tl.float32
        )
        y15 = x15.to(tl.float32) * inv[:, None] * (1.0 + w15)
        tl.store(Y + row[:, None] * N + c[None, :] + 1920, y15)


@triton.jit
def _narrow_tile(
    X, W, Y, N: tl.constexpr, E: tl.constexpr, M: tl.constexpr, R: tl.constexpr
):
    rows = tl.program_id(0).to(tl.int64) * R + tl.arange(0, R)
    cols = tl.arange(0, N)
    offsets = rows[:, None] * N + cols[None, :]
    if M % R == 0:
        x = tl.load(X + offsets).to(tl.float32)
    else:
        x = tl.load(X + offsets, rows[:, None] < M, other=0).to(tl.float32)
    w = tl.load(W + tl.broadcast_to(cols[None, :], (R, N))).to(tl.float32)
    inv = tl.rsqrt(tl.sum(x * x, 1) / N + E)
    y = x * inv[:, None] * (1.0 + w)
    if M % R == 0:
        tl.store(Y + offsets, y)
    else:
        tl.store(Y + offsets, y, rows[:, None] < M)


@triton.jit
def _norm(
    X,
    W,
    Y,
    SH: tl.constexpr,
    ST: tl.constexpr,
    WS: tl.constexpr,
    N: tl.constexpr,
    E: tl.constexpr,
    B: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    rem = row
    base = 0
    for a in tl.static_range(len(SH) - 2, -1, -1):
        base += rem % SH[a] * ST[a]
        rem = rem // SH[a]
    c = tl.arange(0, B).to(tl.int64)
    x = tl.load(X + base + c * ST[-1], c < N, other=0).to(tl.float32)
    w = tl.load(W + c * WS, c < N, other=0).to(tl.float32)
    v = tl.sum(x * x, 0) / N
    y = x * tl.rsqrt(v + E) * (1.0 + w)
    tl.store(Y + row * N + c, y, c < N)


@triton.jit
def _parts8(X, W, Y, N: tl.constexpr, E: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    c = tl.arange(0, N // 8).to(tl.int64)
    x0 = tl.load(X + row * N + 0 * (N // 8) + c).to(tl.float32)
    x1 = tl.load(X + row * N + 1 * (N // 8) + c).to(tl.float32)
    x2 = tl.load(X + row * N + 2 * (N // 8) + c).to(tl.float32)
    x3 = tl.load(X + row * N + 3 * (N // 8) + c).to(tl.float32)
    x4 = tl.load(X + row * N + 4 * (N // 8) + c).to(tl.float32)
    x5 = tl.load(X + row * N + 5 * (N // 8) + c).to(tl.float32)
    x6 = tl.load(X + row * N + 6 * (N // 8) + c).to(tl.float32)
    x7 = tl.load(X + row * N + 7 * (N // 8) + c).to(tl.float32)
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
        / N
        + E
    )
    w0 = tl.load(W + 0 * (N // 8) + c).to(tl.float32)
    tl.store(Y + row * N + 0 * (N // 8) + c, x0 * inv * (1.0 + w0))
    w1 = tl.load(W + 1 * (N // 8) + c).to(tl.float32)
    tl.store(Y + row * N + 1 * (N // 8) + c, x1 * inv * (1.0 + w1))
    w2 = tl.load(W + 2 * (N // 8) + c).to(tl.float32)
    tl.store(Y + row * N + 2 * (N // 8) + c, x2 * inv * (1.0 + w2))
    w3 = tl.load(W + 3 * (N // 8) + c).to(tl.float32)
    tl.store(Y + row * N + 3 * (N // 8) + c, x3 * inv * (1.0 + w3))
    w4 = tl.load(W + 4 * (N // 8) + c).to(tl.float32)
    tl.store(Y + row * N + 4 * (N // 8) + c, x4 * inv * (1.0 + w4))
    w5 = tl.load(W + 5 * (N // 8) + c).to(tl.float32)
    tl.store(Y + row * N + 5 * (N // 8) + c, x5 * inv * (1.0 + w5))
    w6 = tl.load(W + 6 * (N // 8) + c).to(tl.float32)
    tl.store(Y + row * N + 6 * (N // 8) + c, x6 * inv * (1.0 + w6))
    w7 = tl.load(W + 7 * (N // 8) + c).to(tl.float32)
    tl.store(Y + row * N + 7 * (N // 8) + c, x7 * inv * (1.0 + w7))


@triton.jit(do_not_specialize=["X", "W", "Y"])
def _small_dense(X, W, Y, N: tl.constexpr, E: tl.constexpr, B: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    c = tl.arange(0, B).to(tl.int64)
    x = tl.load(X + row * N + c, c < N, other=0).to(tl.float32)
    w = tl.load(W + c, c < N, other=0).to(tl.float32)
    v = tl.sum(x * x, 0) / N
    y = x * tl.rsqrt(v + E) * (1.0 + w)
    tl.store(Y + row * N + c, y, c < N)


@triton.jit
def _dense(X, W, Y, N: tl.constexpr, E: tl.constexpr, B: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    c = tl.arange(0, B).to(tl.int64)
    x = tl.load(X + row * N + c, c < N, other=0).to(tl.float32)
    w = tl.load(W + c, c < N, other=0).to(tl.float32)
    v = tl.sum(x * x, 0) / N
    y = x * tl.rsqrt(v + E) * (1.0 + w)
    tl.store(Y + row * N + c, y, c < N)


def gemma_rmsnorm(x, weight, eps):
    n = x.shape[-1]
    out = torch.empty_like(x, memory_format=torch.contiguous_format)
    m = x.numel() // n
    if (
        m >= 4096
        and 1 <= n <= 64
        and (not n & n - 1)
        and x.is_contiguous()
        and (weight.stride(0) == 1)
        and (x.dtype == weight.dtype)
        and (x.dtype in (torch.bfloat16, torch.float16))
    ):
        r = 2048 // n
        _narrow_tile.run(
            x,
            weight,
            out,
            n,
            eps,
            m,
            r,
            num_warps=4,
            enable_fp_fusion=False,
            grid=(triton.cdiv(m, r),),
            warmup=False,
        )
        return out
    if (
        m >= 128
        and 128 <= n <= 8192
        and (not n & n - 1)
        and (x.ndim == 2)
        and (x.stride(0) == 1)
        and (x.stride(1) >= m)
        and (weight.stride(0) == 1)
    ) and (
        x.dtype in (torch.bfloat16, torch.float16)
        or (x.dtype == torch.float32 and n >= 4096)
    ):
        r = 8 if x.dtype == torch.float32 else min(16, 32768 // n)
        _transposed.run(
            x,
            weight,
            out,
            n,
            eps,
            m,
            x.stride(1),
            r,
            num_warps=8 if n >= 2048 else 4,
            enable_fp_fusion=False,
            grid=(triton.cdiv(m, r),),
            warmup=False,
        )
        return out
    if (
        m >= 1024
        and 128 <= n <= 2048
        and (not n & n - 1)
        and (
            x.ndim == 2
            or (
                x.ndim > 2
                and all(
                    (
                        x.stride(d) == x.stride(d + 1) * x.shape[d + 1]
                        for d in range(x.ndim - 2)
                    )
                )
            )
        )
        and (x.stride(-1) == 1)
        and (not x.is_contiguous())
        and (weight.stride(0) == 1)
        and (
            x.dtype in (torch.bfloat16, torch.float16)
            and weight.dtype in (torch.bfloat16, torch.float16)
            or (
                x.dtype == torch.float32
                and weight.dtype
                in (torch.bfloat16, torch.float16, torch.float32)
            )
        )
    ):
        _rowfragment.run(
            x,
            weight,
            out,
            n,
            eps,
            1,
            x.stride(-2),
            num_warps=1,
            enable_fp_fusion=False,
            grid=(m,),
            warmup=False,
        )
        return out
    if (
        m >= 4096
        and 128 <= n <= 256
        and (not n & n - 1)
        and x.is_contiguous()
        and (weight.stride(0) == 1)
        and (x.dtype == torch.bfloat16)
        and (weight.dtype == torch.bfloat16)
    ):
        r = 2048 // n
        _narrow_tile.run(
            x,
            weight,
            out,
            n,
            eps,
            m,
            r,
            num_warps=4,
            enable_fp_fusion=False,
            grid=(triton.cdiv(m, r),),
            warmup=False,
        )
        return out
    if m:
        b = triton.next_power_of_2(n)
        fast = m >= 512 and x.is_contiguous() and (weight.stride(0) == 1)
        if (
            (m >= 1024 or (n == 1024 and m >= 512))
            and fast
            and (128 <= n <= 2048)
            and (b == n)
            and (x.dtype in (torch.bfloat16, torch.float16))
            and (weight.dtype in (torch.bfloat16, torch.float16))
        ):
            _rowfragment.run(
                x,
                weight,
                out,
                n,
                eps,
                1,
                n,
                num_warps=1,
                enable_fp_fusion=False,
                grid=(m,),
                warmup=False,
            )
            return out
        if fast and 1024 < n <= 8192 and (b == n):
            _parts8.run(
                x,
                weight,
                out,
                n,
                eps,
                num_warps=4 if b <= 4096 else 8,
                enable_fp_fusion=False,
                grid=(m,),
                warmup=False,
            )
        elif x.is_contiguous() and weight.stride(0) == 1:
            (
                _small_dense if m <= 64 and n <= 8192 and (b == n) else _dense
            ).run(
                x,
                weight,
                out,
                n,
                eps,
                b,
                num_warps=4 if b <= 4096 else 8,
                enable_fp_fusion=False,
                grid=(m,),
                warmup=False,
            )
        else:
            if (
                x.ndim > 2
                and x.stride(-1) > 1
                and (1024 <= x.shape[-1] <= 8192)
                and (not x.shape[-1] & x.shape[-1] - 1)
                and (x.numel() // x.shape[-1] >= 1024)
                and (weight.stride(0) == 1)
                and (
                    not all(
                        (
                            x.stride(d) == x.stride(d + 1) * x.shape[d + 1]
                            for d in range(x.ndim - 2)
                        )
                    )
                )
            ):
                dense = True
                expected = 1
                for stride, size in sorted(zip(x.stride()[:-1], x.shape[:-1])):
                    if size > 1:
                        if stride != expected:
                            dense = False
                        expected *= size
                if dense and x.stride(-1) >= expected:
                    r = min(16, 32768 // n)
                    if n >= 4096 and m >= 2048:
                        _permuted_chunked.run(
                            x,
                            weight,
                            out,
                            n,
                            eps,
                            m,
                            x.stride(-1),
                            tuple(x.shape),
                            tuple(x.stride()),
                            num_warps=4,
                            enable_fp_fusion=False,
                            grid=(triton.cdiv(m, 16),),
                            warmup=False,
                        )
                    else:
                        _permuted_transposed.run(
                            x,
                            weight,
                            out,
                            n,
                            eps,
                            m,
                            x.stride(-1),
                            r,
                            tuple(x.shape),
                            tuple(x.stride()),
                            num_warps=8 if n >= 2048 else 4,
                            enable_fp_fusion=False,
                            grid=(triton.cdiv(m, r),),
                            warmup=False,
                        )
                    return out
            _norm.run(
                x,
                weight,
                out,
                tuple(x.shape),
                tuple(x.stride()),
                weight.stride(0),
                n,
                eps,
                b,
                num_warps=4 if b <= 4096 else 8,
                enable_fp_fusion=False,
                grid=(m,),
                warmup=False,
            )
    return out


__all__ = ["gemma_rmsnorm"]
