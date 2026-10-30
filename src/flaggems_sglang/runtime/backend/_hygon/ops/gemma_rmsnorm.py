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
        col = start + c
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
def _small_tile(
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


@triton.jit(
    do_not_specialize=["X", "W", "Y"],
    do_not_specialize_on_alignment=["X", "W", "Y"],
)
def _bounded(X, W, Y, N: tl.constexpr, E: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    tl.assume(row >= 0)
    tl.assume(row < 512)
    c = tl.arange(0, N)
    xp = X + row * N
    yp = Y + row * N
    x = tl.load(xp + c).to(tl.float32)
    w = tl.load(W + c).to(tl.float32)
    v = tl.sum(x * x, 0) / N
    tl.store(yp + c, x * tl.rsqrt(v + E) * (1.0 + w))


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
    m = x.numel() // n
    if (
        0 < m <= 512
        and 128 <= n <= 8192
        and (not n & n - 1)
        and x.is_contiguous()
        and (weight.stride(0) == 1)
        and (x.dtype in (torch.bfloat16, torch.float16, torch.float32))
        and (weight.dtype in (torch.bfloat16, torch.float16, torch.float32))
    ):
        out = torch.empty_like(x)
        _bounded.run(
            x,
            weight,
            out,
            n,
            eps,
            num_warps=4,
            enable_fp_fusion=False,
            grid=(m,),
            warmup=False,
        )
        return out
    out = torch.empty_like(x, memory_format=torch.contiguous_format)
    if (
        m >= 65536
        and 1 <= n <= 64
        and (not n & n - 1)
        and x.is_contiguous()
        and (weight.stride(0) == 1)
        and (x.dtype == weight.dtype)
        and (x.dtype in (torch.bfloat16, torch.float16))
    ):
        r = 2048 // n
        _small_tile.run(
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
    ):
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
    if m:
        b = triton.next_power_of_2(n)
        if x.is_contiguous() and weight.stride(0) == 1:
            _dense.run(
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
            return out
        if (
            x.ndim > 2
            and x.stride(-1) > 1
            and (2048 <= x.shape[-1] <= 8192)
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
