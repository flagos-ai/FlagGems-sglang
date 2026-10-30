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
def _permuted_chunked_pair(
    X,
    W,
    Y,
    N: tl.constexpr,
    E: tl.constexpr,
    M: tl.constexpr,
    S: tl.constexpr,
    SH: tl.constexpr,
    ST: tl.constexpr,
    C: tl.constexpr,
):
    r = tl.program_id(0).to(tl.int64) * 16 + tl.arange(0, 16)
    c = tl.arange(0, C).to(tl.int64)
    destination = tl.full((16,), 0, tl.int64)
    factor = 1
    for axis in tl.static_range(len(SH) - 2, -1, -1):
        if SH[axis] > 1:
            destination += r // ST[axis] % SH[axis] * factor
        factor *= SH[axis]
    total = tl.full((16,), 0, tl.float32)
    for start in range(0, N, 2 * C):
        col = start + c
        x0 = tl.load(
            X + col[:, None] * S + r[None, :], r[None, :] < M, other=0
        ).to(tl.float32)
        x1 = tl.load(
            X + (col[:, None] + C) * S + r[None, :], r[None, :] < M, other=0
        ).to(tl.float32)
        total += tl.sum(x0 * x0 + x1 * x1, 0)
    inv = tl.rsqrt(total / N + E)
    for start in range(0, N, C):
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
    C: tl.constexpr,
):
    r = tl.program_id(0).to(tl.int64) * 16 + tl.arange(0, 16)
    c = tl.arange(0, C).to(tl.int64)
    destination = tl.full((16,), 0, tl.int64)
    factor = 1
    for axis in tl.static_range(len(SH) - 2, -1, -1):
        if SH[axis] > 1:
            destination += r // ST[axis] % SH[axis] * factor
        factor *= SH[axis]
    total = tl.full((16,), 0, tl.float32)
    for start in range(0, N, C):
        col = start + c
        x = tl.load(
            X + col[:, None] * S + r[None, :], r[None, :] < M, other=0
        ).to(tl.float32)
        total += tl.sum(x * x, 0)
    inv = tl.rsqrt(total / N + E)
    for start in range(0, N, C):
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
def _row_tile(
    X,
    W,
    Y,
    N: tl.constexpr,
    E: tl.constexpr,
    M: tl.constexpr,
    BR: tl.constexpr,
    S: tl.constexpr,
    B: tl.constexpr,
):
    r = tl.program_id(0).to(tl.int64) * BR + tl.arange(0, BR)
    c = tl.arange(0, B)
    valid = (r[:, None] < M) & (c[None, :] < N)
    x = tl.load(X + r[:, None] * S + c[None, :], valid, other=0).to(tl.float32)
    v = tl.sum(x * x, 1) / N
    normalized = x * tl.rsqrt(v[:, None] + E)
    w = tl.load(W + c, c < N, other=0).to(tl.float32)
    y = normalized * (1.0 + w[None, :])
    tl.store(Y + r[:, None] * N + c[None, :], y, valid)


@triton.jit
def _exact(X, W, Y, N: tl.constexpr, E: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    c = tl.arange(0, N).to(tl.int32 if 2048 <= N and N <= 65536 else tl.int64)
    x = tl.load(X + row * N + c).to(tl.float32)
    w = tl.load(W + c).to(tl.float32)
    v = tl.sum(x * x, 0) / N
    tl.store(Y + row * N + c, x * tl.rsqrt(v + E) * (1.0 + w))


@triton.jit
def _dense(X, W, Y, N: tl.constexpr, E: tl.constexpr, B: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    c = tl.arange(0, B).to(tl.int64)
    x = tl.load(X + row * N + c, c < N, other=0).to(tl.float32)
    w = tl.load(W + c, c < N, other=0).to(tl.float32)
    v = tl.sum(x * x, 0) / N
    y = x * tl.rsqrt(v + E) * (1.0 + w)
    tl.store(Y + row * N + c, y, c < N)


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
def _reordered_sum(X, W, Y, N: tl.constexpr, E: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    c = tl.arange(0, N).to(tl.int64)
    x = tl.load(X + row * N + c).to(tl.float32)
    squares = tl.reshape(x * x, (N // 8, 8), can_reorder=False)
    partial = tl.sum(squares, 1)
    variance = tl.sum(partial, 0) / N
    inv = tl.rsqrt(variance + E)
    w = tl.load(W + c).to(tl.float32)
    tl.store(Y + row * N + c, x * inv * (1.0 + w))


def gemma_rmsnorm(x, weight, eps):
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
            n = x.shape[-1]
            m = x.numel() // n
            r = min(
                8 if n <= 1024 else 16,
                65536 // (n * max(x.element_size(), weight.element_size())),
            )
            out = torch.empty_like(x, memory_format=torch.contiguous_format)
            if n >= 4096 and m >= 2048:
                (
                    _permuted_chunked_pair
                    if m >= 8192 and x.dtype in (torch.float16, torch.bfloat16)
                    else _permuted_chunked
                ).run(
                    x,
                    weight,
                    out,
                    n,
                    eps,
                    m,
                    x.stride(-1),
                    tuple(x.shape),
                    tuple(x.stride()),
                    512 if x.dtype in (torch.float16, torch.bfloat16) else 256,
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
    if (
        x.ndim > 2
        and x.stride(-1) > 1
        and (4096 <= x.shape[-1] <= 8192)
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
        x = x.contiguous()
    n = x.shape[-1]
    if x.is_contiguous() and weight.stride(0) == 1:
        out = torch.empty_like(x)
        m = x.numel() // n
        if m >= 8192 and 128 <= n <= 512 and (not n & n - 1 or m >= 16384):
            br = (4096 if not n & n - 1 else 2048) // triton.next_power_of_2(n)
            _row_tile.run(
                x,
                weight,
                out,
                n,
                eps,
                m,
                br,
                n,
                triton.next_power_of_2(n),
                num_warps=4,
                enable_fp_fusion=False,
                grid=(triton.cdiv(m, br),),
                warmup=False,
            )
            return out
        if m:
            if not n & n - 1:
                (_reordered_sum if m >= 1024 and n >= 8192 else _exact).run(
                    x,
                    weight,
                    out,
                    n,
                    eps,
                    num_warps=2 if n <= 2048 else 4 if n <= 4096 else 8,
                    enable_fp_fusion=False,
                    grid=(m,),
                    warmup=False,
                )
                return out
            b = triton.next_power_of_2(n)
            _dense.run(
                x,
                weight,
                out,
                n,
                eps,
                b,
                num_warps=2 if b <= 2048 else 4 if b <= 4096 else 8,
                enable_fp_fusion=False,
                grid=(m,),
                warmup=False,
            )
        return out
    out = torch.empty_like(x, memory_format=torch.contiguous_format)
    m = x.numel() // n
    if (
        m >= 128
        and 128 <= n <= 8192
        and (not n & n - 1)
        and (x.ndim >= 2)
        and (x.stride(-2) == 1)
        and (x.stride(-1) >= m)
        and (weight.stride(0) == 1)
        and (x.ndim == 2 or m * n >= 1048576)
        and (
            x.ndim == 2
            or all(
                (
                    x.stride(d) == x.stride(d + 1) * x.shape[d + 1]
                    for d in range(x.ndim - 2)
                )
            )
        )
    ):
        r = min(
            8 if n <= 1024 else 16,
            65536 // (n * max(x.element_size(), weight.element_size())),
        )
        if n >= 4096 and m >= 2048:
            (
                _permuted_chunked_pair
                if m >= 8192 and x.dtype in (torch.float16, torch.bfloat16)
                else _permuted_chunked
            ).run(
                x,
                weight,
                out,
                n,
                eps,
                m,
                x.stride(-1),
                tuple(x.shape),
                tuple(x.stride()),
                512 if x.dtype in (torch.float16, torch.bfloat16) else 256,
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
                x.stride(-1),
                r,
                num_warps=8 if n >= 2048 else 4,
                enable_fp_fusion=False,
                grid=(triton.cdiv(m, r),),
                warmup=False,
            )
        return out
    if m:
        if (
            m >= 8192
            and 128 <= n <= 512
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
            and (weight.stride(0) == 1)
            and (not n & n - 1 or m >= 16384)
        ):
            br = (4096 if not n & n - 1 else 2048) // triton.next_power_of_2(n)
            _row_tile.run(
                x,
                weight,
                out,
                n,
                eps,
                m,
                br,
                x.stride(-2),
                triton.next_power_of_2(n),
                num_warps=4,
                enable_fp_fusion=False,
                grid=(triton.cdiv(m, br),),
                warmup=False,
            )
            return out
        b = triton.next_power_of_2(n)
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
            num_warps=2 if b <= 2048 else 4 if b <= 4096 else 8,
            enable_fp_fusion=False,
            grid=(m,),
            warmup=False,
        )
    return out


__all__ = ["gemma_rmsnorm"]
