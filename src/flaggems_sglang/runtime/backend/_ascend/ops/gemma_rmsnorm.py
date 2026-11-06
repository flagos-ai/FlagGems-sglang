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


@triton.jit(
    do_not_specialize=["X", "W", "Y"],
    do_not_specialize_on_alignment=["X", "W", "Y"],
)
def _pers_mapped(
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
    if N <= 2048:
        BR: tl.constexpr = (
            4096 // N
            if N < 128
            else (
                8192 // N if M > 65535 and N <= 1024 else 8 if N <= 1024 else 4
            )
        )
        c = tl.arange(0, N)
        scale = 1.0 + tl.load(W + c).to(tl.float32)
        for group in tl.range(
            tl.program_id(0).to(tl.int64),
            tl.cdiv(M, BR),
            tl.num_programs(0),
            num_stages=2,
        ):
            r = group * BR + tl.arange(0, BR)
            destination = tl.full((BR,), 0, tl.int64)
            factor = 1
            for axis in tl.static_range(len(SH) - 2, -1, -1):
                if SH[axis] > 1:
                    destination += r // ST[axis] % SH[axis] * factor
                factor *= SH[axis]
            offset = destination[:, None] * N + c[None, :]
            if M % BR == 0:
                x = tl.load(X + r[:, None] * S + c[None, :]).to(tl.float32)
            else:
                x = tl.load(
                    X + r[:, None] * S + c[None, :], r[:, None] < M, other=0
                ).to(tl.float32)
            variance = tl.sum(x * x, 1) / N
            y = x * tl.rsqrt(variance[:, None] + E) * scale[None, :]
            if M % BR == 0:
                tl.store(Y + offset, y)
            else:
                tl.store(Y + offset, y, r[:, None] < M)
    else:
        c = tl.arange(0, N).to(tl.int64)
        w = tl.load(W + c).to(tl.float32)
        for row in tl.range(
            tl.program_id(0).to(tl.int64),
            M,
            tl.num_programs(0),
            num_stages=2,
            loop_unroll_factor=2,
        ):
            destination = tl.full((), 0, tl.int64)
            factor = 1
            for axis in tl.static_range(len(SH) - 2, -1, -1):
                if SH[axis] > 1:
                    destination += row // ST[axis] % SH[axis] * factor
                factor *= SH[axis]
            o = destination * N + c
            x = tl.load(X + row * S + c).to(tl.float32)
            v = tl.sum(x * x, 0) / N
            tl.store(Y + o, x * tl.rsqrt(v + E) * (1.0 + w))


@triton.jit(
    do_not_specialize=["X", "W", "Y"],
    do_not_specialize_on_alignment=["X", "W", "Y"],
)
def _padded_persistent(
    X,
    W,
    Y,
    N: tl.constexpr,
    E: tl.constexpr,
    M: tl.constexpr,
    S: tl.constexpr,
    B: tl.constexpr,
    R: tl.constexpr,
):
    c = tl.arange(0, B).to(tl.int64)
    w = 1.0 + tl.load(W + c, c < N, other=0).to(tl.float32)
    for group in tl.range(
        tl.program_id(0).to(tl.int64),
        tl.cdiv(M, R),
        tl.num_programs(0),
        num_stages=2,
    ):
        r = group * R + tl.arange(0, R)
        mask = (r[:, None] < M) & (c[None, :] < N)
        x = tl.load(X + r[:, None] * S + c[None, :], mask, other=0).to(
            tl.float32
        )
        inv = tl.rsqrt(tl.sum(x * x, 1) / N + E)
        tl.store(
            Y + r[:, None] * N + c[None, :],
            x * inv[:, None] * w[None, :],
            mask,
        )


@triton.jit(
    do_not_specialize=["X", "W", "Y"],
    do_not_specialize_on_alignment=["X", "W", "Y"],
)
def _triplet_persistent(
    X, W, Y, N: tl.constexpr, E: tl.constexpr, M: tl.constexpr
):
    R: tl.constexpr = 6144 // N
    c = tl.arange(0, N // 3)
    w0 = 1.0 + tl.load(W + c).to(tl.float32)
    w1 = 1.0 + tl.load(W + c + N // 3).to(tl.float32)
    w2 = 1.0 + tl.load(W + c + 2 * (N // 3)).to(tl.float32)
    for group in tl.range(
        tl.program_id(0).to(tl.int64),
        tl.cdiv(M, R),
        tl.num_programs(0),
        num_stages=2,
    ):
        row = group * R + tl.arange(0, R)
        offset = row[:, None] * N + c[None, :]
        if M % R == 0:
            x0 = tl.load(X + offset).to(tl.float32)
            x1 = tl.load(X + offset + N // 3).to(tl.float32)
            x2 = tl.load(X + offset + 2 * (N // 3)).to(tl.float32)
        else:
            x0 = tl.load(X + offset, row[:, None] < M, other=0).to(tl.float32)
            x1 = tl.load(X + offset + N // 3, row[:, None] < M, other=0).to(
                tl.float32
            )
            x2 = tl.load(
                X + offset + 2 * (N // 3), row[:, None] < M, other=0
            ).to(tl.float32)
        inv = tl.rsqrt(tl.sum(x0 * x0 + x1 * x1 + x2 * x2, 1) / N + E)
        y0 = x0 * inv[:, None] * w0[None, :]
        y1 = x1 * inv[:, None] * w1[None, :]
        y2 = x2 * inv[:, None] * w2[None, :]
        if M % R == 0:
            tl.store(Y + offset, y0)
            tl.store(Y + offset + N // 3, y1)
            tl.store(Y + offset + 2 * (N // 3), y2)
        else:
            tl.store(Y + offset, y0, row[:, None] < M)
            tl.store(Y + offset + N // 3, y1, row[:, None] < M)
            tl.store(Y + offset + 2 * (N // 3), y2, row[:, None] < M)


@triton.jit(
    do_not_specialize=["X", "W", "Y"],
    do_not_specialize_on_alignment=["X", "W", "Y"],
)
def _pers(
    X, W, Y, N: tl.constexpr, E: tl.constexpr, M: tl.constexpr, S: tl.constexpr
):
    if N <= 2048:
        BR: tl.constexpr = (
            4096 // N
            if N < 128
            else (
                8192 // N if M > 65535 and N <= 1024 else 8 if N <= 1024 else 4
            )
        )
        c = tl.arange(0, N)
        scale = 1.0 + tl.load(W + c).to(tl.float32)
        for group in tl.range(
            tl.program_id(0).to(tl.int64),
            tl.cdiv(M, BR),
            tl.num_programs(0),
            num_stages=2,
        ):
            r = group * BR + tl.arange(0, BR)
            offset = r[:, None] * N + c[None, :]
            if M % BR == 0:
                x = tl.load(X + r[:, None] * S + c[None, :]).to(tl.float32)
            else:
                x = tl.load(
                    X + r[:, None] * S + c[None, :], r[:, None] < M, other=0
                ).to(tl.float32)
            variance = tl.sum(x * x, 1) / N
            y = x * tl.rsqrt(variance[:, None] + E) * scale[None, :]
            if M % BR == 0:
                tl.store(Y + offset, y)
            else:
                tl.store(Y + offset, y, r[:, None] < M)
    else:
        c = tl.arange(0, N).to(tl.int64)
        w = tl.load(W + c).to(tl.float32)
        for row in tl.range(
            tl.program_id(0).to(tl.int64),
            M,
            tl.num_programs(0),
            num_stages=2,
            loop_unroll_factor=2,
        ):
            o = row * N + c
            x = tl.load(X + row * S + c).to(tl.float32)
            v = tl.sum(x * x, 0) / N
            tl.store(Y + o, x * tl.rsqrt(v + E) * (1.0 + w))


@triton.jit(
    do_not_specialize=["X", "W", "Y"],
    do_not_specialize_on_alignment=["X", "W", "Y"],
)
def _triplet(X, W, Y, N: tl.constexpr, E: tl.constexpr):
    row = tl.program_id(0).to(tl.int64) * 2 + tl.arange(0, 2)
    c = tl.arange(0, N // 3)
    o = row[:, None] * N + c[None, :]
    x0 = tl.load(X + o + 0 * (N // 3)).to(tl.float32)
    x1 = tl.load(X + o + 1 * (N // 3)).to(tl.float32)
    x2 = tl.load(X + o + 2 * (N // 3)).to(tl.float32)
    inv = tl.rsqrt(tl.sum(x0 * x0 + x1 * x1 + x2 * x2, 1) / N + E)
    w = tl.load(W + c + 0 * (N // 3)).to(tl.float32)
    tl.store(Y + o + 0 * (N // 3), x0 * inv[:, None] * (1.0 + w[None, :]))
    w = tl.load(W + c + 1 * (N // 3)).to(tl.float32)
    tl.store(Y + o + 1 * (N // 3), x1 * inv[:, None] * (1.0 + w[None, :]))
    w = tl.load(W + c + 2 * (N // 3)).to(tl.float32)
    tl.store(Y + o + 2 * (N // 3), x2 * inv[:, None] * (1.0 + w[None, :]))


@triton.jit(
    do_not_specialize=["X", "W", "Y"],
    do_not_specialize_on_alignment=["X", "W", "Y"],
)
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
    if B == N and ST[-1] == 1 and (WS == 1):
        x = tl.load(X + base + c).to(tl.float32)
        w = tl.load(W + c).to(tl.float32)
        v = tl.sum(x * x, 0) / N
        y = x * tl.rsqrt(v + E) * (1.0 + w)
        tl.store(Y + row * N + c, y)
    else:
        x = tl.load(X + base + c * ST[-1], c < N, other=0).to(tl.float32)
        w = tl.load(W + c * WS, c < N, other=0).to(tl.float32)
        v = tl.sum(x * x, 0) / N
        y = x * tl.rsqrt(v + E) * (1.0 + w)
        tl.store(Y + row * N + c, y, c < N)


@triton.jit(
    do_not_specialize=["X", "W", "Y"],
    do_not_specialize_on_alignment=["X", "W", "Y"],
)
def _exact(X, W, Y, N: tl.constexpr, E: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    c = tl.arange(0, N).to(tl.int64)
    o = row * N + c
    x = tl.load(X + o).to(tl.float32)
    w = tl.load(W + c).to(tl.float32)
    v = tl.sum(x * x, 0) / N
    tl.store(Y + o, x * tl.rsqrt(v + E) * (1.0 + w))


@triton.jit(
    do_not_specialize=["X", "W", "Y"],
    do_not_specialize_on_alignment=["X", "W", "Y"],
)
def _many(
    X,
    W,
    Y,
    SH: tl.constexpr,
    ST: tl.constexpr,
    WS: tl.constexpr,
    N: tl.constexpr,
    E: tl.constexpr,
    B: tl.constexpr,
    M: tl.constexpr,
):
    for row in tl.range(
        tl.program_id(0).to(tl.int64), M, tl.num_programs(0), num_stages=2
    ):
        rem = row
        base = 0
        for a in tl.static_range(len(SH) - 2, -1, -1):
            base += rem % SH[a] * ST[a]
            rem = rem // SH[a]
        c = tl.arange(0, B).to(tl.int64)
        if B == N and ST[-1] == 1 and (WS == 1):
            x = tl.load(X + base + c).to(tl.float32)
            w = tl.load(W + c).to(tl.float32)
            v = tl.sum(x * x, 0) / N
            y = x * tl.rsqrt(v + E) * (1.0 + w)
            tl.store(Y + row * N + c, y)
        else:
            x = tl.load(X + base + c * ST[-1], c < N, other=0).to(tl.float32)
            w = tl.load(W + c * WS, c < N, other=0).to(tl.float32)
            v = tl.sum(x * x, 0) / N
            y = x * tl.rsqrt(v + E) * (1.0 + w)
            tl.store(Y + row * N + c, y, c < N)


def gemma_rmsnorm(x, weight, eps):
    n = x.shape[-1]
    m = x.numel() // n
    if (
        x.ndim > 2
        and x.stride(-1) > 1
        and (m >= 1024)
        and (2048 <= n <= 8192)
        and (not n & n - 1)
        and (weight.stride(0) == 1)
    ):
        dense = True
        expected = 1
        for stride, size in sorted(zip(x.stride()[:-1], x.shape[:-1])):
            if size > 1:
                if stride != expected:
                    dense = False
                expected *= size
        if dense and x.stride(-1) >= expected:
            if m >= 2048:
                order = tuple(
                    sorted(
                        range(x.ndim - 1),
                        key=lambda axis: x.stride(axis),
                        reverse=True,
                    )
                ) + (x.ndim - 1,)
                prepared = x.permute(order).reshape(m, n).contiguous()
                out = torch.empty_like(
                    x, memory_format=torch.contiguous_format
                )
                _pers_mapped.run(
                    prepared,
                    weight,
                    out,
                    n,
                    eps,
                    m,
                    n,
                    tuple(x.shape),
                    tuple(x.stride()),
                    num_warps=4 if n <= 4096 else 8,
                    num_stages=1,
                    enable_fp_fusion=False,
                    grid=(40,),
                    warmup=False,
                )
                return out
            else:
                x = x.contiguous()
    if (
        0 < m < 1024
        and (not n & n - 1)
        and x.is_contiguous()
        and (weight.stride(0) == 1)
    ):
        out = torch.empty_like(x)
        _exact.run(
            x,
            weight,
            out,
            n,
            eps,
            num_warps=4 if n <= 4096 else 8,
            enable_fp_fusion=False,
            grid=(m,),
            warmup=False,
        )
        return out
    if (
        x.ndim == 2
        and x.stride(0) == 1
        and (x.shape[0] >= 1024 or (x.shape[0] >= 512 and x.shape[1] >= 4096))
        and (128 <= x.shape[1] <= 8192)
        and (x.stride(1) >= x.shape[0])
        and (weight.stride(0) == 1)
        and (x.numel() >= 2097152)
    ):
        x = x.contiguous()
    shape = x.shape
    n = shape[-1]
    m = x.numel() // n
    if (
        m >= 1024
        and 128 <= n <= 8192
        and n & n - 1
        and (not (n % 3 == 0 and (not n // 3 & n // 3 - 1)))
        and (weight.stride(0) == 1)
        and (x.is_contiguous() or (x.ndim == 2 and x.stride(1) == 1))
    ):
        out = torch.empty_like(x, memory_format=torch.contiguous_format)
        b = triton.next_power_of_2(n)
        r = min(8, 8192 // b)
        _padded_persistent.run(
            x,
            weight,
            out,
            n,
            eps,
            m,
            n if x.is_contiguous() else x.stride(0),
            b,
            r,
            num_warps=4 if b <= 4096 else 8,
            num_stages=1,
            enable_fp_fusion=False,
            grid=(40,),
            warmup=False,
        )
        return out
    if (
        m >= 1024
        and (
            n >= 128
            or (
                1 <= n <= 64
                and (x.is_contiguous() or n >= 16)
                and (x.dtype == weight.dtype)
                and (x.dtype in (torch.bfloat16, torch.float16))
            )
        )
        and (not n & n - 1)
        and (
            x.is_contiguous()
            or (
                (
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
                and x.stride(-1) == 1
            )
        )
        and (weight.stride(0) == 1)
    ):
        out = torch.empty_like(x, memory_format=torch.contiguous_format)
        _pers.run(
            x,
            weight,
            out,
            n,
            eps,
            m,
            n if x.is_contiguous() else x.stride(-2),
            num_warps=4 if n <= 4096 else 8,
            num_stages=1,
            enable_fp_fusion=False,
            grid=(min(triton.cdiv(m, 4096 // n), 40) if n < 128 else 40,),
            warmup=False,
        )
        return out
    if (
        m >= 32768
        and 3 <= n <= 3072
        and (n % 3 == 0)
        and (not n // 3 & n // 3 - 1)
        and x.is_contiguous()
        and (weight.stride(0) == 1)
        and (x.dtype == weight.dtype)
        and (x.dtype in (torch.bfloat16, torch.float16))
    ):
        out = torch.empty_like(x)
        _triplet_persistent.run(
            x,
            weight,
            out,
            n,
            eps,
            m,
            num_warps=4,
            num_stages=1,
            enable_fp_fusion=False,
            grid=(40,),
            warmup=False,
        )
        return out
    if (
        m
        and m <= 65535
        and (n >= 3)
        and (n <= 3072)
        and (n % 3 == 0)
        and (not n // 3 & n // 3 - 1)
        and (m % 2 == 0)
        and x.is_contiguous()
        and (weight.stride(0) == 1)
    ):
        out = torch.empty_like(x, memory_format=torch.contiguous_format)
        _triplet.run(
            x,
            weight,
            out,
            n,
            eps,
            num_warps=4 if n <= 4096 else 8,
            enable_fp_fusion=False,
            grid=(m // 2,),
            warmup=False,
        )
        return out
    if n >= 8192 and (not x.is_contiguous() or weight.stride(0) != 1):
        x = x.contiguous()
        weight = weight.contiguous()
    if m > 65535:
        out = torch.empty(shape, dtype=x.dtype, device=x.device)
        b = triton.next_power_of_2(n)
        _many.run(
            x,
            weight,
            out,
            shape,
            tuple(x.stride()),
            weight.stride(0),
            n,
            eps,
            b,
            m,
            num_warps=4 if b <= 4096 else 8,
            num_stages=1,
            enable_fp_fusion=False,
            grid=(40,),
            warmup=False,
        )
        return out
    if x.is_contiguous() and weight.stride(0) == 1:
        out = torch.empty_like(x)
        if m:
            if not n & n - 1:
                _exact.run(
                    x,
                    weight,
                    out,
                    n,
                    eps,
                    num_warps=4 if n <= 4096 else 8,
                    enable_fp_fusion=False,
                    grid=(m,),
                    warmup=False,
                )
                return out
            b = triton.next_power_of_2(n)
            _norm.run(
                x,
                weight,
                out,
                shape,
                tuple(x.stride()),
                1,
                n,
                eps,
                b,
                num_warps=4 if b <= 4096 else 8,
                enable_fp_fusion=False,
                grid=(m,),
                warmup=False,
            )
        return out
    out = torch.empty(shape, dtype=x.dtype, device=x.device)
    if m:
        b = triton.next_power_of_2(n)
        _norm.run(
            x,
            weight,
            out,
            shape,
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
