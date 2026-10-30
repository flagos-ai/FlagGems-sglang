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
def _narrow_persistent(
    X,
    W,
    Y,
    SH: tl.constexpr,
    ST: tl.constexpr,
    WS: tl.constexpr,
    N: tl.constexpr,
    E: tl.constexpr,
    M: tl.constexpr,
):
    c = tl.arange(0, 128).to(tl.int64)
    safe = tl.minimum(c, N - 1)
    w = tl.load(W + safe * WS).to(tl.float32)
    for row in tl.range(
        tl.program_id(0).to(tl.int64), M, tl.num_programs(0), num_stages=2
    ):
        rem = row
        base = tl.full((), 0, tl.int64)
        for axis in tl.static_range(len(SH) - 2, -1, -1):
            base += rem % SH[axis] * ST[axis]
            rem = rem // SH[axis]
        value = tl.load(X + base + safe * ST[-1]).to(tl.float32)
        value = tl.where(c < N, value, 0.0)
        inv = tl.rsqrt(tl.sum(value * value, 0) / N + E)
        tl.store(Y + row * N + c, value * inv * (1 + w), c < N)


@triton.jit
def _wide(
    X,
    W,
    Y,
    SH: tl.constexpr,
    ST: tl.constexpr,
    WS: tl.constexpr,
    N: tl.constexpr,
    E: tl.constexpr,
    C: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    rem = row
    base = 0
    for a in tl.static_range(len(SH) - 2, -1, -1):
        base += rem % SH[a] * ST[a]
        rem = rem // SH[a]
    c = tl.arange(0, C).to(tl.int64)
    acc = tl.full((C,), 0, tl.float32)
    for start in range(tl.cdiv(N, C)):
        col = start.to(tl.int64) * C + c
        value = tl.load(X + base + tl.minimum(col, N - 1) * ST[-1]).to(
            tl.float32
        )
        value = tl.where(col < N, value, 0.0)
        acc += value * value
    inv = tl.rsqrt(tl.sum(acc, 0) / N + E)
    for start in range(tl.cdiv(N, C)):
        col = start.to(tl.int64) * C + c
        safe = tl.minimum(col, N - 1)
        value = tl.load(X + base + safe * ST[-1]).to(tl.float32)
        w = tl.load(W + safe * WS).to(tl.float32)
        tl.store(Y + row * N + col, value * inv * (1.0 + w), col < N)


@triton.jit
def _padded_rows(
    X,
    W,
    Y,
    N: tl.constexpr,
    E: tl.constexpr,
    R: tl.constexpr,
    BR: tl.constexpr,
    S: tl.constexpr,
    D: tl.constexpr,
):
    c = tl.arange(0, N)
    w = tl.load(
        W + tl.zeros((BR, 1), tl.int32) + c[None, :],
        offset_state_policy="continuous",
    ).to(tl.float32)
    for grp in tl.range(
        tl.program_id(0), R // BR, tl.num_programs(0), num_stages=2
    ):
        base = grp.to(tl.int64) * BR * N
        r = tl.arange(0, BR)
        off = r[:, None] * N + c[None, :]
        if S == N:
            x = tl.load(X + base + off, offset_state_policy="continuous").to(
                tl.float32
            )
        else:
            x = tl.load(
                X + grp.to(tl.int64) * BR * S + r[:, None] * S + c[None, :]
            ).to(tl.float32)
        v = tl.sum(x * x, 1) / D
        y = x * tl.rsqrt(v[:, None] + E) * (1.0 + w)
        tl.store(Y + base + off, y, offset_state_policy="continuous")


@triton.jit
def _rows_direct_weight(
    X,
    W,
    Y,
    N: tl.constexpr,
    E: tl.constexpr,
    R: tl.constexpr,
    BR: tl.constexpr,
    S: tl.constexpr,
):
    c = tl.arange(0, N)
    w = tl.load(
        W + tl.zeros((BR, 1), tl.int32) + c[None, :],
        offset_state_policy="continuous",
    ).to(tl.float32)
    for grp in tl.range(
        tl.program_id(0), R // BR, tl.num_programs(0), num_stages=2
    ):
        base = grp.to(tl.int64) * BR * N
        r = tl.arange(0, BR)
        off = r[:, None] * N + c[None, :]
        if S == N:
            x = tl.load(X + base + off, offset_state_policy="continuous").to(
                tl.float32
            )
        else:
            x = tl.load(
                X + grp.to(tl.int64) * BR * S + r[:, None] * S + c[None, :]
            ).to(tl.float32)
        v = tl.sum(x * x, 1) / N
        y = x * tl.rsqrt(v[:, None] + E) * (1.0 + w)
        tl.store(Y + base + off, y, offset_state_policy="continuous")


@triton.jit
def _padded_tail(
    X,
    W,
    Y,
    N: tl.constexpr,
    E: tl.constexpr,
    S: tl.constexpr,
    START: tl.constexpr,
    D: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64) + START
    c = tl.arange(0, N).to(tl.int64)
    x = tl.load(X + row * S + c, offset_state_policy="continuous").to(
        tl.float32
    )
    w = tl.load(W + c, offset_state_policy="continuous").to(tl.float32)
    v = tl.sum(x * x, 0) / D
    tl.store(
        Y + row * N + c,
        x * tl.rsqrt(v + E) * (1.0 + w),
        offset_state_policy="continuous",
    )


@triton.jit
def _tail(
    X,
    W,
    Y,
    N: tl.constexpr,
    E: tl.constexpr,
    S: tl.constexpr,
    START: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64) + START
    c = tl.arange(0, N).to(tl.int64)
    x = tl.load(X + row * S + c, offset_state_policy="continuous").to(
        tl.float32
    )
    w = tl.load(W + c, offset_state_policy="continuous").to(tl.float32)
    v = tl.sum(x * x, 0) / N
    tl.store(
        Y + row * N + c,
        x * tl.rsqrt(v + E) * (1.0 + w),
        offset_state_policy="continuous",
    )


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
    if B == N and ST[-1] == 1 and (WS == 1):
        x = tl.load(X + base + c, offset_state_policy="continuous").to(
            tl.float32
        )
        w = tl.load(W + c, offset_state_policy="continuous").to(tl.float32)
        v = tl.sum(x * x, 0) / N
        y = x * tl.rsqrt(v + E) * (1.0 + w)
        tl.store(Y + row * N + c, y, offset_state_policy="continuous")
    else:
        x = tl.where(
            c < N,
            tl.load(X + base + tl.minimum(c, N - 1) * ST[-1]).to(tl.float32),
            0.0,
        )
        w = tl.load(W + tl.minimum(c, N - 1) * WS).to(tl.float32)
        v = tl.sum(x * x, 0) / N
        y = x * tl.rsqrt(v + E) * (1.0 + w)
        tl.store(Y + row * N + c, y, c < N)


@triton.jit
def _rows(
    X,
    W,
    Y,
    N: tl.constexpr,
    E: tl.constexpr,
    R: tl.constexpr,
    BR: tl.constexpr,
):
    c = tl.arange(0, N)
    w = tl.load(W + c, offset_state_policy="continuous").to(tl.float32)
    for grp in tl.range(
        tl.program_id(0), R // BR, tl.num_programs(0), num_stages=2
    ):
        base = grp.to(tl.int64) * BR * N
        r = tl.arange(0, BR)
        off = r[:, None] * N + c[None, :]
        x = tl.load(X + base + off, offset_state_policy="continuous").to(
            tl.float32
        )
        v = tl.sum(x * x, 1) / N
        y = x * tl.rsqrt(v[:, None] + E) * (1.0 + w[None, :])
        tl.store(Y + base + off, y, offset_state_policy="continuous")


def gemma_rmsnorm(x, weight, eps):
    if (
        x.ndim > 2
        and x.stride(-1) >= x.numel() // x.shape[-1]
        and (weight.stride(0) == 1)
        and (128 <= x.shape[-1] <= 8192)
        and (not x.shape[-1] & x.shape[-1] - 1)
        and (
            x.numel() // x.shape[-1] >= 4096
            or (x.numel() // x.shape[-1] >= 512 and x.shape[-1] >= 4096)
        )
    ):
        x = x.contiguous()
    n = x.shape[-1]
    m = x.numel() // n
    if (
        m >= 4096
        and 128 <= n <= 8192
        and n & n - 1
        and (weight.stride(0) == 1)
        and (x.is_contiguous() or (x.ndim == 2 and x.stride(-1) == 1))
    ):
        b = max(128, triton.next_power_of_2(n))
        packed = torch.nn.functional.pad(x.reshape(m, n), (0, b - n))
        packed_weight = torch.nn.functional.pad(weight, (0, b - n))
        result = torch.empty_like(packed)
        br = min(
            64, 131072 // (b * max(x.element_size(), weight.element_size()))
        )
        full = m // br * br
        _padded_rows.run(
            packed,
            packed_weight,
            result,
            b,
            eps,
            full,
            br,
            b,
            n,
            groups_per_cluster=br,
            isCloseUnrollControl=False,
            enable_fp_fusion=False,
            isCloseMemoryAsync=False,
            grid=(12,),
            warmup=False,
            vrf_budget=16 if b >= 4096 else 24,
        )
        if full < m:
            _padded_tail.run(
                packed,
                packed_weight,
                result,
                b,
                eps,
                b,
                full,
                n,
                num_warps=4 if b <= 4096 else 8,
                enable_fp_fusion=False,
                isCloseMemoryAsync=False,
                grid=(m - full,),
                warmup=False,
            )
        return result[:, :n].contiguous().reshape(x.shape)
    out = torch.empty(x.shape, dtype=x.dtype, device=x.device)
    if 0 < n < 128 and m >= 1024:
        _narrow_persistent.run(
            x,
            weight,
            out,
            tuple(x.shape),
            tuple(x.stride()),
            weight.stride(0),
            n,
            eps,
            m,
            num_warps=4,
            enable_fp_fusion=False,
            isCloseMemoryAsync=False,
            grid=(12,),
            warmup=False,
        )
        return out
    if n > 8192 and m:
        _wide.run(
            x,
            weight,
            out,
            tuple(x.shape),
            tuple(x.stride()),
            weight.stride(0),
            n,
            eps,
            1024,
            num_warps=4,
            enable_fp_fusion=False,
            isCloseMemoryAsync=False,
            grid=(m,),
            warmup=False,
        )
        return out
    if (
        (m >= 4096 or (n >= 4096 and m >= 512))
        and 128 <= n <= 8192
        and (not n & n - 1)
        and (x.ndim == 2)
        and (x.stride(0) == 1)
        and (x.stride(1) >= m)
        and (weight.stride(0) == 1)
    ):
        x = x.contiguous()
    if (
        (m >= 1024 or (n >= 4096 and m >= 128))
        and (n != 1024 or m >= 4096)
        and (128 <= n <= 8192)
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
    ):
        br = min(
            64, 131072 // (n * max(x.element_size(), weight.element_size()))
        )
        s = x.stride(-2)
        if (0 <= s and (br - 1) * s + n - 1 <= 2147483647) and (
            m % 64 == 0 or m >= 4096
        ):
            full = m // br * br
            _rows_direct_weight.run(
                x,
                weight,
                out,
                n,
                eps,
                full,
                br,
                s,
                groups_per_cluster=br,
                isCloseUnrollControl=False,
                enable_fp_fusion=False,
                isCloseMemoryAsync=False,
                grid=(12,),
                warmup=False,
                vrf_budget=16 if m >= 1024 and n >= 4096 else 24,
            )
            if full < m:
                _tail.run(
                    x,
                    weight,
                    out,
                    n,
                    eps,
                    s,
                    full,
                    num_warps=4 if n <= 4096 else 8,
                    enable_fp_fusion=False,
                    isCloseMemoryAsync=False,
                    grid=(m - full,),
                    warmup=False,
                )
            return out
    if m:
        b = max(128, triton.next_power_of_2(n))
        fast = (
            b == n
            and x.stride(-1) == 1
            and (weight.stride(0) == 1)
            and (x.is_contiguous() or len(x.shape) == 2)
        )
        if (
            fast
            and 128 <= n <= 8192
            and (m >= 1024 or (n >= 4096 and m >= 128))
            and (n != 1024 or m >= 4096)
            and (x.stride(-2) == n)
        ) and (m % 64 == 0 or m >= 4096):
            br = min(
                64,
                131072 // (n * max(x.element_size(), weight.element_size())),
            )
            full = m // br * br
            _rows_direct_weight.run(
                x,
                weight,
                out,
                n,
                eps,
                full,
                br,
                n,
                groups_per_cluster=br,
                isCloseUnrollControl=False,
                enable_fp_fusion=False,
                isCloseMemoryAsync=False,
                grid=(12,),
                warmup=False,
                vrf_budget=16 if m >= 1024 and n >= 4096 else 24,
            )
            if full < m:
                _tail.run(
                    x,
                    weight,
                    out,
                    n,
                    eps,
                    n,
                    full,
                    num_warps=4 if n <= 4096 else 8,
                    enable_fp_fusion=False,
                    isCloseMemoryAsync=False,
                    grid=(m - full,),
                    warmup=False,
                )
            return out
        br = 8 if n == 1024 else 2 if n in (2048, 4096) else 1
        if fast and br > 1 and (m % br == 0) and (x.stride(-2) == n):
            _rows.run(
                x,
                weight,
                out,
                n,
                eps,
                m,
                br,
                enable_fp_fusion=False,
                isCloseMemoryAsync=False,
                grid=(12,),
                warmup=False,
            )
        else:
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
                isCloseMemoryAsync=False,
                grid=(m,),
                warmup=False,
            )
    return out


__all__ = ["gemma_rmsnorm"]
