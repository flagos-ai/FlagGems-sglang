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
def _butterfly(v, B: tl.constexpr, S: tl.constexpr):
    a, b = tl.split(tl.permute(tl.reshape(v, (B // (2 * S), 2, S)), (0, 2, 1)))
    return tl.reshape(tl.permute(tl.join(a + b, a - b), (0, 2, 1)), (B,))


@triton.jit
def _contiguous(
    X,
    Y,
    SCALE: tl.constexpr,
    N: tl.constexpr,
    SIZE: tl.constexpr,
    B: tl.constexpr,
):
    idx = tl.program_id(0) * B + tl.arange(0, B)
    if SIZE % B == 0:
        v = tl.load(X + idx).to(tl.float32)
    else:
        v = tl.load(X + idx, idx < SIZE, other=0).to(tl.float32)
    for stage in tl.static_range(N.bit_length() - 1):
        v = _butterfly(v, B, N >> stage + 1)
    if SIZE % B == 0:
        tl.store(Y + idx, v * SCALE)
    else:
        tl.store(Y + idx, v * SCALE, idx < SIZE)


@triton.jit
def _strided(
    X,
    Y,
    SH: tl.constexpr,
    ST: tl.constexpr,
    N: tl.constexpr,
    M: tl.constexpr,
    LOG: tl.constexpr,
    SCALE: tl.constexpr,
    B: tl.constexpr,
):
    flat = tl.program_id(0) * B + tl.arange(0, B)
    row = flat // N
    rest = row
    offset = flat % N * ST[len(SH)]
    for axis in tl.static_range(len(SH) - 1, -1, -1):
        pos = rest % SH[axis]
        rest = rest // SH[axis]
        offset += pos * ST[axis]
    v = tl.load(X + offset, row < M, other=0).to(tl.float32)
    for stage in tl.static_range(LOG):
        v = _butterfly(v, B, 1 << LOG - 1 - stage)
    tl.store(Y + flat, v * SCALE, row < M)


@triton.jit
def _stream_stage(x, N: tl.constexpr, S: tl.constexpr):
    if S == 1:
        a, b = tl.split(tl.reshape(x, (1, N // 2, 2)))
        return tl.reshape(tl.join(a + b, a - b), (1, N))
    a, b = tl.split(
        tl.permute(tl.reshape(x, (1, N // (2 * S), 2, S)), (0, 1, 3, 2))
    )
    return tl.reshape(tl.permute(tl.join(a + b, a - b), (0, 1, 3, 2)), (1, N))


@triton.jit
def _stream_row(X, Y, N: tl.constexpr, SCALE: tl.constexpr):
    rows = tl.program_id(0) + tl.arange(0, 1)
    idx = rows[:, None] * N + tl.arange(0, N)[None, :]
    v = tl.load(X + idx, cache_modifier=".cg").to(tl.float32)
    for stage in tl.static_range(N.bit_length() - 1):
        v = _stream_stage(v, N, N >> stage + 1)
    tl.store(Y + idx, v * SCALE, cache_modifier=".cs")


@triton.jit
def _group_stage(v, R: tl.constexpr, N: tl.constexpr, S: tl.constexpr):
    a, b = tl.split(
        tl.permute(tl.reshape(v, (R, N // (2 * S), 2, S)), (0, 1, 3, 2))
    )
    return tl.reshape(tl.permute(tl.join(a + b, a - b), (0, 1, 3, 2)), (R, N))


@triton.jit
def _group_tile(
    X,
    Y,
    N: tl.constexpr,
    SCALE: tl.constexpr,
    M: tl.constexpr,
    R: tl.constexpr,
):
    r = tl.program_id(0) * R + tl.arange(0, R)
    i = r[:, None] * N + tl.arange(0, N)[None, :]
    if M % R == 0:
        v = tl.load(X + i, cache_modifier=".cg").to(tl.float32)
    else:
        v = tl.load(X + i, r[:, None] < M, other=0, cache_modifier=".cg").to(
            tl.float32
        )
    for s in tl.static_range(N.bit_length() - 1):
        v = _group_stage(v, R, N, N >> s + 1)
    if M % R == 0:
        tl.store(Y + i, v * SCALE, cache_modifier=".cs")
    else:
        tl.store(Y + i, v * SCALE, r[:, None] < M, cache_modifier=".cs")


def hadamard_transform(x, scale):
    n = x.shape[-1]
    size = x.numel()
    if n == 512 and size // n >= 4096 and x.is_contiguous():
        out = torch.empty_like(x)
        m = size // n
        _group_tile.run(
            x,
            out,
            n,
            scale,
            m,
            2,
            grid=((m + 1) // 2,),
            warmup=False,
            num_warps=2,
            num_stages=1,
        )
        return out
    if n == 2048 and x.is_contiguous():
        out = torch.empty_like(x)
        if size:
            m = size // n
            _stream_row.run(
                x,
                out,
                n,
                scale,
                grid=(m,),
                warmup=False,
                num_warps=4 if m >= 512 else 8,
                num_stages=1,
            )
        return out
    b = max(n, 1024)
    warps = 1 if size >= 1048576 else 4
    if x.is_contiguous():
        out = torch.empty_like(x)
        if size:
            _contiguous.run(
                x,
                out,
                scale,
                n,
                size,
                b,
                grid=((size + b - 1) // b,),
                warmup=False,
                num_warps=warps,
                num_stages=1,
            )
    else:
        out = torch.empty(x.shape, device=x.device, dtype=x.dtype)
        if size:
            _strided.run(
                x,
                out,
                tuple(x.shape[:-1]),
                tuple(x.stride()),
                n,
                size // n,
                n.bit_length() - 1,
                float(scale),
                b,
                grid=((size + b - 1) // b,),
                warmup=False,
                num_warps=warps,
                num_stages=1,
            )
    return out


__all__ = ["hadamard_transform"]
