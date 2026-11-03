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
def _flip_stage(v, idx, B: tl.constexpr, S: tl.constexpr):
    partner = tl.reshape(tl.flip(tl.reshape(v, (B // (2 * S), 2, S)), 1), (B,))
    return tl.where(idx & S == 0, v + partner, partner - v)


@triton.jit
def _stage(x, N: tl.constexpr, S: tl.constexpr):
    if S == 1:
        a, b = tl.split(tl.reshape(x, (1, N // 2, 2)))
        return tl.reshape(tl.join(a + b, a - b), (1, N))
    a, b = tl.split(
        tl.permute(tl.reshape(x, (1, N // (2 * S), 2, S)), (0, 1, 3, 2))
    )
    return tl.reshape(tl.permute(tl.join(a + b, a - b), (0, 1, 3, 2)), (1, N))


@triton.jit
def _row(X, Y, N: tl.constexpr, SCALE: tl.constexpr):
    rows = tl.program_id(0) + tl.arange(0, 1)
    idx = rows[:, None] * N + tl.arange(0, N)[None, :]
    v = tl.load(X + idx, cache_modifier=".cg").to(tl.float32)
    for stage in tl.static_range(N.bit_length() - 1):
        v = _stage(v, N, N >> stage + 1)
    tl.store(Y + idx, v * SCALE, cache_modifier=".cs")


@triton.jit
def _grouped(X, Y, N: tl.constexpr, SCALE: tl.constexpr, B: tl.constexpr):
    idx = tl.arange(0, B)
    offset = tl.program_id(0) * B + idx
    v = tl.load(X + offset, cache_modifier=".cg").to(tl.float32)
    for stage in tl.static_range(N.bit_length() - 1):
        v = _flip_stage(v, idx, B, 1 << stage)
    tl.store(Y + offset, v * SCALE, cache_modifier=".cs")


@triton.jit
def _asc_stage(x, N: tl.constexpr, S: tl.constexpr):
    if S == 1:
        a, b = tl.split(tl.reshape(x, (1, N // 2, 2)))
        return tl.reshape(tl.join(a + b, a - b), (1, N))
    a, b = tl.split(
        tl.permute(tl.reshape(x, (1, N // (2 * S), 2, S)), (0, 1, 3, 2))
    )
    return tl.reshape(tl.permute(tl.join(a + b, a - b), (0, 1, 3, 2)), (1, N))


@triton.jit
def _asc_row(X, Y, N: tl.constexpr, SCALE: tl.constexpr):
    rows = tl.program_id(0) + tl.arange(0, 1)
    idx = rows[:, None] * N + tl.arange(0, N)[None, :]
    v = tl.load(X + idx, cache_modifier=".cg").to(tl.float32)
    for stage in tl.static_range(N.bit_length() - 1):
        v = _asc_stage(v, N, 1 << stage)
    tl.store(Y + idx, v * SCALE, cache_modifier=".cs")


def hadamard_transform(x, scale):
    n = x.shape[-1]
    size = x.numel()
    if not x.is_contiguous():
        x = x.contiguous()
    out = torch.empty_like(x)
    if size:
        m = size // n
        if m >= 512 and (n == 2048 or n == 512):
            _asc_row.run(
                x,
                out,
                n,
                scale,
                grid=(m,),
                warmup=False,
                num_warps=2 if n == 2048 else 1,
                num_stages=1,
            )
            return out
        if m >= 64 and n <= 128 and (m % 4 == 0):
            _grouped.run(
                x,
                out,
                n,
                scale,
                n * 4,
                grid=(m // 4,),
                warmup=False,
                num_warps=1,
                num_stages=1,
            )
            return out
        warps = 1 if n <= 128 or size >= 1048576 else 2 if n <= 512 else 4
        _row.run(
            x,
            out,
            n,
            scale,
            grid=(m,),
            warmup=False,
            num_warps=warps,
            num_stages=1,
        )
    return out


__all__ = ["hadamard_transform"]
