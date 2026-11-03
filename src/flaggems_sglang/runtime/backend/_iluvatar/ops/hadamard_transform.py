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
        v = _butterfly(v, B, N >> (stage + 1))
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
    offset = (flat % N) * ST[len(SH)]
    for axis in tl.static_range(len(SH) - 1, -1, -1):
        pos = rest % SH[axis]
        rest = rest // SH[axis]
        offset += pos * ST[axis]
    v = tl.load(X + offset, row < M, other=0).to(tl.float32)
    for stage in tl.static_range(LOG):
        v = _butterfly(v, B, 1 << (LOG - 1 - stage))
    tl.store(Y + flat, v * SCALE, row < M)


def _config(m, n):
    if n == 128:
        if m == 1:
            return 128, 1
        if m == 8:
            return 1024, 8
        if m == 64:
            return 1024, 8
        if m == 512:
            return 512, 4
        if m == 4096:
            return 256, 2
    elif n == 256 and m == 37:
        return 2048, 8
    elif n == 512:
        if m == 1:
            return 512, 4
        if m == 8 or m == 64 or m == 512:
            return 1024, 8
        if m == 4096:
            return 512, 4
    elif n == 1024 and m == 64:
        return 1024, 8
    elif n == 2048:
        return 2048, 8
    b = max(n, 1024)
    return b, 1 if m * n >= 1048576 else 4


def hadamard_transform(x, scale):
    n = x.shape[-1]
    size = x.numel()
    out = torch.empty(x.shape, device=x.device, dtype=x.dtype)
    if not size:
        return out
    m = size // n
    b, warps = _config(m, n)
    if x.is_contiguous():
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
        _strided.run(
            x,
            out,
            tuple(x.shape[:-1]),
            tuple(x.stride()),
            n,
            m,
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
