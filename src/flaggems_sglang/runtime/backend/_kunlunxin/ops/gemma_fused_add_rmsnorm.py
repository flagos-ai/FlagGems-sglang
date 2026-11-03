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
def _flat(X, R, W, Y, RO, N: tl.constexpr, EPS: tl.constexpr, B: tl.constexpr):
    base = tl.program_id(0) * N
    c = tl.arange(0, B)
    if B == N:
        x = tl.load(X + base + c).to(tl.float32)
        r = tl.load(R + base + c).to(tl.float32)
        w = tl.load(W + c).to(tl.float32)
    else:
        m = c < N
        x = tl.load(X + base + c, m, other=0).to(tl.float32)
        r = tl.load(R + base + c, m, other=0).to(tl.float32)
        w = tl.load(W + c, m, other=0).to(tl.float32)
    s = x + r
    y = s * tl.rsqrt(tl.sum(s * s, 0) / N + EPS) * (1.0 + w)
    if B == N:
        tl.store(Y + base + c, y)
        tl.store(RO + base + c, s)
    else:
        tl.store(Y + base + c, y, c < N)
        tl.store(RO + base + c, s, c < N)


@triton.jit
def _tile(
    X,
    R,
    W,
    Y,
    O,
    M: tl.constexpr,
    N: tl.constexpr,
    EPS: tl.constexpr,
    B: tl.constexpr,
    T: tl.constexpr,
):
    rows = tl.program_id(0) * T + tl.arange(0, T)[:, None]
    c = tl.arange(0, B)[None, :]
    off = rows * N + c
    if B == N:
        x = tl.load(X + off).to(tl.float32)
        r = tl.load(R + off).to(tl.float32)
        w = tl.load(W + c).to(tl.float32)
    else:
        x = tl.load(X + off, c < N, other=0).to(tl.float32)
        r = tl.load(R + off, c < N, other=0).to(tl.float32)
        w = tl.load(W + c, c < N, other=0).to(tl.float32)
    s = x + r
    y = s * tl.rsqrt(tl.sum(s * s, 1)[:, None] / N + EPS) * (1.0 + w)
    if B == N:
        tl.store(Y + off, y)
        tl.store(O + off, s)
    else:
        tl.store(Y + off, y, c < N)
        tl.store(O + off, s, c < N)


def gemma_fused_add_rmsnorm(x, residual, weight, eps):
    n = x.size(-1)
    rows = x.numel() // n
    out = torch.empty(x.shape, device=x.device, dtype=x.dtype)
    res = torch.empty(x.shape, device=x.device, dtype=x.dtype)
    if rows:
        if not x.is_contiguous():
            x = x.contiguous()
        if not residual.is_contiguous():
            residual = residual.contiguous()
        if not weight.is_contiguous():
            weight = weight.contiguous()
        b = 1 << (n - 1).bit_length()
        if rows >= 512 and b == n and b <= 4096:
            t = max(1, 16384 // b)
            while rows % t:
                t //= 2
        else:
            t = 1
        if t > 1:
            _tile[(rows // t,)](
                x,
                residual,
                weight,
                out,
                res,
                rows,
                n,
                eps,
                b,
                t,
                num_warps=4,
                num_stages=1,
                isCloseMemoryAsync=False,
            )
        else:
            _flat[(rows,)](
                x,
                residual,
                weight,
                out,
                res,
                n,
                eps,
                b,
                num_warps=4 if b <= 4096 else 8,
                num_stages=1,
                isCloseMemoryAsync=False,
            )
    return out, res


__all__ = ["gemma_fused_add_rmsnorm"]
