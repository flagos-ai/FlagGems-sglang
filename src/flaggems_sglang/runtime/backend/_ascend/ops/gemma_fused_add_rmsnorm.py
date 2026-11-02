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
    R,
    W,
    Y,
    O,
    M: tl.constexpr,
    N: tl.constexpr,
    EPS: tl.constexpr,
    B: tl.constexpr,
    T: tl.constexpr,
    G: tl.constexpr,
    P: tl.constexpr,
):
    cols = tl.arange(0, B)
    g = 1.0 + tl.load(W + cols, cols < N, other=0).to(tl.float32)[None, :]
    c = cols[None, :]
    if G > P:
        count = tl.cdiv(G, P)
        start = tl.program_id(0) * count
        end = tl.minimum(start + count, G)
        step = 1
    else:
        start = tl.program_id(0)
        end = G
        step = P
    for tile in tl.range(start, end, step):
        rows = tile * T + tl.arange(0, T)[:, None]
        off = rows * N + c
        m = (rows < M) & (c < N)
        s = tl.load(X + off, m, other=0).to(tl.float32) + tl.load(
            R + off, m, other=0
        ).to(tl.float32)
        tl.store(O + off, s, m)
        tl.store(
            Y + off, s * tl.rsqrt(tl.sum(s * s, 1)[:, None] / N + EPS) * g, m
        )


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
        t = max(1, 8192 // b)
        g = (rows + t - 1) // t
        p = min(40, g)
        _kernel[(p,)](
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
            g,
            p,
            num_warps=1,
        )
    return out, res


__all__ = ["gemma_fused_add_rmsnorm"]
