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
    z = s * tl.rsqrt(tl.sum(s * s, 0) / N + EPS)
    y = tl.fma(z, w, z)
    if B == N:
        tl.store(Y + base + c, y)
        tl.store(RO + base + c, s)
    else:
        tl.store(Y + base + c, y, c < N)
        tl.store(RO + base + c, s, c < N)


def gemma_fused_add_rmsnorm(x, residual, weight, eps):
    n = x.size(-1)
    rows = x.numel() // n
    out = torch.empty_like(x, memory_format=torch.contiguous_format)
    res = torch.empty_like(x, memory_format=torch.contiguous_format)
    if rows:
        if not x.is_contiguous():
            x = x.contiguous()
        if not residual.is_contiguous():
            residual = residual.contiguous()
        if not weight.is_contiguous():
            weight = weight.contiguous()
        b = 1 << (n - 1).bit_length()
        _flat.run(
            x,
            residual,
            weight,
            out,
            res,
            n,
            eps,
            b,
            grid=(rows,),
            warmup=False,
            num_warps=(
                4
                if b <= 2048 or (b <= 4096 and rows <= 1024)
                else 8 if b <= 4096 else 16
            ),
        )
    return out, res


__all__ = ["gemma_fused_add_rmsnorm"]
