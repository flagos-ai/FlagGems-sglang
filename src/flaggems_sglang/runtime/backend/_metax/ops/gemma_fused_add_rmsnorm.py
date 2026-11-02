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
    SH: tl.constexpr,
    XS: tl.constexpr,
    RS: tl.constexpr,
    WS: tl.constexpr,
    N: tl.constexpr,
    EPS: tl.constexpr,
    B: tl.constexpr,
):
    row = tl.program_id(0)
    rest = row
    xo = 0
    ro = 0
    for axis in tl.static_range(len(SH) - 1, -1, -1):
        pos = rest % SH[axis]
        rest = rest // SH[axis]
        xo += pos * XS[axis]
        ro += pos * RS[axis]
    c = tl.arange(0, B)
    x = tl.load(X + xo + c * XS[len(SH)], c < N, other=0).to(tl.float32)
    r = tl.load(R + ro + c * RS[len(SH)], c < N, other=0).to(tl.float32)
    w = tl.load(W + c * WS, c < N, other=0).to(tl.float32)
    s = x + r
    var = tl.sum(s * s, 0) / N
    y = s * tl.rsqrt(var + EPS) * (1.0 + w)
    tl.store(Y + row * N + c, y, c < N)
    tl.store(O + row * N + c, s, c < N)


def gemma_fused_add_rmsnorm(x, residual, weight, eps):
    out = torch.empty(x.shape, device=x.device, dtype=x.dtype)
    residual_out = torch.empty(x.shape, device=x.device, dtype=x.dtype)
    if x.numel() == 0:
        return out, residual_out
    n = x.shape[-1]
    block = triton.next_power_of_2(n) * (
        1
        if x.is_contiguous()
        and residual.is_contiguous()
        and weight.is_contiguous()
        else 2
    )
    _kernel[(x.numel() // n,)](
        x,
        residual,
        weight,
        out,
        residual_out,
        tuple(x.shape[:-1]),
        tuple(x.stride()),
        tuple(residual.stride()),
        weight.stride(0),
        n,
        float(eps),
        block,
        num_warps=4 if block <= 4096 else 8,
        enable_fp_fusion=False,
    )
    return out, residual_out


__all__ = ["gemma_fused_add_rmsnorm"]
