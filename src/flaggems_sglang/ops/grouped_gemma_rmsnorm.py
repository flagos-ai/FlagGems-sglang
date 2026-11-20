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

"""Triton grouped_gemma_rmsnorm.

Splits the last dimension of ``x`` into groups of ``group_size``, applies
RMS normalization independently inside each group, scales by ``1 + weight``
(Gemma parameterization), and returns in the input dtype. The math runs in
fp32; the reciprocal square root is ``tl.rsqrt`` and the per-group mean is
one fused integer-indexed reduction, so every element is a single DRAM
read and write (1R+1W).

One CTA covers ``groups_per_block`` groups of one row. The tile is sized
to ~1024 lanes (2 warps) for latency, widening to 2048 lanes (4 warps)
for big groups; rows with few groups collapse to a single CTA per row.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _grouped_gemma_rmsnorm_kernel(
    X,
    W,
    O,
    NG,
    RS: tl.constexpr,
    OS: tl.constexpr,
    GS: tl.constexpr,
    GPB: tl.constexpr,
    GSP: tl.constexpr,
    EPS,
):
    row = tl.program_id(0)
    chunk = tl.program_id(1)
    g = chunk * GPB + tl.arange(0, GPB)
    s = tl.arange(0, GSP)
    cols = g[:, None] * GS + s[None, :]
    m = (g[:, None] < NG) & (s[None, :] < GS)
    x = tl.load(X + row.to(tl.int64) * RS + cols, mask=m, other=0.0).to(
        tl.float32
    )
    var = tl.sum(x * x, 1) / GS
    rstd = tl.rsqrt(var + EPS)
    w = tl.load(W + cols, mask=m, other=0.0).to(tl.float32)
    y = x * rstd[:, None] * (1.0 + w)
    tl.store(O + row.to(tl.int64) * OS + cols, y, mask=m)


def grouped_gemma_rmsnorm(x, weight, group_size, eps=1e-6):
    n = x.shape[-1]
    orig_shape = x.shape
    if x.dim() > 2:
        x = x.reshape(-1, n)
    rows = x.shape[0]
    ng = n // group_size

    gsp = max(16, triton.next_power_of_2(group_size))
    gpb = max(1, 1024 // gsp)
    if ng <= 2 * gpb:
        # The row fits in one or two chunks; a single CTA per row wins.
        gpb = ng
    lanes = gpb * gsp
    warps = 2 if lanes <= 1024 else 4 if lanes <= 2048 else 8

    out = torch.empty_like(x)
    grid = (rows, triton.cdiv(ng, gpb))
    _grouped_gemma_rmsnorm_kernel[grid](
        x,
        weight,
        out,
        ng,
        x.stride(0),
        out.stride(0),
        group_size,
        gpb,
        gsp,
        eps,
        num_warps=warps,
    )
    return out.reshape(orig_shape)


__all__ = ["grouped_gemma_rmsnorm"]
