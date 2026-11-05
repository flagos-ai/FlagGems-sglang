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
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied. See the License for the specific language governing
# permissions and limitations under the License.

"""Fused RMS normalization, tanh gating, and residual addition."""

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"ROWS": 1}, num_warps=4),
        triton.Config({"ROWS": 1}, num_warps=8),
        triton.Config({"ROWS": 2}, num_warps=4),
        triton.Config({"ROWS": 4}, num_warps=4),
    ],
    key=["D", "nrows"],
)
@triton.jit
def _rmsnorm_tanh_residual_kernel(
    x_ptr,
    g_ptr,
    r_ptr,
    w_ptr,
    out_ptr,
    eps,
    nrows,
    repeat,
    D: tl.constexpr,
    BD: tl.constexpr,
    ROWS: tl.constexpr,
    MASKED: tl.constexpr,
    INT64: tl.constexpr,
):
    pid = tl.program_id(0)
    ridx = pid * ROWS + tl.arange(0, ROWS)
    rmask = ridx < nrows
    cols = tl.arange(0, BD)
    if MASKED:
        m = rmask[:, None] & (cols < D)[None, :]
    else:
        m = rmask[:, None]

    if INT64:
        xoff = ridx[:, None].to(tl.int64) * D + cols[None, :]
        goff = (ridx // repeat)[:, None].to(tl.int64) * D + cols[None, :]
    else:
        xoff = ridx[:, None] * D + cols[None, :]
        goff = (ridx // repeat)[:, None] * D + cols[None, :]
    x = tl.load(x_ptr + xoff, mask=m, other=0.0)
    g = tl.load(g_ptr + goff, mask=m, other=0.0)
    r = tl.load(r_ptr + xoff, mask=m, other=0.0)
    w = (
        tl.load(w_ptr + cols, mask=cols < D, other=0.0)
        if MASKED
        else tl.load(w_ptr + cols)
    )

    # fp32 only where the reference is fp32: the reduction and the tanh.
    xf = x.to(tl.float32)
    rstd = tl.rsqrt(tl.sum(xf * xf, axis=1) / D + eps)[:, None]

    normed = (xf * rstd).to(tl.bfloat16) * w
    e = tl.exp2(g.to(tl.float32) * -2.8853900817779268)
    gated = (2.0 / (1.0 + e) - 1.0).to(tl.bfloat16)
    y = r + gated * normed

    tl.store(out_ptr + xoff, y.to(out_ptr.dtype.element_ty), mask=m)


def _next_pow2(n):
    p = 1
    while p < n:
        p *= 2
    return p


def rmsnorm_tanh_residual(x, gate, residual, weight, eps):
    """out = residual + tanh(gate) * (RMSNorm(x) * weight), gate."""
    dim = x.shape[-1]
    x2 = x.reshape(-1, dim)
    rows = x2.shape[0]
    g2 = gate.reshape(-1, dim)
    r2 = residual.reshape(-1, dim)
    repeat = rows // g2.shape[0]

    out = torch.empty_like(x2)

    BD = _next_pow2(dim)
    MASKED = BD != dim
    # Largest byte index touched by any tile: row < rows, col < BD <= 2*dim.
    # int32 addressing is ample below 2**31 and noticeably cheaper per lane.
    INT64 = rows * BD >= 2**31

    _rmsnorm_tanh_residual_kernel[
        (lambda META: (triton.cdiv(rows, META["ROWS"]),))
    ](
        x2,
        g2,
        r2,
        weight,
        out,
        eps,
        rows,
        repeat,
        D=dim,
        BD=BD,
        MASKED=MASKED,
        INT64=INT64,
    )
    return out.reshape(x.shape)


__all__ = ["rmsnorm_tanh_residual"]
