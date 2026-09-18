# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Kunlunxin vendor: one program per row (no grid-stride loop, which the
# S0 generic used and timed out at 1830s on this backend); tl.rsqrt
# and tl.sigmoid for the norm and gate (platform-proven on the T20
# sister task at 0.509x).

import torch
import triton
import triton.language as tl


@triton.jit
def _fla_ln_gated_enflame(
    x_ptr,
    g_ptr,
    w_ptr,
    b_ptr,
    out_ptr,
    dim,
    eps,
    x_stride,
    g_stride,
    o_stride,
    IS_RMS: tl.constexpr,
    HAS_W: tl.constexpr,
    HAS_B: tl.constexpr,
    ACT_SWISH: tl.constexpr,
    ACT_SIGMOID: tl.constexpr,
    BLOCK_D: tl.constexpr,
    isCloseCoreTiling: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_D)
    mask = offs < dim
    x = tl.load(x_ptr + row * x_stride + offs, mask=mask, other=0.0).to(tl.float32)
    g = tl.load(g_ptr + row * g_stride + offs, mask=mask, other=0.0).to(tl.float32)

    if IS_RMS:
        var = tl.sum(x * x, axis=0) / dim
        x_hat = x * tl.rsqrt(var + eps)
    else:
        mean = tl.sum(x, axis=0) / dim
        xc = tl.where(mask, x - mean, 0.0)
        var = tl.sum(xc * xc, axis=0) / dim
        x_hat = xc * tl.rsqrt(var + eps)

    y = x_hat
    if HAS_W:
        y = y * tl.load(w_ptr + offs, mask=mask, other=1.0).to(tl.float32)
    if HAS_B:
        y = y + tl.load(b_ptr + offs, mask=mask, other=0.0).to(tl.float32)

    sig_g = tl.sigmoid(g)
    if ACT_SWISH:
        y = y * g * sig_g
    elif ACT_SIGMOID:
        y = y * sig_g

    tl.store(
        out_ptr + row * o_stride + offs,
        y.to(out_ptr.dtype.element_ty),
        mask=mask,
    )


def fla_layernorm_gated(
    x, g, weight, bias, activation="swish", eps=1e-5, is_rms_norm=True
):
    x = x.contiguous()
    g = g.contiguous()
    rows = x.shape[0]
    dim = x.shape[-1]
    out = torch.empty_like(x)
    if rows * dim == 0:
        return out

    act_swish = activation in ("swish", "silu")
    act_sigmoid = activation == "sigmoid" and not act_swish

    if weight is None:
        weight = x  # placeholder, HAS_W=False skips the load
    if bias is None:
        bias = x
    HAS_W = weight is not x
    HAS_B = bias is not x

    grid = (rows,)  # one program per row, no grid-stride loop
    _fla_ln_gated_enflame[grid](
        x,
        g,
        weight,
        bias,
        out,
        dim,
        float(eps),
        x.stride(0),
        g.stride(0),
        out.stride(0),
        IS_RMS=is_rms_norm,
        HAS_W=HAS_W,
        HAS_B=HAS_B,
        ACT_SWISH=act_swish,
        ACT_SIGMOID=act_sigmoid,
        BLOCK_D=max(triton.next_power_of_2(dim), 16),
        isCloseCoreTiling=True,
        num_warps=4,
        num_stages=1,
    )
    return out


__all__ = ["fla_layernorm_gated"]
