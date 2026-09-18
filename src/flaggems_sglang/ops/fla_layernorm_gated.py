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

import torch
import triton
import triton.language as tl

_BLOCK_D = 1024
_MAX_GRID = 65535


@triton.jit
def _fla_ln_gated_kernel(
    x_ptr,
    g_ptr,
    w_ptr,
    b_ptr,
    out_ptr,
    rows,
    dim,
    eps,
    x_stride_row,
    g_stride_row,
    o_stride_row,
    w_stride,
    b_stride,
    IS_RMS: tl.constexpr,
    HAS_W: tl.constexpr,
    HAS_B: tl.constexpr,
    ACT_SWISH: tl.constexpr,
    ACT_SIGMOID: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0)
    grid_size = tl.num_programs(0)
    for row in range(pid, rows, grid_size):
        offs = tl.arange(0, BLOCK_D)
        mask = offs < dim
        x = tl.load(x_ptr + row * x_stride_row + offs, mask=mask, other=0.0).to(
            tl.float32
        )
        g = tl.load(g_ptr + row * g_stride_row + offs, mask=mask, other=0.0).to(
            tl.float32
        )

        if IS_RMS:
            var = tl.sum(x * x, axis=0) / dim
            x_hat = x * 1.0 / tl.sqrt(var + eps)
        else:
            mean = tl.sum(x, axis=0) / dim
            xc = tl.where(mask, x - mean, 0.0)
            var = tl.sum(xc * xc, axis=0) / dim
            x_hat = xc * 1.0 / tl.sqrt(var + eps)

        y = x_hat
        if HAS_W:
            y = y * tl.load(w_ptr + offs * w_stride, mask=mask, other=1.0).to(
                tl.float32
            )
        if HAS_B:
            y = y + tl.load(b_ptr + offs * b_stride, mask=mask, other=0.0).to(
                tl.float32
            )

        sig_g = 1.0 / (1.0 + tl.exp(-g))
        if ACT_SWISH:
            y = y * g * sig_g
        elif ACT_SIGMOID:
            y = y * sig_g

        tl.store(
            out_ptr + row * o_stride_row + offs,
            y.to(out_ptr.dtype.element_ty),
            mask=mask,
        )


def fla_layernorm_gated(
    x,
    g,
    weight,
    bias,
    activation="swish",
    eps=1e-5,
    is_rms_norm=True,
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
        bias = x  # placeholder, HAS_B=False skips the load

    HAS_W_FLAG = weight is not x
    HAS_B_FLAG = bias is not x
    block_d = max(triton.next_power_of_2(dim), 16)
    grid = (min(rows, _MAX_GRID),)
    _fla_ln_gated_kernel[grid](
        x,
        g,
        weight,
        bias,
        out,
        rows,
        dim,
        float(eps),
        x.stride(0),
        g.stride(0),
        out.stride(0),
        weight.stride(0) if HAS_W_FLAG else 0,
        bias.stride(0) if HAS_B_FLAG else 0,
        IS_RMS=is_rms_norm,
        HAS_W=HAS_W_FLAG,
        HAS_B=HAS_B_FLAG,
        ACT_SWISH=act_swish,
        ACT_SIGMOID=act_sigmoid,
        BLOCK_D=block_d,
        # wide rows need more warps to keep vector lanes fed (T51 e6,
        # scheduling-tier scan: BLOCK_D>=2048 warps 4->8, +10.9% proxy mean)
        num_warps=8 if block_d >= 2048 else 4,
        num_stages=1,
    )
    return out


__all__ = ["fla_layernorm_gated"]
