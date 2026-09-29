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

# Ascend vendor: identical math to the generic kernel, with dim as a
# tl.constexpr so `offs < D` folds at compile time - runtime int32
# vector compares lower to scalar code on Ascend (official perf guide;
# T40 +130% / T42 +35% prior). The grid-stride row loop and the
# division form stay byte-equivalent to the platform-validated E1.
# Large-D rows (dim > 1024, 256 | dim) take the sub-block kernel below:
# the row splits into exact D_TILE chunks so no phase carries a bounds
# mask, and x is re-read instead of staying live across the reduction.

import torch
import triton
import triton.language as tl

_BLOCK_D = 1024
_MAX_GRID = 65535


@triton.jit
def _fla_ln_gated_subblock_asc(
    x_ptr,
    g_ptr,
    w_ptr,
    b_ptr,
    out_ptr,
    rows,
    D: tl.constexpr,
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
    D_TILE: tl.constexpr,
    N_SUB: tl.constexpr,
):
    pid = tl.program_id(0)
    grid_size = tl.num_programs(0)
    offs = tl.arange(0, D_TILE)
    for row in range(pid, rows, grid_size):
        x_row = x_ptr + row * x_stride_row
        sum_sq = 0.0
        sum_x = 0.0
        for s in range(N_SUB):
            xv = tl.load(x_row + s * D_TILE + offs).to(tl.float32)
            sum_sq += tl.sum(xv * xv)
            if not IS_RMS:
                sum_x += tl.sum(xv)
        if IS_RMS:
            var = sum_sq / D
            rstd = 1.0 / tl.sqrt(var + eps)
            mean = 0.0
        else:
            mean = sum_x / D
            cent_sq = 0.0
            for s in range(N_SUB):
                xv = tl.load(x_row + s * D_TILE + offs).to(tl.float32)
                xc = xv - mean
                cent_sq += tl.sum(xc * xc)
            var = cent_sq / D
            rstd = 1.0 / tl.sqrt(var + eps)
        g_row = g_ptr + row * g_stride_row
        o_row = out_ptr + row * o_stride_row
        for s in range(N_SUB):
            o = s * D_TILE + offs
            xv = tl.load(x_row + o).to(tl.float32)
            if IS_RMS:
                y = xv * rstd
            else:
                y = (xv - mean) * rstd
            if HAS_W:
                y = y * tl.load(w_ptr + o * w_stride).to(tl.float32)
            if HAS_B:
                y = y + tl.load(b_ptr + o * b_stride).to(tl.float32)
            g = tl.load(g_row + o).to(tl.float32)
            sig_g = 1.0 / (1.0 + tl.exp(-g))
            if ACT_SWISH:
                y = y * g * sig_g
            elif ACT_SIGMOID:
                y = y * sig_g
            tl.store(o_row + o, y.to(out_ptr.dtype.element_ty))


@triton.jit
def _fla_ln_gated_kernel_asc(
    x_ptr,
    g_ptr,
    w_ptr,
    b_ptr,
    out_ptr,
    rows,
    D: tl.constexpr,
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
        mask = offs < D
        x = tl.load(
            x_ptr + row * x_stride_row + offs, mask=mask, other=0.0
        ).to(tl.float32)
        if IS_RMS:
            var = tl.sum(x * x, axis=0) / D
            x_hat = x * 1.0 / tl.sqrt(var + eps)
        else:
            mean = tl.sum(x, axis=0) / D
            xc = tl.where(mask, x - mean, 0.0)
            var = tl.sum(xc * xc, axis=0) / D
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

        # Gate is independent of the reduction; keep it out of its live set.
        g = tl.load(
            g_ptr + row * g_stride_row + offs, mask=mask, other=0.0
        ).to(tl.float32)
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
    grid = (min(rows, _MAX_GRID),)
    # Sub-block path for large rows whose lowest power-of-two factor is
    # at least 256 (any such dim splits into exact tiles); small or
    # awkward dims keep the platform-proven whole-row kernel.
    if dim > 1024 and dim % 256 == 0:
        d_tile = min(dim & (-dim), 1024)
        _fla_ln_gated_subblock_asc[grid](
            x,
            g,
            weight,
            bias,
            out,
            rows,
            D=dim,
            eps=float(eps),
            x_stride_row=x.stride(0),
            g_stride_row=g.stride(0),
            o_stride_row=out.stride(0),
            w_stride=weight.stride(0) if HAS_W_FLAG else 0,
            b_stride=bias.stride(0) if HAS_B_FLAG else 0,
            IS_RMS=is_rms_norm,
            HAS_W=HAS_W_FLAG,
            HAS_B=HAS_B_FLAG,
            ACT_SWISH=act_swish,
            ACT_SIGMOID=act_sigmoid,
            D_TILE=d_tile,
            N_SUB=dim // d_tile,
            num_warps=4,
            num_stages=1,
        )
        return out
    _fla_ln_gated_kernel_asc[grid](
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
        BLOCK_D=max(triton.next_power_of_2(dim), 16),
        num_warps=4,
        num_stages=1,
    )
    return out


__all__ = ["fla_layernorm_gated"]
