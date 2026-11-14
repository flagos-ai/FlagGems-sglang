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

# Enflame vendor dispatch (GCU400, do_bench kernel-mode medians):
# * rows <= 48: one flat program per row (launch-floor bound, ~3-4us; same
#   shape as the previous revision - zero regression risk for decode-size
#   batches).
# * rows > 48: the flat grid pays ~50ns/CTA dispatch per row, which measured
#   78% of the wall at 4096x2048 (245us total, ~191us dispatch). A persistent
#   2D row-batch kernel keeps 48 programs resident: each iterates over
#   [U, BLOCK_D] tiles with axis=1 reductions and weight/bias hoisted out of
#   the loop. num_warps=1 - cross-warp reduction tax makes nw2/nw4 1.1-1.4x
#   slower. U ladder = clamp(next_pow2(rows // 48), 2, 32), measured optimum
#   at t=64->U2 (5.8us), t=512->U16 (11.2us), t=4096->U32 (41.3us); nw/grid
#   sweeps and U=64 are documented regressions. With a bias term the tile
#   register budget cliffs: U=32+bias spills and measures 135us at t=4096
#   (3.3x worse than U=16+bias's 45.7us), so the U cap is 16 when HAS_B.
# * The 2D affine-index form aborts the SIP in long loops (>=~200 iters/CTA,
#   compiler_issues/sglang_apply_token_bitmask_2d_index_longloop.md), so once
#   cdiv(nblk, 96) exceeds 48 the grid grows past 48 to keep per-program
#   iterations <= 96 (capped at the 65535 grid.x hardware limit).
# * Known limit shared with the flat path: row*stride offsets stay in int32
#   (rows > ~1M at D=2048 would overflow; such tensors exceed 4GB anyway).
# * tl.rsqrt/tl.sigmoid kept from the platform-proven T20 rewrite.

import torch
import triton
import triton.language as tl

_MAX_GRID = 65535
_DISPATCH_FLOOR = 48  # GCU400: one wave = 48 CTAs (2/SIP); extras cost ~50ns
_MAX_ITERS = 96  # keep 2D-index loop trips below the SIP-abort zone


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
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_D)
    mask = offs < dim
    x = tl.load(x_ptr + row * x_stride + offs, mask=mask, other=0.0).to(
        tl.float32
    )
    g = tl.load(g_ptr + row * g_stride + offs, mask=mask, other=0.0).to(
        tl.float32
    )

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


@triton.jit
def _fla_ln_gated_rows_enflame(
    x_ptr,
    g_ptr,
    w_ptr,
    b_ptr,
    out_ptr,
    rows,
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
    U: tl.constexpr,
    EXACT: tl.constexpr,
):
    pid = tl.program_id(0)
    npg = tl.num_programs(0)
    offs = tl.arange(0, BLOCK_D)
    u = tl.arange(0, U)
    cm = offs < dim
    if HAS_W:
        wv = tl.load(w_ptr + offs, mask=cm, other=1.0).to(tl.float32)
    if HAS_B:
        bv = tl.load(b_ptr + offs, mask=cm, other=0.0).to(tl.float32)
    nblk = (rows + U - 1) // U
    # num_stages=1: the enflame backend defaults to 3, which pipelines three
    # [U, BLOCK_D] fp32 tiles in flight (~1.5MB at U=32) and measures 3.3x
    # slower at 4096x2048; depth 1 is the measured optimum (depth 2 ties).
    for blk in tl.range(pid, nblk, npg, num_stages=1):
        row0 = blk * U
        if EXACT:
            x = tl.load(
                x_ptr + (row0 + u)[:, None] * x_stride + offs[None, :]
            ).to(tl.float32)
            g = tl.load(
                g_ptr + (row0 + u)[:, None] * g_stride + offs[None, :]
            ).to(tl.float32)
        else:
            m2 = ((row0 + u) < rows)[:, None] & cm[None, :]
            x = tl.load(
                x_ptr + (row0 + u)[:, None] * x_stride + offs[None, :],
                mask=m2,
                other=0.0,
            ).to(tl.float32)
            g = tl.load(
                g_ptr + (row0 + u)[:, None] * g_stride + offs[None, :],
                mask=m2,
                other=0.0,
            ).to(tl.float32)
        if IS_RMS:
            var = tl.sum(x * x, axis=1) / dim
            y = x * tl.rsqrt(var + eps)[:, None]
        else:
            mean = tl.sum(x, axis=1) / dim
            xc = tl.where(cm[None, :], x - mean[:, None], 0.0)
            var = tl.sum(xc * xc, axis=1) / dim
            y = xc * tl.rsqrt(var + eps)[:, None]
        if HAS_W:
            y = y * wv[None, :]
        if HAS_B:
            y = y + bv[None, :]
        sig_g = tl.sigmoid(g)
        if ACT_SWISH:
            y = y * g * sig_g
        elif ACT_SIGMOID:
            y = y * sig_g
        if EXACT:
            tl.store(
                out_ptr + (row0 + u)[:, None] * o_stride + offs[None, :],
                y.to(out_ptr.dtype.element_ty),
            )
        else:
            tl.store(
                out_ptr + (row0 + u)[:, None] * o_stride + offs[None, :],
                y.to(out_ptr.dtype.element_ty),
                mask=m2,
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
    if HAS_W:
        # kernel indexes weight/bias with unit stride; strided args (e.g.
        # weight[::2]) must be materialized before the launch
        weight = weight.contiguous()
    if HAS_B:
        bias = bias.contiguous()

    block_d = max(triton.next_power_of_2(dim), 16)
    if rows <= _DISPATCH_FLOOR:
        _fla_ln_gated_enflame[(rows,)](
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
            BLOCK_D=block_d,
        )
        return out

    u_rows = max(
        2,
        min(16 if HAS_B else 32, triton.next_power_of_2(rows // _DISPATCH_FLOOR)),
    )
    nblk = triton.cdiv(rows, u_rows)
    if nblk <= _MAX_ITERS * _DISPATCH_FLOOR:
        grid_n = min(nblk, _DISPATCH_FLOOR)
    else:
        grid_n = min(triton.cdiv(nblk, _MAX_ITERS), _MAX_GRID)
    _fla_ln_gated_rows_enflame[(grid_n,)](
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
        IS_RMS=is_rms_norm,
        HAS_W=HAS_W,
        HAS_B=HAS_B,
        ACT_SWISH=act_swish,
        ACT_SIGMOID=act_sigmoid,
        BLOCK_D=block_d,
        U=u_rows,
        EXACT=(rows % u_rows == 0) and (dim == block_d),
        num_warps=1,
    )
    return out


__all__ = ["fla_layernorm_gated"]
