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
def _kernel010(
    x_ptr,
    g_ptr,
    b_ptr,
    o_ptr,
    n_rows,
    scale,
    eps,
    C: tl.constexpr,
    BLOCK: tl.constexpr,
    ROWS: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    EVEN: tl.constexpr,
):
    tile = tl.program_id(0)
    row_ids = tile.to(tl.int64) * ROWS + tl.arange(0, ROWS).to(tl.int64)
    cols = tl.max_contiguous(tl.multiple_of(tl.arange(0, BLOCK), BLOCK), BLOCK)
    offsets = row_ids[:, None] * C + cols[None, :]
    if EVEN:
        x = tl.load(x_ptr + offsets).to(tl.float32)
    else:
        mask = (row_ids[:, None] < n_rows) & (cols[None, :] < C)
        x = tl.load(x_ptr + offsets, mask=mask, other=0).to(tl.float32)
    norm = tl.sqrt(tl.sum(x * x, axis=1))
    inv = 1.0 / tl.maximum(norm, eps)
    if EVEN:
        gamma = tl.load(g_ptr + cols).to(tl.float32)
    else:
        gamma = tl.load(g_ptr + cols, mask=cols < C, other=0).to(tl.float32)
    y = x * inv[:, None] * scale * gamma[None, :]
    if HAS_BIAS:
        if EVEN:
            bias = tl.load(b_ptr + cols).to(tl.float32)
        else:
            bias = tl.load(b_ptr + cols, mask=cols < C, other=0).to(tl.float32)
        y = y + bias[None, :]
    y = y / (1.0 + tl.exp(-y))
    if EVEN:
        tl.store(o_ptr + offsets, y.to(o_ptr.dtype.element_ty))
    else:
        tl.store(o_ptr + offsets, y.to(o_ptr.dtype.element_ty), mask=mask)


def _entry010(x, gamma, bias, rms_scale, eps):
    b, c, t, h, w = x.shape
    scale = c**0.5 if rms_scale is None else float(rms_scale)
    out_dtype = torch.promote_types(x.dtype, gamma.dtype)
    if not (
        x.stride(1) == 1
        and x.stride(4) == c
        and (x.stride(3) == w * c)
        and (x.stride(2) == h * w * c)
        and (x.stride(0) == t * h * w * c)
    ):
        x = x.contiguous(memory_format=torch.channels_last_3d)
    if (
        out_dtype == x.dtype
        and (not torch._C._overlaps(x, gamma))
        and (bias is None or not torch._C._overlaps(x, bias))
    ):
        out = x
    else:
        out = torch.empty_strided(
            tuple(x.shape), tuple(x.stride()), dtype=out_dtype, device=x.device
        )
    n_rows = b * t * h * w
    if n_rows == 0:
        return out
    block = triton.next_power_of_2(int(c))
    rows = max(4, min(32, 8192 // block))
    tiles = (n_rows + rows - 1) // rows
    has_bias = bias is not None
    g = gamma.reshape(c)
    if not g.is_contiguous():
        g = g.contiguous()
    bias_arg = g
    if has_bias:
        bias_arg = bias.reshape(c)
        if not bias_arg.is_contiguous():
            bias_arg = bias_arg.contiguous()
    even = n_rows % rows == 0 and block == c
    _kernel010[tiles,](
        x,
        g,
        bias_arg,
        out,
        n_rows,
        scale,
        float(eps),
        C=c,
        BLOCK=block,
        ROWS=rows,
        HAS_BIAS=has_bias,
        EVEN=even,
        num_warps=8 if rows * block >= 2048 else 4,
    )
    return out


@triton.jit
def _kernel110(
    x_ptr,
    g_ptr,
    b_ptr,
    o_ptr,
    n_rows,
    scale,
    eps,
    C: tl.constexpr,
    BLOCK: tl.constexpr,
    ROWS: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    EVEN: tl.constexpr,
):
    tile = tl.program_id(0)
    row_ids = tile.to(tl.int64) * ROWS + tl.arange(0, ROWS).to(tl.int64)
    local_rows = tl.arange(0, ROWS)
    cols = tl.max_contiguous(tl.multiple_of(tl.arange(0, BLOCK), BLOCK), BLOCK)
    offsets = row_ids[:, None] * C + cols[None, :]
    if EVEN:
        x = tl.load(x_ptr + offsets).to(tl.float32)
    else:
        mask = (row_ids[:, None] < n_rows) & (cols[None, :] < C)
        x = tl.load(x_ptr + offsets, mask=mask, other=0).to(tl.float32)
    norm = tl.sqrt(tl.sum(x * x, axis=1))
    inv = 1.0 / tl.maximum(norm, eps)
    if EVEN:
        gamma = tl.load(g_ptr + local_rows[:, None] * C + cols[None, :]).to(
            tl.float32
        )
    else:
        gamma = tl.load(
            g_ptr + local_rows[:, None] * C + cols[None, :],
            mask=cols[None, :] < C,
            other=0,
        ).to(tl.float32)
    y = x * inv[:, None] * scale * gamma
    if HAS_BIAS:
        if EVEN:
            bias = tl.load(b_ptr + local_rows[:, None] * C + cols[None, :]).to(
                tl.float32
            )
        else:
            bias = tl.load(
                b_ptr + local_rows[:, None] * C + cols[None, :],
                mask=cols[None, :] < C,
                other=0,
            ).to(tl.float32)
        y = y + bias
    y = y / (1.0 + tl.exp(-y))
    if EVEN:
        tl.store(o_ptr + offsets, y.to(o_ptr.dtype.element_ty))
    else:
        tl.store(o_ptr + offsets, y.to(o_ptr.dtype.element_ty), mask=mask)


def _entry110(x, gamma, bias, rms_scale, eps):
    b, c, t, h, w = x.shape
    scale = c**0.5 if rms_scale is None else float(rms_scale)
    out_dtype = torch.promote_types(x.dtype, gamma.dtype)
    if not (
        x.stride(1) == 1
        and x.stride(4) == c
        and (x.stride(3) == w * c)
        and (x.stride(2) == h * w * c)
        and (x.stride(0) == t * h * w * c)
    ):
        x = x.contiguous(memory_format=torch.channels_last_3d)
    if (
        out_dtype == x.dtype
        and (not torch._C._overlaps(x, gamma))
        and (bias is None or not torch._C._overlaps(x, bias))
    ):
        out = x
    else:
        out = torch.empty_strided(
            tuple(x.shape), tuple(x.stride()), dtype=out_dtype, device=x.device
        )
    n_rows = b * t * h * w
    if n_rows == 0:
        return out
    block = triton.next_power_of_2(int(c))
    rows = max(4, min(64, 8192 // block))
    tiles = (n_rows + rows - 1) // rows
    has_bias = bias is not None
    g = gamma.reshape(c)
    if not g.is_contiguous():
        g = g.contiguous()
    bias_arg = g
    if has_bias:
        bias_arg = bias.reshape(c)
        if not bias_arg.is_contiguous():
            bias_arg = bias_arg.contiguous()
    rg = torch.empty((rows * c,), dtype=torch.float32, device=x.device)
    rb = (
        torch.empty((rows * c,), dtype=torch.float32, device=x.device)
        if has_bias
        else rg
    )
    if c == 128:
        _affine110[rows,](
            g,
            bias_arg,
            rg,
            rb,
            c,
            block,
            g.stride(0),
            bias_arg.stride(0),
            has_bias,
            num_warps=4,
            enable_fp_fusion=False,
            groups_per_cluster=8,
        )
    else:
        _affine110[rows,](
            g,
            bias_arg,
            rg,
            rb,
            c,
            block,
            g.stride(0),
            bias_arg.stride(0),
            has_bias,
            num_warps=4,
            enable_fp_fusion=False,
        )
    g = rg
    bias_arg = rb
    even = n_rows % rows == 0 and block == c
    _kernel110[tiles,](
        x,
        g,
        bias_arg,
        out,
        n_rows,
        scale,
        float(eps),
        C=c,
        BLOCK=block,
        ROWS=rows,
        HAS_BIAS=has_bias,
        EVEN=even,
        num_warps=8 if rows * block >= 2048 else 4,
    )
    return out


@triton.jit
def _affine110(
    G,
    B,
    RG,
    RB,
    C: tl.constexpr,
    BC: tl.constexpr,
    GS: tl.constexpr,
    BS: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    row = tl.program_id(0)
    col = tl.arange(0, BC)[None, :]
    g = tl.load(G + col * GS, col < C, 0).to(tl.float32)
    tl.store(RG + row * C + col, g, col < C)
    if HAS_BIAS:
        b = tl.load(B + col * BS, col < C, 0).to(tl.float32)
        tl.store(RB + row * C + col, b, col < C)


def _entry10(x, gamma, bias, rms_scale, eps):
    if x.numel() // x.size(1) >= 4096:
        return _entry110(x, gamma, bias, rms_scale, eps)
    return _entry010(x, gamma, bias, rms_scale, eps)


def _entry011(x, gamma, bias, rms_scale, eps):
    b, c, t, h, w = x.shape
    scale = c**0.5 if rms_scale is None else float(rms_scale)
    out_dtype = torch.promote_types(x.dtype, gamma.dtype)
    if not (
        x.stride(1) == 1
        and x.stride(4) == c
        and (x.stride(3) == w * c)
        and (x.stride(2) == h * w * c)
        and (x.stride(0) == t * h * w * c)
    ):
        x = x.contiguous(memory_format=torch.channels_last_3d)
    if (
        out_dtype == x.dtype
        and (not torch._C._overlaps(x, gamma))
        and (bias is None or not torch._C._overlaps(x, bias))
    ):
        out = x
    else:
        out = torch.empty_strided(
            tuple(x.shape), tuple(x.stride()), dtype=out_dtype, device=x.device
        )
    n_rows = b * t * h * w
    if n_rows == 0:
        return out
    block = triton.next_power_of_2(int(c))
    rows = max(4, min(32, 8192 // block))
    tiles = (n_rows + rows - 1) // rows
    has_bias = bias is not None
    g = gamma.reshape(c)
    if not g.is_contiguous():
        g = g.contiguous()
    bias_arg = g
    if has_bias:
        bias_arg = bias.reshape(c)
        if not bias_arg.is_contiguous():
            bias_arg = bias_arg.contiguous()
    even = n_rows % rows == 0 and block == c
    _kernel010[tiles,](
        x,
        g,
        bias_arg,
        out,
        n_rows,
        scale,
        float(eps),
        C=c,
        BLOCK=block,
        ROWS=rows,
        HAS_BIAS=has_bias,
        EVEN=even,
        num_warps=8 if rows * block >= 2048 else 4,
    )
    return out


def _entry111(x, gamma, bias, rms_scale, eps):
    b, c, t, h, w = x.shape
    scale = c**0.5 if rms_scale is None else float(rms_scale)
    out_dtype = torch.promote_types(x.dtype, gamma.dtype)
    if not (
        x.stride(1) == 1
        and x.stride(4) == c
        and (x.stride(3) == w * c)
        and (x.stride(2) == h * w * c)
        and (x.stride(0) == t * h * w * c)
    ):
        x = x.contiguous(memory_format=torch.channels_last_3d)
    if (
        out_dtype == x.dtype
        and (not torch._C._overlaps(x, gamma))
        and (bias is None or not torch._C._overlaps(x, bias))
    ):
        out = x
    else:
        out = torch.empty_strided(
            tuple(x.shape), tuple(x.stride()), dtype=out_dtype, device=x.device
        )
    n_rows = b * t * h * w
    if n_rows == 0:
        return out
    block = triton.next_power_of_2(int(c))
    rows = 64
    tiles = (n_rows + rows - 1) // rows
    has_bias = bias is not None
    g = gamma.reshape(c)
    if not g.is_contiguous():
        g = g.contiguous()
    bias_arg = g
    if has_bias:
        bias_arg = bias.reshape(c)
        if not bias_arg.is_contiguous():
            bias_arg = bias_arg.contiguous()
    rg = torch.empty((rows * c,), dtype=torch.float32, device=x.device)
    rb = (
        torch.empty((rows * c,), dtype=torch.float32, device=x.device)
        if has_bias
        else rg
    )
    if c == 128:
        _affine110[rows,](
            g,
            bias_arg,
            rg,
            rb,
            c,
            block,
            g.stride(0),
            bias_arg.stride(0),
            has_bias,
            num_warps=4,
            enable_fp_fusion=False,
            groups_per_cluster=8,
        )
    else:
        _affine110[rows,](
            g,
            bias_arg,
            rg,
            rb,
            c,
            block,
            g.stride(0),
            bias_arg.stride(0),
            has_bias,
            num_warps=4,
            enable_fp_fusion=False,
        )
    g = rg
    bias_arg = rb
    even = n_rows % rows == 0 and block == c
    _kernel110[tiles,](
        x,
        g,
        bias_arg,
        out,
        n_rows,
        scale,
        float(eps),
        C=c,
        BLOCK=block,
        ROWS=rows,
        HAS_BIAS=has_bias,
        EVEN=even,
        num_warps=8 if rows * block >= 2048 else 4,
        groups_per_cluster=64,
    )
    return out


def _entry11(x, gamma, bias, rms_scale, eps):
    if (
        x.numel() // x.size(1) >= 4096
        and x.size(1) == triton.next_power_of_2(x.size(1))
        and (x.numel() // x.size(1) % 64 == 0)
    ):
        return _entry111(x, gamma, bias, rms_scale, eps)
    return _entry011(x, gamma, bias, rms_scale, eps)


def wan_rmsnorm_silu(x, gamma, bias, rms_scale, eps):
    c = x.size(1)
    n = x.numel() // c
    if (
        x.dtype == torch.bfloat16
        and gamma.dtype == torch.bfloat16
        and (c >= 64)
        and (c <= 512)
        and (c == triton.next_power_of_2(c))
        and (n >= 4096)
        and (n % 64 == 0)
    ):
        return _entry11(x, gamma, bias, rms_scale, eps)
    return _entry10(x, gamma, bias, rms_scale, eps)


__all__ = ["wan_rmsnorm_silu"]
