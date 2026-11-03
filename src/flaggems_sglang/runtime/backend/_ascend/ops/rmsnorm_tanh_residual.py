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


@triton.jit
def _rmsnorm_tanh_residual_kernel(
    x_ptr,
    g_ptr,
    res_ptr,
    w_ptr,
    out_ptr,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_S: tl.constexpr,
    REPEAT: tl.constexpr,
    EPS: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = pid * BLOCK_S + tl.arange(0, BLOCK_S)
    cols = tl.arange(0, BLOCK_D)

    offs = rows[:, None] * D + cols[None, :]

    x = tl.load(x_ptr + offs).to(tl.float32)
    rstd = tl.rsqrt(tl.sum(x * x, axis=1) / D + EPS)

    # Reference rounding 1: (x * rsqrt) lands in the storage dtype before
    # the weight multiply.
    normed = (x * rstd[:, None]).to(x_ptr.dtype.element_ty).to(tl.float32)

    w = tl.load(w_ptr + cols).to(tl.float32)
    # Reference rounding 2: normed * weight is a storage-dtype multiply.
    normed = (normed * w[None, :]).to(x_ptr.dtype.element_ty).to(tl.float32)

    g_rows = rows // REPEAT
    g = tl.load(g_ptr + g_rows[:, None] * D + cols[None, :]).to(tl.float32)
    # Reference rounding 3: tanh is evaluated in fp32, then rounded.
    gated = tl.math.tanh(g).to(x_ptr.dtype.element_ty).to(tl.float32)

    res = tl.load(res_ptr + offs).to(tl.float32)
    # Reference roundings 4+5: gated * normed rounds, then the residual add
    # rounds once more.
    prod = (gated * normed).to(x_ptr.dtype.element_ty).to(tl.float32)
    y = res + prod
    tl.store(out_ptr + offs, y.to(out_ptr.dtype.element_ty))


@triton.jit
def _rmsnorm_tanh_residual_kernel_masked(
    x_ptr,
    g_ptr,
    res_ptr,
    w_ptr,
    out_ptr,
    n_rows,
    eps,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_S: tl.constexpr,
    REPEAT: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = pid * BLOCK_S + tl.arange(0, BLOCK_S)
    cols = tl.arange(0, BLOCK_D)

    row_ok = rows < n_rows
    col_ok = cols < D
    mask = row_ok[:, None] & col_ok[None, :]

    offs = rows[:, None] * D + cols[None, :]

    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    ssq = tl.sum(x * x, axis=1)
    rstd = tl.rsqrt(ssq / D + eps)

    normed = (x * rstd[:, None]).to(x_ptr.dtype.element_ty).to(tl.float32)

    w = tl.load(w_ptr + cols, mask=col_ok, other=0.0).to(tl.float32)
    normed = (normed * w[None, :]).to(x_ptr.dtype.element_ty).to(tl.float32)

    g_rows = rows // REPEAT
    g = tl.load(
        g_ptr + g_rows[:, None] * D + cols[None, :], mask=mask, other=0.0
    ).to(tl.float32)
    gated = tl.math.tanh(g).to(x_ptr.dtype.element_ty).to(tl.float32)

    res = tl.load(res_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    prod = (gated * normed).to(x_ptr.dtype.element_ty).to(tl.float32)
    y = res + prod
    tl.store(out_ptr + offs, y.to(out_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _rmsnorm_tanh_residual_kernel_cols(
    x_ptr,
    g_ptr,
    res_ptr,
    w_ptr,
    out_ptr,
    n_rows,
    eps,
    D: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_C: tl.constexpr,
    REPEAT: tl.constexpr,
):
    """Column-loop geometry for large batches."""
    pid = tl.program_id(0)
    rows = pid * BLOCK_S + tl.arange(0, BLOCK_S)
    row_mask = (rows < n_rows)[:, None]
    cols = tl.max_contiguous(
        tl.multiple_of(tl.arange(0, BLOCK_C), BLOCK_C), BLOCK_C
    )

    # Pass 1: sum of squares over all column tiles.
    ssq = tl.zeros((BLOCK_S,), dtype=tl.float32)
    for c0 in range(0, D, BLOCK_C):
        c = c0 + cols
        m = row_mask & (c < D)[None, :]
        xv = tl.load(
            x_ptr + rows[:, None] * D + c[None, :], mask=m, other=0.0
        ).to(tl.float32)
        ssq += tl.sum(xv * xv, axis=1)
    rstd = tl.rsqrt(ssq / D + eps)

    # Pass 2: normalize / gate / residual for each column tile.
    for c0 in range(0, D, BLOCK_C):
        c = c0 + cols
        m = row_mask & (c < D)[None, :]
        offs = rows[:, None] * D + c[None, :]
        x = tl.load(x_ptr + offs, mask=m, other=0.0).to(tl.float32)

        # Reference rounding 1: (x * rsqrt) lands in the storage dtype
        # before the weight multiply.
        normed = (x * rstd[:, None]).to(x_ptr.dtype.element_ty).to(tl.float32)

        w = tl.load(w_ptr + c, mask=c < D, other=0.0).to(tl.float32)
        # Reference rounding 2: normed * weight is a storage-dtype multiply.
        normed = (
            (normed * w[None, :]).to(x_ptr.dtype.element_ty).to(tl.float32)
        )

        g_rows = rows // REPEAT
        g = tl.load(
            g_ptr + g_rows[:, None] * D + c[None, :], mask=m, other=0.0
        ).to(tl.float32)
        # Reference rounding 3: tanh in fp32, then rounded.
        gated = tl.math.tanh(g).to(x_ptr.dtype.element_ty).to(tl.float32)

        res = tl.load(res_ptr + offs, mask=m, other=0.0).to(tl.float32)
        # Reference roundings 4+5.
        prod = (gated * normed).to(x_ptr.dtype.element_ty).to(tl.float32)
        y = res + prod
        tl.store(out_ptr + offs, y.to(out_ptr.dtype.element_ty), mask=m)


_TILE_BUDGET = 8192

_ROWS_COLS_KERNEL = 4096

_COLS_BLOCK_S = 16
_COLS_BLOCK_C = 512
_COLS_NUM_WARPS = 1
_COLS_NUM_STAGES = 1


def _next_pow2(n):
    return 1 << (n - 1).bit_length() if n > 1 else 1


def _launch_plan(dim, n_rows, g_rows, eps):
    """Pick geometry and tile shape for one problem signature."""
    repeat = n_rows // g_rows if g_rows else 1
    if repeat <= 0:
        repeat = 1

    if n_rows >= _ROWS_COLS_KERNEL:
        block_s = _COLS_BLOCK_S
        block_c = min(_COLS_BLOCK_C, _next_pow2(dim))
        grid0 = ((n_rows + block_s - 1) // block_s,)
        return (
            grid0,
            "cols",
            dict(D=dim, BLOCK_S=block_s, BLOCK_C=block_c, REPEAT=repeat),
            (_COLS_NUM_WARPS, _COLS_NUM_STAGES),
        )

    block_d = _next_pow2(dim)
    block_s = min(n_rows, max(1, _TILE_BUDGET // block_d))
    if n_rows % block_s != 0:
        block_s = 1
    # Two warps only on the 4096-wide shapes; one warp wins or ties elsewhere.
    num_warps = 2 if dim > 2048 else 1
    grid0 = ((n_rows + block_s - 1) // block_s,)

    if dim == block_d and n_rows % block_s == 0:
        # Exact tiling over a power-of-two hidden size: no masks anywhere,
        # and eps/repeat ride along as compile-time constants.
        return (
            grid0,
            "fast",
            dict(
                D=dim, BLOCK_D=block_d, BLOCK_S=block_s, REPEAT=repeat, EPS=eps
            ),
            (num_warps,),
        )

    # Padded columns and/or ragged final tile: masked variant with
    # n_rows/eps as runtime values.
    return (
        grid0,
        "masked",
        dict(D=dim, BLOCK_D=block_d, BLOCK_S=block_s, REPEAT=repeat),
        (num_warps,),
    )


def rmsnorm_tanh_residual(x, gate, residual, weight, eps):
    if not x.is_contiguous():
        x = x.contiguous()
    if not gate.is_contiguous():
        gate = gate.contiguous()
    if not residual.is_contiguous():
        residual = residual.contiguous()
    if not weight.is_contiguous():
        weight = weight.contiguous()

    dim = x.shape[-1]
    out = torch.empty_like(x)

    n_rows = x.numel()
    if n_rows == 0 or dim == 0:
        return out
    n_rows //= dim

    g_rows = gate.numel() // dim

    grid0, kind, block, warps = _launch_plan(dim, n_rows, g_rows, eps)

    if kind == "cols":
        nw, ns = warps
        _rmsnorm_tanh_residual_kernel_cols[grid0](
            x,
            gate,
            residual,
            weight,
            out,
            n_rows,
            eps,
            **block,
            num_warps=nw,
            num_stages=ns,
        )
    elif kind == "fast":
        (nw,) = warps
        _rmsnorm_tanh_residual_kernel[grid0](
            x,
            gate,
            residual,
            weight,
            out,
            **block,
            num_warps=nw,
        )
    else:
        (nw,) = warps
        _rmsnorm_tanh_residual_kernel_masked[grid0](
            x,
            gate,
            residual,
            weight,
            out,
            n_rows,
            eps,
            **block,
            num_warps=nw,
        )
    return out


__all__ = ["rmsnorm_tanh_residual"]
