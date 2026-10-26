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

"""Fused affine-free LayerNorm over the head dimension of Q and K."""

import torch
import triton
import triton.language as tl

# Largest head dim handled by the tiled paths (fp32 tile = 16 KB at Dh=4096).
_MAX_SINGLE_TILE = 4096

_FLAT_TARGET_LANES = 4096
_FLAT_MAX_ROWS = 64

# Flat rows per program for the 2D-tile path (v7's bandwidth-bound config;
# the non-power-of-two head dims in the test suite are small, latency-bound).
_2D_ROWS = 32

_I32_MAX_ELEMS = 1 << 31


@triton.jit
def _qk_head_layernorm_flat_kernel(
    q_ptr,
    k_ptr,
    q_out_ptr,
    k_out_ptr,
    n_rows,
    eps,
    D: tl.constexpr,
    ROWS: tl.constexpr,
    BLOCK_D: tl.constexpr,
    NITER: tl.constexpr,
    WIDE: tl.constexpr,
):
    """Program ``(r, t)`` normalizes ``NITER`` flat tiles of tensor."""
    pid = tl.program_id(0)
    tens = tl.program_id(1)
    offs = tl.arange(0, ROWS * BLOCK_D)
    for i in tl.static_range(NITER):
        r0 = (pid + i * tl.num_programs(0)) * ROWS
        if WIDE:
            ptrs = r0.to(tl.int64) * D + offs
            total = n_rows.to(tl.int64) * D
        else:
            ptrs = r0 * D + offs
            total = n_rows * D
        rmask = ptrs < total
        if tens == 0:
            x = tl.load(q_ptr + ptrs, mask=rmask, other=0.0).to(tl.float32)
            x2 = tl.reshape(x, (ROWS, BLOCK_D))
            mean = tl.sum(x2, axis=1) / D
            d = x2 - mean[:, None]
            var = tl.sum(d * d, axis=1) / D
            y = (d * tl.rsqrt(var + eps)[:, None]).to(
                q_out_ptr.dtype.element_ty
            )
            tl.store(
                q_out_ptr + ptrs, tl.reshape(y, (ROWS * BLOCK_D,)), mask=rmask
            )
        else:
            x = tl.load(k_ptr + ptrs, mask=rmask, other=0.0).to(tl.float32)
            x2 = tl.reshape(x, (ROWS, BLOCK_D))
            mean = tl.sum(x2, axis=1) / D
            d = x2 - mean[:, None]
            var = tl.sum(d * d, axis=1) / D
            y = (d * tl.rsqrt(var + eps)[:, None]).to(
                k_out_ptr.dtype.element_ty
            )
            tl.store(
                k_out_ptr + ptrs, tl.reshape(y, (ROWS * BLOCK_D,)), mask=rmask
            )


@triton.jit
def _qk_head_layernorm_2d_kernel(
    q_ptr,
    k_ptr,
    q_out_ptr,
    k_out_ptr,
    n_rows,
    eps,
    D,
    ROWS: tl.constexpr,
    BLOCK_D: tl.constexpr,
    NITER: tl.constexpr,
    EVEN: tl.constexpr,
    WIDE: tl.constexpr,
):
    """Row-tile kernel for non-power-of-two head dims (v7 path kept)."""
    pid = tl.program_id(0)
    tens = tl.program_id(1)
    offs = tl.arange(0, BLOCK_D)
    for i in tl.static_range(NITER):
        tile = pid + i * tl.num_programs(0)
        row0 = tile * ROWS
        rows = row0 + tl.arange(0, ROWS)
        rmask = rows < n_rows
        if WIDE:
            ptrs = rows.to(tl.int64)[:, None] * D + offs[None, :]
        else:
            ptrs = rows[:, None] * D + offs[None, :]
        if tens == 0:
            if EVEN:
                x = tl.load(q_ptr + ptrs, mask=rmask[:, None], other=0.0).to(
                    tl.float32
                )
            else:
                x = tl.load(
                    q_ptr + ptrs,
                    mask=rmask[:, None] & (offs[None, :] < D),
                    other=0.0,
                ).to(tl.float32)
            mean = tl.sum(x, axis=1) / D
            d = x - mean[:, None]
            if not EVEN:
                # Padded lanes loaded as 0 give d = -mean; zero them so the
                # centered-variance reduction stays exact.
                d = tl.where(offs[None, :] < D, d, 0.0)
            var = tl.sum(d * d, axis=1) / D
            y = (d * tl.rsqrt(var + eps)[:, None]).to(
                q_out_ptr.dtype.element_ty
            )
            if EVEN:
                tl.store(q_out_ptr + ptrs, y, mask=rmask[:, None])
            else:
                tl.store(
                    q_out_ptr + ptrs,
                    y,
                    mask=rmask[:, None] & (offs[None, :] < D),
                )
        else:
            if EVEN:
                x = tl.load(k_ptr + ptrs, mask=rmask[:, None], other=0.0).to(
                    tl.float32
                )
            else:
                x = tl.load(
                    k_ptr + ptrs,
                    mask=rmask[:, None] & (offs[None, :] < D),
                    other=0.0,
                ).to(tl.float32)
            mean = tl.sum(x, axis=1) / D
            d = x - mean[:, None]
            if not EVEN:
                d = tl.where(offs[None, :] < D, d, 0.0)
            var = tl.sum(d * d, axis=1) / D
            y = (d * tl.rsqrt(var + eps)[:, None]).to(
                k_out_ptr.dtype.element_ty
            )
            if EVEN:
                tl.store(k_out_ptr + ptrs, y, mask=rmask[:, None])
            else:
                tl.store(
                    k_out_ptr + ptrs,
                    y,
                    mask=rmask[:, None] & (offs[None, :] < D),
                )


@triton.jit
def _layernorm_row_2pass(x_ptr, out_ptr, base, eps, D, BLOCK_D: tl.constexpr):
    """Two-pass fallback for rows longer than the register tile."""
    offs = tl.arange(0, BLOCK_D)
    acc_s = 0.0
    acc_sq = 0.0
    for start in range(0, D, BLOCK_D):
        idx = start + offs
        x = tl.load(x_ptr + base + idx, mask=idx < D, other=0.0).to(tl.float32)
        acc_s += tl.sum(x, axis=0)
        acc_sq += tl.sum(x * x, axis=0)
    mean = acc_s / D
    var = acc_sq / D - mean * mean
    rstd = tl.rsqrt(var + eps)
    for start in range(0, D, BLOCK_D):
        idx = start + offs
        x = tl.load(x_ptr + base + idx, mask=idx < D, other=0.0).to(tl.float32)
        y = (x - mean) * rstd
        tl.store(
            out_ptr + base + idx, y.to(out_ptr.dtype.element_ty), mask=idx < D
        )


@triton.jit
def _qk_head_layernorm_2pass_kernel(
    q_ptr,
    k_ptr,
    q_out_ptr,
    k_out_ptr,
    eps,
    D,
    BLOCK_D: tl.constexpr,
):
    """Program (row, reg): reg 0 normalizes Q's row, reg 1 K's row."""
    row = tl.program_id(0).to(tl.int64)
    reg = tl.program_id(1)
    base = row * D
    if reg == 0:
        _layernorm_row_2pass(q_ptr, q_out_ptr, base, eps, D, BLOCK_D)
    else:
        _layernorm_row_2pass(k_ptr, k_out_ptr, base, eps, D, BLOCK_D)


def qk_head_layernorm(q, k, eps):
    """Per-head affine-free LayerNorm of ``q`` and ``k`` in one launch.

    q, k: (B, S, H, Dh); the last dim is the head dim.  Returns (q_out, k_out).
    """
    if not q.is_contiguous():
        q = q.contiguous()
    if not k.is_contiguous():
        k = k.contiguous()

    b, s, h, d = q.shape
    rows = b * s * h

    q_out = torch.empty_like(q)
    k_out = torch.empty_like(k)

    block_d = triton.next_power_of_2(d)
    if block_d <= _MAX_SINGLE_TILE:
        wide = 1 if rows * block_d >= _I32_MAX_ELEMS else 0
        if block_d == d:
            flat_rows = min(
                triton.cdiv(_FLAT_TARGET_LANES, block_d), _FLAT_MAX_ROWS
            )
            grid0 = triton.cdiv(rows, flat_rows)
            _qk_head_layernorm_flat_kernel[(grid0, 2)](
                q,
                k,
                q_out,
                k_out,
                rows,
                eps,
                D=d,
                ROWS=flat_rows,
                BLOCK_D=block_d,
                NITER=1,
                WIDE=wide,
                num_warps=8,
            )
        else:
            # Non-power-of-two head dim: row tile with column masking.
            n_tiles = triton.cdiv(rows, _2D_ROWS)
            _qk_head_layernorm_2d_kernel[(n_tiles, 2)](
                q,
                k,
                q_out,
                k_out,
                rows,
                eps,
                d,
                ROWS=_2D_ROWS,
                BLOCK_D=block_d,
                NITER=1,
                EVEN=0,
                WIDE=wide,
                num_warps=8,
            )
    else:
        _qk_head_layernorm_2pass_kernel[(rows, 2)](
            q,
            k,
            q_out,
            k_out,
            eps,
            d,
            BLOCK_D=_MAX_SINGLE_TILE,
            num_warps=8,
        )
    return q_out, k_out


__all__ = ["qk_head_layernorm"]
