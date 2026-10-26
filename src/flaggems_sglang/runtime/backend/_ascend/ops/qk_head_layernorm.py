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


@triton.jit
def _qk_ln_rows(
    q_ptr,
    k_ptr,
    q_out_ptr,
    k_out_ptr,
    total_rows,
    eps,
    D: tl.constexpr,
    ROWS: tl.constexpr,
    EVEN_ROWS: tl.constexpr,
):
    """Fast path: Q and K tiles back to back per program, no column."""
    pid = tl.program_id(0)
    rows = pid * ROWS + tl.arange(0, ROWS)
    cols = tl.arange(0, D)
    base = rows[:, None] * D + cols[None, :]

    if EVEN_ROWS:
        # ---- Q ----
        x = tl.load(q_ptr + base).to(tl.float32)
        mean = tl.sum(x, axis=1) / D
        xc = x - mean[:, None]
        var = tl.sum(xc * xc, axis=1) / D
        y = xc * tl.rsqrt(var + eps)[:, None]
        tl.store(q_out_ptr + base, y.to(q_out_ptr.dtype.element_ty))
        # ---- K ----
        x = tl.load(k_ptr + base).to(tl.float32)
        mean = tl.sum(x, axis=1) / D
        xc = x - mean[:, None]
        var = tl.sum(xc * xc, axis=1) / D
        y = xc * tl.rsqrt(var + eps)[:, None]
        tl.store(k_out_ptr + base, y.to(k_out_ptr.dtype.element_ty))
    else:
        m = (rows < total_rows)[:, None]
        # ---- Q ----
        x = tl.load(q_ptr + base, mask=m, other=0.0).to(tl.float32)
        mean = tl.sum(x, axis=1) / D
        xc = x - mean[:, None]
        var = tl.sum(xc * xc, axis=1) / D
        y = xc * tl.rsqrt(var + eps)[:, None]
        tl.store(q_out_ptr + base, y.to(q_out_ptr.dtype.element_ty), mask=m)
        # ---- K ----
        x = tl.load(k_ptr + base, mask=m, other=0.0).to(tl.float32)
        mean = tl.sum(x, axis=1) / D
        xc = x - mean[:, None]
        var = tl.sum(xc * xc, axis=1) / D
        y = xc * tl.rsqrt(var + eps)[:, None]
        tl.store(k_out_ptr + base, y.to(k_out_ptr.dtype.element_ty), mask=m)


@triton.jit
def _qk_ln_masked(
    q_ptr,
    k_ptr,
    q_out_ptr,
    k_out_ptr,
    total_rows,
    eps,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
    ROWS: tl.constexpr,
):
    """Generic fallback: row-count tail and/or non-pow2 head dim.

    Masked lanes are zeroed after the mean subtraction so they drop from the
    variance sum; row-tile tails are masked on load and store.
    """
    pid = tl.program_id(0)
    rows = pid * ROWS + tl.arange(0, ROWS)
    cols = tl.arange(0, BLOCK_D)
    rmask = (rows < total_rows)[:, None]
    cmask = (cols < D)[None, :]
    m = rmask & cmask
    base = rows[:, None] * D + cols[None, :]

    # ---- Q ----
    x = tl.load(q_ptr + base, mask=m, other=0.0).to(tl.float32)
    mean = tl.sum(x, axis=1) / D
    d = tl.where(cmask, x - mean[:, None], 0.0)
    var = tl.sum(d * d, axis=1) / D
    y = d * (1.0 / tl.sqrt(var + eps))[:, None]
    tl.store(q_out_ptr + base, y.to(q_out_ptr.dtype.element_ty), mask=m)

    # ---- K ----
    x = tl.load(k_ptr + base, mask=m, other=0.0).to(tl.float32)
    mean = tl.sum(x, axis=1) / D
    d = tl.where(cmask, x - mean[:, None], 0.0)
    var = tl.sum(d * d, axis=1) / D
    y = d * (1.0 / tl.sqrt(var + eps))[:, None]
    tl.store(k_out_ptr + base, y.to(k_out_ptr.dtype.element_ty), mask=m)


def _next_pow2(n):
    """Inline bit_length shift; same result as triton.next_power_of_2."""
    return 1 << (max(n, 1) - 1).bit_length()


_ROWS_PER_PROG = 128
_NUM_WARPS = 4


def qk_head_layernorm(q, k, eps):
    B, S, H, D = q.shape
    if q.stride(3) != 1 or not q.is_contiguous():
        q = q.contiguous()
    if k.stride(3) != 1 or not k.is_contiguous():
        k = k.contiguous()

    q_out = torch.empty_like(q)
    k_out = torch.empty_like(k)

    total_rows = B * S * H
    if total_rows == 0 or D == 0:
        return q_out, k_out

    block_d = _next_pow2(D)
    if block_d == D:
        rows = _ROWS_PER_PROG
        even_rows = (total_rows % rows) == 0
        grid0 = (total_rows + rows - 1) // rows
        _qk_ln_rows[(grid0,)](
            q,
            k,
            q_out,
            k_out,
            total_rows,
            eps,
            D=D,
            ROWS=rows,
            EVEN_ROWS=even_rows,
            num_warps=_NUM_WARPS,
        )
    else:
        rows = 64
        grid0 = (total_rows + rows - 1) // rows
        _qk_ln_masked[(grid0,)](
            q,
            k,
            q_out,
            k_out,
            total_rows,
            eps,
            D=D,
            BLOCK_D=block_d,
            ROWS=rows,
            num_warps=_NUM_WARPS,
        )
    return q_out, k_out


__all__ = ["qk_head_layernorm"]
