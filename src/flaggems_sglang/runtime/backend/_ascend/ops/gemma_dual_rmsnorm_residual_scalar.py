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

"""Operator: activation_norm/gemma_dual_rmsnorm_residual_scalar"""

import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["S"])
def _gemma_dual_rmsnorm_pad_kernel(
    x1_ptr,
    x2_ptr,
    res_ptr,
    w1_ptr,
    w2_ptr,
    w3_ptr,
    scalar_ptr,
    out_ptr,
    S,
    eps1,
    eps2,
    eps3,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_S: tl.constexpr,
    ROW_MASK: tl.constexpr,
):
    """Resident one-tile path: whole rows in one program (D <= 4096)."""
    pid = tl.program_id(0)
    rows = pid * BLOCK_S + tl.arange(0, BLOCK_S)
    cols = tl.arange(0, BLOCK_D)
    base = rows[:, None] * D
    if ROW_MASK:
        m = (rows < S)[:, None] & (cols < D)[None, :]
        f1 = tl.load(x1_ptr + base + cols[None, :], mask=m, other=0.0).to(
            tl.float32
        )
        f2 = tl.load(x2_ptr + base + cols[None, :], mask=m, other=0.0).to(
            tl.float32
        )
        fr = tl.load(res_ptr + base + cols[None, :], mask=m, other=0.0).to(
            tl.float32
        )
        rstd1 = tl.rsqrt(tl.sum(f1 * f1, axis=1) / D + eps1)
        rstd2 = tl.rsqrt(tl.sum(f2 * f2, axis=1) / D + eps2)
        w1 = tl.load(w1_ptr + cols, mask=cols < D, other=0.0).to(tl.float32)
        w2 = tl.load(w2_ptr + cols, mask=cols < D, other=0.0).to(tl.float32)
        s12 = (
            f1 * rstd1[:, None] * w1[None, :]
            + f2 * rstd2[:, None] * w2[None, :]
        )
        rstd3 = tl.rsqrt(tl.sum(s12 * s12, axis=1) / D + eps3)
        w3 = tl.load(w3_ptr + cols, mask=cols < D, other=0.0).to(tl.float32)
        scalar = tl.load(scalar_ptr).to(tl.float32)
        out = (s12 * rstd3[:, None] * w3[None, :] + fr) * scalar
        tl.store(
            out_ptr + base + cols[None, :],
            out.to(out_ptr.dtype.element_ty),
            mask=m,
        )
    else:
        m = (cols < D)[None, :]
        f1 = tl.load(x1_ptr + base + cols[None, :], mask=m, other=0.0).to(
            tl.float32
        )
        f2 = tl.load(x2_ptr + base + cols[None, :], mask=m, other=0.0).to(
            tl.float32
        )
        fr = tl.load(res_ptr + base + cols[None, :], mask=m, other=0.0).to(
            tl.float32
        )
        rstd1 = tl.rsqrt(tl.sum(f1 * f1, axis=1) / D + eps1)
        rstd2 = tl.rsqrt(tl.sum(f2 * f2, axis=1) / D + eps2)
        w1 = tl.load(w1_ptr + cols, mask=cols < D, other=0.0).to(tl.float32)
        w2 = tl.load(w2_ptr + cols, mask=cols < D, other=0.0).to(tl.float32)
        s12 = (
            f1 * rstd1[:, None] * w1[None, :]
            + f2 * rstd2[:, None] * w2[None, :]
        )
        rstd3 = tl.rsqrt(tl.sum(s12 * s12, axis=1) / D + eps3)
        w3 = tl.load(w3_ptr + cols, mask=cols < D, other=0.0).to(tl.float32)
        scalar = tl.load(scalar_ptr).to(tl.float32)
        out = (s12 * rstd3[:, None] * w3[None, :] + fr) * scalar
        tl.store(
            out_ptr + base + cols[None, :],
            out.to(out_ptr.dtype.element_ty),
            mask=m,
        )


@triton.jit(do_not_specialize=["S"])
def _gemma_dual_rmsnorm_loop_kernel(
    x1_ptr,
    x2_ptr,
    res_ptr,
    w1_ptr,
    w2_ptr,
    w3_ptr,
    scalar_ptr,
    out_ptr,
    S,
    eps1,
    eps2,
    eps3,
    D: tl.constexpr,
    CHUNK: tl.constexpr,
    NCHUNK: tl.constexpr,
    BLOCK_S: tl.constexpr,
    ROW_MASK: tl.constexpr,
):
    """Chunked loop path for wide rows (D > 4096, D % CHUNK == 0)."""
    pid = tl.program_id(0)
    rows = pid * BLOCK_S + tl.arange(0, BLOCK_S)
    cols = tl.arange(0, CHUNK)
    base = rows[:, None] * D
    if ROW_MASK:
        m = (rows < S)[:, None]
        ssq1 = tl.zeros([BLOCK_S], tl.float32)
        ssq2 = tl.zeros([BLOCK_S], tl.float32)
        for c in tl.range(0, NCHUNK, num_stages=2):
            off = c * CHUNK + cols[None, :]
            f1 = tl.load(x1_ptr + base + off, mask=m, other=0.0).to(tl.float32)
            f2 = tl.load(x2_ptr + base + off, mask=m, other=0.0).to(tl.float32)
            ssq1 += tl.sum(f1 * f1, 1)
            ssq2 += tl.sum(f2 * f2, 1)
        rstd1 = tl.rsqrt(ssq1 / D + eps1)
        rstd2 = tl.rsqrt(ssq2 / D + eps2)
        ssq3 = tl.zeros([BLOCK_S], tl.float32)
        for c in tl.range(0, NCHUNK, num_stages=2):
            off = c * CHUNK + cols[None, :]
            f1 = tl.load(x1_ptr + base + off, mask=m, other=0.0).to(tl.float32)
            f2 = tl.load(x2_ptr + base + off, mask=m, other=0.0).to(tl.float32)
            w1 = tl.load(w1_ptr + c * CHUNK + cols).to(tl.float32)
            w2 = tl.load(w2_ptr + c * CHUNK + cols).to(tl.float32)
            s12 = (
                f1 * rstd1[:, None] * w1[None, :]
                + f2 * rstd2[:, None] * w2[None, :]
            )
            ssq3 += tl.sum(s12 * s12, 1)
        rstd3 = tl.rsqrt(ssq3 / D + eps3)
        scalar = tl.load(scalar_ptr).to(tl.float32)
        for c in tl.range(0, NCHUNK, num_stages=2):
            off = c * CHUNK + cols[None, :]
            f1 = tl.load(x1_ptr + base + off, mask=m, other=0.0).to(tl.float32)
            f2 = tl.load(x2_ptr + base + off, mask=m, other=0.0).to(tl.float32)
            fr = tl.load(res_ptr + base + off, mask=m, other=0.0).to(
                tl.float32
            )
            w1 = tl.load(w1_ptr + c * CHUNK + cols).to(tl.float32)
            w2 = tl.load(w2_ptr + c * CHUNK + cols).to(tl.float32)
            w3 = tl.load(w3_ptr + c * CHUNK + cols).to(tl.float32)
            s12 = (
                f1 * rstd1[:, None] * w1[None, :]
                + f2 * rstd2[:, None] * w2[None, :]
            )
            out = (s12 * rstd3[:, None] * w3[None, :] + fr) * scalar
            tl.store(
                out_ptr + base + off, out.to(out_ptr.dtype.element_ty), mask=m
            )
    else:
        ssq1 = tl.zeros([BLOCK_S], tl.float32)
        ssq2 = tl.zeros([BLOCK_S], tl.float32)
        for c in tl.range(0, NCHUNK, num_stages=2):
            off = c * CHUNK + cols[None, :]
            f1 = tl.load(x1_ptr + base + off).to(tl.float32)
            f2 = tl.load(x2_ptr + base + off).to(tl.float32)
            ssq1 += tl.sum(f1 * f1, 1)
            ssq2 += tl.sum(f2 * f2, 1)
        rstd1 = tl.rsqrt(ssq1 / D + eps1)
        rstd2 = tl.rsqrt(ssq2 / D + eps2)
        ssq3 = tl.zeros([BLOCK_S], tl.float32)
        for c in tl.range(0, NCHUNK, num_stages=2):
            off = c * CHUNK + cols[None, :]
            f1 = tl.load(x1_ptr + base + off).to(tl.float32)
            f2 = tl.load(x2_ptr + base + off).to(tl.float32)
            w1 = tl.load(w1_ptr + c * CHUNK + cols).to(tl.float32)
            w2 = tl.load(w2_ptr + c * CHUNK + cols).to(tl.float32)
            s12 = (
                f1 * rstd1[:, None] * w1[None, :]
                + f2 * rstd2[:, None] * w2[None, :]
            )
            ssq3 += tl.sum(s12 * s12, 1)
        rstd3 = tl.rsqrt(ssq3 / D + eps3)
        scalar = tl.load(scalar_ptr).to(tl.float32)
        for c in tl.range(0, NCHUNK, num_stages=2):
            off = c * CHUNK + cols[None, :]
            f1 = tl.load(x1_ptr + base + off).to(tl.float32)
            f2 = tl.load(x2_ptr + base + off).to(tl.float32)
            fr = tl.load(res_ptr + base + off).to(tl.float32)
            w1 = tl.load(w1_ptr + c * CHUNK + cols).to(tl.float32)
            w2 = tl.load(w2_ptr + c * CHUNK + cols).to(tl.float32)
            w3 = tl.load(w3_ptr + c * CHUNK + cols).to(tl.float32)
            s12 = (
                f1 * rstd1[:, None] * w1[None, :]
                + f2 * rstd2[:, None] * w2[None, :]
            )
            out = (s12 * rstd3[:, None] * w3[None, :] + fr) * scalar
            tl.store(out_ptr + base + off, out.to(out_ptr.dtype.element_ty))


def _next_pow2(n):
    return 1 << (n - 1).bit_length() if n > 1 else 1


def _prev_pow2(n):
    return 1 << (n.bit_length() - 1) if n > 0 else 0


_PAD_TILE_ELEMS = 8192

_LOOP_CHUNK = 256
_LOOP_MAX_BLOCK_S = 16


def gemma_dual_rmsnorm_residual_scalar(
    x1, weight1, x2, weight2, weight3, residual, scalar, eps1, eps2, eps3
):
    if not x1.is_contiguous():
        x1 = x1.contiguous()
    if not x2.is_contiguous():
        x2 = x2.contiguous()
    if not residual.is_contiguous():
        residual = residual.contiguous()
    M, D = x1.shape
    S = M
    out = torch.empty_like(x1)
    if S == 0 or D == 0:
        return out

    if _next_pow2(D) > _PAD_TILE_ELEMS and D % _LOOP_CHUNK == 0:
        block_s = min(_prev_pow2(S), _LOOP_MAX_BLOCK_S)
        row_mask = S % block_s != 0
        grid = ((S + block_s - 1) // block_s,)
        _gemma_dual_rmsnorm_loop_kernel[grid](
            x1,
            x2,
            residual,
            weight1,
            weight2,
            weight3,
            scalar,
            out,
            S,
            eps1,
            eps2,
            eps3,
            D=D,
            CHUNK=_LOOP_CHUNK,
            NCHUNK=D // _LOOP_CHUNK,
            BLOCK_S=block_s,
            ROW_MASK=row_mask,
            num_warps=8,
        )
        return out

    block_d = _next_pow2(D)
    cap = min(max(_PAD_TILE_ELEMS // block_d, 1), 4)
    block_s = cap
    if block_s > S:
        block_s = max(_next_pow2(S), 1)
    row_mask = S % block_s != 0
    grid = ((S + block_s - 1) // block_s,)
    num_warps = max(min((block_s * block_d) // 1024, 8), 1)
    _gemma_dual_rmsnorm_pad_kernel[grid](
        x1,
        x2,
        residual,
        weight1,
        weight2,
        weight3,
        scalar,
        out,
        S,
        eps1,
        eps2,
        eps3,
        D=D,
        BLOCK_D=block_d,
        BLOCK_S=block_s,
        ROW_MASK=row_mask,
        num_warps=num_warps,
    )
    return out


__all__ = ["gemma_dual_rmsnorm_residual_scalar"]
