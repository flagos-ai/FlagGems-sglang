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

"""Operator: activation_norm/gemma_rmsnorm_residual_scalar"""

import torch
import triton
import triton.language as tl


@triton.jit
def _gemma_rmsnorm_residual_scalar_kernel(
    x_ptr,
    weight_ptr,
    residual_ptr,
    scalar_ptr,
    out_ptr,
    n_rows,
    n_cols,
    eps,
    BLOCK_N: tl.constexpr,
    ROWS: tl.constexpr,
    NUM_CHUNKS: tl.constexpr,
    MASKED: tl.constexpr,
    EVEN_M: tl.constexpr,
):
    pid = tl.program_id(0)
    scale = tl.load(scalar_ptr).to(tl.float32)

    if NUM_CHUNKS == 1:
        # Whole row fits one tile: reduce and store from the same load.
        cols = tl.max_contiguous(
            tl.multiple_of(tl.arange(0, BLOCK_N), BLOCK_N), BLOCK_N
        )
        if MASKED:
            cmask = cols < n_cols
            weight = tl.load(weight_ptr + cols, mask=cmask, other=0.0).to(
                tl.float32
            )
        else:
            weight = tl.load(weight_ptr + cols).to(tl.float32)
        for i in tl.static_range(ROWS):
            if EVEN_M or (pid * ROWS + i) < n_rows:
                row = pid * ROWS + i
                base = row * n_cols
                if MASKED:
                    x = tl.load(x_ptr + base + cols, mask=cmask, other=0.0).to(
                        tl.float32
                    )
                    residual = tl.load(
                        residual_ptr + base + cols, mask=cmask, other=0.0
                    ).to(tl.float32)
                    inv = tl.math.rsqrt(tl.sum(x * x, axis=0) / n_cols + eps)
                    tl.store(
                        out_ptr + base + cols,
                        ((x * inv * weight + residual) * scale).to(
                            out_ptr.dtype.element_ty
                        ),
                        mask=cmask,
                    )
                else:
                    x = tl.load(x_ptr + base + cols).to(tl.float32)
                    residual = tl.load(residual_ptr + base + cols).to(
                        tl.float32
                    )
                    inv = tl.math.rsqrt(tl.sum(x * x, axis=0) / n_cols + eps)
                    tl.store(
                        out_ptr + base + cols,
                        ((x * inv * weight + residual) * scale).to(
                            out_ptr.dtype.element_ty
                        ),
                    )
    else:
        # Row wider than the tile cap: two passes per row.
        cols = tl.arange(0, BLOCK_N)
        for i in tl.static_range(ROWS):
            if (pid * ROWS + i) < n_rows:
                row = pid * ROWS + i
                base = row.to(tl.int64) * n_cols
                acc = 0.0
                for c in tl.static_range(NUM_CHUNKS):
                    off = c * BLOCK_N + cols
                    m = off < n_cols
                    x = tl.load(x_ptr + base + off, mask=m, other=0.0).to(
                        tl.float32
                    )
                    acc += tl.sum(x * x, axis=0)
                inv = tl.math.rsqrt(acc / n_cols + eps)
                for c in tl.static_range(NUM_CHUNKS):
                    off = c * BLOCK_N + cols
                    m = off < n_cols
                    x = tl.load(x_ptr + base + off, mask=m, other=0.0).to(
                        tl.float32
                    )
                    residual = tl.load(
                        residual_ptr + base + off, mask=m, other=0.0
                    ).to(tl.float32)
                    weight = tl.load(weight_ptr + off, mask=m, other=0.0).to(
                        tl.float32
                    )
                    tl.store(
                        out_ptr + base + off,
                        ((x * inv * weight + residual) * scale).to(
                            out_ptr.dtype.element_ty
                        ),
                        mask=m,
                    )


def gemma_rmsnorm_residual_scalar(x, weight, residual, scalar, eps):
    m, n = x.shape
    out = torch.empty_like(x)
    if m == 0:
        return out

    BLOCK_N = triton.next_power_of_2(n)
    if BLOCK_N <= 16384:
        num_chunks = 1
    else:
        BLOCK_N = 16384
        num_chunks = triton.cdiv(triton.next_power_of_2(n), BLOCK_N)

    if num_chunks == 1:
        if m >= 2048:
            if n >= 8192:
                rows, num_warps, num_stages = 4, 8, 3
            elif n >= 4096:
                rows, num_warps, num_stages = 4, 8, 2
            else:
                rows, num_warps, num_stages = 16, 4, 2
        elif m >= 256:
            if n >= 8192:
                rows, num_warps, num_stages = 4, 8, 2
            elif n >= 4096:
                rows, num_warps, num_stages = 4, 8, 1
            else:
                rows, num_warps, num_stages = 16, 8, 1
        elif m >= 128:
            rows, num_warps, num_stages = 8, 4, 2
        elif m >= 16:
            rows, num_warps, num_stages = 2, 4, 2
        else:
            rows = 1
            num_warps = 8 if n >= 4096 else 4
            num_stages = 2
        masked = (n & (n - 1)) != 0
        even_m = (m % rows) == 0
    else:
        rows, num_warps, num_stages = 4, 8, 2
        masked = True
        even_m = False

    grid = (triton.cdiv(m, rows),)
    _gemma_rmsnorm_residual_scalar_kernel[grid](
        x,
        weight,
        residual,
        scalar,
        out,
        m,
        n,
        eps,
        BLOCK_N=BLOCK_N,
        ROWS=rows,
        NUM_CHUNKS=num_chunks,
        MASKED=masked,
        EVEN_M=even_m,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return out


__all__ = ["gemma_rmsnorm_residual_scalar"]
