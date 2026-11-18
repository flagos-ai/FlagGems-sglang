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

import torch
import triton
import triton.language as tl


@triton.jit
def _zimage_qk_rmsnorm_contig_kernel(
    X,
    W,
    Y,
    N,
    eps,
    BLOCK_M: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    TILES: tl.constexpr,
    EVEN: tl.constexpr,
):
    pid = tl.program_id(0)
    d = tl.arange(0, HEAD_DIM)
    w = tl.load(W + d).to(tl.float32)

    for t in tl.static_range(TILES):
        tile = pid * TILES + t
        # Flat element range covering BLOCK_M full rows; one scalar base plus
        # one arange keeps the address chain to a single add per element.
        offs = tile * (BLOCK_M * HEAD_DIM) + tl.arange(0, BLOCK_M * HEAD_DIM)

        if EVEN:
            x = tl.load(X + offs)
        else:
            x = tl.load(X + offs, mask=offs < N * HEAD_DIM, other=0.0)

        xf = x.to(tl.float32)
        x2 = tl.reshape(xf, (BLOCK_M, HEAD_DIM))
        ss = tl.sum(x2 * x2, axis=1)
        rms = tl.rsqrt(ss / HEAD_DIM + eps)

        y = x2 * rms[:, None] * w[None, :]
        y = tl.reshape(y, (BLOCK_M * HEAD_DIM,)).to(x.dtype)

        if EVEN:
            tl.store(Y + offs, y)
        else:
            tl.store(Y + offs, y, mask=offs < N * HEAD_DIM)


@triton.jit
def _zimage_qk_rmsnorm_strided_kernel(
    X,
    W,
    Y,
    N,
    H,
    S,
    stride_b,
    stride_s,
    stride_h,
    eps,
    BLOCK_M: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = rows < N

    # Decompose the flat row index into (b, s, h) to honour arbitrary strides.
    h = rows % H
    bs = rows // H
    s = bs % S
    b = bs // S
    base = b * stride_b + s * stride_s + h * stride_h

    d = tl.arange(0, HEAD_DIM)
    offs = base[:, None] + d[None, :]
    mask = mask_m[:, None]

    x = tl.load(X + offs, mask=mask, other=0.0)
    xf = x.to(tl.float32)

    ss = tl.sum(xf * xf, axis=1)
    rms = tl.rsqrt(ss / HEAD_DIM + eps)

    w = tl.load(W + d).to(tl.float32)
    y = (xf * rms[:, None] * w[None, :]).to(x.dtype)

    tl.store(Y + rows[:, None] * HEAD_DIM + d[None, :], y, mask=mask)


def _contig_config(n_rows):
    """Launch config measured per size on the reference platform."""
    if n_rows <= 16384:
        return 8, 2, 1, 1
    if n_rows <= 65536:
        return 16, 1, 2, 1
    return 16, 2, 2, 1


def zimage_qk_rmsnorm(x, weight, eps):
    """Per-head QK RMSNorm; signature matches the reference exactly."""
    head_dim = x.shape[-1]
    H = x.shape[-2]
    S = x.shape[-3] if x.dim() >= 3 else 1
    B = x.shape[-4] if x.dim() >= 4 else 1

    x_flat_rows = B * S * H
    out = torch.empty((B, S, H, head_dim), dtype=x.dtype, device=x.device)
    if x_flat_rows == 0:
        return out

    if x.is_contiguous():
        block_m, num_warps, num_stages, tiles = _contig_config(x_flat_rows)
        blocks_per_prog = block_m * tiles
        grid = (triton.cdiv(x_flat_rows, blocks_per_prog),)
        _zimage_qk_rmsnorm_contig_kernel[grid](
            x,
            weight,
            out,
            x_flat_rows,
            eps,
            BLOCK_M=block_m,
            HEAD_DIM=head_dim,
            TILES=tiles,
            # rows % 64 == 0 implies rows % BLOCK_M == 0 for every configured
            # BLOCK_M, so the masked variant only compiles for ragged sizes.
            EVEN=x_flat_rows % 64 == 0,
            num_warps=num_warps,
            num_stages=num_stages,
        )
        return out

    # Heads are packed (stride(-2) == head_dim); spatial strides are uniform.
    stride_h = x.stride(-2)
    stride_s = x.stride(-3) if x.dim() >= 3 else 0
    stride_b = x.stride(-4) if x.dim() >= 4 else 0

    # 8 rows x 2 warps measured fastest across the fused-qkv slice sizes.
    grid = (triton.cdiv(x_flat_rows, 8),)
    _zimage_qk_rmsnorm_strided_kernel[grid](
        x,
        weight,
        out,
        x_flat_rows,
        H,
        S,
        stride_b,
        stride_s,
        stride_h,
        eps,
        BLOCK_M=8,
        HEAD_DIM=head_dim,
        num_warps=2,
        num_stages=1,
    )
    return out


__all__ = ["zimage_qk_rmsnorm"]
