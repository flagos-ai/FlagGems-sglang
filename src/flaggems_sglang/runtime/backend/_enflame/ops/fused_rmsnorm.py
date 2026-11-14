# Copyright 2026, The FlagOS Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
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

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _fused_rmsnorm_kernel(
    x_ptr,
    weight_ptr,
    out_ptr,
    eps,
    row_width: tl.constexpr,
    block_size: tl.constexpr,
):
    row = tl.program_id(0)
    columns = tl.arange(0, block_size)
    mask = columns < row_width
    row_offsets = row * row_width + columns
    x = tl.load(x_ptr + row_offsets, mask=mask, other=0.0).to(tl.float32)
    mean_square = tl.sum(x * x, axis=0) / row_width
    inv_rms = 1.0 / tl.sqrt(mean_square + eps)
    weight = tl.load(weight_ptr + columns, mask=mask, other=0.0).to(tl.float32)
    tl.store(out_ptr + row_offsets, x * inv_rms * weight, mask=mask)


@triton.jit
def _fused_rmsnorm_rows_kernel(
    x_ptr,
    weight_ptr,
    out_ptr,
    M,
    N,
    NUM_BLOCKS,
    eps,
    R: tl.constexpr,
    BLOCK_N: tl.constexpr,
    NPROG: tl.constexpr,
    EVEN_ROWS: tl.constexpr,
    EXACT_COLS: tl.constexpr,
):
    # Row-batched persistent kernel: one program covers R consecutive rows,
    # amortizing the per-program dispatch and the weight load over R rows.
    # All memory ops stay strictly 1-D ([BLOCK_N] vectors): on this backend
    # every 2-D-tile formulation of this reduction either fails to compile
    # (memref_ext.slice_(pad_)start operandSegmentSizes, see compiler_issues/
    # vllm_silu_flat_slice_start.md) or silently miscompiles the broadcast
    # multiply chain (compiler_issues/
    # vllm_fused_q_kv_rmsnorm_rowbatch_bcast_miscompile.md).  The R row chains
    # are independent, so the 2-stage pipeline over blocks overlaps the next
    # block's loads with the current block's reduction trees and stores.
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N)
    if EXACT_COLS:
        w = tl.load(weight_ptr + offs)
    else:
        w = tl.load(weight_ptr + offs, mask=offs < N, other=0.0)
    for blk in tl.range(pid, NUM_BLOCKS, NPROG, num_stages=2):
        r0 = blk * R
        for r in tl.static_range(R):
            row = r0 + r
            if EVEN_ROWS and EXACT_COLS:
                x = tl.load(x_ptr + row * N + offs).to(tl.float32)
                var = tl.sum(x * x, axis=0) / N
                rrms = tl.rsqrt(var + eps)
                y = (x * rrms * w).to(out_ptr.dtype.element_ty)
                tl.store(out_ptr + row * N + offs, y)
            else:
                m = (row < M) & (offs < N)
                x = tl.load(x_ptr + row * N + offs, mask=m, other=0.0).to(
                    tl.float32
                )
                var = tl.sum(x * x, axis=0) / N
                rrms = tl.rsqrt(var + eps)
                y = (x * rrms * w).to(out_ptr.dtype.element_ty)
                tl.store(out_ptr + row * N + offs, y, mask=m)


@triton.jit
def _fused_rmsnorm_tile_kernel(
    x_ptr,
    weight_ptr,
    out_ptr,
    M,
    NUM_BLOCKS,
    eps,
    N: tl.constexpr,
    RT: tl.constexpr,
    BLOCK_N: tl.constexpr,
    NPROG: tl.constexpr,
    EVEN_ROWS: tl.constexpr,
    EXACT_COLS: tl.constexpr,
):
    # Persistent [RT, BLOCK_N] row-tile kernel: one axis=1 reduction tree per
    # tile amortizes the per-row tree cost over RT rows and merges the loads
    # and stores into single 2-D ops.  The weight is loaded with a 2-D shape
    # ([1, BLOCK_N]) on purpose: a 1-D fp32 load broadcast as a 2-D tile
    # multiply operand hits a known silent miscompile (P14 broadcast-chain
    # family); the 2-D-shaped load is the verified immune recipe (same as the
    # delivered vllm fused_add_rms_norm persist kernel).
    pid = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    if EXACT_COLS:
        w = tl.load(weight_ptr + cols[None, :]).to(tl.float32)
    else:
        w = tl.load(
            weight_ptr + cols[None, :], mask=(cols < N)[None, :], other=0.0
        ).to(tl.float32)
    for blk in tl.range(pid, NUM_BLOCKS, NPROG, num_stages=2):
        rows = blk * RT + tl.arange(0, RT)
        offs = rows[:, None] * N + cols[None, :]
        if EVEN_ROWS and EXACT_COLS:
            x = tl.load(x_ptr + offs).to(tl.float32)
            var = tl.sum(x * x, axis=1) / N
            rrms = tl.rsqrt(var + eps)
            y = (x * rrms[:, None] * w).to(out_ptr.dtype.element_ty)
            tl.store(out_ptr + offs, y)
        else:
            if EVEN_ROWS:
                m = (cols < N)[None, :]
            elif EXACT_COLS:
                m = (rows < M)[:, None]
            else:
                m = (rows < M)[:, None] & (cols < N)[None, :]
            x = tl.load(x_ptr + offs, mask=m, other=0.0).to(tl.float32)
            var = tl.sum(x * x, axis=1) / N
            rrms = tl.rsqrt(var + eps)
            y = (x * rrms[:, None] * w).to(out_ptr.dtype.element_ty)
            tl.store(out_ptr + offs, y, mask=m)


# Dispatch calibration (gcu400/Libra-class 48 SPs, bf16/fp16/fp32, do_bench
# kernel mode w1000/r100 median, 2026-11-12):
#   - M >= 256 (and BLOCK_N <= 8192) takes the persistent [RT, BLOCK_N] tile
#     kernel: (4096,1024) 89.3->18.1us (RT32 nw2), (4096,4096) 103.3->40.6us
#     (RT16), (4096,8192) 146.0->74.0us (RT8), (512,*) 1.38-1.97x,
#     (1000,4096) odd-M 1.79x.  Tile budget 65536 elems for 2-byte dtypes
#     (measured DSM 266-332KB, inside the 448KB dual-program line) and 32768
#     for fp32 (299KB); RT capped at 32 rows (validated broadcast-chain
#     envelope; RT64 measured no gain at (4096,1024) and stays outside).
#     nw2 beats nw1 by 8-20%; ns1 == ns2; nw4 loses (94.9 vs 74.0us at 8192).
#   - 64 <= M < 256 keeps the 1-D rows kernel: the 2-D tile loses or ties
#     there ((128,*) 0.87-1.00x, (83,1024) 0.81x, (64,*) 0.75-0.90x) -- the
#     ~1-2us 2-D small-tile structural tax dominates at low block counts.
#   - Below 64 rows (or BLOCK_N > 8192) the launch-latency-bound whole-row
#     kernel is kept.
_ROWS_MIN_ROWS = 64
_TILE_MIN_ROWS = 256
_NUM_SPC = 48
_TILE_ELEMS_NARROW = 65536
_TILE_ELEMS_WIDE = 32768
_TILE_MAX_ROWS = 32
_TILE_MIN_BLOCKS = 8
_TILE_NUM_WARPS = 2


def _rows_config(row_count, block_size, exact_cols):
    if block_size <= 1024:
        rows_per_cta = 4 if row_count >= 256 else 2
    else:
        rows_per_cta = 2
    if exact_cols:
        num_warps = 1 if block_size <= 4096 else 4
    else:
        num_warps = 2 if block_size <= 4096 else 4
    return rows_per_cta, num_warps


def fused_rmsnorm(
    x: torch.Tensor, weight: torch.Tensor, eps: float
) -> torch.Tensor:
    x = x.contiguous()
    weight = weight.contiguous()
    if x.ndim == 0:
        raise ValueError("x must have at least one dimension")
    if weight.ndim != 1 or weight.numel() != x.shape[-1]:
        raise ValueError("weight must match the last dimension of x")
    if x.numel() == 0:
        raise ValueError("x must be non-empty")
    row_width = x.shape[-1]
    output = torch.empty_like(x)
    row_count = x.numel() // row_width
    block_size = triton.next_power_of_2(row_width)
    if row_count >= _TILE_MIN_ROWS and block_size <= 8192:
        tile_elems = (
            _TILE_ELEMS_NARROW
            if x.element_size() < 4
            else _TILE_ELEMS_WIDE
        )
        rows_per_cta = min(_TILE_MAX_ROWS, tile_elems // block_size)
        while (
            rows_per_cta > 2
            and triton.cdiv(row_count, rows_per_cta) < _TILE_MIN_BLOCKS
        ):
            rows_per_cta //= 2
        num_blocks = triton.cdiv(row_count, rows_per_cta)
        _fused_rmsnorm_tile_kernel[(_NUM_SPC,)](
            x,
            weight,
            output,
            row_count,
            num_blocks,
            eps,
            N=row_width,
            RT=rows_per_cta,
            BLOCK_N=block_size,
            NPROG=_NUM_SPC,
            EVEN_ROWS=(row_count % rows_per_cta == 0),
            EXACT_COLS=(block_size == row_width),
            num_warps=_TILE_NUM_WARPS,
            num_stages=2,
        )
    elif row_count >= _ROWS_MIN_ROWS and block_size <= 8192:
        exact_cols = block_size == row_width
        rows_per_cta, num_warps = _rows_config(row_count, block_size, exact_cols)
        num_blocks = triton.cdiv(row_count, rows_per_cta)
        grid = (min(_NUM_SPC, num_blocks),)
        _fused_rmsnorm_rows_kernel[grid](
            x,
            weight,
            output,
            row_count,
            row_width,
            num_blocks,
            eps,
            R=rows_per_cta,
            BLOCK_N=block_size,
            NPROG=grid[0],
            EVEN_ROWS=(row_count % rows_per_cta == 0),
            EXACT_COLS=exact_cols,
            num_warps=num_warps,
        )
    else:
        _fused_rmsnorm_kernel[(row_count,)](
            x,
            weight,
            output,
            eps,
            row_width=row_width,
            block_size=block_size,
            num_warps=2,
        )
    return output


__all__ = ["fused_rmsnorm"]
