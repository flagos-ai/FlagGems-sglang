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

"""Single-pass dual residual RMSNorm implemented in portable Triton."""

import torch
import triton
import triton.language as tl

__all__ = ["fused_dual_residual_rmsnorm"]


@triton.jit
def _fused_dual_residual_rmsnorm_kernel(
    x_ptr,
    residual_ptr,
    weight1_ptr,
    weight2_ptr,
    output_ptr,
    mid_ptr,
    stride_x,
    stride_residual,
    stride_output,
    stride_mid,
    hidden: tl.constexpr,
    eps,
    RMS1_SCALE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < hidden
    x = tl.load(x_ptr + row * stride_x + cols, mask=mask, other=0.0).to(
        tl.float32
    )
    rms1 = tl.sqrt(tl.sum(x * x, axis=0) * (1.0 / hidden) + eps) * RMS1_SCALE
    weight1 = tl.load(weight1_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    normalized1 = ((x / rms1) * weight1).to(residual_ptr.dtype.element_ty)
    residual = tl.load(
        residual_ptr + row * stride_residual + cols, mask=mask, other=0.0
    )
    mid = (residual + normalized1).to(residual_ptr.dtype.element_ty)
    tl.store(mid_ptr + row * stride_mid + cols, mid, mask=mask)
    mid32 = mid.to(tl.float32)
    rms2 = tl.sqrt(tl.sum(mid32 * mid32, axis=0) * (1.0 / hidden) + eps)
    weight2 = tl.load(weight2_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    tl.store(
        output_ptr + row * stride_output + cols,
        (mid32 / rms2) * weight2,
        mask=mask,
    )


@triton.jit
def _fused_dual_residual_rmsnorm_rows_kernel(
    x_ptr,
    residual_ptr,
    weight1_ptr,
    weight2_ptr,
    output_ptr,
    mid_ptr,
    stride_x,
    stride_residual,
    stride_output,
    stride_mid,
    batch: tl.constexpr,
    hidden: tl.constexpr,
    eps,
    ROWS_PER_PROGRAM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    COLS_EVEN: tl.constexpr,
    ROWS_EVEN: tl.constexpr,
):
    row_ids = tl.program_id(0) * ROWS_PER_PROGRAM + tl.arange(
        0, ROWS_PER_PROGRAM
    )
    cols = tl.arange(0, BLOCK_SIZE)
    x_ptrs = x_ptr + row_ids[:, None] * stride_x + cols[None, :]
    r_ptrs = residual_ptr + row_ids[:, None] * stride_residual + cols[None, :]
    if COLS_EVEN and ROWS_EVEN:
        x = tl.load(x_ptrs).to(tl.float32)
        residual = tl.load(r_ptrs)
    else:
        mask = (row_ids[:, None] < batch) & (cols[None, :] < hidden)
        x = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)
        residual = tl.load(r_ptrs, mask=mask, other=0.0)
    inv_rms1 = tl.rsqrt(tl.sum(x * x, axis=1) * (1.0 / hidden) + eps)
    if COLS_EVEN:
        weight1 = tl.load(weight1_ptr + cols).to(tl.float32)
        weight2 = tl.load(weight2_ptr + cols).to(tl.float32)
    else:
        col_mask = cols < hidden
        weight1 = tl.load(weight1_ptr + cols, mask=col_mask, other=0.0).to(
            tl.float32
        )
        weight2 = tl.load(weight2_ptr + cols, mask=col_mask, other=0.0).to(
            tl.float32
        )
    normalized1 = (x * inv_rms1[:, None] * weight1[None, :]).to(
        residual_ptr.dtype.element_ty
    )
    mid = (residual + normalized1).to(residual_ptr.dtype.element_ty)
    mid_ptrs = mid_ptr + row_ids[:, None] * stride_mid + cols[None, :]
    out_ptrs = output_ptr + row_ids[:, None] * stride_output + cols[None, :]
    if COLS_EVEN and ROWS_EVEN:
        tl.store(mid_ptrs, mid)
    else:
        tl.store(mid_ptrs, mid, mask=mask)

    mid32 = mid.to(tl.float32)
    inv_rms2 = tl.rsqrt(tl.sum(mid32 * mid32, axis=1) * (1.0 / hidden) + eps)
    output = mid32 * inv_rms2[:, None] * weight2[None, :]
    if COLS_EVEN and ROWS_EVEN:
        tl.store(out_ptrs, output)
    else:
        tl.store(out_ptrs, output, mask=mask)


@triton.jit
def _fused_dual_residual_rmsnorm_tile_kernel(
    x_ptr,
    residual_ptr,
    weight1_ptr,
    weight2_ptr,
    output_ptr,
    mid_ptr,
    batch,
    num_blocks,
    hidden: tl.constexpr,
    eps,
    BLOCK_SIZE: tl.constexpr,
    ROWS_TILE: tl.constexpr,
    NPROG: tl.constexpr,
    EVEN_ROWS: tl.constexpr,
    EXACT_COLS: tl.constexpr,
):
    # Persistent [ROWS_TILE, BLOCK_SIZE] row-tile kernel (2026-11-13 probes):
    # NPROG == min(48, num_blocks) resident programs each stream row tiles via
    # tl.range, so the wave-synchronized one-shot grid's per-wave "load
    # dispatch + reduction tree" chain is paid once per program instead of
    # once per wave, and both weights are loaded once per program and reused
    # across every tile the program owns.  A 32768-element tile rides the
    # single-program/SP DSM window (526336B, one 48-program grid fills all
    # SPs, so the one-shot form's dual-program DSM line no longer binds).
    pid = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    inv_hidden = 1.0 / hidden
    # The weights are loaded with a 2-D shape on purpose: a 1-D fp32 load
    # broadcast as a 2-D tile multiply operand hits a known silent miscompile
    # (P14 broadcast-chain family); the 2-D-shaped load is the verified immune
    # recipe (same as the delivered gemma_fused_add_rmsnorm tile kernel).
    if EXACT_COLS:
        weight1 = tl.load(weight1_ptr + cols[None, :]).to(tl.float32)
        weight2 = tl.load(weight2_ptr + cols[None, :]).to(tl.float32)
    else:
        col_mask = (cols < hidden)[None, :]
        weight1 = tl.load(
            weight1_ptr + cols[None, :], mask=col_mask, other=0.0
        ).to(tl.float32)
        weight2 = tl.load(
            weight2_ptr + cols[None, :], mask=col_mask, other=0.0
        ).to(tl.float32)
    for block in tl.range(pid, num_blocks, NPROG):
        row_ids = block * ROWS_TILE + tl.arange(0, ROWS_TILE)
        offsets = row_ids[:, None] * hidden + cols[None, :]
        if EVEN_ROWS and EXACT_COLS:
            x = tl.load(x_ptr + offsets).to(tl.float32)
            residual = tl.load(residual_ptr + offsets)
        else:
            if EVEN_ROWS:
                tile_mask = (cols < hidden)[None, :]
            elif EXACT_COLS:
                tile_mask = (row_ids < batch)[:, None]
            else:
                tile_mask = (row_ids < batch)[:, None] & (cols < hidden)[None, :]
            x = tl.load(x_ptr + offsets, mask=tile_mask, other=0.0).to(
                tl.float32
            )
            residual = tl.load(
                residual_ptr + offsets, mask=tile_mask, other=0.0
            )
        inv_rms1 = tl.rsqrt(tl.sum(x * x, axis=1) * inv_hidden + eps)
        normalized1 = (x * inv_rms1[:, None] * weight1).to(
            residual_ptr.dtype.element_ty
        )
        mid = (residual + normalized1).to(residual_ptr.dtype.element_ty)
        mid32 = mid.to(tl.float32)
        inv_rms2 = tl.rsqrt(tl.sum(mid32 * mid32, axis=1) * inv_hidden + eps)
        output = mid32 * inv_rms2[:, None] * weight2
        if EVEN_ROWS and EXACT_COLS:
            tl.store(mid_ptr + offsets, mid)
            tl.store(output_ptr + offsets, output)
        else:
            tl.store(mid_ptr + offsets, mid, mask=tile_mask)
            tl.store(output_ptr + offsets, output, mask=tile_mask)


@triton.jit
def _fused_dual_residual_rmsnorm_persist_kernel(
    x_ptr,
    residual_ptr,
    weight1_ptr,
    weight2_ptr,
    output_ptr,
    mid_ptr,
    batch,
    rows_per_program,
    hidden: tl.constexpr,
    eps,
    RMS1_INV_SCALE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Persistent 1-D row-stream kernel (2026-11-13 probes): GRID programs
    # each own a contiguous block of rows_per_program rows and stream them
    # one row at a time.  Both weights stay resident across the program's
    # rows (the one-row-per-program grids re-read them for every row: at
    # hidden=8192 that is +50% read traffic), and the per-program "load
    # dispatch + two serial reduction trees" chain is paid once per program
    # instead of once per row.  Covers BLOCK_SIZE == 8192 (the 2-D tile at
    # that width loses: (4096,8192) 250.9 vs 225.8us) and the 48 < batch <
    # 128 underfilled-grid band at BLOCK_SIZE <= 4096.
    pid = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    inv_hidden = 1.0 / hidden
    lo = pid * rows_per_program
    hi = tl.minimum(lo + rows_per_program, batch)
    if BLOCK_SIZE == hidden:
        weight1 = tl.load(weight1_ptr + cols).to(tl.float32)
        weight2 = tl.load(weight2_ptr + cols).to(tl.float32)
        for row in range(lo, hi):
            base = row * hidden
            x = tl.load(x_ptr + base + cols).to(tl.float32)
            residual = tl.load(residual_ptr + base + cols)
            inv_rms1 = tl.rsqrt(
                tl.sum(x * x, axis=0) * inv_hidden + eps
            ) * RMS1_INV_SCALE
            normalized1 = (x * inv_rms1 * weight1).to(
                residual_ptr.dtype.element_ty
            )
            mid = (residual + normalized1).to(residual_ptr.dtype.element_ty)
            tl.store(mid_ptr + base + cols, mid)
            mid32 = mid.to(tl.float32)
            inv_rms2 = tl.rsqrt(tl.sum(mid32 * mid32, axis=0) * inv_hidden + eps)
            tl.store(output_ptr + base + cols, mid32 * inv_rms2 * weight2)
    else:
        col_mask = cols < hidden
        weight1 = tl.load(
            weight1_ptr + cols, mask=col_mask, other=0.0
        ).to(tl.float32)
        weight2 = tl.load(
            weight2_ptr + cols, mask=col_mask, other=0.0
        ).to(tl.float32)
        for row in range(lo, hi):
            base = row * hidden
            x = tl.load(x_ptr + base + cols, mask=col_mask, other=0.0).to(
                tl.float32
            )
            residual = tl.load(
                residual_ptr + base + cols, mask=col_mask, other=0.0
            )
            inv_rms1 = tl.rsqrt(
                tl.sum(x * x, axis=0) * inv_hidden + eps
            ) * RMS1_INV_SCALE
            normalized1 = (x * inv_rms1 * weight1).to(
                residual_ptr.dtype.element_ty
            )
            mid = (residual + normalized1).to(residual_ptr.dtype.element_ty)
            tl.store(mid_ptr + base + cols, mid, mask=col_mask)
            mid32 = mid.to(tl.float32)
            inv_rms2 = tl.rsqrt(tl.sum(mid32 * mid32, axis=0) * inv_hidden + eps)
            tl.store(
                output_ptr + base + cols, mid32 * inv_rms2 * weight2,
                mask=col_mask,
            )


@triton.jit
def _normalized1_kernel(
    x_ptr,
    weight_ptr,
    normalized_ptr,
    stride_x,
    stride_normalized,
    hidden: tl.constexpr,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < hidden
    values = tl.load(x_ptr + row * stride_x + cols, mask=mask, other=0.0).to(
        tl.float32
    )
    rms = tl.sqrt(tl.sum(values * values, axis=0) * (1.0 / hidden) + eps)
    weight = tl.load(weight_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    normalized = ((values / rms) * weight).to(normalized_ptr.dtype.element_ty)
    tl.store(
        normalized_ptr + row * stride_normalized + cols, normalized, mask=mask
    )


@triton.jit
def _add_residual_kernel(
    normalized_ptr,
    residual_ptr,
    mid_ptr,
    stride_normalized,
    stride_residual,
    stride_mid,
    hidden: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < hidden
    normalized = tl.load(
        normalized_ptr + row * stride_normalized + cols, mask=mask, other=0.0
    )
    residual = tl.load(
        residual_ptr + row * stride_residual + cols, mask=mask, other=0.0
    )
    tl.store(
        mid_ptr + row * stride_mid + cols, residual + normalized, mask=mask
    )


@triton.jit
def _normalized2_kernel(
    mid_ptr,
    weight_ptr,
    output_ptr,
    stride_mid,
    stride_output,
    hidden: tl.constexpr,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < hidden
    values = tl.load(
        mid_ptr + row * stride_mid + cols, mask=mask, other=0.0
    ).to(tl.float32)
    rms = tl.sqrt(tl.sum(values * values, axis=0) * (1.0 / hidden) + eps)
    weight = tl.load(weight_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    tl.store(
        output_ptr + row * stride_output + cols,
        (values / rms) * weight,
        mask=mask,
    )


@triton.jit
def _quotient_kernel(
    value_ptr,
    quotient_ptr,
    stride_value,
    stride_quotient,
    hidden: tl.constexpr,
    eps,
    RMS_SCALE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < hidden
    values = tl.load(
        value_ptr + row * stride_value + cols, mask=mask, other=0.0
    ).to(tl.float32)
    rms = tl.sqrt(tl.sum(values * values, axis=0) * (1.0 / hidden) + eps)
    rms = rms * RMS_SCALE
    tl.store(
        quotient_ptr + row * stride_quotient + cols, values / rms, mask=mask
    )


@triton.jit
def _apply_weight_kernel(
    quotient_ptr,
    weight_ptr,
    result_ptr,
    stride_quotient,
    stride_result,
    hidden: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < hidden
    quotient = tl.load(
        quotient_ptr + row * stride_quotient + cols, mask=mask, other=0.0
    )
    weight = tl.load(weight_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    tl.store(
        result_ptr + row * stride_result + cols, quotient * weight, mask=mask
    )


# Upper bound on rows-kernel tile elements (ROWS_PER_PROGRAM * BLOCK_SIZE).
# gcu400 DSM is driven by the 2D tile size only (dtype/batch invariant,
# probe-measured 2026-11-07): 32768 elems -> 526336B (fits), 65536 elems ->
# 917632B > 917504B hardware limit (OutOfResources).
# 2026-11-08 probe: 32768-elem tiles also exceed the 458752B dual-program
# threshold (1 program/SP -> the two reduction trees cannot overlap with other
# programs' DMA), so the dispatch below caps the tile at 16384 elems to keep
# 2 programs/SP ((4096,4096): 201.8us at 16384-elem tile vs 257-266us at 32768).
ROWS_TILE_ELEMS = 16384

# Persistent-arm dispatch (gcu400/Libra-class 48 SPs, official do_bench kernel
# mode w1000ms/r100ms, probes 2026-11-13; see kernel docstrings):
#   - _tile: rows*hidden >= 2M elements and 1024 <= BLOCK_SIZE <= 4096; a
#     32768-element tile is single-program/SP DSM-wise, and the persistent
#     48-program grid fills all SPs: (4096,4096) 203->89.4us, (4096,1024)
#     54->30.9us, (512,4096) 31->20.5us, (4096,3072) 86.0us.
#   - _persist: BLOCK_SIZE == 8192 ((512,8192) 48.4->33.7us, (4096,8192)
#     351.5->225.8us, (64,8192) 10.0->8.8us) and the 48 < batch < 128
#     underfilled-grid band at BLOCK_SIZE <= 4096 ((64,1024) 6.6->6.0us,
#     (64,4096) 8.8->7.7us).
#   - The 128 <= batch mid band keeps the one-shot rows kernel (rsqrt form),
#     which still wins there ((512,1024): rows 10.2 vs persist 18.8, tile
#     RT16 10.4us).
_NUM_SPC = 48
_TILE_ELEMS = 32768
_TILE_MAX_ROWS = 32
_TILE_MIN_ELEMS = 1 << 21
_TILE_MIN_BLOCK = 1024
_TILE_NUM_WARPS = 2
_TILE_NUM_STAGES = 1
_PERSIST_SMALL_ROWS = 128
_PERSIST_NUM_STAGES = 2
_PERSIST_NUM_WARPS = {4096: 2, 8192: 4}
# 1 / RMS1_SCALE (0.9999998807907104): x / (sqrt(m) * s) == x * rsqrt(m) * (1/s),
# so the rsqrt form applies the first-norm scale as a reciprocal multiply.
_RMS1_INV_SCALE = 1.0000001192092896


def _prev_pow2(value):
    return 1 << (value.bit_length() - 1)


def fused_dual_residual_rmsnorm(x, residual, weight1, weight2, eps):
    batch, hidden = x.shape
    output, mid = torch.empty_like(x), torch.empty_like(residual)
    block_size = triton.next_power_of_2(hidden)
    if hidden > 8192:
        quotient = torch.empty_like(x, dtype=torch.float32)
        normalized = torch.empty_like(residual)
        num_warps = 8
        _quotient_kernel[(batch,)](
            x,
            quotient,
            x.stride(0),
            quotient.stride(0),
            hidden=hidden,
            eps=eps,
            RMS_SCALE=0.9999998807907104,
            BLOCK_SIZE=block_size,
            num_warps=num_warps,
            num_stages=1,
        )
        _apply_weight_kernel[(batch,)](
            quotient,
            weight1,
            normalized,
            quotient.stride(0),
            normalized.stride(0),
            hidden=hidden,
            BLOCK_SIZE=block_size,
            num_warps=num_warps,
            num_stages=1,
        )
        _add_residual_kernel[(batch,)](
            normalized,
            residual,
            mid,
            normalized.stride(0),
            residual.stride(0),
            mid.stride(0),
            hidden=hidden,
            BLOCK_SIZE=block_size,
            num_warps=num_warps,
            num_stages=1,
        )
        _quotient_kernel[(batch,)](
            mid,
            quotient,
            mid.stride(0),
            quotient.stride(0),
            hidden=hidden,
            eps=eps,
            RMS_SCALE=1.0,
            BLOCK_SIZE=block_size,
            num_warps=num_warps,
            num_stages=1,
        )
        _apply_weight_kernel[(batch,)](
            quotient,
            weight2,
            output,
            quotient.stride(0),
            output.stride(0),
            hidden=hidden,
            BLOCK_SIZE=block_size,
            num_warps=num_warps,
            num_stages=1,
        )
        return output, mid
    if (
        batch > 48
        and x.element_size() < 4
        and x.stride(1) == 1
        and x.stride(0) == hidden
        and residual.stride(1) == 1
        and residual.stride(0) == hidden
        and output.stride(1) == 1
        and output.stride(0) == hidden
        and mid.stride(1) == 1
        and mid.stride(0) == hidden
        and weight1.stride(0) == 1
        and weight2.stride(0) == 1
    ):
        # Persistent arms (probe 2026-11-13): BLOCK_SIZE == 8192 rows are
        # streamed by the 1-D persist kernel (weights resident: the per-row
        # grid re-reads both weights for every row = +50% read traffic at
        # hidden=8192); the 48 < batch < 128 band at BLOCK_SIZE <= 4096 is
        # grid-underfilled on the one-shot arms and also prefers persist.
        if block_size > 4096 or batch < _PERSIST_SMALL_ROWS:
            grid = min(batch, _NUM_SPC)
            rows_per_program = triton.cdiv(batch, grid)
            _fused_dual_residual_rmsnorm_persist_kernel.run(
                x,
                residual,
                weight1,
                weight2,
                output,
                mid,
                batch,
                rows_per_program,
                hidden,
                eps,
                _RMS1_INV_SCALE if hidden >= 8192 else 1.0,
                block_size,
                grid=(grid,),
                warmup=False,
                num_warps=_PERSIST_NUM_WARPS.get(block_size, 1),
                num_stages=_PERSIST_NUM_STAGES,
                enable_fp_fusion=False,
            )
            return output, mid
        # Large-row tiles (>= 2M elements): persistent 2-D tile with a
        # 32768-element budget, one axis=1 reduction tree per ROWS_TILE rows
        # and both weights resident per program.
        if batch * hidden >= _TILE_MIN_ELEMS and block_size >= _TILE_MIN_BLOCK:
            rows_tile = min(_TILE_MAX_ROWS, _TILE_ELEMS // block_size)
            num_blocks = triton.cdiv(batch, rows_tile)
            nprog = min(_NUM_SPC, num_blocks)
            _fused_dual_residual_rmsnorm_tile_kernel.run(
                x,
                residual,
                weight1,
                weight2,
                output,
                mid,
                batch,
                num_blocks,
                hidden,
                eps,
                block_size,
                grid=(nprog,),
                warmup=False,
                ROWS_TILE=rows_tile,
                NPROG=nprog,
                EVEN_ROWS=(batch % rows_tile == 0),
                EXACT_COLS=(block_size == hidden),
                num_warps=_TILE_NUM_WARPS,
                num_stages=_TILE_NUM_STAGES,
                enable_fp_fusion=False,
            )
            return output, mid
    if batch >= 64 and block_size <= 4096:
        # Rows-kernel tile shaping (probe 2026-11-08, nw=1 sweeps):
        # - tile <= 16384 elems keeps 2 programs/SP (DSM dual-program window);
        # - rpp ~ batch//32 (pow2-floored) keeps the grid filled at small batch
        #   ((64,1024): rpp 8 -> 2 = 7.7 -> 6.5us) while large batch rides the
        #   tile cap ((4096,1024): rpp 16 = 77.2 -> 53.9us; (4096,4096):
        #   rpp 4 = 310.5 -> 201.8us).
        rows_per_program = min(
            ROWS_TILE_ELEMS // block_size, _prev_pow2(batch // 32)
        )
        _fused_dual_residual_rmsnorm_rows_kernel[
            (triton.cdiv(batch, rows_per_program),)
        ](
            x,
            residual,
            weight1,
            weight2,
            output,
            mid,
            x.stride(0),
            residual.stride(0),
            output.stride(0),
            mid.stride(0),
            batch=batch,
            hidden=hidden,
            eps=eps,
            ROWS_PER_PROGRAM=rows_per_program,
            BLOCK_SIZE=block_size,
            COLS_EVEN=(block_size == hidden),
            ROWS_EVEN=(batch % rows_per_program == 0),
            num_warps=1,
            num_stages=1,
        )
        return output, mid
    if block_size <= 2048:
        num_warps = 1 if batch >= 64 else 4
    else:
        # nw=1 wins up to BLOCK_SIZE=4096 (probe 2026-11-07: (64..4096,4096)
        # 8.9/42.9/311.7us vs 9.0/44.4/327.6us at nw=8); at BLOCK_SIZE=8192
        # nw=1 loses ~1.5x vs nw=8 (559.7 vs 355.7us at (4096,8192)).
        num_warps = 1 if (batch >= 64 and block_size <= 4096) else 8
    _fused_dual_residual_rmsnorm_kernel[(batch,)](
        x,
        residual,
        weight1,
        weight2,
        output,
        mid,
        x.stride(0),
        residual.stride(0),
        output.stride(0),
        mid.stride(0),
        hidden=hidden,
        eps=eps,
        RMS1_SCALE=0.9999998807907104 if hidden >= 8192 else 1.0,
        BLOCK_SIZE=block_size,
        num_warps=num_warps,
        num_stages=1,
    )
    return output, mid
