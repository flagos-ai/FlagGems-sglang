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
from triton.language.extra import libdevice


@triton.jit
def _rmsnorm_weight(x, inv, weight):
    # Preserve both BF16 rounding stages from the original implementation.
    norm = (x * inv).to(tl.bfloat16).to(tl.float32)
    return (norm * weight).to(tl.bfloat16).to(tl.float32)


@triton.jit
def _apply_gate(gate, norm):
    gate = libdevice.tanh(gate).to(tl.bfloat16).to(tl.float32)
    return (gate * norm).to(tl.bfloat16).to(tl.float32)


@triton.jit
def _apply_gate_native(gate, norm):
    # Tile-form gate: libdevice.tanh lowers pathologically on [RT, BLOCK]
    # tiles on this backend (probe 2026-11-12: full tile chain 116.1us vs
    # 27.4us with an exp2 identity and 22.7us with this sigmoid form at
    # (1024,2048) RT8; the extern-call tanh interacts catastrophically with
    # the multi-stream register pressure, and tl.sigmoid lowers to the
    # native tops_sigmoid_v2_fp32 instruction -- same lever as the
    # softcap_inplace_logits tanh->sigmoid rewrite).  Identity:
    # tanh(g) = 2*sigmoid(2g) - 1; saturates exactly to +/-1 for large |g|
    # (no overflow window), keeps the same tanh->bf16 rounding stage as
    # _apply_gate, and is bitwise identical to the libdevice form on bf16
    # tiles for benchmark inputs.
    t = 2.0 * tl.sigmoid(gate * 2.0) - 1.0
    gate = t.to(tl.bfloat16).to(tl.float32)
    return (gate * norm).to(tl.bfloat16).to(tl.float32)


@triton.jit
def _row_offset(row, ROW_SHAPE: tl.constexpr, ROW_STRIDES: tl.constexpr):
    offset = tl.full((), 0, tl.int64)
    for index in tl.static_range(len(ROW_SHAPE) - 1, -1, -1):
        offset += (row % ROW_SHAPE[index]).to(tl.int64) * ROW_STRIDES[index]
        row = row // ROW_SHAPE[index]
    return offset


@triton.jit
def _rmsnorm_tanh_residual_strided_kernel(
    x_ptr,
    gate_ptr,
    residual_ptr,
    weight_ptr,
    out_ptr,
    NUM_ROWS: tl.constexpr,
    D: tl.constexpr,
    ROWS_PER_GATE: tl.constexpr,
    ROW_SHAPE: tl.constexpr,
    X_STRIDES: tl.constexpr,
    GATE_SHAPE: tl.constexpr,
    GATE_STRIDES: tl.constexpr,
    RESIDUAL_STRIDES: tl.constexpr,
    W_S: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    cols = tl.arange(0, BLOCK_SIZE)
    w = tl.load(weight_ptr + cols * W_S, cols < D, 0).to(tl.float32)
    for row in range(tl.program_id(0), NUM_ROWS, tl.num_programs(0)):
        x = tl.load(
            x_ptr
            + _row_offset(row, ROW_SHAPE, X_STRIDES)
            + cols * X_STRIDES[len(ROW_SHAPE)],
            cols < D,
            0,
        ).to(tl.float32)
        g = tl.load(
            gate_ptr
            + _row_offset(row // ROWS_PER_GATE, GATE_SHAPE, GATE_STRIDES)
            + cols * GATE_STRIDES[len(GATE_SHAPE)],
            cols < D,
            0,
        ).to(tl.float32)
        r = tl.load(
            residual_ptr
            + _row_offset(row, ROW_SHAPE, RESIDUAL_STRIDES)
            + cols * RESIDUAL_STRIDES[len(ROW_SHAPE)],
            cols < D,
            0,
        ).to(tl.float32)
        inv = tl.rsqrt(tl.sum(x * x, 0) / D + EPS)
        norm = _rmsnorm_weight(x, inv, w)
        product = _apply_gate(g, norm)
        tl.store(out_ptr + row * D + cols, r + product, cols < D)


@triton.jit
def _rmsnorm_tanh_residual_row_kernel(
    x_ptr,
    gate_ptr,
    residual_ptr,
    weight_ptr,
    out_ptr,
    D: tl.constexpr,
    ROWS_PER_GATE: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    CACHE_MODIFIER: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    x = tl.load(
        x_ptr + row * D + cols, cols < D, 0, cache_modifier=CACHE_MODIFIER
    ).to(tl.float32)
    inv = tl.rsqrt(tl.sum(x * x, 0) / D + EPS)
    w = tl.load(
        weight_ptr + cols, cols < D, 0, cache_modifier=CACHE_MODIFIER
    ).to(tl.float32)
    norm = _rmsnorm_weight(x, inv, w)
    g = tl.load(
        gate_ptr + row // ROWS_PER_GATE * D + cols,
        cols < D,
        0,
        cache_modifier=CACHE_MODIFIER,
    ).to(tl.float32)
    product = _apply_gate(g, norm)
    r = tl.load(
        residual_ptr + row * D + cols,
        cols < D,
        0,
        cache_modifier=CACHE_MODIFIER,
    ).to(tl.float32)
    tl.store(out_ptr + row * D + cols, r + product, cols < D)


@triton.jit
def _rmsnorm_tanh_residual_split_kernel(
    x_ptr,
    gate_ptr,
    residual_ptr,
    weight_ptr,
    out_ptr,
    D: tl.constexpr,
    ROWS_PER_GATE: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    PARTS: tl.constexpr,
    NUM_COLS: tl.constexpr,
):
    row = tl.program_id(0) // PARTS
    part = tl.program_id(0) % PARTS
    full_c = tl.arange(0, BLOCK_SIZE)
    full_x = tl.load(x_ptr + row * D + full_c, full_c < D, 0).to(tl.float32)
    inv = tl.rsqrt(tl.sum(full_x * full_x, 0) / D + EPS)
    cols = part * NUM_COLS + tl.arange(0, NUM_COLS)
    x = tl.load(x_ptr + row * D + cols, cols < D, 0).to(tl.float32)
    w = tl.load(weight_ptr + cols, cols < D, 0).to(tl.float32)
    norm = _rmsnorm_weight(x, inv, w)
    g = tl.load(gate_ptr + row // ROWS_PER_GATE * D + cols, cols < D, 0).to(
        tl.float32
    )
    product = _apply_gate(g, norm)
    r = tl.load(residual_ptr + row * D + cols, cols < D, 0).to(tl.float32)
    tl.store(out_ptr + row * D + cols, r + product, cols < D)


@triton.jit
def _rmsnorm_tanh_residual_rows_kernel(
    x_ptr,
    gate_ptr,
    residual_ptr,
    weight_ptr,
    out_ptr,
    NUM_ROWS,
    NUM_BLOCKS,
    D: tl.constexpr,
    ROWS_PER_GATE: tl.constexpr,
    EPS: tl.constexpr,
    R: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    NPROG: tl.constexpr,
    LOOP_STAGES: tl.constexpr,
    EARLY_LOADS: tl.constexpr,
    EVEN_ROWS: tl.constexpr,
    EXACT_COLS: tl.constexpr,
):
    # Row-batched persistent kernel (port of the fused_rmsnorm rows recipe,
    # same board/toolchain): one program walks R-row blocks with a grid-stride
    # loop, so the per-program dispatch cost and the weight load amortize over
    # many rows instead of paying one wave per row.  All loads/stores stay
    # strictly 1-D ([BLOCK_SIZE] vectors): on this backend 2-D tile forms of
    # this reduction family fail to compile or silently miscompile (see
    # knowledge/perf_playbook.md rmsnorm rows entries).  The R row chains are
    # independent, so the pipelined block loop overlaps the next block's loads
    # with the current block's reduction trees and stores.
    pid = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    if EXACT_COLS:
        w = tl.load(weight_ptr + cols).to(tl.float32)
    else:
        w = tl.load(weight_ptr + cols, cols < D, 0).to(tl.float32)
    for blk in tl.range(pid, NUM_BLOCKS, NPROG, num_stages=LOOP_STAGES):
        row0 = blk * R
        for r in tl.static_range(R):
            row = row0 + r
            if ROWS_PER_GATE == 1:
                gate_row = row
            else:
                gate_row = row // ROWS_PER_GATE
            if EVEN_ROWS and EXACT_COLS:
                x = tl.load(x_ptr + row * D + cols).to(tl.float32)
                if EARLY_LOADS:
                    g = tl.load(gate_ptr + gate_row * D + cols).to(tl.float32)
                    res = tl.load(residual_ptr + row * D + cols).to(tl.float32)
                inv = tl.rsqrt(tl.sum(x * x, 0) / D + EPS)
                norm = _rmsnorm_weight(x, inv, w)
                if not EARLY_LOADS:
                    g = tl.load(gate_ptr + gate_row * D + cols).to(tl.float32)
                product = _apply_gate(g, norm)
                if not EARLY_LOADS:
                    res = tl.load(residual_ptr + row * D + cols).to(tl.float32)
                tl.store(out_ptr + row * D + cols, res + product)
            else:
                m = (row < NUM_ROWS) & (cols < D)
                x = tl.load(x_ptr + row * D + cols, m, 0).to(tl.float32)
                if EARLY_LOADS:
                    g = tl.load(gate_ptr + gate_row * D + cols, m, 0).to(
                        tl.float32
                    )
                    res = tl.load(residual_ptr + row * D + cols, m, 0).to(
                        tl.float32
                    )
                inv = tl.rsqrt(tl.sum(x * x, 0) / D + EPS)
                norm = _rmsnorm_weight(x, inv, w)
                if not EARLY_LOADS:
                    g = tl.load(gate_ptr + gate_row * D + cols, m, 0).to(
                        tl.float32
                    )
                product = _apply_gate(g, norm)
                if not EARLY_LOADS:
                    res = tl.load(residual_ptr + row * D + cols, m, 0).to(
                        tl.float32
                    )
                tl.store(out_ptr + row * D + cols, res + product, m)


@triton.jit
def _rmsnorm_tanh_residual_tile_kernel(
    x_ptr,
    gate_ptr,
    residual_ptr,
    weight_ptr,
    out_ptr,
    NUM_ROWS,
    NUM_BLOCKS,
    D: tl.constexpr,
    ROWS_PER_GATE: tl.constexpr,
    EPS: tl.constexpr,
    RT: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    NPROG: tl.constexpr,
    EVEN_ROWS: tl.constexpr,
    EXACT_COLS: tl.constexpr,
):
    # Persistent [RT, BLOCK_SIZE] row-tile kernel (same recipe as the
    # delivered fused_rmsnorm/rmsnorm_scale tile kernels on this board): one
    # axis=1 reduction tree per tile amortizes the per-row tree cost over RT
    # rows and merges the four streams' loads/stores into single 2-D ops.
    # The weight is loaded with a 2-D shape ([1, BLOCK_SIZE]) on purpose: a
    # 1-D fp32 load broadcast into a 2-D tile multiply hits the P14
    # broadcast-chain silent-miscompile family; the 2-D-shaped load is the
    # verified immune recipe.
    pid = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    if EXACT_COLS:
        w = tl.load(weight_ptr + cols[None, :]).to(tl.float32)
    else:
        w = tl.load(
            weight_ptr + cols[None, :], mask=(cols < D)[None, :], other=0.0
        ).to(tl.float32)
    for blk in tl.range(pid, NUM_BLOCKS, NPROG, num_stages=2):
        rows = blk * RT + tl.arange(0, RT)
        offs = rows[:, None] * D + cols[None, :]
        if ROWS_PER_GATE == 1:
            gate_offs = offs
        else:
            gate_rows = rows // ROWS_PER_GATE
            gate_offs = gate_rows[:, None] * D + cols[None, :]
        if EVEN_ROWS and EXACT_COLS:
            x = tl.load(x_ptr + offs).to(tl.float32)
            g = tl.load(gate_ptr + gate_offs).to(tl.float32)
            res = tl.load(residual_ptr + offs).to(tl.float32)
            inv = tl.rsqrt(tl.sum(x * x, axis=1) / D + EPS)
            norm = _rmsnorm_weight(x, inv[:, None], w)
            product = _apply_gate_native(g, norm)
            tl.store(out_ptr + offs, res + product)
        else:
            if EVEN_ROWS:
                m = (cols < D)[None, :]
            elif EXACT_COLS:
                m = (rows < NUM_ROWS)[:, None]
            else:
                m = (rows < NUM_ROWS)[:, None] & (cols < D)[None, :]
            x = tl.load(x_ptr + offs, mask=m, other=0.0).to(tl.float32)
            g = tl.load(gate_ptr + gate_offs, mask=m, other=0.0).to(tl.float32)
            res = tl.load(residual_ptr + offs, mask=m, other=0.0).to(
                tl.float32
            )
            inv = tl.rsqrt(tl.sum(x * x, axis=1) / D + EPS)
            norm = _rmsnorm_weight(x, inv[:, None], w)
            product = _apply_gate_native(g, norm)
            tl.store(out_ptr + offs, res + product, mask=m)


# Rows-kernel dispatch calibration (gcu400/Libra-class 48 SPs, bf16, official
# bench do_bench kernel mode, probe 2026-11-08):
#   - R=1 for wide rows (block_size >= 2048): the 3 row loads (x/gate/residual)
#     already keep enough bytes in flight, and deeper batching (R>=2) regresses
#     badly at 4096 (block-level pipeline pressure).  R=2 wins ~7-9% on narrow
#     rows (block_size <= 1024) where per-row fixed cost dominates.
#   - num_warps=1 for exact-width loads (shuffle-only reduction tree, no
#     inter-warp DSM hop); masked (non-pow2 D) loads prefer nw=2
#     ((1024,1536): 46.0 vs 49.2us).
#   - Early loads (gate/residual issued before the reduction tree) are the
#     single biggest lever: -21~-23% at 2048/4096 wide shapes.
#   - stages=3/4 tie with 2; grid 96/192 regress; exp-form tanh regresses
#     (+18%) vs libdevice.tanh -- do not swap the transcendental.
#   - Below 64 rows the launch-bound split/row kernels are kept.
_ROWS_MIN_ROWS = 64
_NUM_SPC = 48

# Tile-kernel dispatch calibration (gcu400/Libra-class 48 SPs, bf16, official
# bench do_bench kernel mode w1000/r100 median, 2026-11-12 geometry sweep;
# recipe ported from the delivered fused_rmsnorm / rmsnorm_scale tile kernels
# on the same board/toolchain):
#   - M >= 256 (and block_size <= 4096) takes the persistent [RT, BLOCK_SIZE]
#     tile kernel: one axis=1 tree per RT-row tile amortizes the per-row
#     "load dispatch + tree" serial chain that floors the 1-D rows kernel.
#   - 64 <= M < 256 keeps the 1-D rows kernel: the 2-D small-tile machinery
#     tax loses or ties below M=256 (fused_rmsnorm/rmsnorm_scale same-day
#     crossovers: (128,*) 0.87-1.00x, (64,*) 0.75-0.90x).
#   - RT by width/M (bf16, nw1): block 2048 -> RT8 (26.6us @1024) / RT16 for
#     M >= 4096 (129.3us @8192, 1.04TB/s); block 4096 -> RT4 (44.2us @1024,
#     252.5us @8192, 1.06TB/s); narrower rows keep the 16384-elem budget rule.
#     num_warps=1 wins or ties every measured point (nw2 regresses badly on
#     wide rows for this 4-stream op: 82.6 vs 53.3us @ (1024,4096) RT8).
#   - tile budget 65536 elems is NOT usable for this op: its 4 streams
#     (x/gate/residual/out) measure 528384B DSM at 65536 (llir
#     gcu.dsm_memory_size, over the 448KB dual-program line even at nw1:
#     526336B) vs 266240B at 32768 (all mask arms identical); fp32 steps
#     down to 16384 (32768-fp32 measures 544768B).
#   - the gate uses the native-instruction sigmoid identity instead of
#     libdevice.tanh: on [RT, BLOCK] tiles the extern-call tanh interacts
#     pathologically with the multi-stream register pressure (full chain
#     116.1us vs 27.4us with an exp2 identity and 22.7us with the sigmoid
#     identity at (1024,2048) RT8; see _apply_gate_native).
_TILE_MIN_ROWS = 256
_TILE_ELEMS_NARROW = 16384
_TILE_MAX_ROWS = 32
_TILE_MIN_BLOCKS = 8


def _tile_config(num_rows, block_size):
    # Measured optima (see calibration block above); nw1 everywhere.
    if block_size > 2048:
        return 4, 1
    if block_size == 2048:
        return (16, 1) if num_rows >= 4096 else (8, 1)
    rows_per_cta = min(_TILE_MAX_ROWS, _TILE_ELEMS_NARROW // block_size)
    return rows_per_cta, 1


def _rows_config(num_rows, block_size, exact_cols):
    if block_size <= 1024:
        rows_per_cta = 2
    else:
        rows_per_cta = 1
    return rows_per_cta, (1 if exact_cols else 2), 2, True


def rmsnorm_tanh_residual(x, gate, residual, weight, eps):
    hidden_size = x.shape[-1]
    num_rows = x.numel() // hidden_size
    out = torch.empty_like(x, memory_format=torch.contiguous_format)
    if num_rows == 0:
        return out

    is_hip = torch.version.hip is not None
    rows_per_gate = num_rows // (gate.numel() // hidden_size)
    block_size = triton.next_power_of_2(hidden_size)
    num_warps = 4 if block_size <= 2048 else 8
    dense = all(t.is_contiguous() for t in (x, gate, residual, weight))

    if dense and num_rows >= _TILE_MIN_ROWS and block_size <= 4096:
        # Persistent [RT, BLOCK_SIZE] tile kernel (see calibration block above).
        rows_per_cta, tile_warps = _tile_config(num_rows, block_size)
        if x.element_size() >= 4:
            # fp32: halve RT to stay under the 448KB DSM line (a 32768-elem
            # fp32 tile measures 544768B; see calibration block above).
            rows_per_cta = max(2, rows_per_cta // 2)
        while (
            rows_per_cta > 2
            and triton.cdiv(num_rows, rows_per_cta) < _TILE_MIN_BLOCKS
        ):
            rows_per_cta //= 2
        num_blocks = triton.cdiv(num_rows, rows_per_cta)
        _rmsnorm_tanh_residual_tile_kernel[(_NUM_SPC,)](
            x,
            gate,
            residual,
            weight,
            out,
            num_rows,
            num_blocks,
            hidden_size,
            rows_per_gate,
            eps,
            RT=rows_per_cta,
            BLOCK_SIZE=block_size,
            NPROG=_NUM_SPC,
            EVEN_ROWS=(num_rows % rows_per_cta == 0),
            EXACT_COLS=(block_size == hidden_size),
            num_warps=tile_warps,
            num_stages=2,
            enable_fp_fusion=False,
        )
    elif dense and num_rows >= _ROWS_MIN_ROWS and block_size <= 4096:
        # Persistent rows kernel: one wave of <= 48 programs covers any row
        # count (this also lifts the grid.x > 65535 case off the strided
        # kernel's per-row div/mod path).  Wider rows (block_size > 4096)
        # keep the original per-row/split dispatch: their register footprint
        # is outside this kernel's calibrated DSM budget.
        exact_cols = block_size == hidden_size
        rows_per_cta, rows_warps, loop_stages, early_loads = _rows_config(
            num_rows, block_size, exact_cols
        )
        num_blocks = triton.cdiv(num_rows, rows_per_cta)
        grid = (min(_NUM_SPC, num_blocks),)
        _rmsnorm_tanh_residual_rows_kernel[grid](
            x,
            gate,
            residual,
            weight,
            out,
            num_rows,
            num_blocks,
            hidden_size,
            rows_per_gate,
            eps,
            R=rows_per_cta,
            BLOCK_SIZE=block_size,
            NPROG=grid[0],
            LOOP_STAGES=loop_stages,
            EARLY_LOADS=early_loads,
            EVEN_ROWS=(num_rows % rows_per_cta == 0),
            EXACT_COLS=exact_cols,
            num_warps=rows_warps,
            enable_fp_fusion=False,
        )
    elif not dense or num_rows > 65535:
        _rmsnorm_tanh_residual_strided_kernel[min(num_rows, 65535),](
            x,
            gate,
            residual,
            weight,
            out,
            num_rows,
            hidden_size,
            rows_per_gate,
            tuple(x.shape[:-1]),
            tuple(x.stride()),
            tuple(gate.shape[:-1]),
            tuple(gate.stride()),
            tuple(residual.stride()),
            weight.stride(0),
            eps,
            block_size,
            num_warps=num_warps,
            enable_fp_fusion=False,
        )
    elif num_rows <= 64 and block_size >= 1024:
        num_parts = 4
        _rmsnorm_tanh_residual_split_kernel[num_rows * num_parts,](
            x,
            gate,
            residual,
            weight,
            out,
            hidden_size,
            rows_per_gate,
            eps,
            block_size,
            num_parts,
            block_size // num_parts,
            num_warps=4,
            enable_fp_fusion=False,
        )
    else:
        cache_modifier = ".cg" if is_hip and num_rows >= 4096 else ""
        _rmsnorm_tanh_residual_row_kernel[num_rows,](
            x,
            gate,
            residual,
            weight,
            out,
            hidden_size,
            rows_per_gate,
            eps,
            block_size,
            cache_modifier,
            num_warps=num_warps,
            enable_fp_fusion=False,
        )
    return out


__all__ = ["rmsnorm_tanh_residual"]
