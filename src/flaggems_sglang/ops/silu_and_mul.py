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

"""Fused SiLU gate + multiply Triton kernel.

Computes ``out[..., c] = silu(x[..., c]) * x[..., d + c]`` where ``d`` is
half of the last dimension, entirely in fp32 and cast back to the input
dtype.

The reference
``F.silu(x[..., :d].float()) * x[..., d:].float()).to(dtype)`` is ~7
memory round-trips (two fp32 casts, silu, mul, cast-back).  This kernel
does a single pass (2 reads + 1 write), so for prefill-sized batches it
wins by roughly the memory-traffic ratio.

Two things dominate the measured runtime and shape the design:

1. **Tile width drives memory throughput on Ascend.**  The vector cores
   get far more bytes/cycle out of a wide contiguous tile than a narrow
   one, so we use the widest tile that still fits the unified buffer:
   8192 (16384 overflows the UB with the multi-buffer feature enabled).

2. **Launch/dispatch overhead (~70 us) is fixed per kernel** and
   dominates decode-sized batches.  Everything the wrapper does on the
   critical path is therefore minimized: no ``.view()`` calls; ``n_rows``
   is derived from ``numel()`` and the contiguous input/output tensors
   are passed straight to the kernel.

Two launch paths are used because of how Ascend schedules CTAs:

* **Direct 2D grid** (one program per ``(row, col-block)``).  Lowest
  overhead; used when the total block count is small
  (``<= _MAX_DIRECT_CTAS``), which is the small/medium batch sizes that
  dominate decode.  Those sizes are launch bound, so a masked
  (wasted-lane) tail costs nothing there.

* **Row grid-stride** (1D grid over rows).  Each CTA owns a *whole* row
  and walks it as ``N_FULL`` full-width 8192 blocks followed by the
  remainder ``d % 8192``, which is decomposed into its *exact*
  power-of-two set bits (e.g. ``2816 = 2048 + 512 + 256``) rather than
  one masked ``next_power_of_2`` block.  Ascend's ``exp``/``div`` run on
  every lane of a tile regardless of a store mask, *and* masked
  loads/stores are themselves slow (predicated memory).  The naive
  single masked tail wastes ``8192 - (d % 8192)`` transcendental lanes
  per row **and** pays a masked access; the exact decomposition emits
  only in-bounds, unmasked tiles, which for ``d = 11008`` cuts the
  per-row tile work from 12288 lanes (8192 + 4096) down to 11008 lanes
  (8192 + 2048 + 512 + 256) with zero masking -- a ~12% prefill speedup
  versus a single 4096-wide masked tail.

Both paths share the same address-arithmetic tricks:

1. **int32 lane offsets.**  Only the per-row *scalar* base uses int64
   (so ``n_rows * 2*d`` cannot overflow); the per-lane ``col_offs`` stay
   int32, which is much cheaper in the Ascend vector pipeline.
2. **Single fused compute.**  ``silu(x) = x / (1 + exp(-x))`` is one
   ``exp`` + one ``div`` instead of ``x * sigmoid(x)``.
"""

import torch
import triton
import triton.language as tl

# Ascend caps the *product* of the grid dimensions at 65535; the direct
# 2D path is only used below a much smaller threshold where launching
# one CTA per tile is still cheaper than the grid-stride loop's extra
# bookkeeping.  decode-sized batches are launch bound (fixed ~76 us
# Triton dispatch dominates), so both paths cost the same there and the
# simpler direct kernel (fewer constexpr, no ``tl.num_programs``) wins
# by a hair; prefill-sized batches are memory bound and route to the
# row kernel.
_MAX_DIRECT_CTAS = 2048

# Upper bound on the number of CTAs for the row grid-stride path.
# Ascend 910B4 exposes ~40 vector cores; a few hundred CTAs saturate
# memory bandwidth without the scheduling overhead of the naive
# many-block launch.  Sweeping over the 4096-row workload, 512 CTAs is
# the sweet spot (more CTAs re-pay scheduling overhead, fewer leave the
# vector cores idle).
_MAX_NUM_CTAS = 512

# Largest full-width tile per CTA (elements along the halved last dim).
# 8192 is the widest tile that fits the Ascend unified buffer (16384
# overflows it).
_MAX_BLOCK_SIZE = 8192

# Max number of exact power-of-two tail tiles emitted per row.  Four
# covers every remainder under 8192 whose popcount is <= 4 (including
# d=11008's remainder 2816 = 2048+512+256); anything more falls back to
# one masked tile for the leftover low bits.
_MAX_TAIL_BLOCKS = 4


@triton.jit
def _silu_and_mul_kernel(
    x_ptr,
    out_ptr,
    d: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    col_block_idx = tl.program_id(1)

    col_offs = col_block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = col_offs < d

    # int64 scalar row base (overflow-safe), int32 lane offsets (fast).
    base = x_ptr + row_idx.to(tl.int64) * (2 * d)
    x1 = tl.load(base + col_offs, mask=mask, other=0.0).to(tl.float32)
    x2 = tl.load(base + d + col_offs, mask=mask, other=0.0).to(tl.float32)

    silu_x1 = tl.fdiv(x1, 1.0 + tl.exp(-x1))

    out_base = out_ptr + row_idx.to(tl.int64) * d
    tl.store(out_base + col_offs, silu_x1 * x2, mask=mask)


@triton.jit
def _silu_and_mul_row_kernel(
    x_ptr,
    out_ptr,
    d: tl.constexpr,
    n_rows,
    MAIN: tl.constexpr,
    N_FULL: tl.constexpr,
    T0: tl.constexpr,
    T1: tl.constexpr,
    T2: tl.constexpr,
    T3: tl.constexpr,
    REM: tl.constexpr,
    HAS_REM: tl.constexpr,
):
    """One CTA per (strided) row: full blocks + exact pow2 tail tiles."""
    row_idx = tl.program_id(0)
    n_row_ctas = tl.num_programs(0)

    main_offs = tl.arange(0, MAIN)

    for r in range(row_idx, n_rows, n_row_ctas):
        base = x_ptr + r.to(tl.int64) * (2 * d)
        out_base = out_ptr + r.to(tl.int64) * d

        # Full 8192-wide blocks: always in-bounds (MAIN <= d), no mask.
        for b in tl.static_range(N_FULL):
            col_offs = b * MAIN + main_offs
            x1 = tl.load(base + col_offs).to(tl.float32)
            x2 = tl.load(base + d + col_offs).to(tl.float32)
            silu_x1 = tl.fdiv(x1, 1.0 + tl.exp(-x1))
            tl.store(out_base + col_offs, silu_x1 * x2)

        # Exact power-of-two tail tiles (unmasked; sizes sum to d % MAIN).
        if T0 > 0:
            col_offs = N_FULL * MAIN + tl.arange(0, T0)
            x1 = tl.load(base + col_offs).to(tl.float32)
            x2 = tl.load(base + d + col_offs).to(tl.float32)
            silu_x1 = tl.fdiv(x1, 1.0 + tl.exp(-x1))
            tl.store(out_base + col_offs, silu_x1 * x2)
        if T1 > 0:
            col_offs = N_FULL * MAIN + T0 + tl.arange(0, T1)
            x1 = tl.load(base + col_offs).to(tl.float32)
            x2 = tl.load(base + d + col_offs).to(tl.float32)
            silu_x1 = tl.fdiv(x1, 1.0 + tl.exp(-x1))
            tl.store(out_base + col_offs, silu_x1 * x2)
        if T2 > 0:
            col_offs = N_FULL * MAIN + T0 + T1 + tl.arange(0, T2)
            x1 = tl.load(base + col_offs).to(tl.float32)
            x2 = tl.load(base + d + col_offs).to(tl.float32)
            silu_x1 = tl.fdiv(x1, 1.0 + tl.exp(-x1))
            tl.store(out_base + col_offs, silu_x1 * x2)
        if T3 > 0:
            col_offs = N_FULL * MAIN + T0 + T1 + T2 + tl.arange(0, T3)
            x1 = tl.load(base + col_offs).to(tl.float32)
            x2 = tl.load(base + d + col_offs).to(tl.float32)
            silu_x1 = tl.fdiv(x1, 1.0 + tl.exp(-x1))
            tl.store(out_base + col_offs, silu_x1 * x2)

        # Leftover low bits (only when the tail has > _MAX_TAIL_BLOCKS
        # set bits): a single masked tile covering the last `REM`
        # elements.
        if HAS_REM:
            tail_base = N_FULL * MAIN + T0 + T1 + T2 + T3
            col_offs = tail_base + tl.arange(0, REM)
            mask = col_offs < d
            x1 = tl.load(base + col_offs, mask=mask, other=0.0).to(tl.float32)
            x2 = tl.load(base + d + col_offs, mask=mask, other=0.0).to(
                tl.float32
            )
            silu_x1 = tl.fdiv(x1, 1.0 + tl.exp(-x1))
            tl.store(out_base + col_offs, silu_x1 * x2, mask=mask)


def _decompose_tail(tail: int, max_blocks: int):
    """Split ``tail`` into its power-of-two set bits (largest first).

    Returns ``(blocks, rem)`` where ``sum(blocks) + rem == tail``,
    ``len(blocks) <= max_blocks``, and ``rem`` holds the low bits that
    did not fit (0 unless ``tail`` has more than ``max_blocks`` set
    bits).  ``blocks`` are exact power-of-two tile widths; ``rem`` is
    covered by one masked tile.
    """
    blocks = []
    while tail > 0 and len(blocks) < max_blocks:
        p = 1 << (tail.bit_length() - 1)
        blocks.append(p)
        tail -= p
    return blocks, tail


def silu_and_mul(hidden_states: torch.Tensor) -> torch.Tensor:
    """Gated SiLU activation: ``out = silu(x[..., :d]) * x[..., d:]``.

    Args:
        hidden_states: Input tensor of any fp dtype whose last dim is
            even.  Shape ``[..., 2d]``.
    Returns:
        Output tensor of shape ``[..., d]`` and the same dtype as the
        input.
    """
    shape = hidden_states.shape
    d = shape[-1] // 2
    numel = hidden_states.numel()

    # ``new_empty`` inherits dtype/device from the input and skips the
    # explicit dtype/device lookups in ``torch.empty``; the launch-bound
    # decode path is sensitive to every microsecond of host-side work.
    out = hidden_states.new_empty(shape[:-1] + (d,))
    if d == 0 or numel == 0:
        return out

    # Flatten the leading dims without materializing a view: the kernel
    # only needs the contiguous flat pointer and the total row count.
    x = hidden_states.contiguous()
    n_rows = numel // (2 * d)

    # next_power_of_2 / cdiv inlined (one-liners) to keep the host-side
    # critical path minimal.
    block_size = min(_MAX_BLOCK_SIZE, 1 << (d - 1).bit_length())
    n_col_blocks = (d + block_size - 1) // block_size
    total_tiles = n_rows * n_col_blocks

    if total_tiles <= _MAX_DIRECT_CTAS:
        # Small/medium batch: one program per (row, col-block).
        _silu_and_mul_kernel[(n_rows, n_col_blocks)](
            x, out, d=d, BLOCK_SIZE=block_size
        )
    else:
        # Large batch: cap the grid and loop over rows, one CTA per row.
        # The remainder d % MAIN is decomposed into exact pow2 tiles to
        # avoid both wasted transcendental lanes and masked loads/stores.
        main = _MAX_BLOCK_SIZE
        n_full = d // main
        tail = d - n_full * main
        tail_blocks, rem = _decompose_tail(tail, _MAX_TAIL_BLOCKS)
        t = tail_blocks + [0] * (_MAX_TAIL_BLOCKS - len(tail_blocks))
        rem_block = 1 << (rem - 1).bit_length() if rem > 0 else 0
        n_row_ctas = min(n_rows, _MAX_NUM_CTAS)
        _silu_and_mul_row_kernel[(n_row_ctas,)](
            x,
            out,
            d=d,
            n_rows=n_rows,
            MAIN=main,
            N_FULL=n_full,
            T0=t[0],
            T1=t[1],
            T2=t[2],
            T3=t[3],
            REM=rem_block,
            HAS_REM=rem > 0,
        )

    return out


__all__ = ["silu_and_mul"]
