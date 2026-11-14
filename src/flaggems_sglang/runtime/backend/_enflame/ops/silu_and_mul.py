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
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied. See the License for the specific language governing
# permissions and limitations under the License.

"""Fused SiLU gate + multiply Triton kernel (GCU tuning).

Computes ``out[..., c] = silu(x[..., c]) * x[..., d + c]`` where ``d`` is
half of the last dimension, entirely in fp32 and cast back to the input
dtype: a single pass (2 reads + 1 write) instead of the ~7 memory
round-trips of the eager reference.

GCU measurements (GCU410, 24 SIP, bf16, do_bench kernel mode) that shape
this file:

1. **The grid is capped at 48 CTAs and must stay a multiple of the SIP
   count.**  The row grid-stride kernel saturates streaming bandwidth at
   2 CTAs per SIP (48 CTAs on a 24-SIP part); 1 CTA/SIP leaves ~40%
   bandwidth unused, and off-multiple counts pay wave-quantization tails
   (52 CTAs measured 1.6x slower than 48).  The inherited Ascend tuning
   (512 CTAs) re-paid the Fixed-Grid dispatch tax on every medium/large
   shape: capping at 48 took the (4096, 8192) core shape from 82.4us to
   63.6us (-23%) and (512, 8192) from 32.3us to 12.8us (-60%).  Small
   inputs launch one program per row (grid = min(n_rows, 48)).

2. **Tile widths in [1024, 8192] are bandwidth-neutral** (BLOCK=1024 /
   2048 / 4096 measure identically), so the wrapper keeps the widest
   tile that compiles: MAIN=8192 with num_warps=4.  Wider tiles, or the
   same width with fewer warps, exceed the vectorizer's per-warp element
   budget on this stack and fail to compile; num_warps=4 is pinned on
   the launch for that reason.

3. **num_stages pipelining, row unrolling, and blocked (contiguous
   chunk) row assignment are neutral-to-negative** here -- loads on this
   stack have no software pipelining -- so the kernel keeps strided row
   assignment plus exact power-of-two tail tiles (e.g. d=11008 walks
   8192 + 2048 + 512 + 256 lanes with zero masking).

4. **Load depth, not width, gates the row bandwidth (round 2, E-1).**
   The round-1 grid-stride loop issues the loads of row ``r + stride``
   only after the stores of row ``r`` have been emitted, so the DMA
   engine never sees more than one row (24 KB) in flight per CTA.
   Hoisting the four loads of *two* rows ahead of both compute/store
   stages doubles the in-flight depth and lifts the d=4096 bandwidth
   from 1.59 to 2.05 TB/s: (4096, 8192) 63.5 -> 49.0 us (-23%),
   (512, 8192) 12.7 -> 11.1 us (-12%), bs<=64 launch-bound shapes
   within noise.  The "sublinear row-width curve" of round 1 is
   therefore partly an issue-depth artifact, not pure hardware limit.
   Hoisted U=4 groups and BLOCK=d/2 x 2 tiles both regress (register
   pressure / small-tile DMA efficiency); the masked odd-tail-row guard
   is free (strided vs blocked pair assignment measure identically);
   and the round-1 launch parameters remain optimal under the new form
   (num_warps=4, 48 CTAs, single full-width tile).  The pair kernel
   only takes power-of-two d <= 4096, where one unmasked tile covers a
   row; every other d keeps the general kernel below.

Address arithmetic keeps the overflow-safe int64 *scalar* row base with
int32 lane offsets, and ``silu(x) = x / (1 + exp(-x))`` as one exp + one
fdiv (the transcendental pair costs ~6% over a pure load-mul-store under
the paired form, so no cheaper rearrangement is worth chasing).
"""

import torch
import triton
import triton.language as tl

# Upper bound on the number of CTAs for the row grid-stride path.  GCU
# Fixed-Grid dispatch serializes CTAs per SIP; on the 24-SIP GCU410 the
# streaming bandwidth of this kernel saturates at 2 CTAs/SIP (48), and
# counts that are not multiples of the SIP count pay wave-quantization
# tails (measured: 52 CTAs is 1.6x slower than 48, 96 is slightly
# slower, 24 leaves ~40% bandwidth unused).  48 is also a safe choice on
# 48-SP parts (1 CTA/SIP).
_MAX_NUM_CTAS = 48

# Largest full-width tile per CTA (elements along the halved last dim).
# 8192 with num_warps=4 is the widest tile this stack compiles for the
# flat 1D loads/stores used here (per-warp element budget ~2048).
_MAX_BLOCK_SIZE = 8192

# Max number of exact power-of-two tail tiles emitted per row.  Four
# covers every remainder under 8192 whose popcount is <= 4 (including
# d=11008's remainder 2816 = 2048+512+256); anything more falls back to
# one masked tile for the leftover low bits.
_MAX_TAIL_BLOCKS = 4

# Largest power-of-two d served by the paired-row fast-path kernel
# (one unmasked BLOCK=d tile per row, two rows hoisted).  d=8192 would
# double the live tile registers per CTA (4 x 8192 fp32 lanes) and hit
# the register-pressure wall measured for U=4; non-power-of-two d needs
# the tail decomposition of the general kernel.
_FAST_PATH_MAX_D = 4096


@triton.jit
def _silu_and_mul_pair_kernel(
    x_ptr,
    out_ptr,
    d: tl.constexpr,
    n_rows,
    BLOCK: tl.constexpr,
):
    """Strided row pairs with all four loads hoisted (pow2-d fast path).

    One unmasked BLOCK=d tile per row; each iteration grabs two rows
    (r, r + num_programs), issues the four loads together, then computes
    and stores both rows.  The masked guard covers the odd tail row only
    (measured free); see module docstring point 4 for the numbers.
    """
    row_idx = tl.program_id(0)
    n_row_ctas = tl.num_programs(0)
    offs = tl.arange(0, BLOCK)

    for r in range(row_idx, n_rows, 2 * n_row_ctas):
        r2 = r + n_row_ctas
        base1 = x_ptr + r.to(tl.int64) * (2 * d)
        x1a = tl.load(base1 + offs).to(tl.float32)
        x2a = tl.load(base1 + d + offs).to(tl.float32)
        mask = r2 < n_rows
        base2 = x_ptr + r2.to(tl.int64) * (2 * d)
        x1b = tl.load(base2 + offs, mask=mask, other=0.0).to(tl.float32)
        x2b = tl.load(base2 + d + offs, mask=mask, other=0.0).to(tl.float32)
        tl.store(out_ptr + r.to(tl.int64) * d + offs,
                 tl.fdiv(x1a, 1.0 + tl.exp(-x1a)) * x2a)
        tl.store(out_ptr + r2.to(tl.int64) * d + offs,
                 tl.fdiv(x1b, 1.0 + tl.exp(-x1b)) * x2b, mask=mask)


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
    # explicit dtype/device lookups in ``torch.empty``; small inputs are
    # launch bound and sensitive to every microsecond of host work.
    out = hidden_states.new_empty(shape[:-1] + (d,))
    if d == 0 or numel == 0:
        return out

    # Flatten the leading dims without materializing a view: the kernel
    # only needs the contiguous flat pointer and the total row count.
    x = hidden_states.contiguous()
    n_rows = numel // (2 * d)

    # Power-of-two d <= 4096: one unmasked tile covers a row, so take
    # the paired-row kernel that hoists two rows' loads (docstring 4).
    # Checked before the tail decomposition so the fast path skips that
    # host work (small inputs are launch bound).  Every other d
    # (non-pow2 tails, d = 8192) keeps the general kernel.
    n_row_ctas = n_rows if n_rows < _MAX_NUM_CTAS else _MAX_NUM_CTAS
    if d <= _FAST_PATH_MAX_D and (d & (d - 1)) == 0:
        _silu_and_mul_pair_kernel[(n_row_ctas,)](
            x,
            out,
            d=d,
            n_rows=n_rows,
            BLOCK=d,
            num_warps=4,
        )
        return out

    # One row grid-stride program per CTA, capped at the SIP-multiple
    # bandwidth sweet spot.  The remainder d % MAIN is decomposed into
    # exact pow2 tiles to avoid both wasted transcendental lanes and
    # masked loads/stores.
    main = _MAX_BLOCK_SIZE
    n_full = d // main
    tail = d - n_full * main
    tail_blocks, rem = _decompose_tail(tail, _MAX_TAIL_BLOCKS)
    t = tail_blocks + [0] * (_MAX_TAIL_BLOCKS - len(tail_blocks))
    rem_block = 1 << (rem - 1).bit_length() if rem > 0 else 0

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
        num_warps=4,
    )

    return out


__all__ = ["silu_and_mul"]
