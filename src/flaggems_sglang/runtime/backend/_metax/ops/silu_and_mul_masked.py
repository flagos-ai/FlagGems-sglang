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

"""Optimised fused SiLU-and-mul with per-expert mask.

The op is memory-bandwidth-bound: elementwise ``silu(gate) * up`` over an
``[E, T, H]`` tensor where, per expert ``e``, only the first
``masked_m[e]`` rows are valid (DeepGEMM-style grouped-MoE layout).

Bench shapes are ``[E, 256, 4096]`` with ``masked_m[e] == 256 == T`` (every
row valid).  The two performance regimes:

  * **e32** (bandwidth-saturated, ~1.17 TB/s on Metax C550): wants one
    program per output row owning the *whole* hidden half (``BLOCK_H ==
    half``), a thin row tile (``BLOCK_T == 1``), loads streamed with
    ``evict_first``.  This is the v9-v11 sweet spot.
  * **e4 / e8** (launch / occupancy bound — ~0.5 / 0.7 TB/s, far below the
    memory ceiling): the per-expert count ``E`` caps how many programs run
    concurrently along the T axis.  v11's one-program-per-row grid still
    issues ``E*T`` programs but each is a heavyweight whole-row tile, so
    launch / scheduling overhead dominates.  v7 showed the fix: split the
    hidden half so several programs from the *same* expert run
    concurrently on different SMs (``num_h > 1``), raising occupancy.

This iteration (v13) keeps the v12 3D grid and widens the autotune space
to recover the per-shape peaks the narrower v12 sweep had dropped:

  * ``BLOCK_T`` back to ``{1,2,4,8,16}`` (v7's set — the v12 sweep only
    kept ``{1,4,16}`` and lost the 2/8 the e8 winner used).
  * ``BLOCK_H`` adds ``128`` so e4 (4 experts) can fan out to ``num_h=16``
    → 64 concurrent programs, near the 104-SM fleet (the v12 sweep only
    went down to 256 → num_h=8 → 32 programs, half the fleet).
  * ``num_stages`` back to ``{1,2}`` (a one-pass streaming kernel has no
    long pipeline to fill; stages=3 never won the v7-v11 micro A/B and
    bloats compile time across the per-(half,E) autotune keys).
  * The SiLU form reverts to ``gate * tl.sigmoid(gate)`` — the form the
    e4 (v9) and e8 (v7) best kernels used; the v11 reciprocal-folded
    divide form regressed both.

Kept from v12/v9-v11: the **3D grid ``(E, num_t, num_h)``** with
axis-aligned program ids (no per-program div-mod), the scalar
``masked_m[e]`` load per program, the **loads-only ``evict_first``**
cache policy (a store-side hint cost as much as it saved in the v9 micro
A/B), and the ``up``-stays-input-dtype trick (gate promoted to fp32 for
the sigmoid; ``up`` promoted by the fp32*bf16 multiply — one cast saved
per element, still within the 1.5e-2 tolerance).

The H axis carries an element mask ``h_off < half`` for the (power-of-two)
padded-tile tail; every swept ``BLOCK_H`` divides ``half=2048`` so the
mask is all-true in production and only bites untested non-divisible tails.
The row mask ``t_off < n`` is the exact per-expert validity test — an
expert with ``n == 0`` (or a tile entirely past ``n``) masks its whole
load/store to a no-op, so padded rows stay untouched (never checked).

The kernel never hardcodes a device or vendor — it runs on
``flaggems_sglang.device`` and uses only portable Triton primitives.
No module-level mutable state; ``@triton.autotune`` caches live in the
Triton runtime, not hand-rolled globals.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


def _configs():
    # 3D grid (E, num_t, num_h).  Each program owns a [BLOCK_T, BLOCK_H]
    # tile.  The autotune searches the row-tile width (BLOCK_T), the
    # hidden-tile width (BLOCK_H), and the launch knobs (num_warps,
    # num_stages) so each expert count lands on its own regime.
    #
    # Measured ceiling on Metax (104 SMs, ~1.1 TB/s peak):
    #   * e32 (32 experts) is bandwidth-saturated — ~96 MB in ~86 us hits
    #     the memory ceiling, so it wants BLOCK_H == half (num_h == 1,
    #     one program per output row owning the whole hidden half) with a
    #     small BLOCK_T so the E*T grid still fans out across all SMs.
    #     This is the v10/v11 whole-row pattern.
    #   * e4 / e8 are launch/occupancy-bound — e4 moves only 12 MB yet
    #     takes ~25 us (~0.48 TB/s, 44% of ceiling), so ~14 us is fixed
    #     launch/scheduling cost that only more *concurrent* programs can
    #     hide.  They want a *small* BLOCK_H so num_h is large (several
    #     programs from the same expert run concurrently on different
    #     SMs) paired with a moderate BLOCK_T.  v7's H-split took e8 to
    #     26.86x with BLOCK_H=512; the prior configs lacked the finer
    #     BLOCK_H=128 (num_h=16, so e4 fans out to 64 programs — near the
    #     104-SM fleet) and the mid BLOCK_T={2,8} v7 used.
    #
    # This sweep widens that space: BLOCK_T in {1,2,4,8,16} (v7's set,
    # including the 2/8 the v11 sweep dropped), BLOCK_H in
    # {128,256,512,1024,2048} (adds 128 for e4 occupancy).  Every BLOCK_H
    # divides half=2048 (production shapes), so the H tail mask is
    # all-true in production and only bites untested non-divisible tails.
    #
    # Private-memory budget on Metax is 4 KB/thread.  The register / spill
    # footprint of a [BLOCK_T, BLOCK_H] tile grows with the product
    # BLOCK_T*BLOCK_H (each element holds a fp32 gate + the input-dtype
    # up + intermediates), so an oversized tile (e.g. 64*2048 = 131072
    # elements) blows the budget and the launch fails with
    # "private memory ... too large".  v9 measured the safe ceiling at a
    # ~8192-element tile; we cap BLOCK_T*BLOCK_H <= 8192 to stay inside
    # that ceiling for every swept combination.  num_warps stay in
    # {2,4,8} (16 blew the budget at large tiles); num_stages in {1,2}
    # (v7's set — a one-pass streaming kernel has no long pipeline to
    # fill, and stages=3 never won the v7-v11 micro A/B while it bloats
    # compile time across the per-(half,E) autotune keys).
    cfgs = []
    for bt in (1, 2, 4, 8, 16):
        for bh in (128, 256, 512, 1024, 2048):
            if bt * bh > 8192:
                continue
            for nw in (2, 4, 8):
                for ns in (1, 2):
                    cfgs.append(
                        triton.Config(
                            {"BLOCK_T": bt, "BLOCK_H": bh},
                            num_warps=nw,
                            num_stages=ns,
                        )
                    )
    return cfgs


@triton.autotune(configs=_configs(), key=("half", "E"))
@triton.jit
def _silu_and_mul_masked_kernel(
    input_ptr,
    out_ptr,
    masked_m_ptr,
    # strides
    in_e_stride,  # T*H
    in_t_stride,  # H
    out_e_stride,  # T*half
    out_t_stride,  # half
    # sizes
    T,
    E,
    half: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    # 3D grid: (expert, t-tile, h-tile) — axis-aligned program ids, no
    # div/mod (the v11 per-row grid recomputed e=row//T, t=row%T per
    # program; here they come straight from program_id).
    pid_e = tl.program_id(0)
    pid_t = tl.program_id(1)
    pid_h = tl.program_id(2)

    # Scalar per-expert mask: one ``masked_m`` load per program.
    n = tl.load(masked_m_ptr + pid_e)

    t_start = pid_t * BLOCK_T
    t_off = t_start + tl.arange(0, BLOCK_T)  # [BLOCK_T]
    t_mask = t_off < n  # within masked_m[e]

    # Hidden tile for this program.  BLOCK_H may be < half (small-E
    # occupancy path), so the H axis needs its own element mask for the
    # padded-tile tail.  For production shapes every swept BLOCK_H divides
    # half, so the last h-tile is exact and the mask is all-true there.
    h_start = pid_h * BLOCK_H
    h_off = h_start + tl.arange(0, BLOCK_H)  # [BLOCK_H]
    h_mask = h_off < half  # [BLOCK_H]

    # Row mask broadcast over H, H mask broadcast over T -> combined mask.
    row_mask = t_mask[:, None]  # [BLOCK_T, 1]
    col_mask = h_mask[None, :]  # [1, BLOCK_H]
    mask = row_mask & col_mask

    in_e_base = pid_e * in_e_stride
    out_e_base = pid_e * out_e_stride

    row_base = t_off[:, None] * in_t_stride  # [BLOCK_T, 1]
    g_off = in_e_base + row_base + h_off[None, :]  # [BLOCK_T, BLOCK_H]
    u_off = g_off + half  # [BLOCK_T, BLOCK_H]

    # Streaming loads (read once, don't churn the L2 working set): evict
    # the one-pass input data immediately.  The store takes the *default*
    # cache policy — on Metax the store-side evict_first hint costs as
    # much as it saves (v9 micro A/B: loads-only 85.76us vs loads+stores
    # 86.02us on e32).
    gate = tl.load(
        input_ptr + g_off, mask=mask, other=0.0, eviction_policy="evict_first"
    )
    up = tl.load(
        input_ptr + u_off, mask=mask, other=0.0, eviction_policy="evict_first"
    )

    # gate must be fp32 for the exp; up is only multiplied in, so leave
    # it in the input dtype and let Triton promote the fp32*bf16
    # multiply — saves one cast per element and still matches the
    # all-fp32 reference within tolerance.
    gate = gate.to(tl.float32)
    # silu(gate) = gate * sigmoid(gate).  This is the form the e4 (v9) and
    # e8 (v7) best kernels used; the v11 reciprocal-folded divide form
    # (gate / (1 + exp(-gate))) regressed both.  tl.sigmoid is the exact
    # 1/(1+exp(-x)), no fast-math; gate is already fp32 for it.
    val = gate * tl.sigmoid(gate) * up

    out_off = out_e_base + t_off[:, None] * out_t_stride + h_off[None, :]
    tl.store(out_ptr + out_off, val.to(out_ptr.dtype.element_ty), mask=mask)


def silu_and_mul_masked(input, masked_m):
    E, T, H = input.shape
    half = H // 2
    # Only valid rows [e, :masked_m[e]] are written; padded rows are never
    # checked, so there is no need to zero-initialise the output (saves a
    # full-tensor memset that dominated small-expert latency).
    out = torch.empty(E, T, half, dtype=input.dtype, device=input.device)

    in_e_stride, in_t_stride, _ = input.stride()
    out_e_stride, out_t_stride, _ = out.stride()

    def grid(meta):
        num_t = triton.cdiv(T, meta["BLOCK_T"])
        num_h = triton.cdiv(half, meta["BLOCK_H"])
        return (E, num_t, num_h)

    _silu_and_mul_masked_kernel[grid](
        input,
        out,
        masked_m,
        in_e_stride,
        in_t_stride,
        out_e_stride,
        out_t_stride,
        T,
        E,
        half,
    )
    return out


__all__ = ["silu_and_mul_masked"]
