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

import triton
import triton.language as tl

# One GCU400 wave holds 48 programs (SIPs); the E/H split doubles the grid,
# so 24 programs per plane fill exactly one wave. Overflowing the wave
# (e.g. 28 per plane) measured +34% on (512,7168).
_WAVE_PROGRAMS_PER_PLANE = 24
# Max R*B elements per program tile (DSM-safe, verified: 64K bf16-input
# lanes compile and run fastest; 128K is either rejected by the compiler
# (R=32/B=8192: dsm 1118208 > 917504) or slow (R=16/B=8192)).
_TILE_ELEM_BUDGET = 1 << 16


@triton.jit
def _norm(
    X,
    W,
    OUT,
    N: tl.constexpr,
    D: tl.constexpr,
    EPS: tl.constexpr,
    R: tl.constexpr,
    B: tl.constexpr,
):
    col = tl.arange(0, B)
    ptr = tl.make_block_ptr(
        X, (N, D), (D, 1), (tl.program_id(0) * R, 0), (R, B), (1, 0)
    )
    x = tl.load(ptr, boundary_check=(0, 1), padding_option="zero").to(
        tl.float32
    )
    w = tl.load(W + col, col < D, other=0).to(tl.float32)
    inv = tl.rsqrt(tl.sum(x * x, 1) / D + EPS)
    value = x * inv[:, None] * w[None, :]
    ptr = tl.make_block_ptr(
        OUT, (N, D), (2 * D, 1), (tl.program_id(0) * R, 0), (R, B), (1, 0)
    )
    tl.store(ptr, value.to(OUT.dtype.element_ty), boundary_check=(0, 1))


@triton.jit
def _eh(
    E,
    H,
    EW,
    HW,
    OUT,
    N: tl.constexpr,
    D: tl.constexpr,
    EPS: tl.constexpr,
    R: tl.constexpr,
    B: tl.constexpr,
):
    if tl.program_id(1) == 0:
        _norm(E, EW, OUT, N, D, EPS, R, B)
    else:
        _norm(H, HW, OUT + D, N, D, EPS, R, B)


@triton.jit
def _norm_tiled(
    X,
    W,
    OUT,
    D: tl.constexpr,
    EPS: tl.constexpr,
    R: tl.constexpr,
    B: tl.constexpr,
    TILES: tl.constexpr,
    EXACT: tl.constexpr,
):
    # Persistent grid-stride over row tiles. Weights are loaded once per
    # program as a [1, B] 2-D shape (the immune form for axis=1 row reduce
    # under multi-warp launches). The host guarantees N % R == 0, so row
    # masking is unnecessary; column masking disappears when B == D.
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    cols = tl.arange(0, B)
    col_mask = cols < D
    if EXACT:
        w = tl.load(W + cols[None, :]).to(tl.float32)
    else:
        w = tl.load(
            W + cols[None, :], mask=col_mask[None, :], other=0.0
        ).to(tl.float32)
    for tile in tl.range(pid, TILES, nprog):
        rows = tile * R + tl.arange(0, R)
        x_ptrs = X + rows[:, None] * D + cols[None, :]
        if EXACT:
            x = tl.load(x_ptrs).to(tl.float32)
        else:
            x = tl.load(
                x_ptrs, mask=col_mask[None, :], other=0.0
            ).to(tl.float32)
        inv = tl.rsqrt(tl.sum(x * x, 1) / D + EPS)
        value = x * inv[:, None] * w
        o_ptrs = OUT + rows[:, None] * (2 * D) + cols[None, :]
        if EXACT:
            tl.store(o_ptrs, value.to(OUT.dtype.element_ty))
        else:
            tl.store(
                o_ptrs,
                value.to(OUT.dtype.element_ty),
                mask=col_mask[None, :],
            )


@triton.jit
def _eh_tiled(
    E,
    H,
    EW,
    HW,
    OUT,
    D: tl.constexpr,
    EPS: tl.constexpr,
    R: tl.constexpr,
    B: tl.constexpr,
    TILES: tl.constexpr,
    EXACT: tl.constexpr,
):
    if tl.program_id(1) == 0:
        _norm_tiled(E, EW, OUT, D, EPS, R, B, TILES, EXACT)
    else:
        _norm_tiled(H, HW, OUT + D, D, EPS, R, B, TILES, EXACT)


def fused_eh_norm(
    inputs_embeds, previous_hidden, enorm_weight, hnorm_weight, eps
):
    e = inputs_embeds
    h = previous_hidden
    ew = enorm_weight
    hw = hnorm_weight
    shape = e.shape
    n = shape[0]
    d = shape[1]
    out = e.new_empty(n, d + d)
    if n:
        if not (
            e.is_contiguous()
            and h.is_contiguous()
            and ew.is_contiguous()
            and hw.is_contiguous()
        ):
            e = e.contiguous()
            h = h.contiguous()
            ew = ew.contiguous()
            hw = hw.contiguous()
        b = triton.next_power_of_2(d)
        if n * b >= (1 << 20):
            # Wide-tile region. num_warps >= 2 is unsafe for B in
            # {512, 1024} (narrow-tile tl.sum(axis=1) row-reduce
            # miscompile; see compiler_issues/
            # sglang_fused_eh_norm_narrow_tile_nw2_rowreduce.md).
            r = min((1 << 16) // b, triton.next_power_of_2(n))
            num_warps = 2 if b >= 2048 else 1
            if n % r == 0:
                # Wave-packed persistent tiles: at most one wave of
                # programs stays resident, eliminating inter-wave retire
                # gaps (measured -16% .. -30% on bs=512..4096 shapes).
                tiles = n // r
                p = min(tiles, _WAVE_PROGRAMS_PER_PLANE)
                _eh_tiled[(p, 2)](
                    e,
                    h,
                    ew,
                    hw,
                    out,
                    d,
                    eps,
                    r,
                    b,
                    tiles,
                    b == d,
                    num_warps=num_warps,
                    num_stages=1,
                    enable_fp_fusion=False,
                )
                return out
            # Non-divisible row count: fall through to the boundary-check
            # kernel below with the wide-tile r.
        else:
            # Parallelism-first arm: more programs beat wider tiles while
            # the grid still fits a single wave.
            r = min(
                max(1, triton.next_power_of_2(n) // 16), max(1, 32768 // b)
            )
            num_warps = 2 if (r == 1 and b >= 8192) else 1
        _eh[(triton.cdiv(n, r), 2)](
            e,
            h,
            ew,
            hw,
            out,
            n,
            d,
            eps,
            r,
            b,
            num_warps=num_warps,
            num_stages=1,
            enable_fp_fusion=False,
        )
    return out


__all__ = ["fused_eh_norm"]
