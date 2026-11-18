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

from __future__ import annotations

import torch
import triton
import triton.language as tl

# One CTA per SP: the GCU exposes 48 SPs, and a persistent grid-stride
# launch at 48 CTAs beats both smaller (24: -40%) and larger (96: +6%)
# grids on every benchmark shape (probe-verified; multi_processor_count
# reports 4 and is NOT the SP count).
_MAX_CTAS = 48

# Widest per-tile column block (bf16 2048 elems = 4 KB slice, far below the
# 32 KB/CTA single-slice compile limit in compiler_issues/
# vllm_silu_flat_slice_start.md). The GLU family ceiling for this
# (rows, 2H)-split layout is ~1.6 TB/s (geglu precedent).
_MAX_BH = 2048

# Smallest column block worth a 1-D contiguous slice; smaller `half`
# lowbit divisors fall back to the masked 2-D path.
_MIN_BH = 16

# Literal unroll depth of the flat kernel: 8 row-tiles in flight per loop
# iteration. The whole batch is held in registers as the *storage* dtype
# (bf16 -> 128 regs/lane at num_warps=2) and each tile is promoted to fp32
# only at its compute+store turn ("late cast"), so the fp32 working set never
# exceeds one tile. Promoting at load time instead ("early cast") makes the
# U=8 batch 256+ fp32 regs/lane and spills: measured 62.5 -> 60.1us at the
# largest bench shape only via the U=4 early-cast compromise (probe-verified
# 2026-11-13; U=8 late-cast reaches 56.8us). U=16 late-cast re-spills (256
# bf16 regs + fp32 temps) and regresses 18%. The load hoisting only happens
# when written out literally -- the scheduler does not discover it from a
# static_range loop (~40% slower), and ns=2/3 software pipelining of the
# range loop is a no-op on this backend.
_UNROLL = 8


@triton.jit
def _silu_and_mul_masked_flat_kernel(
    in_ptr,  # bf16 [E, T, H], row-major contiguous
    out_ptr,  # bf16 [E, T, half], row-major contiguous
    n_tiles,  # n_rows * HTILES
    H: tl.constexpr,  # full hidden width (= 2 * half)
    half: tl.constexpr,
    BH: tl.constexpr,  # per-tile column block (divides `half`)
    HTILES: tl.constexpr,  # half // BH
):
    """Flat 1-D row-tile path with a literal 8-tile load batch.

    Each program strides over groups of ``_UNROLL`` consecutive tiles; a
    tile is one ``[BH]`` slice of one row's gate half plus the matching
    ``[BH]`` slice of the up half, both addressed from a scalar row base
    (``tile // HTILES`` row, ``tile % HTILES`` column block) so every
    load/store is a plain contiguous 1-D access -- no rank-2 tensors,
    which the current toolchain miscompiles (see
    compiler_issues/vllm_silu_flat_slice_start.md).

    All ``2 * _UNROLL`` loads of a group are issued before the first
    compute/store, and each tile's compute+store pair follows immediately.
    The batch stays in registers at the storage dtype and each tile is
    promoted to fp32 only when consumed ("late cast"), which keeps the
    live set at 128 bf16 regs/lane (num_warps=2) instead of 256+ fp32
    regs/lane -- early-cast at this depth spills and loses the DMA-depth
    win (probe-verified 2026-11-13).

    Out-of-range trailing tiles are clamped to the last tile for the
    loads (in-bounds by construction) and simply not stored.
    """
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    U: tl.constexpr = 8
    last = n_tiles - 1
    for t0 in tl.range(pid * U, n_tiles, nprog * U):
        t0_ = min(t0, last)
        t1_ = min(t0 + 1, last)
        t2_ = min(t0 + 2, last)
        t3_ = min(t0 + 3, last)
        t4_ = min(t0 + 4, last)
        t5_ = min(t0 + 5, last)
        t6_ = min(t0 + 6, last)
        t7_ = min(t0 + 7, last)
        r0 = t0_ // HTILES
        r1 = t1_ // HTILES
        r2 = t2_ // HTILES
        r3 = t3_ // HTILES
        r4 = t4_ // HTILES
        r5 = t5_ // HTILES
        r6 = t6_ // HTILES
        r7 = t7_ // HTILES
        c0 = (t0_ % HTILES) * BH
        c1 = (t1_ % HTILES) * BH
        c2 = (t2_ % HTILES) * BH
        c3 = (t3_ % HTILES) * BH
        c4 = (t4_ % HTILES) * BH
        c5 = (t5_ % HTILES) * BH
        c6 = (t6_ % HTILES) * BH
        c7 = (t7_ % HTILES) * BH
        offs = tl.arange(0, BH)
        ib0 = in_ptr + r0.to(tl.int64) * H
        ib1 = in_ptr + r1.to(tl.int64) * H
        ib2 = in_ptr + r2.to(tl.int64) * H
        ib3 = in_ptr + r3.to(tl.int64) * H
        ib4 = in_ptr + r4.to(tl.int64) * H
        ib5 = in_ptr + r5.to(tl.int64) * H
        ib6 = in_ptr + r6.to(tl.int64) * H
        ib7 = in_ptr + r7.to(tl.int64) * H
        g0 = tl.load(ib0 + c0 + offs)
        u0 = tl.load(ib0 + half + c0 + offs)
        g1 = tl.load(ib1 + c1 + offs)
        u1 = tl.load(ib1 + half + c1 + offs)
        g2 = tl.load(ib2 + c2 + offs)
        u2 = tl.load(ib2 + half + c2 + offs)
        g3 = tl.load(ib3 + c3 + offs)
        u3 = tl.load(ib3 + half + c3 + offs)
        g4 = tl.load(ib4 + c4 + offs)
        u4 = tl.load(ib4 + half + c4 + offs)
        g5 = tl.load(ib5 + c5 + offs)
        u5 = tl.load(ib5 + half + c5 + offs)
        g6 = tl.load(ib6 + c6 + offs)
        u6 = tl.load(ib6 + half + c6 + offs)
        g7 = tl.load(ib7 + c7 + offs)
        u7 = tl.load(ib7 + half + c7 + offs)
        ob0 = out_ptr + r0.to(tl.int64) * half
        ob1 = out_ptr + r1.to(tl.int64) * half
        ob2 = out_ptr + r2.to(tl.int64) * half
        ob3 = out_ptr + r3.to(tl.int64) * half
        ob4 = out_ptr + r4.to(tl.int64) * half
        ob5 = out_ptr + r5.to(tl.int64) * half
        ob6 = out_ptr + r6.to(tl.int64) * half
        ob7 = out_ptr + r7.to(tl.int64) * half
        m0 = t0 < n_tiles
        m1 = t0 + 1 < n_tiles
        m2 = t0 + 2 < n_tiles
        m3 = t0 + 3 < n_tiles
        m4 = t0 + 4 < n_tiles
        m5 = t0 + 5 < n_tiles
        m6 = t0 + 6 < n_tiles
        m7 = t0 + 7 < n_tiles
        if m0:
            gf0 = g0.to(tl.float32)
            uf0 = u0.to(tl.float32)
            tl.store(
                ob0 + c0 + offs,
                (tl.fdiv(gf0, 1.0 + tl.exp(-gf0)) * uf0).to(
                    out_ptr.dtype.element_ty
                ),
            )
        if m1:
            gf1 = g1.to(tl.float32)
            uf1 = u1.to(tl.float32)
            tl.store(
                ob1 + c1 + offs,
                (tl.fdiv(gf1, 1.0 + tl.exp(-gf1)) * uf1).to(
                    out_ptr.dtype.element_ty
                ),
            )
        if m2:
            gf2 = g2.to(tl.float32)
            uf2 = u2.to(tl.float32)
            tl.store(
                ob2 + c2 + offs,
                (tl.fdiv(gf2, 1.0 + tl.exp(-gf2)) * uf2).to(
                    out_ptr.dtype.element_ty
                ),
            )
        if m3:
            gf3 = g3.to(tl.float32)
            uf3 = u3.to(tl.float32)
            tl.store(
                ob3 + c3 + offs,
                (tl.fdiv(gf3, 1.0 + tl.exp(-gf3)) * uf3).to(
                    out_ptr.dtype.element_ty
                ),
            )
        if m4:
            gf4 = g4.to(tl.float32)
            uf4 = u4.to(tl.float32)
            tl.store(
                ob4 + c4 + offs,
                (tl.fdiv(gf4, 1.0 + tl.exp(-gf4)) * uf4).to(
                    out_ptr.dtype.element_ty
                ),
            )
        if m5:
            gf5 = g5.to(tl.float32)
            uf5 = u5.to(tl.float32)
            tl.store(
                ob5 + c5 + offs,
                (tl.fdiv(gf5, 1.0 + tl.exp(-gf5)) * uf5).to(
                    out_ptr.dtype.element_ty
                ),
            )
        if m6:
            gf6 = g6.to(tl.float32)
            uf6 = u6.to(tl.float32)
            tl.store(
                ob6 + c6 + offs,
                (tl.fdiv(gf6, 1.0 + tl.exp(-gf6)) * uf6).to(
                    out_ptr.dtype.element_ty
                ),
            )
        if m7:
            gf7 = g7.to(tl.float32)
            uf7 = u7.to(tl.float32)
            tl.store(
                ob7 + c7 + offs,
                (tl.fdiv(gf7, 1.0 + tl.exp(-gf7)) * uf7).to(
                    out_ptr.dtype.element_ty
                ),
            )


_FALLBACK_CONFIGS = [
    triton.Config(
        {"BLOCK_ROW": 16, "BLOCK_H": 2048}, num_warps=4, num_stages=1
    ),
    triton.Config(
        {"BLOCK_ROW": 32, "BLOCK_H": 2048}, num_warps=4, num_stages=1
    ),
    triton.Config(
        {"BLOCK_ROW": 64, "BLOCK_H": 2048}, num_warps=4, num_stages=1
    ),
]


@triton.autotune(configs=_FALLBACK_CONFIGS, key=["n_rows"])
@triton.jit
def _silu_and_mul_masked_fallback_kernel(
    in_ptr,
    out_ptr,
    n_rows,
    H: tl.constexpr,
    half: tl.constexpr,
    BLOCK_ROW: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    """Masked 2-D block-pointer path for degenerate column widths.

    Only reached when ``half`` has no power-of-two divisor >= _MIN_BH
    (e.g. odd or 8-mod-16 halves); every bench/correctness shape takes
    the flat kernel instead. One program per output tile, boundary
    checks keep the tail of the last tile in bounds.
    """
    pid_row = tl.program_id(0)
    pid_col = tl.program_id(1)
    row_off = pid_row * BLOCK_ROW
    col_off = pid_col * BLOCK_H
    in_gate = tl.make_block_ptr(
        in_ptr,
        (n_rows, H),
        (H, 1),
        (row_off, col_off),
        (BLOCK_ROW, BLOCK_H),
        (1, 0),
    )
    in_up = tl.make_block_ptr(
        in_ptr,
        (n_rows, H),
        (H, 1),
        (row_off, col_off + half),
        (BLOCK_ROW, BLOCK_H),
        (1, 0),
    )
    out_block = tl.make_block_ptr(
        out_ptr,
        (n_rows, half),
        (half, 1),
        (row_off, col_off),
        (BLOCK_ROW, BLOCK_H),
        (1, 0),
    )
    gate = tl.load(in_gate, boundary_check=(0, 1), padding_option="zero").to(
        tl.float32
    )
    up = tl.load(in_up, boundary_check=(0, 1), padding_option="zero").to(
        tl.float32
    )
    val = (gate * tl.sigmoid(gate) * up).to(out_ptr.dtype.element_ty)
    tl.store(out_block, val, boundary_check=(0, 1))


def silu_and_mul_masked(input, masked_m):
    """silu(gate) * up with per-expert row masking, bf16 out.

    Matches the reference's result on every valid row ``[e, :masked_m[e]]``.
    Padding rows (``t >= masked_m[e]``) are *also* written (their outputs are
    unspecified by the op's contract, so writing them is harmless), which lets
    the kernel skip the per-row ``masked_m[e]`` load and lets the host use
    ``torch.empty`` instead of ``torch.zeros`` -- neither zero-init nor a
    per-element write mask is needed. ``masked_m`` is therefore accepted to
    keep the public signature identical to the reference but is not consumed
    by the kernel.

    The main path flattens ``[E, T, H]`` to ``n_rows = E * T`` rows of the
    GLU layout ``gate = row[:, :half] / up = row[:, half:]`` and processes
    them as ``BH``-wide 1-D column tiles on a persistent grid of at most
    ``_MAX_CTAS`` programs (see the kernel docstring for the batching
    rationale). It requires ``half`` to have a power-of-two divisor of at
    least ``_MIN_BH``; otherwise the masked fallback kernel runs.

    ``H`` and ``half`` are passed to the kernels as ``tl.constexpr`` so the
    compiler folds the row / output strides into immediate addressing.
    """
    E, T, H = input.shape
    half = H // 2
    # ``torch.empty`` (not ``torch.zeros``): the kernel writes every row
    # unconditionally (including padding rows, whose outputs the correctness
    # contract leaves unspecified), so we never need a zero-init pass.
    out = torch.empty(E, T, half, dtype=input.dtype, device=input.device)
    if half == 0 or E == 0 or T == 0:
        return out

    # The kernels index the buffers linearly via the flat row id, so they only
    # work on contiguous input. ``contiguous()`` is a no-op when the input is
    # already contiguous -- the case for every bench/correctness input.
    inp = input.contiguous()

    n_rows = E * T
    bh = half & -half  # largest power-of-two divisor of `half`

    if bh >= _MIN_BH:
        bh = min(bh, _MAX_BH)
        htiles = half // bh
        n_tiles = n_rows * htiles
        nprog = max(1, min(_MAX_CTAS, triton.cdiv(n_tiles, _UNROLL)))
        _silu_and_mul_masked_flat_kernel[(nprog,)](
            inp,
            out,
            n_tiles,
            H=H,
            half=half,
            BH=bh,
            HTILES=htiles,
            num_warps=2,
            num_stages=1,
        )
    else:
        grid = lambda meta: (
            triton.cdiv(n_rows, meta["BLOCK_ROW"]),
            triton.cdiv(half, meta["BLOCK_H"]),
        )
        _silu_and_mul_masked_fallback_kernel[grid](
            inp,
            out,
            n_rows,
            H=H,
            half=half,
        )
    return out


__all__ = ["silu_and_mul_masked"]
