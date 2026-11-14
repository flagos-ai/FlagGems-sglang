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

import torch
import triton
import triton.language as tl

_TILE_MAX_D = 8192

# Flat 1-D fast path (gcu400, probed 2026-11-08). When D has a power-of-two
# divisor >= 256 every BLOCK-sized chunk of the flattened tensor lies inside
# one row, so the gate reduces to one scalar load + sigmoid per chunk and the
# kernel becomes a pure 1-D stream with no masks at all. The enflame toolchain
# caps a 1-D load tile at 32 KiB per CTA / 8 KiB per warp (same
# memref_ext.slice_pad_start wall as fused_sigmoid_mul), hence the itemsize-
# scaled block ladder. Grids stay within one wave (24 SIP x 2 programs = 48):
# mono one-shot for <= 48 chunks, persistent grid-stride (P=48) above, and a
# 16-row-min exact 2-D tile (one wave, no loop, 2-D loads exempt from the 1-D
# wall) for large-N/pow2-D shapes where per-iteration scalar chains would
# dominate the flat loop.
_FLAT_CTA_BYTES = 32768
_FLAT_WARP_BYTES = 8192
_FLAT_MIN_LOWBIT = 256
_PERSISTENT_GRID = 48


@triton.jit
def _sgmb_row1(x_ptr, g_ptr, out_ptr, BLOCK_D: tl.constexpr):
    cols = tl.arange(0, BLOCK_D)
    x = tl.load(x_ptr + cols)
    sig = tl.sigmoid(tl.load(g_ptr).to(tl.float32)).to(x_ptr.dtype.element_ty)
    tl.store(out_ptr + cols, x * sig)


@triton.jit
def _sgmb_row1_masked(x_ptr, g_ptr, out_ptr, D, BLOCK_D: tl.constexpr):
    cols = tl.arange(0, BLOCK_D)
    mask = cols < D
    x = tl.load(x_ptr + cols, mask=mask, other=0.0)
    sig = tl.sigmoid(tl.load(g_ptr).to(tl.float32)).to(x_ptr.dtype.element_ty)
    tl.store(out_ptr + cols, x * sig, mask=mask)


@triton.jit
def _sgmb_flat(
    x_ptr, g_ptr, out_ptr, D, BPR: tl.constexpr, BLOCK: tl.constexpr
):
    pid = tl.program_id(0)
    row = pid // BPR
    sig = tl.sigmoid(tl.load(g_ptr + row).to(tl.float32)).to(
        x_ptr.dtype.element_ty
    )
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    tl.store(out_ptr + offs, tl.load(x_ptr + offs) * sig)


@triton.jit
def _sgmb_flat_wide(
    x_ptr,
    g_ptr,
    out_ptr,
    NT,
    D,
    BPR: tl.constexpr,
    BLOCK: tl.constexpr,
    P: tl.constexpr,
):
    for tile in tl.range(tl.program_id(0), NT, P):
        row = tile // BPR
        sig = tl.sigmoid(tl.load(g_ptr + row).to(tl.float32)).to(
            x_ptr.dtype.element_ty
        )
        offs = tile * BLOCK + tl.arange(0, BLOCK)
        tl.store(out_ptr + offs, tl.load(x_ptr + offs) * sig)


@triton.jit
def _sgmb_tile_exact(
    x_ptr, g_ptr, out_ptr, D, BLOCK_M: tl.constexpr, BLOCK_D: tl.constexpr
):
    pid = tl.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    g = tl.load(g_ptr + rows).to(tl.float32)
    sig = tl.sigmoid(g).to(x_ptr.dtype.element_ty)
    offs = rows[:, None] * BLOCK_D + tl.arange(0, BLOCK_D)[None, :]
    x = tl.load(x_ptr + offs)
    y = x * sig[:, None]
    tl.store(out_ptr + offs, y)


@triton.jit
def _sgmb_tile_exact_i64(
    x_ptr, g_ptr, out_ptr, D, BLOCK_M: tl.constexpr, BLOCK_D: tl.constexpr
):
    pid = tl.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    g = tl.load(g_ptr + rows).to(tl.float32)
    sig = tl.sigmoid(g).to(x_ptr.dtype.element_ty)
    offs = (
        rows[:, None].to(tl.int64) * BLOCK_D + tl.arange(0, BLOCK_D)[None, :]
    )
    x = tl.load(x_ptr + offs)
    y = x * sig[:, None]
    tl.store(out_ptr + offs, y)


@triton.jit
def _sgmb_tile_masked(
    x_ptr, g_ptr, out_ptr, N, D, BLOCK_M: tl.constexpr, BLOCK_D: tl.constexpr
):
    pid = tl.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    rmask = rows < N
    g = tl.load(g_ptr + rows, mask=rmask, other=0.0).to(tl.float32)
    sig = tl.sigmoid(g).to(x_ptr.dtype.element_ty)
    cols = tl.arange(0, BLOCK_D)
    mask = rmask[:, None] & (cols < D)[None, :]
    offs = rows[:, None] * D + cols[None, :]
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    y = x * sig[:, None]
    tl.store(out_ptr + offs, y, mask=mask)


@triton.jit
def _sgmb_tile_masked_i64(
    x_ptr, g_ptr, out_ptr, N, D, BLOCK_M: tl.constexpr, BLOCK_D: tl.constexpr
):
    pid = tl.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    rmask = rows < N
    g = tl.load(g_ptr + rows, mask=rmask, other=0.0).to(tl.float32)
    sig = tl.sigmoid(g).to(x_ptr.dtype.element_ty)
    cols = tl.arange(0, BLOCK_D)
    mask = rmask[:, None] & (cols < D)[None, :]
    offs = rows[:, None].to(tl.int64) * D + cols[None, :]
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    y = x * sig[:, None]
    tl.store(out_ptr + offs, y, mask=mask)


@triton.jit
def _sgmb_loop(
    x_ptr, g_ptr, out_ptr, N, D, BLOCK_M: tl.constexpr, BLOCK_H: tl.constexpr
):
    pid = tl.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    rmask = rows < N
    g = tl.load(g_ptr + rows, mask=rmask, other=0.0).to(tl.float32)
    sig = tl.sigmoid(g).to(x_ptr.dtype.element_ty)
    for h0 in range(0, D, BLOCK_H):
        cols = h0 + tl.arange(0, BLOCK_H)
        mask = rmask[:, None] & (cols < D)[None, :]
        offs = rows[:, None] * D + cols[None, :]
        x = tl.load(x_ptr + offs, mask=mask, other=0.0)
        y = x * sig[:, None]
        tl.store(out_ptr + offs, y, mask=mask)


def _next_pow2(v):
    return 1 << (max(v, 1) - 1).bit_length()


def _pick_tile(N, D):
    if N <= 1:
        return 1, (1 if D <= 2048 else 4)
    if N <= 8:
        return (8, 1) if D <= 2048 else (8, 8)
    if N <= 64:
        if D <= 2048:
            return 64, 4
        return 16, 8
    if N <= 512:
        if D <= 2048:
            return 128, 4
        if D <= 4096:
            return 64, 2
        return 16, 2
    if D <= 2048:
        return 128, 2
    if D <= 4096:
        return 64, 2
    return 16, 2


# gcu400 DSM limit 917504 B. Empirical model (probe, 2026-11-07):
# dsm = 2 * BLOCK_M * BLOCK_D * itemsize + overhead (nw1: +2K, nw2: +6K,
# nw>=4: +10K; num_stages-independent; identical for exact/masked/loop
# variants). 16384 B covers the worst observed overhead with margin.
_DSM_LIMIT_BYTES = 917504
_DSM_STAGING = 2
_DSM_OVERHEAD_BYTES = 16384


def _dsm_cap_block_m(block_m, block_d, item_size):
    while (
        block_m > 1
        and _DSM_STAGING * block_m * block_d * item_size + _DSM_OVERHEAD_BYTES
        > _DSM_LIMIT_BYTES
    ):
        block_m //= 2
    return block_m


def _pick_exact_bm(n, d, item_size):
    """Smallest >=16-row exact tile keeping the launch within one 48-program
    wave for large-N pow2-D shapes; 0 = not eligible."""
    if n < 256 or d > 4096 or d != _next_pow2(d):
        return 0
    for block_m in (16, 32, 64, 128, 256):
        if n % block_m == 0 and n // block_m <= _PERSISTENT_GRID:
            if _dsm_cap_block_m(block_m, d, item_size) == block_m:
                return block_m
            return 0
    return 0


def sigmoid_gate_mul_broadcast(x, gate):
    xc = x if x.is_contiguous() else x.contiguous()
    gc = gate if gate.is_contiguous() else gate.contiguous()
    out = torch.empty_like(xc)
    if xc.numel() == 0:
        return out

    D = xc.shape[-1] if xc.dim() >= 1 else 1
    N = xc.numel() // max(D, 1)
    numel = xc.numel()
    item_size = xc.element_size()

    if D <= _TILE_MAX_D:
        block_d = _next_pow2(D)
        if N == 1:
            if block_d == D:
                _sgmb_row1[(1,)](xc, gc, out, BLOCK_D=block_d, num_warps=2)
            else:
                _sgmb_row1_masked[(1,)](
                    xc, gc, out, D, BLOCK_D=block_d, num_warps=2
                )
            return out

        lowbit = D & (-D)
        if lowbit >= _FLAT_MIN_LOWBIT and numel < 2**31:
            block_m = _pick_exact_bm(N, D, item_size)
            if block_m:
                _sgmb_tile_exact[(N // block_m,)](
                    xc,
                    gc,
                    out,
                    D,
                    BLOCK_M=block_m,
                    BLOCK_D=D,
                    num_warps=1,
                )
                return out
            warp_cap = _FLAT_WARP_BYTES // item_size
            block = min(_FLAT_CTA_BYTES // item_size, lowbit)
            num_tiles = numel // block
            if num_tiles <= _PERSISTENT_GRID:
                _sgmb_flat[(num_tiles,)](
                    xc,
                    gc,
                    out,
                    D,
                    D // block,
                    BLOCK=block,
                    num_warps=max(1, triton.cdiv(block, warp_cap)),
                )
                return out
            block = min(_FLAT_CTA_BYTES // item_size // 2, lowbit)
            num_tiles = numel // block
            persistent = min(_PERSISTENT_GRID, num_tiles)
            if block * item_size >= _FLAT_WARP_BYTES or (
                persistent > 0 and triton.cdiv(num_tiles, persistent) <= 4
            ):
                _sgmb_flat_wide[(persistent,)](
                    xc,
                    gc,
                    out,
                    num_tiles,
                    D,
                    D // block,
                    BLOCK=block,
                    P=persistent,
                    num_warps=max(1, triton.cdiv(block, warp_cap)),
                )
                return out

        block_m, num_warps = _pick_tile(N, D)
        block_m = _dsm_cap_block_m(block_m, block_d, xc.element_size())
        if block_d == D and N % block_m == 0:
            if xc.numel() < 2**31:
                _sgmb_tile_exact[(N // block_m,)](
                    xc,
                    gc,
                    out,
                    D,
                    BLOCK_M=block_m,
                    BLOCK_D=block_d,
                    num_warps=num_warps,
                )
            else:
                _sgmb_loop[(triton.cdiv(N, block_m),)](
                    xc,
                    gc,
                    out,
                    N,
                    D,
                    BLOCK_M=block_m,
                    BLOCK_H=block_d,
                    num_warps=num_warps,
                )
        elif xc.numel() < 2**31:
            _sgmb_tile_masked[(triton.cdiv(N, block_m),)](
                xc,
                gc,
                out,
                N,
                D,
                BLOCK_M=block_m,
                BLOCK_D=block_d,
                num_warps=num_warps,
            )
        else:
            _sgmb_tile_masked_i64[(triton.cdiv(N, block_m),)](
                xc,
                gc,
                out,
                N,
                D,
                BLOCK_M=block_m,
                BLOCK_D=block_d,
                num_warps=num_warps,
            )
    else:
        block_m, num_warps = _pick_tile(N, min(D, 8192))
        block_m = _dsm_cap_block_m(block_m, 4096, xc.element_size())
        _sgmb_loop[(triton.cdiv(N, block_m),)](
            xc,
            gc,
            out,
            N,
            D,
            BLOCK_M=block_m,
            BLOCK_H=4096,
            num_warps=num_warps,
        )
    return out


__all__ = ["sigmoid_gate_mul_broadcast"]
