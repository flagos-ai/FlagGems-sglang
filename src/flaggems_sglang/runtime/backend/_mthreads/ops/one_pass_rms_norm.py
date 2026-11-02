# Copyright 2026, The FlagOS Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License")
# you may not use this file except in compliance with the License
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Triton RMSNorm without residual (SGLang `RMSNorm.forward_native` semantics).

    out = cast_to(dtype_x)( (xf * rsqrt(mean_i(xf^2) + eps)) * wf )    xf, wf in fp32

Arithmetic is fp32 end to end with exactly one rounding at the store, matching the
reference's single cast, and the normalizer is the true runtime H.

Three exact-vector kernels, all one program per row (or per row pair), chosen by
measured cost on the target:

  * `_rms_norm_row`   - BLOCK_H = next_power_of_2(H); power-of-two H compiles
    mask-free through the EVEN constexpr.
  * `_rms_norm_row2`  - two consecutive rows per program, both row loads issued
    before the reduction; fastest only in a narrow mid-grid band.
  * `_rms_norm_chunk` - the row is walked in chunks that divide H exactly, so no
    lane is ever masked off. This is the cheapest option for the large grids of
    the power-of-two hidden size.
  * `_rms_norm_split` - H = 2048 + 4096 for the non-power-of-two hidden size,
    again with every lane carrying data. Using BLOCK_H = 8192 for H = 6144 would
    waste a quarter of every vector operation.

All launch parameters come from L2-flushed sweeps on the target, whose dominant
effect is the row count M: with few rows the device fills better with wide CTAs,
while with many rows narrow CTAs pack more CTAs per SM and win.
"""

import torch
import triton
import triton.language as tl

_INF = 1 << 60


@triton.jit
def _rms_norm_row(
    X,  # *[M, H] contiguous
    W,  # *[H] contiguous
    Y,  # *[M, H] contiguous
    eps,  # float32
    H: tl.constexpr,
    BLOCK_H: tl.constexpr,
    EVEN: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK_H)
    base = pid.to(tl.int64) * H
    if EVEN:
        x = tl.load(X + base + offs)
        w = tl.load(W + offs)
    else:
        m = offs < H
        x = tl.load(X + base + offs, mask=m, other=0.0)
        w = tl.load(W + offs, mask=m, other=0.0)
    xf = x.to(tl.float32)
    rstd = tl.rsqrt(tl.sum(xf * xf, axis=0) / H + eps)
    y = (xf * rstd) * w.to(tl.float32)
    if EVEN:
        tl.store(Y + base + offs, y.to(Y.dtype.element_ty))
    else:
        tl.store(Y + base + offs, y.to(Y.dtype.element_ty), mask=offs < H)


@triton.jit
def _rms_norm_row2(
    X,
    W,
    Y,
    eps,
    H: tl.constexpr,
    BLOCK_H: tl.constexpr,
    EVEN: tl.constexpr,
):
    """Two consecutive rows per program; both row loads issue before the reduce."""
    ROWS: tl.constexpr = 2
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK_H)
    ptrs = (
        pid.to(tl.int64) * (ROWS * H)
        + tl.arange(0, ROWS)[:, None].to(tl.int64) * H
        + offs[None, :]
    )
    if EVEN:
        x = tl.load(X + ptrs)
        w = tl.load(W + offs)
    else:
        cm = offs[None, :] < H
        x = tl.load(X + ptrs, mask=cm, other=0.0)
        w = tl.load(W + offs, mask=offs < H, other=0.0)
    xf = x.to(tl.float32)
    rstd = tl.rsqrt(tl.sum(xf * xf, axis=1) / H + eps)
    y = (xf * rstd[:, None]) * w.to(tl.float32)[None, :]
    if EVEN:
        tl.store(Y + ptrs, y.to(Y.dtype.element_ty))
    else:
        tl.store(Y + ptrs, y.to(Y.dtype.element_ty), mask=cm)


@triton.jit
def _rms_norm_chunk(
    X,
    W,
    Y,
    eps,
    H: tl.constexpr,
    CH: tl.constexpr,
):
    """CH divides H: every chunk load and store is exact, nothing is masked."""
    NCH: tl.constexpr = H // CH
    pid = tl.program_id(0)
    base = pid.to(tl.int64) * H
    ss = tl.zeros((1,), dtype=tl.float32)
    for c in tl.static_range(NCH):
        offs = c * CH + tl.arange(0, CH)
        v = tl.load(X + base + offs).to(tl.float32)
        ss += tl.sum(v * v, axis=0)
    rstd = tl.rsqrt(ss / H + eps)
    for c in tl.static_range(NCH):
        offs = c * CH + tl.arange(0, CH)
        v = tl.load(X + base + offs).to(tl.float32)
        w = tl.load(W + offs).to(tl.float32)
        tl.store(Y + base + offs, ((v * rstd) * w).to(Y.dtype.element_ty))


@triton.jit
def _rms_norm_split(
    X,
    W,
    Y,
    eps,
    H: tl.constexpr,
    B1: tl.constexpr,
    B2: tl.constexpr,
):
    """H = B1 + B2 with both parts powers of two: every lane carries data."""
    pid = tl.program_id(0)
    base = pid.to(tl.int64) * H
    o1 = tl.arange(0, B1)
    o2 = B1 + tl.arange(0, B2)
    f1 = tl.load(X + base + o1).to(tl.float32)
    f2 = tl.load(X + base + o2).to(tl.float32)
    w1 = tl.load(W + o1).to(tl.float32)
    w2 = tl.load(W + o2).to(tl.float32)
    rstd = tl.rsqrt(
        (tl.sum(f1 * f1, axis=0) + tl.sum(f2 * f2, axis=0)) / H + eps
    )
    tl.store(Y + base + o1, ((f1 * rstd) * w1).to(Y.dtype.element_ty))
    tl.store(Y + base + o2, ((f2 * rstd) * w2).to(Y.dtype.element_ty))


# (inclusive row-count bound, num_warps) per measured BLOCK_H bucket.
_TABLE = {
    4096: ((64, 32), (128, 16), (1024, 8), (16384, 4), (_INF, 16)),
    8192: (
        (64, 32),
        (128, 16),
        (256, 8),
        (512, 4),
        (1024, 8),
        (8192, 4),
        (_INF, 16),
    ),
}
# BLOCK_H -> inclusive row-count band where two rows per program measured fastest.
_ROWS2_BAND = {4096: (256, 768)}
# H = 4096 walked in exact 1024-element chunks: measured faster than the
# single-vector row kernel across the wide mid-grid band.
_CHUNK_H, _CHUNK_CH, _CHUNK_NW = 4096, 1024, 4
_CHUNK_BAND = (4096, 16384)
# H = 6144 = 2048 + 4096 exact split, with its own measured num_warps table.
_SPLIT_H = 6144
_SPLIT_MIN_M = 256
_SPLIT_NW = ((2048, 8), (8192, 4), (_INF, 16))


def _plan(block_h, m):
    """Return (rows_per_program, num_warps) for this (BLOCK_H, row count)."""
    band = _ROWS2_BAND.get(block_h)
    if band is not None and band[0] < m <= band[1] and m % 2 == 0:
        return 2, 8
    table = _TABLE.get(block_h)
    if table is not None:
        for limit, nw in table:
            if m <= limit:
                return 1, nw
    # Hidden sizes outside the measured buckets: conservative width rule.
    return 1, max(4, min(16, block_h // 512))


def one_pass_rms_norm(x, weight, eps=1e-06):
    H = x.shape[-1]
    xr = x.reshape(-1, H)
    M = xr.shape[0]
    out = torch.empty_like(x)
    if M == 0:
        return out
    yr = out.reshape(-1, H)
    w = weight.reshape(-1)
    BLOCK_H = max(triton.next_power_of_2(H), 16)

    if H == _SPLIT_H and M >= _SPLIT_MIN_M:
        nw = _SPLIT_NW[-1][1]
        for limit, cand in _SPLIT_NW:
            if M <= limit:
                nw = cand
                break
        _rms_norm_split[(M,)](
            xr,
            w,
            yr,
            float(eps),
            H=H,
            B1=2048,
            B2=4096,
            num_warps=nw,
        )
        return out

    if H == _CHUNK_H and _CHUNK_BAND[0] <= M <= _CHUNK_BAND[1]:
        _rms_norm_chunk[(M,)](
            xr,
            w,
            yr,
            float(eps),
            H=H,
            CH=_CHUNK_CH,
            num_warps=_CHUNK_NW,
        )
        return out

    rows, num_warps = _plan(BLOCK_H, M)
    if rows == 2:
        _rms_norm_row2[(M // rows,)](
            xr,
            w,
            yr,
            float(eps),
            H=H,
            BLOCK_H=BLOCK_H,
            EVEN=(BLOCK_H == H),
            num_warps=num_warps,
        )
    else:
        _rms_norm_row[(M,)](
            xr,
            w,
            yr,
            float(eps),
            H=H,
            BLOCK_H=BLOCK_H,
            EVEN=(BLOCK_H == H),
            num_warps=num_warps,
        )
    return out


__all__ = ["one_pass_rms_norm"]
