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
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import triton
import triton.language as tl


@triton.jit
def _offset(row, SH: tl.constexpr, ST: tl.constexpr):
    base = tl.full((), 0, tl.int64)
    for j in tl.static_range(len(SH) - 1, -1, -1):
        base += (row % SH[j]).to(tl.int64) * ST[j]
        row //= SH[j]
    return base


@triton.jit
def _row(
    X,
    W,
    S,
    Y,
    D: tl.constexpr,
    REP: tl.constexpr,
    XSH: tl.constexpr,
    XST: tl.constexpr,
    SSH: tl.constexpr,
    SST: tl.constexpr,
    EPS: tl.constexpr,
    B: tl.constexpr,
):
    row = tl.program_id(0)
    c = tl.arange(0, B)
    x = tl.load(X + _offset(row, XSH, XST) + c * XST[-1], c < D, other=0).to(
        tl.float32
    )
    w = tl.load(W + c, c < D, other=0).to(tl.float32)
    s = tl.load(
        S + _offset(row // REP, SSH, SST) + c * SST[-1], c < D, other=0
    ).to(tl.float32)
    inv = tl.rsqrt(tl.sum(x * x, 0) / D + EPS)
    n = (x * inv).to(tl.bfloat16).to(tl.float32)
    z = (n * w).to(tl.bfloat16).to(tl.float32)
    tl.store(Y + row * D + c, z * s, c < D)


@triton.jit
def _lean(X, W, S, Y, P: tl.constexpr):
    D: tl.constexpr = P[0]
    REP: tl.constexpr = P[1]
    EPS: tl.constexpr = P[2]
    B: tl.constexpr = P[3]
    CACHE: tl.constexpr = P[4]
    STORE: tl.constexpr = P[5]
    WC: tl.constexpr = P[6]
    row = tl.program_id(0)
    col = tl.arange(0, B)
    x = tl.load(X + row * D + col, col < D, other=0, cache_modifier=CACHE).to(
        tl.float32
    )
    w = tl.load(W + col, col < D, other=0, cache_modifier=WC).to(tl.float32)
    s = tl.load(
        S + row // REP * D + col, col < D, other=0, cache_modifier=CACHE
    ).to(tl.float32)
    inv = tl.rsqrt(tl.sum(x * x, 0) / D + EPS)
    n = (x * inv).to(tl.bfloat16).to(tl.float32)
    z = (n * w).to(tl.bfloat16).to(tl.float32)
    tl.store(Y + row * D + col, z * s, col < D, cache_modifier=STORE)


def _flat_rows(t):
    if t.ndim <= 2:
        return True
    row_stride = None
    rows = 1
    for size, stride in zip(reversed(t.shape[:-1]), reversed(t.stride()[:-1])):
        if size == 1:
            continue
        if row_stride is None:
            row_stride = stride
        elif stride != row_stride * rows:
            return False
        rows *= size
    return True


@triton.jit
def _rows_scale(
    X,
    W,
    S,
    Y,
    M,
    NUM_BLOCKS,
    R: tl.constexpr,
    REP: tl.constexpr,
    D: tl.constexpr,
    B: tl.constexpr,
    NPROG: tl.constexpr,
    EPS: tl.constexpr,
    EVEN_ROWS: tl.constexpr,
    EXACT_COLS: tl.constexpr,
):
    # Row-batched persistent kernel (GCU): one program covers R consecutive
    # rows, amortizing the per-program dispatch and the weight load.  All
    # memory ops stay strictly 1-D ([B] vectors); 2-D tile formulations of
    # this reduction family fail to compile or silently miscompile the
    # broadcast multiply chain on this backend (same reasoning as the
    # fused_rmsnorm rows kernel).  The R row chains are independent, so the
    # 2-stage pipeline over blocks overlaps the next block's loads with the
    # current block's reduction trees and stores.
    pid = tl.program_id(0)
    col = tl.arange(0, B)
    if EXACT_COLS:
        w = tl.load(W + col).to(tl.float32)
    else:
        w = tl.load(W + col, col < D, other=0).to(tl.float32)
    for blk in tl.range(pid, NUM_BLOCKS, NPROG, num_stages=2):
        r0 = blk * R
        for r in tl.static_range(R):
            row = r0 + r
            if EVEN_ROWS and EXACT_COLS:
                x = tl.load(X + row * D + col).to(tl.float32)
                s = tl.load(S + (row // REP) * D + col).to(tl.float32)
                inv = tl.rsqrt(tl.sum(x * x, 0) / D + EPS)
                n = (x * inv).to(tl.bfloat16).to(tl.float32)
                z = (n * w).to(tl.bfloat16).to(tl.float32)
                tl.store(Y + row * D + col, z * s)
            else:
                m = (row < M) & (col < D)
                x = tl.load(X + row * D + col, m, other=0).to(tl.float32)
                s = tl.load(S + (row // REP) * D + col, m, other=0).to(
                    tl.float32
                )
                inv = tl.rsqrt(tl.sum(x * x, 0) / D + EPS)
                n = (x * inv).to(tl.bfloat16).to(tl.float32)
                z = (n * w).to(tl.bfloat16).to(tl.float32)
                tl.store(Y + row * D + col, z * s, m)


@triton.jit
def _rows_scale_tile(
    X,
    W,
    S,
    Y,
    M,
    NUM_BLOCKS,
    eps,
    REP: tl.constexpr,
    N: tl.constexpr,
    RT: tl.constexpr,
    B: tl.constexpr,
    NPROG: tl.constexpr,
    EVEN_ROWS: tl.constexpr,
    EXACT_COLS: tl.constexpr,
):
    # Persistent [RT, B] row-tile kernel: one axis=1 reduction tree per tile
    # amortizes the per-row tree cost over RT rows and merges the per-row
    # loads/stores into single 2-D ops.  The 2-D tile path of this reduction
    # family used to be blocked by P01a (slice_(pad_)start fresh-compile
    # failure) and the P14 broadcast-multiply miscompile; both are resolved
    # (P01a lifted 2026-11-10, w 2-D-shaped load is the verified P14-immune
    # recipe, same as the fused_rmsnorm / fused_add_rmsnorm tile kernels).
    # The numeric chain keeps the three chained bf16 roundings of the
    # reference (norm -> *weight -> *scale) bit-for-bit.
    pid = tl.program_id(0)
    cols = tl.arange(0, B)
    if EXACT_COLS:
        w = tl.load(W + cols[None, :]).to(tl.float32)
    else:
        w = tl.load(W + cols[None, :], mask=(cols < N)[None, :], other=0.0).to(
            tl.float32
        )
    for blk in tl.range(pid, NUM_BLOCKS, NPROG, num_stages=2):
        rows = blk * RT + tl.arange(0, RT)
        offs = rows[:, None] * N + cols[None, :]
        if REP == 1:
            s_offs = offs
        else:
            s_offs = (rows // REP)[:, None] * N + cols[None, :]
        if EVEN_ROWS and EXACT_COLS:
            x = tl.load(X + offs).to(tl.float32)
            s = tl.load(S + s_offs).to(tl.float32)
            var = tl.sum(x * x, axis=1) / N
            rrms = tl.rsqrt(var + eps)
            n = (x * rrms[:, None]).to(tl.bfloat16).to(tl.float32)
            z = (n * w).to(tl.bfloat16).to(tl.float32)
            tl.store(Y + offs, z * s)
        else:
            if EVEN_ROWS:
                m = (cols < N)[None, :]
            elif EXACT_COLS:
                m = (rows < M)[:, None]
            else:
                m = (rows < M)[:, None] & (cols < N)[None, :]
            x = tl.load(X + offs, mask=m, other=0.0).to(tl.float32)
            s = tl.load(S + s_offs, mask=m, other=0.0).to(tl.float32)
            var = tl.sum(x * x, axis=1) / N
            rrms = tl.rsqrt(var + eps)
            n = (x * rrms[:, None]).to(tl.bfloat16).to(tl.float32)
            z = (n * w).to(tl.bfloat16).to(tl.float32)
            tl.store(Y + offs, z * s, mask=m)


# Dispatch calibration (gcu400/Libra-class 48 SPs, bf16, benchmark kernel
# mode, 2026-11-08; mirrors the fused_rmsnorm rows kernel on the same
# toolchain):
#   - num_warps=1 wins for every exact-width load <= 4096 (shuffle-only
#     reduction tree, no inter-warp DSM hop); masked columns use nw2;
#     B=8192 needs nw4.
#   - R=2 for wide rows (B > 1024), R=4 for narrow ones; below 64 rows the
#     launch-latency-bound whole-row kernel is kept.
#
# Tile calibration (same board/protocol, 2026-11-12, E-1 round 2): M >= 256
# takes the persistent [RT, B] tile kernel (one axis=1 tree per RT-row tile,
# w loaded with a 2-D shape).  Measured DSM (ASTSource llir
# gcu.dsm_memory_size): 65536-elem tile = 395264B @nw1 / 397312B @nw2,
# 32768-elem = ~199-201KB — both inside the 448KB dual-program line, and
# nw2 costs only +2KB here (unlike the scale_residual nw2 doubling).
# Tile budget is work-dependent: 65536 elems when M*N >= 2**23
# ((4096,2048)/(2048,4096)+ prefer RT32/RT16) else 32768 ((1024,*) and
# (2048,2048) prefer the smaller tile — at low block counts the smaller
# tile keeps more programs busy).  RT shrinks while blocks < 32.  nw2 beats
# nw1 on every measured shape.  Below 256 rows the 2-D tile loses or ties
# ((128,2048) 7.19 vs 7.39, (64,*) +9-16%) — the 1-D rows arm is kept.
_ROWS_MIN_ROWS = 64
_TILE_MIN_ROWS = 256
_TILE_ELEMS_LARGE = 65536
_TILE_ELEMS_SMALL = 32768
_TILE_LARGE_WORK = 1 << 23
_TILE_MAX_ROWS = 32
_TILE_MIN_BLOCKS = 32
_NUM_SPC = 48


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


def rmsnorm_scale(x, weight, scale, eps):
    xshape = x.shape
    sshape = scale.shape
    if not xshape or not sshape:
        return None
    if not x.dtype == weight.dtype == scale.dtype == torch.bfloat16:
        return None
    if not x.device == weight.device == scale.device:
        return None
    d = xshape[-1]
    if (
        not 0 < d <= 8192
        or weight.shape != (d,)
        or (not weight.is_contiguous())
        or (sshape[-1] != d)
    ):
        return None
    m, sm = (xshape.numel() // d, sshape.numel() // d)
    if sm == 0 or m % sm != 0:
        return None
    xc, sc = (x.is_contiguous(), scale.is_contiguous())
    if not xc and (not _flat_rows(x)) or (not sc and (not _flat_rows(scale))):
        return None
    y = (
        torch.empty_like(x)
        if xc
        else torch.empty_like(x, memory_format=torch.contiguous_format)
    )
    b = 1 << max(6, (d - 1).bit_length())
    if m:
        if xc and sc and torch.version.hip is None and m >= _ROWS_MIN_ROWS:
            if m >= _TILE_MIN_ROWS:
                tile_elems = (
                    _TILE_ELEMS_LARGE
                    if m * d >= _TILE_LARGE_WORK
                    else _TILE_ELEMS_SMALL
                )
                rows_per_cta = min(_TILE_MAX_ROWS, tile_elems // b)
                while (
                    rows_per_cta > 2
                    and -(-m // rows_per_cta) < _TILE_MIN_BLOCKS
                ):
                    rows_per_cta //= 2
                num_blocks = -(-m // rows_per_cta)
                _rows_scale_tile[(_NUM_SPC,)](
                    x,
                    weight,
                    scale,
                    y,
                    m,
                    num_blocks,
                    float(eps),
                    REP=m // sm,
                    N=d,
                    RT=rows_per_cta,
                    B=b,
                    NPROG=_NUM_SPC,
                    EVEN_ROWS=(m % rows_per_cta == 0),
                    EXACT_COLS=(b == d),
                    num_warps=2,
                    num_stages=2,
                    enable_fp_fusion=False,
                )
                return y
            exact_cols = b == d
            rows_per_cta, num_warps = _rows_config(m, b, exact_cols)
            num_blocks = -(-m // rows_per_cta)
            grid = (min(_NUM_SPC, num_blocks),)
            _rows_scale[grid](
                x,
                weight,
                scale,
                y,
                m,
                num_blocks,
                R=rows_per_cta,
                REP=m // sm,
                D=d,
                B=b,
                NPROG=grid[0],
                EPS=float(eps),
                EVEN_ROWS=(m % rows_per_cta == 0),
                EXACT_COLS=exact_cols,
                num_warps=num_warps,
                enable_fp_fusion=False,
            )
            return y
        if xc and sc:
            xp, wp, sp, yp = (
                x.data_ptr(),
                weight.data_ptr(),
                scale.data_ptr(),
                y.data_ptr(),
            )
            input_alignment = xp | wp | sp
            aligned = (input_alignment | yp) & 15 == 0
            if not aligned:
                xp, wp, sp, yp = (x, weight, scale, y)
        else:
            aligned = False
            xp, wp, sp, yp = (x, weight, scale, y)
        if xc and sc:
            bf = (
                torch.version.hip is not None
                and d == 2048
                and (1024 <= m < 8192)
            )
            (
                (
                    _address_hint_lean_delayed_scale
                    if aligned
                    else _lean_delayed_scale
                )
                if torch.version.hip is not None and d == 4096 and (m >= 1024)
                else (
                    (_address_hint_bf_lean if aligned else _bf_lean)
                    if bf
                    else _address_hint_lean if aligned else _lean
                )
            ).run(
                xp,
                wp,
                sp,
                yp,
                (
                    d,
                    m // sm,
                    float(eps),
                    b,
                    (
                        ".cg"
                        if torch.version.hip is not None or m >= 1024
                        else ""
                    ),
                    (
                        ".cs"
                        if torch.version.hip is not None and m >= 1024
                        else ""
                    ),
                    ".cg" if torch.version.hip is not None else "",
                ),
                grid=(m,),
                warmup=False,
                num_warps=(
                    2
                    if bf
                    else (
                        4
                        if torch.version.hip is None
                        and d <= 2048
                        and (m >= 1024)
                        else 8
                    )
                ),
                enable_fp_fusion=False,
            )
        else:
            _row[m,](
                x,
                weight,
                scale,
                y,
                d,
                m // sm,
                tuple(x.shape[:-1]),
                tuple(x.stride()),
                tuple(scale.shape[:-1]),
                tuple(scale.stride()),
                float(eps),
                b,
                num_warps=4 if b <= 2048 else 8,
                num_stages=1,
                enable_fp_fusion=False,
            )
    return y


@triton.jit
def _bf_lean(X, W, S, Y, P: tl.constexpr):
    D: tl.constexpr = P[0]
    REP: tl.constexpr = P[1]
    EPS: tl.constexpr = P[2]
    B: tl.constexpr = P[3]
    CACHE: tl.constexpr = P[4]
    STORE: tl.constexpr = P[5]
    WC: tl.constexpr = P[6]
    row = tl.program_id(0)
    col = tl.arange(0, B)
    x = tl.load(X + row * D + col, col < D, other=0, cache_modifier=CACHE).to(
        tl.float32
    )
    w = tl.load(W + col, col < D, other=0, cache_modifier=WC)
    s = tl.load(
        S + row // REP * D + col, col < D, other=0, cache_modifier=CACHE
    )
    inv = tl.rsqrt(tl.sum(x * x, 0) / D + EPS)
    n = (x * inv).to(tl.bfloat16)
    z = (n * w).to(tl.bfloat16)
    tl.store(Y + row * D + col, z * s, col < D, cache_modifier=STORE)


@triton.jit
def _lean_delayed_scale(X, W, S, Y, P: tl.constexpr):
    D: tl.constexpr = P[0]
    REP: tl.constexpr = P[1]
    EPS: tl.constexpr = P[2]
    B: tl.constexpr = P[3]
    CACHE: tl.constexpr = P[4]
    STORE: tl.constexpr = P[5]
    WC: tl.constexpr = P[6]
    row = tl.program_id(0)
    col = tl.arange(0, B)
    x = tl.load(X + row * D + col, col < D, other=0, cache_modifier=CACHE).to(
        tl.float32
    )
    w = tl.load(W + col, col < D, other=0, cache_modifier=WC).to(tl.float32)
    inv = tl.rsqrt(tl.sum(x * x, 0) / D + EPS)
    s = tl.load(
        S + row // REP * D + col, col < D, other=0, cache_modifier=CACHE
    ).to(tl.float32)
    n = (x * inv).to(tl.bfloat16).to(tl.float32)
    z = (n * w).to(tl.bfloat16).to(tl.float32)
    tl.store(Y + row * D + col, z * s, col < D, cache_modifier=STORE)


@triton.jit(do_not_specialize=["X", "W", "S", "Y"])
def _address_hint_bf_lean(X, W, S, Y, P: tl.constexpr):
    X = X.to(tl.pointer_type(tl.bfloat16))
    W = W.to(tl.pointer_type(tl.bfloat16))
    S = S.to(tl.pointer_type(tl.bfloat16))
    Y = Y.to(tl.pointer_type(tl.bfloat16))
    X = tl.multiple_of(X, 16)
    W = tl.multiple_of(W, 16)
    S = tl.multiple_of(S, 16)
    Y = tl.multiple_of(Y, 16)
    D: tl.constexpr = P[0]
    REP: tl.constexpr = P[1]
    EPS: tl.constexpr = P[2]
    B: tl.constexpr = P[3]
    CACHE: tl.constexpr = P[4]
    STORE: tl.constexpr = P[5]
    WC: tl.constexpr = P[6]
    row = tl.program_id(0)
    col = tl.arange(0, B)
    x = tl.load(X + row * D + col, col < D, other=0, cache_modifier=CACHE).to(
        tl.float32
    )
    w = tl.load(W + col, col < D, other=0, cache_modifier=WC)
    s = tl.load(
        S + row // REP * D + col, col < D, other=0, cache_modifier=CACHE
    )
    inv = tl.rsqrt(tl.sum(x * x, 0) / D + EPS)
    n = (x * inv).to(tl.bfloat16)
    z = (n * w).to(tl.bfloat16)
    tl.store(Y + row * D + col, z * s, col < D, cache_modifier=STORE)


@triton.jit(do_not_specialize=["X", "W", "S", "Y"])
def _address_hint_lean(X, W, S, Y, P: tl.constexpr):
    X = X.to(tl.pointer_type(tl.bfloat16))
    W = W.to(tl.pointer_type(tl.bfloat16))
    S = S.to(tl.pointer_type(tl.bfloat16))
    Y = Y.to(tl.pointer_type(tl.bfloat16))
    X = tl.multiple_of(X, 16)
    W = tl.multiple_of(W, 16)
    S = tl.multiple_of(S, 16)
    Y = tl.multiple_of(Y, 16)
    D: tl.constexpr = P[0]
    REP: tl.constexpr = P[1]
    EPS: tl.constexpr = P[2]
    B: tl.constexpr = P[3]
    CACHE: tl.constexpr = P[4]
    STORE: tl.constexpr = P[5]
    WC: tl.constexpr = P[6]
    row = tl.program_id(0)
    col = tl.arange(0, B)
    x = tl.load(X + row * D + col, col < D, other=0, cache_modifier=CACHE).to(
        tl.float32
    )
    w = tl.load(W + col, col < D, other=0, cache_modifier=WC).to(tl.float32)
    s = tl.load(
        S + row // REP * D + col, col < D, other=0, cache_modifier=CACHE
    ).to(tl.float32)
    inv = tl.rsqrt(tl.sum(x * x, 0) / D + EPS)
    n = (x * inv).to(tl.bfloat16).to(tl.float32)
    z = (n * w).to(tl.bfloat16).to(tl.float32)
    tl.store(Y + row * D + col, z * s, col < D, cache_modifier=STORE)


@triton.jit(do_not_specialize=["X", "W", "S", "Y"])
def _address_hint_lean_delayed_scale(X, W, S, Y, P: tl.constexpr):
    X = X.to(tl.pointer_type(tl.bfloat16))
    W = W.to(tl.pointer_type(tl.bfloat16))
    S = S.to(tl.pointer_type(tl.bfloat16))
    Y = Y.to(tl.pointer_type(tl.bfloat16))
    X = tl.multiple_of(X, 16)
    W = tl.multiple_of(W, 16)
    S = tl.multiple_of(S, 16)
    Y = tl.multiple_of(Y, 16)
    D: tl.constexpr = P[0]
    REP: tl.constexpr = P[1]
    EPS: tl.constexpr = P[2]
    B: tl.constexpr = P[3]
    CACHE: tl.constexpr = P[4]
    STORE: tl.constexpr = P[5]
    WC: tl.constexpr = P[6]
    row = tl.program_id(0)
    col = tl.arange(0, B)
    x = tl.load(X + row * D + col, col < D, other=0, cache_modifier=CACHE).to(
        tl.float32
    )
    w = tl.load(W + col, col < D, other=0, cache_modifier=WC).to(tl.float32)
    inv = tl.rsqrt(tl.sum(x * x, 0) / D + EPS)
    s = tl.load(
        S + row // REP * D + col, col < D, other=0, cache_modifier=CACHE
    ).to(tl.float32)
    n = (x * inv).to(tl.bfloat16).to(tl.float32)
    z = (n * w).to(tl.bfloat16).to(tl.float32)
    tl.store(Y + row * D + col, z * s, col < D, cache_modifier=STORE)


__all__ = ["rmsnorm_scale"]
