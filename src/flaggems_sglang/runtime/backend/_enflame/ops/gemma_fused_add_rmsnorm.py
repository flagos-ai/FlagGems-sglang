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
def _flat(X, R, W, Y, RO, N: tl.constexpr, EPS: tl.constexpr, B: tl.constexpr):
    base = tl.program_id(0) * N
    c = tl.arange(0, B)
    if B == N:
        x = tl.load(X + base + c, cache_modifier=".cg").to(tl.float32)
        r = tl.load(R + base + c, cache_modifier=".cg").to(tl.float32)
        w = tl.load(W + c).to(tl.float32)
    else:
        m = c < N
        x = tl.load(X + base + c, m, other=0, cache_modifier=".cg").to(
            tl.float32
        )
        r = tl.load(R + base + c, m, other=0, cache_modifier=".cg").to(
            tl.float32
        )
        w = tl.load(W + c, m, other=0).to(tl.float32)
    s = x + r
    y = s * tl.rsqrt(tl.sum(s * s, 0) / N + EPS) * (1.0 + w)
    if B == N:
        tl.store(Y + base + c, y, cache_modifier=".cs")
        tl.store(RO + base + c, s, cache_modifier=".cs")
    else:
        tl.store(Y + base + c, y, c < N, cache_modifier=".cs")
        tl.store(RO + base + c, s, c < N, cache_modifier=".cs")


@triton.jit
def _persist(
    X,
    R,
    W,
    Y,
    RO,
    M,
    RPG,
    N: tl.constexpr,
    EPS: tl.constexpr,
    B: tl.constexpr,
):
    # Persistent 1-D row-stream: GRID programs each own a contiguous block of
    # RPG rows and stream them as scalar-base 1-D affine tiles. The pre-scaled
    # weight stays resident and the per-program "load dispatch + reduction
    # tree" serial chain that dominates one-row-per-program grids is paid
    # once instead of once per row.  This arm covers 48 < rows < 256 (2-D
    # small-tile structural tax dominates there), BLOCK_N == 8192 and fp32
    # rows (the 4-stream [RT, B] tile of this op crosses the 448KB
    # dual-program DSM line at those widths, measured 2026-11-13); the 2-D
    # tile form itself compiles fine since the 2026-11-10 P01a toolchain
    # resolution (see _tile below).  Row unrolling (U>=2) was probed and is
    # consistently slower than U=1 on B >= 1024.
    pid = tl.program_id(0)
    c = tl.arange(0, B)
    if B == N:
        wp = 1.0 + tl.load(W + c).to(tl.float32)
        lo = pid * RPG
        hi = tl.minimum(lo + RPG, M)
        for row in range(lo, hi):
            base = row * N
            x = tl.load(X + base + c, cache_modifier=".cg").to(tl.float32)
            r = tl.load(R + base + c, cache_modifier=".cg").to(tl.float32)
            s = x + r
            y = s * tl.rsqrt(tl.sum(s * s, 0) / N + EPS) * wp
            tl.store(Y + base + c, y, cache_modifier=".cs")
            tl.store(RO + base + c, s, cache_modifier=".cs")
    else:
        m = c < N
        wp = 1.0 + tl.load(W + c, m, other=0).to(tl.float32)
        lo = pid * RPG
        hi = tl.minimum(lo + RPG, M)
        for row in range(lo, hi):
            base = row * N
            x = tl.load(X + base + c, m, other=0, cache_modifier=".cg").to(
                tl.float32
            )
            r = tl.load(R + base + c, m, other=0, cache_modifier=".cg").to(
                tl.float32
            )
            s = x + r
            y = s * tl.rsqrt(tl.sum(s * s, 0) / N + EPS) * wp
            tl.store(Y + base + c, y, m, cache_modifier=".cs")
            tl.store(RO + base + c, s, m, cache_modifier=".cs")


@triton.jit
def _tile(
    X,
    R,
    W,
    Y,
    RO,
    M,
    NUM_BLOCKS,
    N: tl.constexpr,
    EPS: tl.constexpr,
    B: tl.constexpr,
    RT: tl.constexpr,
    NPROG: tl.constexpr,
    EVEN_ROWS: tl.constexpr,
    EXACT_COLS: tl.constexpr,
):
    # Persistent [RT, B] row-tile kernel (2026-11-13 E-1 second round; the
    # first round was forced into the 1-D _persist form by the since-resolved
    # P01a slice_start fresh-compile regression).  One axis=1 reduction tree
    # per tile amortizes the per-row "load dispatch + tree" chain over RT
    # rows and merges the x/r loads and y/RO stores into single 2-D ops.
    # The weight is loaded with a 2-D shape ([1, B]) on purpose: a 1-D fp32
    # load broadcast as a 2-D tile multiply operand hits a known silent
    # miscompile (P14 broadcast-chain family); the 2-D-shaped load is the
    # verified immune recipe (same as the delivered vllm fused_add_rms_norm
    # and sglang fused_rmsnorm tile kernels).
    pid = tl.program_id(0)
    cols = tl.arange(0, B)
    if EXACT_COLS:
        wp = 1.0 + tl.load(W + cols[None, :]).to(tl.float32)
    else:
        wp = 1.0 + tl.load(
            W + cols[None, :], mask=(cols < N)[None, :], other=0.0
        ).to(tl.float32)
    for blk in tl.range(pid, NUM_BLOCKS, NPROG, num_stages=2):
        rows = blk * RT + tl.arange(0, RT)
        offs = rows[:, None] * N + cols[None, :]
        if EVEN_ROWS and EXACT_COLS:
            x = tl.load(X + offs).to(tl.float32)
            r = tl.load(R + offs).to(tl.float32)
            s = x + r
            rrms = tl.rsqrt(tl.sum(s * s, axis=1) / N + EPS)
            y = s * rrms[:, None] * wp
            tl.store(Y + offs, y)
            tl.store(RO + offs, s)
        else:
            if EVEN_ROWS:
                m = (cols < N)[None, :]
            elif EXACT_COLS:
                m = (rows < M)[:, None]
            else:
                m = (rows < M)[:, None] & (cols < N)[None, :]
            x = tl.load(X + offs, mask=m, other=0.0).to(tl.float32)
            r = tl.load(R + offs, mask=m, other=0.0).to(tl.float32)
            s = x + r
            rrms = tl.rsqrt(tl.sum(s * s, axis=1) / N + EPS)
            y = s * rrms[:, None] * wp
            tl.store(Y + offs, y, mask=m)
            tl.store(RO + offs, s, mask=m)


# Persistent row-stream dispatch (L600: 48 SPs). Above ~48 rows the
# one-row-per-program grid pays the per-program "load dispatch + reduction
# tree" serial chain per row (retirement-bound, ~15 programs/us); the
# persistent kernel keeps a fixed grid of 48 programs streaming contiguous
# row blocks instead. num_warps by row width: 1 up to B=2048, 2 at B=4096,
# 4 at B=8192 (B=8192 also needs num_warps >= 2 to compile: the fresh-compile
# slice_start regression is nw1-only for 1-D N=8192). Wider rows (B > 8192)
# keep the legacy kernel.
_NUM_SPC = 48
_PERSIST_MAX_BLOCK = 8192
_PERSIST_NUM_WARPS = {4096: 2, 8192: 4}
_PERSIST_NUM_STAGES = 2

# 2-D [RT, B] row-tile dispatch (gcu400/Libra-class 48 SPs, 2-byte dtypes,
# do_bench kernel mode w1000/r100 median, 2026-11-13):
#   - rows >= 256 and BLOCK_N <= 4096 take the persistent tile kernel:
#     (512,1024) 13.7->9.6us (RT32 nw2, 1.43x), (512,4096) 20.8->19.1us
#     (RT8, 1.09x), (256,1024) 8.9->7.9us (1.13x), (1024,4096) 36.3->27.8us
#     (1.31x), (4096,1024) 76.7->27.2us (2.82x), (4096,4096) 127.5->84.5us
#     (1.51x).  Tile budget 32768 elems (bf16/fp16): the 4 full streams
#     (x/r loads + y/RO stores) of this op stay inside the 448KB
#     dual-program DSM line there.
#   - BLOCK_N == 8192 keeps _persist: the same 32768-elem tile at that width
#     loses 27-34% ((512,8192) 33.8 vs 26.6us; (4096,8192) 228 vs 170us; the
#     65536-elem tile does not even fit DSM).  fp32 keeps _persist too
#     (untested perf; half-size 16384-elem fp32 tiles compile and are exact).
#   - rows < 256 keeps _persist/legacy: the 2-D tile ties at (256,4096) and
#     loses below ((128,4096) 9.6 vs 8.4us, (64,*) all negative) -- the 2-D
#     small-tile structural tax, same crossover as the fused_rmsnorm sibling.
_TILE_MIN_ROWS = 256
_TILE_MAX_BLOCK = 4096
_TILE_ELEMS = 32768
_TILE_MAX_ROWS = 32
_TILE_MIN_BLOCKS = 8
_TILE_NUM_WARPS = 2
_TILE_NUM_STAGES = 2


def _persist_num_warps(block):
    return _PERSIST_NUM_WARPS.get(block, 1)


def gemma_fused_add_rmsnorm(x, residual, weight, eps):
    n = x.size(-1)
    rows = x.numel() // n
    out = torch.empty_like(x, memory_format=torch.contiguous_format)
    res = torch.empty_like(x, memory_format=torch.contiguous_format)
    if rows:
        if not x.is_contiguous():
            x = x.contiguous()
        if not residual.is_contiguous():
            residual = residual.contiguous()
        if not weight.is_contiguous():
            weight = weight.contiguous()
        b = 1 << (n - 1).bit_length()
        if (
            rows >= _TILE_MIN_ROWS
            and b <= _TILE_MAX_BLOCK
            and x.element_size() < 4
        ):
            rt = min(_TILE_MAX_ROWS, _TILE_ELEMS // b)
            while rt > 2 and triton.cdiv(rows, rt) < _TILE_MIN_BLOCKS:
                rt //= 2
            num_blocks = triton.cdiv(rows, rt)
            nprog = min(_NUM_SPC, num_blocks)
            _tile.run(
                x,
                residual,
                weight,
                out,
                res,
                rows,
                num_blocks,
                n,
                eps,
                b,
                grid=(nprog,),
                warmup=False,
                RT=rt,
                NPROG=nprog,
                EVEN_ROWS=(rows % rt == 0),
                EXACT_COLS=(b == n),
                num_warps=_TILE_NUM_WARPS,
                num_stages=_TILE_NUM_STAGES,
                enable_fp_fusion=False,
            )
        elif rows > _NUM_SPC and b <= _PERSIST_MAX_BLOCK:
            grid = min(rows, _NUM_SPC)
            rpg = triton.cdiv(rows, grid)
            _persist.run(
                x,
                residual,
                weight,
                out,
                res,
                rows,
                rpg,
                n,
                eps,
                b,
                grid=(grid,),
                warmup=False,
                num_warps=_persist_num_warps(b),
                num_stages=_PERSIST_NUM_STAGES,
                enable_fp_fusion=False,
            )
        else:
            _flat.run(
                x,
                residual,
                weight,
                out,
                res,
                n,
                eps,
                b,
                grid=(rows,),
                warmup=False,
                num_warps=8,
                enable_fp_fusion=False,
            )
    return out, res


__all__ = ["gemma_fused_add_rmsnorm"]
