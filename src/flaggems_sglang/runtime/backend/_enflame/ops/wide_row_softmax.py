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

"""wide_row_softmax -- Triton kernel (FlagOS S2 Track-1, Batch 8, Task 126).

Version: enflame r7 (2026-11-13) -- GCU perf rewrite of the qoder r5 layout.

r7 = r6 + the multi-program BLOCK_V=8192 dispatch arm (see the r7 bullet
below); kernels themselves are unchanged from r6 (2026-11-08).

The r5 two-launch (partial + normalise) structure is kept, but both kernels
are re-tiled for the GCU retirement/dispatch floor that pinned the original
2048-wide one-tile-per-program form at ~4-10% of streaming bandwidth
(bench core: (4,131072) 48.9 -> 13.0us, (1,131072) 17.9 -> 7.9us):

- B >= 2 rows: 2D [R, BLOCK_V] row-batch tiles (R = min(prev_pow2(B), 4)),
  num_warps=2 -- merges the per-row load/reduce/store chains the way the
  rmsnorm_hf rowbatch does and quadruples the bytes per program retirement.
- r7 (2026-11-13) column-tile re-scan for the multi-program regime: the r6
  tuning was done at B in {1, 4} where grid <= 48 (one wave) -- at B >= 8 the
  grid enters the multi-wave regime and the per-program dispatch cost
  (~0.2-0.4us/program, measured) dominates.  Shapes with enough programs
  (grid_8192 = ceil(B/4) * ceil(V/8192) >= 16, B >= 4, V >= 8192) now use
  BLOCK_V=8192: [4, 8192] tiles are 260KB of program DSM (llir local_mem,
  still under the 448KB double-program line), halving the program count --
  official-bench-protocol A/B, do_bench w1000/r100 median, 3 interleaved
  rounds: (32,131072) 69.3 -> 49.3us, (32,32000) 40.4 -> 31.1us,
  (4,131072) 12.96 -> 13.0us WITHIN-NOISE.  Small grids keep the r6 tiles
  ((4,32000) measured 17.5 -> 20.4us with BLOCK_V=8192, grid=4 only).
  Tiles beyond 128KB are out: [8,8192]/[4,16384] exceed the 448KB/SP
  double-program line ([4,16384] norm needs 1MB > 896KB hardware limit,
  OutOfResources; [8,8192] compiles but measured slower), and [2,8192]
  hits a compiler DSM allocation pathology (1MB requested) -- the R=4
  requirement of this arm avoids it by construction.
- B == 1: 1D tiles (2D [1,N] tiles measured ~1.7x slower); V >= 65536 uses
  BLOCK_V=2048 with a textually batched pair of tiles per program (K=2, all
  loads traced before the compute chains) and num_warps=1; smaller V uses
  BLOCK_V=1024, num_warps=4.
- V % BLOCK_V == 0 (and B % R == 0) dispatches an EXACT arm with fully
  unmasked affine loads; measured ~1.4x over the clamped-index masked arm.
- Temperature divides become one reciprocal per row, multiplied per tile.
- Grid is capped at 65535 (grid.x hardware limit) by growing R up to 8; the
  r5 form launched B*C programs with no cap.
"""
import torch
import triton
import triton.language as tl

OP = "wide_row_softmax"
NEG_INF = tl.constexpr(float("-inf"))


@triton.jit
def _wsr_partial_1d(
    logits_ptr,
    temp_ptr,
    pmax_ptr,
    psum_ptr,
    L_S0: tl.constexpr,
    L_S1: tl.constexpr,
    V: tl.constexpr,
    C: tl.constexpr,
    HAS_TEMP: tl.constexpr,
    BLOCK_V: tl.constexpr,
    K: tl.constexpr,
    EXACT: tl.constexpr,
):
    pid = tl.program_id(0)
    nck = (C + K - 1) // K
    b = pid // nck
    ck = pid - b * nck
    cb0 = ck * K
    base = b * C
    lbase = b * L_S0
    if HAS_TEMP:
        inv_t = 1.0 / tl.load(temp_ptr + b)
    if EXACT:
        cb0c = tl.minimum(cb0, C - 1)
        cb1c = tl.minimum(cb0 + 1, C - 1)
        ar = tl.arange(0, BLOCK_V)
        x0 = tl.load(logits_ptr + lbase + (cb0c * BLOCK_V + ar) * L_S1)
        if K >= 2:
            x1 = tl.load(logits_ptr + lbase + (cb1c * BLOCK_V + ar) * L_S1)
        if HAS_TEMP:
            x0 = x0 * inv_t
            if K >= 2:
                x1 = x1 * inv_t
        lm0 = tl.max(x0)
        e0 = tl.exp(x0 - lm0)
        tl.store(pmax_ptr + base + cb0c, lm0)
        tl.store(psum_ptr + base + cb0c, tl.sum(e0))
        if K >= 2:
            lm1 = tl.max(x1)
            e1 = tl.exp(x1 - lm1)
            tl.store(pmax_ptr + base + cb1c, lm1)
            tl.store(psum_ptr + base + cb1c, tl.sum(e1))
    else:
        ar = tl.arange(0, BLOCK_V)
        cb0c = tl.minimum(cb0, C - 1)
        cb1c = tl.minimum(cb0 + 1, C - 1)
        cols0 = cb0c * BLOCK_V + ar
        cm0 = cols0 < V
        x0 = tl.load(logits_ptr + lbase + tl.where(cm0, cols0, V - 1) * L_S1)
        if K >= 2:
            cols1 = cb1c * BLOCK_V + ar
            cm1 = cols1 < V
            x1 = tl.load(logits_ptr + lbase + tl.where(cm1, cols1, V - 1) * L_S1)
        if HAS_TEMP:
            x0 = x0 * inv_t
            if K >= 2:
                x1 = x1 * inv_t
        x0 = tl.where(cm0, x0, NEG_INF)
        lm0 = tl.max(x0)
        e0 = tl.exp(x0 - lm0)
        e0 = tl.where(cm0, e0, 0.0)
        tl.store(pmax_ptr + base + cb0c, lm0)
        tl.store(psum_ptr + base + cb0c, tl.sum(e0))
        if K >= 2:
            x1 = tl.where(cm1, x1, NEG_INF)
            lm1 = tl.max(x1)
            e1 = tl.exp(x1 - lm1)
            e1 = tl.where(cm1, e1, 0.0)
            tl.store(pmax_ptr + base + cb1c, lm1)
            tl.store(psum_ptr + base + cb1c, tl.sum(e1))


@triton.jit
def _wsr_norm_1d(
    logits_ptr,
    temp_ptr,
    pmax_ptr,
    psum_ptr,
    out_ptr,
    L_S0: tl.constexpr,
    L_S1: tl.constexpr,
    O_S0: tl.constexpr,
    O_S1: tl.constexpr,
    V: tl.constexpr,
    C: tl.constexpr,
    BLOCK_C: tl.constexpr,
    HAS_TEMP: tl.constexpr,
    BLOCK_V: tl.constexpr,
    K: tl.constexpr,
    EXACT: tl.constexpr,
):
    pid = tl.program_id(0)
    nck = (C + K - 1) // K
    b = pid // nck
    ck = pid - b * nck
    cb0 = ck * K
    offs = tl.arange(0, BLOCK_C)
    m = offs < C
    osafe = tl.where(m, offs, 0)
    pm = tl.load(pmax_ptr + b * C + osafe)
    pm = tl.where(m, pm, NEG_INF)
    gmax = tl.max(pm)
    ps = tl.load(psum_ptr + b * C + osafe)
    ps = tl.where(m, ps, 0.0)
    gsum = tl.sum(ps * tl.exp(pm - gmax))
    inv = 1.0 / gsum
    lbase = b * L_S0
    obase = b * O_S0
    if HAS_TEMP:
        inv_t = 1.0 / tl.load(temp_ptr + b)
    if EXACT:
        cb0c = tl.minimum(cb0, C - 1)
        cb1c = tl.minimum(cb0 + 1, C - 1)
        ar = tl.arange(0, BLOCK_V)
        x0 = tl.load(logits_ptr + lbase + (cb0c * BLOCK_V + ar) * L_S1)
        if K >= 2:
            x1 = tl.load(logits_ptr + lbase + (cb1c * BLOCK_V + ar) * L_S1)
        if HAS_TEMP:
            x0 = x0 * inv_t
            if K >= 2:
                x1 = x1 * inv_t
        y0 = tl.exp(x0 - gmax) * inv
        tl.store(out_ptr + obase + (cb0c * BLOCK_V + ar) * O_S1, y0)
        if K >= 2:
            y1 = tl.exp(x1 - gmax) * inv
            tl.store(out_ptr + obase + (cb1c * BLOCK_V + ar) * O_S1, y1)
    else:
        ar = tl.arange(0, BLOCK_V)
        cb0c = tl.minimum(cb0, C - 1)
        cb1c = tl.minimum(cb0 + 1, C - 1)
        cols0 = cb0c * BLOCK_V + ar
        cm0 = cols0 < V
        x0 = tl.load(logits_ptr + lbase + tl.where(cm0, cols0, V - 1) * L_S1)
        if K >= 2:
            cols1 = cb1c * BLOCK_V + ar
            cm1 = cols1 < V
            x1 = tl.load(logits_ptr + lbase + tl.where(cm1, cols1, V - 1) * L_S1)
        if HAS_TEMP:
            x0 = x0 * inv_t
            if K >= 2:
                x1 = x1 * inv_t
        y0 = tl.exp(x0 - gmax) * inv
        tl.store(out_ptr + obase + cols0 * O_S1, y0, mask=cm0)
        if K >= 2:
            y1 = tl.exp(x1 - gmax) * inv
            tl.store(out_ptr + obase + cols1 * O_S1, y1, mask=cm1)


@triton.jit
def _wsr_partial_2d(
    logits_ptr,
    temp_ptr,
    pmax_ptr,
    psum_ptr,
    L_S0: tl.constexpr,
    L_S1: tl.constexpr,
    B: tl.constexpr,
    V: tl.constexpr,
    C: tl.constexpr,
    HAS_TEMP: tl.constexpr,
    BLOCK_V: tl.constexpr,
    R: tl.constexpr,
    EXACT: tl.constexpr,
):
    pid = tl.program_id(0)
    g = pid // C
    cb = pid - g * C
    rows = g * R + tl.arange(0, R)
    if EXACT:
        cols = cb * BLOCK_V + tl.arange(0, BLOCK_V)
        x = tl.load(logits_ptr + rows[:, None] * L_S0 + cols[None, :] * L_S1)
        if HAS_TEMP:
            inv_t = (1.0 / tl.load(temp_ptr + rows))[:, None]
            x = x * inv_t
        lm = tl.max(x, axis=1)
        e = tl.exp(x - lm[:, None])
        lse = tl.sum(e, axis=1)
        tl.store(pmax_ptr + rows * C + cb, lm)
        tl.store(psum_ptr + rows * C + cb, lse)
    else:
        rmask = rows < B
        rsafe = tl.where(rmask, rows, B - 1)
        cols = cb * BLOCK_V + tl.arange(0, BLOCK_V)
        cm = cols < V
        lmask = rmask[:, None] & cm[None, :]
        x = tl.load(
            logits_ptr + rsafe[:, None] * L_S0
            + tl.where(cm, cols, V - 1)[None, :] * L_S1
        )
        if HAS_TEMP:
            inv_t = (1.0 / tl.load(temp_ptr + rows, mask=rmask, other=1.0))[:, None]
            x = x * inv_t
        x = tl.where(lmask, x, NEG_INF)
        lm = tl.max(x, axis=1)
        e = tl.exp(x - lm[:, None])
        e = tl.where(lmask, e, 0.0)
        lse = tl.sum(e, axis=1)
        tl.store(pmax_ptr + rows * C + cb, lm, mask=rmask)
        tl.store(psum_ptr + rows * C + cb, lse, mask=rmask)


@triton.jit
def _wsr_norm_2d(
    logits_ptr,
    temp_ptr,
    pmax_ptr,
    psum_ptr,
    out_ptr,
    L_S0: tl.constexpr,
    L_S1: tl.constexpr,
    O_S0: tl.constexpr,
    O_S1: tl.constexpr,
    B: tl.constexpr,
    V: tl.constexpr,
    C: tl.constexpr,
    BLOCK_C: tl.constexpr,
    HAS_TEMP: tl.constexpr,
    BLOCK_V: tl.constexpr,
    R: tl.constexpr,
    EXACT: tl.constexpr,
):
    pid = tl.program_id(0)
    g = pid // C
    cb = pid - g * C
    rows = g * R + tl.arange(0, R)
    offs = tl.arange(0, BLOCK_C)
    rmask = rows < B
    mo = offs < C
    mmask = rmask[:, None] & mo[None, :]
    pm = tl.load(pmax_ptr + rows[:, None] * C + offs[None, :],
                 mask=mmask, other=NEG_INF)
    gmax = tl.max(pm, axis=1)
    ps = tl.load(psum_ptr + rows[:, None] * C + offs[None, :],
                 mask=mmask, other=0.0)
    gsum = tl.sum(ps * tl.exp(pm - gmax[:, None]), axis=1)
    inv = (1.0 / gsum)[:, None]
    if EXACT:
        cols = cb * BLOCK_V + tl.arange(0, BLOCK_V)
        x = tl.load(logits_ptr + rows[:, None] * L_S0 + cols[None, :] * L_S1)
        if HAS_TEMP:
            inv_t = (1.0 / tl.load(temp_ptr + rows))[:, None]
            x = x * inv_t
        y = tl.exp(x - gmax[:, None]) * inv
        tl.store(out_ptr + rows[:, None] * O_S0 + cols[None, :] * O_S1, y)
    else:
        rsafe = tl.where(rmask, rows, B - 1)
        cols = cb * BLOCK_V + tl.arange(0, BLOCK_V)
        cm = cols < V
        lmask = rmask[:, None] & cm[None, :]
        x = tl.load(
            logits_ptr + rsafe[:, None] * L_S0
            + tl.where(cm, cols, V - 1)[None, :] * L_S1
        )
        if HAS_TEMP:
            inv_t = (1.0 / tl.load(temp_ptr + rows, mask=rmask, other=1.0))[:, None]
            x = x * inv_t
        y = tl.exp(x - gmax[:, None]) * inv
        tl.store(out_ptr + rows[:, None] * O_S0 + cols[None, :] * O_S1,
                 y, mask=lmask)


_GRID_X_MAX = 65535


def wide_row_softmax(logits, temperatures, out):
    B = int(logits.shape[0])
    V = int(logits.shape[1])
    dev = logits.device
    if B == 0 or V == 0:
        return out
    has_temp = temperatures is not None
    if has_temp:
        temp = temperatures.to(torch.float32)
        if not temp.is_contiguous():
            temp = temp.contiguous()
        temp = temp.reshape(-1)
    else:
        temp = None
    L_S0 = int(logits.stride(0))
    L_S1 = int(logits.stride(1))
    O_S0 = int(out.stride(0))
    O_S1 = int(out.stride(1))

    if B == 1:
        if V >= 65536:
            block_v, k, nw = 2048, 2, 1
        else:
            block_v = min(1024, max(1, triton.next_power_of_2(V)))
            k = 1
            nw = 4 if block_v >= 512 else 1
        C = int(triton.cdiv(V, block_v))
        block_c = int(triton.next_power_of_2(max(C, 1)))
        exact = V % block_v == 0
        pmax = torch.empty((B, C), device=dev, dtype=torch.float32)
        psum = torch.empty((B, C), device=dev, dtype=torch.float32)
        if temp is None:
            temp = pmax  # dummy, never dereferenced
        nck = (C + k - 1) // k
        grid = (B * nck,)
        _wsr_partial_1d[grid](
            logits,
            temp,
            pmax,
            psum,
            L_S0=L_S0,
            L_S1=L_S1,
            V=V,
            C=C,
            HAS_TEMP=has_temp,
            BLOCK_V=block_v,
            K=k,
            EXACT=exact,
            num_warps=nw,
        )
        _wsr_norm_1d[grid](
            logits,
            temp,
            pmax,
            psum,
            out,
            L_S0=L_S0,
            L_S1=L_S1,
            O_S0=O_S0,
            O_S1=O_S1,
            V=V,
            C=C,
            BLOCK_C=block_c,
            HAS_TEMP=has_temp,
            BLOCK_V=block_v,
            K=k,
            EXACT=exact,
            num_warps=nw,
        )
        return out

    # B >= 2: 2D row-batch tiles.
    # r7: multi-program shapes (enough programs to keep the dispatch cost
    # amortised) double the column tile to cut the program count in half.
    if B >= 4 and V >= 8192 and (((B + 3) // 4) * triton.cdiv(V, 8192)) >= 16:
        block_v = 8192
    else:
        block_v = 4096 if V >= 65536 else 2048
    block_v = min(block_v, max(1, triton.next_power_of_2(V)))
    C = int(triton.cdiv(V, block_v))
    block_c = int(triton.next_power_of_2(max(C, 1)))
    R = min(triton.next_power_of_2(B), 4)
    while ((B + R - 1) // R) * C > _GRID_X_MAX and R < 8:
        R *= 2
    grid = (((B + R - 1) // R) * C,)
    if grid[0] > _GRID_X_MAX:
        # Would need B*V >= 2**31 fp32 elements (~8 GiB of logits); the r5
        # form failed earlier (uncapped B*C grid). Fail loudly instead.
        raise RuntimeError(
            f"wide_row_softmax grid {grid[0]} exceeds GCU grid.x limit "
            f"{_GRID_X_MAX} (B={B}, V={V})"
        )
    exact = (B % R == 0) and (V % block_v == 0)
    pmax = torch.empty((B, C), device=dev, dtype=torch.float32)
    psum = torch.empty((B, C), device=dev, dtype=torch.float32)
    if temp is None:
        temp = pmax  # dummy, never dereferenced
    _wsr_partial_2d[grid](
        logits,
        temp,
        pmax,
        psum,
        L_S0=L_S0,
        L_S1=L_S1,
        B=B,
        V=V,
        C=C,
        HAS_TEMP=has_temp,
        BLOCK_V=block_v,
        R=R,
        EXACT=exact,
        num_warps=2,
    )
    _wsr_norm_2d[grid](
        logits,
        temp,
        pmax,
        psum,
        out,
        L_S0=L_S0,
        L_S1=L_S1,
        O_S0=O_S0,
        O_S1=O_S1,
        B=B,
        V=V,
        C=C,
        BLOCK_C=block_c,
        HAS_TEMP=has_temp,
        BLOCK_V=block_v,
        R=R,
        EXACT=exact,
        num_warps=2,
    )
    return out


__all__ = ["wide_row_softmax"]
