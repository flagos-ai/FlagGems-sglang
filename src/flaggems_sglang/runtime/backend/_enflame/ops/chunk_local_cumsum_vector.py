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
# implied. See the License for the specific language
# governing permissions and limitations under the License.

"""chunk_local_cumsum_vector — vector-mode local cumulative sum within chunks.

Splits the input `g` of shape [B, T, H, S] into chunks of `chunk_size` along the
time axis (T must be divisible by chunk_size), then does a cumulative sum along
the time-within-chunk axis for every (head, vector) column. Supports a reversed
cumsum and an optional scalar scale.

Semantics (matches the PyTorch reference exactly):

    B, T, H, S = g.shape
    BT = chunk_size;  NT = T // BT
    g_c = g.float().view(B, NT, BT, H, S)
    if reverse: g_c = g_c.flip(2)
    out = g_c.cumsum(dim=2)
    if scale is not None: out = out * scale
    if reverse: out = out.flip(2)
    return out.reshape(B, T, H, S)

The output is always float32 (the reference never casts back to the input
dtype). This kernel is written in portable Triton only and never calls any
pre-compiled / vendor-specific cached operator.

v9 design (2026-11-08).  Three parts:

1. Triangular-dot main path.  The within-chunk scan is computed as a single
   ``tl.dot`` of a loop-invariant triangular mask with the loaded tile —
   the delivered form of the vllm sibling kernel
   (``flaggems_vllm/ops/FLA/cumsum.py::chunk_local_cumsum_vector_kernel``):
   on the GCU backend the ``tl.cumsum`` intrinsic is an order of magnitude
   slower than the surrounding memory traffic, while the masked matmul runs
   at the load+store floor.  The mask is inclusive-lower-triangular for the
   forward direction and inclusive-upper-triangular for reverse, so both
   directions share one canonical addressing pattern (no reversed rows, no
   flip arithmetic).  Operands stay in the input dtype (bf16/fp16 products
   are exact in the fp32 accumulator; for fp32 inputs the stripped-input
   matmul measured IEEE-level accuracy, max abs err 3.8e-6 on the probe
   shapes).  MMA bounds require BLOCK_BT >= 16, BLOCK_HS >= 16 and keep
   BLOCK_BT <= 128 (the [BT, BT] mask stays <= 32K elements); the per-program
   tile is capped at [BLOCK_BT, 512] (bf16 tile + fp32 accumulator + mask
   <= ~576 KB DSM at the largest BLOCK_BT = 128, under the 917504 B limit).
   Measured on the GCU backend (kernel wall): b8-t1024-h8-s64-c64 forward
   20.2 us / reverse 22.0 us; b32-t256-h8-s128-c64 (two [64, 512] column
   blocks) 32.0 / 31.9 us — with num_warps=8, NCHUNK=4 chunks per program.

2. Toolchain-compat shim (2026-11-07 20:22 build skew).  First compile of
   every canonical 2D tile load (``memref_ext.slice_*``) and every
   ``tl.dot`` (``gcu.matmul`` inputPrecision / operandSegmentSizes) fails on
   this build: the in-process GCUIR generator emits new-format op encodings
   that the older external ``gcu-compiler-opt`` rejects.  A scoped retry
   shim — the delivered vllm pattern (see
   compiler_issues/vllm_parallel_nsa_gcu_matmul.md and
   compiler_issues/sglang_chunk_local_cumsum_vector.md) — rewrites those
   encodings back to the old form and retries.  It is fallback-only (the
   original compile is attempted first) and scoped to this module's kernel
   names, so already-compilable kernels are unaffected.  Remove once the
   toolchain is re-synced.

3. Wave-aware NCHUNK dispatch (2026-11-12 re-tune).  With the toolchain
   re-synced (the shim below stays disabled), the single-wave grid optimum
   moved: one program per chunk-group per 512-wide column block, with the
   grid held inside one 48-program wave, beats the old fixed NCHUNK=4 cap
   whenever NCOL >= 2 — the wide bench shape [32, 256, 8, 128] drops from
   34.4 to 30.4 us (NCHUNK 4 -> 8, grid 64 -> 32), sitting on its copy
   floor.  Deeper loops and multi-wave grids both lose (see _pick_nchunk).

Shapes outside the dot domain (chunk_size < 16 or > 128, H*S < 16) take a
per-row 1D recurrence fallback (static_range row loop over [BLOCK_HS]
vectors; scan direction = loop order), which compiles without the shim on
every configuration probed.
"""

import re

import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# Toolchain-compat shim (2026-11-07 20:22 skew): retry-path GCUIR downgrade
# scoped to this module's kernels. See compiler_issues/
# vllm_parallel_nsa_gcu_matmul.md for the full analysis.
# ---------------------------------------------------------------------------
_COMPAT_SCOPED_KERNELS = (
    "_chunk_local_cumsum_vector_kernel",
    "_chunk_local_cumsum_rowloop_kernel",
)


def _downgrade_gcuir_for_external_tool(text):
    """Rewrite new-format GCUIR op encodings back to the old external-tool form."""

    def _strip_trailing_zero_segments(match):
        entries = [entry.strip() for entry in match.group(2).split(",")]
        while len(entries) > 1 and entries[-1] == "0":
            entries.pop()
        return match.group(1) + ", ".join(entries) + match.group(3)

    out_lines = []
    for line in text.splitlines():
        if '"memref_ext.slice_' in line or '"gcu.matmul"' in line:
            if "operandSegmentSizes = array<i32:" in line:
                line = re.sub(
                    r"(operandSegmentSizes = array<i32: )"
                    r"((?:\s*\d+\s*,\s*)*\s*\d+\s*)(>)",
                    _strip_trailing_zero_segments,
                    line,
                )
            if '"gcu.matmul"' in line:
                # Legacy gcu.matmul keeps one empty variadic segment entry.
                line = re.sub(
                    r"operandSegmentSizes = array<i32: (\d+(?:, \d+){3})>",
                    r"operandSegmentSizes = array<i32: \1, 0>",
                    line,
                )
                line = re.sub(r"inputPrecision = \d+ : i32, ", "", line)
                line = re.sub(r", inputPrecision = \d+ : i32", "", line)
        if '"memref_ext.memset_start"' in line:
            line = re.sub(
                r" <\{operandSegmentSizes = array<i32: [0-9, ]+>\}>", "", line
            )
        out_lines.append(line)
    return "\n".join(out_lines) + "\n"


def _install_gcuir_text_compat():
    """Wrap enflame toolkit.gcu_compiler_opt with a scoped downgrade retry."""
    try:
        from triton.backends.enflame import toolkit
    except ImportError:
        return
    if getattr(toolkit.gcu_compiler_opt, "_cumsum_vector_gcu_compat", False):
        return
    original = toolkit.gcu_compiler_opt

    def gcu_compiler_opt_compat(content, *args):
        try:
            return original(content, *args)
        except Exception as first_error:
            text = content if isinstance(content, str) else str(content)
            if not any(name in text for name in _COMPAT_SCOPED_KERNELS):
                raise first_error
            try:
                return toolkit._run_command(
                    "gcu-compiler-opt",
                    _downgrade_gcuir_for_external_tool(text),
                    "-mlir-print-op-generic",
                    *args,
                )
            except Exception:
                raise first_error

    gcu_compiler_opt_compat._cumsum_vector_gcu_compat = True
    toolkit.gcu_compiler_opt = gcu_compiler_opt_compat


# _install_gcuir_text_compat()  # DISABLED 2026-11-09: 私有 GCUIR shim 实测引入回归(降级重试路径
# 搞挂合法 kernel,同 conftest shim 事故);详见 commit b333be6/e1acdf6 与
# compiler_issues/vllm_memref_slice_pad_start.md。编译器修复后统一处置。


# ---------------------------------------------------------------------------
# Tunables.
# ---------------------------------------------------------------------------
# Dot-path launch config (measured on the GCU backend; nw=8 won every probed
# tile and stages are flat for NCHUNK >= 2).
NUM_WARPS_DOT = 8
NUM_STAGES_DOT = 3
# Wave-aware chunk-loop dispatch (measured 2026-11-12 on GCU400 / 48 SIPs):
# latency is minimized by a single-wave grid (<= 48 programs) with the
# program count as large as the wave allows.  Grids past one wave lose
# (64-256 programs: +8-15% on the official shapes and synthetics), and too
# few resident programs lose harder (grid <= 16: +30-300%).  Chunk loops
# deeper than ~32 always lose (NCHUNK=97 on prime nchunks measured 10x), so
# shapes whose single-wave grid needs a deeper loop fall back to a
# 128-program multi-wave cap before giving up on loop depth.
GRID_WAVE = 48
GRID_MULTIWAVE = 128
NCHUNK_WAVE_MAX = 16
NCHUNK_MAX = 32
# Dot-path geometry: column blocks at most 512 wide (two [64, 512] blocks
# beat one [64, 1024] tile by ~22% on the wide bench shape), chunk (BT) side
# bounded by the MMA constraints and the [BT, BT] mask DSM budget.
BLOCK_HS_MAX = 512
BLOCK_BT_MIN = 16
BLOCK_BT_MAX = 128
BLOCK_HS_MIN = 16
# Fallback launch config (per-row recurrence, [BLOCK_HS] 1D row vectors).
LOOP_BLOCK_HS_MAX = 512
NUM_WARPS_LOOP = 1


def _next_pow2(n):
    p = 1
    while p < n:
        p *= 2
    return p


def _largest_pow2_le(n):
    p = 1
    while p * 2 <= n:
        p *= 2
    return p


def _pick_nchunk(nchunks, ncol):
    """Smallest chunk loop at the measured latency optimum.

    Tier 1 keeps the grid within one 48-program wave (loop <= 16 chunks);
    tier 2 allows a 128-program multi-wave grid (loop <= 32); tier 3 takes
    the shallowest divisor-based loop available.  NCHUNK always divides
    nchunks, so the grid covers the chunks exactly (no guard, no OOB).
    """
    for grid_cap, nchunk_cap in (
        (GRID_WAVE, NCHUNK_WAVE_MAX),
        (GRID_MULTIWAVE, NCHUNK_MAX),
    ):
        for nprog in range(min(max(grid_cap // ncol, 1), nchunks), 0, -1):
            if nchunks % nprog == 0:
                nchunk = nchunks // nprog
                if nchunk <= nchunk_cap:
                    return nchunk
                break
    for nchunk in range(NCHUNK_MAX, 0, -1):
        if nchunks % nchunk == 0:
            return nchunk
    return 1


@triton.jit
def _chunk_local_cumsum_vector_kernel(
    g_ptr,
    out_ptr,
    scale,
    BT: tl.constexpr,
    HS: tl.constexpr,
    NCOL: tl.constexpr,
    NCHUNK: tl.constexpr,
    BLOCK_BT: tl.constexpr,
    BLOCK_HS: tl.constexpr,
    HAS_SCALE: tl.constexpr,
    IS_REVERSE: tl.constexpr,
    HAS_ROW_MASK: tl.constexpr,
    HAS_COL_MASK: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    # Triangular-dot scan: one loop-inclusive triangular mask per program,
    # applied to every chunk tile the program owns.  The mask is lower
    # triangular for the forward direction (inclusive prefix sums) and upper
    # triangular for reverse (inclusive suffix sums), so both directions use
    # the same canonical ascending addresses.  Operands stay in the input
    # dtype; the fp32 accumulator keeps bf16/fp16 products exact.
    pid = tl.program_id(0)
    cb = pid % NCOL
    rem = pid // NCOL
    chunk0 = rem * NCHUNK

    rows = tl.arange(0, BLOCK_BT)
    cols = tl.arange(0, BLOCK_HS)
    o_i = tl.arange(0, BLOCK_BT)
    if IS_REVERSE:
        m = tl.where(o_i[:, None] <= o_i[None, :], 1.0, 0.0)
    else:
        m = tl.where(o_i[:, None] >= o_i[None, :], 1.0, 0.0)
    mb = m.to(g_ptr.dtype.element_ty)
    off_cols = cb * BLOCK_HS + tl.max_contiguous(
        tl.multiple_of(cols, BLOCK_HS), BLOCK_HS
    )
    row_off = rows[:, None] * HS + off_cols[None, :]
    base = (chunk0 * BT) * HS

    if HAS_ROW_MASK and HAS_COL_MASK:
        ncols = HS - cb * BLOCK_HS
        m2 = (rows[:, None] < BT) & (cols[None, :] < ncols)
        for k in tl.range(0, NCHUNK, num_stages=NUM_STAGES):
            off = base + row_off
            x = tl.load(g_ptr + off, mask=m2, other=0.0)
            y = tl.dot(mb, x)
            if HAS_SCALE:
                y = y * scale
            tl.store(out_ptr + off, y, mask=m2)
            base += BT * HS
    elif HAS_ROW_MASK:
        mr = rows[:, None] < BT
        for k in tl.range(0, NCHUNK, num_stages=NUM_STAGES):
            off = base + row_off
            x = tl.load(g_ptr + off, mask=mr, other=0.0)
            y = tl.dot(mb, x)
            if HAS_SCALE:
                y = y * scale
            tl.store(out_ptr + off, y, mask=mr)
            base += BT * HS
    elif HAS_COL_MASK:
        ncols = HS - cb * BLOCK_HS
        mc = cols[None, :] < ncols
        for k in tl.range(0, NCHUNK, num_stages=NUM_STAGES):
            off = base + row_off
            x = tl.load(g_ptr + off, mask=mc, other=0.0)
            y = tl.dot(mb, x)
            if HAS_SCALE:
                y = y * scale
            tl.store(out_ptr + off, y, mask=mc)
            base += BT * HS
    else:
        for k in tl.range(0, NCHUNK, num_stages=NUM_STAGES):
            off = base + row_off
            x = tl.load(g_ptr + off)
            y = tl.dot(mb, x)
            if HAS_SCALE:
                y = y * scale
            tl.store(out_ptr + off, y)
            base += BT * HS


@triton.jit
def _chunk_local_cumsum_rowloop_kernel(
    g_ptr,
    out_ptr,
    scale,
    BT: tl.constexpr,
    HS: tl.constexpr,
    NCOL: tl.constexpr,
    BLOCK_HS: tl.constexpr,
    IS_REVERSE: tl.constexpr,
    HAS_SCALE: tl.constexpr,
    HAS_COL_MASK: tl.constexpr,
):
    # Fallback for shapes outside the dot domain (BT < 16 or > 128, HS < 16):
    # per-row 1D recurrence.  One program owns one (chunk, column block);
    # the BT rows stream through a running fp32 accumulator, storing each
    # partial sum.  The scan direction is the loop order.
    pid = tl.program_id(0)
    cb = pid % NCOL
    chunk = pid // NCOL

    cols = tl.arange(0, BLOCK_HS)
    off_cols = cb * BLOCK_HS + tl.max_contiguous(
        tl.multiple_of(cols, BLOCK_HS), BLOCK_HS
    )
    if HAS_COL_MASK:
        cmask = cols < HS - cb * BLOCK_HS

    base = (chunk * BT) * HS
    acc = tl.zeros((BLOCK_HS,), dtype=tl.float32)
    if IS_REVERSE:
        for r in tl.static_range(BT - 1, -1, -1):
            if HAS_COL_MASK:
                x = tl.load(
                    g_ptr + base + r * HS + off_cols, mask=cmask, other=0.0
                ).to(tl.float32)
            else:
                x = tl.load(g_ptr + base + r * HS + off_cols).to(tl.float32)
            acc += x
            if HAS_SCALE:
                y = acc * scale
            else:
                y = acc
            if HAS_COL_MASK:
                tl.store(out_ptr + base + r * HS + off_cols, y, mask=cmask)
            else:
                tl.store(out_ptr + base + r * HS + off_cols, y)
    else:
        for r in tl.static_range(0, BT):
            if HAS_COL_MASK:
                x = tl.load(
                    g_ptr + base + r * HS + off_cols, mask=cmask, other=0.0
                ).to(tl.float32)
            else:
                x = tl.load(g_ptr + base + r * HS + off_cols).to(tl.float32)
            acc += x
            if HAS_SCALE:
                y = acc * scale
            else:
                y = acc
            if HAS_COL_MASK:
                tl.store(out_ptr + base + r * HS + off_cols, y, mask=cmask)
            else:
                tl.store(out_ptr + base + r * HS + off_cols, y)


def chunk_local_cumsum_vector(g, chunk_size, reverse=False, scale=None):
    """Vector-mode local cumulative sum within chunks (Triton implementation).

    Args:
        g:           [B, T, H, S] input (float32 / bfloat16 / float16).
        chunk_size:  within-chunk time width; T must be a multiple of it.
        reverse:     if True, cumsum runs from the end of each chunk.
        scale:       optional scalar multiplier applied after the cumsum.

    Returns:
        out: [B, T, H, S] float32.
    """
    B, T, H, S = g.shape
    BT = chunk_size
    HS = H * S
    nchunks = B * (T // BT)
    out = torch.empty_like(g, dtype=torch.float32)

    scale_f = float(scale) if scale is not None else 1.0
    has_scale = scale is not None
    is_reverse = bool(reverse)

    BLOCK_BT = _next_pow2(BT)
    if BLOCK_BT_MIN <= BLOCK_BT <= BLOCK_BT_MAX:
        # Dot path.  BLOCK_HS covers all pow2 HS up to 512 in one block;
        # wider or non-pow2 HS splits into column blocks (masked tail).
        BLOCK_HS = min(_largest_pow2_le(HS), BLOCK_HS_MAX)
        if BLOCK_HS >= BLOCK_HS_MIN:
            NCOL = (HS + BLOCK_HS - 1) // BLOCK_HS
            NCHUNK = _pick_nchunk(nchunks, NCOL)
            grid = ((nchunks + NCHUNK - 1) // NCHUNK * NCOL,)
            _chunk_local_cumsum_vector_kernel[grid](
                g,
                out,
                scale_f,
                BT=BT,
                HS=HS,
                NCOL=NCOL,
                NCHUNK=NCHUNK,
                BLOCK_BT=BLOCK_BT,
                BLOCK_HS=BLOCK_HS,
                HAS_SCALE=has_scale,
                IS_REVERSE=is_reverse,
                HAS_ROW_MASK=(BT != BLOCK_BT),
                HAS_COL_MASK=(HS % BLOCK_HS != 0),
                NUM_STAGES=NUM_STAGES_DOT if NCHUNK >= 2 else 1,
                num_warps=NUM_WARPS_DOT,
            )
            return out

    # Fallback: per-row recurrence over [BLOCK_HS] column blocks.
    BLOCK_HS_L = min(_largest_pow2_le(HS), LOOP_BLOCK_HS_MAX)
    NCOL = (HS + BLOCK_HS_L - 1) // BLOCK_HS_L
    grid = (nchunks * NCOL,)
    _chunk_local_cumsum_rowloop_kernel[grid](
        g,
        out,
        scale_f,
        BT=BT,
        HS=HS,
        NCOL=NCOL,
        BLOCK_HS=BLOCK_HS_L,
        IS_REVERSE=is_reverse,
        HAS_SCALE=has_scale,
        HAS_COL_MASK=(HS % BLOCK_HS_L != 0),
        num_warps=NUM_WARPS_LOOP,
    )
    return out


__all__ = ["chunk_local_cumsum_vector"]
