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

"""DeepGEMM MoE post-reorder epilogue -- Enflame GCU specialization.

v3 (2026-11-13, E-1 second round): hidden >= 4096 with topk == 8 now runs
a wide-column row kernel (`_post_reorder_kernel_row4096`): one program
per token, BLOCK_H=4096 unmasked chunks (NF = hidden // 4096) plus a
single masked 4096 tail chunk, all 24 slot scalars hoisted as independent
loads, and a SCALE_ONE constexpr arm for the routing factor. Probe matrix
(kernel-direct do_bench, w1000/r100 median, GCU400, H=7168, topk=8, bf16):

- T=1/8/64/512/2048: 5.30/5.70/12.18/52.94/191.02 us (nw4, TAIL form)
  vs v2 direct/row/persist 5.26/9.31/32.30/175.52/676.44 us.
- num_warps: 4 wins on the masked-tail form (7168/6144/5120), 2 wins on
  the exact-divisible form (4096/8192); host picks by TAIL.
- Guard shapes: H=4096 T=1/64/512 = 4.20/8.94/40.04 (nw2) vs v2 wrap
  5.40/12.24/61.36; H=6144/8192 T=512 = 51.39/54.60 vs 144.83/205.14.
- Rejected with numbers (see job j20261113174112 probe logs): ep_gather
  PAIR pair4k_nw2 T=512 87.96 us (loses 66% at H=7168; pair's win is
  H=4096-specific); single masked 8192 tile 66.71 us (249.37 at T=2048);
  persistent pair 60.91 us; column-split pair 113.90 us.
- Narrow hidden (< 4096) keeps the v2 dispatch: at H=3072/T=512 the v2
  persist48@1024 form measured 33.69 us vs 41.27 (2048+tail row) -- the
  wide-tail row form is not universally better, so only the measured
  winning region (H >= 4096) switches over.

v2 (2026-11-08): first GCU rewrite of the generic autotuned 2D-tile kernel
(``down_ptr + row_base[:, None] + offs_h[None, :]`` gather). On GCU that
vector-indexed row gather is the slow form (knowledge/perf_playbook.md:
3-30x slower than scalar-base row loads), and after the 11-07 20:22
toolchain regression the 2D masked tile face no longer fresh-compiles at
all (compiler_issues/sglang_2d_tile_fresh_compile_slice_pad_start.md) --
the recorded baseline numbers were pre-regression cache hits.

Structure follows the sibling deepep_post_reorder v10 template (measured
same-day on this stack, kernel-direct do_bench, H=7168, topk=8):
- 1D loads/stores only, fully unmasked column tiles walked with
  tl.static_range -- the compile-gate-proven form.
- Slot ids/src2dst/weight are read with independent scalar loads (no
  vector-load + tl.sum extraction chain), so the row gathers behind
  them issue at full memory-level parallelism.
- Narrow-hidden (< 4096) topk == 8: T < _ROW_MIN_TOKENS direct-mapped
  grid = num_tokens * NB; whole-row kernel beyond; persistent whole-row
  kernel (fixed 48 programs) past the per-tile-count crossover.
- Validity is ids >= 0 (not src2dst >= 0): invalid slots read row 0 and
  contribute an exact zero weight, matching the generic masked-load
  semantics; scaling stays applied once after the fp32 accumulation
  (SCALE_ONE skips the exact-identity mul when scaling == 1.0).
- Non-divisible / >8192 hidden sizes, non-power-of-two topk, int64-range
  tensors (WIDE) or grids past the 65535 program cap fall back to a
  persistent masked BLOCK_H=1024 chunk walk (masked 1D tiles at 1024
  stay outside the regression's trigger face).
"""

import triton
import triton.language as tl

# Column tile for the narrow-hidden fast path and the masked fallback.
# Production hidden sizes (2048/4096/7168) are all multiples of 1024; the
# masked fallback keeps the same tile width.
_BLOCK_H = 1024

# Hidden size at (and above) which the wide-column row kernel
# (_post_reorder_kernel_row4096) takes over for topk == 8.
_ROW_WIDE_H = 4096

# Column tile for the wide-hidden row kernel (cutlass-v2 / E-1 probe form).
_WIDE_BLOCK_H = 4096

# Minimum token count before the whole-row kernel takes over from the
# direct-mapped (token x column-tile) kernel. Crossover measured on the
# sibling deepep op on this stack (H=7168, topk=8): T=8 direct 12.5us
# vs row 16.4us; T=16 direct 18.3us vs row 16.0us.
_ROW_MIN_TOKENS = 16

# Upper bound on concurrently resident programs for the fallback kernel.
_MAX_PROGRAMS = 96

# Max hidden size for the unmasked fast path; wider or non-divisible
# hidden sizes take the masked chunk-loop fallback.
_MAX_BLOCK_H = 8192

# GCU grid.x hardware cap (compiler_issues/grid_hard_limits.md).
_GRID_X_LIMIT = 65535

# Persistent whole-row kernel: fixed program count and the token counts
# past which it beats the one-program-per-token form (measured on GCU400,
# kernel-direct do_bench; NB=4 / hidden 4096 always prefers the row
# kernel and never takes the persistent path).
_PERSIST_PROGRAMS = 48
_PERSIST_MIN_TOKENS_WIDE = 96  # NB >= 5 (hidden >= 5120)
_PERSIST_MIN_TOKENS_NARROW = 128  # NB <= 3 (hidden <= 3072)


@triton.jit
def _post_reorder_kernel(
    down_ptr,  # [num_tokens * topk, hidden_size] expert outputs
    out_ptr,  # [num_tokens, hidden_size] merged output
    src2dst_ptr,  # [num_tokens, topk] int32 destination rows
    ids_ptr,  # [num_tokens, topk] int32 expert ids, -1 = padding
    w_ptr,  # [num_tokens, topk] float32 routing weights
    scaling,
    stride_down_m,
    stride_out_m,
    stride_s2d_t,
    stride_ids_t,
    stride_w_t,
    TOPK: tl.constexpr,
    BLOCK_H: tl.constexpr,
    NB: tl.constexpr,
):
    # Direct-mapped grid: program pid owns (token t, column tile hb).
    pid = tl.program_id(0)
    t = pid // NB
    hb = pid % NB
    offs = hb * BLOCK_H + tl.arange(0, BLOCK_H)
    tbase_s = t * stride_s2d_t
    tbase_i = t * stride_ids_t
    tbase_w = t * stride_w_t
    acc = tl.zeros([BLOCK_H], dtype=tl.float32)
    for i in tl.static_range(TOPK):
        # Scalar slot loads: independent, so the row gathers behind them
        # issue at full memory-level parallelism.
        id_i = tl.load(ids_ptr + tbase_i + i)
        dst_i = tl.load(src2dst_ptr + tbase_s + i)
        w_i = tl.load(w_ptr + tbase_w + i)
        valid = id_i >= 0
        # Invalid slots read row 0 and contribute exactly zero, matching
        # the generic masked-load semantics.
        d = tl.where(valid, dst_i, 0)
        w_f = tl.where(valid, w_i, 0.0)
        row = tl.load(down_ptr + d * stride_down_m + offs)
        acc += row.to(tl.float32) * w_f
    tl.store(
        out_ptr + t * stride_out_m + offs,
        (acc * scaling).to(out_ptr.dtype.element_ty),
    )


@triton.jit
def _post_reorder_kernel_padded(
    down_ptr,
    out_ptr,
    src2dst_ptr,
    ids_ptr,
    w_ptr,
    topk,
    scaling,
    stride_down_m,
    stride_out_m,
    stride_s2d_t,
    stride_ids_t,
    stride_w_t,
    TOPK: tl.constexpr,
    BLOCK_H: tl.constexpr,
    NB: tl.constexpr,
):
    """Fallback for topk values that are not powers of two.

    Same direct-mapped body as _post_reorder_kernel, but lane i >= topk
    re-reads slot topk-1 (clamped scalar read) and is nullified by a
    zero weight plus a failed validity check, so every scalar load stays
    unmasked.
    """
    pid = tl.program_id(0)
    t = pid // NB
    hb = pid % NB
    offs = hb * BLOCK_H + tl.arange(0, BLOCK_H)
    tbase_s = t * stride_s2d_t
    tbase_i = t * stride_ids_t
    tbase_w = t * stride_w_t
    acc = tl.zeros([BLOCK_H], dtype=tl.float32)
    for i in tl.static_range(TOPK):
        ic = tl.minimum(i, topk - 1)
        id_i = tl.load(ids_ptr + tbase_i + ic)
        dst_i = tl.load(src2dst_ptr + tbase_s + ic)
        w_i = tl.load(w_ptr + tbase_w + ic)
        valid = (i < topk) & (id_i >= 0)
        d = tl.where(valid, dst_i, 0)
        w_f = tl.where(valid, w_i, 0.0)
        row = tl.load(down_ptr + d * stride_down_m + offs)
        acc += row.to(tl.float32) * w_f
    tl.store(
        out_ptr + t * stride_out_m + offs,
        (acc * scaling).to(out_ptr.dtype.element_ty),
    )


@triton.jit
def _post_reorder_kernel_row(
    down_ptr,
    out_ptr,
    src2dst_ptr,
    ids_ptr,
    w_ptr,
    scaling,
    stride_down_m,
    stride_out_m,
    stride_s2d_t,
    stride_ids_t,
    stride_w_t,
    BLOCK_H: tl.constexpr,
    NB: tl.constexpr,
):
    """Whole-row kernel for topk == 8, T >= _ROW_MIN_TOKENS.

    One program per token: all slot ids/dst/weight scalars are read once
    up front (independent scalar loads, full memory-level parallelism),
    then the hidden dimension is walked in NB unmasked column tiles with
    a balanced fp32 tree per tile. scaling is applied once after the
    accumulation, exactly like the generic kernel.
    """
    t = tl.program_id(0)
    tbase_s = t * stride_s2d_t
    tbase_i = t * stride_ids_t
    tbase_w = t * stride_w_t
    out_base = t * stride_out_m
    id0 = tl.load(ids_ptr + tbase_i + 0)
    id1 = tl.load(ids_ptr + tbase_i + 1)
    id2 = tl.load(ids_ptr + tbase_i + 2)
    id3 = tl.load(ids_ptr + tbase_i + 3)
    id4 = tl.load(ids_ptr + tbase_i + 4)
    id5 = tl.load(ids_ptr + tbase_i + 5)
    id6 = tl.load(ids_ptr + tbase_i + 6)
    id7 = tl.load(ids_ptr + tbase_i + 7)
    dst0 = tl.load(src2dst_ptr + tbase_s + 0)
    dst1 = tl.load(src2dst_ptr + tbase_s + 1)
    dst2 = tl.load(src2dst_ptr + tbase_s + 2)
    dst3 = tl.load(src2dst_ptr + tbase_s + 3)
    dst4 = tl.load(src2dst_ptr + tbase_s + 4)
    dst5 = tl.load(src2dst_ptr + tbase_s + 5)
    dst6 = tl.load(src2dst_ptr + tbase_s + 6)
    dst7 = tl.load(src2dst_ptr + tbase_s + 7)
    w0 = tl.load(w_ptr + tbase_w + 0)
    w1 = tl.load(w_ptr + tbase_w + 1)
    w2 = tl.load(w_ptr + tbase_w + 2)
    w3 = tl.load(w_ptr + tbase_w + 3)
    w4 = tl.load(w_ptr + tbase_w + 4)
    w5 = tl.load(w_ptr + tbase_w + 5)
    w6 = tl.load(w_ptr + tbase_w + 6)
    w7 = tl.load(w_ptr + tbase_w + 7)
    d0 = tl.where(id0 >= 0, dst0, 0) * stride_down_m
    d1 = tl.where(id1 >= 0, dst1, 0) * stride_down_m
    d2 = tl.where(id2 >= 0, dst2, 0) * stride_down_m
    d3 = tl.where(id3 >= 0, dst3, 0) * stride_down_m
    d4 = tl.where(id4 >= 0, dst4, 0) * stride_down_m
    d5 = tl.where(id5 >= 0, dst5, 0) * stride_down_m
    d6 = tl.where(id6 >= 0, dst6, 0) * stride_down_m
    d7 = tl.where(id7 >= 0, dst7, 0) * stride_down_m
    wv0 = tl.where(id0 >= 0, w0, 0.0)
    wv1 = tl.where(id1 >= 0, w1, 0.0)
    wv2 = tl.where(id2 >= 0, w2, 0.0)
    wv3 = tl.where(id3 >= 0, w3, 0.0)
    wv4 = tl.where(id4 >= 0, w4, 0.0)
    wv5 = tl.where(id5 >= 0, w5, 0.0)
    wv6 = tl.where(id6 >= 0, w6, 0.0)
    wv7 = tl.where(id7 >= 0, w7, 0.0)
    for c in tl.static_range(NB):
        offs = c * BLOCK_H + tl.arange(0, BLOCK_H)
        r0 = tl.load(down_ptr + d0 + offs)
        r1 = tl.load(down_ptr + d1 + offs)
        r2 = tl.load(down_ptr + d2 + offs)
        r3 = tl.load(down_ptr + d3 + offs)
        r4 = tl.load(down_ptr + d4 + offs)
        r5 = tl.load(down_ptr + d5 + offs)
        r6 = tl.load(down_ptr + d6 + offs)
        r7 = tl.load(down_ptr + d7 + offs)
        acc = (
            (r0.to(tl.float32) * wv0 + r1.to(tl.float32) * wv1)
            + (r2.to(tl.float32) * wv2 + r3.to(tl.float32) * wv3)
        ) + (
            (r4.to(tl.float32) * wv4 + r5.to(tl.float32) * wv5)
            + (r6.to(tl.float32) * wv6 + r7.to(tl.float32) * wv7)
        )
        tl.store(
            out_ptr + out_base + offs,
            (acc * scaling).to(out_ptr.dtype.element_ty),
        )


@triton.jit
def _post_reorder_kernel_row_persist(
    down_ptr,
    out_ptr,
    src2dst_ptr,
    ids_ptr,
    w_ptr,
    num_tokens,
    scaling,
    stride_down_m,
    stride_out_m,
    stride_s2d_t,
    stride_ids_t,
    stride_w_t,
    BLOCK_H: tl.constexpr,
    NB: tl.constexpr,
):
    """Persistent whole-row kernel for topk == 8 at large token counts.

    Same per-token body as _post_reorder_kernel_row, walked by a fixed
    grid of 48 programs with a grid-stride token loop. Beyond the row
    kernel's measured crossover the CTA dispatch floor (~52ns/CTA on
    GCU400) outweighs the extra per-program serialization, and the fixed
    grid also sidesteps the 65535 grid.x cap for arbitrarily large
    num_tokens. Measured (kernel-direct do_bench, nw=2): NB=7 T=512
    177.3us vs row 198.7us (-10.8%), T=2048 680us vs 769us (-11.6%);
    NB=5/6/8 win from T>=256..512; NB=4 (hidden 4096) always prefers
    the row kernel, so it is excluded from this path.
    """
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    for t in range(pid, num_tokens, nprog):
        tbase_s = t * stride_s2d_t
        tbase_i = t * stride_ids_t
        tbase_w = t * stride_w_t
        out_base = t * stride_out_m
        id0 = tl.load(ids_ptr + tbase_i + 0)
        id1 = tl.load(ids_ptr + tbase_i + 1)
        id2 = tl.load(ids_ptr + tbase_i + 2)
        id3 = tl.load(ids_ptr + tbase_i + 3)
        id4 = tl.load(ids_ptr + tbase_i + 4)
        id5 = tl.load(ids_ptr + tbase_i + 5)
        id6 = tl.load(ids_ptr + tbase_i + 6)
        id7 = tl.load(ids_ptr + tbase_i + 7)
        dst0 = tl.load(src2dst_ptr + tbase_s + 0)
        dst1 = tl.load(src2dst_ptr + tbase_s + 1)
        dst2 = tl.load(src2dst_ptr + tbase_s + 2)
        dst3 = tl.load(src2dst_ptr + tbase_s + 3)
        dst4 = tl.load(src2dst_ptr + tbase_s + 4)
        dst5 = tl.load(src2dst_ptr + tbase_s + 5)
        dst6 = tl.load(src2dst_ptr + tbase_s + 6)
        dst7 = tl.load(src2dst_ptr + tbase_s + 7)
        w0 = tl.load(w_ptr + tbase_w + 0)
        w1 = tl.load(w_ptr + tbase_w + 1)
        w2 = tl.load(w_ptr + tbase_w + 2)
        w3 = tl.load(w_ptr + tbase_w + 3)
        w4 = tl.load(w_ptr + tbase_w + 4)
        w5 = tl.load(w_ptr + tbase_w + 5)
        w6 = tl.load(w_ptr + tbase_w + 6)
        w7 = tl.load(w_ptr + tbase_w + 7)
        d0 = tl.where(id0 >= 0, dst0, 0) * stride_down_m
        d1 = tl.where(id1 >= 0, dst1, 0) * stride_down_m
        d2 = tl.where(id2 >= 0, dst2, 0) * stride_down_m
        d3 = tl.where(id3 >= 0, dst3, 0) * stride_down_m
        d4 = tl.where(id4 >= 0, dst4, 0) * stride_down_m
        d5 = tl.where(id5 >= 0, dst5, 0) * stride_down_m
        d6 = tl.where(id6 >= 0, dst6, 0) * stride_down_m
        d7 = tl.where(id7 >= 0, dst7, 0) * stride_down_m
        wv0 = tl.where(id0 >= 0, w0, 0.0)
        wv1 = tl.where(id1 >= 0, w1, 0.0)
        wv2 = tl.where(id2 >= 0, w2, 0.0)
        wv3 = tl.where(id3 >= 0, w3, 0.0)
        wv4 = tl.where(id4 >= 0, w4, 0.0)
        wv5 = tl.where(id5 >= 0, w5, 0.0)
        wv6 = tl.where(id6 >= 0, w6, 0.0)
        wv7 = tl.where(id7 >= 0, w7, 0.0)
        for c in tl.static_range(NB):
            offs = c * BLOCK_H + tl.arange(0, BLOCK_H)
            r0 = tl.load(down_ptr + d0 + offs)
            r1 = tl.load(down_ptr + d1 + offs)
            r2 = tl.load(down_ptr + d2 + offs)
            r3 = tl.load(down_ptr + d3 + offs)
            r4 = tl.load(down_ptr + d4 + offs)
            r5 = tl.load(down_ptr + d5 + offs)
            r6 = tl.load(down_ptr + d6 + offs)
            r7 = tl.load(down_ptr + d7 + offs)
            acc = (
                (r0.to(tl.float32) * wv0 + r1.to(tl.float32) * wv1)
                + (r2.to(tl.float32) * wv2 + r3.to(tl.float32) * wv3)
            ) + (
                (r4.to(tl.float32) * wv4 + r5.to(tl.float32) * wv5)
                + (r6.to(tl.float32) * wv6 + r7.to(tl.float32) * wv7)
            )
            tl.store(
                out_ptr + out_base + offs,
                (acc * scaling).to(out_ptr.dtype.element_ty),
            )


@triton.jit
def _post_reorder_kernel_row4096(
    down_ptr,
    out_ptr,
    src2dst_ptr,
    ids_ptr,
    w_ptr,
    hidden_size,
    scaling,
    stride_down_m,
    stride_out_m,
    stride_s2d_t,
    stride_ids_t,
    stride_w_t,
    BLOCK_H: tl.constexpr,
    NF: tl.constexpr,
    TAIL: tl.constexpr,
    SCALE_ONE: tl.constexpr,
):
    """Wide-hidden topk == 8 kernel: one program per token.

    All 24 slot scalars are hoisted once as independent scalar loads
    (full memory-level parallelism), then the hidden dimension is walked
    as NF unmasked BLOCK_H chunks plus one masked tail chunk when
    hidden_size % BLOCK_H != 0 (a single wide masked tail beats several
    narrow unmasked tiles on this device -- post_reorder_cutlass v2
    precedent, re-measured here at H=7168/5120/6144). Accumulation keeps
    the v2 row-kernel fp32 4+4 balanced tree; SCALE_ONE skips the exact
    identity ``acc * 1.0`` multiply when the routing factor is 1.0
    (bitwise-preserving, moe_fused_mul_sum precedent). DSM peak ~96KB
    per program (8 bf16 tiles + fp32 acc + store) -- well under the
    448KB dual-program line, so 96-way concurrency is preserved.
    """
    t = tl.program_id(0)
    tbase_s = t * stride_s2d_t
    tbase_i = t * stride_ids_t
    tbase_w = t * stride_w_t
    out_base = t * stride_out_m
    id0 = tl.load(ids_ptr + tbase_i + 0)
    id1 = tl.load(ids_ptr + tbase_i + 1)
    id2 = tl.load(ids_ptr + tbase_i + 2)
    id3 = tl.load(ids_ptr + tbase_i + 3)
    id4 = tl.load(ids_ptr + tbase_i + 4)
    id5 = tl.load(ids_ptr + tbase_i + 5)
    id6 = tl.load(ids_ptr + tbase_i + 6)
    id7 = tl.load(ids_ptr + tbase_i + 7)
    dst0 = tl.load(src2dst_ptr + tbase_s + 0)
    dst1 = tl.load(src2dst_ptr + tbase_s + 1)
    dst2 = tl.load(src2dst_ptr + tbase_s + 2)
    dst3 = tl.load(src2dst_ptr + tbase_s + 3)
    dst4 = tl.load(src2dst_ptr + tbase_s + 4)
    dst5 = tl.load(src2dst_ptr + tbase_s + 5)
    dst6 = tl.load(src2dst_ptr + tbase_s + 6)
    dst7 = tl.load(src2dst_ptr + tbase_s + 7)
    w0 = tl.load(w_ptr + tbase_w + 0)
    w1 = tl.load(w_ptr + tbase_w + 1)
    w2 = tl.load(w_ptr + tbase_w + 2)
    w3 = tl.load(w_ptr + tbase_w + 3)
    w4 = tl.load(w_ptr + tbase_w + 4)
    w5 = tl.load(w_ptr + tbase_w + 5)
    w6 = tl.load(w_ptr + tbase_w + 6)
    w7 = tl.load(w_ptr + tbase_w + 7)
    d0 = tl.where(id0 >= 0, dst0, 0) * stride_down_m
    d1 = tl.where(id1 >= 0, dst1, 0) * stride_down_m
    d2 = tl.where(id2 >= 0, dst2, 0) * stride_down_m
    d3 = tl.where(id3 >= 0, dst3, 0) * stride_down_m
    d4 = tl.where(id4 >= 0, dst4, 0) * stride_down_m
    d5 = tl.where(id5 >= 0, dst5, 0) * stride_down_m
    d6 = tl.where(id6 >= 0, dst6, 0) * stride_down_m
    d7 = tl.where(id7 >= 0, dst7, 0) * stride_down_m
    wv0 = tl.where(id0 >= 0, w0, 0.0)
    wv1 = tl.where(id1 >= 0, w1, 0.0)
    wv2 = tl.where(id2 >= 0, w2, 0.0)
    wv3 = tl.where(id3 >= 0, w3, 0.0)
    wv4 = tl.where(id4 >= 0, w4, 0.0)
    wv5 = tl.where(id5 >= 0, w5, 0.0)
    wv6 = tl.where(id6 >= 0, w6, 0.0)
    wv7 = tl.where(id7 >= 0, w7, 0.0)
    for j in tl.static_range(NF):
        offs = j * BLOCK_H + tl.arange(0, BLOCK_H)
        r0 = tl.load(down_ptr + d0 + offs)
        r1 = tl.load(down_ptr + d1 + offs)
        r2 = tl.load(down_ptr + d2 + offs)
        r3 = tl.load(down_ptr + d3 + offs)
        r4 = tl.load(down_ptr + d4 + offs)
        r5 = tl.load(down_ptr + d5 + offs)
        r6 = tl.load(down_ptr + d6 + offs)
        r7 = tl.load(down_ptr + d7 + offs)
        acc = (
            (r0.to(tl.float32) * wv0 + r1.to(tl.float32) * wv1)
            + (r2.to(tl.float32) * wv2 + r3.to(tl.float32) * wv3)
        ) + (
            (r4.to(tl.float32) * wv4 + r5.to(tl.float32) * wv5)
            + (r6.to(tl.float32) * wv6 + r7.to(tl.float32) * wv7)
        )
        if SCALE_ONE:
            tl.store(
                out_ptr + out_base + offs, acc.to(out_ptr.dtype.element_ty)
            )
        else:
            tl.store(
                out_ptr + out_base + offs,
                (acc * scaling).to(out_ptr.dtype.element_ty),
            )
    if TAIL:
        offs = NF * BLOCK_H + tl.arange(0, BLOCK_H)
        mask = offs < hidden_size
        r0 = tl.load(down_ptr + d0 + offs, mask, other=0.0)
        r1 = tl.load(down_ptr + d1 + offs, mask, other=0.0)
        r2 = tl.load(down_ptr + d2 + offs, mask, other=0.0)
        r3 = tl.load(down_ptr + d3 + offs, mask, other=0.0)
        r4 = tl.load(down_ptr + d4 + offs, mask, other=0.0)
        r5 = tl.load(down_ptr + d5 + offs, mask, other=0.0)
        r6 = tl.load(down_ptr + d6 + offs, mask, other=0.0)
        r7 = tl.load(down_ptr + d7 + offs, mask, other=0.0)
        acc = (
            (r0.to(tl.float32) * wv0 + r1.to(tl.float32) * wv1)
            + (r2.to(tl.float32) * wv2 + r3.to(tl.float32) * wv3)
        ) + (
            (r4.to(tl.float32) * wv4 + r5.to(tl.float32) * wv5)
            + (r6.to(tl.float32) * wv6 + r7.to(tl.float32) * wv7)
        )
        if SCALE_ONE:
            tl.store(
                out_ptr + out_base + offs,
                acc.to(out_ptr.dtype.element_ty),
                mask,
            )
        else:
            tl.store(
                out_ptr + out_base + offs,
                (acc * scaling).to(out_ptr.dtype.element_ty),
                mask,
            )


@triton.jit
def _post_reorder_kernel_wide(
    down_ptr,
    out_ptr,
    src2dst_ptr,
    ids_ptr,
    w_ptr,
    hidden_size,
    num_tokens,
    topk,
    scaling,
    stride_down_m,
    stride_out_m,
    stride_s2d_t,
    stride_ids_t,
    stride_w_t,
    TOPK: tl.constexpr,
    BLOCK_H: tl.constexpr,
    WIDE: tl.constexpr,
):
    """Persistent fallback: masked BLOCK_H chunks for hidden sizes that
    are not divisible by the fast-path tile (or wider than the tile cap),
    any topk (padded lanes nullified), and int64-range tensors (WIDE).

    Masked 1D tiles at 1024 stay outside the 11-07 regression's trigger
    face; the token walk is a grid-stride loop so the launch stays small
    for arbitrarily large num_tokens.
    """
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    kis = tl.arange(0, TOPK)
    live = kis < topk
    for t in range(pid, num_tokens, nprog):
        # Padded lanes read as invalid (id -1 -> zero weight, row 0).
        idv = tl.load(ids_ptr + t * stride_ids_t + kis, mask=live, other=-1)
        dstv = tl.load(
            src2dst_ptr + t * stride_s2d_t + kis, mask=live, other=0
        )
        wv = tl.load(w_ptr + t * stride_w_t + kis, mask=live, other=0.0)
        valid = idv >= 0
        dstc = tl.where(valid, dstv, 0)
        wf = tl.where(valid, wv, 0.0)
        if WIDE:
            t_out = t.to(tl.int64) * stride_out_m
        else:
            t_out = t * stride_out_m
        for c0 in range(0, hidden_size, BLOCK_H):
            cols = c0 + tl.arange(0, BLOCK_H)
            cmask = cols < hidden_size
            acc = tl.zeros([BLOCK_H], dtype=tl.float32)
            for i in tl.static_range(TOPK):
                di = tl.sum(tl.where(kis == i, dstc, 0))
                wi = tl.sum(tl.where(kis == i, wf, 0.0))
                if WIDE:
                    row = tl.load(
                        down_ptr + di.to(tl.int64) * stride_down_m + cols,
                        mask=cmask,
                        other=0.0,
                    )
                else:
                    row = tl.load(
                        down_ptr + di * stride_down_m + cols,
                        mask=cmask,
                        other=0.0,
                    )
                acc += row.to(tl.float32) * wi
            tl.store(
                out_ptr + t_out + cols,
                (acc * scaling).to(out_ptr.dtype.element_ty),
                mask=cmask,
            )


def post_reorder_deepgemm(
    down_output,
    output,
    src2dst,
    topk_ids,
    topk_weights,
    topk,
    num_tokens,
    hidden_size,
    routed_scaling_factor,
):
    if num_tokens == 0 or hidden_size == 0 or topk == 0:
        return output

    # 64-bit offsets are only needed when a linear index can exceed int32.
    max_elem = down_output.numel() + num_tokens * hidden_size
    wide = max_elem > 2**31 - 1
    topk_pad = max(triton.next_power_of_2(topk), 1)
    scaling = float(routed_scaling_factor)
    strides = (
        down_output.stride(0),
        output.stride(0),
        src2dst.stride(0),
        topk_ids.stride(0),
        topk_weights.stride(0),
    )

    def launch_wide():
        grid = (min(num_tokens, _MAX_PROGRAMS),)
        _post_reorder_kernel_wide[grid](
            down_output,
            output,
            src2dst,
            topk_ids,
            topk_weights,
            hidden_size,
            num_tokens,
            topk,
            scaling,
            *strides,
            TOPK=topk_pad,
            BLOCK_H=_BLOCK_H,
            WIDE=wide,
            num_warps=1,
            num_stages=1,
        )

    if wide or hidden_size > _MAX_BLOCK_H or hidden_size % _BLOCK_H != 0:
        # Fallback: hidden not divisible by the fast-path column tile
        # (or wider than the tile cap) / int64-range tensors.
        launch_wide()
        return output

    nb = hidden_size // _BLOCK_H

    if topk == 8:
        if hidden_size >= _ROW_WIDE_H:
            # Wide-hidden fast path: BLOCK=4096 row kernel. num_warps
            # follows the masked-tail form (4 on TAIL, 2 on exact
            # division -- E-1 probe). Token counts the 1D row grid
            # cannot carry keep the v2 persistent whole-row walk.
            if num_tokens > _GRID_X_LIMIT:
                _post_reorder_kernel_row_persist[(_PERSIST_PROGRAMS,)](
                    down_output,
                    output,
                    src2dst,
                    topk_ids,
                    topk_weights,
                    num_tokens,
                    scaling,
                    *strides,
                    BLOCK_H=_BLOCK_H,
                    NB=nb,
                    num_warps=2,
                    num_stages=1,
                )
                return output
            nf = hidden_size // _WIDE_BLOCK_H
            tail = hidden_size % _WIDE_BLOCK_H != 0
            if num_tokens == 1 and tail:
                # T=1 on masked-tail wide shapes keeps the v2 direct-mapped
                # form: interleaved A/B measured direct 4.97-5.23us vs row
                # 5.27-5.30us at H=7168 (latency-bound; 7 parallel column
                # programs hide more gather latency than one wide program).
                # Exact-division wide shapes stay on the row kernel (H=4096
                # T=1: row 4.20us wins).
                _post_reorder_kernel[(nb,)](
                    down_output,
                    output,
                    src2dst,
                    topk_ids,
                    topk_weights,
                    scaling,
                    *strides,
                    TOPK=8,
                    BLOCK_H=_BLOCK_H,
                    NB=nb,
                    num_warps=2,
                    num_stages=1,
                )
                return output
            _post_reorder_kernel_row4096[(num_tokens,)](
                down_output,
                output,
                src2dst,
                topk_ids,
                topk_weights,
                hidden_size,
                scaling,
                *strides,
                BLOCK_H=_WIDE_BLOCK_H,
                NF=nf,
                TAIL=tail,
                SCALE_ONE=(scaling == 1.0),
                num_warps=4 if tail else 2,
                num_stages=1,
            )
            return output
        if num_tokens >= _ROW_MIN_TOKENS:
            # Narrow-hidden v2 dispatch: NB=4 (hidden 4096) keeps the row
            # kernel at every measured token count; wider/narrower tiles
            # hand over to the persistent form past their measured
            # crossovers, and any token count the row grid cannot carry
            # (>65535) does too.
            persist_min = (
                _PERSIST_MIN_TOKENS_WIDE
                if nb >= 5
                else _PERSIST_MIN_TOKENS_NARROW
            )
            if num_tokens > _GRID_X_LIMIT or (
                nb != 4 and num_tokens >= persist_min
            ):
                _post_reorder_kernel_row_persist[(_PERSIST_PROGRAMS,)](
                    down_output,
                    output,
                    src2dst,
                    topk_ids,
                    topk_weights,
                    num_tokens,
                    scaling,
                    *strides,
                    BLOCK_H=_BLOCK_H,
                    NB=nb,
                    num_warps=2,
                    num_stages=1,
                )
                return output
            # Whole-row kernel: one program per token; measured crossover
            # against the direct-mapped grid sits between T=8 and T=16.
            _post_reorder_kernel_row[(num_tokens,)](
                down_output,
                output,
                src2dst,
                topk_ids,
                topk_weights,
                scaling,
                *strides,
                BLOCK_H=_BLOCK_H,
                NB=nb,
                num_warps=2,
                num_stages=1,
            )
            return output

    # Direct-mapped grid: one program per (token, column tile).
    if num_tokens * nb <= _GRID_X_LIMIT:
        grid = (num_tokens * nb,)
        if topk_pad == topk:
            _post_reorder_kernel[grid](
                down_output,
                output,
                src2dst,
                topk_ids,
                topk_weights,
                scaling,
                *strides,
                TOPK=topk_pad,
                BLOCK_H=_BLOCK_H,
                NB=nb,
                num_warps=2,
                num_stages=1,
            )
        else:
            # Rare path: pad topk to the next power of two.
            _post_reorder_kernel_padded[grid](
                down_output,
                output,
                src2dst,
                topk_ids,
                topk_weights,
                topk,
                scaling,
                *strides,
                TOPK=topk_pad,
                BLOCK_H=_BLOCK_H,
                NB=nb,
                num_warps=1,
                num_stages=1,
            )
        return output

    # Huge token count that neither 1D grid can carry: persistent walk.
    launch_wide()
    return output


__all__ = ["post_reorder_deepgemm"]
