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

"""Causal 1-D convolution over continuous-batched sequences (Enflame GCU
/ NVIDIA portable).

    out[d, start + t] = silu(
        sum_{k=0}^{WIDTH-1} weight[d, k]
            * x[d, start + t - (WIDTH-1) + k]
        + bias[d]
    )

where ``start``/``end`` come from ``query_start_loc`` (a per-sequence
prefix-sum into a packed ``x`` of shape ``(dim, total_t)``), and inputs
before ``start`` are zero-padded.

GCU-specific constraints (triton 3.6.0 backend, measured 2026-11)
------------------------------------------------------------------
1. **The window shift lives in the *index tensor*, never in the base
   pointer** (``src_t = offs_t + k - (WIDTH-1)``): scalar base-pointer
   shifts on tile loads are silently dropped and affine masks mangled
   by the backend (mis-lowering, 73-75% of elements wrong; filed in
   ``compiler_issues/sglang_causal_conv1d_fn.md`` with a minimal repro).

2. **Load indices stay *affine*; bounds are enforced with affine
   compile-time masks, never value clamps and never vector predicates.**
   Three measured regimes on the benchmark shapes (dim=4096/total=16384,
   bf16, W=4):
   * min/max-clamped indices (``tl.minimum(tl.maximum(src_t, 0), T-1)``):
     the nonlinear index destroys the affine structure of the load ->
     per-element gather degradation. Full kernel 1.10 ms.
   * affine mask ``src_t >= 0`` + post-load elementwise ``tl.where`` for
     the per-position segment bound: 0.73 ms (-34%).
   * folding the segment bound *into* the load mask (vector-valued
     predicate): 4.14 ms (7.6x collapse) -- the mask predicate itself
     must be compile-time affine; anything per-position stays in the
     cheap select after the load.

3. **T_EVEN / D_EVEN constexpr specialization.** ``total_t % BLOCK_T ==
   0`` and ``dim % ROWS == 0`` hold for the benchmark shapes; the
   constexpr arms then drop the tail predicates on loads and the store
   masks entirely. Dead-but-unprovable predicates and masked stores
   defeat wide vector codegen even when they are all-true at runtime
   (-20% on top of the affine masks). Mixed/odd shapes take the fully
   masked arm (bitwise-identical results, verified).

4. **BLOCK_T = 4096 sweet spot (DSM budget).** Per-program live state is
   ~6-8 ``[BLOCK_T]`` fp32/int32 vectors (offs/seg_start/lo/acc/tap/
   predicates): ~96-128 KiB at BLOCK_T=4096, inside the ~448 KiB
   dual-program line; 8192 (~192-256 KiB) starts to regress and 16384
   blows through it (2.6 ms pathology). The earlier 2048 cap was
   measured on the mis-vectorized clamp form; on the affine-mask form
   4096 beats 2048 by a further -24%.

5. **1-D per-row shape (all tile loads/stores are 1-D ``[BLOCK_T]``
   vectors).** The 2026-11-07 20:22 toolchain made every 2-D tile load
   fail to compile (``memref_ext.slice_start ... 7 vs 6``, same family
   as pre_reorder_cutlass; filed in the compiler_issues archive). The
   1-D per-row form is immune, so ROWS channel rows run per program
   with purely 1-D vector memory ops and scalar weight/bias loads.

Grid / tiling
-------------
* ``grid.x = cdiv(total_t, BLOCK_T)`` (packed time, <= 65535) and
  ``grid.y = cdiv(dim, ROWS)`` (row blocks, <= 255; ``ROWS`` ~ dim/128,
  floor 16, doubling only to respect the cap).
* ``num_warps`` follows the vector width (8 at BLOCK_T=4096; nw4 equal,
  nw2 10-15% slower).

Measured (benchmark shapes, bf16, W=4, torch-event kernel timing):
dim=4096/total=16384 0.542 ms and dim=2048/total=16384 0.294 ms, vs
1.10 / 0.57 ms for the clamp-indexed predecessor (this file, 2026-11-08)
and 2.50 / 1.00 ms for the retired 2-D tile kernel on the pre-2026-11-07
toolchain. Output is bitwise-identical to both predecessors (same fp32
tap order; verified with torch.equal across uniform, mixed-length and
odd-dim shapes).

Segment-start index (host pre-processing)
-----------------------------------------
The per-position ``seg_start`` vector is built on-device with a tiny
Triton kernel (``build_seg_start_kernel``) instead of
``torch.searchsorted`` (~1.1ms for a single 4096-length sequence); the
branchless max-scan over segment boundaries is 1.5-45x cheaper and stays
entirely in Triton (portable, no torch fallback). At ~5 us it is ~1% of
the op and is kept: replacing it with per-segment scalar loads inside
the main kernel measured 1.6-3.8 ms (the runtime-scalar segment offset
in the load address defeats vector codegen), so the packed-time grid +
per-position seg vector is the fast shape on this backend.

Portability
-----------
The kernels are plain Triton: no device-specific branching, no torch
fallback. int32 offset math stays within ``dim * total_t < 2**31``.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def build_seg_start_kernel(
    qsl_ptr,
    seg_start_ptr,
    NSEGS,
    TOTAL_T: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """seg_start[t] = max_{i} qsl[i] subject to qsl[i] <= t.

    i.e. the start of the segment containing packed position t.  ``qsl``
    is the sorted prefix-sum of sequence lengths, so the running max over
    all boundaries <= t is exactly the right boundary.

    NSEGS is a *runtime* scalar (number of boundaries = n_seqs + 1), so
    the loop stays dynamic and does not fully unroll for large n_seqs.
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < TOTAL_T
    t = offs

    # Start from the first boundary qsl[0] (== 0 in practice) and keep
    # the running max of every boundary <= t.
    seg = tl.load(qsl_ptr + 0) + tl.zeros((BLOCK,), dtype=tl.int32)
    for i in range(1, NSEGS):
        b = tl.load(qsl_ptr + i)
        seg = tl.maximum(seg, tl.where(b <= t, b, seg))
    tl.store(seg_start_ptr + offs, seg, mask=mask)


@triton.jit
def causal_conv1d_fn_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    seg_start_ptr,
    out_ptr,
    TOTAL_T: tl.constexpr,
    DIM: tl.constexpr,
    WIDTH: tl.constexpr,
    BLOCK_T: tl.constexpr,
    ROWS: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    HAS_ACTIVATION: tl.constexpr,
    T_EVEN: tl.constexpr,
    D_EVEN: tl.constexpr,
):
    # grid = (num_t_blocks, num_row_blocks): packed time on grid.x
    # (<= 65535), ROWS channel rows per program on grid.y (<= 255).
    # 1-D per-row shape: all tile loads/stores are [BLOCK_T] vectors
    # (2-D tile loads are un-compilable on the 2026-11-07+ toolchain,
    # see module docstring / compiler_issues archive).
    pid_t = tl.program_id(0)  # packed-time block
    pid_d = tl.program_id(1)  # row block

    offs_t = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)  # (BLOCK_T,) pos

    # seg_start[t] = start of the segment that packed position t belongs
    # to. T_EVEN shapes have no tail lanes, so the gather needs no
    # clamp; the clamped form only serves the masked general arm.
    if T_EVEN:
        seg_start = tl.load(seg_start_ptr + offs_t)
    else:
        seg_start = tl.load(seg_start_ptr + tl.minimum(offs_t, TOTAL_T - 1))

    # Left limit of valid taps: tap k reads x[t - (WIDTH-1) + k], which
    # is causal-valid iff t - (WIDTH-1) + k >= seg_start[t]
    # <=> t >= seg_start[t] + (WIDTH-1) - k.
    lo = seg_start + (WIDTH - 1)  # (BLOCK_T,)

    d0 = pid_d * ROWS
    for r in tl.static_range(ROWS):
        d = d0 + r
        if D_EVEN:
            d_safe = d
        else:
            d_safe = tl.minimum(d, DIM - 1)

        acc = tl.zeros((BLOCK_T,), dtype=tl.float32)
        for k in tl.static_range(WIDTH):
            # Window shift lives in the index tensor (NOT the base
            # pointer): the backend drops scalar-pointer deltas on tile
            # loads -- see module docstring / compiler_issues archive.
            src_t = offs_t + k - (WIDTH - 1)
            # Affine, compile-time-provable mask keeps the load
            # vectorized (a min/max clamp on src_t collapses it to a
            # per-element gather; a vector-valued mask predicate
            # collapses it even harder -- see module docstring).
            if T_EVEN:
                m = src_t >= 0
            else:
                m = (src_t >= 0) & (offs_t < TOTAL_T)
            v = tl.load(
                x_ptr + d_safe * TOTAL_T + src_t, mask=m, other=0.0
            ).to(tl.float32)
            # Segment boundary: zero lanes whose window reaches left of
            # the segment start. Per-position (vector) condition, so it
            # must stay a post-load select -- folding it into the load
            # mask would collapse vectorization (0.54 -> 4.14 ms).
            ok = offs_t >= (lo - k)
            v = tl.where(ok, v, 0.0)

            w = tl.load(weight_ptr + d_safe * WIDTH + k)
            acc += v * w

        if HAS_BIAS:
            acc += tl.load(bias_ptr + d_safe)

        if HAS_ACTIVATION:
            acc = acc / (1.0 + tl.exp(-acc))  # SiLU / swish

        # T_EVEN/D_EVEN constexpr arms drop dead store masks (masked
        # stores defeat wide vector codegen even when all-true).
        if T_EVEN and D_EVEN:
            tl.store(out_ptr + d_safe * TOTAL_T + offs_t, acc)
        elif T_EVEN:
            tl.store(out_ptr + d_safe * TOTAL_T + offs_t, acc, mask=d < DIM)
        elif D_EVEN:
            tl.store(
                out_ptr + d_safe * TOTAL_T + offs_t, acc, mask=offs_t < TOTAL_T
            )
        else:
            tl.store(
                out_ptr + d_safe * TOTAL_T + offs_t,
                acc,
                mask=(offs_t < TOTAL_T) & (d < DIM),
            )


# ---- host-side launch configuration and wrapper ----------------------


def _next_pow2(n):
    """Smallest power of two >= n (n >= 1)."""
    return 1 << (n - 1).bit_length()


def _launch_config(dim, total_t):
    """Pick BLOCK_T / ROWS / num_warps.

    Honors the GCU grid caps (grid.y <= 255 row blocks, grid.x <= 65535
    time blocks). Per-program live DSM is ~6-8 [BLOCK_T] vectors, so the
    4096-wide tile sits ~96-128 KiB/program, inside the ~448 KiB
    dual-program line (8192 regresses, 16384 blows through it).

    Tuned on the benchmark shapes (dim=4096 / dim=2048, total_t=16384,
    bf16, W=4) on the affine-mask kernel by single-config sweeps:
    BLOCK_T=4096 is the sweet spot (~25% faster than 2048 on both
    shapes; 8192 regresses again), ~128 row blocks on grid.y balance
    per-program work against occupancy (ROWS 16-64 all within 2% at
    BLOCK_T=4096; the rule picks 32/16 for the two benchmark dims), and
    num_warps follows the vector width (nw 4 and 8 measured equivalent,
    nw2 10-15% slower).
    """
    # Time block: 4096 preferred (contiguous axis, measured sweet spot);
    # shrink for tiny total_t so small problems stay one wave.
    block_t = min(4096, _next_pow2(max(total_t, 1)))

    # Rows per program: ~128 row blocks on grid.y (dim/128, pow2, floor
    # 16); grow only to respect the grid.y <= 255 cap.
    rows = max(16, _next_pow2(max(dim // 128, 1)))
    while triton.cdiv(dim, rows) > 255:
        rows *= 2

    # num_warps scales with the vector width (nw 8 and 4 measured
    # equivalent at BLOCK_T=4096; the rule keeps the scaling).
    num_warps = min(8, max(2, block_t // 256))

    # Guard grid.x <= 65535 for pathological total_t (grow the tile;
    # such shapes also approach the int32 offset limit and are not
    # supported anyway).
    while triton.cdiv(total_t, block_t) > 65535:
        block_t *= 2

    return block_t, rows, num_warps


def causal_conv1d_fn(
    x, weight, bias, query_start_loc, seq_lens_cpu, activation="silu"
):
    """Fused causal depthwise conv1d + optional silu.

    Signature matches the reference.
    """
    dim, total_t = x.shape
    width = weight.shape[1]

    # Per packed-position segment start. Built on-device with a Triton
    # kernel (see ``build_seg_start_kernel``): ``torch.searchsorted`` +
    # gather is surprisingly slow on the GCU (~1.1ms for a single long
    # sequence, ~350us for many segments), while the branchless max-scan
    # below is 1.5-45x cheaper and stays entirely in Triton.
    # (seq_lens_cpu is accepted for API compatibility but not needed:
    # query_start_loc already encodes every segment boundary.)
    qsl = query_start_loc.to(device=x.device, dtype=torch.int32)
    seg_start = torch.empty(total_t, device=x.device, dtype=torch.int32)
    seg_block = min(1024, _next_pow2(max(total_t, 1)))
    build_seg_start_kernel[(triton.cdiv(total_t, seg_block),)](
        qsl,
        seg_start,
        qsl.numel(),
        TOTAL_T=total_t,
        BLOCK=seg_block,
        num_warps=1,
    )

    block_t, rows, num_warps = _launch_config(dim, total_t)
    num_t_blocks = triton.cdiv(total_t, block_t)
    num_row_blocks = triton.cdiv(dim, rows)

    out = torch.empty_like(x)
    has_activation = activation in ("silu", "swish")

    causal_conv1d_fn_kernel[(num_t_blocks, num_row_blocks)](
        x,
        weight,
        x if bias is None else bias,
        seg_start,
        out,
        TOTAL_T=total_t,
        DIM=dim,
        WIDTH=width,
        BLOCK_T=block_t,
        ROWS=rows,
        HAS_BIAS=bias is not None,
        HAS_ACTIVATION=has_activation,
        T_EVEN=total_t % block_t == 0,
        D_EVEN=dim % rows == 0,
        num_warps=num_warps,
    )
    return out


__all__ = ["causal_conv1d_fn"]
