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

"""Fused affine-free LayerNorm over the head dimension of Q and K."""

import torch
import triton
import triton.language as tl

# Flat-chunk elements per single-warp program: above this the current
# GCU toolchain fails 1D tile accesses (memref_ext.slice_start segment
# mismatch) or, with the register reshape consumed by axis-1 reductions,
# aborts the device with a Sip exception (see
# compiler_issues/sglang_qk_head_layernorm.md). 2048-8192 plain copies
# probe OK, but 4096 is the safe bound once tl.reshape enters the picture.
_FLAT_ELEMS_MAX = 4096
# Persistent grid-stride launch: programs per tensor (Q/K live on grid
# axis 1), i.e. 48 resident CTAs with the Q/K axis on the 48-SP part.
# Measured on GCU400, bf16, DH=128 (do_bench median, w1000/r100):
# 24 >= 48 >= 96 at every core shape; non-multiples of the SP count
# (e.g. 32/64) lose to wave quantization.
_GRID_TARGET = 24
# Multi-warp lowering of the reshape + axis-1 reduction fails to compile
# on this toolchain (slice_start 6 vs 7, not rescued by the raw-IR retry),
# so single-warp programs with in-warp reductions are the only form.
_WARPS = 1


@triton.jit
def _qk_head_ln_row(
    q_ptr,
    k_ptr,
    qo_ptr,
    ko_ptr,
    is_k,
    row,
    DH: tl.constexpr,  # true head dim (mean/var divisor, column bound)
    eps,
    BLOCK: tl.constexpr,  # next_pow2(DH), row is padded + masked up to it
):
    # Single-row body with strictly 1D affine access.
    cols = tl.max_contiguous(tl.multiple_of(tl.arange(0, BLOCK), BLOCK), BLOCK)
    offs = row * DH + cols
    PADDED: tl.constexpr = DH != BLOCK

    if PADDED:
        mask = cols < DH
        if is_k:
            x = tl.load(
                k_ptr + offs, mask=mask, other=0.0, cache_modifier=".cg"
            ).to(tl.float32)
        else:
            x = tl.load(
                q_ptr + offs, mask=mask, other=0.0, cache_modifier=".cg"
            ).to(tl.float32)
        mean = tl.sum(x, 0) / DH
        d = tl.where(mask, x - mean, 0.0)
        var = tl.sum(d * d, 0) / DH
        y = d * tl.rsqrt(var + eps)
        y = y.to(qo_ptr.dtype.element_ty)
        if is_k:
            tl.store(ko_ptr + offs, y, mask=mask, cache_modifier=".cg")
        else:
            tl.store(qo_ptr + offs, y, mask=mask, cache_modifier=".cg")
    else:
        if is_k:
            x = tl.load(k_ptr + offs, cache_modifier=".cg").to(tl.float32)
        else:
            x = tl.load(q_ptr + offs, cache_modifier=".cg").to(tl.float32)
        mean = tl.sum(x, 0) / DH
        d = x - mean
        var = tl.sum(d * d, 0) / DH
        y = d * tl.rsqrt(var + eps)
        y = y.to(qo_ptr.dtype.element_ty)
        if is_k:
            tl.store(ko_ptr + offs, y, cache_modifier=".cg")
        else:
            tl.store(qo_ptr + offs, y, cache_modifier=".cg")


@triton.jit
def _qk_head_ln_kernel(
    q_ptr,
    k_ptr,
    qo_ptr,
    ko_ptr,
    N_ROWS_Q,
    N_ROWS_K,
    DH: tl.constexpr,  # true head dim (mean/var divisor, column bound)
    eps,
    BLOCK: tl.constexpr,  # next_pow2(DH), row is padded + masked up to it
    ROWS: tl.constexpr,  # rows of (B*S*H) per grid-stride group
):
    # Persistent grid-stride over row groups; grid axis 1 selects the
    # tensor. Q and K have their own head counts under GQA, so their own
    # row counts.
    pid = tl.program_id(0)
    is_k = tl.program_id(1)
    if is_k:
        N_ROWS = N_ROWS_K
        src = k_ptr
        dst = ko_ptr
    else:
        N_ROWS = N_ROWS_Q
        src = q_ptr
        dst = qo_ptr
    # Full-chunk flat path needs DH == BLOCK (rows are one contiguous
    # span) and the flat tile inside the toolchain's safe bound.
    FLAT_OK: tl.constexpr = (DH == BLOCK) and (ROWS * BLOCK <= 4096)
    n_groups = tl.cdiv(N_ROWS, ROWS)
    if FLAT_OK:
        for grp in range(pid, n_groups, tl.num_programs(0)):
            # Tail-anchored full chunk: the last (partial) group is slid
            # back to end at N_ROWS, re-deriving a few already-stored rows.
            # LayerNorm rows are independent, so the overlap is an
            # idempotent rewrite and no row mask / tail branch is needed.
            # The clamp is a scalar; the address vector stays affine.
            base = tl.minimum(grp * ROWS, N_ROWS - ROWS)
            offs = base * DH + tl.arange(0, ROWS * BLOCK)
            x = tl.load(src + offs, cache_modifier=".cg").to(tl.float32)
            x2 = tl.reshape(x, (ROWS, BLOCK))
            # Single-pass variance E[x^2] - mean^2 (clamped at 0): both
            # row reductions depend only on the loaded tile, which takes
            # the mean broadcast off the reduction critical path and drops
            # one intermediate tile. Bit-exact against the two-pass form
            # after the bf16 output rounding on measured cases.
            mean = tl.sum(x2, axis=1) / DH
            msq = tl.sum(x2 * x2, axis=1) / DH
            var = tl.maximum(msq - mean * mean, 0.0)
            rstd = tl.rsqrt(var + eps)
            d = x2 - mean[:, None]
            y = (d * rstd[:, None]).to(dst.dtype.element_ty)
            yf = tl.reshape(y, (ROWS * BLOCK,))
            tl.store(dst + offs, yf, cache_modifier=".cg")
    else:
        # Padded head dim (or a flat tile beyond the safe bound): fall
        # back to per-row 1D.
        for grp in range(pid, n_groups, tl.num_programs(0)):
            base = grp * ROWS
            for r in tl.static_range(ROWS):
                row = base + r
                if row < N_ROWS:
                    _qk_head_ln_row(
                        q_ptr, k_ptr, qo_ptr, ko_ptr, is_k, row, DH, eps, BLOCK
                    )


@triton.jit
def _ln_subtile(
    xb, DH: tl.constexpr, eps, BLOCK: tl.constexpr, ROWS: tl.constexpr, out_ty
):
    # One flat (ROWS * BLOCK)-element subtile: reshape to (ROWS, BLOCK),
    # single-pass moments (same math and op order as the single-chunk
    # FLAT_OK arm, so results are bit-identical), normalize, cast.
    x2 = tl.reshape(xb, (ROWS, BLOCK)).to(tl.float32)
    mean = tl.sum(x2, axis=1) / DH
    msq = tl.sum(x2 * x2, axis=1) / DH
    var = tl.maximum(msq - mean * mean, 0.0)
    rstd = tl.rsqrt(var + eps)
    return ((x2 - mean[:, None]) * rstd[:, None]).to(out_ty)


@triton.jit
def _qk_head_ln_multi(
    q_ptr,
    k_ptr,
    qo_ptr,
    ko_ptr,
    N_ROWS_Q,
    N_ROWS_K,
    DH: tl.constexpr,  # true head dim (mean/var divisor, column bound)
    eps,
    BLOCK: tl.constexpr,  # next_pow2(DH) == DH on this arm
    ROWS: tl.constexpr,  # rows per flat subtile; ROWS * BLOCK <= 4096
    MULT: tl.constexpr,  # subtile loads issued up front per group (2/4/8)
):
    # Interleaved multi-subtile arm of the same persistent grid-stride
    # scheme. All MULT flat loads of a group are issued up front, so the
    # DMA fill of subtile i+1 overlaps the reduction trees of subtile i:
    # the single-chunk loop pays memory and tree time additively
    # (99us floor + 243us trees @ s=4360), while this form hides the load
    # latency behind tree compute (measured -19/-32/-38% at the
    # s=128/1024/4360 core shapes, kernel-only do_bench). Stores stay
    # interleaved (one store right after each subtile's trees): batching
    # all stores to the end of the group instead measures +64% (register
    # liveness of the y tiles; same mechanism as the earlier U2 veto).
    # E-1's "+69% U2 double chunk" veto was that batched-store form.
    pid = tl.program_id(0)
    is_k = tl.program_id(1)
    if is_k:
        n_rows = N_ROWS_K
        src = k_ptr
        dst = ko_ptr
    else:
        n_rows = N_ROWS_Q
        src = q_ptr
        dst = qo_ptr
    n_groups = tl.cdiv(n_rows, MULT * ROWS)
    col = tl.arange(0, ROWS * BLOCK)
    for grp in range(pid, n_groups, tl.num_programs(0)):
        # Tail-anchored group of MULT subtile spans (same idempotent
        # rewrite trick as the single-chunk arm).
        base = tl.minimum(grp * MULT * ROWS, n_rows - MULT * ROWS)
        if MULT == 2:
            xa = tl.load(src + base * DH + col, cache_modifier=".cg")
            xb = tl.load(src + (base + ROWS) * DH + col, cache_modifier=".cg")
            ya = _ln_subtile(xa, DH, eps, BLOCK, ROWS, dst.dtype.element_ty)
            tl.store(
                dst + base * DH + col,
                tl.reshape(ya, (ROWS * BLOCK,)),
                cache_modifier=".cg",
            )
            yb = _ln_subtile(xb, DH, eps, BLOCK, ROWS, dst.dtype.element_ty)
            tl.store(
                dst + (base + ROWS) * DH + col,
                tl.reshape(yb, (ROWS * BLOCK,)),
                cache_modifier=".cg",
            )
        elif MULT == 4:
            xa = tl.load(src + base * DH + col, cache_modifier=".cg")
            xb = tl.load(src + (base + ROWS) * DH + col, cache_modifier=".cg")
            xc = tl.load(
                src + (base + 2 * ROWS) * DH + col, cache_modifier=".cg"
            )
            xd = tl.load(
                src + (base + 3 * ROWS) * DH + col, cache_modifier=".cg"
            )
            ya = _ln_subtile(xa, DH, eps, BLOCK, ROWS, dst.dtype.element_ty)
            tl.store(
                dst + base * DH + col,
                tl.reshape(ya, (ROWS * BLOCK,)),
                cache_modifier=".cg",
            )
            yb = _ln_subtile(xb, DH, eps, BLOCK, ROWS, dst.dtype.element_ty)
            tl.store(
                dst + (base + ROWS) * DH + col,
                tl.reshape(yb, (ROWS * BLOCK,)),
                cache_modifier=".cg",
            )
            yc = _ln_subtile(xc, DH, eps, BLOCK, ROWS, dst.dtype.element_ty)
            tl.store(
                dst + (base + 2 * ROWS) * DH + col,
                tl.reshape(yc, (ROWS * BLOCK,)),
                cache_modifier=".cg",
            )
            yd = _ln_subtile(xd, DH, eps, BLOCK, ROWS, dst.dtype.element_ty)
            tl.store(
                dst + (base + 3 * ROWS) * DH + col,
                tl.reshape(yd, (ROWS * BLOCK,)),
                cache_modifier=".cg",
            )
        else:
            x0 = tl.load(src + base * DH + col, cache_modifier=".cg")
            x1 = tl.load(src + (base + ROWS) * DH + col, cache_modifier=".cg")
            x2 = tl.load(
                src + (base + 2 * ROWS) * DH + col, cache_modifier=".cg"
            )
            x3 = tl.load(
                src + (base + 3 * ROWS) * DH + col, cache_modifier=".cg"
            )
            x4 = tl.load(
                src + (base + 4 * ROWS) * DH + col, cache_modifier=".cg"
            )
            x5 = tl.load(
                src + (base + 5 * ROWS) * DH + col, cache_modifier=".cg"
            )
            x6 = tl.load(
                src + (base + 6 * ROWS) * DH + col, cache_modifier=".cg"
            )
            x7 = tl.load(
                src + (base + 7 * ROWS) * DH + col, cache_modifier=".cg"
            )
            y0 = _ln_subtile(x0, DH, eps, BLOCK, ROWS, dst.dtype.element_ty)
            tl.store(
                dst + base * DH + col,
                tl.reshape(y0, (ROWS * BLOCK,)),
                cache_modifier=".cg",
            )
            y1 = _ln_subtile(x1, DH, eps, BLOCK, ROWS, dst.dtype.element_ty)
            tl.store(
                dst + (base + ROWS) * DH + col,
                tl.reshape(y1, (ROWS * BLOCK,)),
                cache_modifier=".cg",
            )
            y2 = _ln_subtile(x2, DH, eps, BLOCK, ROWS, dst.dtype.element_ty)
            tl.store(
                dst + (base + 2 * ROWS) * DH + col,
                tl.reshape(y2, (ROWS * BLOCK,)),
                cache_modifier=".cg",
            )
            y3 = _ln_subtile(x3, DH, eps, BLOCK, ROWS, dst.dtype.element_ty)
            tl.store(
                dst + (base + 3 * ROWS) * DH + col,
                tl.reshape(y3, (ROWS * BLOCK,)),
                cache_modifier=".cg",
            )
            y4 = _ln_subtile(x4, DH, eps, BLOCK, ROWS, dst.dtype.element_ty)
            tl.store(
                dst + (base + 4 * ROWS) * DH + col,
                tl.reshape(y4, (ROWS * BLOCK,)),
                cache_modifier=".cg",
            )
            y5 = _ln_subtile(x5, DH, eps, BLOCK, ROWS, dst.dtype.element_ty)
            tl.store(
                dst + (base + 5 * ROWS) * DH + col,
                tl.reshape(y5, (ROWS * BLOCK,)),
                cache_modifier=".cg",
            )
            y6 = _ln_subtile(x6, DH, eps, BLOCK, ROWS, dst.dtype.element_ty)
            tl.store(
                dst + (base + 6 * ROWS) * DH + col,
                tl.reshape(y6, (ROWS * BLOCK,)),
                cache_modifier=".cg",
            )
            y7 = _ln_subtile(x7, DH, eps, BLOCK, ROWS, dst.dtype.element_ty)
            tl.store(
                dst + (base + 7 * ROWS) * DH + col,
                tl.reshape(y7, (ROWS * BLOCK,)),
                cache_modifier=".cg",
            )


def qk_head_layernorm(q, k, eps):
    # Per-head LayerNorm, no affine: the last dim is the head dim.
    if not q.is_contiguous():
        q = q.contiguous()
    if not k.is_contiguous():
        k = k.contiguous()

    q_out = torch.empty_like(q)
    k_out = torch.empty_like(k)

    dh = q.shape[-1]
    if k.shape[-1] != dh:
        raise ValueError(
            f"q and k must share the head dim, got {dh} and {k.shape[-1]}"
        )
    n_rows_q = q.numel() // dh
    n_rows_k = k.numel() // dh
    n_rows = max(n_rows_q, n_rows_k)
    if n_rows == 0:
        return q_out, k_out
    min_rows = min(n_rows_q, n_rows_k)

    block = triton.next_power_of_2(dh)
    # Rows per flat subtile: fill the flat-tile budget (ROWS * BLOCK <=
    # 4096, the toolchain's reshape+axis-1-reduction safe bound).
    flat_rows = _FLAT_ELEMS_MAX // block
    if flat_rows < 1:
        flat_rows = 1
    # Single-chunk rows, halved against the SMALLER tensor so the
    # tail anchor (N_ROWS - ROWS) never goes negative for either Q or K
    # under GQA; tiny tensors keep at least one full chunk. An empty
    # tensor (0 rows) skips its loop entirely, so halve against the
    # larger tensor in that case (matching the original behavior).
    rows = flat_rows
    limit = min_rows if min_rows >= 1 else n_rows
    while rows > limit:
        rows //= 2

    # Interleaved multi-subtile arm (same math, bit-identical results):
    # for flat head dims with enough rows, MULT flat loads are issued up
    # front per grid-stride group so later subtile DMA overlaps earlier
    # subtile tree compute. MULT is picked by the larger tensor's row
    # count and clamped by the smaller one (keeps the group anchor
    # non-negative for both tensors).
    if (
        dh == block
        and flat_rows * block <= _FLAT_ELEMS_MAX
        and (min_rows >= 2 * flat_rows)
    ):
        if n_rows < 32768:
            mult = 2
        elif n_rows < 131072:
            mult = 4
        else:
            mult = 8
        while mult > 2 and mult * flat_rows > min_rows:
            mult //= 2
        n_groups = -(-n_rows // (mult * flat_rows))
        _qk_head_ln_multi[(min(n_groups, _GRID_TARGET), 2)](
            q,
            k,
            q_out,
            k_out,
            n_rows_q,
            n_rows_k,
            dh,
            eps,
            BLOCK=block,
            ROWS=flat_rows,
            MULT=mult,
            num_warps=_WARPS,
        )
        return q_out, k_out

    # Single-chunk / padded path (unchanged semantics).
    n_groups = -(-n_rows // rows)
    _qk_head_ln_kernel[(min(n_groups, _GRID_TARGET), 2)](
        q,
        k,
        q_out,
        k_out,
        n_rows_q,
        n_rows_k,
        dh,
        eps,
        BLOCK=block,
        ROWS=rows,
        num_warps=_WARPS,
    )
    return q_out, k_out


__all__ = ["qk_head_layernorm"]
