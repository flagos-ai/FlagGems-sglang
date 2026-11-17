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

import torch
import triton
import triton.language as tl

_ARM_C3 = 2
_ARM_EXACT = 1
_ARM_MASKED = 0


@triton.jit
def _rms_norm_row(
    X,
    W,
    Y,
    N,
    NUM_BLOCKS,
    eps,
    B: tl.constexpr,
    CHUNK: tl.constexpr,
    NPROG: tl.constexpr,
    ARM: tl.constexpr,
):
    # Persistent row kernel (one row per block step, 48 programs on this
    # device class).  All memory ops stay strictly 1-D: on this backend every
    # 2-D-tile formulation of this reduction either fails to compile
    # (memref_ext.slice_(pad_)start operandSegmentSizes, see compiler_issues/
    # vllm_silu_flat_slice_start.md) or silently miscompiles the broadcast
    # multiply chain (compiler_issues/triton_fp32_1d_bcast_miscompile.md).
    # The generic one-row-per-CTA grid pays a ~1us dispatch+tree chain per
    # row and wave-quantizes at large M; the persistent form amortizes it.
    pid = tl.program_id(0)
    if ARM == 2:
        # N == 3 * CHUNK with CHUNK a power of two: three independent
        # exact loads issue back-to-back, per-chunk trees stay short and
        # unmasked.  Measured -14..-18% vs the masked whole-block arm and
        # unlocks the shuffle-only (num_warps=1) reduction tree.
        offs = tl.arange(0, CHUNK)
        w0 = tl.load(W + offs).to(tl.float32)
        w1 = tl.load(W + CHUNK + offs).to(tl.float32)
        w2 = tl.load(W + 2 * CHUNK + offs).to(tl.float32)
        for blk in tl.range(pid, NUM_BLOCKS, NPROG, num_stages=2):
            base = blk * N
            c0 = tl.load(X + base + offs).to(tl.float32)
            c1 = tl.load(X + base + CHUNK + offs).to(tl.float32)
            c2 = tl.load(X + base + 2 * CHUNK + offs).to(tl.float32)
            var = (
                tl.sum(c0 * c0, axis=0)
                + tl.sum(c1 * c1, axis=0)
                + tl.sum(c2 * c2, axis=0)
            ) / N
            rrms = tl.rsqrt(var + eps)
            tl.store(Y + base + offs, (c0 * rrms * w0).to(Y.dtype.element_ty))
            tl.store(
                Y + base + CHUNK + offs,
                (c1 * rrms * w1).to(Y.dtype.element_ty),
            )
            tl.store(
                Y + base + 2 * CHUNK + offs,
                (c2 * rrms * w2).to(Y.dtype.element_ty),
            )
    else:
        offs = tl.arange(0, B)
        if ARM == 1:
            w = tl.load(W + offs).to(tl.float32)
        else:
            cm = offs < N
            w = tl.load(W + offs, mask=cm, other=0.0).to(tl.float32)
        for blk in tl.range(pid, NUM_BLOCKS, NPROG, num_stages=2):
            base = blk * N
            if ARM == 1:
                x = tl.load(X + base + offs).to(tl.float32)
                var = tl.sum(x * x, axis=0) / N
                rrms = tl.rsqrt(var + eps)
                tl.store(
                    Y + base + offs, (x * rrms * w).to(Y.dtype.element_ty)
                )
            else:
                x = tl.load(X + base + offs, mask=cm, other=0.0).to(tl.float32)
                var = tl.sum(x * x, axis=0) / N
                rrms = tl.rsqrt(var + eps)
                tl.store(
                    Y + base + offs,
                    (x * rrms * w).to(Y.dtype.element_ty),
                    mask=cm,
                )


@triton.jit
def _rms_norm_rows_c3_ub(
    X,
    W,
    Y,
    N,
    NUM_GROUPS,
    LAST0,
    eps,
    CHUNK: tl.constexpr,
    NPROG: tl.constexpr,
    U: tl.constexpr,
):
    # U-row batched persistent kernel for the d == 3 * 2^k arm (2026-11-12
    # scan, gcu400): all U*3 chunk loads issue in one window, then the U
    # independent reduction trees, then the stores.  Versus one row per
    # step this amortizes the per-row load-dispatch + tree chain that
    # otherwise sits fully on the critical path ((1024,1536) 20.9 -> 11.2,
    # (16384,1536) 285 -> 128us).  U is capped at 4: at U=6/8 the 18-24
    # parallel sums/loads overflow the per-program scheduler budget and
    # regress at every row count (unlike the register-byte budget, which
    # stays fine for CHUNK=2048).  The final group is anchored at LAST0 =
    # M - U so tail rows are re-written idempotently instead of masked.
    pid = tl.program_id(0)
    offs = tl.arange(0, CHUNK)
    w0 = tl.load(W + offs).to(tl.float32)
    w1 = tl.load(W + CHUNK + offs).to(tl.float32)
    w2 = tl.load(W + 2 * CHUNK + offs).to(tl.float32)
    for g in tl.range(pid, NUM_GROUPS, NPROG, num_stages=2):
        r0 = tl.minimum(g * U, LAST0)
        base0 = r0 * N
        c0_0 = tl.load(X + base0 + offs).to(tl.float32)
        c0_1 = tl.load(X + base0 + CHUNK + offs).to(tl.float32)
        c0_2 = tl.load(X + base0 + 2 * CHUNK + offs).to(tl.float32)
        c1_0 = tl.load(X + base0 + N + offs).to(tl.float32)
        c1_1 = tl.load(X + base0 + N + CHUNK + offs).to(tl.float32)
        c1_2 = tl.load(X + base0 + N + 2 * CHUNK + offs).to(tl.float32)
        c2_0 = tl.load(X + base0 + 2 * N + offs).to(tl.float32)
        c2_1 = tl.load(X + base0 + 2 * N + CHUNK + offs).to(tl.float32)
        c2_2 = tl.load(X + base0 + 2 * N + 2 * CHUNK + offs).to(tl.float32)
        c3_0 = tl.load(X + base0 + 3 * N + offs).to(tl.float32)
        c3_1 = tl.load(X + base0 + 3 * N + CHUNK + offs).to(tl.float32)
        c3_2 = tl.load(X + base0 + 3 * N + 2 * CHUNK + offs).to(tl.float32)
        var = (
            tl.sum(c0_0 * c0_0, 0)
            + tl.sum(c0_1 * c0_1, 0)
            + tl.sum(c0_2 * c0_2, 0)
        ) / N
        rr = tl.rsqrt(var + eps)
        tl.store(Y + base0 + offs, (c0_0 * rr * w0).to(Y.dtype.element_ty))
        tl.store(
            Y + base0 + CHUNK + offs, (c0_1 * rr * w1).to(Y.dtype.element_ty)
        )
        tl.store(
            Y + base0 + 2 * CHUNK + offs,
            (c0_2 * rr * w2).to(Y.dtype.element_ty),
        )
        var = (
            tl.sum(c1_0 * c1_0, 0)
            + tl.sum(c1_1 * c1_1, 0)
            + tl.sum(c1_2 * c1_2, 0)
        ) / N
        rr = tl.rsqrt(var + eps)
        tl.store(Y + base0 + N + offs, (c1_0 * rr * w0).to(Y.dtype.element_ty))
        tl.store(
            Y + base0 + N + CHUNK + offs,
            (c1_1 * rr * w1).to(Y.dtype.element_ty),
        )
        tl.store(
            Y + base0 + N + 2 * CHUNK + offs,
            (c1_2 * rr * w2).to(Y.dtype.element_ty),
        )
        var = (
            tl.sum(c2_0 * c2_0, 0)
            + tl.sum(c2_1 * c2_1, 0)
            + tl.sum(c2_2 * c2_2, 0)
        ) / N
        rr = tl.rsqrt(var + eps)
        tl.store(
            Y + base0 + 2 * N + offs, (c2_0 * rr * w0).to(Y.dtype.element_ty)
        )
        tl.store(
            Y + base0 + 2 * N + CHUNK + offs,
            (c2_1 * rr * w1).to(Y.dtype.element_ty),
        )
        tl.store(
            Y + base0 + 2 * N + 2 * CHUNK + offs,
            (c2_2 * rr * w2).to(Y.dtype.element_ty),
        )
        var = (
            tl.sum(c3_0 * c3_0, 0)
            + tl.sum(c3_1 * c3_1, 0)
            + tl.sum(c3_2 * c3_2, 0)
        ) / N
        rr = tl.rsqrt(var + eps)
        tl.store(
            Y + base0 + 3 * N + offs, (c3_0 * rr * w0).to(Y.dtype.element_ty)
        )
        tl.store(
            Y + base0 + 3 * N + CHUNK + offs,
            (c3_1 * rr * w1).to(Y.dtype.element_ty),
        )
        tl.store(
            Y + base0 + 3 * N + 2 * CHUNK + offs,
            (c3_2 * rr * w2).to(Y.dtype.element_ty),
        )


@triton.jit
def _rms_norm_rows_exact_ub(
    X,
    W,
    Y,
    N,
    NUM_GROUPS,
    LAST0,
    eps,
    B: tl.constexpr,
    NPROG: tl.constexpr,
    U: tl.constexpr,
):
    # U-row batched persistent kernel for the power-of-two-d arm.  With one
    # exact load per row the op count is half the 3-chunk arm, so U=8 stays
    # inside the per-program budget and wins ((1024,2048) 23.1 -> 10.6us,
    # (1024,1024) 26.7 -> 8.0us); B=8192 spills at U>=4 and uses U=2.
    pid = tl.program_id(0)
    offs = tl.arange(0, B)
    w = tl.load(W + offs).to(tl.float32)
    for g in tl.range(pid, NUM_GROUPS, NPROG, num_stages=2):
        r0 = tl.minimum(g * U, LAST0)
        base0 = r0 * N
        if U == 8:
            x0 = tl.load(X + base0 + offs).to(tl.float32)
            x1 = tl.load(X + base0 + N + offs).to(tl.float32)
            x2 = tl.load(X + base0 + 2 * N + offs).to(tl.float32)
            x3 = tl.load(X + base0 + 3 * N + offs).to(tl.float32)
            x4 = tl.load(X + base0 + 4 * N + offs).to(tl.float32)
            x5 = tl.load(X + base0 + 5 * N + offs).to(tl.float32)
            x6 = tl.load(X + base0 + 6 * N + offs).to(tl.float32)
            x7 = tl.load(X + base0 + 7 * N + offs).to(tl.float32)
            var = tl.sum(x0 * x0, 0) / N
            rr = tl.rsqrt(var + eps)
            tl.store(Y + base0 + offs, (x0 * rr * w).to(Y.dtype.element_ty))
            var = tl.sum(x1 * x1, 0) / N
            rr = tl.rsqrt(var + eps)
            tl.store(
                Y + base0 + N + offs, (x1 * rr * w).to(Y.dtype.element_ty)
            )
            var = tl.sum(x2 * x2, 0) / N
            rr = tl.rsqrt(var + eps)
            tl.store(
                Y + base0 + 2 * N + offs, (x2 * rr * w).to(Y.dtype.element_ty)
            )
            var = tl.sum(x3 * x3, 0) / N
            rr = tl.rsqrt(var + eps)
            tl.store(
                Y + base0 + 3 * N + offs, (x3 * rr * w).to(Y.dtype.element_ty)
            )
            var = tl.sum(x4 * x4, 0) / N
            rr = tl.rsqrt(var + eps)
            tl.store(
                Y + base0 + 4 * N + offs, (x4 * rr * w).to(Y.dtype.element_ty)
            )
            var = tl.sum(x5 * x5, 0) / N
            rr = tl.rsqrt(var + eps)
            tl.store(
                Y + base0 + 5 * N + offs, (x5 * rr * w).to(Y.dtype.element_ty)
            )
            var = tl.sum(x6 * x6, 0) / N
            rr = tl.rsqrt(var + eps)
            tl.store(
                Y + base0 + 6 * N + offs, (x6 * rr * w).to(Y.dtype.element_ty)
            )
            var = tl.sum(x7 * x7, 0) / N
            rr = tl.rsqrt(var + eps)
            tl.store(
                Y + base0 + 7 * N + offs, (x7 * rr * w).to(Y.dtype.element_ty)
            )
        else:
            x0 = tl.load(X + base0 + offs).to(tl.float32)
            x1 = tl.load(X + base0 + N + offs).to(tl.float32)
            var = tl.sum(x0 * x0, 0) / N
            rr = tl.rsqrt(var + eps)
            tl.store(Y + base0 + offs, (x0 * rr * w).to(Y.dtype.element_ty))
            var = tl.sum(x1 * x1, 0) / N
            rr = tl.rsqrt(var + eps)
            tl.store(
                Y + base0 + N + offs, (x1 * rr * w).to(Y.dtype.element_ty)
            )


@triton.jit
def _rms_norm_whole_row(
    X,
    W,
    Y,
    N,
    eps,
    B: tl.constexpr,
    CHUNK: tl.constexpr,
    ARM: tl.constexpr,
):
    # Launch-latency-bound fallback for tiny row counts: one row per program.
    row = tl.program_id(0)
    if ARM == 2:
        offs = tl.arange(0, CHUNK)
        w0 = tl.load(W + offs).to(tl.float32)
        w1 = tl.load(W + CHUNK + offs).to(tl.float32)
        w2 = tl.load(W + 2 * CHUNK + offs).to(tl.float32)
        base = row * N
        c0 = tl.load(X + base + offs).to(tl.float32)
        c1 = tl.load(X + base + CHUNK + offs).to(tl.float32)
        c2 = tl.load(X + base + 2 * CHUNK + offs).to(tl.float32)
        var = (
            tl.sum(c0 * c0, axis=0)
            + tl.sum(c1 * c1, axis=0)
            + tl.sum(c2 * c2, axis=0)
        ) / N
        rrms = tl.rsqrt(var + eps)
        tl.store(Y + base + offs, (c0 * rrms * w0).to(Y.dtype.element_ty))
        tl.store(
            Y + base + CHUNK + offs, (c1 * rrms * w1).to(Y.dtype.element_ty)
        )
        tl.store(
            Y + base + 2 * CHUNK + offs,
            (c2 * rrms * w2).to(Y.dtype.element_ty),
        )
    elif ARM == 1:
        offs = tl.arange(0, B)
        w = tl.load(W + offs).to(tl.float32)
        base = row * N
        x = tl.load(X + base + offs).to(tl.float32)
        var = tl.sum(x * x, axis=0) / N
        rrms = tl.rsqrt(var + eps)
        tl.store(Y + base + offs, (x * rrms * w).to(Y.dtype.element_ty))
    else:
        offs = tl.arange(0, B)
        cm = offs < N
        w = tl.load(W + offs, mask=cm, other=0.0).to(tl.float32)
        base = row * N
        x = tl.load(X + base + offs, mask=cm, other=0.0).to(tl.float32)
        var = tl.sum(x * x, axis=0) / N
        rrms = tl.rsqrt(var + eps)
        tl.store(
            Y + base + offs, (x * rrms * w).to(Y.dtype.element_ty), mask=cm
        )


# Column-arm dispatch (gcu400/Libra-class 48 SPs, bf16, do_bench median).
# 2026-11-08 first round: persistent-48 1-D row-batch, three arms (masked /
# pow2-exact / 3*2^k exact), num_warps=1 shuffle-only trees.
# 2026-11-12 second round: the per-row "load dispatch + reduction tree"
# chain still sat fully on the critical path (L0 copy floor 14.5us vs full
# 20.9us at (1024,1536)), so the row-batch arms now process U rows per
# loop step with all U*chunks loads issued in one window (U-batch):
#   - 3*2^k arm, m >= 192: U=4 ((1024,1536) 20.9->11.2us, (16384,1536)
#     285->128us); U=6/8 overflow the per-program op budget (24 sums) and
#     regress at every row count; interleaved per-row order and num_warps=2
#     regress 2x; merged [2*CHUNK]+[CHUNK] loads are WITHIN-NOISE.
#   - pow2 arm, B <= 4096, m >= 384: U=8 (one load/row halves the op
#     count, so U=8 fits: (1024,2048) 23.1->10.6us, (1024,1024) 26.7->8.0,
#     (1024,4096) 29.0->15.2); B=8192 spills at U>=4, uses U=2
#     ((1024,8192) 39.9->35.6us); B=16384 stays on the U=1 kernel.
#   - masked arm and m < 64: unchanged (no scan coverage, keep round-1
#     shape).  U-batch groups are tail-anchored at M-U (idempotent
#     re-write of overlap rows, no row mask), and each arm keeps
#     cdiv(m, U) >= 48 so the persistent grid stays fully populated.
_ROWS_MIN_ROWS = 64
_NUM_SPC = 48
_ROWS_MAX_BLOCK = 16384
_C3_MAX_CHUNK = 2048
_UB_C3_ROWS = 4 * _NUM_SPC
_UB_EXACT8_ROWS = 8 * _NUM_SPC
_UB_EXACT2_ROWS = 2 * _NUM_SPC
_UB_EXACT8_MAX_BLOCK = 4096
_UB_EXACT2_BLOCK = 8192


def _col_plan(d):
    block = triton.next_power_of_2(d)
    if block == d:
        return _ARM_EXACT, block, 0
    chunk = d // 3
    if d % 3 == 0 and chunk & (chunk - 1) == 0 and chunk <= _C3_MAX_CHUNK:
        return _ARM_C3, block, chunk
    return _ARM_MASKED, block, 0


def _rows_warps(arm, block):
    if arm == _ARM_C3:
        return 1
    if arm == _ARM_EXACT:
        return 1 if block <= 4096 else 4
    return 1 if block <= 2048 else 4


def _whole_row_warps(arm, block):
    if arm == _ARM_C3:
        return 1
    if arm == _ARM_EXACT:
        return 1 if block <= 4096 else 4
    return 2 if block <= 4096 else 4


def one_pass_rms_norm(x, w, eps):
    values = x.contiguous()
    weights = w.contiguous()
    out = torch.empty_like(values)
    d = values.shape[-1]
    m = values.numel() // d
    if m == 0:
        return out
    arm, block, chunk = _col_plan(d)
    if m >= _ROWS_MIN_ROWS and block <= _ROWS_MAX_BLOCK:
        if arm == _ARM_C3 and m >= _UB_C3_ROWS:
            groups = (m + 3) // 4
            _rms_norm_rows_c3_ub[(min(_NUM_SPC, groups),)](
                values,
                weights,
                out,
                d,
                groups,
                m - 4,
                eps,
                CHUNK=chunk,
                NPROG=min(_NUM_SPC, groups),
                U=4,
                num_warps=1,
            )
        elif (
            arm == _ARM_EXACT
            and block <= _UB_EXACT8_MAX_BLOCK
            and m >= _UB_EXACT8_ROWS
        ):
            groups = (m + 7) // 8
            _rms_norm_rows_exact_ub[(min(_NUM_SPC, groups),)](
                values,
                weights,
                out,
                d,
                groups,
                m - 8,
                eps,
                B=block,
                NPROG=min(_NUM_SPC, groups),
                U=8,
                num_warps=1,
            )
        elif (
            arm == _ARM_EXACT
            and block == _UB_EXACT2_BLOCK
            and m >= _UB_EXACT2_ROWS
        ):
            groups = (m + 1) // 2
            _rms_norm_rows_exact_ub[(min(_NUM_SPC, groups),)](
                values,
                weights,
                out,
                d,
                groups,
                m - 2,
                eps,
                B=block,
                NPROG=min(_NUM_SPC, groups),
                U=2,
                num_warps=1,
            )
        else:
            grid = (min(_NUM_SPC, m),)
            _rms_norm_row[grid](
                values,
                weights,
                out,
                d,
                m,
                eps,
                B=block,
                CHUNK=chunk,
                NPROG=grid[0],
                ARM=arm,
                num_warps=_rows_warps(arm, block),
            )
    else:
        _rms_norm_whole_row[(m,)](
            values,
            weights,
            out,
            d,
            eps,
            B=block,
            CHUNK=chunk,
            ARM=arm,
            num_warps=_whole_row_warps(arm, block),
        )
    return out


__all__ = ["one_pass_rms_norm"]
