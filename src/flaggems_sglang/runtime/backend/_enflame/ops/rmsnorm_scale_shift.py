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
# implied. See the License for the specific language governing
# permissions and limitations under the License.

"""Fused RMSNorm with per-batch scale and shift.

Inputs: x [B, S, D], weight [D], scale/shift [B, 1, D].
Rows are padded to the next power of two, so any positive D works. This
GCU copy reduces the sum of squares with a plain tree sum instead of
FlashInfer's thread mapping: tl.gather is not implemented on the GCU
backend, and the mapping only fixed the fp32 summation order.

On the GCU the dense large-M path runs a row-batched persistent kernel:
one program covers R consecutive rows over a grid-stride block loop, so
the per-row program dispatch of the fallback kernel (~3us per 48-program
wave, M=4096 rows = 85 waves ~= 260us) is amortized, and x is loaded once
per row instead of twice. All memory ops stay strictly 1-D: 2-D tile
formulations of this reduction family fail to compile or silently
miscompile on this backend (compiler_issues/vllm_silu_flat_slice_start.md,
compiler_issues/vllm_fused_q_kv_rmsnorm_rowbatch_bcast_miscompile.md).
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _refined_rsqrt(variance, EPS: tl.constexpr):
    # GCU: scalar tl.fma lowers to an out-of-line tops_fma_fp32 libcall that
    # ld.lld cannot relocate (R_DTU_ADDR16_*_ICALL). Plain rsqrt is ~1e-7
    # relative, far inside the bf16 tolerance; every passing rmsnorm sibling
    # on GCU uses it.
    return tl.rsqrt(variance)


@triton.jit
def _scale_shift(norm, scale, shift, out_ptr, scale_ptr, IS_HIP: tl.constexpr):
    if IS_HIP:
        norm = norm.to(out_ptr.dtype.element_ty).to(tl.float32)
        gain = (scale + 1.0).to(scale_ptr.dtype.element_ty).to(tl.float32)
        product = (norm * gain).to(out_ptr.dtype.element_ty).to(tl.float32)
        return product + shift
    else:
        # Keep the BF16 rounding stages explicit, without inline PTX.
        norm = norm.to(tl.bfloat16).to(tl.float32)
        scale = scale.to(tl.bfloat16).to(tl.float32)
        shift = shift.to(tl.bfloat16).to(tl.float32)
        gain = (scale + 1.0).to(tl.bfloat16).to(tl.float32)
        product = (norm * gain).to(tl.bfloat16).to(tl.float32)
        return (product + shift).to(tl.bfloat16)


@triton.jit
def _rmsnorm_scale_shift_kernel(
    x_ptr,
    weight_ptr,
    scale_ptr,
    shift_ptr,
    out_ptr,
    SEQ_LEN: tl.constexpr,
    D: tl.constexpr,
    X_STRIDES: tl.constexpr,
    W_S: tl.constexpr,
    SCALE_STRIDES: tl.constexpr,
    SHIFT_STRIDES: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK_D: tl.constexpr,
    IS_HIP: tl.constexpr,
    USE_CG: tl.constexpr,
):
    row = tl.program_id(0)
    batch = row // SEQ_LEN
    seq = row % SEQ_LEN
    ptr = x_ptr + batch * X_STRIDES[0] + seq * X_STRIDES[1]
    q = tl.arange(0, 128)
    accum = tl.full((128,), 0, tl.float32)
    for start in tl.static_range(0, D, 128):
        if start + 128 <= D:
            v = tl.load(
                ptr + (start + q) * X_STRIDES[2],
                cache_modifier=".cg" if USE_CG else "",
            ).to(tl.float32)
        else:
            v = tl.load(
                ptr + (start + q) * X_STRIDES[2],
                start + q < D,
                other=0,
                cache_modifier=".cg" if USE_CG else "",
            ).to(tl.float32)
        accum = accum + v * v
    # GCU: tl.gather is not implemented (tt.gather illegal, PassManager fails),
    # so this copy reduces with a plain tree sum. The FlashInfer lane
    # permutation only fixed the fp32 summation order; the deviation is far
    # below the bf16 comparison tolerance.
    variance = tl.sum(accum, 0) / D + EPS
    inv = _refined_rsqrt(variance, EPS)
    cols = tl.arange(0, BLOCK_D)
    x = tl.load(
        ptr + cols * X_STRIDES[2],
        cols < D,
        other=0,
        cache_modifier=".cg" if USE_CG else "",
    ).to(tl.float32)
    w = tl.load(
        weight_ptr + cols * W_S,
        cols < D,
        other=0,
        eviction_policy="" if IS_HIP else "evict_last",
    ).to(tl.float32)
    scale = tl.load(
        scale_ptr + batch * SCALE_STRIDES[0] + cols * SCALE_STRIDES[2],
        cols < D,
        other=0,
        eviction_policy="" if IS_HIP else "evict_last",
    ).to(tl.float32)
    shift = tl.load(
        shift_ptr + batch * SHIFT_STRIDES[0] + cols * SHIFT_STRIDES[2],
        cols < D,
        other=0,
        eviction_policy="" if IS_HIP else "evict_last",
    ).to(tl.float32)
    value = _scale_shift(x * inv * w, scale, shift, out_ptr, scale_ptr, IS_HIP)
    tl.store(
        out_ptr + row * D + cols,
        value,
        cols < D,
        cache_modifier="" if IS_HIP else ".cs",
    )


@triton.jit
def _rmsnorm_scale_shift_rows_kernel(
    x_ptr,
    weight_ptr,
    scale_ptr,
    shift_ptr,
    out_ptr,
    M,
    N,
    SEQ_LEN,
    NUM_BLOCKS,
    eps,
    R: tl.constexpr,
    BLOCK_N: tl.constexpr,
    NPROG: tl.constexpr,
    EXACT_COLS: tl.constexpr,
    BATCH1: tl.constexpr,
    NS: tl.constexpr,
):
    # GCU row-batched persistent kernel (see module docstring). The R row
    # chains inside a block are independent, so the 2-stage pipeline over
    # blocks overlaps the next block's loads with this block's reduction
    # trees and stores. The bf16 rounding chain is shared with the
    # fallback kernel via _scale_shift. The tail block slides back to
    # re-write the final R rows (idempotent overwrite: a row's value only
    # depends on its own inputs), so no row mask is ever needed -- the
    # masked-row arm measured >=8x slower than the clean arm on odd-M
    # shapes. scale/shift only vary with the batch index, so they are
    # loaded once per program (BATCH1) or once per block (the dispatcher
    # guarantees every block is batch-uniform: R divides SEQ_LEN, or R=1);
    # the per-row reload they replace cost ~18% on large-M shapes.
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N)
    if EXACT_COLS:
        w = tl.load(weight_ptr + offs).to(tl.float32)
    else:
        w = tl.load(weight_ptr + offs, mask=offs < N, other=0.0).to(tl.float32)
    if BATCH1:
        if EXACT_COLS:
            sc = tl.load(scale_ptr + offs).to(tl.float32)
            sh = tl.load(shift_ptr + offs).to(tl.float32)
        else:
            m0 = offs < N
            sc = tl.load(scale_ptr + offs, mask=m0, other=0.0).to(tl.float32)
            sh = tl.load(shift_ptr + offs, mask=m0, other=0.0).to(tl.float32)
    for blk in tl.range(pid, NUM_BLOCKS, NPROG, num_stages=NS):
        r0 = tl.minimum(blk * R, M - R)
        if not BATCH1:
            b = r0 // SEQ_LEN
            if EXACT_COLS:
                sc = tl.load(scale_ptr + b * N + offs).to(tl.float32)
                sh = tl.load(shift_ptr + b * N + offs).to(tl.float32)
            else:
                m0 = offs < N
                sc = tl.load(scale_ptr + b * N + offs, mask=m0, other=0.0).to(
                    tl.float32
                )
                sh = tl.load(shift_ptr + b * N + offs, mask=m0, other=0.0).to(
                    tl.float32
                )
        for r in tl.static_range(R):
            row = r0 + r
            if EXACT_COLS:
                x = tl.load(x_ptr + row * N + offs).to(tl.float32)
                variance = tl.sum(x * x, axis=0) / N + eps
                inv = tl.rsqrt(variance)
                value = _scale_shift(x * inv * w, sc, sh, out_ptr, scale_ptr, False)
                tl.store(out_ptr + row * N + offs, value)
            else:
                m = offs < N
                x = tl.load(x_ptr + row * N + offs, mask=m, other=0.0).to(
                    tl.float32
                )
                variance = tl.sum(x * x, axis=0) / N + eps
                inv = tl.rsqrt(variance)
                value = _scale_shift(x * inv * w, sc, sh, out_ptr, scale_ptr, False)
                tl.store(out_ptr + row * N + offs, value, mask=m)


@triton.jit
def _split_2048_chunks(x, BLOCK_ROWS: tl.constexpr):
    # Original split layout; tuple order is ascending 128-column chunks.
    group_0 = tl.trans(tl.reshape(x, (BLOCK_ROWS, 16, 128)), (0, 2, 1))
    (group_1, group_2) = tl.split(tl.reshape(group_0, (BLOCK_ROWS, 128, 8, 2)))
    (group_3, group_4) = tl.split(tl.reshape(group_1, (BLOCK_ROWS, 128, 4, 2)))
    (group_5, group_6) = tl.split(tl.reshape(group_3, (BLOCK_ROWS, 128, 2, 2)))
    (chunk_0, chunk_8) = tl.split(group_5)
    (chunk_4, chunk_12) = tl.split(group_6)
    (group_7, group_8) = tl.split(tl.reshape(group_4, (BLOCK_ROWS, 128, 2, 2)))
    (chunk_2, chunk_10) = tl.split(group_7)
    (chunk_6, chunk_14) = tl.split(group_8)
    (group_9, group_10) = tl.split(
        tl.reshape(group_2, (BLOCK_ROWS, 128, 4, 2))
    )
    (group_11, group_12) = tl.split(
        tl.reshape(group_9, (BLOCK_ROWS, 128, 2, 2))
    )
    (chunk_1, chunk_9) = tl.split(group_11)
    (chunk_5, chunk_13) = tl.split(group_12)
    (group_13, group_14) = tl.split(
        tl.reshape(group_10, (BLOCK_ROWS, 128, 2, 2))
    )
    (chunk_3, chunk_11) = tl.split(group_13)
    (chunk_7, chunk_15) = tl.split(group_14)
    return (
        chunk_0,
        chunk_1,
        chunk_2,
        chunk_3,
        chunk_4,
        chunk_5,
        chunk_6,
        chunk_7,
        chunk_8,
        chunk_9,
        chunk_10,
        chunk_11,
        chunk_12,
        chunk_13,
        chunk_14,
        chunk_15,
    )


@triton.jit
def _rmsnorm_scale_shift_dense_split_kernel(
    x_ptr,
    weight_ptr,
    scale_ptr,
    shift_ptr,
    out_ptr,
    SEQ_LEN: tl.constexpr,
    D: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
):
    row = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    batch = tl.program_id(1)
    ptr = x_ptr + batch * (SEQ_LEN * D) + row[:, None] * D
    cols = tl.arange(0, BLOCK_D)
    mask = (row[:, None] < SEQ_LEN) & (cols[None, :] < D)
    x = tl.load(ptr + cols[None, :], mask, other=0, cache_modifier=".cg").to(
        tl.float32
    )
    # Static tuple indexing keeps the original split DAG and addition order.
    chunks = _split_2048_chunks(x, BLOCK_ROWS)
    accum = tl.full((BLOCK_ROWS, 128), 0, tl.float32)
    for i in tl.static_range(16):
        v = chunks[i]
        accum = accum + v * v

    # Keep the original grouping and HIP lane permutation before the final sum.
    even, odd = tl.split(tl.reshape(accum, (BLOCK_ROWS, 32, 2, 2)))
    a0, a2 = tl.split(even)
    a1, a3 = tl.split(odd)
    acc = a0 + a1 + a2 + a3
    lane = tl.arange(0, 32)
    reverse = (
        (lane & 1) << 3
        | (lane & 2) << 1
        | (lane & 4) >> 1
        | (lane & 8) >> 3
        | lane & 16
    )
    acc = tl.gather(
        acc, tl.broadcast_to(reverse[None, :], (BLOCK_ROWS, 32)), axis=1
    )
    variance = tl.sum(acc, axis=1) / D + EPS
    inv = _refined_rsqrt(variance, EPS)
    w = tl.load(
        weight_ptr + cols,
        cols < D,
        other=0,
    ).to(tl.float32)
    scale = tl.load(
        scale_ptr + batch * D + cols,
        cols < D,
        other=0,
    ).to(tl.float32)
    shift = tl.load(
        shift_ptr + batch * D + cols,
        cols < D,
        other=0,
    ).to(tl.float32)
    value = _scale_shift(
        x * inv[:, None] * w[None, :],
        scale[None, :],
        shift[None, :],
        out_ptr,
        scale_ptr,
        True,
    )
    tl.store(
        out_ptr + (batch * SEQ_LEN + row[:, None]) * D + cols[None, :],
        value,
        mask,
        cache_modifier=".cs",
    )


# Dispatch calibration for the GCU row-batched kernel (gcu400/Libra-class
# 48 SPs, bf16, triton.testing.do_bench kernel mode, warmup 1000ms /
# rep 100ms median, 2026-11-08), mirroring the fused_rmsnorm sibling:
#   - num_warps=1 wins for every exact-width load <= 4096 (shuffle-only
#     reduction tree, no inter-warp DSM hop); masked columns need nw2.
#   - R=2 for BLOCK_N >= 2048, R=4 for multi-batch rows >= 2048 with
#     4 | SEQ_LEN (amortizes the per-block scale/shift load, see
#     _rows_config); below 64 rows the launch-latency-bound whole-row
#     fallback kernel is kept.
#   - num_stages=2 on the block loop (3 measured within noise).
#   - Above 8192 columns the fallback kernel keeps handling the row in
#     128-wide chunks (register footprint of a full padded row).
_ROWS_MIN_ROWS = 64
_MAX_BLOCK_N = 8192
_NUM_SPC = 48


def _rows_config(row_count, block_size, exact_cols, seq_len, batch1):
    if block_size <= 1024:
        rows_per_cta = 4 if row_count >= 256 else 2
    else:
        rows_per_cta = 2
        # Multi-batch shapes reload scale/shift once per block, so wider
        # blocks amortize that load: R=4 measured -9..-11% for rows >= 2048
        # with 4 | SEQ_LEN ((2,4096,2048) 201.7->183.5, (8,4096,2048)
        # 793.2->714.6, (2,2048,4096) 138.2->125.6us). Batch-1 shapes keep
        # R=2: their scale/shift is hoisted out of the loop entirely, so
        # the amortization does not apply and R=4 measured +4..5%
        # ((1,4096) 87.8->91.4, (1,1024) 25.2->26.5us).
        if (
            not batch1
            and row_count >= 2048
            and seq_len % 4 == 0
            and exact_cols
            and block_size <= 4096
        ):
            rows_per_cta = 4
    # Blocks must stay batch-uniform so scale/shift load once per block:
    # single-row blocks whenever R would straddle a batch edge (batch==1
    # is uniform by construction).
    if not batch1 and seq_len % rows_per_cta != 0:
        rows_per_cta = 1
    if exact_cols:
        num_warps = 1 if block_size <= 4096 else 4
    else:
        num_warps = 2 if block_size <= 4096 else 4
    return rows_per_cta, num_warps


def rmsnorm_scale_shift(x, weight, scale, shift, eps):
    batch, seq_len, hidden_size = x.shape
    rows = batch * seq_len
    is_hip = torch.version.hip is not None
    dense = all(t.is_contiguous() for t in (x, weight, scale, shift))
    out = torch.empty(x.shape, dtype=x.dtype, device=x.device)
    if rows == 0:
        return out

    block_size = triton.next_power_of_2(hidden_size)
    if (
        not is_hip
        and dense
        and rows >= _ROWS_MIN_ROWS
        and block_size <= _MAX_BLOCK_N
    ):
        exact_cols = block_size == hidden_size
        batch1 = batch == 1
        rows_per_cta, num_warps = _rows_config(
            rows, block_size, exact_cols, seq_len, batch1
        )
        num_blocks = triton.cdiv(rows, rows_per_cta)
        grid = (min(_NUM_SPC, num_blocks),)
        _rmsnorm_scale_shift_rows_kernel[grid](
            x,
            weight,
            scale,
            shift,
            out,
            rows,
            hidden_size,
            seq_len,
            num_blocks,
            float(eps),
            R=rows_per_cta,
            BLOCK_N=block_size,
            NPROG=grid[0],
            EXACT_COLS=exact_cols,
            BATCH1=batch1,
            NS=2,
            num_warps=num_warps,
            enable_fp_fusion=False,
        )
    elif dense and is_hip and hidden_size == 2048 and rows >= 2048:
        _rmsnorm_scale_shift_dense_split_kernel[
            triton.cdiv(seq_len, 4), batch
        ](
            x,
            weight,
            scale,
            shift,
            out,
            seq_len,
            hidden_size,
            float(eps),
            hidden_size,
            4,
            num_warps=8 if rows > 16384 else 4,
            num_stages=1,
            enable_fp_fusion=False,
        )
    else:
        # constexpr strides specialize to the same addressing for dense inputs.
        _rmsnorm_scale_shift_kernel[rows,](
            x,
            weight,
            scale,
            shift,
            out,
            seq_len,
            hidden_size,
            tuple(x.stride()),
            weight.stride(0),
            tuple(scale.stride()),
            tuple(shift.stride()),
            float(eps),
            triton.next_power_of_2(hidden_size),
            is_hip,
            is_hip and rows <= 16384,
            num_warps=8 if is_hip else 4,
            num_stages=1,
            enable_fp_fusion=False,
        )
    return out


__all__ = ["rmsnorm_scale_shift"]
