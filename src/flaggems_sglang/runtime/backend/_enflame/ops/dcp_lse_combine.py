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

"""GCU rewrite of dcp_lse_combine.

The generic kernel uses 2D [BLOCK_M, BLOCK_D] tile loads/stores, which the
current GCU toolchain fails to compile (memref_ext.slice_(pad_)start
operandSegmentSizes 8 vs 7; every rank-2 store and every unmasked rank-2
load triggers it -- see compiler_issues/vllm_silu_flat_slice_start.md and
compiler_issues/sglang_qk_head_layernorm.md). Beyond compiling again, the
rewrite targets the GCU memory system directly:

- Row groups are processed as single flat 1D spans ([ROWS, D] with power-of-
  two D is contiguous in memory), loaded wide and reshaped in registers for
  the per-row weight broadcast. Per-row 256B-class accesses top out ~15x
  below bandwidth on GCU400 (qk_head_layernorm precedent).
- The N (data-parallel) axis is a constexpr static_range unroll, so all N
  flat loads issue independently and overlap.
- recv_lse rows are read with [ROWS]-wide 1D affine loads; NaN/+inf shards
  are filtered exactly like the reference before the max/exp math.
- All stores are 1D (out tile flat, out_lse per row group).

Non-power-of-two head dims and tail row groups fall back to a strictly 1D
per-row body (masked loads), which never touches the broken lowering path.

Second round (2026-11-12 E-1): n == 8 shapes with m % rows == 0 dispatch to
_dcp_lse_combine_flat — the same flat body without the never-taken tail
fallback, whose dead else arm drags the flat loop lowering for the N=8
constexpr combo on this toolchain (-6.1% at the (8, 64, 16, 128) core shape;
N=4 stays on the generic kernel where the dead branch is neutral or better).
n == 2 shapes dispatch to _dcp_lse_combine_pair (loads issued ahead of the
lse math, product-start accumulator; ~4-6% at (2, 64, 16, 128), at most one
bf16 ulp from the zeros-start form).
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _dcp_lse_combine_pair(
    out_ptr,
    out_lse_ptr,
    recv_out_ptr,
    recv_lse_ptr,
    M,
    D,
    stride_rn,
    stride_ln,
    IS_E: tl.constexpr,
    RETURN_LSE: tl.constexpr,
    N: tl.constexpr,
    ROWS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # N == 2 unrolled arm (host guarantees N == 2 and m % ROWS == 0): both
    # flat span loads issue before the lse math, and the accumulator starts
    # from the w0 * x0 product instead of a zeros tensor. On the 2026-11-12
    # toolchain this exact lowering form runs the (2, 64, 16, 128) core
    # shape ~4-6% faster than the generic interleaved loop; the zeros-start
    # form and the generic form are bitwise identical, while this form can
    # differ by at most one bf16 ulp on rounding-boundary outputs (fp32
    # lse path is bitwise identical; well inside the op tolerances).
    pid = tl.program_id(0)
    n_groups = tl.cdiv(M, ROWS)
    for grp in range(pid, n_groups, tl.num_programs(0)):
        base = grp * ROWS
        if base + ROWS <= M:
            row_offs = tl.arange(0, ROWS)
            flat_offs = tl.arange(0, ROWS * BLOCK)
            rbase = base * D
            x0 = tl.load(
                recv_out_ptr + rbase + flat_offs, cache_modifier=".cg"
            )
            x1 = tl.load(
                recv_out_ptr + stride_rn + rbase + flat_offs,
                cache_modifier=".cg",
            )
            l0 = tl.load(recv_lse_ptr + base + row_offs)
            l0 = tl.where((l0 != l0) | (l0 == float("inf")), float("-inf"), l0)
            l1 = tl.load(recv_lse_ptr + stride_ln + base + row_offs)
            l1 = tl.where((l1 != l1) | (l1 == float("inf")), float("-inf"), l1)
            m_max = tl.maximum(l0, l1)
            m_max = tl.where(m_max == float("-inf"), 0.0, m_max)
            if IS_E:
                w0 = tl.exp(l0 - m_max)
                w1 = tl.exp(l1 - m_max)
            else:
                w0 = tl.exp2(l0 - m_max)
                w1 = tl.exp2(l1 - m_max)
            wsum = w0 + w1
            acc = w0[:, None] * tl.reshape(x0, (ROWS, BLOCK)).to(tl.float32)
            acc += w1[:, None] * tl.reshape(x1, (ROWS, BLOCK)).to(tl.float32)
            res = acc / wsum[:, None]
            tl.store(
                out_ptr + rbase + flat_offs,
                tl.reshape(res.to(out_ptr.dtype.element_ty), (ROWS * BLOCK,)),
                cache_modifier=".cg",
            )
            if RETURN_LSE:
                if IS_E:
                    merged = tl.log(wsum) + m_max
                else:
                    merged = tl.log2(wsum) + m_max
                tl.store(
                    out_lse_ptr + base + row_offs,
                    merged.to(out_lse_ptr.dtype.element_ty),
                )


# Flat-span element budget: 1D tile accesses above this fail to compile on
# the current toolchain (per-warp lanes must stay <= 4096; num_warps>1 fails
# to compile for this kernel family on the 2026-11-07 toolchain -- see
# compiler_issues/vllm_silu_flat_slice_start.md). 4096 probed OK on GCU400.
_FLAT_ELEMS_MAX = 4096
# Per-row fallback chunk width (1D masked), bounded by the 32KB slice rule.
_ROW_BLOCK_MAX = 16384
# Rows per group in the per-row fallback arm.
_ROW_GROUP_ROWS = 8
# Total unrolled span elements (N * rows * block) kept at or below this:
# above it the in-flight flat loads overshoot and regress (measured: n=8
# rows=32 is 2.2x slower than rows=16 at block=128).
_SPAN_ELEMS_TARGET = 16384
# Persistent grid-stride cap: keeps CTA dispatch cost (~52ns each) flat for
# large row counts and measured best for the many-group shapes.
_GRID_CAP = 48


@triton.jit
def _dcp_lse_combine_row(
    out_ptr,
    out_lse_ptr,
    recv_out_ptr,
    recv_lse_ptr,
    row,
    D,
    stride_rn,
    stride_ln,
    IS_E: tl.constexpr,
    RETURN_LSE: tl.constexpr,
    N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    # Single row, strictly 1D accesses. Scalar weight chain + one masked
    # [BLOCK_D] chunk loop over D.
    m_max = float("-inf")
    for n in tl.static_range(0, N):
        lse_n = tl.load(recv_lse_ptr + n * stride_ln + row)
        lse_n = tl.where(
            (lse_n != lse_n) | (lse_n == float("inf")),
            float("-inf"),
            lse_n,
        )
        m_max = tl.maximum(m_max, lse_n)
    m_max = tl.where(m_max == float("-inf"), 0.0, m_max)

    wsum = 0.0
    for n in tl.static_range(0, N):
        lse_n = tl.load(recv_lse_ptr + n * stride_ln + row)
        lse_n = tl.where(
            (lse_n != lse_n) | (lse_n == float("inf")),
            float("-inf"),
            lse_n,
        )
        if IS_E:
            wsum += tl.exp(lse_n - m_max)
        else:
            wsum += tl.exp2(lse_n - m_max)

    cols = tl.arange(0, BLOCK_D)
    mask_d = cols < D
    for d0 in range(0, D, BLOCK_D):
        offs = row * D + d0 + cols
        acc = tl.zeros([BLOCK_D], dtype=tl.float32)
        for n in tl.static_range(0, N):
            lse_n = tl.load(recv_lse_ptr + n * stride_ln + row)
            lse_n = tl.where(
                (lse_n != lse_n) | (lse_n == float("inf")),
                float("-inf"),
                lse_n,
            )
            if IS_E:
                w_n = tl.exp(lse_n - m_max)
            else:
                w_n = tl.exp2(lse_n - m_max)
            x = tl.load(
                recv_out_ptr + n * stride_rn + offs,
                mask=mask_d,
                other=0.0,
            )
            acc += w_n * x.to(tl.float32)
        tl.store(
            out_ptr + offs,
            (acc / wsum).to(out_ptr.dtype.element_ty),
            mask=mask_d,
        )

    if RETURN_LSE:
        if IS_E:
            merged = tl.log(wsum) + m_max
        else:
            merged = tl.log2(wsum) + m_max
        tl.store(out_lse_ptr + row, merged.to(out_lse_ptr.dtype.element_ty))


@triton.jit
def _dcp_lse_combine_flat(
    out_ptr,
    out_lse_ptr,
    recv_out_ptr,
    recv_lse_ptr,
    M,
    D,
    stride_rn,
    stride_ln,
    IS_E: tl.constexpr,
    RETURN_LSE: tl.constexpr,
    N: tl.constexpr,
    ROWS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # Full-group-only arm of _dcp_lse_combine_kernel (host guarantees
    # m % ROWS == 0), without the per-row tail fallback. Dispatched for
    # n == 8 only: on the 2026-11-12 toolchain the never-taken tail else
    # inside the shared kernel drags the flat main loop's lowering for the
    # N=8/rows=16 constexpr combo (10.27 -> 9.64 us at the (8, 64, 16, 128)
    # core shape, -6.1%), while the same dead branch is neutral-or-better
    # for the N=2/4 combos (so those keep the generic kernel; same
    # dead-branch mechanism as the vetoed runtime-false tail branches in
    # perf_playbook).
    pid = tl.program_id(0)
    n_groups = tl.cdiv(M, ROWS)
    for grp in range(pid, n_groups, tl.num_programs(0)):
        base = grp * ROWS
        if base + ROWS <= M:
            row_offs = tl.arange(0, ROWS)

            m_max = tl.full([ROWS], float("-inf"), tl.float32)
            for n in tl.static_range(0, N):
                lse_n = tl.load(recv_lse_ptr + n * stride_ln + base + row_offs)
                lse_n = tl.where(
                    (lse_n != lse_n) | (lse_n == float("inf")),
                    float("-inf"),
                    lse_n,
                )
                m_max = tl.maximum(m_max, lse_n)
            m_max = tl.where(m_max == float("-inf"), 0.0, m_max)

            wsum = tl.zeros([ROWS], dtype=tl.float32)
            acc = tl.zeros([ROWS, BLOCK], dtype=tl.float32)
            flat_offs = tl.arange(0, ROWS * BLOCK)
            for n in tl.static_range(0, N):
                lse_n = tl.load(recv_lse_ptr + n * stride_ln + base + row_offs)
                lse_n = tl.where(
                    (lse_n != lse_n) | (lse_n == float("inf")),
                    float("-inf"),
                    lse_n,
                )
                if IS_E:
                    w_n = tl.exp(lse_n - m_max)
                else:
                    w_n = tl.exp2(lse_n - m_max)
                wsum += w_n
                x = tl.load(
                    recv_out_ptr + n * stride_rn + base * D + flat_offs,
                    cache_modifier=".cg",
                )
                acc += w_n[:, None] * tl.reshape(x, (ROWS, BLOCK)).to(
                    tl.float32
                )

            out = tl.reshape(
                (acc / wsum[:, None]).to(out_ptr.dtype.element_ty),
                (ROWS * BLOCK,),
            )
            tl.store(
                out_ptr + base * D + flat_offs,
                out,
                cache_modifier=".cg",
            )
            if RETURN_LSE:
                if IS_E:
                    merged = tl.log(wsum) + m_max
                else:
                    merged = tl.log2(wsum) + m_max
                tl.store(
                    out_lse_ptr + base + row_offs,
                    merged.to(out_lse_ptr.dtype.element_ty),
                )


@triton.jit
def _dcp_lse_combine_kernel(
    out_ptr,
    out_lse_ptr,
    recv_out_ptr,
    recv_lse_ptr,
    M,
    D,
    stride_rn,
    stride_ln,
    IS_E: tl.constexpr,
    RETURN_LSE: tl.constexpr,
    N: tl.constexpr,
    ROWS: tl.constexpr,
    BLOCK: tl.constexpr,
    ROW_BLOCK: tl.constexpr,
    FLAT: tl.constexpr,
):
    pid = tl.program_id(0)
    n_groups = tl.cdiv(M, ROWS)
    for grp in range(pid, n_groups, tl.num_programs(0)):
        base = grp * ROWS
        if FLAT and base + ROWS <= M:
            # Full row group with power-of-two D: the ROWS rows form one
            # contiguous [ROWS * D] span per split n. Load it flat, reshape
            # in registers, broadcast the per-row weight down the columns.
            row_offs = tl.arange(0, ROWS)

            m_max = tl.full([ROWS], float("-inf"), tl.float32)
            for n in tl.static_range(0, N):
                lse_n = tl.load(recv_lse_ptr + n * stride_ln + base + row_offs)
                lse_n = tl.where(
                    (lse_n != lse_n) | (lse_n == float("inf")),
                    float("-inf"),
                    lse_n,
                )
                m_max = tl.maximum(m_max, lse_n)
            m_max = tl.where(m_max == float("-inf"), 0.0, m_max)

            wsum = tl.zeros([ROWS], dtype=tl.float32)
            acc = tl.zeros([ROWS, BLOCK], dtype=tl.float32)
            flat_offs = tl.arange(0, ROWS * BLOCK)
            for n in tl.static_range(0, N):
                lse_n = tl.load(recv_lse_ptr + n * stride_ln + base + row_offs)
                lse_n = tl.where(
                    (lse_n != lse_n) | (lse_n == float("inf")),
                    float("-inf"),
                    lse_n,
                )
                if IS_E:
                    w_n = tl.exp(lse_n - m_max)
                else:
                    w_n = tl.exp2(lse_n - m_max)
                wsum += w_n
                x = tl.load(
                    recv_out_ptr + n * stride_rn + base * D + flat_offs,
                    cache_modifier=".cg",
                )
                acc += w_n[:, None] * tl.reshape(x, (ROWS, BLOCK)).to(
                    tl.float32
                )

            out = tl.reshape(
                (acc / wsum[:, None]).to(out_ptr.dtype.element_ty),
                (ROWS * BLOCK,),
            )
            tl.store(
                out_ptr + base * D + flat_offs,
                out,
                cache_modifier=".cg",
            )
            if RETURN_LSE:
                if IS_E:
                    merged = tl.log(wsum) + m_max
                else:
                    merged = tl.log2(wsum) + m_max
                tl.store(
                    out_lse_ptr + base + row_offs,
                    merged.to(out_lse_ptr.dtype.element_ty),
                )
        else:
            # Tail group or non-power-of-two D: strictly 1D per-row body.
            for r in tl.static_range(0, ROWS):
                row = base + r
                if row < M:
                    _dcp_lse_combine_row(
                        out_ptr,
                        out_lse_ptr,
                        recv_out_ptr,
                        recv_lse_ptr,
                        row,
                        D,
                        stride_rn,
                        stride_ln,
                        IS_E,
                        RETURN_LSE,
                        N,
                        ROW_BLOCK,
                    )


def dcp_lse_combine(recv_output, recv_lse, is_lse_base_on_e, return_lse):
    recv_output = recv_output.contiguous()
    recv_lse = recv_lse.contiguous()

    n, b, h, d = recv_output.shape
    m = b * h

    out = torch.empty(
        (b, h, d), dtype=recv_output.dtype, device=recv_output.device
    )
    out_lse = (
        torch.empty((b, h), dtype=recv_lse.dtype, device=recv_lse.device)
        if return_lse
        else None
    )
    if m == 0:
        return out, out_lse

    block = triton.next_power_of_2(d)
    row_block = min(block, _ROW_BLOCK_MAX)
    flat = block == d and block >= 4 and block <= _FLAT_ELEMS_MAX
    if flat:
        # Balance span width against the unrolled in-flight loads: keep
        # N * rows * block near _SPAN_ELEMS_TARGET, then keep at least ~8
        # row groups for small m so tiny shapes stay parallel.
        rows = min(_SPAN_ELEMS_TARGET // (n * block), _FLAT_ELEMS_MAX // block)
        rows = max(2, min(rows, 1 << (max(m, 2).bit_length() - 1)))
        while rows > 2 and triton.cdiv(m, rows) < 8:
            rows //= 2
    else:
        rows = _ROW_GROUP_ROWS

    n_groups = triton.cdiv(m, rows)
    grid = (min(n_groups, _GRID_CAP),)
    if flat and n == 2 and m % rows == 0:
        _dcp_lse_combine_pair[grid](
            out,
            out_lse if out_lse is not None else out,
            recv_output,
            recv_lse,
            m,
            d,
            recv_output.stride(0),
            recv_lse.stride(0),
            IS_E=bool(is_lse_base_on_e),
            RETURN_LSE=bool(return_lse),
            N=n,
            ROWS=rows,
            BLOCK=block,
            num_warps=1,
        )
        return out, out_lse
    if flat and n == 8 and m % rows == 0:
        _dcp_lse_combine_flat[grid](
            out,
            out_lse if out_lse is not None else out,
            recv_output,
            recv_lse,
            m,
            d,
            recv_output.stride(0),
            recv_lse.stride(0),
            IS_E=bool(is_lse_base_on_e),
            RETURN_LSE=bool(return_lse),
            N=n,
            ROWS=rows,
            BLOCK=block,
            num_warps=1,
        )
        return out, out_lse
    _dcp_lse_combine_kernel[grid](
        out,
        out_lse if out_lse is not None else out,
        recv_output,
        recv_lse,
        m,
        d,
        recv_output.stride(0),
        recv_lse.stride(0),
        IS_E=bool(is_lse_base_on_e),
        RETURN_LSE=bool(return_lse),
        N=n,
        ROWS=rows,
        BLOCK=block,
        ROW_BLOCK=row_block,
        FLAT=bool(flat),
        num_warps=1,
    )
    return out, out_lse


__all__ = ["dcp_lse_combine"]
