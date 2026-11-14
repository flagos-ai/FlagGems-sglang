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

import torch
import triton
import triton.language as tl

# Fresh-compile boundaries on the current GCU400 stack (2026-11-08,
# gcu-compiler-opt slice_(pad_)start 8-vs-7 lockout): a 1D flat kernel
# compiles only with per-CTA slice <= 32KB AND <= 4096 elements per warp
# (bf16 16384@nw4 OK, 16384@nw2 / 32768@any FAIL). The block budgets below
# keep every dispatched configuration inside that envelope for all dtypes.
_CTA_BYTE_BUDGET = 32768
_MAX_LANES = 16384
_LANES_PER_WARP_CAP = 4096


@triton.jit
def _fill_padded_rows_copyfirst_kernel(
    x_ptr,
    out_ptr,
    n_ptr,
    total,
    fill_value,
    NCOLS: tl.constexpr,
    BLOCK: tl.constexpr,
    EXACT: tl.constexpr,
):
    # Copy-first split: the whole-tensor copy is issued without waiting for
    # the device-read pad threshold, so the scalar-load latency of ``n``
    # overlaps the copy traffic instead of sitting on the critical path.
    # Only the fill store waits for ``n``. The copy does read the
    # soon-to-be-padded region and writes it twice; the multi-CTA grid
    # keeps that extra traffic cheap enough that this form stays the
    # fastest of masked / where / copy-first at every measured size.
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    if EXACT:
        xv = tl.load(x_ptr + offs)
        tl.store(out_ptr + offs, xv)
        thresh = tl.load(n_ptr) * NCOLS
        fv = tl.full([BLOCK], fill_value, tl.float32).to(out_ptr.dtype.element_ty)
        tl.store(out_ptr + offs, fv, mask=offs >= thresh)
    else:
        in_bounds = offs < total
        xv = tl.load(x_ptr + offs, mask=in_bounds, other=fill_value)
        tl.store(out_ptr + offs, xv, mask=in_bounds)
        thresh = tl.load(n_ptr) * NCOLS
        fv = tl.full([BLOCK], fill_value, tl.float32).to(out_ptr.dtype.element_ty)
        tl.store(out_ptr + offs, fv, mask=in_bounds & (offs >= thresh))


@triton.jit
def _fill_padded_rows_strided_rows_kernel(
    x_ptr,
    out_ptr,
    n_ptr,
    n_rows,
    n_cols,
    x_row_stride,
    out_row_stride,
    fill_value,
    BLOCK_C: tl.constexpr,
    ROWS_PER_PROG: tl.constexpr,
):
    # Row-strided inputs (stride(1) == 1 per the op contract): one program
    # covers ROWS_PER_PROG rows with per-row 1D column chunks and explicit
    # row strides. Rank-2 broadcast-offset tiles are avoided on purpose --
    # they fail fresh compilation on the current stack -- and the previous
    # 2D tile form also addressed rows as ``row * n_cols``, which read the
    # wrong elements whenever x.stride(0) != n_cols. Closed lanes clamp the
    # row index to 0 so masked accesses stay inside the allocation.
    pid = tl.program_id(0)
    n = tl.load(n_ptr)
    for r in tl.static_range(ROWS_PER_PROG):
        row = pid * ROWS_PER_PROG + r
        row_ok = row < n_rows
        row_c = tl.where(row_ok, row, 0)
        keep = row_ok & (row < n)
        for c0 in range(0, n_cols, BLOCK_C):
            cols = c0 + tl.arange(0, BLOCK_C)
            c_ok = cols < n_cols
            src = tl.load(
                x_ptr + row_c * x_row_stride + cols,
                mask=keep & c_ok,
                other=fill_value,
            )
            tl.store(
                out_ptr + row_c * out_row_stride + cols,
                src,
                mask=row_ok & c_ok,
            )


def fill_padded_rows(x, num_token_non_padded, fill_value):
    out = torch.empty_like(x)
    n_rows, n_cols = x.shape
    total = n_rows * n_cols
    if total == 0:
        return out

    if torch.is_tensor(num_token_non_padded):
        n_dev = num_token_non_padded
    else:
        # Scalar fallback: keep the device-tensor interface so the kernel
        # always reads the count from memory.
        n_dev = torch.tensor(
            [num_token_non_padded], dtype=torch.int32, device=x.device
        )

    if not x.is_contiguous():
        # Strided rows: per-row 1D pass with real strides (empty_like may
        # hand back a contiguous out while x is strided, so the output
        # stride must come from out itself, not from x).
        block_c = triton.next_power_of_2(max(n_cols, 1))
        # Lane width 1 hits the width-1 tile pathology (PassManager fail on
        # this stack); pad to 2 and let the column mask keep values right.
        block_c = max(block_c, 2)
        block_c = min(block_c, 8192, _CTA_BYTE_BUDGET // x.element_size())
        rows_per_prog = max(4, triton.cdiv(n_rows, 65535))
        _fill_padded_rows_strided_rows_kernel[
            (triton.cdiv(n_rows, rows_per_prog),)
        ](
            x,
            out,
            n_dev,
            n_rows,
            n_cols,
            x.stride(0),
            out.stride(0),
            fill_value,
            BLOCK_C=block_c,
            ROWS_PER_PROG=rows_per_prog,
            num_warps=4,
        )
        return out

    # Contiguous flat pass, dtype-aware block budget: single CTA while the
    # whole tensor fits one copy-first program, then a multi-CTA grid of
    # 32KB-per-CTA blocks. Measured on GCU400 (do_bench w1000/r100 median,
    # bf16): total=32768 4.71 -> 3.8-4.0us with BLOCK=16384 grid=2; the
    # previous single-CTA BLOCK=32768 form cannot fresh-compile on this
    # stack (64KB slice) and relied on stale cache binaries.
    max_block = min(_MAX_LANES, _CTA_BYTE_BUDGET // x.element_size())
    if total <= max_block:
        block = triton.next_power_of_2(total)
        num_warps = 2 if block <= 2 * _LANES_PER_WARP_CAP else 4
        grid = (1,)
    else:
        block = max_block
        num_warps = max(4, triton.cdiv(block, _LANES_PER_WARP_CAP))
        grid = (triton.cdiv(total, block),)

    _fill_padded_rows_copyfirst_kernel[grid](
        x,
        out,
        n_dev,
        total,
        fill_value,
        NCOLS=n_cols,
        BLOCK=block,
        EXACT=(total % block == 0),
        num_warps=num_warps,
    )
    return out


__all__ = ["fill_padded_rows"]
