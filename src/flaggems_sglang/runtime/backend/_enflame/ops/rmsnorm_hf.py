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

_MAX_SINGLE_PASS = (
    1 << 14
)  # max hidden size handled by the register-resident kernel

# Row-batch tile budget (elements). 32768 elems measured ~526KB peak DSM on
# this kernel family (fused_dual_residual_rmsnorm calibration), safely below
# the 917504B compile-time limit; larger tiles also enter the documented
# 2D-broadcast miscompile window (tile=65536 / R>=32).
_MAX_ROWBATCH_TILE = 1 << 15


def _small_cfgs():
    cfgs = []
    for rows, warps, stages in (
        (8, 1, 1),
        (8, 2, 1),
        (8, 4, 1),
        (8, 4, 2),
        (16, 4, 1),
        (16, 8, 1),
        (32, 8, 1),
    ):
        cfgs.append(
            triton.Config({"ROWS": rows}, num_warps=warps, num_stages=stages)
        )
    return cfgs


def _row_cfgs():
    cfgs = []
    for warps in (4, 8, 16):
        for stages in (1, 3):
            cfgs.append(triton.Config({}, num_warps=warps, num_stages=stages))
    return cfgs


def _prev_pow2(value):
    return 1 << max(0, value.bit_length() - 1)


@triton.autotune(configs=_small_cfgs(), key=["M", "N"])
@triton.jit
def _rmsnorm_hf_kernel_rows(
    x_ptr,
    w_ptr,
    out_ptr,
    M,
    N,
    stride_x,
    stride_o,
    eps,
    BLOCK_N: tl.constexpr,
    ROWS: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    rows = pid * ROWS + tl.arange(0, ROWS)
    rmask = rows < M
    cols = tl.arange(0, BLOCK_N)
    cmask = cols < N
    mask = rmask[:, None] & cmask[None, :]
    w = tl.load(
        w_ptr + cols, mask=cmask, other=0.0, eviction_policy="evict_last"
    ).to(tl.float32)
    x = tl.load(
        x_ptr + rows[:, None] * stride_x + cols[None, :],
        mask=mask,
        other=0.0,
        eviction_policy="evict_first",
    ).to(tl.float32)
    ms = tl.sum(x * x, axis=1) / N
    rstd = tl.math.rsqrt(ms + eps)
    # HF semantics: round the normalized value to the input dtype first.
    y = (x * rstd[:, None]).to(x_ptr.dtype.element_ty)
    tl.store(
        out_ptr + rows[:, None] * stride_o + cols[None, :],
        (w[None, :] * y.to(tl.float32)).to(out_ptr.dtype.element_ty),
        mask=mask,
        eviction_policy="evict_first",
    )


@triton.autotune(configs=_row_cfgs(), key=["M", "N"])
@triton.jit
def _rmsnorm_hf_kernel(
    x_ptr,
    w_ptr,
    out_ptr,
    M,
    N: tl.constexpr,
    stride_x,
    stride_o,
    eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    # Issue the weight load first (evict_last: shared by all programs) so its
    # latency overlaps the streamed input load (evict_first).
    w = tl.load(
        w_ptr + cols, mask=mask, other=0.0, eviction_policy="evict_last"
    ).to(tl.float32)
    x = tl.load(
        x_ptr + row * stride_x + cols,
        mask=mask,
        other=0.0,
        eviction_policy="evict_first",
    ).to(tl.float32)
    ms = tl.sum(x * x, axis=0) / N
    rstd = tl.math.rsqrt(ms + eps)
    y = (x * rstd).to(x_ptr.dtype.element_ty)
    tl.store(
        out_ptr + row * stride_o + cols,
        (w * y.to(tl.float32)).to(out_ptr.dtype.element_ty),
        mask=mask,
        eviction_policy="evict_first",
    )


@triton.jit
def _rmsnorm_hf_kernel_rowbatch(
    x_ptr,
    w_ptr,
    out_ptr,
    M,
    stride_x,
    stride_o,
    eps,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
    R: tl.constexpr,
    EXACT: tl.constexpr,
):
    # R rows per program: one wide [R, BLOCK] tile merges the per-row
    # load/reduce/store chains and amortizes the weight load, replacing the
    # one-row-per-program serial chain that retires ~17 programs/us.
    # num_warps is pinned to 1 by the host: nw>=2 enters the documented
    # 2D double-broadcast miscompile windows (BLOCK 512/1024 family) and
    # measured 20-25% slower here with the 1D tail arm coexisting.
    pid = tl.program_id(0)
    row0 = pid * R
    if row0 + R <= M:
        rows = (row0 + tl.arange(0, R)).to(tl.int64)
        cols = tl.arange(0, BLOCK)
        if EXACT:
            x = tl.load(
                x_ptr + rows[:, None] * stride_x + cols[None, :]
            ).to(tl.float32)
            # [1, BLOCK] 2D-shaped weight load: keeps the broadcast chain on
            # the proven-safe form (1D vector broadcast is a miscompile face).
            wv = tl.load(w_ptr + cols[None, :]).to(tl.float32)
            ms = tl.sum(x * x, axis=1) / N
            rrms = tl.math.rsqrt(ms + eps)
            # HF semantics: round the normalized value to the input dtype first.
            y = (x * rrms[:, None]).to(x_ptr.dtype.element_ty).to(tl.float32)
            tl.store(
                out_ptr + rows[:, None] * stride_o + cols[None, :],
                (wv * y).to(out_ptr.dtype.element_ty),
            )
        else:
            cmask = cols < N
            x = tl.load(
                x_ptr + rows[:, None] * stride_x + cols[None, :],
                mask=cmask[None, :],
                other=0.0,
            ).to(tl.float32)
            wv = tl.load(
                w_ptr + cols[None, :], mask=cmask[None, :], other=0.0
            ).to(tl.float32)
            ms = tl.sum(x * x, axis=1) / N
            rrms = tl.math.rsqrt(ms + eps)
            y = (x * rrms[:, None]).to(x_ptr.dtype.element_ty).to(tl.float32)
            tl.store(
                out_ptr + rows[:, None] * stride_o + cols[None, :],
                (wv * y).to(out_ptr.dtype.element_ty),
                mask=cmask[None, :],
            )
    else:
        # 1D per-row tail (static_range, nw=1): the 2D row-masked tail block
        # is a documented silent-miscompile window on BLOCK 512/1024.
        for i in tl.static_range(R):
            row = row0 + i
            if row < M:
                cols = tl.arange(0, BLOCK)
                cmask = cols < N
                x = tl.load(
                    x_ptr + row.to(tl.int64) * stride_x + cols,
                    mask=cmask,
                    other=0.0,
                ).to(tl.float32)
                ms = tl.sum(x * x, axis=0) / N
                rrms = tl.math.rsqrt(ms + eps)
                y = (x * rrms).to(x_ptr.dtype.element_ty).to(tl.float32)
                wv = tl.load(
                    w_ptr + cols, mask=cmask, other=0.0
                ).to(tl.float32)
                tl.store(
                    out_ptr + row.to(tl.int64) * stride_o + cols,
                    (wv * y).to(out_ptr.dtype.element_ty),
                    mask=cmask,
                )


@triton.jit
def _rmsnorm_hf_kernel_two_pass(
    x_ptr,
    w_ptr,
    out_ptr,
    stride_x,
    stride_o,
    N,
    eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    base = row * stride_x
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for off in range(0, N, BLOCK):
        cols = off + tl.arange(0, BLOCK)
        mask = cols < N
        x = tl.load(
            x_ptr + base + cols,
            mask=mask,
            other=0.0,
            eviction_policy="evict_first",
        ).to(tl.float32)
        acc += x * x
    ms = tl.sum(acc, axis=0) / N
    rstd = tl.math.rsqrt(ms + eps)
    obase = row * stride_o
    for off in range(0, N, BLOCK):
        cols = off + tl.arange(0, BLOCK)
        mask = cols < N
        x = tl.load(
            x_ptr + base + cols,
            mask=mask,
            other=0.0,
            eviction_policy="evict_first",
        ).to(tl.float32)
        y = (x * rstd).to(x_ptr.dtype.element_ty)
        w = tl.load(
            w_ptr + cols, mask=mask, other=0.0, eviction_policy="evict_last"
        ).to(tl.float32)
        out = w * y.to(tl.float32)
        tl.store(
            out_ptr + obase + cols,
            out.to(out_ptr.dtype.element_ty),
            mask=mask,
            eviction_policy="evict_first",
        )


def rmsnorm_hf(input, weight, eps):
    n = input.shape[-1]
    m = input.numel() // n
    xc = input if input.is_contiguous() else input.contiguous()
    w = weight if weight.is_contiguous() else weight.contiguous()
    out_dtype = w.dtype
    if out_dtype == input.dtype:
        out = torch.empty_like(xc)
    else:
        out = torch.empty(
            xc.shape,
            dtype=torch.promote_types(out_dtype, input.dtype),
            device=xc.device,
        )
    if m == 0 or n == 0:
        return out
    sx = xc.stride(0)
    so = out.stride(0)

    if n <= 512:
        # Small hidden: batch several rows per program to keep HBM busy.
        # ROWS / num_warps / num_stages are picked per (M, N) by autotune.
        _rmsnorm_hf_kernel_rows[lambda meta: (triton.cdiv(m, meta["ROWS"]),)](
            xc,
            w,
            out,
            m,
            n,
            sx,
            so,
            eps,
            BLOCK_N=triton.next_power_of_2(n),
        )
    elif n <= _MAX_SINGLE_PASS:
        block = triton.next_power_of_2(n)
        # Row-batch dispatch (M >= 64): R grows with M, capped by the tile
        # budget; probed optima were R=8@BLOCK4096/M512, R=4@BLOCK8192/M512,
        # R=2@M64 -- i.e. R = min(tile_cap, prev_pow2(M/32)).
        rows_per = min(
            max(1, _MAX_ROWBATCH_TILE // block), _prev_pow2(max(1, m // 32))
        )
        if rows_per >= 2:
            _rmsnorm_hf_kernel_rowbatch[(triton.cdiv(m, rows_per),)](
                xc,
                w,
                out,
                m,
                sx,
                so,
                eps,
                N=n,
                BLOCK=block,
                R=rows_per,
                EXACT=(block == n),
                num_warps=1,
            )
        else:
            # Few rows: one row per program keeps max device parallelism
            # (row-batching starves the machine below ~64 programs).
            _rmsnorm_hf_kernel[(m,)](
                xc,
                w,
                out,
                M=m,
                stride_x=sx,
                stride_o=so,
                eps=eps,
                N=n,
                BLOCK=block,
            )
    else:
        block = 2048
        _rmsnorm_hf_kernel_two_pass[(m,)](
            xc,
            w,
            out,
            stride_x=sx,
            stride_o=so,
            N=n,
            eps=eps,
            BLOCK=block,
            num_warps=8,
            num_stages=2,
        )
    return out


__all__ = ["rmsnorm_hf"]
