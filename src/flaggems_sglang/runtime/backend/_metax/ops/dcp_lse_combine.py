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

"""Operator: attention/dcp_lse_combine"""

import torch
import triton
import triton.language as tl

_LN2 = tl.constexpr(0.6931471805599453)
_LOG2E = tl.constexpr(1.4426950408889634)


@triton.jit
def _dcp_lse_combine_kernel(
    recv_out_ptr,  # [N, TOTAL_BH, D] contiguous shards
    recv_lse_ptr,  # [N, TOTAL_BH] float32 per-shard LSEs
    out_ptr,  # [TOTAL_BH, D] combined output
    out_lse_ptr,  # [TOTAL_BH] combined LSE (only touched when RETURN_LSE)
    lse_n_stride,  # elements between consecutive shards in recv_lse
    out_n_stride,  # elements between consecutive shards in recv_output
    TOTAL_BH,
    N: tl.constexpr,
    D: tl.constexpr,
    BLOCK_BH: tl.constexpr,
    BLOCK_D: tl.constexpr,
    IS_BASE_E: tl.constexpr,
    RETURN_LSE: tl.constexpr,
    NS: tl.constexpr,  # pipeline stages for the accumulate loop
    EVEN: tl.constexpr,  # TOTAL_BH % BLOCK_BH == 0 and D == BLOCK_D
):
    pid = tl.program_id(0)
    offs_bh = pid * BLOCK_BH + tl.arange(0, BLOCK_BH)
    offs_d = tl.arange(0, BLOCK_D)

    if EVEN:
        # --- pass 1: per-shard max, NaN / +inf sanitized to -inf ------------
        m = tl.full((BLOCK_BH,), float("-inf"), tl.float32)
        for n in tl.range(0, N):
            lse_n = tl.load(recv_lse_ptr + n * lse_n_stride + offs_bh)
            bad = (lse_n != lse_n) | (lse_n == float("inf"))
            m = tl.maximum(m, tl.where(bad, float("-inf"), lse_n))
        # Every shard dead: force m = 0 so weights stay finite (matches ref).
        m = tl.where(m == float("-inf"), 0.0, m)

        acc = tl.zeros((BLOCK_BH, BLOCK_D), dtype=tl.float32)
        wsum = tl.zeros((BLOCK_BH,), dtype=tl.float32)
        for n in tl.range(0, N, num_stages=NS):
            lse_n = tl.load(recv_lse_ptr + n * lse_n_stride + offs_bh)
            bad = (lse_n != lse_n) | (lse_n == float("inf"))
            lse_n = tl.where(bad, float("-inf"), lse_n)
            if IS_BASE_E:
                w = tl.exp2((lse_n - m) * _LOG2E)
            else:
                w = tl.exp2(lse_n - m)
            wsum += w
            row_ptrs = (
                recv_out_ptr
                + n * out_n_stride
                + offs_bh[:, None] * D
                + offs_d[None, :]
            )
            rows = tl.load(row_ptrs, cache_modifier=".cg")
            acc += rows.to(tl.float32) * w[:, None]

        # One reciprocal per (b,h) instead of a divide per output element.
        rcp = 1.0 / wsum
        out_ptrs = out_ptr + offs_bh[:, None] * D + offs_d[None, :]
        tl.store(out_ptrs, (acc * rcp[:, None]).to(out_ptr.dtype.element_ty))

        if RETURN_LSE:
            if IS_BASE_E:
                combined = tl.log2(wsum) * _LN2 + m
            else:
                combined = tl.log2(wsum) + m
            tl.store(out_lse_ptr + offs_bh, combined)
    else:
        bh_mask = offs_bh < TOTAL_BH
        d_mask = offs_d < D

        m = tl.full((BLOCK_BH,), float("-inf"), tl.float32)
        for n in tl.range(0, N):
            lse_n = tl.load(
                recv_lse_ptr + n * lse_n_stride + offs_bh,
                mask=bh_mask,
                other=float("-inf"),
            )
            bad = (lse_n != lse_n) | (lse_n == float("inf"))
            m = tl.maximum(m, tl.where(bad, float("-inf"), lse_n))
        m = tl.where(m == float("-inf"), 0.0, m)

        acc = tl.zeros((BLOCK_BH, BLOCK_D), dtype=tl.float32)
        wsum = tl.zeros((BLOCK_BH,), dtype=tl.float32)
        for n in tl.range(0, N, num_stages=NS):
            lse_n = tl.load(
                recv_lse_ptr + n * lse_n_stride + offs_bh,
                mask=bh_mask,
                other=float("-inf"),
            )
            bad = (lse_n != lse_n) | (lse_n == float("inf"))
            lse_n = tl.where(bad, float("-inf"), lse_n)
            if IS_BASE_E:
                w = tl.exp2((lse_n - m) * _LOG2E)
            else:
                w = tl.exp2(lse_n - m)
            wsum += w
            row_ptrs = (
                recv_out_ptr
                + n * out_n_stride
                + offs_bh[:, None] * D
                + offs_d[None, :]
            )
            rows = tl.load(
                row_ptrs,
                mask=bh_mask[:, None] & d_mask[None, :],
                other=0.0,
                cache_modifier=".cg",
            )
            acc += rows.to(tl.float32) * w[:, None]

        rcp = 1.0 / wsum
        out_ptrs = out_ptr + offs_bh[:, None] * D + offs_d[None, :]
        tl.store(
            out_ptrs,
            (acc * rcp[:, None]).to(out_ptr.dtype.element_ty),
            mask=bh_mask[:, None] & d_mask[None, :],
        )

        if RETURN_LSE:
            if IS_BASE_E:
                combined = tl.log2(wsum) * _LN2 + m
            else:
                combined = tl.log2(wsum) + m
            tl.store(out_lse_ptr + offs_bh, combined, mask=bh_mask)


@triton.jit
def _dcp_lse_combine_kernel_ds(
    recv_out_ptr,  # [N, TOTAL_BH, D] contiguous shards
    recv_lse_ptr,  # [N, TOTAL_BH] float32 per-shard LSEs
    out_ptr,  # [TOTAL_BH, D] combined output
    out_lse_ptr,  # [TOTAL_BH] combined LSE (only touched when RETURN_LSE)
    lse_n_stride,  # elements between consecutive shards in recv_lse
    out_n_stride,  # elements between consecutive shards in recv_output
    TOTAL_BH,
    N: tl.constexpr,
    D: tl.constexpr,
    BLOCK_BH: tl.constexpr,
    BLOCK_D: tl.constexpr,
    IS_BASE_E: tl.constexpr,
    RETURN_LSE: tl.constexpr,
    EVEN: tl.constexpr,  # TOTAL_BH % BLOCK_BH == 0 and D % BLOCK_D == 0
):
    """D-split twin of the main kernel for latency-bound small shapes."""
    pid = tl.program_id(0)
    pid_d = tl.program_id(1)
    offs_bh = pid * BLOCK_BH + tl.arange(0, BLOCK_BH)
    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)

    if EVEN:
        # --- pass 1: per-shard max (identical in every pid_d column) --------
        m = tl.full((BLOCK_BH,), float("-inf"), tl.float32)
        for n in tl.static_range(N):
            lse_n = tl.load(
                recv_lse_ptr + n * lse_n_stride + offs_bh,
                eviction_policy="evict_last",
            )
            bad = (lse_n != lse_n) | (lse_n == float("inf"))
            m = tl.maximum(m, tl.where(bad, float("-inf"), lse_n))
        m = tl.where(m == float("-inf"), 0.0, m)

        # --- pass 2: this column's slice of each shard ----------------------
        acc = tl.zeros((BLOCK_BH, BLOCK_D), dtype=tl.float32)
        wsum = tl.zeros((BLOCK_BH,), dtype=tl.float32)
        for n in tl.static_range(N):
            lse_n = tl.load(
                recv_lse_ptr + n * lse_n_stride + offs_bh,
                eviction_policy="evict_first",
            )
            bad = (lse_n != lse_n) | (lse_n == float("inf"))
            lse_n = tl.where(bad, float("-inf"), lse_n)
            if IS_BASE_E:
                w = tl.exp2((lse_n - m) * _LOG2E)
            else:
                w = tl.exp2(lse_n - m)
            wsum += w
            row_ptrs = (
                recv_out_ptr
                + n * out_n_stride
                + offs_bh[:, None] * D
                + offs_d[None, :]
            )
            rows = tl.load(
                row_ptrs, cache_modifier=".cg", eviction_policy="evict_first"
            )
            acc += rows.to(tl.float32) * w[:, None]

        rcp = 1.0 / wsum
        out_ptrs = out_ptr + offs_bh[:, None] * D + offs_d[None, :]
        tl.store(out_ptrs, (acc * rcp[:, None]).to(out_ptr.dtype.element_ty))

        if RETURN_LSE:
            if pid_d == 0:
                if IS_BASE_E:
                    combined = tl.log2(wsum) * _LN2 + m
                else:
                    combined = tl.log2(wsum) + m
                tl.store(out_lse_ptr + offs_bh, combined)
    else:
        bh_mask = offs_bh < TOTAL_BH
        d_mask = offs_d < D

        m = tl.full((BLOCK_BH,), float("-inf"), tl.float32)
        for n in tl.static_range(N):
            lse_n = tl.load(
                recv_lse_ptr + n * lse_n_stride + offs_bh,
                mask=bh_mask,
                other=float("-inf"),
            )
            bad = (lse_n != lse_n) | (lse_n == float("inf"))
            m = tl.maximum(m, tl.where(bad, float("-inf"), lse_n))
        m = tl.where(m == float("-inf"), 0.0, m)

        acc = tl.zeros((BLOCK_BH, BLOCK_D), dtype=tl.float32)
        wsum = tl.zeros((BLOCK_BH,), dtype=tl.float32)
        for n in tl.static_range(N):
            lse_n = tl.load(
                recv_lse_ptr + n * lse_n_stride + offs_bh,
                mask=bh_mask,
                other=float("-inf"),
            )
            bad = (lse_n != lse_n) | (lse_n == float("inf"))
            lse_n = tl.where(bad, float("-inf"), lse_n)
            if IS_BASE_E:
                w = tl.exp2((lse_n - m) * _LOG2E)
            else:
                w = tl.exp2(lse_n - m)
            wsum += w
            row_ptrs = (
                recv_out_ptr
                + n * out_n_stride
                + offs_bh[:, None] * D
                + offs_d[None, :]
            )
            rows = tl.load(
                row_ptrs,
                mask=bh_mask[:, None] & d_mask[None, :],
                other=0.0,
                cache_modifier=".cg",
            )
            acc += rows.to(tl.float32) * w[:, None]

        rcp = 1.0 / wsum
        out_ptrs = out_ptr + offs_bh[:, None] * D + offs_d[None, :]
        tl.store(
            out_ptrs,
            (acc * rcp[:, None]).to(out_ptr.dtype.element_ty),
            mask=bh_mask[:, None] & d_mask[None, :],
        )

        if RETURN_LSE:
            if pid_d == 0:
                if IS_BASE_E:
                    combined = tl.log2(wsum) * _LN2 + m
                else:
                    combined = tl.log2(wsum) + m
                tl.store(out_lse_ptr + offs_bh, combined, mask=bh_mask)


def _pick_cfg(total_bh, d, n):
    """Static launch config, per size class."""
    if total_bh <= 32:
        # Tiny (b=1): 2 (b,h) x 64-wide D slice per program, 2D grid.
        return _dcp_lse_combine_kernel_ds, 2, min(64, d), 1, 1, 1
    if total_bh <= 1024 and d >= 128:
        # Small (b=64): 4 (b,h) x 64-wide D slice, 2D grid.
        return _dcp_lse_combine_kernel_ds, 4, 64, 1, 1, 1
    # Bandwidth-bound (b=1024 and up): unsplit, pipelined shard loop.
    if n <= 4:
        return _dcp_lse_combine_kernel, 4, d, 1, 1, 3
    return _dcp_lse_combine_kernel, 16, d, 2, 1, 2


def dcp_lse_combine(recv_output, recv_lse, is_lse_base_on_e, return_lse):
    n, b, h, d = recv_output.shape
    # No-op for the contiguous tensors the framework hands us; keeps the
    # kernel's flattened-stride view valid for arbitrary inputs.
    recv_output = recv_output.contiguous()
    recv_lse = recv_lse.contiguous()

    out = torch.empty(
        (b, h, d), dtype=recv_output.dtype, device=recv_output.device
    )
    if return_lse:
        out_lse = torch.empty(
            (b, h), dtype=recv_lse.dtype, device=recv_lse.device
        )
    else:
        out_lse = out  # placeholder; RETURN_LSE=False compiles the store away

    total_bh = b * h
    kern, block_bh, block_d, num_warps, num_stages, ns = _pick_cfg(
        total_bh, d, n
    )

    if kern is _dcp_lse_combine_kernel_ds:
        grid = (triton.cdiv(total_bh, block_bh), triton.cdiv(d, block_d))
        even = total_bh % block_bh == 0 and d % block_d == 0
        kern[grid](
            recv_output,
            recv_lse,
            out,
            out_lse,
            recv_lse.stride(0),
            recv_output.stride(0),
            total_bh,
            N=n,
            D=d,
            BLOCK_BH=block_bh,
            BLOCK_D=block_d,
            IS_BASE_E=bool(is_lse_base_on_e),
            RETURN_LSE=bool(return_lse),
            EVEN=even,
            num_warps=num_warps,
            num_stages=num_stages,
        )
    else:
        grid = (triton.cdiv(total_bh, block_bh),)
        even = total_bh % block_bh == 0 and d == block_d
        kern[grid](
            recv_output,
            recv_lse,
            out,
            out_lse,
            recv_lse.stride(0),
            recv_output.stride(0),
            total_bh,
            N=n,
            D=d,
            BLOCK_BH=block_bh,
            BLOCK_D=block_d,
            IS_BASE_E=bool(is_lse_base_on_e),
            RETURN_LSE=bool(return_lse),
            NS=ns,
            EVEN=even,
            num_warps=num_warps,
            num_stages=num_stages,
        )
    return out, (out_lse if return_lse else None)


__all__ = ["dcp_lse_combine"]
