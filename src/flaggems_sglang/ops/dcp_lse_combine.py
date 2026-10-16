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


@triton.jit
def _dcp_lse_combine_kernel(
    out_ptr,
    out_lse_ptr,
    recv_out_ptr,
    recv_lse_ptr,
    M,
    D,
    stride_rn,
    stride_rm,
    stride_ln,
    stride_lm,
    IS_E: tl.constexpr,
    RETURN_LSE: tl.constexpr,
    N: tl.constexpr,
    EVEN_M: tl.constexpr,
    EVEN_D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_d = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    if EVEN_D:
        offs_d = tl.max_contiguous(tl.multiple_of(offs_d, BLOCK_D), BLOCK_D)
    if EVEN_M:
        if EVEN_D:
            m_max = tl.full([BLOCK_M], float("-inf"), tl.float32)
            for n in tl.static_range(0, N):
                lse_n = tl.load(
                    recv_lse_ptr + n * stride_ln + offs_m * stride_lm
                )
                lse_n = tl.where(
                    (lse_n != lse_n) | (lse_n == float("inf")),
                    float("-inf"),
                    lse_n,
                )
                m_max = tl.maximum(m_max, lse_n)
            m_max = tl.where(m_max == float("-inf"), 0.0, m_max)

            acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)
            wsum = tl.zeros([BLOCK_M], dtype=tl.float32)
            for n in tl.static_range(0, N):
                lse_n = tl.load(
                    recv_lse_ptr + n * stride_ln + offs_m * stride_lm
                )
                lse_n = tl.where(
                    (lse_n != lse_n) | (lse_n == float("inf")),
                    float("-inf"),
                    lse_n,
                )
                if IS_E:
                    w = tl.exp(lse_n - m_max)
                else:
                    w = tl.exp2(lse_n - m_max)
                wsum += w
                tile = tl.load(
                    recv_out_ptr
                    + n * stride_rn
                    + offs_m[:, None] * stride_rm
                    + offs_d[None, :],
                    cache_modifier=".cg",
                )
                acc += w[:, None] * tile.to(tl.float32)

            acc = acc / wsum[:, None]
            tl.store(
                out_ptr + offs_m[:, None] * D + offs_d[None, :],
                acc.to(out_ptr.dtype.element_ty),
            )
            if RETURN_LSE and pid_d == 0:
                if IS_E:
                    merged = tl.log(wsum) + m_max
                else:
                    merged = tl.log2(wsum) + m_max
                tl.store(
                    out_lse_ptr + offs_m,
                    merged.to(out_lse_ptr.dtype.element_ty),
                )
        else:
            m_max = tl.full([BLOCK_M], float("-inf"), tl.float32)
            for n in tl.static_range(0, N):
                lse_n = tl.load(
                    recv_lse_ptr + n * stride_ln + offs_m * stride_lm
                )
                lse_n = tl.where(
                    (lse_n != lse_n) | (lse_n == float("inf")),
                    float("-inf"),
                    lse_n,
                )
                m_max = tl.maximum(m_max, lse_n)
            m_max = tl.where(m_max == float("-inf"), 0.0, m_max)

            acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)
            wsum = tl.zeros([BLOCK_M], dtype=tl.float32)
            mask_d = offs_d < D
            for n in tl.static_range(0, N):
                lse_n = tl.load(
                    recv_lse_ptr + n * stride_ln + offs_m * stride_lm
                )
                lse_n = tl.where(
                    (lse_n != lse_n) | (lse_n == float("inf")),
                    float("-inf"),
                    lse_n,
                )
                if IS_E:
                    w = tl.exp(lse_n - m_max)
                else:
                    w = tl.exp2(lse_n - m_max)
                wsum += w
                tile = tl.load(
                    recv_out_ptr
                    + n * stride_rn
                    + offs_m[:, None] * stride_rm
                    + offs_d[None, :],
                    mask=mask_d[None, :],
                    other=0.0,
                    cache_modifier=".cg",
                )
                acc += w[:, None] * tile.to(tl.float32)

            acc = acc / wsum[:, None]
            tl.store(
                out_ptr + offs_m[:, None] * D + offs_d[None, :],
                acc.to(out_ptr.dtype.element_ty),
                mask=mask_d[None, :],
            )
            if RETURN_LSE and pid_d == 0:
                if IS_E:
                    merged = tl.log(wsum) + m_max
                else:
                    merged = tl.log2(wsum) + m_max
                tl.store(
                    out_lse_ptr + offs_m,
                    merged.to(out_lse_ptr.dtype.element_ty),
                )
    else:
        mask_m = offs_m < M
        mask_d = offs_d < D

        m_max = tl.full([BLOCK_M], float("-inf"), tl.float32)
        for n in tl.static_range(0, N):
            lse_n = tl.load(
                recv_lse_ptr + n * stride_ln + offs_m * stride_lm,
                mask=mask_m,
                other=float("-inf"),
            )
            lse_n = tl.where(
                (lse_n != lse_n) | (lse_n == float("inf")),
                float("-inf"),
                lse_n,
            )
            m_max = tl.maximum(m_max, lse_n)
        m_max = tl.where(m_max == float("-inf"), 0.0, m_max)

        acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)
        wsum = tl.zeros([BLOCK_M], dtype=tl.float32)
        for n in tl.static_range(0, N):
            lse_n = tl.load(
                recv_lse_ptr + n * stride_ln + offs_m * stride_lm,
                mask=mask_m,
                other=float("-inf"),
            )
            lse_n = tl.where(
                (lse_n != lse_n) | (lse_n == float("inf")),
                float("-inf"),
                lse_n,
            )
            if IS_E:
                w = tl.exp(lse_n - m_max)
            else:
                w = tl.exp2(lse_n - m_max)
            wsum += w
            tile = tl.load(
                recv_out_ptr
                + n * stride_rn
                + offs_m[:, None] * stride_rm
                + offs_d[None, :],
                mask=mask_m[:, None] & mask_d[None, :],
                other=0.0,
                cache_modifier=".cg",
            )
            acc += w[:, None] * tile.to(tl.float32)

        acc = acc / wsum[:, None]
        tl.store(
            out_ptr + offs_m[:, None] * D + offs_d[None, :],
            acc.to(out_ptr.dtype.element_ty),
            mask=mask_m[:, None] & mask_d[None, :],
        )
        if RETURN_LSE and pid_d == 0:
            if IS_E:
                merged = tl.log(wsum) + m_max
            else:
                merged = tl.log2(wsum) + m_max
            tl.store(
                out_lse_ptr + offs_m,
                merged.to(out_lse_ptr.dtype.element_ty),
                mask=mask_m,
            )


_SMALL_M = 128


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

    if m <= _SMALL_M:
        block_m, block_d, num_warps = 4, min(16, triton.next_power_of_2(d)), 2
    elif m <= 4096:
        block_m, block_d, num_warps = 16, min(64, triton.next_power_of_2(d)), 8
    else:
        block_m, block_d, num_warps = (
            16,
            min(128, triton.next_power_of_2(d)),
            16,
        )

    grid = (triton.cdiv(m, block_m), triton.cdiv(d, block_d))
    _dcp_lse_combine_kernel[grid](
        out,
        out_lse if out_lse is not None else out,
        recv_output,
        recv_lse,
        m,
        d,
        recv_output.stride(0),
        d,
        recv_lse.stride(0),
        1,
        IS_E=bool(is_lse_base_on_e),
        RETURN_LSE=bool(return_lse),
        N=n,
        EVEN_M=(m % block_m == 0),
        EVEN_D=(d % block_d == 0),
        BLOCK_M=block_m,
        BLOCK_D=block_d,
        num_warps=num_warps,
    )
    return out, out_lse


__all__ = ["dcp_lse_combine"]
