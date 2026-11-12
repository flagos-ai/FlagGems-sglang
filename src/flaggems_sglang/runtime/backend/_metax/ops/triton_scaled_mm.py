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
def _dequant_flat_kernel(
    a_ptr,
    b_ptr,
    ya_ptr,
    yb_ptr,
    num_a,
    num_b,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    nblk_a = tl.cdiv(num_a, BLOCK)
    if pid < nblk_a:
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < num_a
        x = tl.load(a_ptr + offs, mask=mask, other=0.0)
        tl.store(ya_ptr + offs, x.to(tl.bfloat16), mask=mask)
    else:
        offs = (pid - nblk_a) * BLOCK + tl.arange(0, BLOCK)
        mask = offs < num_b
        x = tl.load(b_ptr + offs, mask=mask, other=0.0)
        tl.store(yb_ptr + offs, x.to(tl.bfloat16), mask=mask)


@triton.jit
def _dequant_both_kernel(
    a_ptr,
    b_ptr,
    ya_ptr,
    yb_ptr,
    ROWS_A,
    ROWS_B,
    cols_a,
    cols_b,
    stride_am,
    stride_ak,
    stride_bm,
    stride_bk,
    RPP: tl.constexpr,
    BLOCK_COLS: tl.constexpr,
):
    pid = tl.program_id(0)
    nblk_a = tl.cdiv(cols_a, BLOCK_COLS)
    nblk_b = tl.cdiv(cols_b, BLOCK_COLS)
    progs_a = (ROWS_A + RPP - 1) // RPP * nblk_a
    is_b = pid >= progs_a
    if is_b:
        local = pid - progs_a
        row0 = (local // nblk_b) * RPP
        col_blk = local % nblk_b
        offs_c = col_blk * BLOCK_COLS + tl.arange(0, BLOCK_COLS)
        offs_r = row0 + tl.arange(0, RPP)
        mask_c = offs_c < cols_b
        mask_r = offs_r < ROWS_B
        x = tl.load(
            b_ptr + offs_r[:, None] * stride_bm + offs_c[None, :] * stride_bk,
            mask=mask_r[:, None] & mask_c[None, :],
            other=0.0,
        )
        tl.store(
            yb_ptr + offs_r[:, None] * cols_b + offs_c[None, :],
            x.to(tl.bfloat16),
            mask=mask_r[:, None] & mask_c[None, :],
        )
    else:
        row0 = (pid // nblk_a) * RPP
        col_blk = pid % nblk_a
        offs_c = col_blk * BLOCK_COLS + tl.arange(0, BLOCK_COLS)
        offs_r = row0 + tl.arange(0, RPP)
        mask_c = offs_c < cols_a
        mask_r = offs_r < ROWS_A
        x = tl.load(
            a_ptr + offs_r[:, None] * stride_am + offs_c[None, :] * stride_ak,
            mask=mask_r[:, None] & mask_c[None, :],
            other=0.0,
        )
        tl.store(
            ya_ptr + offs_r[:, None] * cols_a + offs_c[None, :],
            x.to(tl.bfloat16),
            mask=mask_r[:, None] & mask_c[None, :],
        )


@triton.jit
def _scaled_mm_kernel(
    a_ptr,
    b_ptr,
    sa_ptr,
    sb_ptr,
    bias_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_sam,
    stride_sbn,
    stride_bm,
    stride_bn2,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
    EVEN_K: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    SA_SCALAR: tl.constexpr,
    SB_SCALAR: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + (
        offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    )
    b_ptrs = b_ptr + (
        offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    )

    mask_m = offs_m < M
    mask_n = offs_n < N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    if EVEN_K:
        for _ in range(0, tl.cdiv(K, BLOCK_K)):
            if EVEN_M:
                a = tl.load(a_ptrs)
            else:
                a = tl.load(a_ptrs, mask=mask_m[:, None], other=0.0)
            if EVEN_N:
                b = tl.load(b_ptrs)
            else:
                b = tl.load(b_ptrs, mask=mask_n[None, :], other=0.0)
            acc = tl.dot(a, b, acc)
            a_ptrs += BLOCK_K * stride_ak
            b_ptrs += BLOCK_K * stride_bk
    else:
        for k_idx in range(0, tl.cdiv(K, BLOCK_K)):
            mask_k = offs_k < K - k_idx * BLOCK_K
            if EVEN_M:
                mask_a = mask_k[None, :]
            else:
                mask_a = mask_m[:, None] & mask_k[None, :]
            if EVEN_N:
                mask_b = mask_k[:, None]
            else:
                mask_b = mask_k[:, None] & mask_n[None, :]
            a = tl.load(a_ptrs, mask=mask_a, other=0.0)
            b = tl.load(b_ptrs, mask=mask_b, other=0.0)
            acc = tl.dot(a, b, acc)
            a_ptrs += BLOCK_K * stride_ak
            b_ptrs += BLOCK_K * stride_bk

    if SA_SCALAR:
        acc = acc * tl.load(sa_ptr).to(tl.float32)
    else:
        sa = tl.load(sa_ptr + offs_m * stride_sam, mask=mask_m, other=1.0)
        acc = acc * sa.to(tl.float32)[:, None]
    if SB_SCALAR:
        acc = acc * tl.load(sb_ptr).to(tl.float32)
    else:
        sb = tl.load(sb_ptr + offs_n * stride_sbn, mask=mask_n, other=1.0)
        acc = acc * sb.to(tl.float32)[None, :]

    if HAS_BIAS:
        bias_ptrs = bias_ptr + (
            offs_m[:, None] * stride_bm + offs_n[None, :] * stride_bn2
        )
        if EVEN_M and EVEN_N:
            bias = tl.load(bias_ptrs)
        else:
            bias = tl.load(
                bias_ptrs, mask=mask_m[:, None] & mask_n[None, :], other=0.0
            )
        acc += bias.to(tl.float32)

    c_ptrs = c_ptr + (
        offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    )
    if EVEN_M and EVEN_N:
        tl.store(c_ptrs, acc.to(c_ptr.dtype.element_ty))
    else:
        tl.store(
            c_ptrs,
            acc.to(c_ptr.dtype.element_ty),
            mask=mask_m[:, None] & mask_n[None, :],
        )


@triton.jit
def _splitk_part_kernel(
    a_ptr,
    b_ptr,
    ws_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_wk,
    stride_wm,
    stride_wn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SPLIT_K: tl.constexpr,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
    EVEN_K: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_k = pid % SPLIT_K
    pid_g = pid // SPLIT_K
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid_g // num_pid_n
    pid_n = pid_g % num_pid_n

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    K_PER = tl.cdiv(tl.cdiv(K, SPLIT_K), BLOCK_K) * BLOCK_K
    k_start = pid_k * K_PER

    a_ptrs = a_ptr + (
        offs_m[:, None] * stride_am + (k_start + offs_k)[None, :] * stride_ak
    )
    b_ptrs = (
        b_ptr
        + (k_start + offs_k)[:, None] * stride_bk
        + offs_n[None, :] * stride_bn
    )
    mask_m = offs_m < M
    mask_n = offs_n < N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k_idx in range(0, K_PER // BLOCK_K):
        kk = k_start + k_idx * BLOCK_K
        if kk < K:
            if EVEN_K:
                if EVEN_M:
                    a = tl.load(a_ptrs)
                else:
                    a = tl.load(a_ptrs, mask=mask_m[:, None], other=0.0)
                if EVEN_N:
                    b = tl.load(b_ptrs)
                else:
                    b = tl.load(b_ptrs, mask=mask_n[None, :], other=0.0)
            else:
                mask_k = offs_k < K - kk
                a = tl.load(
                    a_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0
                )
                b = tl.load(
                    b_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0
                )
            acc = tl.dot(a, b, acc)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    ws_ptrs = (
        ws_ptr
        + pid_k * stride_wk
        + offs_m[:, None] * stride_wm
        + offs_n[None, :] * stride_wn
    )
    if EVEN_M and EVEN_N:
        tl.store(ws_ptrs, acc.to(ws_ptr.dtype.element_ty))
    else:
        tl.store(
            ws_ptrs,
            acc.to(ws_ptr.dtype.element_ty),
            mask=mask_m[:, None] & mask_n[None, :],
        )


@triton.jit
def _splitk_reduce_kernel(
    ws_ptr,
    sa_ptr,
    sb_ptr,
    bias_ptr,
    c_ptr,
    M,
    N,
    stride_wk,
    stride_wm,
    stride_wn,
    stride_sam,
    stride_sbn,
    stride_bm,
    stride_bn2,
    stride_cm,
    stride_cn,
    SPLIT_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    SA_SCALAR: tl.constexpr,
    SB_SCALAR: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = offs_m < M
    mask_n = offs_n < N
    mask = mask_m[:, None] & mask_n[None, :]

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for s in range(SPLIT_K):
        acc += tl.load(
            ws_ptr
            + s * stride_wk
            + offs_m[:, None] * stride_wm
            + offs_n[None, :] * stride_wn,
            mask=mask,
            other=0.0,
        ).to(tl.float32)

    if SA_SCALAR:
        acc = acc * tl.load(sa_ptr).to(tl.float32)
    else:
        sa = tl.load(sa_ptr + offs_m * stride_sam, mask=mask_m, other=1.0)
        acc = acc * sa.to(tl.float32)[:, None]
    if SB_SCALAR:
        acc = acc * tl.load(sb_ptr).to(tl.float32)
    else:
        sb = tl.load(sb_ptr + offs_n * stride_sbn, mask=mask_n, other=1.0)
        acc = acc * sb.to(tl.float32)[None, :]

    if HAS_BIAS:
        bias_ptrs = bias_ptr + (
            offs_m[:, None] * stride_bm + offs_n[None, :] * stride_bn2
        )
        bias = tl.load(
            bias_ptrs, mask=mask_m[:, None] & mask_n[None, :], other=0.0
        )
        acc += bias.to(tl.float32)

    c_ptrs = c_ptr + (
        offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    )
    tl.store(c_ptrs, acc.to(c_ptr.dtype.element_ty), mask=mask)


def _splitk_cfg(m, n):
    if m <= 64:
        return 6, 64, 256, 64, 8, 4
    return 3, 256, 256, 32, 8, 3


def _gemm_config(m, n):
    if m <= 16:
        if n <= 4096:
            return 16, 64, 64, 1, 4, 4
        return 16, 128, 64, 1, 8, 4
    return 256, 256, 32, 8, 8, 2


_SPLITK_REDUCE_TILE = (16, 256, 8, 4)
_DQ_BLOCK = 2048


def triton_scaled_mm(input, weight, scale_a, scale_b, out_dtype, bias=None):
    m, k = input.shape
    n = weight.shape[1]
    out = torch.empty((m, n), dtype=out_dtype, device=input.device)
    if m == 0 or n == 0:
        return out

    a_bf = torch.empty((m, k), dtype=torch.bfloat16, device=input.device)
    b_bf = torch.empty((k, n), dtype=torch.bfloat16, device=input.device)
    if k != 0:
        if input.stride(1) == 1 and weight.stride(1) == 1:
            num_a = m * k
            num_b = k * n
            _dequant_flat_kernel[
                (
                    triton.cdiv(num_a, _DQ_BLOCK)
                    + triton.cdiv(num_b, _DQ_BLOCK),
                )
            ](
                input,
                weight,
                a_bf,
                b_bf,
                num_a,
                num_b,
                BLOCK=_DQ_BLOCK,
                num_warps=1,
                num_stages=2,
            )
        else:
            _dequant_both_kernel[
                (
                    triton.cdiv(m, 2) * triton.cdiv(k, 1024)
                    + triton.cdiv(k, 2) * triton.cdiv(n, 1024),
                )
            ](
                input,
                weight,
                a_bf,
                b_bf,
                m,
                k,
                k,
                n,
                input.stride(0),
                input.stride(1),
                weight.stride(0),
                weight.stride(1),
                RPP=2,
                BLOCK_COLS=1024,
                num_warps=1,
                num_stages=2,
            )

    if 16 < m <= 1024 and k >= 4096 and n > 2048:
        splits, bm, bn, bk, warps, stages = _splitk_cfg(m, n)
        even_m = m % bm == 0
        even_n = n % bn == 0
        if not (even_m and even_n):
            bm, bn, bk, warps, stages = 128, 128, 64, 8, 4
        k_per = triton.cdiv(triton.cdiv(k, splits), bk) * bk
        if k_per * splits <= 2 * k:
            ws = torch.empty(
                (splits, m, n), dtype=torch.bfloat16, device=input.device
            )
            _splitk_part_kernel[
                (triton.cdiv(m, bm) * triton.cdiv(n, bn) * splits,)
            ](
                a_bf,
                b_bf,
                ws,
                m,
                n,
                k,
                a_bf.stride(0),
                a_bf.stride(1),
                b_bf.stride(0),
                b_bf.stride(1),
                ws.stride(0),
                ws.stride(1),
                ws.stride(2),
                BLOCK_M=bm,
                BLOCK_N=bn,
                BLOCK_K=bk,
                SPLIT_K=splits,
                EVEN_M=even_m,
                EVEN_N=even_n,
                EVEN_K=(k % bk == 0),
                num_warps=warps,
                num_stages=stages,
            )
            rbm, rbn, rw, rs = _SPLITK_REDUCE_TILE
            _splitk_reduce_kernel[(triton.cdiv(m, rbm), triton.cdiv(n, rbn))](
                ws,
                scale_a,
                scale_b,
                bias if bias is not None else scale_a,
                out,
                m,
                n,
                ws.stride(0),
                ws.stride(1),
                ws.stride(2),
                scale_a.stride(0),
                scale_b.stride(0),
                bias.stride(0) if bias is not None else 0,
                bias.stride(1) if bias is not None else 0,
                out.stride(0),
                out.stride(1),
                SPLIT_K=splits,
                BLOCK_M=rbm,
                BLOCK_N=rbn,
                HAS_BIAS=(bias is not None),
                SA_SCALAR=(scale_a.shape[0] == 1),
                SB_SCALAR=(scale_b.shape[0] == 1),
                num_warps=rw,
                num_stages=rs,
            )
            return out

    bm, bn, bk, group_m, warps, stages = _gemm_config(m, n)
    if bm > 128 and not (m % bm == 0 and n % bn == 0):
        bm, bn, bk, group_m, warps, stages = 128, 128, 64, 8, 8, 4
    grid = (triton.cdiv(m, bm) * triton.cdiv(n, bn),)
    _scaled_mm_kernel[grid](
        a_bf,
        b_bf,
        scale_a,
        scale_b,
        bias if bias is not None else scale_a,
        out,
        m,
        n,
        k,
        a_bf.stride(0),
        a_bf.stride(1),
        b_bf.stride(0),
        b_bf.stride(1),
        scale_a.stride(0),
        scale_b.stride(0),
        bias.stride(0) if bias is not None else 0,
        bias.stride(1) if bias is not None else 0,
        out.stride(0),
        out.stride(1),
        BLOCK_M=bm,
        BLOCK_N=bn,
        BLOCK_K=bk,
        GROUP_M=group_m,
        EVEN_M=(m % bm == 0),
        EVEN_N=(n % bn == 0),
        EVEN_K=(k % bk == 0),
        HAS_BIAS=(bias is not None),
        SA_SCALAR=(scale_a.shape[0] == 1),
        SB_SCALAR=(scale_b.shape[0] == 1),
        num_warps=warps,
        num_stages=stages,
    )
    return out


__all__ = ["triton_scaled_mm"]
