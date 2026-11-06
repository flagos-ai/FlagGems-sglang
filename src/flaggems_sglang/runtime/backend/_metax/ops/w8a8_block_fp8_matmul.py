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

"""Triton w8a8 block fp8 matmul for MetaX (block-wise FP8 GEMM)."""

import torch
import triton
import triton.language as tl


@triton.jit
def _fp8_to_bf16_kernel(
    x_ptr,
    y_ptr,
    R,
    K,
    stride_xr,
    stride_xk,
    stride_yr,
    stride_yk,
    BLOCK_R: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Widen a [R, K] fp8 tensor to bf16 (bit-exact) on the way to the copy."""
    pid_r = tl.program_id(0)
    pid_k = tl.program_id(1)

    offs_r = pid_r * BLOCK_R + tl.arange(0, BLOCK_R)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    mask = (offs_r < R)[:, None] & (offs_k < K)[None, :]

    x = tl.load(
        x_ptr + offs_r[:, None] * stride_xr + offs_k[None, :] * stride_xk,
        mask=mask,
        other=0.0,
    )
    tl.store(
        y_ptr + offs_r[:, None] * stride_yr + offs_k[None, :] * stride_yk,
        x.to(tl.bfloat16),
        mask=mask,
    )


@triton.jit
def _zero_fill_kernel(
    c_ptr,
    numel,
    BLOCK: tl.constexpr,
):
    """Zero-fill the output for a K == 0 call (empty reduction over k)."""
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    zero = tl.zeros((BLOCK,), dtype=tl.int32).to(c_ptr.dtype.element_ty)
    tl.store(c_ptr + offs, zero, mask=mask)


@triton.jit
def _w8a8_block_fp8_matmul_kernel(
    a_ptr,
    b_ptr,
    as_ptr,
    bs_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bn,
    stride_bk,
    stride_asm,
    stride_ask,
    stride_bsn,
    stride_bsk,
    stride_cm,
    stride_cn,
    scale_block_n,
    scale_block_k,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    EVEN_K: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    m_mask = offs_m < M
    n_mask = offs_n < N
    # Row of Bs shared by every n in this tile (BLOCK_N <= block_n).
    n_scale_row = (pid_n * BLOCK_N) // scale_block_n

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_base in range(0, K, BLOCK_K):
        k_idx = k_base + offs_k
        a_ptrs = (
            a_ptr + offs_m[:, None] * stride_am + k_idx[None, :] * stride_ak
        )
        # B is [N, K]; address it as a [BLOCK_K, BLOCK_N] dot operand directly.
        b_ptrs = (
            b_ptr + k_idx[:, None] * stride_bk + offs_n[None, :] * stride_bn
        )

        if EVEN_K:
            a = tl.load(a_ptrs, mask=m_mask[:, None], other=0.0)
            b = tl.load(b_ptrs, mask=n_mask[None, :], other=0.0)
        else:
            k_mask = k_idx < K
            a = tl.load(
                a_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=0.0
            )
            b = tl.load(
                b_ptrs, mask=k_mask[:, None] & n_mask[None, :], other=0.0
            )

        # Whole tile shares one scale pair, applied in fp32 after the MAC.
        s_col = k_base // scale_block_k
        a_s = tl.load(
            as_ptr + offs_m * stride_asm + s_col * stride_ask,
            mask=m_mask,
            other=0.0,
        )
        b_s = tl.load(bs_ptr + n_scale_row * stride_bsn + s_col * stride_bsk)
        acc += tl.dot(a, b) * (a_s * b_s)[:, None]

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(
        c_ptrs,
        acc.to(c_ptr.dtype.element_ty),
        mask=m_mask[:, None] & n_mask[None, :],
    )


def w8a8_block_fp8_matmul(A, B, As, Bs, block_size, output_dtype):
    block_n, block_k = block_size
    M, K = A.shape
    N = B.shape[0]

    c = torch.empty((M, N), dtype=output_dtype, device=A.device)
    if M == 0 or N == 0:
        return c
    if K == 0:
        # The k reduction is empty: the definition yields an all-zero output.
        _zero_fill_kernel[(triton.cdiv(c.numel(), 4096),)](
            c,
            c.numel(),
            BLOCK=4096,
            num_warps=4,
        )
        return c

    a_bf16 = torch.empty((M, K), dtype=torch.bfloat16, device=A.device)
    b_bf16 = torch.empty((N, K), dtype=torch.bfloat16, device=B.device)
    _fp8_to_bf16_kernel[(triton.cdiv(M, 2), triton.cdiv(K, 1024))](
        A,
        a_bf16,
        M,
        K,
        A.stride(0),
        A.stride(1),
        a_bf16.stride(0),
        a_bf16.stride(1),
        BLOCK_R=2,
        BLOCK_K=1024,
        num_warps=4,
    )
    _fp8_to_bf16_kernel[(triton.cdiv(N, 2), triton.cdiv(K, 1024))](
        B,
        b_bf16,
        N,
        K,
        B.stride(0),
        B.stride(1),
        b_bf16.stride(0),
        b_bf16.stride(1),
        BLOCK_R=2,
        BLOCK_K=1024,
        num_warps=4,
    )

    if M <= 16:
        bm, bn, bk, group_m, warps, stages = 16, 32, 128, 1, 4, 4
    elif M <= 64:
        bm, bn, bk, group_m, warps, stages = 64, 64, 64, 8, 8, 4
    else:
        bm, bk, group_m, warps, stages = 128, 64, 8, 8, 2
        bn = 64 if N <= 2048 else 128

    # A k tile must sit inside one k-scale block and an n tile inside one
    # n-scale block, so shrink both (power-of-two steps) until they fit.
    while block_k % bk != 0:
        bk //= 2
    while block_n % bn != 0:
        bn //= 2

    grid = (triton.cdiv(M, bm) * triton.cdiv(N, bn),)
    _w8a8_block_fp8_matmul_kernel[grid](
        a_bf16,
        b_bf16,
        As,
        Bs,
        c,
        M,
        N,
        K,
        a_bf16.stride(0),
        a_bf16.stride(1),
        b_bf16.stride(0),
        b_bf16.stride(1),
        As.stride(0),
        As.stride(1),
        Bs.stride(0),
        Bs.stride(1),
        c.stride(0),
        c.stride(1),
        block_n,
        block_k,
        BLOCK_M=bm,
        BLOCK_N=bn,
        BLOCK_K=bk,
        GROUP_SIZE_M=group_m,
        EVEN_K=(K % bk == 0),
        num_warps=warps,
        num_stages=stages,
    )
    return c


__all__ = ["w8a8_block_fp8_matmul"]
