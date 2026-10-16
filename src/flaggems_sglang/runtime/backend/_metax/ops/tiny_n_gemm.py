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
def _tiny_n_gemm_kernel(
    x_ptr,
    w_ptr,
    y_ptr,
    M,
    N,
    K,
    K_PER_SPLIT,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SPLIT_K: tl.constexpr,
    FULL_K: tl.constexpr,
    EVEN_N: tl.constexpr,
    EVEN_M: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_n = pid // SPLIT_K
    pid_k = pid % SPLIT_K

    offs_m = tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = offs_m < M
    mask_n = offs_n < N

    k_start = pid_k * K_PER_SPLIT
    k_end = tl.minimum(k_start + K_PER_SPLIT, K)

    offs_k = tl.arange(0, BLOCK_K)
    k_off = k_start + offs_k
    x_ptrs = x_ptr + offs_m[:, None] * K + k_off[None, :]
    w_ptrs = w_ptr + k_off[:, None] + offs_n[None, :] * K

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(k_start, k_end, BLOCK_K):
        if FULL_K:
            if EVEN_M:
                a = tl.load(x_ptrs)
            else:
                a = tl.load(x_ptrs, mask=mask_m[:, None], other=0.0)
            if EVEN_N:
                b = tl.load(w_ptrs)
            else:
                b = tl.load(w_ptrs, mask=mask_n[None, :], other=0.0)
        else:
            k_mask = k0 + offs_k < k_end
            if EVEN_M:
                a = tl.load(x_ptrs, mask=k_mask[None, :], other=0.0)
            else:
                a = tl.load(
                    x_ptrs, mask=mask_m[:, None] & k_mask[None, :], other=0.0
                )
            b = tl.load(
                w_ptrs, mask=k_mask[:, None] & mask_n[None, :], other=0.0
            )
        acc = tl.dot(a, b, acc)
        x_ptrs += BLOCK_K
        w_ptrs += BLOCK_K

    if EVEN_M:
        out_mask = mask_n[None, :]
    else:
        out_mask = mask_m[:, None] & mask_n[None, :]
    if SPLIT_K == 1:
        tl.store(
            y_ptr + offs_m[:, None] * N + offs_n[None, :],
            acc,
            mask=out_mask,
        )
    else:
        base = pid_k * M * N
        tl.store(
            y_ptr + base + offs_m[:, None] * N + offs_n[None, :],
            acc,
            mask=out_mask,
        )


@triton.jit
def _split_k_reduce_kernel(
    partial_ptr,
    out_ptr,
    MN,
    SPLIT_K: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < MN
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for s in range(SPLIT_K):
        acc += tl.load(partial_ptr + s * MN + offs, mask=mask, other=0.0)
    tl.store(out_ptr + offs, acc, mask=mask)


def _pick_config(n, k):
    block_n = 64
    block_k = 128
    block_k = min(block_k, max(triton.next_power_of_2(k), 16))

    n_blocks = triton.cdiv(n, block_n)
    split_k = max(triton.cdiv(96, n_blocks), 1)
    split_k = min(split_k, triton.cdiv(k, block_k), 32)
    return block_n, block_k, split_k, 4, 3


def tiny_n_gemm(x, w, out_dtype):
    m, k = x.shape
    n = w.shape[0]
    out = torch.empty((m, n), dtype=out_dtype, device=x.device)

    block_n, block_k, split_k, num_warps, num_stages = _pick_config(n, k)

    k_per_split = triton.cdiv(triton.cdiv(k, split_k), block_k) * block_k
    split_k = triton.cdiv(k, k_per_split)
    n_blocks = triton.cdiv(n, block_n)
    grid = (n_blocks * split_k,)

    full_k = k % block_k == 0
    even_n = n % block_n == 0
    even_m = m == 16

    if split_k == 1:
        _tiny_n_gemm_kernel[grid](
            x,
            w,
            out,
            m,
            n,
            k,
            k_per_split,
            BLOCK_M=16,
            BLOCK_N=block_n,
            BLOCK_K=block_k,
            SPLIT_K=1,
            FULL_K=full_k,
            EVEN_N=even_n,
            EVEN_M=even_m,
            num_warps=num_warps,
            num_stages=num_stages,
        )
        return out

    partial = torch.empty(
        (split_k, m, n), dtype=torch.float32, device=x.device
    )
    _tiny_n_gemm_kernel[grid](
        x,
        w,
        partial,
        m,
        n,
        k,
        k_per_split,
        BLOCK_M=16,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        SPLIT_K=split_k,
        FULL_K=full_k,
        EVEN_N=even_n,
        EVEN_M=even_m,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    mn = m * n
    _split_k_reduce_kernel[(triton.cdiv(mn, 64),)](
        partial,
        out,
        mn,
        SPLIT_K=split_k,
        BLOCK=64,
        num_warps=1,
    )
    return out


__all__ = ["tiny_n_gemm"]
