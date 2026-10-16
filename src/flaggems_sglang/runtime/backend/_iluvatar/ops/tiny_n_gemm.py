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

from flaggems_sglang import device


@triton.jit
def _tiny_n_gemm_fused_kernel(
    x_ptr,
    w_ptr,
    part_ptr,
    cnt_ptr,
    out_ptr,
    M,
    N,
    K,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    S: tl.constexpr,
    EVEN_M: tl.constexpr,
    IEEE: tl.constexpr,
):
    pid = tl.program_id(0)
    kid = tl.program_id(1)

    offs_m = tl.arange(0, BM)
    offs_n = pid * BN + tl.arange(0, BN)

    k_per = K // S
    k_start = kid * k_per

    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(k_start, k_start + k_per, BK):
        offs_k = k0 + tl.arange(0, BK)
        if EVEN_M:
            x = tl.load(x_ptr + offs_m[:, None] * K + offs_k[None, :])
        else:
            x = tl.load(
                x_ptr + offs_m[:, None] * K + offs_k[None, :],
                mask=offs_m[:, None] < M,
                other=0.0,
            )
        w = tl.load(w_ptr + offs_n[None, :] * K + offs_k[:, None])
        if IEEE:
            acc += tl.dot(x, w, input_precision="ieee")
        else:
            acc += tl.dot(x, w)

    offs_out = offs_m[:, None] * N + offs_n[None, :]
    out_mask = offs_m[:, None] < M
    tl.store(part_ptr + kid * M * N + offs_out, acc, mask=out_mask)

    old = tl.atomic_add(cnt_ptr + pid, 1)
    if old == S - 1:
        red = tl.zeros((BM, BN), dtype=tl.float32)
        for s in range(S):
            red += tl.load(
                part_ptr + s * M * N + offs_out, mask=out_mask, other=0.0
            )
        tl.store(
            out_ptr + offs_out,
            red.to(out_ptr.dtype.element_ty),
            mask=out_mask,
        )
        tl.store(cnt_ptr + pid, 0)


@triton.jit
def _tiny_n_gemm_kernel(
    x_ptr,
    w_ptr,
    out_ptr,
    M,
    N,
    K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    IEEE: tl.constexpr,
):
    pid = tl.program_id(0)

    offs_m = tl.arange(0, BLOCK_M)
    offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        k_mask = offs_k < K
        x = tl.load(
            x_ptr + offs_m[:, None] * K + offs_k[None, :],
            mask=(offs_m[:, None] < M) & k_mask[None, :],
            other=0.0,
        )
        w = tl.load(
            w_ptr + offs_n[None, :] * K + offs_k[:, None],
            mask=(offs_n[None, :] < N) & k_mask[:, None],
            other=0.0,
        )
        if IEEE:
            acc += tl.dot(x, w, input_precision="ieee")
        else:
            acc += tl.dot(x, w)

    out = acc.to(out_ptr.dtype.element_ty)
    tl.store(
        out_ptr + offs_m[:, None] * N + offs_n[None, :],
        out,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


def tiny_n_gemm(x, w, out_dtype):
    m, k = x.shape
    n = w.shape[0]

    if not x.is_contiguous():
        x = x.contiguous()
    if not w.is_contiguous():
        w = w.contiguous()

    block_m = max(16, triton.next_power_of_2(m))
    out = torch.empty((m, n), device=device, dtype=out_dtype)

    ieee = x.dtype == torch.float32

    if n >= 1024:
        block_n, num_warps = 64, 4
    else:
        block_n, num_warps = 32, 2
    block_k = 128
    split_k = 8

    if (
        not ieee
        and n % block_n == 0
        and k % split_k == 0
        and (k // split_k) % block_k == 0
    ):
        part = torch.empty((split_k, m, n), device=device, dtype=torch.float32)
        counter = torch.zeros(n // block_n, device=device, dtype=torch.int32)
        _tiny_n_gemm_fused_kernel[(n // block_n, split_k)](
            x,
            w,
            part,
            counter,
            out,
            m,
            n,
            k,
            BM=block_m,
            BN=block_n,
            BK=block_k,
            S=split_k,
            EVEN_M=(block_m == m),
            IEEE=ieee,
            num_warps=num_warps,
            num_stages=2,
        )
    else:
        block_n, block_k = 32, 64
        _tiny_n_gemm_kernel[(triton.cdiv(n, block_n),)](
            x,
            w,
            out,
            m,
            n,
            k,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_K=block_k,
            IEEE=ieee,
            num_warps=4,
            num_stages=2,
        )
    return out


__all__ = ["tiny_n_gemm"]
