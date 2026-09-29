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
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import triton
import triton.language as tl


@triton.jit
def _w8a8_int8_dot(
    A,
    B,
    C,
    As,
    Bs,
    group_n,
    group_k,
    stride_am,
    stride_ak,
    stride_bn,
    stride_bk,
    stride_cm,
    stride_cn,
    stride_As_m,
    stride_As_k,
    stride_Bs_n,
    stride_Bs_k,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = A + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = B + offs_n[None, :] * stride_bn + offs_k[:, None] * stride_bk

    As_ptrs = As + offs_m * stride_As_m
    offs_bsn = offs_n // group_n
    Bs_ptrs = Bs + offs_bsn * stride_Bs_n

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_remaining = K - k * BLOCK_K
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < k_remaining)
        b_mask = (offs_k[:, None] < k_remaining) & (offs_n[None, :] < N)
        a = tl.load(a_ptrs, mask=a_mask, other=0)
        b = tl.load(b_ptrs, mask=b_mask, other=0)
        offs_ks = (k * BLOCK_K) // group_k
        a_s = tl.load(
            As_ptrs + offs_ks * stride_As_k, mask=offs_m < M, other=0.0
        )
        b_s = tl.load(
            Bs_ptrs + offs_ks * stride_Bs_k, mask=offs_n < N, other=0.0
        )
        prod = tl.dot(a, b, out_dtype=tl.int32)
        acc += prod.to(tl.float32) * a_s[:, None] * b_s[None, :]
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c = acc.to(C.dtype.element_ty)
    c_ptrs = C + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, c, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def _pow2_at_most(value, cap):
    size = 16
    while size * 2 <= value and size * 2 <= cap:
        size *= 2
    return size


def w8a8_block_int8_matmul(A, B, As, Bs, block_size, output_dtype):
    A = A.contiguous()
    B = B.contiguous()
    As = As.contiguous()
    Bs = Bs.contiguous()
    block_n, block_k = int(block_size[0]), int(block_size[1])
    M, K = A.shape
    N = B.shape[0]
    C = torch.empty((M, N), device=A.device, dtype=output_dtype)
    if M == 0 or N == 0:
        return C

    BLOCK_M = _pow2_at_most(M, 64)
    BLOCK_N = 64 if block_n >= 64 else block_n
    BLOCK_K = block_k
    GROUP_M = 8
    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)
    _w8a8_int8_dot[grid](
        A,
        B,
        C,
        As,
        Bs,
        block_n,
        block_k,
        A.stride(0),
        A.stride(1),
        B.stride(0),
        B.stride(1),
        C.stride(0),
        C.stride(1),
        As.stride(0),
        As.stride(1),
        Bs.stride(0),
        Bs.stride(1),
        M=M,
        N=N,
        K=K,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        GROUP_M=GROUP_M,
        num_warps=4,
        num_stages=3,
    )
    return C


__all__ = ["w8a8_block_int8_matmul"]
