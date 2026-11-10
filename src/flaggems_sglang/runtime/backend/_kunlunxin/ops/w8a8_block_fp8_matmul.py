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
def _w8a8_gemm_kernel_kl(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    sam,
    sak,
    sbn,
    sbk,
    scm,
    scn,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    GROUP_M: tl.constexpr,
    EVEN_K: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BM)
    num_pid_n = tl.cdiv(N, BN)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    rm = pid_m * BM + tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    wrapped_m = tl.max_contiguous(tl.multiple_of(rm % M, BM), BM)
    wrapped_n = tl.max_contiguous(tl.multiple_of(rn % N, BN), BN)
    a_ptrs = a_ptr + wrapped_m[:, None] * sam + rk[None, :] * sak
    b_ptrs = b_ptr + wrapped_n[None, :] * sbn + rk[:, None] * sbk

    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, K, BK):
        if EVEN_K:
            a = tl.load(a_ptrs)
            b = tl.load(b_ptrs)
        else:
            mk = rk < K - k0
            a = tl.load(a_ptrs, mask=mk[None, :], other=0)
            b = tl.load(b_ptrs, mask=mk[:, None], other=0)
        acc = tl.dot(a, b, acc, input_precision="ieee", out_dtype=tl.float32)
        a_ptrs += BK * sak
        b_ptrs += BK * sbk

    c_ptrs = c_ptr + rm[:, None] * scm + rn[None, :] * scn
    tl.store(c_ptrs, acc, mask=(rm[:, None] < M) & (rn[None, :] < N))


def _pow2_at_most(value, cap):
    size = 16
    while size * 2 <= value and size * 2 <= cap:
        size *= 2
    return size


def _decode_e4m3(u8):
    u = u8.to(torch.int32)
    sign = (u >> 7) & 1
    e = (u >> 3) & 15
    m = u & 7
    is_norm = (e > 0).to(torch.int32)
    val = (m + 8.0 * is_norm) * torch.exp2(e.to(torch.float32) - 9.0 - is_norm)
    return val * (1.0 - 2.0 * sign)


def _prepack_fp32(u8, scale, block_row, block_k):
    vals = _decode_e4m3(u8.contiguous().view(torch.uint8))
    R, K = vals.shape
    sc = scale.contiguous().repeat_interleave(block_k, dim=1)
    if sc.shape[1] < K:
        sc = torch.cat([sc, sc.new_ones(sc.shape[0], K - sc.shape[1])], dim=1)
    sc = sc.repeat_interleave(block_row, dim=0)
    if sc.shape[0] < R:
        sc = torch.cat([sc, sc.new_ones(R - sc.shape[0], K)], dim=0)
    return vals * sc[:R, :K]


def w8a8_block_fp8_matmul(A, B, As, Bs, block_size, output_dtype):
    block_n = int(block_size[0])
    block_k = int(block_size[1])
    A2 = _prepack_fp32(A, As, 1, block_k)
    B2 = _prepack_fp32(B, Bs, block_n, block_k)
    M, K = A2.shape
    N = B2.shape[0]
    if M == 0 or N == 0:
        return torch.empty((M, N), dtype=output_dtype, device=A2.device)

    BM = 64
    BN = _pow2_at_most(N, 64)
    BK = 32 if block_k % 32 == 0 else 16
    even_k = K % BK == 0
    grid = (triton.cdiv(M, BM) * triton.cdiv(N, BN),)
    tmp = torch.empty((M, N), dtype=torch.float32, device=A2.device)
    _w8a8_gemm_kernel_kl[grid](
        A2,
        B2,
        tmp,
        M,
        N,
        K,
        A2.stride(0),
        A2.stride(1),
        B2.stride(0),
        B2.stride(1),
        tmp.stride(0),
        tmp.stride(1),
        BM=BM,
        BN=BN,
        BK=BK,
        GROUP_M=8,
        EVEN_K=even_k,
        num_warps=4,
        num_stages=2,
    )
    return tmp.to(output_dtype)


__all__ = ["w8a8_block_fp8_matmul"]
