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

_THIN_CONFIGS = tuple(
    triton.Config(
        {"BLOCK_SIZE_M": bm, "BLOCK_SIZE_N": bn},
        num_warps=warps,
        num_stages=stages,
    )
    for bm, bn, warps, stages in (
        (16, 32, 4, 3),
        (16, 64, 4, 3),
        (16, 64, 4, 4),
        (16, 128, 4, 2),
        (16, 128, 4, 3),
        (16, 128, 8, 3),
        (16, 256, 8, 1),
        (32, 64, 4, 3),
        (32, 128, 4, 2),
        (32, 128, 4, 3),
        (32, 256, 8, 1),
        (64, 64, 4, 3),
        (64, 128, 4, 2),
        (64, 128, 8, 2),
        (64, 256, 8, 1),
    )
)


@triton.autotune(configs=_THIN_CONFIGS, key=["M", "N", "K"])
@triton.jit
def _w8a8_block_fp8_thin_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    a_s_ptr,
    b_s_ptr,
    M,
    N,
    K,
    group_n,
    split_k,
    stride_am,
    stride_bn,
    stride_asm,
    stride_bsn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    GROUP_K: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    pid_n = pid % num_pid_n
    rest = pid // num_pid_n
    pid_m = rest % num_pid_m
    pid_k = rest // num_pid_m

    k_tiles_total = K // GROUP_K
    tiles_per_split = (k_tiles_total + split_k - 1) // split_k
    k_first = pid_k * tiles_per_split
    k_last = tl.minimum(k_tiles_total, k_first + tiles_per_split)

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    m_mask = offs_m < M
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    n_mask = offs_n < N
    offs_k = tl.arange(0, GROUP_K)

    k0 = k_first * GROUP_K
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + k0 + offs_k[None, :]
    b_ptrs = b_ptr + offs_n[None, :] * stride_bn + k0 + offs_k[:, None]
    a_s_base = a_s_ptr + offs_m * stride_asm
    b_s_base = b_s_ptr + (offs_n // group_n) * stride_bsn

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for i in range(k_first, k_last):
        a_s = tl.load(a_s_base + i, mask=m_mask, other=0.0)
        b_s = tl.load(b_s_base + i, mask=n_mask, other=0.0)
        a = tl.load(a_ptrs, mask=m_mask[:, None], other=0.0)
        b = tl.load(b_ptrs, mask=n_mask[None, :], other=0.0)
        accumulator += tl.dot(a.to(tl.bfloat16), b.to(tl.bfloat16)) * (
            a_s[:, None] * b_s[None, :]
        )
        a_ptrs += GROUP_K
        b_ptrs += GROUP_K

    c_ptrs = c_ptr + pid_k * M * N + offs_m[:, None] * N + offs_n[None, :]
    tl.store(
        c_ptrs,
        accumulator.to(c_ptr.dtype.element_ty),
        mask=m_mask[:, None] & n_mask[None, :],
    )


@triton.jit
def _thin_reduce_kernel(
    src_ptr,
    dst_ptr,
    total,
    split_k,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < total
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for i in range(split_k):
        acc += tl.load(src_ptr + i * total + offs, mask=mask, other=0.0)
    tl.store(dst_ptr + offs, acc.to(dst_ptr.dtype.element_ty), mask=mask)


_SPLITK_CONFIGS = tuple(
    triton.Config(
        {
            "SPLIT_K": sk,
            "BLOCK_SIZE_M": bm,
            "BLOCK_SIZE_N": bn,
            "BLOCK_SIZE_K": bk,
        },
        num_warps=warps,
        num_stages=stages,
    )
    for sk, bm, bn, bk, warps, stages in (
        (16, 16, 128, 128, 4, 2),
        (8, 16, 128, 128, 4, 2),
        (8, 16, 128, 64, 4, 2),
        (4, 16, 128, 128, 4, 2),
        (4, 16, 128, 64, 4, 2),
        (16, 16, 256, 128, 8, 2),
        (8, 16, 256, 128, 8, 2),
        (8, 16, 64, 128, 4, 2),
        (8, 64, 128, 128, 4, 2),
        (8, 64, 64, 128, 4, 2),
        (4, 64, 128, 128, 4, 2),
        (8, 64, 128, 256, 8, 2),
        (2, 64, 128, 128, 4, 2),
        (1, 64, 128, 128, 4, 2),
    )
)


@triton.autotune(
    configs=_SPLITK_CONFIGS,
    key=["M", "N", "K"],
    reset_to_zero=["c_ptr"],
)
@triton.jit
def _w8a8_block_fp8_splitk_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    a_s_ptr,
    b_s_ptr,
    M,
    N,
    K,
    group_n,
    group_k,
    stride_am,
    stride_bn,
    stride_asm,
    stride_bsn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    SPLIT_K: tl.constexpr,
    EVEN_K: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    pid_k = pid % SPLIT_K
    pid_rest = pid // SPLIT_K
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    pid_m = pid_rest // num_pid_n
    pid_n = pid_rest % num_pid_n

    offs_m = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    k_tiles_total = tl.cdiv(K, BLOCK_SIZE_K)
    k_tiles_per_split = tl.cdiv(k_tiles_total, SPLIT_K)
    k_first = pid_k * k_tiles_per_split
    k_last = min(k_tiles_total, k_first + k_tiles_per_split)
    k_offset = k_first * BLOCK_SIZE_K
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + (k_offset + offs_k[None, :])
    b_ptrs = b_ptr + offs_n[None, :] * stride_bn + (k_offset + offs_k[:, None])
    a_s_base = a_s_ptr + offs_m * stride_asm
    b_s_base = b_s_ptr + (offs_n // group_n) * stride_bsn

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for i in range(k_first, k_last):
        s_idx = i * BLOCK_SIZE_K // group_k
        a_s = tl.load(a_s_base + s_idx)
        b_s = tl.load(b_s_base + s_idx)
        if EVEN_K:
            a = tl.load(a_ptrs)
            b = tl.load(b_ptrs)
        else:
            k_rem = K - i * BLOCK_SIZE_K
            a = tl.load(a_ptrs, mask=offs_k[None, :] < k_rem, other=0.0)
            b = tl.load(b_ptrs, mask=offs_k[:, None] < k_rem, other=0.0)
        accumulator += tl.dot(a.to(tl.bfloat16), b.to(tl.bfloat16)) * (
            a_s[:, None] * b_s[None, :]
        )
        a_ptrs += BLOCK_SIZE_K
        b_ptrs += BLOCK_SIZE_K

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    c_ptrs = c_ptr + offs_cm[:, None] * N + offs_n[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_n[None, :] < N)
    tl.atomic_add(c_ptrs, accumulator, mask=c_mask)


_WIDEN_CONFIGS = tuple(
    triton.Config(
        {"BLOCK_ROWS": rm, "BLOCK_BLOCKS": rb},
        num_warps=warps,
        num_stages=1,
    )
    for rm, rb, warps in (
        (2, 32, 8),
        (2, 16, 8),
        (4, 16, 8),
        (4, 8, 4),
        (8, 16, 8),
        (16, 8, 4),
        (16, 16, 8),
        (32, 8, 4),
        (32, 4, 4),
        (64, 4, 4),
        (64, 2, 4),
        (128, 2, 8),
        (1, 32, 8),
        (1, 64, 8),
        (256, 1, 8),
    )
)

_BLOCK_K = 128


@triton.autotune(configs=_WIDEN_CONFIGS, key=["R", "K"])
@triton.jit
def _w8a8_widen_kernel(
    src_ptr,
    o_ptr,
    R,
    K,
    stride_sr,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_BLOCKS: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_r = tl.program_id(0)
    pid_b = tl.program_id(1)
    rows = pid_r * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    row_mask = rows < R
    n_blocks = tl.cdiv(K, BLOCK_K)
    blocks = pid_b * BLOCK_BLOCKS + tl.arange(0, BLOCK_BLOCKS)
    blk_mask = blocks < n_blocks
    offs_k = tl.arange(0, BLOCK_K)
    base = (
        rows[:, None, None] * stride_sr
        + blocks[None, :, None] * BLOCK_K
        + offs_k[None, None, :]
    )
    mask = row_mask[:, None, None] & blk_mask[None, :, None]
    x = tl.load(src_ptr + base, mask=mask, other=0.0)
    tl.store(o_ptr + base, x.to(tl.bfloat16), mask=mask)


_GEMM_CONFIGS = tuple(
    triton.Config(
        {"BLOCK_SIZE_M": bm, "BLOCK_SIZE_N": bn, "BLOCK_SIZE_K": 64},
        num_warps=warps,
        num_stages=stages,
    )
    for bm, bn, warps, stages in (
        (256, 128, 8, 1),
        (128, 256, 8, 1),
        (128, 128, 8, 1),
        (128, 128, 8, 2),
        (128, 128, 4, 1),
        (128, 128, 4, 2),
        (256, 128, 4, 1),
        (64, 256, 8, 1),
        (64, 128, 8, 1),
        (64, 128, 4, 2),
        (128, 64, 4, 2),
        (64, 64, 4, 2),
    )
)

_GEMM_GROUP_M = 8


@triton.autotune(configs=_GEMM_CONFIGS, key=["M", "N", "K"])
@triton.jit
def _w8a8_block_fp8_gemm_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    a_s_ptr,
    b_s_ptr,
    M,
    N,
    K,
    group_n,
    stride_asm,
    stride_bsn,
    DOTS_PER_SCALE: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    cols = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_n = cols % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + offs_m[:, None] * K + offs_k[None, :]
    b_ptrs = b_ptr + offs_k[:, None] + offs_n[None, :] * K
    a_s_base = a_s_ptr + offs_m * stride_asm
    b_s_base = b_s_ptr + (cols // group_n) * stride_bsn

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for i in range(0, tl.cdiv(K, BLOCK_SIZE_K * DOTS_PER_SCALE)):
        a_s = tl.load(a_s_base + i)
        b_s = tl.load(b_s_base + i, mask=cols < N, other=0.0)
        part = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for _ in tl.static_range(DOTS_PER_SCALE):
            a = tl.load(a_ptrs)
            b = tl.load(b_ptrs)
            part = tl.dot(a, b, part, out_dtype=tl.float32)
            a_ptrs += BLOCK_SIZE_K
            b_ptrs += BLOCK_SIZE_K
        accumulator += part * (a_s[:, None] * b_s[None, :])

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + offs_cm[:, None] * N + offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator.to(c_ptr.dtype.element_ty), mask=c_mask)


def _thin_split_k(M, N, K, block_k):
    est_progs = triton.cdiv(M, 16) * triton.cdiv(N, 64)
    split_k = 4
    while split_k < 8 and est_progs * split_k < 256:
        split_k *= 2
    return min(split_k, K // block_k)


def w8a8_block_fp8_matmul(A, B, As, Bs, block_size, output_dtype):
    block_n, block_k = block_size
    M, K = A.shape
    N, _ = B.shape

    even_k = K % block_k == 0
    pow2_gk = block_k >= 16 and (block_k & (block_k - 1)) == 0

    if M <= 64:
        if even_k and pow2_gk:
            split_k = _thin_split_k(M, N, K, block_k)

            def grid(meta):
                return (
                    triton.cdiv(M, meta["BLOCK_SIZE_M"])
                    * triton.cdiv(N, meta["BLOCK_SIZE_N"])
                    * split_k,
                )

            if split_k == 1:
                C = torch.empty((M, N), dtype=output_dtype, device=A.device)
                _w8a8_block_fp8_thin_kernel[grid](
                    A,
                    B,
                    C,
                    As,
                    Bs,
                    M,
                    N,
                    K,
                    block_n,
                    split_k,
                    A.stride(0),
                    B.stride(0),
                    As.stride(0),
                    Bs.stride(0),
                    GROUP_K=block_k,
                )
                return C
            C32 = torch.empty(
                (split_k, M, N), dtype=torch.float32, device=A.device
            )
            _w8a8_block_fp8_thin_kernel[grid](
                A,
                B,
                C32,
                As,
                Bs,
                M,
                N,
                K,
                block_n,
                split_k,
                A.stride(0),
                B.stride(0),
                As.stride(0),
                Bs.stride(0),
                GROUP_K=block_k,
            )
            C = torch.empty((M, N), dtype=output_dtype, device=A.device)
            total = M * N
            _thin_reduce_kernel[(triton.cdiv(total, 4096),)](
                C32,
                C,
                total,
                split_k,
                BLOCK=4096,
                num_warps=8,
            )
            return C

        C32 = torch.zeros((M, N), dtype=torch.float32, device=A.device)

        def grid(meta):
            return (
                triton.cdiv(M, meta["BLOCK_SIZE_M"])
                * triton.cdiv(N, meta["BLOCK_SIZE_N"])
                * meta["SPLIT_K"],
            )

        _w8a8_block_fp8_splitk_kernel[grid](
            A,
            B,
            C32,
            As,
            Bs,
            M,
            N,
            K,
            block_n,
            block_k,
            A.stride(0),
            B.stride(0),
            As.stride(0),
            Bs.stride(0),
            EVEN_K=even_k,
        )
        return C32.to(output_dtype)

    Abf = torch.empty((M, K), dtype=torch.bfloat16, device=A.device)
    Bbf = torch.empty((N, K), dtype=torch.bfloat16, device=B.device)

    def grid_a(meta):
        return (
            triton.cdiv(M, meta["BLOCK_ROWS"]),
            triton.cdiv(K, meta["BLOCK_BLOCKS"] * _BLOCK_K),
        )

    def grid_b(meta):
        return (
            triton.cdiv(N, meta["BLOCK_ROWS"]),
            triton.cdiv(K, meta["BLOCK_BLOCKS"] * _BLOCK_K),
        )

    _w8a8_widen_kernel[grid_a](A, Abf, M, K, A.stride(0), BLOCK_K=_BLOCK_K)
    _w8a8_widen_kernel[grid_b](B, Bbf, N, K, B.stride(0), BLOCK_K=_BLOCK_K)
    C = torch.empty((M, N), dtype=output_dtype, device=A.device)

    def grid(meta):
        return (
            triton.cdiv(M, meta["BLOCK_SIZE_M"])
            * triton.cdiv(N, meta["BLOCK_SIZE_N"]),
        )

    _w8a8_block_fp8_gemm_kernel[grid](
        Abf,
        Bbf,
        C,
        As,
        Bs,
        M,
        N,
        K,
        block_n,
        As.stride(0),
        Bs.stride(0),
        DOTS_PER_SCALE=block_k // 64,
        GROUP_M=_GEMM_GROUP_M,
    )
    return C


__all__ = ["w8a8_block_fp8_matmul"]
