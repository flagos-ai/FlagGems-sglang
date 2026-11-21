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
def _dsv3_router_gemm_copy2d_kernel(
    src_ptr,
    dst_ptr,
    M,
    K,
    stride_sm,
    stride_sk,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    num_m = tl.cdiv(M, BLOCK_M)
    pid_m = pid % num_m
    pid_k = pid // num_m
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    # Modulo-index trick keeps pointer arithmetic in-bounds for masked lanes.
    offs_m_mod = tl.where(offs_m < M, offs_m, 0)
    offs_k_mod = tl.where(offs_k < K, offs_k, 0)
    mask = (offs_m < M)[:, None] & (offs_k < K)[None, :]
    v = tl.load(
        src_ptr
        + offs_m_mod[:, None] * stride_sm
        + offs_k_mod[None, :] * stride_sk,
        mask=mask,
        other=0.0,
    )
    tl.store(dst_ptr + offs_m[:, None] * K + offs_k[None, :], v, mask=mask)


@triton.jit
def _dsv3_router_gemm_gemv_m1_kernel(
    h_ptr,
    w_ptr,
    c_ptr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_K: tl.constexpr,
    FULL: tl.constexpr,
):
    """Single-launch decode-GEMV for the M == 1 region (the M=1."""
    n = tl.program_id(0)
    acc = tl.zeros((BLOCK_K,), dtype=tl.float32)
    w_row = w_ptr + n * K
    if FULL:
        for k0 in range(0, K, BLOCK_K):
            offs_k = k0 + tl.arange(0, BLOCK_K)
            hv = tl.load(h_ptr + offs_k).to(tl.float32)
            wv = tl.load(w_row + offs_k).to(tl.float32)
            acc += hv * wv
    else:
        for k0 in range(0, K, BLOCK_K):
            offs_k = k0 + tl.arange(0, BLOCK_K)
            mask = offs_k < K
            hv = tl.load(h_ptr + offs_k, mask=mask, other=0.0).to(tl.float32)
            wv = tl.load(w_row + offs_k, mask=mask, other=0.0).to(tl.float32)
            acc += hv * wv
    tl.store(c_ptr + n, tl.sum(acc).to(c_ptr.dtype.element_ty))


@triton.jit
def _dsv3_router_gemm_splitk_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    TILES: tl.constexpr,
):
    """Exact-tiling split-K GEMM: requires K % (BLOCK_K*TILES) == 0 and"""
    pid_s = tl.program_id(0)
    pid = tl.program_id(1)
    num_pid_n: tl.constexpr = N // BLOCK_N
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    # M <= 16 always fits one 16-row tile in the decode regime; larger M (not
    # hit by the benchmark) is handled by the wrapper folding it into the grid.
    offs_m = pid_m * 16 + tl.arange(0, 16)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_m_mod = tl.where(offs_m < M, offs_m, 0)
    mask_m = offs_m < M

    k_base = pid_s * (BLOCK_K * TILES)
    acc = tl.zeros((16, BLOCK_N), dtype=tl.float32)
    a_ptrs = (
        a_ptr
        + offs_m_mod[:, None] * K
        + (k_base + tl.arange(0, BLOCK_K))[None, :]
    )
    b_ptrs = (
        b_ptr + offs_n[:, None] * K + (k_base + tl.arange(0, BLOCK_K))[None, :]
    )
    for _ in tl.static_range(TILES):
        acc = tl.dot(tl.load(a_ptrs), tl.trans(tl.load(b_ptrs)), acc)
        a_ptrs += BLOCK_K
        b_ptrs += BLOCK_K

    tl.store(
        c_ptr + pid_s * M * N + offs_m[:, None] * N + offs_n[None, :],
        acc,
        mask=mask_m[:, None],
    )


@triton.jit
def _dsv3_router_gemm_reduce_kernel(
    c_ptr,
    out_ptr,
    N_ELEMENTS: tl.constexpr,
    SPLITS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N_ELEMENTS
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for s in tl.static_range(SPLITS):
        acc += tl.load(c_ptr + s * N_ELEMENTS + offs, mask=mask, other=0.0)
    tl.store(out_ptr + offs, acc.to(out_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _dsv3_router_gemm_splitk_general_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SPLIT_K: tl.constexpr,
):
    """Masked fallback: arbitrary M, N and K. One split covers SPLIT_K
    elements of K; the final split masks the tail."""
    pid = tl.program_id(0)
    pid_s = tl.program_id(1)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    offs_m = pid_m * 16 + tl.arange(0, 16)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_m_mod = tl.where(offs_m < M, offs_m, 0)
    offs_n_mod = tl.where(offs_n < N, offs_n, 0)
    mask_m = offs_m < M
    mask_n = offs_n < N

    k_base = pid_s * SPLIT_K
    acc = tl.zeros((16, BLOCK_N), dtype=tl.float32)
    offs_k = tl.arange(0, BLOCK_K)
    for _ in tl.static_range(SPLIT_K // BLOCK_K):
        k_cur = k_base + offs_k
        k_mod = tl.where(k_cur < K, k_cur, 0)
        a = tl.load(
            a_ptr + offs_m_mod[:, None] * K + k_mod[None, :],
            mask=(k_cur < K)[None, :],
            other=0.0,
        )
        b = tl.load(
            b_ptr + offs_n_mod[:, None] * K + k_mod[None, :],
            mask=mask_n[:, None] & (k_cur < K)[None, :],
            other=0.0,
        )
        acc = tl.dot(a, tl.trans(b), acc)
        offs_k += BLOCK_K

    tl.store(
        c_ptr + pid_s * M * N + offs_m[:, None] * N + offs_n[None, :],
        acc,
        mask=mask_m[:, None] & mask_n[None, :],
    )


@triton.jit
def _dsv3_router_gemm_reduce_general_kernel(
    c_ptr,
    out_ptr,
    N_ELEMENTS,
    SPLITS,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N_ELEMENTS
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for s in range(0, SPLITS):
        acc += tl.load(c_ptr + s * N_ELEMENTS + offs, mask=mask, other=0.0)
    tl.store(out_ptr + offs, acc.to(out_ptr.dtype.element_ty), mask=mask)


def dsv3_router_gemm(hidden_states, router_weights, out_dtype):
    M, K = hidden_states.shape
    N, _ = router_weights.shape
    if M == 0 or N == 0 or K == 0:
        return torch.empty(
            (M, N), dtype=out_dtype, device=hidden_states.device
        )

    h_s1 = hidden_states.stride(1)
    h_s0 = hidden_states.stride(0)
    if h_s1 != 1 or (M > 1 and h_s0 != K):
        h = torch.empty(
            (M, K), dtype=hidden_states.dtype, device=hidden_states.device
        )
        _dsv3_router_gemm_copy2d_kernel[
            (triton.cdiv(M, 16) * triton.cdiv(K, 512),)
        ](
            hidden_states,
            h,
            M,
            K,
            h_s0,
            h_s1,
            BLOCK_M=16,
            BLOCK_K=512,
            num_warps=4,
        )
    else:
        h = hidden_states
    w_s1 = router_weights.stride(1)
    w_s0 = router_weights.stride(0)
    if w_s1 != 1 or w_s0 != K:
        w = torch.empty(
            (N, K), dtype=router_weights.dtype, device=router_weights.device
        )
        _dsv3_router_gemm_copy2d_kernel[
            (triton.cdiv(N, 64) * triton.cdiv(K, 512),)
        ](
            router_weights,
            w,
            N,
            K,
            w_s0,
            w_s1,
            BLOCK_M=64,
            BLOCK_K=512,
            num_warps=4,
        )
    else:
        w = router_weights

    out = torch.empty((M, N), dtype=out_dtype, device=hidden_states.device)
    n_elements = M * N

    block_n, block_k, tiles = 16, 256, 2
    split_k = block_k * tiles

    if M == 1:
        _dsv3_router_gemm_gemv_m1_kernel[(N,)](
            h,
            w,
            out,
            N,
            K,
            512,
            K % 512 == 0,
            num_warps=4,
            num_stages=2,
        )
        return out

    if M <= 16 and K % split_k == 0 and N % block_n == 0:
        splits = K // split_k
        part = torch.empty(
            splits * n_elements,
            dtype=torch.float32,
            device=hidden_states.device,
        )
        _dsv3_router_gemm_splitk_kernel[
            (splits, (M + 15) // 16 * (N // block_n))
        ](
            h,
            w,
            part,
            M,
            N,
            K,
            block_n,
            block_k,
            tiles,
            num_warps=2,
            num_stages=2,
        )
        _dsv3_router_gemm_reduce_kernel[((n_elements + 127) // 128,)](
            part,
            out,
            n_elements,
            splits,
            128,
            num_warps=1,
            num_stages=4,
        )
        return out

    block_n, block_k, split_k = 64, 128, 256
    splits = triton.cdiv(K, split_k)
    part = torch.empty(
        splits * n_elements, dtype=torch.float32, device=hidden_states.device
    )
    _dsv3_router_gemm_splitk_general_kernel[
        (triton.cdiv(M, 16) * triton.cdiv(N, block_n), splits)
    ](
        h,
        w,
        part,
        M,
        N,
        K,
        block_n,
        block_k,
        split_k,
        num_warps=4,
        num_stages=2,
    )
    _dsv3_router_gemm_reduce_general_kernel[((n_elements + 127) // 128,)](
        part,
        out,
        n_elements,
        splits,
        128,
        num_warps=2,
        num_stages=1,
    )
    return out


__all__ = ["dsv3_router_gemm"]
