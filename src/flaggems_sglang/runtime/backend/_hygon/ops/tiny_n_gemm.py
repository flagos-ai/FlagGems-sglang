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
def _tiny_n_gemm_splitk_kernel(
    x_ptr,
    w_ptr,
    part_ptr,
    m,
    n,
    k,
    m_pad,
    stride_xm,
    stride_xk,
    stride_wn,
    stride_wk,
    INPUT_F32: tl.constexpr,
    STATIC_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    CHUNKS_PER: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_s = tl.program_id(1)

    k0 = pid_s * (CHUNKS_PER * BLOCK_K)

    offs_m = tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    m_mask = offs_m < m
    n_mask = offs_n < n

    x_ptrs = (
        x_ptr
        + offs_m[:, None] * stride_xm
        + (k0 + offs_k[None, :]) * stride_xk
    )
    w_ptrs = (
        w_ptr
        + offs_n[None, :] * stride_wn
        + (k0 + offs_k[:, None]) * stride_wk
    )

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    if STATIC_K:
        for i in range(CHUNKS_PER):
            x_tile = tl.load(x_ptrs, mask=m_mask[:, None], other=0.0)
            w_tile = tl.load(w_ptrs, mask=n_mask[None, :], other=0.0)
            if INPUT_F32:
                acc = tl.dot(x_tile, w_tile, acc, input_precision="ieee")
            else:
                acc = tl.dot(x_tile, w_tile, acc)
            x_ptrs += BLOCK_K * stride_xk
            w_ptrs += BLOCK_K * stride_wk
    else:
        k1 = tl.minimum(k0 + CHUNKS_PER * BLOCK_K, k)
        for kk in range(k0, k1, BLOCK_K):
            k_mask = (kk + offs_k) < k
            x_tile = tl.load(
                x_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=0.0
            )
            w_tile = tl.load(
                w_ptrs, mask=k_mask[:, None] & n_mask[None, :], other=0.0
            )
            if INPUT_F32:
                acc = tl.dot(x_tile, w_tile, acc, input_precision="ieee")
            else:
                acc = tl.dot(x_tile, w_tile, acc)
            x_ptrs += BLOCK_K * stride_xk
            w_ptrs += BLOCK_K * stride_wk

    p_ptrs = (
        part_ptr + pid_s * m_pad * n + offs_m[:, None] * n + offs_n[None, :]
    )
    tl.store(p_ptrs, acc, mask=m_mask[:, None] & n_mask[None, :])


@triton.jit
def _tiny_n_gemm_reduce_kernel(
    part_ptr,
    out_ptr,
    numel,
    m_pad,
    n,
    BLOCK: tl.constexpr,
    S: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for s in range(S):
        acc += tl.load(part_ptr + s * m_pad * n + offs, mask=mask, other=0.0)
    tl.store(out_ptr + offs, acc.to(out_ptr.dtype.element_ty), mask=mask)


_BLOCK_M = 16
_CFG_NARROW = (16, 256, 2, 2, 8)
_CFG_WIDE = (32, 128, 2, 2, 4)
_NARROW_MAX_N = 1024
_MIN_PROGRAMS = 2 * 80
_REDUCE_BLOCK = 1024


def _split_count(k, block_k, max_split):
    n_chunks = triton.cdiv(k, block_k)
    split = min(max_split, n_chunks)
    chunks_per = triton.cdiv(n_chunks, split)
    split = triton.cdiv(n_chunks, chunks_per)
    return max(1, split), chunks_per


def _select_config(m, n, k):
    if n <= _NARROW_MAX_N:
        block_n, block_k, num_warps, num_stages, max_split = _CFG_NARROW
    else:
        block_n, block_k, num_warps, num_stages, max_split = _CFG_WIDE

    n_panels = triton.cdiv(n, block_n)
    if n_panels < _MIN_PROGRAMS:
        max_split = max(max_split, triton.cdiv(_MIN_PROGRAMS, n_panels))
    split, chunks_per = _split_count(k, block_k, max_split)
    return block_n, block_k, num_warps, num_stages, split, chunks_per


def tiny_n_gemm(x, w, out_dtype):
    m, k = x.shape
    n = w.shape[0]
    out = x.new_empty((m, n), dtype=out_dtype)

    m_pad = triton.cdiv(m, _BLOCK_M) * _BLOCK_M
    block_n, block_k, num_warps, num_stages, split, chunks_per = (
        _select_config(m, n, k)
    )
    n_panels = triton.cdiv(n, block_n)

    static_k = (split * chunks_per * block_k == k) and (k % block_k == 0)

    part = torch.empty((split, m_pad, n), device=x.device, dtype=torch.float32)

    grid = (n_panels, split, triton.cdiv(m, _BLOCK_M))
    _tiny_n_gemm_splitk_kernel[grid](
        x,
        w,
        part,
        m,
        n,
        k,
        m_pad,
        x.stride(0),
        x.stride(1),
        w.stride(0),
        w.stride(1),
        INPUT_F32=(x.dtype == torch.float32),
        STATIC_K=static_k,
        BLOCK_M=_BLOCK_M,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        CHUNKS_PER=chunks_per,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    numel = m * n
    _tiny_n_gemm_reduce_kernel[(triton.cdiv(numel, _REDUCE_BLOCK),)](
        part,
        out,
        numel,
        m_pad,
        n,
        BLOCK=_REDUCE_BLOCK,
        S=split,
        num_warps=4,
    )
    return out


__all__ = ["tiny_n_gemm"]
