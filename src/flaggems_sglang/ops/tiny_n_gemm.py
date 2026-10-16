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

"""tiny_n_gemm: thin-M bf16 GEMM for decode-phase projections."""

import torch
import triton
import triton.language as tl

from flaggems_sglang import runtime as _runtime


@triton.jit
def _tiny_n_dot_kernel(
    x_ptr,
    w_ptr,
    out_ptr,
    M,
    N,
    K,
    stride_xm,
    stride_wn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    EVEN_K: tl.constexpr,
    EVEN_N: tl.constexpr,
    IEEE: tl.constexpr,
):
    """One program computes a [BLOCK_M, BLOCK_N] output tile over."""
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)

    pid_m = pid % num_pid_m
    pid_n = pid // num_pid_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    m_mask = offs_m[:, None] < M
    n_col = offs_n[:, None] < N
    n_row = offs_n[None, :] < N

    x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :]
    # w is [N, K] and K is the contiguous axis, so load the [BLOCK_N, BLOCK_K]
    # tile in its natural layout and transpose it in registers.
    w_ptrs = w_ptr + offs_n[:, None] * stride_wn + offs_k[None, :]

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    if EVEN_N:
        if EVEN_K:
            for _ in range(0, K, BLOCK_K):
                a = tl.load(x_ptrs, mask=m_mask, other=0.0)
                b = tl.load(w_ptrs)
                if IEEE:
                    acc = tl.dot(a, tl.trans(b), acc, input_precision="ieee")
                else:
                    acc = tl.dot(a, tl.trans(b), acc)
                x_ptrs += BLOCK_K
                w_ptrs += BLOCK_K
        else:
            for k0 in range(0, K, BLOCK_K):
                k_mask = (offs_k + k0) < K
                a = tl.load(x_ptrs, mask=m_mask & k_mask[None, :], other=0.0)
                b = tl.load(w_ptrs, mask=k_mask[None, :], other=0.0)
                if IEEE:
                    acc = tl.dot(a, tl.trans(b), acc, input_precision="ieee")
                else:
                    acc = tl.dot(a, tl.trans(b), acc)
                x_ptrs += BLOCK_K
                w_ptrs += BLOCK_K
    else:
        if EVEN_K:
            for _ in range(0, K, BLOCK_K):
                a = tl.load(x_ptrs, mask=m_mask, other=0.0)
                b = tl.load(w_ptrs, mask=n_col, other=0.0)
                if IEEE:
                    acc = tl.dot(a, tl.trans(b), acc, input_precision="ieee")
                else:
                    acc = tl.dot(a, tl.trans(b), acc)
                x_ptrs += BLOCK_K
                w_ptrs += BLOCK_K
        else:
            for k0 in range(0, K, BLOCK_K):
                k_mask = (offs_k + k0) < K
                a = tl.load(x_ptrs, mask=m_mask & k_mask[None, :], other=0.0)
                b = tl.load(w_ptrs, mask=n_col & k_mask[None, :], other=0.0)
                if IEEE:
                    acc = tl.dot(a, tl.trans(b), acc, input_precision="ieee")
                else:
                    acc = tl.dot(a, tl.trans(b), acc)
                x_ptrs += BLOCK_K
                w_ptrs += BLOCK_K

    out = acc.to(out_ptr.dtype.element_ty)
    out_ptrs = out_ptr + offs_m[:, None] * N + offs_n[None, :]
    tl.store(out_ptrs, out, mask=m_mask & n_row)


@triton.jit
def _tiny_n_dot_splitk_kernel(
    x_ptr,
    w_ptr,
    part_ptr,  # [split_k, M, N] fp32 partial sums
    M,
    N,
    K,
    split_k,
    k_steps,  # uniform number of BLOCK_K steps per K-chunk
    stride_xm,
    stride_wn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    IEEE: tl.constexpr,
):
    """One program computes a [BLOCK_M, BLOCK_N] tile over one K-chunk."""
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    tile = num_pid_m * num_pid_n

    pid_m = pid % num_pid_m
    pid_n = (pid // num_pid_m) % num_pid_n
    pid_k = pid // tile
    pid_k = (pid_k + pid_m * num_pid_n + pid_n) % split_k

    k_start = pid_k * k_steps * BLOCK_K
    k_end = tl.minimum(k_start + k_steps * BLOCK_K, K)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    m_mask = offs_m[:, None] < M
    n_col = offs_n[:, None] < N
    n_row = offs_n[None, :] < N

    x_ptrs = x_ptr + offs_m[:, None] * stride_xm + k_start + offs_k[None, :]
    w_ptrs = w_ptr + offs_n[:, None] * stride_wn + k_start + offs_k[None, :]

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for _ in range(k_start, k_end, BLOCK_K):
        a = tl.load(x_ptrs, mask=m_mask, other=0.0)
        b = tl.load(w_ptrs, mask=n_col, other=0.0)
        if IEEE:
            acc = tl.dot(a, tl.trans(b), acc, input_precision="ieee")
        else:
            acc = tl.dot(a, tl.trans(b), acc)
        x_ptrs += BLOCK_K
        w_ptrs += BLOCK_K

    part_ptrs = (
        part_ptr + pid_k * M * N + offs_m[:, None] * N + offs_n[None, :]
    )
    tl.store(part_ptrs, acc, mask=m_mask & n_row)


@triton.jit
def _tiny_n_reduce_kernel(
    part_ptr,  # [split_k, M, N] fp32
    out_ptr,
    total,
    split_k,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < total
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for s in range(0, split_k):
        acc += tl.load(part_ptr + s * total + offs, mask=mask, other=0.0)
    tl.store(out_ptr + offs, acc.to(out_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _tiny_n_gemv_kernel(
    x_ptr,
    w_ptr,
    out_ptr,
    N,
    K,
    stride_wn,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    EVEN_K: tl.constexpr,
):
    """m == 1 fast path: broadcast-MAC over a [BLOCK_N, BLOCK_K] fp32."""
    pid = tl.program_id(0)
    offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    n_mask = offs_n[:, None] < N
    w_ptrs = w_ptr + offs_n[:, None] * stride_wn + offs_k[None, :]

    acc = tl.zeros((BLOCK_N, BLOCK_K), dtype=tl.float32)
    if EVEN_K:
        for k0 in range(0, K, BLOCK_K):
            a = tl.load(x_ptr + k0 + offs_k).to(tl.float32)
            b = tl.load(w_ptrs, mask=n_mask, other=0.0).to(tl.float32)
            acc += b * a[None, :]
            w_ptrs += BLOCK_K
    else:
        for k0 in range(0, K, BLOCK_K):
            k_mask = (offs_k + k0) < K
            a = tl.load(x_ptr + k0 + offs_k, mask=k_mask, other=0.0).to(
                tl.float32
            )
            b = tl.load(w_ptrs, mask=n_mask & k_mask[None, :], other=0.0).to(
                tl.float32
            )
            acc += b * a[None, :]
            w_ptrs += BLOCK_K

    out = tl.sum(acc, axis=1).to(out_ptr.dtype.element_ty)
    tl.store(out_ptr + offs_n, out, mask=offs_n < N)


_BLOCK_M = 16
_DOT_BLOCK_N = 16
_DOT_BLOCK_K = 512
_DOT_WARPS = 2
_DOT_STAGES = 4

_SK_WAVES = 2.5

# GEMV path (m == 1): N threshold between the "thin n" and "wide n" configs.
_GEMV_N_WIDE = 1024


def _num_sms():
    """SM count through the framework's portable torch device object."""
    fn = _runtime.torch_device_fn
    return fn.get_device_properties(fn.current_device()).multi_processor_count


def tiny_n_gemm(x, w, out_dtype):
    m, k = x.shape
    n, k2 = w.shape
    assert k == k2, "tiny_n_gemm: x and w must share the K dimension"

    x = x.contiguous()
    w = w.contiguous()

    out = torch.empty((m, n), device=x.device, dtype=out_dtype)

    if m == 1:
        if n < _GEMV_N_WIDE:
            block_n, block_k, warps, stages = 4, 2048, 4, 2
        else:
            block_n, block_k, warps, stages = 4, 1024, 2, 2
        block_k = min(block_k, triton.next_power_of_2(max(k, 16)))
        grid = (triton.cdiv(n, block_n),)
        _tiny_n_gemv_kernel[grid](
            x,
            w,
            out,
            n,
            k,
            w.stride(0),
            BLOCK_N=block_n,
            BLOCK_K=block_k,
            EVEN_K=(k % block_k == 0),
            num_warps=warps,
            num_stages=stages,
        )
        return out

    block_k = min(_DOT_BLOCK_K, triton.next_power_of_2(max(k, 16)))
    num_pid_n = triton.cdiv(n, _DOT_BLOCK_N)
    tiles = triton.cdiv(m, _BLOCK_M) * num_pid_n

    split_k = 1
    if tiles < _num_sms():
        block_k = min(256, block_k)
        max_steps = max(triton.cdiv(k, block_k), 1)
        split_k = max(
            1, min(triton.cdiv(int(_num_sms() * _SK_WAVES), tiles), max_steps)
        )
        if tiles * split_k < tiles + 4:
            split_k = (
                1  # a couple of extra programs cannot pay for the workspace
            )

    if split_k > 1:
        k_steps = triton.cdiv(triton.cdiv(k, split_k), block_k)
        part = torch.empty(
            (split_k, m, n), device=x.device, dtype=torch.float32
        )
        grid = (tiles * split_k,)
        _tiny_n_dot_splitk_kernel[grid](
            x,
            w,
            part,
            m,
            n,
            k,
            split_k,
            k_steps,
            x.stride(0),
            w.stride(0),
            BLOCK_M=_BLOCK_M,
            BLOCK_N=_DOT_BLOCK_N,
            BLOCK_K=block_k,
            IEEE=(out_dtype == torch.float32),
            num_warps=_DOT_WARPS,
            num_stages=_DOT_STAGES,
        )
        total = m * n
        rgrid = (triton.cdiv(total, 256),)
        _tiny_n_reduce_kernel[rgrid](
            part,
            out,
            total,
            split_k,
            BLOCK=256,
            num_warps=1,
            num_stages=1,
        )
        return out

    grid = (tiles,)
    _tiny_n_dot_kernel[grid](
        x,
        w,
        out,
        m,
        n,
        k,
        x.stride(0),
        w.stride(0),
        BLOCK_M=_BLOCK_M,
        BLOCK_N=_DOT_BLOCK_N,
        BLOCK_K=block_k,
        EVEN_K=(k % block_k == 0),
        EVEN_N=(n % _DOT_BLOCK_N == 0),
        IEEE=(out_dtype == torch.float32),
        num_warps=_DOT_WARPS,
        num_stages=_DOT_STAGES,
    )
    return out


__all__ = ["tiny_n_gemm"]
