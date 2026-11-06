# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""tiny_n_gemm for RDNA3 (gfx1100): ``x @ w.T`` for an x of a few rows
(Triton).

The generic op takes its split-K kernel only while the output tiles are
fewer than ``multi_processor_count``, which on gfx1100 counts WGPs (48), not
CUs (96): n = 1536 makes 96 tiles and falls to the single-pass kernel, whose
512-wide K tile spills past the scratch limit and runs 5.6x slower than
split-K. Here every x of more than one row takes split-K, with one pipeline
stage, since more stages spill and run 1.6x slower. A program takes three
256-element K steps: two or four steps, which start the K chunks a power of
two of bytes apart, run 1.4-2.2x slower. A single row keeps the generic
GEMV kernel, and K need not be a multiple of the K step.

This kernel is written in portable Triton only. It must NOT call any
pre-compiled / vendor-specific cached operator — the whole computation is done
inside the Triton kernel.
"""

import torch
import triton
import triton.language as tl

_BLOCK_M = 16
_BLOCK_N = 16
_BLOCK_K = 256
_K_STEPS = 3
_WARPS = 2

# GEMV path (m == 1): N threshold between the "thin n" and "wide n" configs.
_GEMV_N_WIDE = 1024


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
    EVEN_K: tl.constexpr,
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
    for k in range(k_start, k_end, BLOCK_K):
        if EVEN_K:
            a = tl.load(x_ptrs, mask=m_mask, other=0.0)
            b = tl.load(w_ptrs, mask=n_col, other=0.0)
        else:
            k_mask = (offs_k + k < K)[None, :]
            a = tl.load(x_ptrs, mask=m_mask & k_mask, other=0.0)
            b = tl.load(w_ptrs, mask=n_col & k_mask, other=0.0)
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


def tiny_n_gemm(x, w, out_dtype):
    m, k = x.shape
    n, k2 = w.shape
    assert k == k2, "tiny_n_gemm: x and w must share the K dimension"

    x = x.contiguous()
    w = w.contiguous()

    out = torch.empty((m, n), device=x.device, dtype=out_dtype)
    if m == 0 or n == 0:
        return out

    if m == 1:
        if n < _GEMV_N_WIDE:
            block_n, block_k, warps, stages = 4, 2048, 4, 2
        else:
            block_n, block_k, warps, stages = 4, 1024, 2, 2
        block_k = min(block_k, triton.next_power_of_2(max(k, 16)))
        _tiny_n_gemv_kernel[(triton.cdiv(n, block_n),)](
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

    block_k = min(_BLOCK_K, triton.next_power_of_2(max(k, 16)))
    split_k = triton.cdiv(triton.cdiv(k, block_k), _K_STEPS)
    tiles = triton.cdiv(m, _BLOCK_M) * triton.cdiv(n, _BLOCK_N)
    part = torch.empty((split_k, m, n), device=x.device, dtype=torch.float32)
    _tiny_n_dot_splitk_kernel[(tiles * split_k,)](
        x,
        w,
        part,
        m,
        n,
        k,
        split_k,
        _K_STEPS,
        x.stride(0),
        w.stride(0),
        BLOCK_M=_BLOCK_M,
        BLOCK_N=_BLOCK_N,
        BLOCK_K=block_k,
        EVEN_K=(k % block_k == 0),
        IEEE=(out_dtype == torch.float32),
        num_warps=_WARPS,
        num_stages=1,
    )
    total = m * n
    _tiny_n_reduce_kernel[(triton.cdiv(total, 256),)](
        part,
        out,
        total,
        split_k,
        BLOCK=256,
        num_warps=1,
        num_stages=1,
    )
    return out


__all__ = ["tiny_n_gemm"]
