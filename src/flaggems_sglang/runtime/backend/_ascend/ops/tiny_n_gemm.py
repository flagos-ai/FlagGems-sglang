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

"""Tiny-N GEMM, ``out = x @ w.T``, entirely in Triton."""

import torch
import triton
import triton.language as tl


@triton.jit
def _tiny_n_gemm_kernel(
    x_ptr,
    w_ptr,
    out_ptr,
    M,
    K,
    N,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    EVEN_M: tl.constexpr,
):
    """Runtime-shape fallback: masks on the x load / store (unless EVEN_M) and
    on the K loop tail.  Used only when the shape does not tile evenly."""
    pid = tl.program_id(0)
    offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_m = tl.arange(0, 16)
    mask_n = offs_n < N
    mask_m = offs_m < M
    acc = tl.zeros((16, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K
        if EVEN_M:
            x = tl.load(
                x_ptr + offs_m[:, None] * K + offs_k[None, :],
                mask=mask_k[None, :],
                other=0.0,
            )
        else:
            x = tl.load(
                x_ptr + offs_m[:, None] * K + offs_k[None, :],
                mask=mask_m[:, None] & mask_k[None, :],
                other=0.0,
            )
        w = tl.load(
            w_ptr + offs_n[:, None] * K + offs_k[None, :],
            mask=mask_n[:, None] & mask_k[None, :],
            other=0.0,
        )
        acc = tl.dot(x, tl.trans(w), acc, out_dtype=tl.float32)
    out_ptrs = out_ptr + offs_m[:, None] * N + offs_n[None, :]
    if EVEN_M:
        tl.store(
            out_ptrs, acc.to(out_ptr.dtype.element_ty), mask=mask_n[None, :]
        )
    else:
        tl.store(
            out_ptrs,
            acc.to(out_ptr.dtype.element_ty),
            mask=mask_m[:, None] & mask_n[None, :],
        )


@triton.jit
def _tiny_n_gemm_kernel_const(
    x_ptr,
    w_ptr,
    out_ptr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    K: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr,
    EVEN_M: tl.constexpr,
):
    """M, N and K folded to constants with ``BLOCK_N | N`` and ``BLOCK_K | K``:
    no masks anywhere except the row mask that folds away at ``m == 16``, the
    K loop bounds are immediates, and the integer address math folds."""
    pid = tl.program_id(0)
    offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_m = tl.arange(0, 16)
    acc = tl.zeros((16, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        if EVEN_M:
            x = tl.load(x_ptr + offs_m[:, None] * K + offs_k[None, :])
        else:
            x = tl.load(
                x_ptr + offs_m[:, None] * K + offs_k[None, :],
                mask=(offs_m < M)[:, None],
                other=0.0,
            )
        w = tl.load(w_ptr + offs_n[:, None] * K + offs_k[None, :])
        acc = tl.dot(x, tl.trans(w), acc, out_dtype=tl.float32)
    out_ptrs = out_ptr + offs_m[:, None] * N + offs_n[None, :]
    if EVEN_M:
        tl.store(out_ptrs, acc.to(out_ptr.dtype.element_ty))
    else:
        tl.store(
            out_ptrs,
            acc.to(out_ptr.dtype.element_ty),
            mask=(offs_m < M)[:, None],
        )


@triton.jit
def _tiny_n_gemm_kernel_splitn(
    x_ptr,
    w_ptr,
    out_ptr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SPLIT_N: tl.constexpr,
    K: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr,
):
    """N-range split: ``SPLIT_N`` programs share each column group, so."""
    pid = tl.program_id(0)
    j = pid % SPLIT_N
    offs_n = j * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_m = tl.arange(0, 16)
    acc = tl.zeros((16, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        if M == 16:
            x = tl.load(x_ptr + offs_m[:, None] * K + offs_k[None, :])
        else:
            x = tl.load(
                x_ptr + offs_m[:, None] * K + offs_k[None, :],
                mask=(offs_m < M)[:, None],
                other=0.0,
            )
        w = tl.load(w_ptr + offs_n[:, None] * K + offs_k[None, :])
        acc = tl.dot(x, tl.trans(w), acc, out_dtype=tl.float32)
    out_ptrs = out_ptr + offs_m[:, None] * N + offs_n[None, :]
    if M == 16:
        tl.store(out_ptrs, acc.to(out_ptr.dtype.element_ty))
    else:
        tl.store(
            out_ptrs,
            acc.to(out_ptr.dtype.element_ty),
            mask=(offs_m < M)[:, None],
        )


# Default config for benchmark-like shapes that miss the table below.
_BLOCK_M = 16  # covers every m <= 16 in one row-tile
_BLOCK_N = 64
_BLOCK_K = 256
_NUM_WARPS = 2
_NUM_STAGES = 3
_NUM_WARPS_GEN = 4
_NUM_STAGES_GEN = 2

_TABLE = {
    (1, 512, 7168): (64, 256, 1, 3),
    (1, 1536, 7168): (128, 512, 1, 2),
    (2, 512, 7168): (64, 256, 2, 3),
    (2, 1536, 7168): (32, 512, 2, 2),
    (4, 512, 7168): (64, 128, 4, 3),
    (4, 1536, 7168): (32, 512, 2, 4),
    (8, 512, 7168): (128, 512, 1, 3),
    (8, 1536, 7168): (32, 512, 2, 4),
    (16, 1536, 7168): (128, 512, 4, 3),
}

_SPLITN_MS = (16,)
_SPLITN_BN = 128
_SPLITN_BK = 512
_SPLITN_WARPS = 2
_SPLITN_STAGES = 3
_SPLITN_MAX = 7


def _pick(m, n, k):
    """Choose the instantiation and config for one shape.

    Returns ``(kind, BLOCK_N, BLOCK_K, num_warps, num_stages, split_n)`` where
    ``kind`` is ``"const"``, ``"splitn"`` or ``"generic"``.
    """
    if m in _SPLITN_MS and n % _SPLITN_BN == 0:
        split_n = n // _SPLITN_BN
        if 2 <= split_n <= _SPLITN_MAX:
            return (
                "splitn",
                _SPLITN_BN,
                _SPLITN_BK,
                _SPLITN_WARPS,
                _SPLITN_STAGES,
                split_n,
            )
    cfg = _TABLE.get((m, n, k))
    if cfg is None:
        cfg = (_BLOCK_N, _BLOCK_K, _NUM_WARPS, _NUM_STAGES)
    bn, bk, warps, stages = cfg
    if n % bn == 0 and k % bk == 0:
        return ("const", bn, bk, warps, stages, 1)
    return ("generic", _BLOCK_N, _BLOCK_K, _NUM_WARPS_GEN, _NUM_STAGES_GEN, 1)


def tiny_n_gemm(x, w, out_dtype):
    out = torch.empty(
        (x.shape[0], w.shape[0]), dtype=out_dtype, device=x.device
    )
    m, k = x.shape
    n = w.shape[0]

    if not x.is_contiguous():
        x = x.contiguous()
    if not w.is_contiguous():
        w = w.contiguous()

    kind, bn, bk, warps, stages, split_n = _pick(m, n, k)
    if kind == "splitn":
        _tiny_n_gemm_kernel_splitn[((n // bn) * split_n,)](
            x,
            w,
            out,
            BLOCK_N=bn,
            BLOCK_K=bk,
            SPLIT_N=split_n,
            K=k,
            M=m,
            N=n,
            num_warps=warps,
            num_stages=stages,
        )
        return out
    grid = (triton.cdiv(n, bn), 1, 1)
    if kind == "const":
        _tiny_n_gemm_kernel_const[grid](
            x,
            w,
            out,
            BLOCK_N=bn,
            BLOCK_K=bk,
            K=k,
            M=m,
            N=n,
            EVEN_M=(m % _BLOCK_M == 0),
            num_warps=warps,
            num_stages=stages,
        )
    else:
        _tiny_n_gemm_kernel[grid](
            x,
            w,
            out,
            m,
            k,
            n,
            BLOCK_N=bn,
            BLOCK_K=bk,
            EVEN_M=(m % _BLOCK_M == 0),
            num_warps=warps,
            num_stages=stages,
        )
    return out


__all__ = ["tiny_n_gemm"]
