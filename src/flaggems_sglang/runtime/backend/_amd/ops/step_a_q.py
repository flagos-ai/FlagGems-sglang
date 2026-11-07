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

"""step_a_q implementation for AMD."""

import torch
import triton
import triton.language as tl


@triton.jit
def _step_a_q_kernel(
    x_ptr,
    weight_ptr,
    out_ptr,
    seg_indptr,
    weight_indices_ptr,
    ranks_ptr,
    perm_ptr,
    H: tl.constexpr,
    N: tl.constexpr,
    S: tl.constexpr,
    K: tl.constexpr,
    FULL_K_PER_HEAD: tl.constexpr,
    BS: tl.constexpr,
    X_STRIDE_0: tl.constexpr,
    X_STRIDE_1: tl.constexpr,
    X_STRIDE_2: tl.constexpr,
    W_STRIDE_0: tl.constexpr,
    W_STRIDE_1: tl.constexpr,
    W_STRIDE_2: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    NK: tl.constexpr,
    BRANCHED: tl.constexpr,
    HAS_PERM: tl.constexpr,
):
    h = tl.program_id(0)
    i0 = tl.program_id(1) * BLOCK_M
    idx = i0 + tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, BLOCK_K)
    offs_n = tl.arange(0, BLOCK_N)
    if HAS_PERM:
        keep = idx < tl.load(seg_indptr + BS)
        rows = tl.load(perm_ptr + idx, mask=keep, other=0).to(tl.int32)
    else:
        keep = idx < S
        rows = idx
    xb = (
        x_ptr
        + rows[:, None] * X_STRIDE_0
        + h * X_STRIDE_1
        + offs_k[None, :] * X_STRIDE_2
    )
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for b in range(BS):
        lo = tl.load(seg_indptr + b)
        hi = tl.load(seg_indptr + b + 1)
        if BRANCHED:
            if (lo < i0 + BLOCK_M) & (hi > i0):
                w = tl.load(weight_indices_ptr + b)
                if tl.load(ranks_ptr + w) > 0:
                    rmask = keep & (idx >= lo) & (idx < hi)
                    acc += tl.where(
                        rmask[:, None],
                        _step_a_q_tile(
                            x_ptr,
                            weight_ptr,
                            xb,
                            w,
                            h,
                            offs_k,
                            offs_n,
                            keep,
                            H,
                            N,
                            S,
                            K,
                            FULL_K_PER_HEAD,
                            X_STRIDE_2,
                            W_STRIDE_0,
                            W_STRIDE_1,
                            W_STRIDE_2,
                            BLOCK_M,
                            BLOCK_N,
                            BLOCK_K,
                            NK,
                            HAS_PERM,
                        ),
                        0.0,
                    )
        else:
            w = tl.load(weight_indices_ptr + b)
            rmask = (
                keep & (idx >= lo) & (idx < hi) & (tl.load(ranks_ptr + w) > 0)
            )
            acc += tl.where(
                rmask[:, None],
                _step_a_q_tile(
                    x_ptr,
                    weight_ptr,
                    xb,
                    w,
                    h,
                    offs_k,
                    offs_n,
                    keep,
                    H,
                    N,
                    S,
                    K,
                    FULL_K_PER_HEAD,
                    X_STRIDE_2,
                    W_STRIDE_0,
                    W_STRIDE_1,
                    W_STRIDE_2,
                    BLOCK_M,
                    BLOCK_N,
                    BLOCK_K,
                    NK,
                    HAS_PERM,
                ),
                0.0,
            )
    op = out_ptr + rows[:, None] * (H * N) + h * N + offs_n[None, :]
    val = acc.to(out_ptr.dtype.element_ty)
    if BLOCK_N == N:
        if not HAS_PERM and S % BLOCK_M == 0:
            tl.store(op, val)
        else:
            tl.store(op, val, mask=keep[:, None])
    else:
        tl.store(op, val, mask=keep[:, None] & (offs_n < N)[None, :])


@triton.jit
def _step_a_q_tile(
    x_ptr,
    weight_ptr,
    xb,
    w,
    h,
    offs_k,
    offs_n,
    keep,
    H: tl.constexpr,
    N: tl.constexpr,
    S: tl.constexpr,
    K: tl.constexpr,
    FULL_K_PER_HEAD: tl.constexpr,
    X_STRIDE_2: tl.constexpr,
    W_STRIDE_0: tl.constexpr,
    W_STRIDE_1: tl.constexpr,
    W_STRIDE_2: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    NK: tl.constexpr,
    HAS_PERM: tl.constexpr,
):
    wbase = (
        weight_ptr
        + w * W_STRIDE_0
        + (h * FULL_K_PER_HEAD + offs_k[:, None]) * W_STRIDE_1
        + offs_n[None, :] * W_STRIDE_2
    )
    part = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for ki in range(NK):
        kk = ki * BLOCK_K
        xp = xb + kk * X_STRIDE_2
        wp = wbase + kk * W_STRIDE_1
        if NK * BLOCK_K == K:
            if HAS_PERM or S % BLOCK_M == 0:
                x = tl.load(xp, cache_modifier=".cg")
            else:
                x = tl.load(
                    xp, mask=keep[:, None], other=0.0, cache_modifier=".cg"
                )
            if BLOCK_N == N:
                wt = tl.load(wp)
            else:
                wt = tl.load(wp, mask=(offs_n < N)[None, :], other=0.0)
        else:
            km = kk + offs_k < K
            x = tl.load(
                xp,
                mask=keep[:, None] & km[None, :],
                other=0.0,
                cache_modifier=".cg",
            )
            wt = tl.load(
                wp, mask=km[:, None] & (offs_n < N)[None, :], other=0.0
            )
        part = tl.dot(x, wt, part)
    return part


def step_a_q(q_nope, B_buf, batch_info, full_K_per_head):
    (S, H, K) = q_nope.shape
    N = B_buf.shape[2]
    permutation = batch_info.permutation
    has_permutation = permutation is not None
    if has_permutation:
        covered = permutation.numel() == S
    else:
        permutation = q_nope
        covered = True
    if covered:
        out = torch.empty((S, H, N), dtype=q_nope.dtype, device=q_nope.device)
    else:
        out = torch.zeros((S, H, N), dtype=q_nope.dtype, device=q_nope.device)
    block_m = 16 if S < 64 else 64
    block_n = max(16, triton.next_power_of_2(N))
    block_k = max(16, triton.next_power_of_2(K))
    nk = 1
    _step_a_q_kernel[H, triton.cdiv(S, block_m)](
        q_nope,
        B_buf,
        out,
        batch_info.seg_indptr,
        batch_info.weight_indices,
        batch_info.lora_ranks,
        permutation,
        H,
        N,
        S,
        K,
        full_K_per_head,
        batch_info.bs,
        *q_nope.stride(),
        *B_buf.stride(),
        block_m,
        block_n,
        block_k,
        nk,
        S >= 64,
        has_permutation,
        num_warps=4,
        num_stages=3,
    )
    return out


__all__ = ["step_a_q"]
