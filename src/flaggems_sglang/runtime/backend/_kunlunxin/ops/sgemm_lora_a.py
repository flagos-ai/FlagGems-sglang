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
def _sgemm_lora_a_kernel(
    x_ptr,  # [S, K] in segment order
    w_ptr,  # [num_lora, R, K]
    out_ptr,  # [S, R] in segment order
    seg_indptr_ptr,  # [bs+1] int32
    weight_indices_ptr,  # [bs] int32
    K,
    R,
    stride_xs,
    stride_xk,
    stride_wl,
    stride_wr,
    stride_wk,
    stride_os,
    stride_or_,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    seg = tl.program_id(2)

    start = tl.load(seg_indptr_ptr + seg).to(tl.int32)
    end = tl.load(seg_indptr_ptr + seg + 1).to(tl.int32)
    seg_len = end - start

    # Adapter weight slice for this segment.
    w_idx = tl.load(weight_indices_ptr + seg).to(tl.int32)
    w_base = w_ptr + w_idx * stride_wl

    # Segment-relative row indices (affine; no gather -> keeps tl.dot lowerable).
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)

    # Early-exit programs whose entire M-tile lies past this segment's rows.
    # The grid's M axis is sized off the *total* token count S
    # (``cdiv(S, BLOCK_M)``), so for multi-segment inputs many M-tiles are
    # entirely out of range. Those programs would otherwise walk the full K
    # loop and reload the (M-independent) weight tile every iteration only
    # to drop the masked accumulator. Returning before the K loop kills that
    # redundant weight traffic. Device-side, no host sync.
    if pid_m * BLOCK_M >= seg_len:
        return

    m_mask = offs_m < seg_len
    row_idx = start + offs_m  # token row in the segment-order buffer

    # Output column tile.
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = offs_n < R

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        k_mask = offs_k < K

        # x[row_idx, offs_k] -> [BLOCK_M, BLOCK_K]
        x_ptrs = (
            x_ptr + row_idx[:, None] * stride_xs + offs_k[None, :] * stride_xk
        )
        x_mask = m_mask[:, None] & k_mask[None, :]
        a = tl.load(x_ptrs, mask=x_mask, other=0.0)

        # w[w_idx, offs_n, offs_k] -> b[k, n] = w[w_idx, n, k], load [BLOCK_K, BLOCK_N]
        w_ptrs = (
            w_base + offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wr
        )
        w_mask = k_mask[:, None] & n_mask[None, :]
        b = tl.load(w_ptrs, mask=w_mask, other=0.0)

        acc += tl.dot(a, b)

    out_tile = acc.to(out_ptr.dtype.element_ty)
    out_ptrs = (
        out_ptr + row_idx[:, None] * stride_os + offs_n[None, :] * stride_or_
    )
    out_mask = m_mask[:, None] & n_mask[None, :]
    tl.store(out_ptrs, out_tile, mask=out_mask)


def _next_pow2(n):
    p = 1
    while p < n:
        p <<= 1
    return p


def _choose_config(R, S, K):
    bn = min(_next_pow2(R), 128)
    num_warps = 4
    num_stages = 2

    if S < 256:
        # Tiny correctness-case inputs: keep the grid cheap.
        block_m = 32
        block_k = 64
        num_warps = 4
        num_stages = 3
    elif K >= 4096:

        if R <= 32:

            block_m = 512
            block_k = 512
            num_warps = 4
            num_stages = 2
        elif R <= 64:

            block_m = 480
            block_k = 256
            num_warps = 8
            num_stages = 4
        else:
            block_m = 256
            block_k = 256
            num_stages = 4
    else:
        block_m = 128
        block_k = 128
        num_stages = 4
    return block_m, bn, block_k, num_warps, num_stages


def sgemm_lora_a(x, weights, batch_info, stack_num=1):
    S, K = x.shape
    num_lora, R, K_w = weights.shape
    assert K == K_w, f"K mismatch: x K={K}, weights K={K_w}"

    seg_indptr = batch_info.seg_indptr
    weight_indices = batch_info.weight_indices
    permutation = batch_info.permutation
    bs = batch_info.bs

    if permutation is not None:
        perm_long = permutation.long()
        x_work = x.index_select(0, perm_long)
    else:
        perm_long = None
        x_work = x
    out_work = torch.zeros((S, R), dtype=x.dtype, device=x.device)

    block_m, block_n, block_k, num_warps, num_stages = _choose_config(R, S, K)

    grid = (triton.cdiv(S, block_m), triton.cdiv(R, block_n), bs)

    _sgemm_lora_a_kernel[grid](
        x_work,
        weights,
        out_work,
        seg_indptr,
        weight_indices,
        K,
        R,
        x_work.stride(0),
        x_work.stride(1),
        weights.stride(0),
        weights.stride(1),
        weights.stride(2),
        out_work.stride(0),
        out_work.stride(1),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    if permutation is not None:
        out = torch.zeros((S, R), dtype=x.dtype, device=x.device)
        # out[perm] = out_work  (perm is a full permutation of 0..S-1, unique)
        out.scatter_(0, perm_long.unsqueeze(1).expand(-1, R), out_work)
        return out

    return out_work


__all__ = ["sgemm_lora_a"]
