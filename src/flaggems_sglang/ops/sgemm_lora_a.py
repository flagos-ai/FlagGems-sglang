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


def _next_pow2(v):
    p = 1
    while p < v:
        p <<= 1
    return p


@triton.jit
def _sgemm_lora_a_stage1(
    x_ptr,
    w_ptr,
    tmp_ptr,
    seg_indptr_ptr,
    weight_indices_ptr,
    perm_ptr,
    K: tl.constexpr,
    R: tl.constexpr,
    S: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    NUM_K_SPLITS: tl.constexpr,
    HAS_PERM: tl.constexpr,
    INPUT_PRECISION: tl.constexpr,
):
    mk_id = tl.program_id(0)
    n_tile_id = tl.program_id(1)
    seg_id = tl.program_id(2)

    m_tile_id = mk_id // NUM_K_SPLITS
    k_split_id = mk_id % NUM_K_SPLITS

    seg_start = tl.load(seg_indptr_ptr + seg_id).to(tl.int64)
    seg_end = tl.load(seg_indptr_ptr + seg_id + 1).to(tl.int64)
    seg_len = seg_end - seg_start

    m_local_start = m_tile_id * BLOCK_M
    # idle tile for this (shorter/empty) segment: nothing to do
    if m_local_start >= seg_len:
        return

    w_idx = tl.load(weight_indices_ptr + seg_id).to(tl.int64)

    # within-segment row index -> global token index -> (optionally) permuted row
    m_idx = m_local_start + tl.arange(0, BLOCK_M)
    valid_m = m_idx < seg_len
    token_idx = seg_start + m_idx
    if HAS_PERM:
        row_idx = tl.load(perm_ptr + token_idx, mask=valid_m, other=0).to(
            tl.int64
        )
    else:
        row_idx = token_idx
    row_idx = tl.where(valid_m, row_idx, 0)

    n_start = n_tile_id * BLOCK_N
    n_off = n_start + tl.arange(0, BLOCK_N)
    n_in = n_off < R

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    w_base = w_idx * (R * K)
    # this program's contiguous slice of K
    for k_start in range(k_split_id * BLOCK_K, K, NUM_K_SPLITS * BLOCK_K):
        k_off = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_off < K

        # x_tile: [BLOCK_M, BLOCK_K] gathered at row_idx
        x_off = row_idx[:, None] * K + k_off[None, :]
        x_tile = tl.load(
            x_ptr + x_off, mask=valid_m[:, None] & k_mask[None, :], other=0.0
        )

        # w_tile: [BLOCK_K, BLOCK_N] = weights[w_idx, n, k].T
        # weights [num_lora, R, K] row-major -> [w,n,k] at w_base + n*K + k
        w_off = w_base + k_off[:, None] * 1 + n_off[None, :] * K
        w_tile = tl.load(
            w_ptr + w_off, mask=k_mask[:, None] & n_in[None, :], other=0.0
        )

        acc += tl.dot(x_tile, w_tile, input_precision=INPUT_PRECISION)

    tmp_off = k_split_id * (S * R) + row_idx[:, None] * R + n_off[None, :]
    tmp_mask = valid_m[:, None] & n_in[None, :]
    tl.store(tmp_ptr + tmp_off, acc, mask=tmp_mask)


@triton.jit
def _sgemm_lora_a_reduce(
    tmp_ptr,
    out_ptr,
    N: tl.constexpr,
    NUM_K_SPLITS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    off = pid * BLOCK + tl.arange(0, BLOCK)
    mask = off < N

    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for s in range(NUM_K_SPLITS):
        acc += tl.load(tmp_ptr + s * N + off, mask=mask)

    tl.store(out_ptr + off, acc.to(out_ptr.dtype.element_ty), mask=mask)


def sgemm_lora_a(x, weights, batch_info, stack_num=1):
    S, K = x.shape
    num_lora, R, _ = weights.shape

    seg_indptr = batch_info.seg_indptr
    weight_indices = batch_info.weight_indices
    permutation = batch_info.permutation
    has_perm = permutation is not None

    bs = batch_info.bs

    out = torch.empty((S, R), dtype=x.dtype, device=x.device)

    if bs <= 0:
        out.zero_()
        return out

    input_precision = "ieee"

    BLOCK_N = min(_next_pow2(R), 128)
    if R <= 32:
        BLOCK_M = 32
        BLOCK_K = 128
        want_k_splits = 8
        num_warps = 4
        num_stages = 2
    else:
        BLOCK_M = 64
        BLOCK_K = 256
        want_k_splits = 4
        num_warps = 4
        num_stages = 2

    is_fp32 = x.dtype == torch.float32
    wide_n = BLOCK_N >= 128
    if is_fp32 or wide_n:
        # Drop to the smaller K-tile (128, still pipelined at 2 stages); if even
        # that is risky (fp32 + wide N), fall back to a single software stage.
        if BLOCK_K > 128:
            BLOCK_K = 128
        if is_fp32 and wide_n:
            num_stages = 1

    max_m_tiles_per_seg = (S + BLOCK_M - 1) // BLOCK_M
    if max_m_tiles_per_seg < 1:
        max_m_tiles_per_seg = 1

    n_total = (R + BLOCK_N - 1) // BLOCK_N

    if is_fp32:
        want_k_splits = max(
            want_k_splits, K // max(BLOCK_K, 1)
        )  # whole-K sweep

    max_k_splits = max(1, K // BLOCK_K)
    num_k_splits = min(want_k_splits, max_k_splits)
    if num_k_splits < 1:
        num_k_splits = 1

    tmp = torch.empty(
        (num_k_splits, S, R), dtype=torch.float32, device=x.device
    )

    grid1 = (max_m_tiles_per_seg * num_k_splits, n_total, bs)
    _sgemm_lora_a_stage1[grid1](
        x,
        weights,
        tmp,
        seg_indptr,
        weight_indices,
        permutation if has_perm else x,
        K=K,
        R=R,
        S=S,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        NUM_K_SPLITS=num_k_splits,
        HAS_PERM=has_perm,
        INPUT_PRECISION=input_precision,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    BLOCK_REDUCE = 512
    N = S * R
    grid2 = ((N + BLOCK_REDUCE - 1) // BLOCK_REDUCE,)
    _sgemm_lora_a_reduce[grid2](
        tmp,
        out,
        N=N,
        NUM_K_SPLITS=num_k_splits,
        BLOCK=BLOCK_REDUCE,
        num_warps=4,
    )

    return out


__all__ = ["sgemm_lora_a"]
