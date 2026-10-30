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

"""Operator: lora/step_a_q"""

import torch
import triton
import triton.language as tl


@triton.jit
def _step_a_q_kernel(
    q_ptr,  # [S, H, K]
    b_ptr,  # [num_lora, H * full_K, R]
    seg_indptr_ptr,  # [bs + 1] int32
    weight_idx_ptr,  # [bs] int32
    lora_ranks_ptr,  # [num_lora] int32
    perm_ptr,  # [S] int32 (unused when PERM is False)
    out_ptr,  # [S, H, R]
    H: tl.constexpr,
    K: tl.constexpr,  # qk_nope
    R: tl.constexpr,  # rank == B_buf.shape[-1]; compile-time column extent
    RN: tl.constexpr,  # rank padded up to >= 16 for tl.dot
    full_K: tl.constexpr,  # qk_nope + v_head_dim: B row stride per head
    PERM: tl.constexpr,
    IEEE: tl.constexpr,  # fp32 inputs -> exact fp32 dot
    BLOCK_S: tl.constexpr,
    BLOCK_K: tl.constexpr,
    K_MASK: tl.constexpr,  # K % BLOCK_K != 0
    TILES_PER_SEG: tl.constexpr,  # cdiv(max_len, BLOCK_S) == grid extent / bs
):
    pid = tl.program_id(0)
    h = tl.program_id(1)

    # TILES_PER_SEG is the same value the host used to build the grid, so
    # ``seg`` is always < bs: no bounds guard is needed.
    seg = pid // TILES_PER_SEG
    tile = pid % TILES_PER_SEG

    start = tl.load(seg_indptr_ptr + seg)
    seg_len = tl.load(seg_indptr_ptr + seg + 1) - start
    tile_start = tile * BLOCK_S
    if tile_start >= seg_len:
        return

    # Rank lookup is an independent load: issue it before the row gather so the
    # load/store latency overlaps with the q/B fetches below.
    w = tl.load(weight_idx_ptr + seg)
    r = tl.load(lora_ranks_ptr + w)

    offs_s = tile_start + tl.arange(0, BLOCK_S)
    s_mask = offs_s < seg_len
    if PERM:
        rows = tl.load(perm_ptr + start + offs_s, mask=s_mask, other=0)
    else:
        rows = start + offs_s

    offs_r = tl.arange(0, RN)
    # R is a constexpr: this mask is a compile-time constant pattern, so the B
    # tile load stays a uniform block load (see module docstring).
    r_mask = offs_r < R

    if r == 0:
        # The reference leaves rank-0 adapters untouched; ``out`` is allocated
        # uninitialized, so this program must write the zeros itself.
        tl.store(
            out_ptr + rows[:, None] * (H * R) + h * R + offs_r[None, :],
            tl.zeros((BLOCK_S, RN), dtype=out_ptr.dtype.element_ty),
            mask=s_mask[:, None] & r_mask[None, :],
        )
        return

    offs_k = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_S, RN), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        offs = k0 + offs_k
        if K_MASK:
            a = tl.load(
                q_ptr + rows[:, None] * (H * K) + h * K + offs[None, :],
                mask=s_mask[:, None] & (offs < K)[None, :],
                other=0.0,
            )
            bt = tl.load(
                b_ptr
                + w * (H * full_K * R)
                + (h * full_K + offs[:, None]) * R
                + offs_r[None, :],
                mask=(offs < K)[:, None] & r_mask[None, :],
                other=0.0,
            )
        else:
            a = tl.load(
                q_ptr + rows[:, None] * (H * K) + h * K + offs[None, :],
                mask=s_mask[:, None],
                other=0.0,
                cache_modifier=".cg",
            )
            bt = tl.load(
                b_ptr
                + w * (H * full_K * R)
                + (h * full_K + offs[:, None]) * R
                + offs_r[None, :],
                mask=r_mask[None, :],
                other=0.0,
                cache_modifier=".cg",
            )
        if IEEE:
            acc = tl.dot(a, bt, acc, input_precision="ieee")
        else:
            acc = tl.dot(a, bt, acc)

    tl.store(
        out_ptr + rows[:, None] * (H * R) + h * R + offs_r[None, :],
        acc.to(out_ptr.dtype.element_ty),
        mask=s_mask[:, None] & r_mask[None, :],
    )


def _pick_block_s(max_len):
    """Row-tile size from the longest segment: small segments want many small
    programs (saturation), long segments want fewer wider tiles per head."""
    if max_len <= 64:
        return 64
    if max_len <= 256:
        return 128
    return 256


def _pick_block_k(qk_nope, max_len):
    """K-tile width."""
    bk = 64 if (max_len >= 128 and qk_nope >= 64) else 32
    return min(bk, triton.next_power_of_2(max(16, qk_nope)))


def step_a_q(q_nope, B_buf, batch_info, full_K_per_head):
    S, H, qk_nope = q_nope.shape
    rank = B_buf.shape[-1]

    bs = batch_info.bs
    # Empty/degenerate shapes produce no programs, so nothing would initialize
    # the output; those go through a zeroed allocation instead.
    if bs == 0 or S == 0 or qk_nope == 0 or rank == 0:
        return q_nope.new_zeros(S, H, rank)

    seg_indptr = batch_info.seg_indptr
    max_len = batch_info.max_len
    if max_len is None or max_len <= 0:
        max_len = int((seg_indptr[1:] - seg_indptr[:-1]).max().item())
        if max_len == 0:
            return q_nope.new_zeros(S, H, rank)

    out = torch.empty(S, H, rank, device=q_nope.device, dtype=q_nope.dtype)

    permutation = batch_info.permutation
    # tl.dot needs every dim >= 16; the store mask trims back to the real rank.
    rn = max(16, triton.next_power_of_2(rank))
    block_s = _pick_block_s(max_len)
    block_k = _pick_block_k(qk_nope, max_len)
    k_mask = (qk_nope % block_k) != 0
    num_stages = 4 if max_len < 128 else 3
    # Two warps is optimal on C550 for both bench shapes (see module
    # docstring).
    num_warps = 2

    tiles_per_seg = triton.cdiv(max_len, block_s)
    grid = (bs * tiles_per_seg, H)
    _step_a_q_kernel[grid](
        q_nope,
        B_buf,
        seg_indptr,
        batch_info.weight_indices,
        batch_info.lora_ranks,
        permutation if permutation is not None else q_nope,
        out,
        H=H,
        K=qk_nope,
        R=rank,
        RN=rn,
        full_K=full_K_per_head,
        PERM=permutation is not None,
        IEEE=q_nope.dtype == torch.float32,
        BLOCK_S=block_s,
        BLOCK_K=block_k,
        K_MASK=k_mask,
        TILES_PER_SEG=tiles_per_seg,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return out


__all__ = ["step_a_q"]
