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

"""Triton kernel for lora/step_a_q."""

import torch
import triton
import triton.language as tl


def _next_pow2(x):
    return max(16, triton.next_power_of_2(int(x)))


_CONFIGS = [
    # MD = 1 -- the previous per-row-block geometry, kept so autotune can
    # always fall back to it.
    triton.Config({"BLOCK_M": 32, "MD": 1, "HG": 1}, num_warps=8),
    triton.Config({"BLOCK_M": 32, "MD": 1, "HG": 2}, num_warps=8),
    triton.Config({"BLOCK_M": 32, "MD": 1, "HG": 1}, num_warps=16),
    triton.Config({"BLOCK_M": 32, "MD": 1, "HG": 2}, num_warps=16),
    triton.Config({"BLOCK_M": 32, "MD": 1, "HG": 4}, num_warps=16),
    triton.Config({"BLOCK_M": 64, "MD": 1, "HG": 1}, num_warps=8),
    triton.Config({"BLOCK_M": 64, "MD": 1, "HG": 2}, num_warps=8),
    triton.Config({"BLOCK_M": 64, "MD": 1, "HG": 1}, num_warps=16),
    triton.Config({"BLOCK_M": 64, "MD": 1, "HG": 2}, num_warps=16),
    triton.Config({"BLOCK_M": 64, "MD": 1, "HG": 2}, num_warps=32),
    triton.Config({"BLOCK_M": 64, "MD": 1, "HG": 4}, num_warps=32),
    triton.Config({"BLOCK_M": 128, "MD": 1, "HG": 1}, num_warps=16),
    triton.Config({"BLOCK_M": 128, "MD": 1, "HG": 2}, num_warps=16),
    triton.Config({"BLOCK_M": 128, "MD": 1, "HG": 2}, num_warps=32),
    triton.Config({"BLOCK_M": 128, "MD": 1, "HG": 4}, num_warps=32),
    # MD > 1 -- several row-blocks share one B-tile load and expose more
    # independent dots to the scheduler.
    triton.Config({"BLOCK_M": 32, "MD": 2, "HG": 1}, num_warps=8),
    triton.Config({"BLOCK_M": 32, "MD": 2, "HG": 1}, num_warps=16),
    triton.Config({"BLOCK_M": 32, "MD": 2, "HG": 2}, num_warps=8),
    triton.Config({"BLOCK_M": 64, "MD": 2, "HG": 1}, num_warps=8),
    triton.Config({"BLOCK_M": 64, "MD": 2, "HG": 1}, num_warps=16),
    triton.Config({"BLOCK_M": 64, "MD": 2, "HG": 1}, num_warps=32),
    triton.Config({"BLOCK_M": 64, "MD": 2, "HG": 2}, num_warps=16),
    triton.Config({"BLOCK_M": 128, "MD": 2, "HG": 1}, num_warps=16),
    triton.Config({"BLOCK_M": 128, "MD": 2, "HG": 1}, num_warps=32),
    triton.Config({"BLOCK_M": 32, "MD": 4, "HG": 1}, num_warps=8),
    triton.Config({"BLOCK_M": 32, "MD": 4, "HG": 1}, num_warps=16),
    triton.Config({"BLOCK_M": 64, "MD": 4, "HG": 1}, num_warps=8),
    triton.Config({"BLOCK_M": 64, "MD": 4, "HG": 1}, num_warps=16),
    triton.Config({"BLOCK_M": 64, "MD": 4, "HG": 1}, num_warps=32),
    triton.Config({"BLOCK_M": 128, "MD": 4, "HG": 1}, num_warps=16),
    triton.Config({"BLOCK_M": 128, "MD": 4, "HG": 1}, num_warps=32),
    triton.Config({"BLOCK_M": 32, "MD": 8, "HG": 1}, num_warps=8),
    triton.Config({"BLOCK_M": 64, "MD": 8, "HG": 1}, num_warps=16),
    triton.Config({"BLOCK_M": 64, "MD": 8, "HG": 1}, num_warps=32),
]


@triton.autotune(configs=_CONFIGS, key=["H", "QK_NOPE", "RANK", "MAX_LEN"])
@triton.jit
def _step_a_q(
    q_ptr,
    b_ptr,
    out_ptr,
    seg_indptr_ptr,
    weight_ptr,
    ranks_ptr,
    perm_ptr,
    H,
    QK_NOPE,
    RANK,
    FULL_K,
    MAX_LEN,
    BS,
    S,
    stride_qs,
    stride_qh,
    stride_bs,
    stride_bh,
    stride_os,
    stride_oh,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_R: tl.constexpr,
    MD: tl.constexpr,
    HG: tl.constexpr,
    HAS_PERM: tl.constexpr,
    IS_F32: tl.constexpr,
):
    n_m = tl.cdiv(MAX_LEN, BLOCK_M * MD)
    nhg = tl.cdiv(H, HG)
    pid = tl.program_id(0)
    hgrp = pid % nhg
    tmp = pid // nhg
    m_blk = tmp % n_m
    b = tmp // n_m

    start = tl.load(seg_indptr_ptr + b)
    end = tl.load(seg_indptr_ptr + b + 1)
    w = tl.load(weight_ptr + b)
    r = tl.load(ranks_ptr + w)

    k_offs = tl.arange(0, BLOCK_K)
    r_offs = tl.arange(0, BLOCK_R)
    mask_k = k_offs < QK_NOPE
    mask_r = r_offs < RANK
    head0 = hgrp * HG
    base = m_blk * BLOCK_M * MD
    # Hoist the per-adapter row base of B_buf out of the head loop: only the
    # intra-head-block row offset changes from one head to the next.
    b_adapter = b_ptr + w.to(tl.int64) * stride_bs

    if (start < end) & (r != 0):
        for i in tl.static_range(HG):
            h = head0 + i
            if h < H:
                # B tile for this head is loaded once and reused by all MD
                # row-blocks below.
                bmat = tl.load(
                    b_adapter
                    + (h * FULL_K + k_offs[:, None]) * stride_bh
                    + r_offs[None, :],
                    mask=mask_k[:, None] & mask_r[None, :],
                    other=0.0,
                )
                for j in tl.static_range(MD):
                    m_offs = base + j * BLOCK_M + tl.arange(0, BLOCK_M)
                    pos = start + m_offs
                    mask_m = pos < end
                    if HAS_PERM:
                        rows = tl.load(
                            perm_ptr + pos, mask=mask_m, other=0
                        ).to(tl.int64)
                    else:
                        rows = pos.to(tl.int64)
                    q = tl.load(
                        q_ptr
                        + rows[:, None] * stride_qs
                        + h * stride_qh
                        + k_offs[None, :],
                        mask=mask_m[:, None] & mask_k[None, :],
                        other=0.0,
                    )
                    if IS_F32:
                        acc = tl.dot(
                            q.to(tl.float32),
                            bmat.to(tl.float32),
                            input_precision="ieee",
                        )
                    else:
                        acc = tl.dot(q, bmat)
                    tl.store(
                        out_ptr
                        + rows[:, None] * stride_os
                        + h * stride_oh
                        + r_offs[None, :],
                        acc.to(out_ptr.dtype.element_ty),
                        mask=mask_m[:, None] & mask_r[None, :],
                    )
    else:
        for i in tl.static_range(HG):
            h = head0 + i
            if h < H:
                for j in tl.static_range(MD):
                    m_offs = base + j * BLOCK_M + tl.arange(0, BLOCK_M)
                    pos = start + m_offs
                    mask_m = pos < end
                    if HAS_PERM:
                        rows = tl.load(
                            perm_ptr + pos, mask=mask_m, other=0
                        ).to(tl.int64)
                    else:
                        rows = pos.to(tl.int64)
                    tl.store(
                        out_ptr
                        + rows[:, None] * stride_os
                        + h * stride_oh
                        + r_offs[None, :],
                        tl.zeros(
                            (BLOCK_M, BLOCK_R), dtype=out_ptr.dtype.element_ty
                        ),
                        mask=mask_m[:, None] & mask_r[None, :],
                    )

    # Rows past the last segment boundary are never owned by a segment program;
    # zero them once from the first program (a no-op when S == seg_indptr[BS]).
    if pid == 0:
        total = tl.load(seg_indptr_ptr + BS)
        if total < S:
            for rr in tl.range(total, S):
                for hh in tl.range(0, H):
                    tl.store(
                        out_ptr + rr * stride_os + hh * stride_oh + r_offs,
                        tl.zeros((BLOCK_R,), dtype=out_ptr.dtype.element_ty),
                        mask=mask_r,
                    )


def step_a_q(q_nope, B_buf, batch_info, full_K_per_head):
    S, H, qk_nope = q_nope.shape
    rank = B_buf.shape[-1]

    # `empty` + a fully-writing kernel avoids a redundant full-tensor `zeros`
    # pass; every output element is produced by the kernel above.
    out = torch.empty(S, H, rank, device=q_nope.device, dtype=q_nope.dtype)

    seg_indptr = batch_info.seg_indptr
    weight_indices = batch_info.weight_indices
    lora_ranks = batch_info.lora_ranks
    permutation = batch_info.permutation

    bs = int(batch_info.bs)
    if bs == 0 or S == 0:
        return out

    max_len = int(batch_info.max_len)
    if max_len <= 0:
        return out

    q_nope = q_nope.contiguous()
    B_buf = B_buf.contiguous()

    device = q_nope.device
    seg_indptr = seg_indptr.to(device=device, dtype=torch.int32).contiguous()
    weight_indices = weight_indices.to(
        device=device, dtype=torch.int32
    ).contiguous()
    lora_ranks = lora_ranks.to(device=device, dtype=torch.int32).contiguous()
    has_perm = permutation is not None
    if has_perm:
        perm = permutation.to(device=device, dtype=torch.int32).contiguous()
    else:
        perm = seg_indptr  # dummy, never read

    def grid(meta):
        return (
            bs
            * triton.cdiv(max_len, meta["BLOCK_M"] * meta["MD"])
            * triton.cdiv(H, meta["HG"]),
        )

    _step_a_q[grid](
        q_nope,
        B_buf,
        out,
        seg_indptr,
        weight_indices,
        lora_ranks,
        perm,
        H,
        qk_nope,
        rank,
        int(full_K_per_head),
        max_len,
        bs,
        S,
        q_nope.stride(0),
        q_nope.stride(1),
        B_buf.stride(0),
        B_buf.stride(1),
        out.stride(0),
        out.stride(1),
        BLOCK_K=_next_pow2(qk_nope),
        BLOCK_R=_next_pow2(rank),
        HAS_PERM=has_perm,
        IS_F32=(q_nope.dtype == torch.float32),
    )

    return out


__all__ = ["step_a_q"]
