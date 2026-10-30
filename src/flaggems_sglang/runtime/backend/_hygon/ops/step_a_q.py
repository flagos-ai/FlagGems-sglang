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
    q_ptr,
    b_ptr,
    out_ptr,
    seg_indptr_ptr,
    weight_ptr,
    lora_ranks_ptr,
    perm_ptr,
    H: tl.constexpr,
    QK: tl.constexpr,
    FK: tl.constexpr,
    RANK: tl.constexpr,
    KH: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HAS_PERM: tl.constexpr,
    IS_FP32: tl.constexpr,
    NEED_KN: tl.constexpr,
    USE_I64: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_shg = tl.program_id(1)
    NH: tl.constexpr = H // KH
    seg = pid_shg // NH
    hg = pid_shg % NH

    start = tl.load(seg_indptr_ptr + seg)
    end = tl.load(seg_indptr_ptr + seg + 1)
    w = tl.load(weight_ptr + seg)

    r = tl.load(lora_ranks_ptr + w)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    m_mask = offs_m < (end - start)
    if HAS_PERM:
        p = tl.load(perm_ptr + start + offs_m, mask=m_mask, other=0)
        phy = p.to(tl.int64) if USE_I64 else p
    else:
        s_off = start + offs_m
        phy = s_off.to(tl.int64) if USE_I64 else s_off

    offs_k = tl.arange(0, BLOCK_K)
    # Contiguity hints so the per-head x rows (offs_k runs) load vectorized.
    offs_k = tl.max_contiguous(tl.multiple_of(offs_k, BLOCK_K), BLOCK_K)
    offs_n = tl.arange(0, BLOCK_N)
    offs_h = tl.arange(0, KH)
    k_mask = offs_k < QK
    n_mask = offs_n < RANK

    # out[rows, hg*KH + i, :] — one 3D tile per token, contiguous over the
    # (head-in-group, rank) axes.
    out_ptrs = (
        out_ptr
        + phy[:, None, None] * (H * RANK)
        + (hg * KH + offs_h)[None, :, None] * RANK
        + offs_n[None, None, :]
    )
    store_mask = m_mask[:, None, None] & n_mask[None, None, :]

    if r > 0:
        # b tile base for head hg*KH: [BLOCK_K, BLOCK_N]
        w64 = w.to(tl.int64)
        bw = w64 if USE_I64 else w
        b_base = (
            b_ptr
            + bw * (H * FK * RANK)
            + (hg * KH) * (FK * RANK)
            + offs_k[:, None] * RANK
            + offs_n[None, :]
        )
        accs = tl.zeros((BLOCK_M, KH, BLOCK_N), dtype=tl.float32)
        for i in tl.static_range(KH):
            x_ptrs = (
                q_ptr
                + phy[:, None] * (H * QK)
                + (hg * KH + i) * QK
                + offs_k[None, :]
            )
            b_ptrs = b_base + i * (FK * RANK)
            if NEED_KN:
                # BLOCK_K/BLOCK_N may exceed QK/RANK (non-power-of-two shapes)
                x = tl.load(
                    x_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=0.0
                )
                b_i = tl.load(
                    b_ptrs, mask=k_mask[:, None] & n_mask[None, :], other=0.0
                )
            else:
                # BLOCK_K == QK and BLOCK_N == RANK: K/N masks are always true.
                x = tl.load(x_ptrs, mask=m_mask[:, None], other=0.0)
                b_i = tl.load(b_ptrs)
            if IS_FP32:
                a = tl.dot(x, b_i, input_precision="ieee")
            else:
                a = tl.dot(x, b_i)
            accs = tl.where(offs_h[None, :, None] == i, a[:, None, :], accs)
        tl.store(out_ptrs, accs.to(q_ptr.dtype.element_ty), mask=store_mask)
    else:
        tl.store(
            out_ptrs,
            tl.zeros((BLOCK_M, KH, BLOCK_N), dtype=q_ptr.dtype.element_ty),
            mask=store_mask,
        )


def _is_contig(t):
    # contiguous() check without the dispatcher round-trip
    return t is None or t.is_contiguous()


def _pick_config(H, qk_nope, rank, max_len):
    """Heuristic launch config (measured on the eval backend)."""
    BLOCK_K = max(16, triton.next_power_of_2(max(1, qk_nope)))
    BLOCK_N = max(16, triton.next_power_of_2(max(1, rank)))
    if H % 4 == 0 and 4 * BLOCK_K <= 1024:
        return 4, 32, BLOCK_K, BLOCK_N, 2, 1
    if H % 2 == 0 and 2 * BLOCK_K <= 1024:
        return 2, 32, BLOCK_K, BLOCK_N, 2, 1
    BM = min(128, max(16, triton.next_power_of_2(max(1, max_len))))
    return 1, BM, BLOCK_K, BLOCK_N, 4, 1


def step_a_q(q_nope, B_buf, batch_info, full_K_per_head):
    S, H, qk_nope = q_nope.shape
    rank = B_buf.shape[-1]
    # Kernel writes every token row (zero blocks for zero-rank adapters),
    # so no zero-init memset is needed — saves a full-output fill launch.
    out = torch.empty(S, H, rank, device=q_nope.device, dtype=q_nope.dtype)

    bs = batch_info.bs
    if S == 0 or H == 0 or rank == 0 or qk_nope == 0 or bs == 0:
        if S and H and rank:
            out.zero_()
        return out

    if not _is_contig(q_nope):
        q_nope = q_nope.contiguous()
    if not _is_contig(B_buf):
        B_buf = B_buf.contiguous()
    seg_indptr = batch_info.seg_indptr
    if not _is_contig(seg_indptr):
        seg_indptr = seg_indptr.contiguous()
    weight_indices = batch_info.weight_indices
    if not _is_contig(weight_indices):
        weight_indices = weight_indices.contiguous()
    lora_ranks = batch_info.lora_ranks
    if not _is_contig(lora_ranks):
        lora_ranks = lora_ranks.contiguous()
    perm_arg = batch_info.permutation
    has_perm = perm_arg is not None
    if has_perm and not _is_contig(perm_arg):
        perm_arg = perm_arg.contiguous()

    KH, BLOCK_M, BLOCK_K, BLOCK_N, num_warps, num_stages = _pick_config(
        H, qk_nope, rank, batch_info.max_len
    )

    max_chunks = (max(1, batch_info.max_len) + BLOCK_M - 1) // BLOCK_M
    num_lora = B_buf.shape[0]
    use_i64 = (
        max(
            S * H * max(1, qk_nope),
            num_lora * H * max(1, full_K_per_head) * max(1, rank),
            S * H * max(1, rank),
        )
        > 2**31 - 1
    )
    grid = (max_chunks, bs * (H // KH))
    _step_a_q_kernel[grid](
        q_nope,
        B_buf,
        out,
        seg_indptr,
        weight_indices,
        lora_ranks,
        perm_arg if has_perm else q_nope,
        H=H,
        QK=qk_nope,
        FK=full_K_per_head,
        RANK=rank,
        KH=KH,
        BLOCK_M=BLOCK_M,
        BLOCK_K=BLOCK_K,
        BLOCK_N=BLOCK_N,
        HAS_PERM=1 if has_perm else 0,
        IS_FP32=1 if q_nope.dtype == torch.float32 else 0,
        NEED_KN=0 if (BLOCK_K == qk_nope and BLOCK_N == rank) else 1,
        USE_I64=1 if use_i64 else 0,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return out


__all__ = ["step_a_q"]
