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


def _next_pow2(n):
    p = 1
    while p < n:
        p *= 2
    return p


@triton.jit
def _step_a_q_kernel(
    x_ptr,  # (S, H, K)          q_nope
    w_ptr,  # (L, H*FULL_K, N)   B_buf
    out_ptr,  # (S, H, N)          output
    S,  # rows of x
    K,  # qk_nope (contraction)
    N,  # rank (output cols)
    x_stride_s,
    x_stride_h,
    x_stride_k,
    w_stride_l,
    w_stride_row,
    w_stride_col,
    o_stride_s,
    o_stride_h,
    o_stride_n,
    seg_indptr_ptr,  # (num_segments + 1,) int32
    weight_indices_ptr,  # (num_segments,)     int32
    lora_ranks_ptr,  # (num_lora,)         int32
    perm_ptr,  # (S,) int32 or dummy
    num_segments,
    FULL_K: tl.constexpr,  # per-head row stride in B (qk_nope + v_head_dim)
    BLOCK_S: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    H_PER: tl.constexpr,  # heads statically unrolled per program
    HAS_PERM: tl.constexpr,
    EVEN_K: tl.constexpr,  # K  % BLOCK_K == 0: drop the k-column mask
    EVEN_N: tl.constexpr,  # N  % BLOCK_N == 0: drop the n-column mask
):
    batch_id = tl.program_id(2)
    head_base = tl.program_id(1) * H_PER
    pid = tl.program_id(0)

    if batch_id >= num_segments:
        return

    w_index = tl.load(weight_indices_ptr + batch_id)
    adapter_rank = tl.load(lora_ranks_ptr + w_index)

    seg_start = tl.load(seg_indptr_ptr + batch_id)
    seg_end = tl.load(seg_indptr_ptr + batch_id + 1)
    seg_len = seg_end - seg_start
    if seg_len == 0:
        return

    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_s = pid // num_pid_n
    pid_n = pid % num_pid_n
    if pid_s * BLOCK_S >= seg_len:
        return

    s_off = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    n_off = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    k_off = tl.arange(0, BLOCK_K)

    row_mask = s_off < seg_len
    if HAS_PERM:
        s_phys = tl.load(
            perm_ptr + seg_start + s_off, mask=row_mask, other=0
        ).to(tl.int32)
    else:
        s_phys = seg_start + s_off

    if EVEN_N:
        n_idx = n_off
    else:
        n_idx = tl.minimum(n_off, N - 1)
    n_mask = n_off < N

    x_base = s_phys[:, None] * x_stride_s

    for hh in tl.static_range(H_PER):
        head_id = head_base + hh
        x_off = x_base + head_id * x_stride_h
        w_base = w_index * w_stride_l + head_id * FULL_K * w_stride_row

        acc = tl.zeros((BLOCK_S, BLOCK_N), dtype=tl.float32)
        if adapter_rank != 0:
            for kb in range(0, tl.cdiv(K, BLOCK_K)):
                if EVEN_K:
                    k_idx = kb * BLOCK_K + k_off
                    x_tile = tl.load(
                        x_ptr + x_off + k_idx[None, :] * x_stride_k,
                        mask=row_mask[:, None],
                        other=0.0,
                    )
                    if EVEN_N:
                        w_tile = tl.load(
                            w_ptr
                            + w_base
                            + k_idx[:, None] * w_stride_row
                            + n_idx[None, :] * w_stride_col
                        )
                    else:
                        w_tile = tl.load(
                            w_ptr
                            + w_base
                            + k_idx[:, None] * w_stride_row
                            + n_idx[None, :] * w_stride_col,
                            mask=n_mask[None, :],
                            other=0.0,
                        )
                else:
                    cur_k = kb * BLOCK_K + k_off
                    k_mask = cur_k < K
                    k_idx = tl.minimum(cur_k, K - 1)
                    x_tile = tl.load(
                        x_ptr + x_off + k_idx[None, :] * x_stride_k,
                        mask=row_mask[:, None] & k_mask[None, :],
                        other=0.0,
                    )
                    if EVEN_N:
                        w_tile = tl.load(
                            w_ptr
                            + w_base
                            + k_idx[:, None] * w_stride_row
                            + n_idx[None, :] * w_stride_col,
                            mask=k_mask[:, None],
                            other=0.0,
                        )
                    else:
                        w_tile = tl.load(
                            w_ptr
                            + w_base
                            + k_idx[:, None] * w_stride_row
                            + n_idx[None, :] * w_stride_col,
                            mask=k_mask[:, None] & n_mask[None, :],
                            other=0.0,
                        )
                acc += tl.dot(x_tile, w_tile, input_precision="ieee")

        acc = acc.to(out_ptr.dtype.element_ty)
        o_off = (
            s_phys[:, None] * o_stride_s
            + head_id * o_stride_h
            + n_idx[None, :] * o_stride_n
        )
        if EVEN_N:
            tl.store(out_ptr + o_off, acc, mask=row_mask[:, None])
        else:
            tl.store(
                out_ptr + o_off, acc, mask=row_mask[:, None] & n_mask[None, :]
            )


def _max_segment_len(batch_info):
    if batch_info.max_len is not None:
        return int(batch_info.max_len)
    if batch_info.seg_lens is not None:
        return int(batch_info.seg_lens.max().item())
    seg = batch_info.seg_indptr
    return int((seg[1:] - seg[:-1]).max().item())


def _pick_h_per(H):
    """Largest head-unroll factor that divides ``H`` (8, 4, 2, else 1)."""
    for hp in (8, 4, 2):
        if H % hp == 0:
            return hp
    return 1


def step_a_q(q_nope, B_buf, batch_info, full_K_per_head):
    S, H, qk_nope = q_nope.shape
    rank = B_buf.shape[-1]
    out = torch.empty((S, H, rank), device=q_nope.device, dtype=q_nope.dtype)

    num_segments = batch_info.bs
    perm = batch_info.permutation
    has_perm = perm is not None

    max_seg = _max_segment_len(batch_info)

    BLOCK_S = max(16, min(256, _next_pow2(max_seg)))
    BLOCK_N = max(16, _next_pow2(rank))
    BLOCK_K = max(16, min(128, _next_pow2(qk_nope)))
    even_k = (qk_nope % BLOCK_K) == 0
    even_n = (rank % BLOCK_N) == 0

    H_PER = _pick_h_per(H)

    num_pid_s = triton.cdiv(max_seg, BLOCK_S)
    num_pid_n = triton.cdiv(rank, BLOCK_N)
    grid = (num_pid_s * num_pid_n, H // H_PER, num_segments)

    _step_a_q_kernel[grid](
        q_nope,
        B_buf,
        out,
        S,
        qk_nope,
        rank,
        q_nope.stride(0),
        q_nope.stride(1),
        q_nope.stride(2),
        B_buf.stride(0),
        B_buf.stride(1),
        B_buf.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        batch_info.seg_indptr,
        batch_info.weight_indices,
        batch_info.lora_ranks,
        perm if has_perm else batch_info.seg_indptr,
        num_segments,
        FULL_K=full_K_per_head,
        BLOCK_S=BLOCK_S,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        H_PER=H_PER,
        HAS_PERM=has_perm,
        EVEN_K=even_k,
        EVEN_N=even_n,
        num_warps=4,
        num_stages=2,
    )
    return out


__all__ = ["step_a_q"]
