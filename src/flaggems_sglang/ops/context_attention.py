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

"""Packed variable-length context attention implemented in Triton.

The online-softmax structure follows the Apache-2.0 SGLang/LightLLM
``context_attention_fwd`` lineage.  This version keeps the competition
interface self-contained and adds explicit strides, actual-head-dimension
masking, float32 output, and conservative cross-backend launch parameters.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _context_attention_kernel(
    q,
    k,
    v,
    b_start_loc,
    b_seq_len,
    out,
    sm_scale,
    stride_qt: tl.constexpr,
    stride_qh: tl.constexpr,
    stride_qd: tl.constexpr,
    stride_kt: tl.constexpr,
    stride_kh: tl.constexpr,
    stride_kd: tl.constexpr,
    stride_vt: tl.constexpr,
    stride_vh: tl.constexpr,
    stride_vd: tl.constexpr,
    stride_ot: tl.constexpr,
    stride_oh: tl.constexpr,
    stride_od: tl.constexpr,
    stride_start: tl.constexpr,
    stride_len: tl.constexpr,
    batch_size,
    group_size: tl.constexpr,
    head_dim: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
):
    work_id = tl.program_id(0)
    q_head_id = tl.program_id(1)

    # Flatten the real per-sequence Q blocks instead of padding every sequence
    # to max_input_len.  Besides avoiding ragged-batch waste, this honors the
    # public contract in which max_input_len is only a planning hint and may be
    # smaller than an actual sequence length.
    block_prefix = 0
    q_block_id = 0
    seq_start = 0
    seq_len = 0
    owns_work = False
    for candidate_batch in range(0, batch_size):
        candidate_len = tl.load(b_seq_len + candidate_batch * stride_len).to(
            tl.int32
        )
        candidate_blocks = (candidate_len + BLOCK_M - 1) // BLOCK_M
        is_owner = (work_id >= block_prefix) & (
            work_id < block_prefix + candidate_blocks
        )
        q_block_id = tl.where(is_owner, work_id - block_prefix, q_block_id)
        seq_len = tl.where(is_owner, candidate_len, seq_len)
        candidate_start = tl.load(
            b_start_loc + candidate_batch * stride_start
        ).to(tl.int32)
        seq_start = tl.where(is_owner, candidate_start, seq_start)
        owns_work |= is_owner
        block_prefix += candidate_blocks
    kv_head_id = q_head_id // group_size

    offs_m = q_block_id * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)
    mask_m = owns_work & (offs_m < seq_len)
    mask_d = offs_d < head_dim

    q_ptrs = (
        q
        + (seq_start + offs_m[:, None]) * stride_qt
        + q_head_id * stride_qh
        + offs_d[None, :] * stride_qd
    )
    q_tile = tl.load(
        q_ptrs,
        mask=mask_m[:, None] & mask_d[None, :],
        other=0.0,
    )

    neg_inf = float("-inf")
    log2e = 1.4426950408889634
    qk_scale = sm_scale * log2e
    running_max = tl.full([BLOCK_M], neg_inf, tl.float32)
    running_sum = tl.zeros([BLOCK_M], tl.float32)
    accumulator = tl.zeros([BLOCK_M, BLOCK_D], tl.float32)

    if IS_CAUSAL:
        end_n = tl.minimum((q_block_id + 1) * BLOCK_M, seq_len)
    else:
        end_n = seq_len

    for start_n in range(0, end_n, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        key_pos = start_n + offs_n
        mask_n = key_pos < seq_len

        k_ptrs = (
            k
            + (seq_start + key_pos[None, :]) * stride_kt
            + kv_head_id * stride_kh
            + offs_d[:, None] * stride_kd
        )
        k_tile = tl.load(
            k_ptrs,
            mask=mask_d[:, None] & mask_n[None, :],
            other=0.0,
        )

        scores = tl.dot(q_tile, k_tile) * qk_scale
        if IS_CAUSAL:
            score_mask = (
                mask_m[:, None]
                & mask_n[None, :]
                & (offs_m[:, None] >= key_pos[None, :])
            )
        else:
            score_mask = mask_m[:, None] & mask_n[None, :]
        scores = tl.where(score_mask, scores, neg_inf)

        block_max = tl.max(scores, axis=1)
        new_max = tl.maximum(running_max, block_max)
        alpha = tl.exp2(running_max - new_max)
        probabilities = tl.exp2(scores - new_max[:, None])

        v_ptrs = (
            v
            + (seq_start + key_pos[:, None]) * stride_vt
            + kv_head_id * stride_vh
            + offs_d[None, :] * stride_vd
        )
        v_tile = tl.load(
            v_ptrs,
            mask=mask_n[:, None] & mask_d[None, :],
            other=0.0,
        )

        accumulator *= alpha[:, None]
        accumulator += tl.dot(probabilities.to(v_tile.dtype), v_tile)
        running_sum = running_sum * alpha + tl.sum(probabilities, axis=1)
        running_max = new_max

    result = accumulator / running_sum[:, None]
    out_ptrs = (
        out
        + (seq_start + offs_m[:, None]) * stride_ot
        + q_head_id * stride_oh
        + offs_d[None, :] * stride_od
    )
    tl.store(out_ptrs, result, mask=mask_m[:, None] & mask_d[None, :])


def _run_context_attention(
    q, k, v, b_start_loc, b_seq_len, max_input_len, is_causal
):
    """Launch the common packed-attention kernel and return float32 output."""
    out = torch.empty(q.shape, device=q.device, dtype=torch.float32)
    batch_size = b_seq_len.numel()
    if batch_size == 0 or q.numel() == 0:
        return out

    q_heads = q.shape[1]
    kv_heads = k.shape[1]
    head_dim = q.shape[2]
    assert kv_heads == v.shape[1]
    assert q_heads % kv_heads == 0
    assert k.shape[2] == head_dim and v.shape[2] == head_dim

    block_d = triton.next_power_of_2(head_dim)
    block_m = 64
    block_n = 64
    num_warps = 4 if head_dim <= 64 else 8
    # sum(ceil(seq_len / block_m)) is at most
    # ceil(total_tokens / block_m) + batch_size - 1 for a packed batch.
    # One extra slot keeps the bound simple and all excess programs are masked.
    grid = (triton.cdiv(q.shape[0], block_m) + batch_size, q_heads)

    _context_attention_kernel[grid](
        q,
        k,
        v,
        b_start_loc,
        b_seq_len,
        out,
        head_dim**-0.5,
        stride_qt=q.stride(0),
        stride_qh=q.stride(1),
        stride_qd=q.stride(2),
        stride_kt=k.stride(0),
        stride_kh=k.stride(1),
        stride_kd=k.stride(2),
        stride_vt=v.stride(0),
        stride_vh=v.stride(1),
        stride_vd=v.stride(2),
        stride_ot=out.stride(0),
        stride_oh=out.stride(1),
        stride_od=out.stride(2),
        stride_start=b_start_loc.stride(0),
        stride_len=b_seq_len.stride(0),
        batch_size=batch_size,
        group_size=q_heads // kv_heads,
        head_dim=head_dim,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_D=block_d,
        IS_CAUSAL=bool(is_causal),
        num_warps=num_warps,
        num_stages=1,
    )
    return out


def context_attention(
    q, k, v, b_start_loc, b_seq_len, max_input_len, is_causal
):
    """Compute packed variable-length scaled dot-product attention."""
    return _run_context_attention(
        q, k, v, b_start_loc, b_seq_len, max_input_len, is_causal
    )


def reference(q, k, v, b_start_loc, b_seq_len, max_input_len, is_causal):
    """Competition-compatible alias implemented as a real function."""
    return _run_context_attention(
        q, k, v, b_start_loc, b_seq_len, max_input_len, is_causal
    )


__all__ = ["context_attention"]
