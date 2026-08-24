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

"""Kunlunxin context attention with a static query-block kernel."""

import torch
import triton
import triton.language as tl

_MAX_GRID_PROGRAMS = 65535


@triton.jit
def _context_attention_static_q_kernel(
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
    batch_head_start,
    q_block_start,
    q_programs,
    q_heads: tl.constexpr,
    group_size: tl.constexpr,
    head_dim: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
):
    # One flat program owns one Q block: no outer grid-stride loop.
    program_id = tl.program_id(0)
    batch_head_offset = program_id // q_programs
    batch_head_id = batch_head_offset + batch_head_start
    batch_id = batch_head_id // q_heads
    q_head_id = batch_head_id - batch_id * q_heads
    q_block_id = q_block_start + program_id - batch_head_offset * q_programs
    seq_len = tl.load(b_seq_len + batch_id * stride_len).to(tl.int32)
    seq_start = tl.load(b_start_loc + batch_id * stride_start).to(tl.int32)
    kv_head_id = q_head_id // group_size
    offs_m = q_block_id * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)
    mask_m, mask_d = offs_m < seq_len, offs_d < head_dim
    q_tile = tl.load(
        q
        + (seq_start + offs_m[:, None]) * stride_qt
        + q_head_id * stride_qh
        + offs_d[None, :] * stride_qd,
        mask=mask_m[:, None] & mask_d[None, :],
        other=0.0,
    )
    neg_inf = float("-inf")
    running_max = tl.full([BLOCK_M], neg_inf, tl.float32)
    running_sum = tl.zeros([BLOCK_M], tl.float32)
    accumulator = tl.zeros([BLOCK_M, BLOCK_D], tl.float32)
    end_n = (
        tl.minimum((q_block_id + 1) * BLOCK_M, seq_len)
        if IS_CAUSAL
        else seq_len
    )
    # total_tokens is only a conservative host-side sequence bound. Empty
    # ragged-tail programs must not scan the full K/V sequence.
    end_n = tl.where(q_block_id * BLOCK_M < seq_len, end_n, 0)
    for start_n in range(0, end_n, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        key_pos = start_n + offs_n
        mask_n = key_pos < seq_len
        k_tile = tl.load(
            k
            + (seq_start + key_pos[None, :]) * stride_kt
            + kv_head_id * stride_kh
            + offs_d[:, None] * stride_kd,
            mask=mask_d[:, None] & mask_n[None, :],
            other=0.0,
        )
        scores = tl.dot(q_tile, k_tile) * (sm_scale * 1.4426950408889634)
        score_mask = mask_m[:, None] & mask_n[None, :]
        if IS_CAUSAL:
            score_mask = score_mask & (offs_m[:, None] >= key_pos[None, :])
        scores = tl.where(score_mask, scores, neg_inf)
        new_max = tl.maximum(running_max, tl.max(scores, axis=1))
        alpha = tl.exp2(running_max - new_max)
        probabilities = tl.exp2(scores - new_max[:, None])
        v_tile = tl.load(
            v
            + (seq_start + key_pos[:, None]) * stride_vt
            + kv_head_id * stride_vh
            + offs_d[None, :] * stride_vd,
            mask=mask_n[:, None] & mask_d[None, :],
            other=0.0,
        )
        accumulator *= alpha[:, None]
        accumulator += tl.dot(probabilities.to(v_tile.dtype), v_tile)
        running_sum = running_sum * alpha + tl.sum(probabilities, axis=1)
        running_max = new_max
    tl.store(
        out
        + (seq_start + offs_m[:, None]) * stride_ot
        + q_head_id * stride_oh
        + offs_d[None, :] * stride_od,
        accumulator / running_sum[:, None],
        mask=mask_m[:, None] & mask_d[None, :],
    )


def context_attention(
    q, k, v, b_start_loc, b_seq_len, max_input_len, is_causal
):
    """Run static Q blocks without truncating on a low max_input_len."""
    del max_input_len
    out = torch.empty(q.shape, device=q.device, dtype=torch.float32)
    batch_size = b_seq_len.numel()
    if batch_size == 0 or q.numel() == 0:
        return out
    q_heads, kv_heads, head_dim = q.shape[1], k.shape[1], q.shape[2]
    assert kv_heads == v.shape[1]
    assert q_heads % kv_heads == 0
    assert k.shape[2] == head_dim and v.shape[2] == head_dim
    # q.shape[0] is host-visible and bounds every packed sequence; no metadata
    # device-to-host read is needed to cover under-reported max_input_len.
    total_q_programs = triton.cdiv(q.shape[0], 32)
    batch_heads = batch_size * q_heads
    for q_block_start in range(0, total_q_programs, _MAX_GRID_PROGRAMS):
        q_programs = min(_MAX_GRID_PROGRAMS, total_q_programs - q_block_start)
        batch_heads_per_launch = max(1, _MAX_GRID_PROGRAMS // q_programs)
        for batch_head_start in range(0, batch_heads, batch_heads_per_launch):
            batch_head_count = min(
                batch_heads_per_launch, batch_heads - batch_head_start
            )
            _context_attention_static_q_kernel[
                (q_programs * batch_head_count,)
            ](
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
                batch_head_start=batch_head_start,
                q_block_start=q_block_start,
                q_programs=q_programs,
                q_heads=q_heads,
                group_size=q_heads // kv_heads,
                head_dim=head_dim,
                BLOCK_M=32,
                BLOCK_N=32,
                BLOCK_D=triton.next_power_of_2(head_dim),
                IS_CAUSAL=bool(is_causal),
                num_warps=4,
                num_stages=1,
            )
    return out


__all__ = ["context_attention"]
