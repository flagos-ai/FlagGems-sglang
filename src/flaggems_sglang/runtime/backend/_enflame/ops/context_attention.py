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
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language
# governing permissions and limitations under the License.

"""context_attention — Prefill/context-stage scaled dot-product attention.

Computes multi-head attention for packed variable-length sequences (causal or
non-causal). Inputs are in *packed* format: all sequences are concatenated
along the token dimension, and b_start_loc / b_seq_len mark each sequence's
slice.

Semantics (matches the PyTorch reference exactly):

    q, k, v: [total_tokens, num_heads, head_dim]
    b_start_loc: [batch]  — start token index for each sequence
    b_seq_len:   [batch]  — number of tokens in each sequence
    max_input_len: int    — max sequence length (reserved for temp allocation)
    is_causal: bool       — apply causal (lower-triangular) mask

    for i in range(batch):
        start = b_start_loc[i]
        end   = start + b_seq_len[i]
        # transpose to [num_heads, seq_len, head_dim] for SDPA
        qi = q[start:end].permute(1, 0, 2).float()
        ki = k[start:end].permute(1, 0, 2).float()
        vi = v[start:end].permute(1, 0, 2).float()
        out[start:end] = F.scaled_dot_product_attention(
            qi, ki, vi, is_causal=is_causal
        ).permute(1, 0, 2)

    return out   # [total_tokens, num_heads, head_dim]  float32

This kernel is written in portable Triton only. It must NOT call any
pre-compiled / vendor-specific cached operator — the whole computation is done
inside the Triton kernel so it is portable across supported chips.

Optimization strategy (v0):
    Baseline: one program per (sequence, head) work-item, persistent loop over
    batch*num_heads total items. For each (seq, head), iterates query tiles of
    BLOCK_M rows; for each query tile does flash-attention style online softmax
    over key/value tiles of BLOCK_N columns. Grid is capped at GRID_CAP.
    No autotune.
"""

import math

import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------

BLOCK_M = 64  # query tile size  (rows of the attention matrix per program)
BLOCK_N = 64  # key/value tile size (columns scanned per inner loop)
GRID_CAP = 512  # max programs to bound dispatch overhead


@triton.jit
def _context_attn_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    out_ptr,
    b_start_loc_ptr,
    b_seq_len_ptr,
    num_heads,
    head_dim,
    sm_scale,
    is_causal,
    batch,
    stride_qt,
    stride_qh,
    stride_qd,
    stride_kt,
    stride_kh,
    stride_kd,
    stride_vt,
    stride_vh,
    stride_vd,
    stride_ot,
    stride_oh,
    stride_od,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Per-(seq, head) persistent kernel with flash-attention style inner loop.

    Each program handles all query tiles for one (seq_id, head_id) pair.
    Online softmax (running max + denominator) avoids materialising full QK^T.
    """
    pid = tl.program_id(0)
    total_items = batch * num_heads

    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < head_dim

    for item in range(pid, total_items, tl.num_programs(0)):
        seq_id = item // num_heads
        head_id = item % num_heads

        seq_start = tl.load(b_start_loc_ptr + seq_id).to(tl.int32)
        seq_length = tl.load(b_seq_len_ptr + seq_id).to(tl.int32)

        # base pointers for this (seq, head)
        q_base = q_ptr + seq_start * stride_qt + head_id * stride_qh
        k_base = k_ptr + seq_start * stride_kt + head_id * stride_kh
        v_base = v_ptr + seq_start * stride_vt + head_id * stride_vh
        o_base = out_ptr + seq_start * stride_ot + head_id * stride_oh

        # --- Outer loop: query tiles (rows) ---
        for m_start in range(0, seq_length, BLOCK_M):
            offs_m = m_start + tl.arange(0, BLOCK_M)
            mask_m = offs_m < seq_length

            # Load Q tile: [BLOCK_M, BLOCK_D]
            q_ptrs = (
                q_base
                + offs_m[:, None] * stride_qt
                + offs_d[None, :] * stride_qd
            )
            q_tile = tl.load(
                q_ptrs, mask=mask_m[:, None] & mask_d[None, :], other=0.0
            ).to(tl.float32)

            # Running max and denominator for online softmax
            m_i = tl.full([BLOCK_M], value=-1e9, dtype=tl.float32)
            l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
            acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

            # --- Inner loop: key/value tiles (columns) ---
            for n_start in range(0, seq_length, BLOCK_N):
                offs_n = n_start + tl.arange(0, BLOCK_N)
                mask_n = offs_n < seq_length

                # Load K tile: [BLOCK_N, BLOCK_D]
                k_ptrs = (
                    k_base
                    + offs_n[:, None] * stride_kt
                    + offs_d[None, :] * stride_kd
                )
                k_tile = tl.load(
                    k_ptrs, mask=mask_n[:, None] & mask_d[None, :], other=0.0
                ).to(tl.float32)

                # QK^T: [BLOCK_M, BLOCK_N]
                qk = tl.dot(q_tile, tl.trans(k_tile)) * sm_scale

                # Causal mask: query absolute position >= key absolute position
                if is_causal:
                    causal_mask = offs_m[:, None] >= offs_n[None, :]
                    qk = tl.where(causal_mask, qk, -1e9)

                # Padding mask for key positions beyond seq_length
                qk = tl.where(mask_n[None, :], qk, -1e9)

                # Online softmax update
                m_new = tl.maximum(m_i, tl.max(qk, axis=1))
                alpha = tl.math.exp(m_i - m_new)
                p = tl.math.exp(qk - m_new[:, None])

                # Load V tile: [BLOCK_N, BLOCK_D]
                v_ptrs = (
                    v_base
                    + offs_n[:, None] * stride_vt
                    + offs_d[None, :] * stride_vd
                )
                v_tile = tl.load(
                    v_ptrs, mask=mask_n[:, None] & mask_d[None, :], other=0.0
                ).to(tl.float32)

                # Accumulate: rescale old acc, add new contribution
                acc = acc * alpha[:, None] + tl.dot(
                    p.to(v_tile.dtype), v_tile
                ).to(tl.float32)
                l_i = l_i * alpha + tl.sum(p, axis=1)
                m_i = m_new

            # Normalise
            acc = acc / l_i[:, None]

            # Store output tile: [BLOCK_M, BLOCK_D]
            o_ptrs = (
                o_base
                + offs_m[:, None] * stride_ot
                + offs_d[None, :] * stride_od
            )
            tl.store(o_ptrs, acc, mask=mask_m[:, None] & mask_d[None, :])


def context_attention(
    q, k, v, b_start_loc, b_seq_len, max_input_len, is_causal
):
    """Context-stage attention over packed variable-length sequences.

    Args:
        q, k, v:       [total_tokens, num_heads, head_dim] — packed input.
        b_start_loc:   [batch] int32 — start token index per sequence.
        b_seq_len:     [batch] int32 — sequence lengths.
        max_input_len: int — reserved; unused in this implementation.
        is_causal:     bool — if True, apply causal mask.

    Returns:
        out: [total_tokens, num_heads, head_dim] float32.
    """
    total_tokens, num_heads, head_dim = q.shape
    batch = b_seq_len.shape[0]

    out = torch.zeros(
        total_tokens, num_heads, head_dim, device=q.device, dtype=torch.float32
    )

    sm_scale = 1.0 / math.sqrt(head_dim)

    # Smallest power-of-two >= head_dim for constexpr tile
    block_d = 1
    while block_d < head_dim:
        block_d *= 2

    total_items = batch * num_heads
    grid = (min(total_items, GRID_CAP),)

    _context_attn_kernel[grid](
        q,
        k,
        v,
        out,
        b_start_loc,
        b_seq_len,
        num_heads,
        head_dim,
        sm_scale,
        1 if is_causal else 0,
        batch,
        # q strides
        q.stride(0),
        q.stride(1),
        q.stride(2),
        # k strides
        k.stride(0),
        k.stride(1),
        k.stride(2),
        # v strides
        v.stride(0),
        v.stride(1),
        v.stride(2),
        # out strides
        out.stride(0),
        out.stride(1),
        out.stride(2),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_D=block_d,
    )

    return out


__all__ = ["context_attention"]
