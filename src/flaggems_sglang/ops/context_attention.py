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

"""context_attention — Prefill/context-stage scaled dot-product attention on
packed variable-length sequences (Triton).

Computes, per sequence i (delimited by ``b_start_loc`` / ``b_seq_len``):

    o[start:end] = SDPA(q[start:end], k[start:end], v[start:end], is_causal)

where the packed tensors are laid out as ``[total_tokens, num_heads, head_dim]``.
The output is ``[total_tokens, num_heads, head_dim]`` in float32, matching the
PyTorch reference exactly.

This kernel is written in portable Triton only. It must NOT call any
pre-compiled / vendor-specific cached operator (no ``F.scaled_dot_product_attention``,
no ``_compiled`` handles, no ``torch.ops.*`` attention) — the whole computation is
done inside the Triton kernel so it is portable across supported chips.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _context_attention_fwd_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    o_ptr,
    b_start_loc_ptr,
    b_seq_len_ptr,
    scale,
    num_heads,
    head_dim: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
):
    """Flash-attention style kernel for packed prefill.

    One program per (sequence, head, query-block).  The grid is::

        grid = (num_seq, num_heads, cdiv(max_input_len, BLOCK_M))

    The kernel computes attention independently for each sequence using its
    own ``start`` / ``seq_len`` boundaries, so variable-length packed sequences
    are handled without padding.

    ``BLOCK_D`` is the next power of two >= ``head_dim``; the head dimension is
    loaded with a mask so that non-power-of-two head dims (e.g. 72/80/96) are
    supported.  Padded head columns are read as 0, so they contribute nothing to
    the score (q=0) nor to the output accumulator (v=0).
    """
    seq_id = tl.program_id(0)
    head_id = tl.program_id(1)
    m_block = tl.program_id(2)

    start = tl.load(b_start_loc_ptr + seq_id)
    seq_len = tl.load(b_seq_len_ptr + seq_id)

    # Offset to the start of this sequence in the packed [total_tokens, H, D] layout.
    # Strides: token stride = num_heads * head_dim, head stride = head_dim.
    off_hz = head_id * head_dim
    qkv_base = start * num_heads * head_dim + off_hz

    # Query block [BLOCK_M, BLOCK_D].  BLOCK_D is a power of two; we mask the
    # head-dim axis so only the first `head_dim` columns are read (rest = 0).
    offs_m = m_block * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)
    d_mask = offs_d < head_dim

    q_ptrs = (
        q_ptr
        + qkv_base
        + offs_m[:, None] * num_heads * head_dim
        + offs_d[None, :]
    )
    m_mask = offs_m < seq_len
    q = tl.load(q_ptrs, mask=m_mask[:, None] & d_mask[None, :], other=0.0).to(
        tl.float32
    )

    # Running statistics for the online softmax.
    m_i = tl.full([BLOCK_M], -float("inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

    if IS_CAUSAL:
        # Causal: a query at row r can attend to keys at rows [0, r].  All keys
        # are needed up to the largest query index in this block, so start at 0.
        # We can stop the N loop once we pass the largest query row (keys beyond
        # it are entirely masked), so n_end = min(seq_len, last query row + 1).
        n_start = 0
        n_end = min(seq_len, (m_block + 1) * BLOCK_M)
    else:
        n_start = 0
        n_end = seq_len

    # Iterate over key/value blocks.
    n_off_base = tl.arange(0, BLOCK_N)
    for n_start_block in range(n_start, n_end, BLOCK_N):
        n_off = n_start_block + n_off_base
        n_mask = n_off < seq_len

        k_ptrs = (
            k_ptr
            + qkv_base
            + n_off[None, :] * num_heads * head_dim
            + offs_d[:, None]
        )
        v_ptrs = (
            v_ptr
            + qkv_base
            + n_off[:, None] * num_heads * head_dim
            + offs_d[None, :]
        )

        k = tl.load(
            k_ptrs, mask=d_mask[:, None] & n_mask[None, :], other=0.0
        ).to(tl.float32)
        v = tl.load(
            v_ptrs, mask=n_mask[:, None] & d_mask[None, :], other=0.0
        ).to(tl.float32)

        # Scores: [BLOCK_M, BLOCK_N]
        qk = tl.dot(q, k) * scale

        if IS_CAUSAL:
            # Mask out keys strictly after each query row, and padding keys.
            attn_mask = (offs_m[:, None] >= n_off[None, :]) & n_mask[None, :]
        else:
            # Only mask out padding keys (beyond seq_len).
            attn_mask = n_mask[None, :]
        qk = tl.where(attn_mask, qk, -1.0e30)

        # Online softmax update.
        m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
        p = tl.exp(qk - m_ij[:, None])
        l_ij = tl.sum(p, axis=1)

        alpha = tl.exp(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m_i = m_ij

    # Normalize and store.  Only write the real `head_dim` columns.
    out = acc / l_i[:, None]
    o_ptrs = (
        o_ptr
        + qkv_base
        + offs_m[:, None] * num_heads * head_dim
        + offs_d[None, :]
    )
    tl.store(
        o_ptrs, out.to(tl.float32), mask=m_mask[:, None] & d_mask[None, :]
    )


def context_attention(
    q, k, v, b_start_loc, b_seq_len, max_input_len, is_causal
):
    """Prefill/context-stage scaled dot-product attention on packed variable-length
    sequences.

    Args:
        q, k, v: packed tensors of shape ``[total_tokens, num_heads, head_dim]``
            (float32 / bfloat16 / float16).  ``total_tokens`` is the sum of the
            per-sequence lengths given by ``b_seq_len``.
        b_start_loc: ``[B]`` int tensor, start offset (in tokens) of each sequence.
        b_seq_len:   ``[B]`` int tensor, length (in tokens) of each sequence.
        max_input_len: reserved for kernel scratch allocation (unused in the
            reference).  Used here only to size the M grid.
        is_causal: whether to apply a causal attention mask per sequence.

    Returns:
        ``[total_tokens, num_heads, head_dim]`` float32 tensor.
    """
    total_tokens, num_heads, head_dim = q.shape
    scale = 1.0 / (head_dim**0.5)

    B = b_seq_len.shape[0]
    # Ensure int32 offsets for the kernel.
    if b_start_loc.dtype != torch.int32:
        b_start_loc = b_start_loc.to(torch.int32)
    if b_seq_len.dtype != torch.int32:
        b_seq_len = b_seq_len.to(torch.int32)

    out = torch.empty(
        (total_tokens, num_heads, head_dim),
        device=q.device,
        dtype=torch.float32,
    )

    # BLOCK_D must be a power of two (Triton's arange / dot require it).  Use the
    # next power of two >= head_dim; the kernel masks the head-dim axis so the
    # padded columns do not affect the result.
    block_d = 1 << (head_dim - 1).bit_length()

    # Tile sizes.  head_dim is the contraction axis; BLOCK_M/BLOCK_N tile the
    # sequence axis.  MetaX C550 has 64KB of shared memory per SM, so the tile
    # sizes (and the software-pipelining factor) must be chosen so that
    #   smem ≈ (q + num_stages * (k + v)) * 4 bytes
    # stays within budget.  These conservative defaults fit both head_dim=64
    # and head_dim=128; the optimization loop can tune them later.
    if block_d <= 64:
        BLOCK_M = 64
        BLOCK_N = 64
        num_stages = 1
    else:
        BLOCK_M = 32
        BLOCK_N = 32
        num_stages = 1

    # max_input_len drives the M grid.  Guard against a caller passing 0 or a
    # value smaller than the actual longest sequence (fall back to the real max
    # so no query row is dropped).
    actual_max = int(b_seq_len.max().item())
    grid_m_max = max(max_input_len, actual_max)
    grid_m = triton.cdiv(grid_m_max, BLOCK_M)

    grid = (B, num_heads, grid_m)
    _context_attention_fwd_kernel[grid](
        q,
        k,
        v,
        out,
        b_start_loc,
        b_seq_len,
        scale,
        num_heads,
        head_dim=head_dim,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_D=block_d,
        IS_CAUSAL=is_causal,
        num_warps=4,
        num_stages=num_stages,
    )
    return out


__all__ = ["context_attention"]
