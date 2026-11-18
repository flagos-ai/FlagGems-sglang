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
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""context_attention — Prefill/context-stage scaled dot-product attention.

Computes multi-head attention for packed variable-length sequences (causal or
non-causal). Inputs are in *packed* format: all sequences are concatenated
along the token dimension, and b_start_loc / b_seq_len mark each sequence's
slice.

Semantics (matches the PyTorch reference exactly):

    q, k, v: [total_tokens, num_heads, head_dim]
    b_start_loc: [batch]  — start token index for each sequence
    b_seq_len:   [batch]  — number of tokens in each sequence
    max_input_len: int    — max sequence length (upper bound of every
                            b_seq_len; also sizes the m-split decomposition)
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

Optimization strategy (GCU400/L600, 2026-11-08 sweep, 8-round A/B):
    Flash-attention style online-softmax kernel with:
    - Causal early stop: the K/V scan of a query tile stops at the tile's
      last query row (halves the tile work under is_causal).
    - Two work layouts picked by host metadata (batch*num_heads vs the
      query-tile count): a (seq, head) persistent layout for large batches
      and an (seq, head, query-tile) split layout for small ones where
      batch*num_heads cannot fill the machine.
    - V loaded together with K *before* the QK dot (VPRE): +6-9% — the
      QK dot covers the V DMA latency instead of stalling behind it.
    - Causal/padding masks applied ARITHMETICALLY
      (qk*ok + (ok-1)*1e9) instead of tl.where selects: the select form
      between the dot and the softmax reduction costs ~40% of runtime on
      this backend (suspected select-vs-MMA-epilogue fusion break); the
      arithmetic form is bit-compatible for every masked lane (-1e9) and
      every valid lane (qk unchanged, multiply by 1.0).
    - IS_CAUSAL / EXACT_D as constexpr specialisations (no runtime
      branches inside the tile loop; head_dim == BLOCK_D drops the d-axis
      masks).
    - One canonical scf.for K/V tile loop per query tile. Measured on this
      backend: any second tile loop, peeled straight-line tail tile, or a
      runtime guard wrapping the loop regresses 25-60% (pipelining) — do
      NOT restructure without re-benching.
    - BLOCK_M/BLOCK_N = 64/128 fp32 tiles, num_warps=2 (re-swept on the
      current toolchain: 32/64/128 x 64/128/256 x nw1/2/4 all lose),
      grid capped per layout. No autotune.
    - torch.empty for the output: every valid token row is written by the
      kernel (query tiles cover [0, seq_length) of every sequence), so no
      memset launch is needed. max_input_len must be an upper bound of the
      per-sequence lengths for the m-split layout to cover all rows.
"""

import math

import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------

BLOCK_M = 64  # query tile size  (rows of the attention matrix per program)
BLOCK_N = 128  # key/value tile size (columns scanned per inner loop)
GRID_CAP_ITEM = 48  # (seq, head) layout: grid near the physical slot count
GRID_CAP_MSPLIT = 512  # m-split layout: large grid spreads the many works
NUM_WARPS = 2  # GCU400 dot-tile family: nw2 < nw1 < nw4 (re-swept 2026-11-08)
SPLIT_ITEMS = 64  # m-split layout when batch*num_heads <= this ...
SPLIT_MIN_TILES = 16  # ... and the per-pair query-tile chain is at least this
# long (long serial chains are what the split amortises; short sequences
# pay the ~0.4us/work metadata for nothing). Both sides of each line are
# anchored by measurements: (items, m_tiles) = (32,32)/(64,32)/(32,64)
# MSPLIT wins 20-41%; (32,2)/(256,2)/(256,32)/(2048,*)/(32,8)/(64,8) ITEM
# wins 3-25%. Re-anchored on the current toolchain: MSPLIT +17% @ (32,32).


@triton.jit
def _attn_m_tile(
    q_ptr,
    k_ptr,
    v_ptr,
    out_ptr,
    seq_start,
    seq_length,
    m_start,
    stride_qt,
    stride_qd,
    stride_kt,
    stride_kd,
    stride_vt,
    stride_vd,
    stride_ot,
    stride_od,
    sm_scale,
    is_causal,
    offs_d,
    mask_d,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    EXACT_D: tl.constexpr,
):
    """Process one query tile [BLOCK_M rows] of one (seq, head) pair."""
    # base pointers for this (seq, head); the head offset is already folded
    # into the pointers passed by the caller.
    q_base = q_ptr + seq_start * stride_qt
    k_base = k_ptr + seq_start * stride_kt
    v_base = v_ptr + seq_start * stride_vt
    o_base = out_ptr + seq_start * stride_ot

    offs_m = m_start + tl.arange(0, BLOCK_M)
    mask_m = offs_m < seq_length

    # Load Q tile: [BLOCK_M, BLOCK_D]
    q_ptrs = q_base + offs_m[:, None] * stride_qt + offs_d[None, :] * stride_qd
    if EXACT_D:
        q_tile = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0).to(
            tl.float32
        )
    else:
        q_tile = tl.load(
            q_ptrs, mask=mask_m[:, None] & mask_d[None, :], other=0.0
        ).to(tl.float32)

    # Running max and denominator for online softmax
    m_i = tl.full([BLOCK_M], value=-1e9, dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

    # Causal early stop: query rows in this tile only attend to keys at
    # absolute position <= m_start + BLOCK_M - 1. Out-of-range works
    # (m_start >= seq_length, possible when the max_input_len bound
    # over-estimates some sequence) clamp the trip count to zero via a
    # scalar select. Skipping the fully-masked tiles is numerically
    # exact: their scores were clamped to -1e9 and exp(-1e9 - m_new)
    # underflows to +0.
    if IS_CAUSAL:
        kv_hi = tl.minimum(m_start + BLOCK_M, seq_length)
    else:
        kv_hi = seq_length
    kv_hi = tl.where(m_start < seq_length, kv_hi, 0)

    # --- Inner loop: key/value tiles (columns) ---
    for n_start in range(0, kv_hi, BLOCK_N):
        offs_n = n_start + tl.arange(0, BLOCK_N)
        mask_n = offs_n < seq_length

        # Load K and V tiles together, before the QK dot: the dot then
        # covers both DMA latencies (measured +6-9% vs loading V after
        # the softmax on this backend).
        k_ptrs = (
            k_base + offs_n[:, None] * stride_kt + offs_d[None, :] * stride_kd
        )
        v_ptrs = (
            v_base + offs_n[:, None] * stride_vt + offs_d[None, :] * stride_vd
        )
        if EXACT_D:
            k_tile = tl.load(k_ptrs, mask=mask_n[:, None], other=0.0).to(
                tl.float32
            )
            v_tile = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0).to(
                tl.float32
            )
        else:
            k_tile = tl.load(
                k_ptrs, mask=mask_n[:, None] & mask_d[None, :], other=0.0
            ).to(tl.float32)
            v_tile = tl.load(
                v_ptrs, mask=mask_n[:, None] & mask_d[None, :], other=0.0
            ).to(tl.float32)

        # QK^T: [BLOCK_M, BLOCK_N]
        qk = tl.dot(q_tile, tl.trans(k_tile)) * sm_scale

        # Combined causal + padding mask, applied arithmetically. For a
        # valid lane (ok=1) the score is unchanged (x*1 + 0); for a masked
        # lane (ok=0) it becomes exactly -1e9, same value the tl.where
        # form produced. The select form here costs ~40% of the kernel
        # runtime on this backend (see module docstring).
        if IS_CAUSAL:
            ok = (offs_m[:, None] >= offs_n[None, :]) & mask_n[None, :]
        else:
            ok = mask_n[None, :] & tl.full([1, BLOCK_N], 1, tl.int1)
        okf = ok.to(tl.float32)
        qk = qk * okf + (okf - 1.0) * 1e9

        # Online softmax update
        m_new = tl.maximum(m_i, tl.max(qk, axis=1))
        alpha = tl.math.exp(m_i - m_new)
        p = tl.math.exp(qk - m_new[:, None])

        # Accumulate: rescale old acc, add new contribution. The rescale
        # feeds the dot accumulator so the add fuses into the MMA epilogue
        # (serial-recurrence acc-fusion pattern).
        acc = tl.dot(p.to(v_tile.dtype), v_tile, acc=acc * alpha[:, None]).to(
            tl.float32
        )
        l_i = l_i * alpha + tl.sum(p, axis=1)
        m_i = m_new

    # Normalise
    acc = acc / l_i[:, None]

    # Store output tile: [BLOCK_M, BLOCK_D]
    o_ptrs = o_base + offs_m[:, None] * stride_ot + offs_d[None, :] * stride_od
    if EXACT_D:
        tl.store(o_ptrs, acc, mask=mask_m[:, None])
    else:
        tl.store(o_ptrs, acc, mask=mask_m[:, None] & mask_d[None, :])


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
    m_tiles,
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
    M_SPLIT: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    EXACT_D: tl.constexpr,
):
    """Persistent kernel with two work layouts:

    - M_SPLIT=False: one work per (seq, head); the program iterates that
      pair's query tiles serially. Minimal per-work metadata cost.
    - M_SPLIT=True: one work per (seq, head, query tile) so parallelism
      also scales with the number of query tiles — best when
      batch*num_heads is too small to fill the machine.
    """
    pid = tl.program_id(0)

    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < head_dim

    if M_SPLIT:
        total_work = batch * num_heads * m_tiles
        for work in range(pid, total_work, tl.num_programs(0)):
            m_idx = work % m_tiles
            item = work // m_tiles
            seq_id = item // num_heads
            head_id = item % num_heads

            seq_start = tl.load(b_start_loc_ptr + seq_id).to(tl.int32)
            seq_length = tl.load(b_seq_len_ptr + seq_id).to(tl.int32)

            _attn_m_tile(
                q_ptr + head_id * stride_qh,
                k_ptr + head_id * stride_kh,
                v_ptr + head_id * stride_vh,
                out_ptr + head_id * stride_oh,
                seq_start,
                seq_length,
                m_idx * BLOCK_M,
                stride_qt,
                stride_qd,
                stride_kt,
                stride_kd,
                stride_vt,
                stride_vd,
                stride_ot,
                stride_od,
                sm_scale,
                is_causal,
                offs_d,
                mask_d,
                BLOCK_M,
                BLOCK_N,
                BLOCK_D,
                IS_CAUSAL,
                EXACT_D,
            )
    else:
        total_items = batch * num_heads
        for item in range(pid, total_items, tl.num_programs(0)):
            seq_id = item // num_heads
            head_id = item % num_heads

            seq_start = tl.load(b_start_loc_ptr + seq_id).to(tl.int32)
            seq_length = tl.load(b_seq_len_ptr + seq_id).to(tl.int32)

            for m_start in range(0, seq_length, BLOCK_M):
                _attn_m_tile(
                    q_ptr + head_id * stride_qh,
                    k_ptr + head_id * stride_kh,
                    v_ptr + head_id * stride_vh,
                    out_ptr + head_id * stride_oh,
                    seq_start,
                    seq_length,
                    m_start,
                    stride_qt,
                    stride_qd,
                    stride_kt,
                    stride_kd,
                    stride_vt,
                    stride_vd,
                    stride_ot,
                    stride_od,
                    sm_scale,
                    is_causal,
                    offs_d,
                    mask_d,
                    BLOCK_M,
                    BLOCK_N,
                    BLOCK_D,
                    IS_CAUSAL,
                    EXACT_D,
                )


def context_attention(
    q, k, v, b_start_loc, b_seq_len, max_input_len, is_causal
):
    """Context-stage attention over packed variable-length sequences.

    Args:
        q, k, v:       [total_tokens, num_heads, head_dim] — packed input.
        b_start_loc:   [batch] int32 — start token index per sequence.
        b_seq_len:     [batch] int32 — sequence lengths.
        max_input_len: int — upper bound of the sequence lengths; used to
                       size the m-split work decomposition (clamped to
                       total_tokens, so over-estimates only cost a skipped
                       empty work each). Must be >= max(b_seq_len) for the
                       m-split layout to cover every query row.
        is_causal:     bool — if True, apply causal mask.

    Returns:
        out: [total_tokens, num_heads, head_dim] float32.
    """
    total_tokens, num_heads, head_dim = q.shape
    batch = b_seq_len.shape[0]

    # Every valid token row is written by the kernel (query tiles cover
    # [0, seq_length) of every sequence), so empty beats zeros: no memset
    # launch, single-kernel contract preserved.
    out = torch.empty(
        total_tokens, num_heads, head_dim, device=q.device, dtype=torch.float32
    )

    sm_scale = 1.0 / math.sqrt(head_dim)

    # Smallest power-of-two >= head_dim for constexpr tile
    block_d = 1
    while block_d < head_dim:
        block_d *= 2

    # When head_dim is exactly the pow2 tile the d-axis masks are
    # all-true: drop them (specialised compile, measured ~1.5%).
    exact_d = block_d == head_dim

    total_items = batch * num_heads
    # Work item granularity choice (host metadata dispatch, zero D2H):
    # - M_SPLIT=True:  work = (seq, head, query tile). Parallelism also
    #   scales with the number of query tiles; pays ~0.4us metadata per
    #   extra work. Wins when batch*num_heads cannot fill the machine.
    # - M_SPLIT=False: work = (seq, head); query tiles of a pair stay
    #   serial in-program. Minimal metadata; best for large batches.
    if max_input_len is None:
        max_input_len = total_tokens
    max_len_bound = (
        min(int(max_input_len), total_tokens) if total_tokens > 0 else 0
    )
    m_tiles = max(1, (max_len_bound + BLOCK_M - 1) // BLOCK_M)
    m_split = total_items <= SPLIT_ITEMS and m_tiles >= SPLIT_MIN_TILES
    if m_split:
        # The m-split layout spreads many small works, so a large grid
        # wins (512 CTAs beat 48 by ~12% on the bs<=2/long-seq shapes).
        grid = (min(total_items * m_tiles, GRID_CAP_MSPLIT),)
    else:
        # The (seq, head) layout prefers a grid near the physical slots;
        # a 512-CTA cap is up to ~5% slower here.
        grid = (min(total_items, GRID_CAP_ITEM),)

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
        m_tiles,
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
        M_SPLIT=m_split,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_D=block_d,
        IS_CAUSAL=bool(is_causal),
        EXACT_D=bool(exact_d),
        num_warps=NUM_WARPS,
    )

    return out


__all__ = ["context_attention"]
