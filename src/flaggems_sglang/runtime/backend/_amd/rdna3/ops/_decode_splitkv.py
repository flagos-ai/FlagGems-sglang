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

"""Split-KV decode attention for RDNA3 (gfx1100), shared by decode_attention
and decode_grouped_attention.

One decode step reads the whole KV cache of a sequence to produce a single
output token a head, so it is bound by memory bandwidth, and the only way to
reach that bandwidth is to keep all 96 compute units busy. A program here owns
one block of query heads of one sequence and one slice of that sequence's KV
cache: it computes the attention of its slice and writes the partial result
together with its log-sum-exp, and a second pass combines the slices.

Both products run on the matrix units for 16-bit inputs; float32 has no
matrix instruction on gfx1100, so its products are asked for explicitly in
IEEE mode and run on the FMA units.

The exponentials are base 2 (``exp2``), which is a single instruction, with
the scale folded into the query scaling.

This kernel is written in portable Triton only. It must NOT call any
pre-compiled / vendor-specific cached operator (no
``F.scaled_dot_product_attention``, no ``_compiled`` handles, no
``torch.ops.*`` attention) — the whole computation is done inside the Triton
kernel.
"""

import torch
import triton
import triton.language as tl

_LOG2E = 1.4426950408889634
# Launch rules measured on the W7900D (96 CUs): a sequence is sliced only while
# the grid is too small to fill the device, never into slices shorter than
# _MIN_SPLIT_TOKENS, never into more than _MAX_SPLITS, and only up to
# _SPLIT_PROGRAMS programs, past which the slices cost more than the compute
# units they fill.
_MIN_PROGRAMS = 96
_MAX_SPLITS = 64
_MIN_SPLIT_TOKENS = 128
_SPLIT_PROGRAMS = 192
# Below this many programs the launch is short enough that prefetching the next
# K/V tile wins over the registers it costs.
_SMALL_LAUNCH = 64
# A K/V tile of 32 tokens: 64 is never more than half a percent faster here and
# spills to scratch at four warps.
_BLOCK_N = 32


@triton.jit
def _decode_split_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    kv_indptr_ptr,
    kv_indices_ptr,
    out_ptr,
    mid_o_ptr,
    mid_lse_ptr,
    qk_scale,
    H_Q: tl.constexpr,
    H_KV: tl.constexpr,
    D: tl.constexpr,
    D_V: tl.constexpr,
    GROUP: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    IEEE: tl.constexpr,
):
    hb = tl.program_id(0)
    split = tl.program_id(1)
    b = tl.program_id(2)
    # the KV head is the slowest axis of the first dimension, so neighbouring
    # programs share a K/V slice
    num_hblocks: tl.constexpr = (GROUP + BLOCK_H - 1) // BLOCK_H
    kv_head = hb // num_hblocks
    h_local = (hb % num_hblocks) * BLOCK_H + tl.arange(0, BLOCK_H)
    h_mask = h_local < GROUP
    heads = kv_head * GROUP + h_local

    start = tl.load(kv_indptr_ptr + b)
    seq_len = tl.load(kv_indptr_ptr + b + 1) - start
    split_len = tl.cdiv(tl.cdiv(seq_len, NUM_SPLITS), BLOCK_N) * BLOCK_N
    s_start = split * split_len
    s_end = tl.minimum(seq_len, s_start + split_len)

    offs_d = tl.arange(0, BLOCK_D)
    offs_dv = tl.arange(0, BLOCK_DV)
    d_mask = offs_d < D
    dv_mask = offs_dv < D_V
    q = tl.load(
        q_ptr + (b * H_Q + heads[:, None]) * D + offs_d[None, :],
        mask=h_mask[:, None] & d_mask[None, :],
        other=0.0,
    )
    m_i = tl.full([BLOCK_H], -float("inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_H], dtype=tl.float32)
    acc = tl.zeros([BLOCK_H, BLOCK_DV], dtype=tl.float32)
    for n0 in range(s_start, s_end, BLOCK_N):
        offs_n = n0 + tl.arange(0, BLOCK_N)
        n_mask = offs_n < s_end
        tok = tl.load(kv_indices_ptr + start + offs_n, mask=n_mask, other=0)
        k = tl.load(
            k_ptr + (tok[:, None] * H_KV + kv_head) * D + offs_d[None, :],
            mask=n_mask[:, None] & d_mask[None, :],
            other=0.0,
        )
        if IEEE:
            qk = tl.dot(q, tl.trans(k), input_precision="ieee") * qk_scale
        else:
            qk = tl.dot(q, tl.trans(k)) * qk_scale
        qk = tl.where(n_mask[None, :], qk, -1.0e30)
        m_new = tl.maximum(m_i, tl.max(qk, axis=1))
        p = tl.math.exp2(qk - m_new[:, None])
        alpha = tl.math.exp2(m_i - m_new)
        l_i = l_i * alpha + tl.sum(p, axis=1)
        v = tl.load(
            v_ptr + (tok[:, None] * H_KV + kv_head) * D_V + offs_dv[None, :],
            mask=n_mask[:, None] & dv_mask[None, :],
            other=0.0,
        )
        if IEEE:
            acc = acc * alpha[:, None] + tl.dot(p, v, input_precision="ieee")
        else:
            acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m_i = m_new

    # a slice that got no tokens leaves l_i at zero and is dropped by the
    # second pass through its -inf log-sum-exp
    o = acc / tl.where(l_i > 0, l_i, 1.0)[:, None]
    if NUM_SPLITS == 1:
        tl.store(
            out_ptr + (b * H_Q + heads[:, None]) * D_V + offs_dv[None, :],
            o,
            mask=h_mask[:, None] & dv_mask[None, :],
        )
    else:
        row = (b * H_Q + heads) * NUM_SPLITS + split
        tl.store(
            mid_o_ptr + row[:, None] * D_V + offs_dv[None, :],
            o,
            mask=h_mask[:, None] & dv_mask[None, :],
        )
        lse = tl.where(l_i > 0, m_i + tl.math.log2(l_i), -float("inf"))
        tl.store(mid_lse_ptr + row, lse, mask=h_mask)


@triton.jit
def _decode_merge_kernel(
    mid_o_ptr,
    mid_lse_ptr,
    out_ptr,
    D_V: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    BLOCK_DV: tl.constexpr,
):
    row = tl.program_id(0)
    offs_s = tl.arange(0, NUM_SPLITS)
    offs_dv = tl.arange(0, BLOCK_DV)
    dv_mask = offs_dv < D_V
    lse = tl.load(mid_lse_ptr + row * NUM_SPLITS + offs_s)
    m = tl.max(lse, axis=0)
    w = tl.where(lse > -float("inf"), tl.math.exp2(lse - m), 0.0)
    o = tl.load(
        mid_o_ptr
        + (row * NUM_SPLITS + offs_s[:, None]) * D_V
        + offs_dv[None, :],
        mask=dv_mask[None, :],
        other=0.0,
    )
    tl.store(
        out_ptr + row * D_V + offs_dv,
        tl.sum(o * w[:, None], axis=0) / tl.sum(w, axis=0),
        mask=dv_mask,
    )


def supported(q, k_buffer, v_buffer, group):
    """Head sizes up to 256 (512-wide tiles do not fit in LDS) and contiguous
    buffers, in 16-bit; float32 products run on the FMA units, where this only
    beats a kernel without tiles once a block of query heads fills the tile."""
    return (
        q.dtype in (torch.bfloat16, torch.float16, torch.float32)
        and k_buffer.dtype == q.dtype
        and v_buffer.dtype == q.dtype
        and (group >= 16 or q.dtype != torch.float32)
        and q.shape[-1] <= 256
        and v_buffer.shape[-1] <= 256
        and q.is_contiguous()
        and k_buffer.is_contiguous()
        and v_buffer.is_contiguous()
    )


def _splits(programs, avg_len):
    if programs >= _MIN_PROGRAMS:
        return 1
    count = min(
        _MAX_SPLITS,
        _SPLIT_PROGRAMS // programs,
        avg_len // _MIN_SPLIT_TOKENS,
    )
    return 1 << (count.bit_length() - 1) if count >= 1 else 1


def _config(batch, kv_heads, group, avg_len, ieee):
    """Head block, number of slices and pipeline stages.

    All the query heads of a block share the K/V tiles their program reads, so
    the fewer blocks a group is cut into the less of the KV cache is read
    twice; the block is only made smaller when the grid would otherwise leave
    compute units idle, and slicing the sequence is the other way of filling
    them. On the FMA units a wider block buys no reuse, only masked rows.
    """
    block_h = 16 if ieee else max(16, min(64, triton.next_power_of_2(group)))
    while True:
        programs = max(1, batch * kv_heads * triton.cdiv(group, block_h))
        splits = _splits(programs, avg_len)
        if block_h == 16 or programs * splits >= _MIN_PROGRAMS:
            break
        block_h //= 2
    stages = 2 if programs * splits <= _SMALL_LAUNCH else 1
    return block_h, splits, stages


def decode_splitkv(q, k_buffer, v_buffer, kv_indptr, kv_indices, sm_scale):
    batch, h_q, d = q.shape
    h_kv = k_buffer.shape[1]
    d_v = v_buffer.shape[-1]
    group = h_q // h_kv
    # the average sequence length, read from the shapes so that no sequence
    # length has to be copied back from the device
    avg_len = kv_indices.numel() // max(batch, 1)
    ieee = q.dtype == torch.float32
    block_h, splits, stages = _config(batch, h_kv, group, avg_len, ieee)
    num_hblocks = triton.cdiv(group, block_h)
    out = torch.empty((batch, h_q, d_v), dtype=torch.float32, device=q.device)
    if splits > 1:
        mid_o = torch.empty(
            (batch * h_q * splits, d_v), dtype=torch.float32, device=q.device
        )
        mid_lse = torch.empty(
            (batch * h_q * splits,), dtype=torch.float32, device=q.device
        )
    else:
        mid_o = mid_lse = out
    block_dv = triton.next_power_of_2(d_v)
    _decode_split_kernel[(h_kv * num_hblocks, splits, batch)](
        q,
        k_buffer,
        v_buffer,
        kv_indptr,
        kv_indices,
        out,
        mid_o,
        mid_lse,
        sm_scale * _LOG2E,
        H_Q=h_q,
        H_KV=h_kv,
        D=d,
        D_V=d_v,
        GROUP=group,
        BLOCK_H=block_h,
        BLOCK_N=_BLOCK_N,
        BLOCK_D=triton.next_power_of_2(d),
        BLOCK_DV=block_dv,
        NUM_SPLITS=splits,
        IEEE=ieee,
        num_warps=4,
        num_stages=stages,
    )
    if splits > 1:
        _decode_merge_kernel[(batch * h_q,)](
            mid_o,
            mid_lse,
            out,
            D_V=d_v,
            NUM_SPLITS=splits,
            BLOCK_DV=block_dv,
            num_warps=4,
        )
    return out
