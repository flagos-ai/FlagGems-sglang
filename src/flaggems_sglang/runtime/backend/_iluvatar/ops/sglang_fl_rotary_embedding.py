# Copyright 2026, The FlagOS Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License")
# you may not use this file except in compliance with the License
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""SGLang fused in-place RoPE as a single Triton kernel.

run(positions, query, key, head_size, cos_sin_cache, is_neox=True) -> None

Semantics (bound Definition):
  * rd = cos_sin_cache.shape[-1]; cos is the first M = rd/2 columns of each cache
    row, sin the second M.  Lanes [rd, head_size) of a head are pass-through and
    are simply never written.
  * neox  : out[j]   = x[j]*c[j]   - x[j+M]*s[j]
            out[j+M] = x[j+M]*c[j] + x[j]*s[j]
    inter : out[2k]   = x[2k]*c[k]   - x[2k+1]*s[k]
            out[2k+1] = x[2k+1]*c[k] + x[2k]*s[k]
  * arithmetic in fp32 with a single round-to-nearest-even cast at the store,
    matching RotaryEmbedding.forward_native; positions are gathered per token and
    clamped into the cache; query/key are mutated in place in their own dtype.

Launch shape: 2D grid (tokens, query-chunks + key-chunks) so a chunk never
straddles the two buffers; each program covers BH = 8 heads of one token and the
M rotary lanes of each head, loading the two lane groups that the rotation pairs
(j, j+M for neox; 2k, 2k+1 for interleaved).  The four payload strides and the
rotary half width are tl.constexpr, which keeps the per-lane accesses wide, and
only the six genuinely per-call scalars remain runtime arguments.

A compile-time DIVISIBLE specialization drops the head and the lane predicates
for shapes where they are provably all-true (BH divides both head counts and
M == BLOCK_M) and marks the payload stores as streaming; the fully masked body
stays the default for partial chunks and odd head counts.  The base pointer is
always selected with tl.where, never with a scalar branch: a branch measurably
cost an extra ~14% on the largest timed shape.
"""

import torch
import triton
import triton.language as tl

_DT_CODE = {
    torch.bfloat16: 0,
    torch.float16: 1,
    torch.float32: 2,
}


@triton.jit
def _rope_inplace_kernel(
    q_ptr,
    k_ptr,
    pos_ptr,
    cs_ptr,
    nq,
    nkv,
    max_pos,
    pos_stride,
    cs_stride,
    CQ,
    QR: tl.constexpr,
    QH: tl.constexpr,
    KR: tl.constexpr,
    KH: tl.constexpr,
    BH: tl.constexpr,
    M: tl.constexpr,
    BLOCK_M: tl.constexpr,
    IS_NEOX: tl.constexpr,
    ODT: tl.constexpr,
    FULL: tl.constexpr,
):
    t = tl.program_id(0)
    c = tl.program_id(1)

    off_m = tl.arange(0, BLOCK_M)
    t64 = t.to(tl.int64)

    p = tl.load(pos_ptr + t64 * pos_stride).to(tl.int64)
    p = tl.minimum(tl.maximum(p, 0), max_pos - 1)
    cs_row = cs_ptr + p * cs_stride

    is_q = c < CQ
    off_h = tl.arange(0, BH)
    h_idx = tl.where(is_q, c * BH + off_h, (c - CQ) * BH + off_h)
    nh = tl.where(is_q, nq, nkv)
    q_base = q_ptr + t64 * QR + h_idx.to(tl.int64) * QH
    k_base = k_ptr + t64 * KR + h_idx.to(tl.int64) * KH
    base = tl.where(is_q, q_base, k_base)

    if IS_NEOX:
        la = off_m
        lb = off_m + M
    else:
        la = off_m * 2
        lb = off_m * 2 + 1

    off = base[:, None] + la[None, :]
    off2 = base[:, None] + lb[None, :]

    if FULL:
        cval = tl.load(cs_row + off_m).to(tl.float32)
        sval = tl.load(cs_row + M + off_m).to(tl.float32)
        xa = tl.load(off).to(tl.float32)
        xb = tl.load(off2).to(tl.float32)
        o1 = xa * cval[None, :] - xb * sval[None, :]
        o2 = xb * cval[None, :] + xa * sval[None, :]
        if ODT == 0:
            tl.store(off, o1.to(tl.bfloat16), cache_modifier=".cs")
            tl.store(off2, o2.to(tl.bfloat16), cache_modifier=".cs")
        elif ODT == 1:
            tl.store(off, o1.to(tl.float16), cache_modifier=".cs")
            tl.store(off2, o2.to(tl.float16), cache_modifier=".cs")
        else:
            tl.store(off, o1, cache_modifier=".cs")
            tl.store(off2, o2, cache_modifier=".cs")
    else:
        m_mask = off_m < M
        mask = (h_idx < nh)[:, None] & m_mask[None, :]
        cval = tl.load(cs_row + off_m, mask=m_mask, other=0.0).to(tl.float32)
        sval = tl.load(cs_row + M + off_m, mask=m_mask, other=0.0).to(
            tl.float32
        )
        xa = tl.load(off, mask=mask, other=0.0).to(tl.float32)
        xb = tl.load(off2, mask=mask, other=0.0).to(tl.float32)
        o1 = xa * cval[None, :] - xb * sval[None, :]
        o2 = xb * cval[None, :] + xa * sval[None, :]
        if ODT == 0:
            tl.store(off, o1.to(tl.bfloat16), mask=mask)
            tl.store(off2, o2.to(tl.bfloat16), mask=mask)
        elif ODT == 1:
            tl.store(off, o1.to(tl.float16), mask=mask)
            tl.store(off2, o2.to(tl.float16), mask=mask)
        else:
            tl.store(off, o1, mask=mask)
            tl.store(off2, o2, mask=mask)


_BH = 8
_NUM_WARPS = 2


def _row_and_head_stride(t, head_size):
    """Row stride and per-head stride of a [NT, nheads*head_size] style tensor."""
    row = int(t.stride(0))
    if t.dim() <= 2:
        head = head_size * int(t.stride(-1))
    else:
        head = int(t.stride(-2))
    return row, head


def sglang_fl_rotary_embedding(
    positions, query, key, head_size, cos_sin_cache, is_neox=True
):
    head_size = int(head_size)
    nt = int(query.shape[0])
    if nt == 0:
        return query, key

    rd = int(cos_sin_cache.shape[-1])
    m = rd // 2
    nq = int(query.numel()) // (nt * head_size)
    nkv = int(key.numel()) // (nt * head_size)
    if m <= 0 or nq <= 0 or nkv <= 0:
        return query, key

    odt = _DT_CODE.get(query.dtype)
    if odt is None:
        return query, key

    max_pos = int(cos_sin_cache.shape[0])
    if max_pos <= 0:
        max_pos = 1

    q_row, q_head = _row_and_head_stride(query, head_size)
    k_row, k_head = _row_and_head_stride(key, head_size)

    pos = positions.reshape(-1)
    pos_stride = int(pos.stride(0))
    cs_stride = int(cos_sin_cache.stride(-2))

    bh = _BH
    cq = (nq + bh - 1) // bh
    ck = (nkv + bh - 1) // bh
    block_m = triton.next_power_of_2(m)
    full = (block_m == m) and (nq % bh == 0) and (nkv % bh == 0)

    grid = (nt, cq + ck)
    _rope_inplace_kernel[grid](
        query,
        key,
        pos,
        cos_sin_cache,
        nq,
        nkv,
        max_pos,
        pos_stride,
        cs_stride,
        cq,
        QR=q_row,
        QH=q_head,
        KR=k_row,
        KH=k_head,
        BH=bh,
        M=m,
        BLOCK_M=block_m,
        IS_NEOX=bool(is_neox),
        ODT=odt,
        FULL=full,
        num_warps=_NUM_WARPS,
    )
    return query, key


__all__ = ["sglang_fl_rotary_embedding"]
