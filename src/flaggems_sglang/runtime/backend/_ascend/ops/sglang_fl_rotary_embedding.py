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

"""Triton fused RoPE (SGLang apply_rope_with_cos_sin_cache_inplace semantics).

One launch rotates the leading ``rotary_dim`` lanes of every head of both
``query`` and ``key`` in place; lanes past ``rotary_dim`` are never touched.

Grid: pid(0) selects a head tile of ``query`` or ``key``, pid(1) a token block.

The cos/sin row for every token is gathered with two loads of ``rd/2`` fp32
values.  A tile whose tokens all share one position -- the common decode case
(``positions = arange(seq_len).repeat(batch)`` with ``seq_len == 1``) -- issues
that gather with a replicated row index tensor, which on this backend is several
times slower than a uniform-row load, so that case is served by a single row
load broadcast over the whole tile instead.
"""

import triton
import triton.language as tl


@triton.jit
def _rope_heads(
    base,
    pos_ptr,
    cache_ptr,
    off_tok,
    tok_mask,
    h0,
    off_h,
    hmask,
    stride_t,
    stride_c,
    hs,
    rd2,
    TOK_BLK: tl.constexpr,
    HT: tl.constexpr,
    PB: tl.constexpr,
    TPB: tl.constexpr,
    IS_NEOX: tl.constexpr,
):
    off_c = tl.arange(0, PB)
    lane = off_c < rd2

    # cos/sin row for every token, or one row broadcast when they all agree.
    pos = tl.load(pos_ptr + off_tok, mask=tok_mask, other=0).to(tl.int32)
    pmax = tl.max(pos, axis=0)
    pmin = tl.min(pos, axis=0)
    same = pmax == pmin

    hoff = (h0 + off_h)[None, :, None] * hs
    ptrs = base + off_tok[:, None, None] * stride_t + hoff

    if IS_NEOX:
        off = off_c[None, None, :]
        m = (
            tok_mask[:, None, None]
            & hmask[None, :, None]
            & lane[None, None, :]
        )
        x1r = tl.load(ptrs + off, mask=m, other=0.0)
        x2r = tl.load(ptrs + (rd2 + off), mask=m, other=0.0)
        x1 = x1r.to(tl.float32)
        x2 = x2r.to(tl.float32)
        if same:
            c1 = tl.load(
                cache_ptr + pmax * stride_c + off_c, mask=lane, other=0.0
            )[None, None, :]
            s1 = tl.load(
                cache_ptr + pmax * stride_c + (rd2 + off_c),
                mask=lane,
                other=0.0,
            )[None, None, :]
            tl.store(ptrs + off, (x1 * c1 - x2 * s1).to(x1r.dtype), mask=m)
            tl.store(
                ptrs + (rd2 + off), (x2 * c1 + x1 * s1).to(x2r.dtype), mask=m
            )
        else:
            cm = tok_mask[:, None] & lane[None, :]
            row = cache_ptr + pos[:, None] * stride_c
            c2 = tl.load(row + off_c[None, :], mask=cm, other=0.0)[:, None, :]
            s2 = tl.load(row + (rd2 + off_c)[None, :], mask=cm, other=0.0)[
                :, None, :
            ]
            tl.store(ptrs + off, (x1 * c2 - x2 * s2).to(x1r.dtype), mask=m)
            tl.store(
                ptrs + (rd2 + off), (x2 * c2 + x1 * s2).to(x2r.dtype), mask=m
            )
    else:
        off = tl.arange(0, TPB)
        m = (
            tok_mask[:, None, None]
            & hmask[None, :, None]
            & (off < 2 * rd2)[None, None, :]
        )
        xr = tl.load(ptrs + off[None, None, :], mask=m, other=0.0)
        xf = xr.to(tl.float32)
        x1, x2 = tl.split(tl.reshape(xf, (TOK_BLK, HT, PB, 2)))
        if same:
            c1 = tl.load(
                cache_ptr + pmax * stride_c + off_c, mask=lane, other=0.0
            )[None, None, :]
            s1 = tl.load(
                cache_ptr + pmax * stride_c + (rd2 + off_c),
                mask=lane,
                other=0.0,
            )[None, None, :]
            of = tl.reshape(
                tl.join(x1 * c1 - x2 * s1, x2 * c1 + x1 * s1),
                (TOK_BLK, HT, TPB),
            )
            tl.store(ptrs + off[None, None, :], of.to(xr.dtype), mask=m)
        else:
            cm = tok_mask[:, None] & lane[None, :]
            row = cache_ptr + pos[:, None] * stride_c
            c2 = tl.load(row + off_c[None, :], mask=cm, other=0.0)[:, None, :]
            s2 = tl.load(row + (rd2 + off_c)[None, :], mask=cm, other=0.0)[
                :, None, :
            ]
            of = tl.reshape(
                tl.join(x1 * c2 - x2 * s2, x2 * c2 + x1 * s2),
                (TOK_BLK, HT, TPB),
            )
            tl.store(ptrs + off[None, None, :], of.to(xr.dtype), mask=m)


@triton.jit
def _rope_kernel(
    q_ptr,
    k_ptr,
    pos_ptr,
    cache_ptr,
    nnz,
    nq,
    nk,
    nq_tiles,
    stride_qt,
    stride_kt,
    stride_c,
    hs,
    rd2,
    TOK_BLK: tl.constexpr,
    HT: tl.constexpr,
    PB: tl.constexpr,
    TPB: tl.constexpr,
    IS_NEOX: tl.constexpr,
):
    pid_h = tl.program_id(0)
    pid_t = tl.program_id(1)

    off_tok = pid_t * TOK_BLK + tl.arange(0, TOK_BLK)
    tok_mask = off_tok < nnz
    off_h = tl.arange(0, HT)

    if pid_h < nq_tiles:
        h0 = pid_h * HT
        _rope_heads(
            q_ptr,
            pos_ptr,
            cache_ptr,
            off_tok,
            tok_mask,
            h0,
            off_h,
            (h0 + off_h) < nq,
            stride_qt,
            stride_c,
            hs,
            rd2,
            TOK_BLK,
            HT,
            PB,
            TPB,
            IS_NEOX,
        )
    else:
        h0 = (pid_h - nq_tiles) * HT
        _rope_heads(
            k_ptr,
            pos_ptr,
            cache_ptr,
            off_tok,
            tok_mask,
            h0,
            off_h,
            (h0 + off_h) < nk,
            stride_kt,
            stride_c,
            hs,
            rd2,
            TOK_BLK,
            HT,
            PB,
            TPB,
            IS_NEOX,
        )


def sglang_fl_rotary_embedding(
    positions, query, key, head_size, cos_sin_cache, is_neox=True
):
    hs = int(head_size)
    rd = int(cos_sin_cache.shape[-1])
    rd2 = rd // 2

    nnz = int(query.shape[0])
    nq = int(query.shape[-1]) // hs
    nk = int(key.shape[-1]) // hs
    if nnz == 0 or nq == 0 or nk == 0 or rd2 == 0:
        return query, key

    # The interleaved path moves a full rd-wide row per head (TPB = 2*PB), so
    # its tile must be smaller to stay inside the 192 KB UB budget.
    if is_neox:
        TOK_BLK = 8
        HT = 8
    else:
        TOK_BLK = 4
        HT = 4
    PB = max(1, triton.next_power_of_2(rd2))
    TPB = 2 * PB

    nq_tiles = (nq + HT - 1) // HT
    nk_tiles = (nk + HT - 1) // HT

    grid = (nq_tiles + nk_tiles, (nnz + TOK_BLK - 1) // TOK_BLK)
    _rope_kernel[grid](
        query,
        key,
        positions,
        cos_sin_cache,
        nnz,
        nq,
        nk,
        nq_tiles,
        query.stride(0),
        key.stride(0),
        cos_sin_cache.stride(0),
        hs,
        rd2,
        TOK_BLK=TOK_BLK,
        HT=HT,
        PB=PB,
        TPB=TPB,
        IS_NEOX=bool(is_neox),
    )
    return query, key


__all__ = ["sglang_fl_rotary_embedding"]
