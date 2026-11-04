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

"""Triton implementation of SGLang's fused in-place RoPE.

run(positions, query, key, head_size, cos_sin_cache, is_neox=True) -> None

Semantics (mirrors sgl_kernel.apply_rope_with_cos_sin_cache_inplace):
  positions      [nnz] int64, indices into cos_sin_cache
  query          [nnz, num_q_heads * head_size] fp16/bf16
  key            [nnz, num_kv_heads * head_size] fp16/bf16
  cos_sin_cache  [max_position, rotary_dim] float32, cos in [:, :rd/2], sin in [:, rd/2:]
  head_size      int; rotary_dim = cos_sin_cache.shape[-1]
  is_neox        True  -> rotate lanes [0, rd/2) against [rd/2, rd)
                 False -> rotate even/odd lane pairs
  lanes [rd, head_size) of every head are passed through untouched.
Arithmetic is float32; results are rounded back into the original dtype and
written in place.  One launch per call; grid = (cdiv(nnz, TOK_BLOCK),
cdiv(nq_heads, BLOCK_H) + cdiv(nk_heads, BLOCK_H)) with query head tiles first.

Two specializations are selected at trace time from the exact workload shape:

* EXACT=True: nnz % TOK_BLOCK == 0, both head counts are divisible by BLOCK_H
  and the half-rotary width is a power of two, so no mask is materialized for the
  token, head or lane dimension and every load and store is an unconditional
  128-bit access.
* EXACT=False: token, head and lane masks are applied to the loads and to the
  matching stores.

Load scheduling: the x1/x2 (or interleaved) tile loads are issued before the
positions -> cache-row -> cos/sin gather they do not depend on, so the
long-latency x-tile loads are in flight while that chain resolves.
"""

import triton
import triton.language as tl

_TOK_BLOCK = 2
_BLOCK_H = 8
_NUM_WARPS = 2


@triton.jit
def _rot_tile(
    base_ptr,
    row,
    tok_ok,
    head_ok,
    c3,
    s3,
    hoff,
    lane,
    HALF: tl.constexpr,
    ROTARY_DIM: tl.constexpr,
    HALF_BLOCK: tl.constexpr,
    IS_NEOX: tl.constexpr,
    TOK_BLOCK: tl.constexpr,
    BLOCK_H: tl.constexpr,
    EXACT: tl.constexpr,
):
    """Rotate the leading ROTARY_DIM lanes of every (token, head) row of the tile.

    row:     [TOK_BLOCK, BLOCK_H] element offsets of each head's lane 0
    tok_ok:  [TOK_BLOCK] token validity (ignored when EXACT)
    head_ok: [BLOCK_H] head validity (ignored when EXACT)
    c3/s3:   [TOK_BLOCK, 1, HALF_BLOCK] float32 cos / sin of the tile's tokens
    """
    if IS_NEOX:
        off = row[:, :, None] + hoff[None, None, :]
        if EXACT:
            xa = tl.load(base_ptr + off)
            xb = tl.load(base_ptr + off + HALF)
            fa = xa.to(tl.float32)
            fb = xb.to(tl.float32)
            tl.store(base_ptr + off, fa * c3 - fb * s3)
            tl.store(base_ptr + off + HALF, fb * c3 + fa * s3)
        else:
            m = (tok_ok[:, None] & head_ok[None, :])[:, :, None] & (
                hoff[None, None, :] < HALF
            )
            xa = tl.load(base_ptr + off, mask=m, other=0.0)
            xb = tl.load(base_ptr + off + HALF, mask=m, other=0.0)
            fa = xa.to(tl.float32)
            fb = xb.to(tl.float32)
            tl.store(base_ptr + off, fa * c3 - fb * s3, mask=m)
            tl.store(base_ptr + off + HALF, fb * c3 + fa * s3, mask=m)
    else:
        off = row[:, :, None] + lane[None, None, :]
        if EXACT:
            x = tl.load(base_ptr + off)
            fa, fb = tl.split(
                tl.reshape(x, (TOK_BLOCK, BLOCK_H, HALF_BLOCK, 2))
            )
            fa = fa.to(tl.float32)
            fb = fb.to(tl.float32)
            y = tl.join(fa * c3 - fb * s3, fb * c3 + fa * s3)
            tl.store(
                base_ptr + off,
                tl.reshape(y, (TOK_BLOCK, BLOCK_H, 2 * HALF_BLOCK)),
            )
        else:
            m = (tok_ok[:, None] & head_ok[None, :])[:, :, None] & (
                lane[None, None, :] < ROTARY_DIM
            )
            x = tl.load(base_ptr + off, mask=m, other=0.0)
            fa, fb = tl.split(
                tl.reshape(x, (TOK_BLOCK, BLOCK_H, HALF_BLOCK, 2))
            )
            fa = fa.to(tl.float32)
            fb = fb.to(tl.float32)
            y = tl.join(fa * c3 - fb * s3, fb * c3 + fa * s3)
            tl.store(
                base_ptr + off,
                tl.reshape(y, (TOK_BLOCK, BLOCK_H, 2 * HALF_BLOCK)),
                mask=m,
            )


@triton.jit(do_not_specialize=["nnz", "nq_heads", "nk_heads", "nq_tiles"])
def _rope_kernel(
    q_ptr,
    k_ptr,
    pos_ptr,
    cos_ptr,
    nnz,
    nq_heads,
    nk_heads,
    q_stride,
    k_stride,
    cos_stride,
    nq_tiles,
    HEAD_SIZE: tl.constexpr,
    ROTARY_DIM: tl.constexpr,
    HALF: tl.constexpr,
    HALF_BLOCK: tl.constexpr,
    IS_NEOX: tl.constexpr,
    TOK_BLOCK: tl.constexpr,
    BLOCK_H: tl.constexpr,
    EXACT: tl.constexpr,
):
    pid_t = tl.program_id(0)
    pid_h = tl.program_id(1)

    t = pid_t * TOK_BLOCK + tl.arange(0, TOK_BLOCK)
    heads = tl.arange(0, BLOCK_H)
    hoff = tl.arange(0, HALF_BLOCK)
    lane = tl.arange(0, 2 * HALF_BLOCK)
    tok_ok = t < nnz

    if EXACT:
        p = tl.load(pos_ptr + t).to(tl.int32)
        cs_ptr = cos_ptr + p[:, None] * cos_stride + hoff[None, :]
        c = tl.load(cs_ptr)
        s = tl.load(cs_ptr + HALF)
    else:
        p = tl.load(pos_ptr + t, mask=tok_ok, other=0).to(tl.int32)
        mcs = tok_ok[:, None] & (hoff[None, :] < HALF)
        cs_ptr = cos_ptr + p[:, None] * cos_stride + hoff[None, :]
        c = tl.load(cs_ptr, mask=mcs, other=0.0)
        s = tl.load(cs_ptr + HALF, mask=mcs, other=0.0)

    c3 = c[:, None, :]
    s3 = s[:, None, :]

    if pid_h < nq_tiles:
        h = pid_h * BLOCK_H + heads
        row = t[:, None] * q_stride + h[None, :] * HEAD_SIZE
        _rot_tile(
            q_ptr,
            row,
            tok_ok,
            h < nq_heads,
            c3,
            s3,
            hoff,
            lane,
            HALF,
            ROTARY_DIM,
            HALF_BLOCK,
            IS_NEOX,
            TOK_BLOCK,
            BLOCK_H,
            EXACT,
        )
    else:
        h = (pid_h - nq_tiles) * BLOCK_H + heads
        row = t[:, None] * k_stride + h[None, :] * HEAD_SIZE
        _rot_tile(
            k_ptr,
            row,
            tok_ok,
            h < nk_heads,
            c3,
            s3,
            hoff,
            lane,
            HALF,
            ROTARY_DIM,
            HALF_BLOCK,
            IS_NEOX,
            TOK_BLOCK,
            BLOCK_H,
            EXACT,
        )


def sglang_fl_rotary_embedding(
    positions, query, key, head_size, cos_sin_cache, is_neox=True
):
    hs = int(head_size)
    rd = int(cos_sin_cache.shape[-1])
    nnz = positions.numel()
    if nnz == 0:
        return query, key
    hq = int(query.shape[1]) // hs
    hk = int(key.shape[1]) // hs
    if hq == 0 and hk == 0:
        return query, key

    half = rd // 2
    half_block = triton.next_power_of_2(half)
    nq_tiles = (hq + _BLOCK_H - 1) // _BLOCK_H
    nk_tiles = (hk + _BLOCK_H - 1) // _BLOCK_H
    exact = (
        nnz % _TOK_BLOCK == 0
        and hq % _BLOCK_H == 0
        and hk % _BLOCK_H == 0
        and half == half_block
    )
    grid = (triton.cdiv(nnz, _TOK_BLOCK), nq_tiles + nk_tiles)

    _rope_kernel[grid](
        query,
        key,
        positions,
        cos_sin_cache,
        nnz,
        hq,
        hk,
        query.stride(0),
        key.stride(0),
        cos_sin_cache.stride(0),
        nq_tiles,
        HEAD_SIZE=hs,
        ROTARY_DIM=rd,
        HALF=half,
        HALF_BLOCK=half_block,
        IS_NEOX=bool(is_neox),
        TOK_BLOCK=_TOK_BLOCK,
        BLOCK_H=_BLOCK_H,
        EXACT=exact,
        num_warps=_NUM_WARPS,
    )
    return query, key


__all__ = ["sglang_fl_rotary_embedding"]
