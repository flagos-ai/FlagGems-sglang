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

_DIM_NOPE = 512
_DIM_ROPE = 64
_ROW_BYTES = 656
_OUT_ROW_ELEMS = 576


@triton.jit
def _decode_fp8_e4m3(c_u8):
    c = c_u8.to(tl.int32)
    mag = c & 0x7F
    exp = mag >> 3
    man = mag & 7
    nbits = ((exp + 120) << 23) | (man << 20)
    normal = nbits.to(tl.float32, bitcast=True)
    sub = man.to(tl.float32) * 0.001953125
    val = tl.where(exp == 0, sub, normal)
    return tl.where((c & 0x80) != 0, -val, val)


@triton.jit
def _dequant_k_cache_fast(
    q_ptr,
    o_ptr,
    NUM_TOKENS: tl.constexpr,
    NUM_TILES: tl.constexpr,
    GROUP: tl.constexpr,
    TROWS: tl.constexpr,
):
    EVEN_T: tl.constexpr = (NUM_TOKENS % TROWS) == 0
    num_tokens = NUM_TOKENS
    pid = tl.program_id(0)
    rows = pid * TROWS + tl.arange(0, TROWS)
    cols = tl.arange(0, GROUP)
    rmask = rows < num_tokens

    if EVEN_T:
        sb = tl.load(
            q_ptr + rows[:, None] * 656 + 512 + tl.arange(0, 16)[None, :]
        )
    else:
        sb = tl.load(
            q_ptr + rows[:, None] * 656 + 512 + tl.arange(0, 16)[None, :],
            mask=rmask[:, None],
            other=0,
        )
    s_even, s_odd = tl.split(tl.reshape(sb, (TROWS, 8, 2)))
    e0, e1 = tl.split(tl.reshape(s_even, (TROWS, 4, 2)))
    o0, o1 = tl.split(tl.reshape(s_odd, (TROWS, 4, 2)))
    e0w, e0q = tl.split(tl.reshape(e0, (TROWS, 2, 2)))
    e1w, e1q = tl.split(tl.reshape(e1, (TROWS, 2, 2)))
    o0w, o0q = tl.split(tl.reshape(o0, (TROWS, 2, 2)))
    o1w, o1q = tl.split(tl.reshape(o1, (TROWS, 2, 2)))
    p0, p2 = tl.split(e0w)
    p1, p3 = tl.split(e0q)
    g0, g2 = tl.split(e1w)
    g1, g3 = tl.split(e1q)
    f0, f2 = tl.split(o0w)
    f1, f3 = tl.split(o0q)
    h0, h2 = tl.split(o1w)
    h1, h3 = tl.split(o1q)
    s0 = (
        p0
        | (f0.to(tl.int32) << 8)
        | (g0.to(tl.int32) << 16)
        | (h0.to(tl.int32) << 24)
    ).to(tl.float32, bitcast=True)
    s1 = (
        p1
        | (f1.to(tl.int32) << 8)
        | (g1.to(tl.int32) << 16)
        | (h1.to(tl.int32) << 24)
    ).to(tl.float32, bitcast=True)
    s2 = (
        p2
        | (f2.to(tl.int32) << 8)
        | (g2.to(tl.int32) << 16)
        | (h2.to(tl.int32) << 24)
    ).to(tl.float32, bitcast=True)
    s3 = (
        p3
        | (f3.to(tl.int32) << 8)
        | (g3.to(tl.int32) << 16)
        | (h3.to(tl.int32) << 24)
    ).to(tl.float32, bitcast=True)

    for t in tl.static_range(NUM_TILES):
        if t == 0:
            scale = s0
        elif t == 1:
            scale = s1
        elif t == 2:
            scale = s2
        else:
            scale = s3
        if EVEN_T:
            c = tl.load(
                q_ptr + rows[:, None] * 656 + t * GROUP + cols[None, :]
            ).to(tl.int32)
            val = _decode_fp8_e4m3(c)
            tl.store(
                o_ptr + rows[:, None] * 576 + t * GROUP + cols[None, :],
                (val * scale[:, None]).to(tl.bfloat16),
            )
        else:
            c = tl.load(
                q_ptr + rows[:, None] * 656 + t * GROUP + cols[None, :],
                mask=rmask[:, None],
                other=0,
            ).to(tl.int32)
            val = _decode_fp8_e4m3(c)
            tl.store(
                o_ptr + rows[:, None] * 576 + t * GROUP + cols[None, :],
                (val * scale[:, None]).to(tl.bfloat16),
                mask=rmask[:, None],
            )

    r_offs = (
        q_ptr
        + rows[:, None] * 656
        + (512 + NUM_TILES * 4)
        + tl.arange(0, 128)[None, :]
    )
    if EVEN_T:
        rraw = tl.load(r_offs)
    else:
        rraw = tl.load(r_offs, mask=rmask[:, None], other=0)
    r2 = tl.reshape(rraw, (TROWS, 64, 2))
    r_lo, r_hi = tl.split(r2)
    word = (r_lo.to(tl.uint16) | (r_hi.to(tl.uint16) << 8)).to(
        tl.bfloat16, bitcast=True
    )
    if EVEN_T:
        tl.store(
            o_ptr + rows[:, None] * 576 + 512 + tl.arange(0, 64)[None, :], word
        )
    else:
        tl.store(
            o_ptr + rows[:, None] * 576 + 512 + tl.arange(0, 64)[None, :],
            word,
            mask=rmask[:, None],
        )


def dequantize_k_cache_fast(quant_k_cache, group_size=128):
    num_tokens, _ = quant_k_cache.shape
    if _DIM_NOPE % group_size:
        raise ValueError("group_size must divide dim_nope=512")
    num_tiles = _DIM_NOPE // group_size

    out = quant_k_cache.new_empty(
        (num_tokens, _DIM_NOPE + _DIM_ROPE), dtype=torch.bfloat16
    )
    if num_tokens == 0:
        return out

    q_u8 = quant_k_cache.view(torch.uint8)
    if not q_u8.is_contiguous():
        q_u8 = q_u8.contiguous()

    trows = 32
    grid = ((num_tokens + 31) // 32,)
    _dequant_k_cache_fast[grid](
        q_u8,
        out,
        num_tokens,
        num_tiles,
        group_size,
        trows,
        num_warps=1,
    )
    return out


__all__ = ["dequantize_k_cache_fast"]
