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

"""Fused in-place RoPE (SGLang apply_rope_with_cos_sin_cache_inplace semantics).

One @triton.jit kernel rotates the leading rotary_dim lanes of every head of
query and key, reading cos/sin from cos_sin_cache and writing back into the
caller's buffers (in place).  The function returns None.
"""

import triton
import triton.language as tl


@triton.jit
def _rope_heads(
    X,
    POS,
    CS,
    t,
    pos_stride,
    h0,
    stride_row,
    H,
    HEAD,
    RD: tl.constexpr,
    HALF: tl.constexpr,
    BLOCK_H: tl.constexpr,
    IS_NEOX: tl.constexpr,
):
    # rotation-table row for this token
    p = tl.load(POS + t * pos_stride).to(tl.int32)
    k = tl.arange(0, HALF)
    csb = CS + p * RD
    if 2 * HALF == RD:
        c = tl.load(csb + k)
        s = tl.load(csb + HALF + k)
    else:
        km = k < (RD // 2)
        c = tl.load(csb + k, mask=km, other=0.0)
        s = tl.load(csb + HALF + k, mask=km, other=0.0)

    r = h0 + tl.arange(0, BLOCK_H)
    rmask = r < H
    base = X + t * stride_row + r[:, None] * HEAD

    if IS_NEOX:
        off1 = k
        off2 = k + HALF
    else:
        off1 = 2 * k
        off2 = 2 * k + 1
    o1 = off1[None, :]
    o2 = off2[None, :]
    m = rmask[:, None]

    if 2 * HALF == RD:
        x1 = tl.load(base + o1, mask=m, other=0.0)
        x2 = tl.load(base + o2, mask=m, other=0.0)
    else:
        m2 = m & (k < (RD // 2))[None, :]
        x1 = tl.load(base + o1, mask=m2, other=0.0)
        x2 = tl.load(base + o2, mask=m2, other=0.0)

    x1f = x1.to(tl.float32)
    x2f = x2.to(tl.float32)
    cf = c.to(tl.float32)[None, :]
    sf = s.to(tl.float32)[None, :]
    y1 = x1f * cf - x2f * sf
    y2 = x2f * cf + x1f * sf

    ety = X.dtype.element_ty
    if 2 * HALF == RD:
        tl.store(base + o1, y1.to(ety), mask=m)
        tl.store(base + o2, y2.to(ety), mask=m)
    else:
        tl.store(base + o1, y1.to(ety), mask=m2)
        tl.store(base + o2, y2.to(ety), mask=m2)


@triton.jit
def _rope_kernel(
    Q,
    K,
    POS,
    CS,
    stride_q,
    stride_k,
    Hq,
    Hk,
    HEAD,
    pos_stride,
    nblkq,
    RD: tl.constexpr,
    HALF: tl.constexpr,
    BLOCK_H: tl.constexpr,
    IS_NEOX: tl.constexpr,
):
    t = tl.program_id(0)
    pid1 = tl.program_id(1)
    if pid1 < nblkq:
        _rope_heads(
            Q,
            POS,
            CS,
            t,
            pos_stride,
            pid1 * BLOCK_H,
            stride_q,
            Hq,
            HEAD,
            RD,
            HALF,
            BLOCK_H,
            IS_NEOX,
        )
    else:
        _rope_heads(
            K,
            POS,
            CS,
            t,
            pos_stride,
            (pid1 - nblkq) * BLOCK_H,
            stride_k,
            Hk,
            HEAD,
            RD,
            HALF,
            BLOCK_H,
            IS_NEOX,
        )


def sglang_fl_rotary_embedding(
    positions, query, key, head_size, cos_sin_cache, is_neox=True
):
    head_size = int(head_size)
    rd = int(cos_sin_cache.shape[-1])
    nnz = query.shape[0]
    hq = query.shape[1] // head_size
    hk = key.shape[1] // head_size

    pos = positions.reshape(-1)
    pos_stride = 0 if pos.numel() == 1 else pos.stride(0)

    half = triton.next_power_of_2((rd + 1) // 2)
    BLOCK_H = 16
    num_warps = 4
    nblkq = (hq + BLOCK_H - 1) // BLOCK_H
    nblkk = (hk + BLOCK_H - 1) // BLOCK_H
    if nblkq + nblkk == 0:
        return query, key

    _rope_kernel[(nnz, nblkq + nblkk)](
        query,
        key,
        pos,
        cos_sin_cache,
        query.stride(0),
        key.stride(0),
        hq,
        hk,
        head_size,
        pos_stride,
        nblkq,
        RD=rd,
        HALF=half,
        BLOCK_H=BLOCK_H,
        IS_NEOX=bool(is_neox),
        num_warps=num_warps,
    )
    return query, key


__all__ = ["sglang_fl_rotary_embedding"]
