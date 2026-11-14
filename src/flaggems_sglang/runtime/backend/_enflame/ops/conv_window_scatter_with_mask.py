# Copyright 2026, The FlagOS Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
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

"""conv_window_scatter_with_mask -- Triton kernel (FlagOS S2 Track-1, Batch 8, Task 111).

GCU-enflame rewrite (2026-11-08): persistent grid-stride segment kernel.

For every request i with step_indices_raw[i] >= 0:
    dst[:, dst_indices_raw[i], :, :] = src[:, i, step_indices_raw[i], :, :]
`src` is an overlapping as_strided conv window, so the (dim, win) block is addressed by its
own strides and adjacent draft steps share elements.  Fills `dst` in place and returns it.

Why not the generic grid=(requests, layers, dim-chunks) launch: every (request, layer)
pair is a 1.5 KB scatter segment, so the benchmark shapes dispatch 64-2048 micro CTAs
whose ramp cost (~48 ns/CTA above the 48 free slots) dominates the wall time.  The
rewrite keeps a single kernel launch but walks a flat segment space from at most 48
persistent CTAs and merges the whole layer axis into one [LBLK, DBLK, WBLK] tile per
segment: step/slot are loaded once per request instead of once per (request, layer),
and each segment issues one wide gather plus one wide scatter.  Segment tiles are
capped at 8192 lanes (DSM budget) by shrinking DBLK then LBLK; when the resulting
segment count cannot fill the 48 free dispatch slots (S < 32) LBLK is halved further
to trade tile width for CTA parallelism.  Per-segment masks are constexpr-bounded
(l/L, d/DIM, w/WIN) combined with the scalar `step >= 0` validity, and row bases stay
pure arithmetic (scalar base + arange affine offsets) to keep DMA lowering vectorized.
A host guard grows the grid past 48 only when a CTA would exceed 128 loop iterations,
staying clear of the 2D-affine-index long-loop device abort threshold.
"""

import triton
import triton.language as tl

OP = "conv_window_scatter_with_mask"

_NUM_SP_SLOTS = 48
_MAX_LANES = 8192
_MIN_SEGMENTS = 32


@triton.jit
def _cws_persist(
    dst_ptr,
    src_ptr,
    dst_idx_ptr,
    step_idx_ptr,
    d_stride_l: tl.constexpr,
    d_stride_c: tl.constexpr,
    d_stride_dim: tl.constexpr,
    d_stride_win: tl.constexpr,
    s_stride_l: tl.constexpr,
    s_stride_r: tl.constexpr,
    s_stride_step: tl.constexpr,
    s_stride_dim: tl.constexpr,
    s_stride_win: tl.constexpr,
    S,
    L: tl.constexpr,
    DIM: tl.constexpr,
    WIN: tl.constexpr,
    SEGP_R: tl.constexpr,
    DC: tl.constexpr,
    LBLK: tl.constexpr,
    DBLK: tl.constexpr,
    WBLK: tl.constexpr,
):
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    l = tl.arange(0, LBLK)
    d = tl.arange(0, DBLK)
    w = tl.arange(0, WBLK)
    m0 = (
        (l < L)[:, None, None]
        & (d < DIM)[None, :, None]
        & (w < WIN)[None, None, :]
    )
    so_l = l[:, None, None] * s_stride_l
    do_l = l[:, None, None] * d_stride_l
    so_d = d[None, :, None] * s_stride_dim
    do_d = d[None, :, None] * d_stride_dim
    so_w = w[None, None, :] * s_stride_win
    do_w = w[None, None, :] * d_stride_win
    for s in tl.range(pid, S, nprog):
        r = s // SEGP_R
        t = s % SEGP_R
        step = tl.load(step_idx_ptr + r).to(tl.int32)
        slot = tl.load(dst_idx_ptr + r).to(tl.int32)
        m = (step >= 0) & m0
        sofs = (
            r * s_stride_r
            + step * s_stride_step
            + (t // DC) * LBLK * s_stride_l
            + (t % DC) * DBLK * s_stride_dim
            + so_l
            + so_d
            + so_w
        )
        dofs = (
            slot * d_stride_c
            + (t // DC) * LBLK * d_stride_l
            + (t % DC) * DBLK * d_stride_dim
            + do_l
            + do_d
            + do_w
        )
        data = tl.load(src_ptr + sofs, mask=m)
        tl.store(
            dst_ptr + dofs,
            data.to(dst_ptr.dtype.element_ty),
            mask=m,
        )


def conv_window_scatter_with_mask(dst, src, dst_indices_raw, step_indices_raw):
    if int(dst.dim()) != 4 or int(src.dim()) != 5:
        raise RuntimeError(
            f"{OP}: expected dst[layers,cache,dim,K-1] and "
            f"src[layers,requests,draft,dim,K-1], got {tuple(dst.shape)}"
            f"/{tuple(src.shape)}"
        )
    L = int(dst.shape[0])
    DIM = int(dst.shape[2])
    WIN = int(dst.shape[3])
    rows = min(
        int(src.shape[1]),
        int(step_indices_raw.shape[0]),
        int(dst_indices_raw.shape[0]),
    )
    if rows == 0 or L == 0 or DIM == 0 or WIN == 0:
        return dst

    wb = 1 << (max(1, WIN) - 1).bit_length()
    db = 1 << (max(1, DIM) - 1).bit_length()
    lb = 1 << (max(1, L) - 1).bit_length()
    while lb * db * wb > _MAX_LANES and db > 1:
        db //= 2
    while lb * db * wb > _MAX_LANES and lb > 1:
        lb //= 2
    # Top up CTA parallelism when the segment count cannot fill the free
    # dispatch slots: halve the layer tile (more segments, same scalars
    # per request) until S >= 32.
    while (
        rows * -(-L // lb) * -(-DIM // db) < _MIN_SEGMENTS
        and lb > 1
    ):
        lb //= 2
    lc = -(-L // lb)
    dc = -(-DIM // db)
    segp_r = lc * dc
    S = rows * segp_r
    grid = min(_NUM_SP_SLOTS, S)
    if -(-S // grid) > 128:
        grid = min(65535, -(-S // 128))

    ds = dst.stride()
    ss = src.stride()
    _cws_persist[(grid,)](
        dst,
        src,
        dst_indices_raw,
        step_indices_raw,
        d_stride_l=int(ds[0]),
        d_stride_c=int(ds[1]),
        d_stride_dim=int(ds[2]),
        d_stride_win=int(ds[3]),
        s_stride_l=int(ss[0]),
        s_stride_r=int(ss[1]),
        s_stride_step=int(ss[2]),
        s_stride_dim=int(ss[3]),
        s_stride_win=int(ss[4]),
        S=S,
        L=L,
        DIM=DIM,
        WIN=WIN,
        SEGP_R=segp_r,
        DC=dc,
        LBLK=lb,
        DBLK=db,
        WBLK=wb,
        num_warps=1,
    )
    return dst


__all__ = ["conv_window_scatter_with_mask"]
