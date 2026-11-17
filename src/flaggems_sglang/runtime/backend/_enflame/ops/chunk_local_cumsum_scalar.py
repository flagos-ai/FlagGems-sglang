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
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied. See the License for the specific language governing
# permissions and limitations under the License.

"""chunk_local_cumsum_scalar — local (per-chunk) cumulative sum along time.

[B, T, H] input (T divisible by chunk_size); output is always float32.
Semantics (matches the PyTorch reference exactly):

    g_c = g.float().view(B, T // BT, BT, H)
    if reverse: g_c = g_c.flip(2)
    out = g_c.cumsum(dim=2)
    if scale is not None: out = out * scale
    if reverse: out = out.flip(2)

GCU delivery notes (L600 / GCU400, triton 3.6 + torch_gcu 2.11.0+4.1.20261015):

* Toolchain constraint (compiler_issues/sglang_chunk_local_cumsum_scalar.md):
  the gcu-compiler-opt attribute-conversion regression blocks fresh compiles
  of ascending-row 2D tiles, num_warps >= 2 and every tl.dot form.  The only
  compile-safe 2D tile here is a DESCENDING-row tile at num_warps = 1 (with
  the head mask when BLOCK_H > H, without it when the tile is exact), so the
  kernel loads and stores each chunk with reversed
  row indices.  cumsum over the reversed tile is the suffix sum — exactly the
  reverse-mode output; forward mode recovers the prefix with the exact
  identity prefix = total - suffix + value (fp32 error <= ~1e-6, far inside
  the bf16 test tolerance).
* Launch shape: one persistent program per SP slot (grid = min(total_chunks,
  48) on the 48-SP L600) walking chunks grid-stride.  Raising the old
  grid=24 to 48 alone takes [8,1024,32] from 48.6us to ~28.4us (do_bench
  incl. wrapper); the flat (non-persistent) launch is slower (wave tail).
* BLOCK_H is capped so that CHUNK * BLOCK_H <= 4096 elements — the per-warp
  slice budget of the toolchain regression; wider H fans out over grid axis 1
  (heads are the contiguous axis, so each head block is still N-contiguous).
  When BLOCK_H == H the tile is exact and the mask is dropped (NOMASK),
  which saves ~0.7us per launch; the unmasked form only compiles for the
  descending-row tile, not for the general masked path.
"""

import torch
import triton
import triton.language as tl

# L600 (GCU400) SP count: grid beyond this adds time-slicing, not parallelism.
_NUM_SP = 48
# Per-warp element budget for the descending masked 2D tile (fresh-compile
# boundary of the gcu-compiler-opt slice_(pad_)start regression at nw=1).
_MAX_TILE_ELEMS = 4096


@triton.jit
def _scan_persistent(
    g,
    out,
    scale,
    T: tl.constexpr,
    H: tl.constexpr,
    CHUNK: tl.constexpr,
    NUM_CHUNKS: tl.constexpr,
    TOTAL_CHUNKS: tl.constexpr,
    GRID_CHUNKS: tl.constexpr,
    BLOCK_H: tl.constexpr,
    FORWARD: tl.constexpr,
    HAS_SCALE: tl.constexpr,
    NOMASK: tl.constexpr,
):
    program_chunk = tl.program_id(0)
    head_block = tl.program_id(1)
    heads = head_block * BLOCK_H + tl.arange(0, BLOCK_H)[None, :]
    times = tl.arange(0, CHUNK)[:, None]
    rev_times = CHUNK - 1 - times
    if not NOMASK:
        mask = heads < H
    for batch_chunk in range(program_chunk, TOTAL_CHUNKS, GRID_CHUNKS):
        batch = batch_chunk // NUM_CHUNKS
        chunk = batch_chunk - batch * NUM_CHUNKS
        offsets = (batch * T + chunk * CHUNK + rev_times) * H + heads
        if NOMASK:
            values = tl.load(g + offsets).to(tl.float32)
        else:
            values = tl.load(g + offsets, mask=mask, other=0.0).to(tl.float32)
        s = tl.cumsum(values, axis=0)
        if FORWARD:
            z = tl.sum(values, axis=0)
            y = z[None, :] - s + values
        else:
            y = s
        if HAS_SCALE:
            y = y * scale
        if NOMASK:
            tl.store(out + offsets, y)
        else:
            tl.store(out + offsets, y, mask=mask)


def chunk_local_cumsum_scalar(g, chunk_size, reverse=False, scale=None):
    batch, time, heads = g.shape
    num_chunks = time // chunk_size
    total_chunks = batch * num_chunks
    grid_chunks = total_chunks if total_chunks < _NUM_SP else _NUM_SP

    block_h = triton.next_power_of_2(heads)
    tile_cap = _MAX_TILE_ELEMS // chunk_size
    while block_h > 1 and block_h > tile_cap:
        block_h //= 2
    num_col_blocks = triton.cdiv(heads, block_h)

    out = torch.empty_like(g, dtype=torch.float32)
    grid = (
        (grid_chunks,)
        if num_col_blocks == 1
        else (grid_chunks, num_col_blocks)
    )
    _scan_persistent[grid](
        g,
        out,
        1.0 if scale is None else scale,
        T=time,
        H=heads,
        CHUNK=chunk_size,
        NUM_CHUNKS=num_chunks,
        TOTAL_CHUNKS=total_chunks,
        GRID_CHUNKS=grid_chunks,
        BLOCK_H=block_h,
        FORWARD=not reverse,
        HAS_SCALE=scale is not None,
        NOMASK=block_h == heads,
        num_warps=1,
        num_stages=1,
    )
    return out


__all__ = ["chunk_local_cumsum_scalar"]
