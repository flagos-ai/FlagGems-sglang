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


@triton.jit
def _scan_3d(
    g,
    out,
    scale,
    T: tl.constexpr,
    H: tl.constexpr,
    CHUNK: tl.constexpr,
    BLOCK_H: tl.constexpr,
    REVERSE: tl.constexpr,
    HAS_SCALE: tl.constexpr,
    TRANSPOSE: tl.constexpr,
):
    head_block = tl.program_id(0)
    chunk = tl.program_id(1)
    batch = tl.program_id(2)
    heads = head_block * BLOCK_H + tl.arange(0, BLOCK_H)[None, :]
    times = tl.arange(0, CHUNK)[:, None]
    if REVERSE:
        times = CHUNK - 1 - times
    offsets = (batch * T + chunk * CHUNK + times) * H + heads
    mask = heads < H
    values = tl.load(g + offsets, mask=mask, other=0.0).to(tl.float32)
    if TRANSPOSE:
        values = tl.trans(tl.cumsum(tl.trans(values), axis=1))
    else:
        values = tl.cumsum(values, axis=0)
    if HAS_SCALE:
        values *= scale
    tl.store(out + offsets, values, mask=mask)


@triton.jit
def _scan_flat(
    g,
    out,
    scale,
    T: tl.constexpr,
    H: tl.constexpr,
    CHUNK: tl.constexpr,
    NUM_CHUNKS: tl.constexpr,
    BLOCK_H: tl.constexpr,
    REVERSE: tl.constexpr,
    HAS_SCALE: tl.constexpr,
):
    batch_chunk = tl.program_id(0)
    head_block = tl.program_id(1)
    batch = batch_chunk // NUM_CHUNKS
    chunk = batch_chunk - batch * NUM_CHUNKS
    heads = head_block * BLOCK_H + tl.arange(0, BLOCK_H)[None, :]
    times = tl.arange(0, CHUNK)[:, None]
    if REVERSE:
        times = CHUNK - 1 - times
    offsets = (batch * T + chunk * CHUNK + times) * H + heads
    mask = heads < H
    values = tl.load(g + offsets, mask=mask, other=0.0).to(tl.float32)
    values = tl.cumsum(values, axis=0)
    if HAS_SCALE:
        values *= scale
    tl.store(out + offsets, values, mask=mask)


@triton.jit
def _scan_persistent(
    g,
    out,
    scale,
    T: tl.constexpr,
    H: tl.constexpr,
    CHUNK: tl.constexpr,
    CHUNKS_PER_BATCH: tl.constexpr,
    TOTAL_CHUNKS: tl.constexpr,
    GRID_CHUNKS: tl.constexpr,
    BLOCK_H: tl.constexpr,
    REVERSE: tl.constexpr,
    HAS_SCALE: tl.constexpr,
):
    program_chunk = tl.program_id(0)
    head_block = tl.program_id(1)
    heads = head_block * BLOCK_H + tl.arange(0, BLOCK_H)[None, :]
    times = tl.arange(0, CHUNK)[:, None]
    if REVERSE:
        times = CHUNK - 1 - times
    mask = heads < H
    for batch_chunk in range(program_chunk, TOTAL_CHUNKS, GRID_CHUNKS):
        batch = batch_chunk // CHUNKS_PER_BATCH
        chunk = batch_chunk - batch * CHUNKS_PER_BATCH
        offsets = (batch * T + chunk * CHUNK + times) * H + heads
        values = tl.load(g + offsets, mask=mask, other=0.0).to(tl.float32)
        values = tl.cumsum(values, axis=0)
        if HAS_SCALE:
            values *= scale
        tl.store(out + offsets, values, mask=mask)


def _output(g, low_precision):
    return torch.empty_like(g, dtype=torch.float32)


def _launch_3d(
    g, chunk_size, reverse, scale, block_h, warps, transpose, low_precision
):
    batch, time, heads = g.shape
    out = _output(g, low_precision)
    grid = (triton.cdiv(heads, block_h), time // chunk_size, batch)
    _scan_3d[grid](
        g,
        out,
        1.0 if scale is None else scale,
        T=time,
        H=heads,
        CHUNK=chunk_size,
        BLOCK_H=block_h,
        REVERSE=reverse,
        HAS_SCALE=scale is not None,
        TRANSPOSE=transpose,
        num_warps=warps,
        num_stages=1,
    )
    return out


def _launch_flat(g, chunk_size, reverse, scale, block_h, warps, low_precision):
    batch, time, heads = g.shape
    chunks = time // chunk_size
    out = _output(g, low_precision)
    grid = (batch * chunks, triton.cdiv(heads, block_h))
    _scan_flat[grid](
        g,
        out,
        1.0 if scale is None else scale,
        T=time,
        H=heads,
        CHUNK=chunk_size,
        NUM_CHUNKS=chunks,
        BLOCK_H=block_h,
        REVERSE=reverse,
        HAS_SCALE=scale is not None,
        num_warps=warps,
        num_stages=1,
    )
    return out


def _launch_persistent(g, chunk_size, reverse, scale):
    batch, time, heads = g.shape
    chunks_per_batch = time // chunk_size
    total_chunks = batch * chunks_per_batch
    grid_chunks = min(total_chunks, 24)
    block_h = triton.next_power_of_2(heads)
    out = _output(g, False)
    _scan_persistent[(grid_chunks, 1)](
        g,
        out,
        1.0 if scale is None else scale,
        T=time,
        H=heads,
        CHUNK=chunk_size,
        CHUNKS_PER_BATCH=chunks_per_batch,
        TOTAL_CHUNKS=total_chunks,
        GRID_CHUNKS=grid_chunks,
        BLOCK_H=block_h,
        REVERSE=reverse,
        HAS_SCALE=scale is not None,
        num_warps=1,
        num_stages=1,
    )
    return out


def chunk_local_cumsum_scalar(g, chunk_size, reverse=False, scale=None):
    chunks = g.shape[1] / chunk_size
    if chunk_size == 64 and g.shape[2] <= 64:
        return _launch_persistent(g, chunk_size, reverse, scale)
    if chunk_size == 16:
        if chunks <= 255:
            return _launch_3d(
                g, chunk_size, reverse, scale, 32, 4, False, False
            )
        return _launch_flat(g, chunk_size, reverse, scale, 32, 4, False)
    if chunk_size == 32:
        return _launch_flat(g, chunk_size, reverse, scale, 32, 4, False)
    if chunk_size == 64 and g.shape[2] >= 128:
        return _launch_flat(g, chunk_size, reverse, scale, 8, 1, False)
    if chunk_size == 128:
        return _launch_flat(g, chunk_size, reverse, scale, 32, 4, False)
    return _launch_3d(g, chunk_size, reverse, scale, 32, 4, False, False)


__all__ = ["chunk_local_cumsum_scalar"]
