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
def _kernel(
    g,
    output,
    sequence_length,
    head_count,
    scale,
    CHUNK_SIZE: tl.constexpr,
    BLOCK_H: tl.constexpr,
    HEAD_BLOCKS: tl.constexpr,
    REVERSE: tl.constexpr,
    HAS_SCALE: tl.constexpr,
):
    program = tl.program_id(0)
    head_block = program % HEAD_BLOCKS
    chunk = program // HEAD_BLOCKS
    heads = head_block * BLOCK_H + tl.arange(0, BLOCK_H)[None, :]
    times = tl.arange(0, CHUNK_SIZE)[:, None]
    chunks_per_batch = sequence_length // CHUNK_SIZE
    batch = chunk // chunks_per_batch
    chunk_in_batch = chunk - batch * chunks_per_batch
    if REVERSE:
        times = CHUNK_SIZE - 1 - times
    offsets = (
        batch * sequence_length * head_count
        + (chunk_in_batch * CHUNK_SIZE + times) * head_count
        + heads
    )
    mask = heads < head_count
    values = tl.load(g + offsets, mask=mask, other=0.0).to(tl.float32)
    values = tl.cumsum(values, axis=0)
    if HAS_SCALE:
        values *= scale
    tl.store(output + offsets, values, mask=mask)


def _launch_direct(g, chunk_size, reverse, scale, block_h, warps):
    batch, sequence, heads = g.shape
    output = torch.empty(g.shape, device=g.device, dtype=torch.float32)
    chunks = batch * (sequence // chunk_size)
    head_blocks = triton.cdiv(heads, block_h)
    _kernel[(chunks * head_blocks,)](
        g,
        output,
        sequence,
        heads,
        1.0 if scale is None else scale,
        CHUNK_SIZE=chunk_size,
        BLOCK_H=block_h,
        HEAD_BLOCKS=head_blocks,
        REVERSE=reverse,
        HAS_SCALE=scale is not None,
        num_warps=warps,
    )
    return output


@triton.jit
def _transpose_scan_kernel(
    g,
    output,
    sequence_length,
    head_count,
    scale,
    CHUNK_SIZE: tl.constexpr,
    BLOCK_H: tl.constexpr,
    HEAD_BLOCKS: tl.constexpr,
    REVERSE: tl.constexpr,
    HAS_SCALE: tl.constexpr,
):
    program = tl.program_id(0)
    head_block = program % HEAD_BLOCKS
    chunk = program // HEAD_BLOCKS
    heads = head_block * BLOCK_H + tl.arange(0, BLOCK_H)[None, :]
    times = tl.arange(0, CHUNK_SIZE)[:, None]
    chunks_per_batch = sequence_length // CHUNK_SIZE
    batch = chunk // chunks_per_batch
    chunk_in_batch = chunk - batch * chunks_per_batch
    if REVERSE:
        times = CHUNK_SIZE - 1 - times
    offsets = (
        batch * sequence_length * head_count
        + (chunk_in_batch * CHUNK_SIZE + times) * head_count
        + heads
    )
    mask = heads < head_count
    values = tl.load(g + offsets, mask=mask, other=0.0).to(tl.float32)
    values = tl.trans(tl.cumsum(tl.trans(values), axis=1))
    if HAS_SCALE:
        values *= scale
    tl.store(output + offsets, values, mask=mask)


def _launch_transposed(g, chunk_size, reverse, scale, block_h, warps):
    batch, sequence, heads = g.shape
    output = torch.empty(g.shape, device=g.device, dtype=torch.float32)
    chunks = batch * (sequence // chunk_size)
    head_blocks = triton.cdiv(heads, block_h)
    _transpose_scan_kernel[(chunks * head_blocks,)](
        g,
        output,
        sequence,
        heads,
        1.0 if scale is None else scale,
        CHUNK_SIZE=chunk_size,
        BLOCK_H=block_h,
        HEAD_BLOCKS=head_blocks,
        REVERSE=reverse,
        HAS_SCALE=scale is not None,
        num_warps=warps,
    )
    return output


def chunk_local_cumsum_scalar(g, chunk_size, reverse=False, scale=None):
    heads = g.shape[2]
    if chunk_size == 16:
        if heads >= 128:
            return _launch_transposed(g, chunk_size, reverse, scale, 32, 4)
        return _launch_direct(g, chunk_size, reverse, scale, 32, 4)
    if chunk_size == 32:
        if heads == 64:
            return _launch_transposed(g, chunk_size, reverse, scale, 32, 4)
        return _launch_direct(g, chunk_size, reverse, scale, 32, 4)
    if chunk_size == 64:
        if heads >= 128:
            return _launch_transposed(g, chunk_size, reverse, scale, 16, 4)
        return _launch_direct(g, chunk_size, reverse, scale, 16, 4)
    if chunk_size == 128:
        return _launch_direct(g, chunk_size, reverse, scale, 16, 4)
    if heads >= 64:
        return _launch_transposed(g, chunk_size, reverse, scale, 16, 4)
    return _launch_direct(g, chunk_size, reverse, scale, 8, 2)


reference = chunk_local_cumsum_scalar

__all__ = ["chunk_local_cumsum_scalar"]
