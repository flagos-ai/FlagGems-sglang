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
def _serial_3d_kernel(
    g,
    output,
    scale,
    SEQUENCE: tl.constexpr,
    HEADS: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
    BLOCK_H: tl.constexpr,
    REVERSE: tl.constexpr,
    HAS_SCALE: tl.constexpr,
):
    head_block = tl.program_id(0)
    chunk = tl.program_id(1)
    batch = tl.program_id(2)
    heads = head_block * BLOCK_H + tl.arange(0, BLOCK_H)
    base = (batch * SEQUENCE + chunk * CHUNK_SIZE) * HEADS + heads
    mask = heads < HEADS
    accumulator = tl.zeros((BLOCK_H,), tl.float32)
    for step in tl.static_range(0, CHUNK_SIZE):
        time = CHUNK_SIZE - 1 - step if REVERSE else step
        accumulator += tl.load(
            g + base + time * HEADS, mask=mask, other=0.0
        ).to(tl.float32)
        result = accumulator * scale if HAS_SCALE else accumulator
        tl.store(output + base + time * HEADS, result, mask=mask)


@triton.jit
def _head_major_kernel(
    g,
    output,
    scale,
    SEQUENCE: tl.constexpr,
    HEADS: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
    BLOCK_H: tl.constexpr,
    REVERSE: tl.constexpr,
    HAS_SCALE: tl.constexpr,
):
    head_block = tl.program_id(0)
    chunk = tl.program_id(1)
    batch = tl.program_id(2)
    heads = head_block * BLOCK_H + tl.arange(0, BLOCK_H)[:, None]
    times = tl.arange(0, CHUNK_SIZE)[None, :]
    if REVERSE:
        times = CHUNK_SIZE - 1 - times
    offsets = (batch * SEQUENCE + chunk * CHUNK_SIZE + times) * HEADS + heads
    mask = heads < HEADS
    values = tl.load(g + offsets, mask=mask, other=0.0).to(tl.float32)
    values = tl.cumsum(values, axis=1)
    if HAS_SCALE:
        values *= scale
    tl.store(output + offsets, values, mask=mask)


def launch(g, chunk_size, reverse, scale, block_h, warps):
    batch, sequence, heads = g.shape
    output = torch.empty(g.shape, device=g.device, dtype=torch.float32)
    _serial_3d_kernel[
        (triton.cdiv(heads, block_h), sequence // chunk_size, batch)
    ](
        g,
        output,
        1.0 if scale is None else scale,
        SEQUENCE=sequence,
        HEADS=heads,
        CHUNK_SIZE=chunk_size,
        BLOCK_H=block_h,
        REVERSE=reverse,
        HAS_SCALE=scale is not None,
        num_warps=warps,
    )
    return output


def launch_head_major(g, chunk_size, reverse, scale, block_h, warps):
    batch, sequence, heads = g.shape
    output = torch.empty_like(g, dtype=torch.float32)
    grid = (triton.cdiv(heads, block_h), sequence // chunk_size, batch)
    _head_major_kernel[grid](
        g,
        output,
        1.0 if scale is None else scale,
        SEQUENCE=sequence,
        HEADS=heads,
        CHUNK_SIZE=chunk_size,
        BLOCK_H=block_h,
        REVERSE=reverse,
        HAS_SCALE=scale is not None,
        num_warps=warps,
        num_stages=1,
    )
    return output


def chunk_local_cumsum_scalar(g, chunk_size, reverse=False, scale=None):
    heads = g.shape[2]
    if (
        chunk_size == 64
        and not reverse
        and scale is None
        and heads in (32, 64)
    ):
        if heads == 32:
            return launch_head_major(g, chunk_size, reverse, scale, 2, 2)
        return launch_head_major(g, chunk_size, reverse, scale, 4, 1)
    block_h = 64 if heads >= 64 else 32
    return launch(g, chunk_size, reverse, scale, block_h, 4)


__all__ = ["chunk_local_cumsum_scalar"]
