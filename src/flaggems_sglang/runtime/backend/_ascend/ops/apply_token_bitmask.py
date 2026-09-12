# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import triton
import triton.language as tl

CORE_NUM = 48
NWB = 64
BLOCK = NWB * 32


@triton.jit
def _decode_store(
    logits_row,
    bitmask_row,
    out_row,
    word_start,
    V,
    NW,
    NWB: tl.constexpr,
    BLOCK: tl.constexpr,
):
    word_lane = tl.arange(0, NWB)
    word_off = word_start + word_lane
    words = tl.load(bitmask_row + word_off, mask=word_off < NW, other=0)

    bit = tl.arange(0, 32)
    one = tl.full((32,), 1, tl.int32)
    powers = one << bit
    keep_2d = (words[:, None] & powers[None, :]) != 0
    keep = tl.reshape(keep_2d, (BLOCK,))

    value_off = word_start * 32 + tl.arange(0, BLOCK)
    valid = value_off < V
    values = tl.load(logits_row + value_off, mask=valid, other=0.0)
    neg_inf = tl.full(values.shape, float("-inf"), values.dtype)
    tl.store(out_row + value_off, tl.where(keep, values, neg_inf), mask=valid)


@triton.jit
def _apply_row_grid(
    logits_ptr,
    bitmask_ptr,
    out_ptr,
    V,
    NW,
    stride_lb,
    stride_bb,
    NWB: tl.constexpr,
    BLOCK: tl.constexpr,
):
    b = tl.program_id(0)
    word_tile = tl.program_id(1)
    b64 = b.to(tl.int64)
    _decode_store(
        logits_ptr + b64 * stride_lb,
        bitmask_ptr + b64 * stride_bb,
        out_ptr + b64 * V,
        word_tile * NWB,
        V,
        NW,
        NWB=NWB,
        BLOCK=BLOCK,
    )


@triton.jit
def _apply_core_grid(
    logits_ptr,
    bitmask_ptr,
    out_ptr,
    V,
    NW,
    n_tiles,
    tiles_per_row,
    stride_lb,
    stride_bb,
    NWB: tl.constexpr,
    BLOCK: tl.constexpr,
    CORES: tl.constexpr,
):
    pid = tl.program_id(0)
    for tile in range(pid, n_tiles, CORES):
        b = tile // tiles_per_row
        word_tile = tile - b * tiles_per_row
        b64 = b.to(tl.int64)
        _decode_store(
            logits_ptr + b64 * stride_lb,
            bitmask_ptr + b64 * stride_bb,
            out_ptr + b64 * V,
            word_tile * NWB,
            V,
            NW,
            NWB=NWB,
            BLOCK=BLOCK,
        )


def apply_token_bitmask(logits, bitmask):
    B, V = logits.shape
    out = torch.empty((B, V), dtype=logits.dtype, device=logits.device)
    if B == 0 or V == 0:
        return out

    if logits.stride(1) != 1:
        logits = logits.contiguous()
    if bitmask.stride(1) != 1:
        bitmask = bitmask.contiguous()

    NW = bitmask.shape[1]
    tiles_per_row = triton.cdiv(NW, NWB)
    n_tiles = B * tiles_per_row
    common = (
        logits,
        bitmask,
        out,
        V,
        NW,
    )

    if n_tiles <= 65535:
        _apply_row_grid[(B, tiles_per_row)](
            *common,
            logits.stride(0),
            bitmask.stride(0),
            NWB=NWB,
            BLOCK=BLOCK,
        )
    else:
        cores = min(CORE_NUM, n_tiles)
        _apply_core_grid[(cores,)](
            *common,
            n_tiles,
            tiles_per_row,
            logits.stride(0),
            bitmask.stride(0),
            NWB=NWB,
            BLOCK=BLOCK,
            CORES=cores,
        )
    return out


reference = apply_token_bitmask

__all__ = ["apply_token_bitmask"]
