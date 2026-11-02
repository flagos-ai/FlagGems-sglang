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
def _vocab_parallel_embedding_kernel(
    tok_ptr,
    weight_ptr,
    out_ptr,
    m,
    ORG_START: tl.constexpr,
    ORG_END: tl.constexpr,
    ADDED_START: tl.constexpr,
    ADDED_END: tl.constexpr,
    ADDED_OFFSET: tl.constexpr,
    HIDDEN: tl.constexpr,
    WEIGHT_STRIDE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_H: tl.constexpr,
    EVEN_M: tl.constexpr,
    EVEN_H: tl.constexpr,
    STREAM_OUT: tl.constexpr,
    IDX32: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_h = tl.program_id(1)

    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)

    if EVEN_M:
        tok = tl.load(tok_ptr + rows)
    else:
        tok = tl.load(tok_ptr + rows, mask=rows < m, other=0)

    in_org = (tok >= ORG_START) & (tok < ORG_END)
    in_added = (tok >= ADDED_START) & (tok < ADDED_END)
    valid = in_org | in_added

    local_id = tl.where(in_org, tok - ORG_START, tok - ADDED_OFFSET)
    local_id = tl.where(valid, local_id, 0)

    if IDX32:
        w_off = local_id[:, None] * WEIGHT_STRIDE + cols[None, :]
        out_off = rows[:, None] * HIDDEN + cols[None, :]
    else:
        w_off = local_id[:, None].to(tl.int64) * WEIGHT_STRIDE + cols[None, :]
        out_off = rows.to(tl.int64)[:, None] * HIDDEN + cols[None, :]

    if EVEN_M and EVEN_H:
        w = tl.load(weight_ptr + w_off)
        w = tl.where(valid[:, None], w, 0.0)
        if STREAM_OUT:
            tl.store(out_ptr + out_off, w, eviction_policy="evict_first")
        else:
            tl.store(out_ptr + out_off, w)
    else:
        mask = (rows < m)[:, None] & (cols < HIDDEN)[None, :]
        w = tl.load(weight_ptr + w_off, mask=mask, other=0.0)
        w = tl.where(valid[:, None], w, 0.0)
        if STREAM_OUT:
            tl.store(
                out_ptr + out_off, w, mask=mask, eviction_policy="evict_first"
            )
        else:
            tl.store(out_ptr + out_off, w, mask=mask)


def vocab_parallel_embedding(
    input_,
    weight,
    org_vocab_start_index,
    org_vocab_end_index,
    num_org_vocab_padding,
    added_vocab_start_index,
    added_vocab_end_index,
):
    org_start = (
        0 if org_vocab_start_index is None else int(org_vocab_start_index)
    )
    org_end = 0 if org_vocab_end_index is None else int(org_vocab_end_index)
    padding = (
        0 if num_org_vocab_padding is None else int(num_org_vocab_padding)
    )
    added_start = (
        0 if added_vocab_start_index is None else int(added_vocab_start_index)
    )
    added_end = (
        0 if added_vocab_end_index is None else int(added_vocab_end_index)
    )

    tok_flat = input_.reshape(-1)
    m = tok_flat.numel()
    hidden = weight.shape[1]

    out = torch.empty(
        (*input_.shape, hidden), dtype=weight.dtype, device=weight.device
    )
    if m == 0:
        return out

    added_offset = added_start - (org_end - org_start) - padding

    if m <= 8:
        block_m, block_h, num_warps = 1, 1024, 4
    elif m < 512:
        if hidden >= 4096:
            block_m, block_h, num_warps = 1, 1024, 1
        else:
            block_m, block_h, num_warps = 8, 1024, 2
    elif m < 4096:
        if hidden >= 4096:
            block_m, block_h, num_warps = 16, 1024, 8
        else:
            block_m, block_h, num_warps = 8, 1024, 2
    elif hidden % 2048 == 0:
        block_m, block_h, num_warps = 8, 2048, 1
    elif hidden % 1024 == 0:
        block_m, block_h, num_warps = 8, 1024, 16
    else:
        block_m, block_h, num_warps = 1, 4096, 8

    stream_out = m * hidden >= 8_000_000

    idx32 = weight.shape[0] * weight.stride(0) < 2**31 and m * hidden < 2**31

    even_m = m % block_m == 0
    even_h = hidden % block_h == 0
    num_h = hidden // block_h if even_h else triton.cdiv(hidden, block_h)
    grid = (triton.cdiv(m, block_m), num_h)

    _vocab_parallel_embedding_kernel[grid](
        tok_flat,
        weight,
        out,
        m,
        ORG_START=org_start,
        ORG_END=org_end,
        ADDED_START=added_start,
        ADDED_END=added_end,
        ADDED_OFFSET=added_offset,
        HIDDEN=hidden,
        WEIGHT_STRIDE=weight.stride(0),
        BLOCK_M=block_m,
        BLOCK_H=block_h,
        EVEN_M=even_m,
        EVEN_H=even_h,
        STREAM_OUT=stream_out,
        IDX32=idx32,
        num_warps=num_warps,
    )
    return out


__all__ = ["vocab_parallel_embedding"]
