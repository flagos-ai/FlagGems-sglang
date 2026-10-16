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
    hidden,
    w_row_stride,
    org_start,
    org_end,
    added_start,
    added_end,
    added_offset,
    n_tokens,
    BLOCK_H: tl.constexpr,
    ROWS: tl.constexpr,
    EVEN_H: tl.constexpr,
    EVEN_T: tl.constexpr,
):
    pid_t = tl.program_id(0)
    pid_h = tl.program_id(1)

    cols = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    out_base = out_ptr + pid_t * ROWS * hidden + cols
    for r in tl.static_range(ROWS):
        row = pid_t * ROWS + r
        if EVEN_H and EVEN_T:
            tok = tl.load(tok_ptr + row).to(tl.int32)
            in_org = (tok >= org_start) & (tok < org_end)
            in_added = (tok >= added_start) & (tok < added_end)
            valid = in_org | in_added
            local_id = tl.where(in_org, tok - org_start, tok - added_offset)
            local_id = tl.where(valid, local_id, 0)
            vals = tl.load(weight_ptr + local_id * w_row_stride + cols)
            vals = tl.where(valid, vals, 0.0)
            tl.store(out_base + r * hidden, vals)
        else:
            rmask = row < n_tokens
            cmask = cols < hidden
            tok = tl.load(tok_ptr + row, mask=rmask, other=0).to(tl.int32)
            in_org = (tok >= org_start) & (tok < org_end)
            in_added = (tok >= added_start) & (tok < added_end)
            valid = (in_org | in_added) & rmask

            local_id = tl.where(in_org, tok - org_start, tok - added_offset)
            local_id = tl.where(valid, local_id, 0)

            vals = tl.load(
                weight_ptr + local_id * w_row_stride + cols,
                mask=cmask & valid,
                other=0.0,
            )
            tl.store(out_base + r * hidden, vals, mask=cmask & rmask)


def vocab_parallel_embedding(
    input_,
    weight,
    org_vocab_start_index,
    org_vocab_end_index,
    num_org_vocab_padding,
    added_vocab_start_index,
    added_vocab_end_index,
):
    n_tokens = input_.numel()
    hidden = weight.shape[1]
    out = torch.empty(
        (*input_.shape, hidden), dtype=weight.dtype, device=weight.device
    )
    if n_tokens == 0:
        return out

    toks = input_ if input_.is_contiguous() else input_.contiguous()

    block_h = 2048 if hidden % 2048 == 0 and n_tokens <= 32 else 1024
    if block_h > hidden:
        block_h = triton.next_power_of_2(hidden)
    even_h = hidden % block_h == 0
    if n_tokens <= 512:
        rows, num_warps, num_stages = 1, 8, 4
    elif hidden > 4096:
        rows, num_warps, num_stages = 2, 4, 2
        block_h = min(block_h, 512)
        even_h = hidden % block_h == 0
    else:
        rows, num_warps, num_stages = 2, 8, 4
    even_t = n_tokens % rows == 0

    added_offset = (
        added_vocab_start_index
        - (org_vocab_end_index - org_vocab_start_index)
        - num_org_vocab_padding
    )

    grid = (triton.cdiv(n_tokens, rows), triton.cdiv(hidden, block_h))
    _vocab_parallel_embedding_kernel[grid](
        toks,
        weight,
        out,
        hidden,
        weight.stride(0),
        org_vocab_start_index,
        org_vocab_end_index,
        added_vocab_start_index,
        added_vocab_end_index,
        added_offset,
        n_tokens,
        BLOCK_H=block_h,
        ROWS=rows,
        EVEN_H=even_h,
        EVEN_T=even_t,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return out


__all__ = ["vocab_parallel_embedding"]
