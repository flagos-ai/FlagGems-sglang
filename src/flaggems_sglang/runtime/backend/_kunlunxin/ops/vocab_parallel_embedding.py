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
def _vocab_embed_kernel(
    input_ptr,
    weight_ptr,
    out_ptr,
    HIDDEN: tl.constexpr,
    W_STRIDE: tl.constexpr,
    O_STRIDE: tl.constexpr,
    N_TILES: tl.constexpr,
    ORG_START: tl.constexpr,
    ORG_END: tl.constexpr,
    NUM_PAD: tl.constexpr,
    ADD_START: tl.constexpr,
    ADD_END: tl.constexpr,
    BLOCK: tl.constexpr,
):
    p = tl.program_id(0)
    row = p // N_TILES
    col = (p % N_TILES) * BLOCK + tl.arange(0, BLOCK)
    token = tl.load(input_ptr + row).to(tl.int64)
    org = (token >= ORG_START) & (token < ORG_END)
    added = (token >= ADD_START) & (token < ADD_END)
    good = org | added
    idx = tl.where(
        org,
        token - ORG_START,
        token - (ADD_START - (ORG_END - ORG_START) - NUM_PAD),
    )
    idx = tl.where(good, idx, 0)
    value = tl.load(weight_ptr + idx * W_STRIDE + col, col < HIDDEN, other=0)
    value = tl.where(good, value, value * 0.0)
    tl.store(out_ptr + row * O_STRIDE + col, value, col < HIDDEN)


def vocab_parallel_embedding(
    input_,
    weight,
    org_vocab_start_index,
    org_vocab_end_index,
    num_org_vocab_padding,
    added_vocab_start_index,
    added_vocab_end_index,
):
    n = input_.numel()
    h = weight.shape[1]
    os = (
        triton.cdiv(h * weight.element_size(), 128)
        * 128
        // weight.element_size()
    )
    out = torch.empty_strided(
        (n, h), (os, 1), device=weight.device, dtype=weight.dtype
    )
    if n and h:
        b = min(2048, triton.next_power_of_2(h))
        nt = triton.cdiv(h, b)
        _vocab_embed_kernel[(n * nt,)](
            input_,
            weight,
            out,
            h,
            weight.stride(0),
            os,
            nt,
            org_vocab_start_index,
            org_vocab_end_index,
            num_org_vocab_padding,
            added_vocab_start_index,
            added_vocab_end_index,
            b,
            num_warps=4,
            num_stages=1,
            isCloseUnrollControl=True,
        )
    return out.view(*input_.shape, h)


__all__ = ["vocab_parallel_embedding"]
