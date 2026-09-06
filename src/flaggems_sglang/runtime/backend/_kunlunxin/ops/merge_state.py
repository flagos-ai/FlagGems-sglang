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

import math

import torch
import triton
import triton.language as tl


@triton.jit
def _single_row_kernel(
    po, pl, so, sl, out, out_lse, HEAD: tl.constexpr, BLOCK: tl.constexpr
):
    cols = tl.arange(0, BLOCK)
    mask = cols < HEAD
    p_lse = tl.load(pl).to(tl.float32)
    s_lse = tl.load(sl).to(tl.float32)
    p_lse = tl.where(p_lse == math.inf, -math.inf, p_lse)
    s_lse = tl.where(s_lse == math.inf, -math.inf, s_lse)
    p_high = p_lse >= s_lse
    max_lse = tl.maximum(p_lse, s_lse)
    low_se = tl.exp(tl.minimum(p_lse, s_lse) - max_lse)
    inv = 1.0 / (1.0 + low_se)
    low_scale = low_se * inv
    p = tl.load(po + cols, mask=mask, other=0.0).to(tl.float32)
    s = tl.load(so + cols, mask=mask, other=0.0).to(tl.float32)
    tl.store(
        out + cols,
        p * tl.where(p_high, inv, low_scale)
        + s * tl.where(p_high, low_scale, inv),
        mask=mask,
    )
    tl.store(out_lse, tl.log(1.0 + low_se) + max_lse)


@triton.jit
def _scale_kernel(
    pl, sl, staged, out_lse, row_count, HEAD: tl.constexpr, BLOCK: tl.constexpr
):
    rows = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = rows < row_count
    p_lse = tl.load(pl + rows, mask=mask, other=-math.inf).to(tl.float32)
    s_lse = tl.load(sl + rows, mask=mask, other=-math.inf).to(tl.float32)
    p_lse = tl.where(p_lse == math.inf, -math.inf, p_lse)
    s_lse = tl.where(s_lse == math.inf, -math.inf, s_lse)
    p_high = p_lse >= s_lse
    max_lse = tl.maximum(p_lse, s_lse)
    low_se = tl.exp(tl.minimum(p_lse, s_lse) - max_lse)
    inv = 1.0 / (1.0 + low_se)
    p_scale = tl.where(p_high, inv, low_se * inv)
    tl.store(staged + rows * HEAD, p_scale, mask=mask)
    tl.store(out_lse + rows, tl.log(1.0 + low_se) + max_lse, mask=mask)


@triton.jit
def _output_kernel(
    po, so, staged, out, total, HEAD: tl.constexpr, BLOCK: tl.constexpr
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < total
    rows = offsets // HEAD
    scale = tl.load(staged + rows * HEAD, mask=mask, other=0.0).to(tl.float32)
    p = tl.load(po + offsets, mask=mask, other=0.0).to(tl.float32)
    s = tl.load(so + offsets, mask=mask, other=0.0).to(tl.float32)
    tl.store(out + offsets, p * scale + s * (1.0 - scale), mask=mask)


def merge_state(prefix_output, prefix_lse, suffix_output, suffix_lse):
    output = torch.empty_like(prefix_output)
    output_lse = torch.empty_like(prefix_lse)
    head_dim = prefix_output.shape[-1]
    rows = prefix_output.numel() // head_dim
    if rows == 1:
        _single_row_kernel[(1,)](
            prefix_output,
            prefix_lse,
            suffix_output,
            suffix_lse,
            output,
            output_lse,
            HEAD=head_dim,
            BLOCK=triton.next_power_of_2(head_dim),
            num_warps=4,
        )
        return output, output_lse
    if rows >= 8192:
        block = 4096 if head_dim <= 64 else 8192 if head_dim <= 128 else 16384
        scale_block = 1024 if head_dim <= 64 else 256
    elif rows >= 1024:
        block = 8192 if head_dim >= 256 else 4096
        scale_block = 256
    elif rows >= 128:
        block = 1024 if head_dim >= 256 else 8192 if head_dim >= 128 else 4096
        scale_block = 256
    else:
        block = 4096
        scale_block = 256
    _scale_kernel[(triton.cdiv(rows, scale_block),)](
        prefix_lse,
        suffix_lse,
        output,
        output_lse,
        rows,
        HEAD=head_dim,
        BLOCK=scale_block,
        num_warps=4,
    )
    _output_kernel[(triton.cdiv(rows * head_dim, block),)](
        prefix_output,
        suffix_output,
        output,
        output,
        rows * head_dim,
        HEAD=head_dim,
        BLOCK=block,
        num_warps=8 if block >= 4096 else 4,
    )
    return output, output_lse


__all__ = ["merge_state"]
