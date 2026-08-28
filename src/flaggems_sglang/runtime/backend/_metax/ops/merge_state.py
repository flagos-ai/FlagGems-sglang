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
def _merge_state_kernel(
    po,
    pl,
    so,
    sl,
    out,
    out_lse,
    total,
    HEAD: tl.constexpr,
    BLOCK: tl.constexpr,
    OPT: tl.constexpr,
    FAST: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < total
    rows = offsets // HEAD
    cols = offsets - rows * HEAD
    p_lse = tl.load(pl + rows, mask=mask, other=-math.inf).to(tl.float32)
    s_lse = tl.load(sl + rows, mask=mask, other=-math.inf).to(tl.float32)
    p_lse = tl.where(p_lse == math.inf, -math.inf, p_lse)
    s_lse = tl.where(s_lse == math.inf, -math.inf, s_lse)
    p_high = p_lse >= s_lse
    max_lse = tl.maximum(p_lse, s_lse)
    delta = tl.minimum(p_lse, s_lse) - max_lse
    if FAST:
        low_se = tl.exp2(delta * 1.4426950408889634)
    else:
        low_se = tl.exp(delta)
    inv = 1.0 / (1.0 + low_se)
    low_scale = low_se * inv
    if OPT:
        t = low_se / (2.0 + low_se)
        t2 = t * t
        log_se = (
            2.0
            * t
            * (
                1.0
                + t2
                * (0.3333333333333333 + t2 * (0.2 + t2 * 0.14285714285714285))
            )
        )
    else:
        log_se = tl.log(1.0 + low_se)
    p = tl.load(po + offsets, mask=mask, other=0.0).to(tl.float32)
    s = tl.load(so + offsets, mask=mask, other=0.0).to(tl.float32)
    if OPT:
        high = tl.where(p_high, p, s)
        low = tl.where(p_high, s, p)
        merged = high + (low - high) * low_scale
    else:
        merged = p * tl.where(p_high, inv, low_scale) + s * tl.where(
            p_high, low_scale, inv
        )
    tl.store(out + offsets, merged, mask=mask)
    tl.store(out_lse + rows, log_se + max_lse, mask=mask & (cols == 0))


def merge_state(prefix_output, prefix_lse, suffix_output, suffix_lse):
    output = torch.empty_like(prefix_output)
    output_lse = torch.empty_like(prefix_lse)
    head_dim = prefix_output.shape[-1]
    rows = prefix_output.numel() // head_dim
    total = rows * head_dim
    if rows >= 8192:
        block = 16384 if head_dim <= 64 else 4096
    elif rows >= 1024 and head_dim >= 256:
        block = 4096
    else:
        block = 1024
    optimize_math = (
        rows <= 16 or rows >= 8192 or (rows == 1024 and head_dim <= 128)
    )
    fast_exp = rows <= 16 or rows >= 8192
    _merge_state_kernel[(triton.cdiv(total, block),)](
        prefix_output,
        prefix_lse,
        suffix_output,
        suffix_lse,
        output,
        output_lse,
        total,
        HEAD=head_dim,
        BLOCK=block,
        OPT=optimize_math,
        FAST=fast_exp,
        num_warps=4 if block == 1024 else 8,
    )
    return output, output_lse


reference = merge_state

__all__ = ["merge_state"]
