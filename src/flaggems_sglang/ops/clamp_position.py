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

"""Operator: attention/clamp_position.

Decode-step position computation: out = (seq_lens - 1).clamp(min=0),
one flat pass, launch config chosen from the tensor length; works for
int32 and int64 (output keeps the input dtype).
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _clamp_position_kernel(
    seq_lens_ptr,
    out_ptr,
    n_elements,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    # Contiguity hints let the compiler emit vectorized (128-bit) loads/stores.
    offs = tl.max_contiguous(tl.multiple_of(offs, BLOCK), BLOCK)
    mask = offs < n_elements
    x = tl.load(seq_lens_ptr + offs, mask=mask, other=1)
    # (x - 1).clamp(min=0): with other=1 masked lanes become 0 and stay 0.
    y = tl.maximum(x - 1, 0)
    tl.store(out_ptr + offs, y, mask=mask)


def clamp_position(seq_lens):
    n = seq_lens.numel()
    out = torch.empty_like(seq_lens)
    if n == 0:
        return out
    # Size-driven launch config (local vars only): small tensors get a
    # single program with the smallest BLOCK that covers them and one warp;
    # large tensors keep BLOCK=1024 so enough programs run concurrently.
    if n <= 16:
        block, num_warps = 16, 1
    elif n <= 64:
        block, num_warps = 64, 1
    elif n <= 256:
        block, num_warps = 256, 2
    else:
        block, num_warps = 1024, 4
    grid = (triton.cdiv(n, block),)
    _clamp_position_kernel[grid](
        seq_lens,
        out,
        n,
        BLOCK=block,
        num_warps=num_warps,
        num_stages=1,
    )
    return out


__all__ = ["clamp_position"]
