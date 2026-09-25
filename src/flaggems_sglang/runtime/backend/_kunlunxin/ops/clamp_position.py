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

"""clamp_position -- Kunlun XPU specialization.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _clamp_position_kernel(
    out_ptr,
    in_ptr,
    IN_STRIDE: tl.constexpr,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(in_ptr + offs * IN_STRIDE, mask=mask, other=1)
    tl.store(out_ptr + offs, tl.maximum(x - 1, 0), mask=mask)


def _block(n):
    if n >= 4096:
        return 4096
    size = 64
    while size < n:
        size = size * 2
    return size


def clamp_position(seq_lens):
    out = torch.empty(
        seq_lens.shape, dtype=seq_lens.dtype, device=seq_lens.device
    )
    n = seq_lens.numel()
    block = _block(n)
    grid = (n + block - 1) // block
    if grid < 1:
        grid = 1
    _clamp_position_kernel[(grid,)](
        out,
        seq_lens,
        IN_STRIDE=seq_lens.stride(0),
        N=n,
        BLOCK=block,
        num_warps=4,
        num_stages=1,
    )
    return out


__all__ = ["clamp_position"]
