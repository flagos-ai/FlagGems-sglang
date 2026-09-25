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

"""clamp_position -- MetaX specialization: exact (mask-free) flat pass for
block-divisible lengths, masked fallback otherwise.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _clamp_position_exact_kernel(
    seq_lens_ptr,
    out_ptr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    # Full blocks: contiguity hints let the compiler emit unmasked vectorized
    # (128-bit) loads/stores.
    offs = tl.max_contiguous(tl.multiple_of(offs, BLOCK), BLOCK)
    x = tl.load(seq_lens_ptr + offs)
    y = tl.maximum(x - 1, 0)
    tl.store(out_ptr + offs, y)


@triton.jit
def _clamp_position_masked_kernel(
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
    # Size-driven launch config (local vars only). Power-of-two lengths run
    # the mask-free exact kernel: a single program for tiny batches (smallest
    # block, one warp), 1024-wide blocks with four warps for large batches so
    # enough programs stay in flight. Other lengths take the masked fallback.
    if n <= 1024 and (n & (n - 1)) == 0:
        _clamp_position_exact_kernel[(1,)](
            seq_lens,
            out,
            BLOCK=n,
            num_warps=1,
            num_stages=1,
        )
    elif n % 1024 == 0:
        _clamp_position_exact_kernel[(n // 1024,)](
            seq_lens,
            out,
            BLOCK=1024,
            num_warps=4,
            num_stages=1,
        )
    else:
        if n <= 16:
            block, num_warps = 16, 1
        elif n <= 64:
            block, num_warps = 64, 1
        elif n <= 256:
            block, num_warps = 256, 2
        else:
            block, num_warps = 1024, 4
        _clamp_position_masked_kernel[(triton.cdiv(n, block),)](
            seq_lens,
            out,
            n,
            BLOCK=block,
            num_warps=num_warps,
            num_stages=1,
        )
    return out


__all__ = ["clamp_position"]
