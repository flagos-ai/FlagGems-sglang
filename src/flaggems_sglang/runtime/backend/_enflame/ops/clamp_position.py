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

"""clamp_position -- Enflame GCU specialization.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _clamp_position_kernel(
    seq_lens_ptr,  # int32 tensor (or int32 word view of an int64 tensor)
    out_ptr,  # int32 tensor / int32 word view of the output
    n,  # number of int32 elements/words to process
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(seq_lens_ptr + offs, mask=mask, other=0)
    y = tl.maximum(x - 1, 0)
    tl.store(out_ptr + offs, y, mask=mask)


def clamp_position(seq_lens):
    if seq_lens.dtype == torch.int64:
        # Value-preserving narrow/widen: exact for seq_lens (< 2**31,
        # non-negative) and independent of any backend storage layout.
        src = seq_lens.to(torch.int32)
        mid = torch.empty_like(src)
        n = src.numel()
        if n == 0:
            return mid.to(seq_lens.dtype)
        _launch(src, mid, n)
        return mid.to(seq_lens.dtype)
    out = torch.empty_like(seq_lens)
    n = seq_lens.numel()
    if n == 0:
        return out
    _launch(seq_lens, out, n)
    return out


def _launch(src, dst, n):
    if n <= 4096:
        _clamp_position_kernel[(1,)](src, dst, n, BLOCK=4096, num_warps=4)
    elif n <= 16384:
        _clamp_position_kernel[(1,)](
            src, dst, n, BLOCK=16384, num_warps=16, num_stages=1
        )
    else:
        grid = (triton.cdiv(n, 32768),)
        _clamp_position_kernel[grid](src, dst, n, BLOCK=32768, num_warps=8)


__all__ = ["clamp_position"]
