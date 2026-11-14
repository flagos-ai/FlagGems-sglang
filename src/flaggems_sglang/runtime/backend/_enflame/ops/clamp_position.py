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

out = (seq_lens - 1).clamp(min=0), flat elementwise pass over int32 words.

int64 inputs: while TORCH_GCU_ENABLE_INT64_AND_UINT64 is off, torch_gcu
stores int64 tensors as physical 4-byte words (values < 2**31, i.e. any
real sequence length, are represented exactly). The kernel then runs over
zero-copy int32 word views of input and output -- a single kernel launch
with no staging copies. With the env on (true 8-byte int64) or for
non-contiguous inputs, a value-preserving cast staging path is used
instead (exact for seq_lens < 2**31, non-negative; output keeps the input
dtype).
"""

import os

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


def _gcu_int64_enabled():
    return os.environ.get("TORCH_GCU_ENABLE_INT64_AND_UINT64", "").lower() not in (
        "",
        "0",
        "false",
        "off",
        "no",
    )


def _int32_word_view(t):
    # Same-shape/same-stride int32 view of an int64 tensor's physical
    # words (valid while int64 is physically 4B and t is contiguous).
    view = torch.empty(0, dtype=torch.int32, device=t.device)
    view.set_(
        t.untyped_storage(),
        t.storage_offset(),
        t.shape,
        t.stride(),
    )
    return view


def clamp_position(seq_lens):
    n = seq_lens.numel()
    if seq_lens.dtype == torch.int64:
        if not _gcu_int64_enabled() and seq_lens.is_contiguous():
            # Zero-copy word views: single kernel launch, in-place into
            # the int64 output; values are exact for seq_lens (< 2**31,
            # non-negative) and independent of any backend storage layout.
            out = torch.empty_like(seq_lens)
            if n == 0:
                return out
            _launch(_int32_word_view(seq_lens), _int32_word_view(out), n)
            return out
        # True 8-byte int64 (env on) or non-contiguous input:
        # value-preserving narrow/widen staging.
        src = seq_lens.to(torch.int32)
        mid = torch.empty_like(src)
        if n == 0:
            return mid.to(seq_lens.dtype)
        _launch(src, mid, n)
        return mid.to(seq_lens.dtype)
    out = torch.empty_like(seq_lens)
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
