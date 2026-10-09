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

"""Operator: moe/compute_src2dst.

Inverts a routing permutation: given reorder_ids (the stable-argsort
permutation of flattened topk_ids), produces src2dst with
src2dst[reorder_ids[d]] = d, in one fused scatter kernel.
"""

import torch
import triton
import triton.language as tl

# Flat length above which the streaming load hint and the narrow block pay off.
_STREAM_MIN = 1 << 17
_STREAM_BLOCK = 128
_DEFAULT_BLOCK = 256
_NUM_WARPS = 4


@triton.jit
def _src2dst_kernel(
    ids_ptr,
    out_ptr,
    N,
    STRIDE_0: tl.constexpr,
    BLOCK: tl.constexpr,
    LOAD_CACHE: tl.constexpr,
    EVEN: tl.constexpr,
):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    # int64 ids: view the buffer as int32 and skip every other word, so only
    # the low word of each element is read (the ids are counts well inside
    # int32 range). The stride doubles because the pointer is now int32-wide.
    if ids_ptr.dtype.element_ty == tl.int64:
        ptrs = ids_ptr.to(tl.pointer_type(tl.int32)) + offs * (2 * STRIDE_0)
    else:
        ptrs = ids_ptr + offs * STRIDE_0
    if EVEN:
        # N % BLOCK == 0: every offset is in-bounds, skip the mask entirely.
        idx = tl.load(ptrs, cache_modifier=LOAD_CACHE).to(tl.int32)
        tl.store(out_ptr + idx, offs)
    else:
        mask = offs < N
        idx = tl.load(ptrs, mask=mask, other=0, cache_modifier=LOAD_CACHE).to(
            tl.int32
        )
        tl.store(out_ptr + idx, offs, mask=mask)


def compute_src2dst(reorder_ids, num_toks):
    """Build ``src2dst`` from the routing permutation ``reorder_ids``.

    ``reorder_ids``: flat [num_toks] permutation of ``d`` values, read through
    its own strides (no host-side contiguous copy). The output is int32.
    """
    out = torch.empty_like(reorder_ids, dtype=torch.int32)

    if num_toks >= _STREAM_MIN:
        block, load_cache = _STREAM_BLOCK, ".cg"
    else:
        block, load_cache = _DEFAULT_BLOCK, ""
    grid = (triton.cdiv(num_toks, block),)
    _src2dst_kernel[grid](
        reorder_ids,
        out,
        num_toks,
        STRIDE_0=reorder_ids.stride(0),
        BLOCK=block,
        LOAD_CACHE=load_cache,
        EVEN=(num_toks % block == 0),
        num_warps=_NUM_WARPS,
    )
    return out


__all__ = ["compute_src2dst"]
