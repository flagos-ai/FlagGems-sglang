# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Operator: moe/fused_moe_dispatch_index.

Builds permuted grouped-GEMM destination indices for masked/DeepGEMM MoE
dispatch: every valid (token, slot) pair routed to expert e gets
src2dst = e * m_max + rank; -1 slots are padding and write 0.
"""

import torch
import triton
import triton.language as tl

# Flat id count above which the sort path runs instead of the per-expert scan.
_RANK_SORT_MIN = 4096
# Tile cap of the scan path's chunk (a power of two no smaller than the id
# count it is chosen for, so the scan always walks a single chunk).
_SCAN_BLOCK_MAX = 4096
# Tile of the sort path: one program per chunk, 2 warps.
_SORT_BLOCK = 128
_SORT_WARPS = 2
_SORT_STAGES = 3
_SCAN_WARPS = 8


@triton.jit
def _moe_dispatch_scan_kernel(
    ids_ptr,
    masked_m_ptr,
    dst_ptr,
    n_elements,
    topk,
    stride_0,
    stride_1,
    m_max,
    BLOCK: tl.constexpr,
    E: tl.constexpr,
):
    expert = tl.program_id(axis=0)
    lane = tl.arange(0, BLOCK)
    # Running count of this expert's hits over the chunks already visited:
    # the bucket rank base of the current chunk. No atomics — every rank is a
    # pure function of the input and the program's own chunk history.
    total = tl.full((), 0, tl.int32)
    for c in range(tl.cdiv(n_elements, BLOCK)):
        offs = c * BLOCK + lane
        # Clamped gather: the tail chunk's out-of-range lanes re-read the last
        # valid id and are dropped by the masks below, so the load itself
        # needs no predicate.
        safe_offs = tl.minimum(offs, n_elements - 1)
        ids = tl.load(
            ids_ptr
            + safe_offs // topk * stride_0
            + safe_offs % topk * stride_1
        )
        hit = (offs < n_elements) & (ids == expert)
        # Rank of each hit inside expert's bucket = pre-chunk total + prefix.
        rank = tl.cumsum(hit.to(tl.int32), axis=0)
        tl.store(dst_ptr + offs, expert * m_max + total + rank - 1, mask=hit)
        # Program e == 0 owns the padding lanes, so every dst row is written
        # and a plain torch.empty allocation is enough for the output.
        tl.store(
            dst_ptr + offs,
            0,
            mask=(offs < n_elements) & (ids < 0) & (expert == 0),
        )
        total += tl.sum(hit.to(tl.int32), axis=0)
    tl.store(masked_m_ptr + expert, total)


@triton.jit
def _moe_dispatch_sort_kernel(
    ids_ptr,
    masked_m_ptr,
    dst_ptr,
    n_elements,
    topk,
    stride_0,
    stride_1,
    m_max,
    BLOCK: tl.constexpr,
    E: tl.constexpr,
    EPOW2: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    lane = tl.arange(0, BLOCK)
    offs = pid * BLOCK + lane
    # Padding (-1) is folded into the sentinel bucket E, which no real expert
    # maps to, so it sorts to the end and is stored as 0 below.
    ids = tl.load(
        ids_ptr + offs // topk * stride_0 + offs % topk * stride_1,
        mask=offs < n_elements,
        other=-1,
    )
    bucket = tl.where(ids >= 0, ids, E)
    # Sort the (bucket, lane) pairs: the lane index rides in the low bits so
    # in-bucket order follows input order, and the sorted lane holds its rank.
    srt = tl.sort(bucket * BLOCK + lane)
    pe = srt // BLOCK
    hist = tl.histogram(bucket, EPOW2)
    e_range = tl.arange(0, EPOW2)
    # Per-expert cursor claimed from the global counter in one vectorized
    # atomic; the sentinel bucket E is excluded so padding never consumes a
    # real expert's ranks.
    base = tl.atomic_add(
        masked_m_ptr + e_range, hist, mask=e_range < E, sem="relaxed"
    )
    # Bucket rank = sorted position - this bucket's in-chunk prefix, offset by
    # the cursor value claimed above (reduction over the block's own history).
    rank = lane - tl.gather(tl.cumsum(hist, axis=0) - hist - base, pe, 0)
    pos = pid * BLOCK + srt % BLOCK
    tl.store(
        dst_ptr + pos,
        tl.where(pe < E, pe * m_max + rank, 0),
        mask=pos < n_elements,
    )


def fused_moe_dispatch_index(topk_ids, num_local_experts, m_max):
    """Build ``(masked_m, src2dst)`` for one MoE dispatch.

    ``topk_ids``: [num_tokens, topk] int32 expert ids (``-1`` = padding);
    ranks are assigned per expert bucket in row-major order.
    """
    n_elements = topk_ids.numel()
    topk = topk_ids.shape[1]
    stride_0 = topk_ids.stride(0)
    stride_1 = topk_ids.stride(1)

    dst = torch.empty((n_elements,), dtype=torch.int32, device=topk_ids.device)

    if n_elements > _RANK_SORT_MIN:
        # masked_m must start zeroed: it holds the atomic cursors.
        masked_m = torch.zeros(
            (num_local_experts,), dtype=torch.int32, device=topk_ids.device
        )
        # Strictly above the expert count so the padding sentinel bucket is
        # representable and never collides with a real expert.
        epow2 = 1 << num_local_experts.bit_length()
        _moe_dispatch_sort_kernel[(triton.cdiv(n_elements, _SORT_BLOCK),)](
            topk_ids,
            masked_m,
            dst,
            n_elements,
            topk,
            stride_0,
            stride_1,
            m_max,
            BLOCK=_SORT_BLOCK,
            E=num_local_experts,
            EPOW2=epow2,
            num_warps=_SORT_WARPS,
            num_stages=_SORT_STAGES,
        )
    else:
        # masked_m is written outright by the scan kernels, so it needs no
        # pre-zeroing.
        masked_m = torch.empty(
            (num_local_experts,), dtype=torch.int32, device=topk_ids.device
        )
        block = min(_SCAN_BLOCK_MAX, 1 << (n_elements - 1).bit_length())
        _moe_dispatch_scan_kernel[(num_local_experts,)](
            topk_ids,
            masked_m,
            dst,
            n_elements,
            topk,
            stride_0,
            stride_1,
            m_max,
            BLOCK=block,
            E=num_local_experts,
            num_warps=_SCAN_WARPS,
        )
    return masked_m, dst


__all__ = ["fused_moe_dispatch_index"]
