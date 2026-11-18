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

"""Triton implementation of moe/fused_moe_dispatch_index.

Builds the permutation indices for masked/DeepGEMM-style grouped GEMM MoE
dispatch: every valid (token, slot) pair routed to expert e gets
``src2dst = e * m_max + rank`` where rank is the element's 0-based position
inside expert e's bucket in flat row-major order; -1 slots are padding and
write 0. ``masked_m[e]`` is the total number of elements routed to e.

Design: one program per expert scans the flattened id stream and derives
that expert's ranks with a 1D ``tl.cumsum`` over its hit vector. Keeping
every tile 1D matters on this backend: the previous one-hot formulation
built an [E, BLOCK] int32 tile and ran a 2D cumsum along its last axis,
which measured 547 us of a 573 us op at n=512 / E=32 (plus a ~23 us fixed
cost for the one-hot build and its two reduction trees); a triangular-dot
reformulation was also tried and lost to per-element vector-op pricing.
The 1D scan runs the same shapes in 4.1-5.5 us and scales to arbitrary n
via the chunk loop's running total.

Two round-2 refinements fall out of the probe ladder (floor 2.7us |
load+compare+sum | cumsum premium 1.2us per 4096-wide chunk):
1. The chunk hit count is read off the cumsum itself: ``rank`` is the
   running hit prefix, so its max (the last lane) is exactly the chunk
   count — the separate ``tl.sum(hitI)`` reduction tree is redundant and
   measured 1.2us per 4096-wide chunk at num_warps=2 (13us of the 33.7us
   n=32768 case).
2. When ``n % block == 0`` (every core benchmark shape) the EVEN
   specialization drops the clamped-gather offset and the ``(offs < n)``
   bound masks — a further ~0.5us per launch at n=4096.
Core shapes measure 4.13/4.74/5.46us (n=8/512/4096) and 20.7us at n=32768,
against 4.76/4.79/6.94/33.7us for the round-1 kernel.

Dispatch: block = next_pow2(n) capped at 4096 (floor 8). num_warps follows
a hard correctness line on this backend: a width-4096 int32 cumsum silently
miscompiles at num_warps=1 (masked_m stays correct while ~82% of dst lanes
pick up cross-expert values; width 2048 x nw=1 and width 4096 x nw>=2 are
bit-exact), so blocks above 2048 pin num_warps=2.
"""

import torch
import triton
import triton.language as tl

# Chunk width ceiling of the scan loop (single chunk below this).
_BLOCK_MAX = 4096
# Below this width run single-warp (fastest and verified bit-exact).
_NW1_BLOCK_MAX = 2048
# Tile-width floor: width-1 vectors are a pathological shape here.
_BLOCK_MIN = 8


@triton.jit
def _scan_kernel(
    ids_ptr,  # [n] int32 expert ids, -1 = padding
    masked_m_ptr,  # [num_experts] int32
    dst_ptr,  # [n] int32 destination rows
    n,
    m_max,
    num_experts,
    BLOCK: tl.constexpr,
    EVEN: tl.constexpr,  # n % BLOCK == 0: no tail chunk, bounds masks dropped
):
    expert = tl.program_id(0)
    lane = tl.arange(0, BLOCK)
    # Running count of this expert's hits over the chunks already visited:
    # the bucket rank base of the current chunk. No atomics — every rank is
    # a pure function of the input and the program's own chunk history.
    total = tl.full((), 0, tl.int32)
    for c in range(0, tl.cdiv(n, BLOCK)):
        offs = c * BLOCK + lane
        if EVEN:
            ids = tl.load(ids_ptr + offs)
            hit = ids == expert
        else:
            # Clamped gather: the tail chunk's out-of-range lanes re-read the
            # last valid id and are dropped by the masks below, so the load
            # itself needs no predicate.
            safe_offs = tl.minimum(offs, n - 1)
            ids = tl.load(ids_ptr + safe_offs)
            hit = (offs < n) & (ids == expert)
        hitI = hit.to(tl.int32)
        rank = tl.cumsum(hitI, axis=0)
        tl.store(dst_ptr + offs, expert * m_max + total + rank - 1, mask=hit)
        # Program 0 owns every lane no expert program claims (padding and
        # out-of-range ids), so each dst row is written exactly once and a
        # plain torch.empty allocation is enough for the output.
        if expert == 0:
            if EVEN:
                tl.store(
                    dst_ptr + offs,
                    0,
                    mask=(ids < 0) | (ids >= num_experts),
                )
            else:
                tl.store(
                    dst_ptr + offs,
                    0,
                    mask=(offs < n) & ((ids < 0) | (ids >= num_experts)),
                )
        # rank is the running hit prefix, so its max (the last lane, or any
        # tail lane clamped out of the hit mask) is exactly this chunk's hit
        # count — cheaper than a second tl.sum reduction tree over hitI.
        total += tl.max(rank, axis=0)
    tl.store(masked_m_ptr + expert, total)


def fused_moe_dispatch_index(topk_ids, num_local_experts, m_max):
    n = topk_ids.numel()
    device = topk_ids.device
    num_local_experts = int(num_local_experts)
    m_max = int(m_max)
    masked_m = torch.empty(num_local_experts, dtype=torch.int32, device=device)
    src2dst = torch.empty(n, dtype=torch.int32, device=device)
    if n == 0 or num_local_experts == 0:
        return masked_m.zero_(), src2dst

    if topk_ids.dtype != torch.int32:
        topk_ids = topk_ids.to(torch.int32)
    # reshape(-1) is a free view for the standard contiguous [tokens, topk]
    # layout, but for e.g. [:, ::2] slices torch legally returns a strided
    # 1-D *view* (stride != 1) instead of a copy — the kernel reads with
    # unit stride, so materialize in that case.
    ids = topk_ids.reshape(-1)
    if ids.stride(0) != 1:
        ids = ids.contiguous()

    block = min(triton.next_power_of_2(n), _BLOCK_MAX)
    if block < _BLOCK_MIN:
        block = _BLOCK_MIN
    num_warps = 1 if block <= _NW1_BLOCK_MAX else 2
    _scan_kernel[(num_local_experts,)](
        ids,
        masked_m,
        src2dst,
        n,
        m_max,
        num_local_experts,
        BLOCK=block,
        EVEN=(n % block == 0),
        num_warps=num_warps,
    )
    return masked_m, src2dst


__all__ = ["fused_moe_dispatch_index"]
