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

"""fused_moe_dispatch_index: build the permutation grouped-GEMM destination
indices for masked/DeepGEMM MoE dispatch.
"""

import torch
import triton
import triton.language as tl

_MAX_BLOCK = 4096
_FILL = 4096


@triton.jit
def _single_kernel(
    topk_ids_ptr,  # [n] int32 flat view of [num_tokens, top_k]
    masked_m_ptr,  # [num_local_experts] int32 output counts
    src2dst_ptr,  # [n] int32 destination row per slot
    n_elements,
    m_max,
    E_POW: tl.constexpr,
    BLOCK: tl.constexpr,
    FILL: tl.constexpr,
):
    # One program does everything: histogram + exclusive bucket bases + fill.
    # The input vector is loaded once; each static expert iteration reuses it.
    offs = tl.arange(0, BLOCK)
    expert = tl.load(topk_ids_ptr + offs, mask=offs < n_elements, other=-1)
    valid = expert >= 0
    base = 0  # exclusive flat start of the current expert's bucket
    for e in tl.static_range(E_POW):
        cnt = tl.sum((valid & (expert == e)).to(tl.int32), axis=0)
        tl.store(masked_m_ptr + e, cnt)
        for off in range(0, cnt, FILL):
            o = off + tl.arange(0, FILL)
            tl.store(src2dst_ptr + base + o, e * m_max + o, mask=o < cnt)
        base = base + cnt


@triton.jit
def _histogram_kernel(
    topk_ids_ptr,  # [n] int32 flat view of [num_tokens, top_k]
    partial_ptr,  # [nblocks * E_POW] int32 per-chunk per-expert counts
    n_elements,
    E_POW: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    expert = tl.load(topk_ids_ptr + offs, mask=offs < n_elements, other=-1)
    valid = expert >= 0
    row = partial_ptr + pid * E_POW
    for e in tl.static_range(E_POW):
        cnt = tl.sum((valid & (expert == e)).to(tl.int32), axis=0)
        tl.store(row + e, cnt)


@triton.jit
def _totals_fill_kernel(
    partial_ptr,  # [nblocks * E_POW] int32 per-chunk per-expert counts
    masked_m_ptr,  # [num_local_experts] int32 output counts
    src2dst_ptr,  # [n] int32 destination row per slot
    nblocks,
    n_experts,
    m_max,
    E_POW: tl.constexpr,
    CHUNKS: tl.constexpr,
    FILL: tl.constexpr,
):
    # One program per expert.  Every program re-walks the (tiny) partial table
    # to rebuild the grand totals and their exclusive cumsum, then keeps only
    # its own entries.  All programs store identical masked_m values, so the
    # redundant stores are benign.
    #
    # Both loops are ``static_range`` (fully unrolled) rather than runtime
    # ``range``: on the XPU backend a runtime loop body inside a grid>1 kernel
    # carries so much per-iteration overhead that the device cost grows with
    # grid size, which made this kernel the dominant cost of the large path.
    # ``CHUNKS`` bounds both loops -- the scan needs one iteration per partial
    # row (``nblocks <= CHUNKS``) and the fill needs ``CHUNKS * FILL >= n``.
    e = tl.program_id(0)
    e_idx = tl.arange(0, E_POW)
    tot = tl.zeros([E_POW], dtype=tl.int32)
    for p in tl.static_range(CHUNKS):
        tot += tl.load(
            partial_ptr + p * E_POW + e_idx, mask=p < nblocks, other=0
        )
    # Exclusive cumsum of the totals = flat start row of each bucket.
    eb = tl.cumsum(tot, axis=0) - tot
    mycnt = tl.sum(tl.where(e_idx == e, tot, 0), axis=0)
    mybase = tl.sum(tl.where(e_idx == e, eb, 0), axis=0)
    # Scalar store of this program's own count.  A vector store of the whole
    # ``tot`` here would corrupt the scalar ``mycnt``/``mybase`` extraction
    # below on the XPU backend, so write only one element per program.
    tl.store(masked_m_ptr + e, mycnt)
    val0 = e * m_max
    for k in tl.static_range(CHUNKS):
        o = k * FILL + tl.arange(0, FILL)
        tl.store(src2dst_ptr + mybase + o, val0 + o, mask=o < mycnt)


def fused_moe_dispatch_index(topk_ids, num_local_experts, m_max):
    flat = topk_ids.reshape(-1)
    n = flat.numel()
    device = flat.device
    masked_m = torch.empty(num_local_experts, dtype=torch.int32, device=device)
    src2dst = torch.empty(n, dtype=torch.int32, device=device)

    if n == 0:
        masked_m.zero_()
        return masked_m, src2dst

    e_pow = triton.next_power_of_2(num_local_experts)
    if n <= _MAX_BLOCK:
        # Single launch: one program covers the whole input.
        block = max(256, triton.next_power_of_2(n))
        _single_kernel[(1,)](
            flat,
            masked_m,
            src2dst,
            n,
            m_max,
            E_POW=e_pow,
            BLOCK=block,
            FILL=512,
            num_warps=4,
        )
        return masked_m, src2dst

    # Two launches: parallel histogram, then per-expert totals + fill.
    # ``chunks`` bounds the fill kernel's unrolled scan (>= nblocks) and its
    # unrolled fill (chunks * FILL >= n), so one constexpr serves both.
    block = _MAX_BLOCK
    nblocks = triton.cdiv(n, block)
    chunks = max(triton.next_power_of_2(nblocks), triton.cdiv(n, _FILL))
    partial = torch.empty(nblocks * e_pow, dtype=torch.int32, device=device)
    _histogram_kernel[(nblocks,)](
        flat,
        partial,
        n,
        E_POW=e_pow,
        BLOCK=block,
        num_warps=8,
    )
    _totals_fill_kernel[(num_local_experts,)](
        partial,
        masked_m,
        src2dst,
        nblocks,
        num_local_experts,
        m_max,
        E_POW=e_pow,
        CHUNKS=chunks,
        FILL=_FILL,
        num_warps=4,
    )
    return masked_m, src2dst


__all__ = ["fused_moe_dispatch_index"]
