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

"""Single-launch segmented dispatch index builder for moe/fused_moe_dispatch_index.
"""

import torch
import triton
import triton.language as tl

# Inputs up to this size are handled by a single CTA in a single launch.
_SINGLE_CTA_MAX = 2048
# Inputs up to this size run the single-launch expert-scan kernel; beyond it
# the O(n * ne) scan traffic beats the launch saving and the single-launch
# segmented range-allocator kernel takes over.
_SCAN_MAX_N = 4096
# Elements per CTA for the segmented kernel (one block, ranks stay in
# registers — no scratch round trip).
_SEG_ELEMS_PER_CTA = 512
# Private counter rows live in registers sized NE_POW2; wider expert counts
# fall back to the single-pass atomic kernel.
_MAX_PRIVATE_NE_POW2 = 128


@triton.jit
def _dispatch_atomic_fallback_kernel(
    topk_ids_ptr,
    masked_m_ptr,
    src2dst_ptr,
    n_elements,
    m_max,
    BLOCK: tl.constexpr,
):
    """Single-pass atomic-cursor kernel for expert counts too wide for the
    private-row scheme (masked_m must be pre-zeroed by the caller)."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    e = tl.load(topk_ids_ptr + offs, mask=mask, other=-1)
    valid = e >= 0
    # Clamp padding lanes to expert 0 so the pointer arithmetic stays in
    # bounds; the atomic itself is masked off so their count is untouched.
    e_safe = tl.where(valid, e, 0)
    slot = tl.atomic_add(masked_m_ptr + e_safe, 1, mask=valid)
    dst = tl.where(valid, e_safe * m_max + slot, 0)
    tl.store(src2dst_ptr + offs, dst, mask=mask)


@triton.jit
def _dispatch_single_cta_kernel(
    topk_ids_ptr,
    masked_m_ptr,
    src2dst_ptr,
    n,
    m_max,
    num_experts,
    NE_POW2: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """One CTA builds both outputs in one launch: zero the counters with
    atomic RMW (ordered at L2, visible to the whole CTA after the barrier),
    then dispatch. No memset launch, no ``torch.zeros``."""
    cols = tl.arange(0, NE_POW2)
    col_mask = cols < num_experts
    tl.atomic_xchg(masked_m_ptr + cols, 0, mask=col_mask)
    tl.debug_barrier()
    offs = tl.arange(0, BLOCK)
    mask = offs < n
    e = tl.load(topk_ids_ptr + offs, mask=mask, other=-1)
    valid = (e >= 0) & (e < num_experts)
    e_safe = tl.where(valid, e, 0)
    slot = tl.atomic_add(masked_m_ptr + e_safe, 1, mask=valid)
    dst = tl.where(valid, e_safe * m_max + slot, 0)
    tl.store(src2dst_ptr + offs, dst, mask=mask)


@triton.jit
def _dispatch_expert_scan_kernel(
    topk_ids_ptr,
    masked_m_ptr,
    src2dst_ptr,
    n,
    m_max,
    num_experts,
    BLOCK: tl.constexpr,
    NSTAGE: tl.constexpr,
):
    """One launch, no atomics: CTA ``e`` owns expert ``e`` and scans the
    whole flat id array, CTA ``num_experts`` owns invalid ids.

    Expert CTA: per block, ``match = (id == e)``; the match's inclusive
    cumsum minus one is the element's 0-based bucket rank *within this
    block*, and a loop-carried scalar adds the block's running match count,
    so ``e * m_max + pos_base + rank`` enumerates the bucket in flat order.
    All state except the scalar base lives in registers — no atomics, no
    scratch, no barrier. The invalid CTA writes ``dst = 0`` for padding ids
    so every slot is covered exactly once.
    """
    pid = tl.program_id(0)
    is_expert_cta = pid < num_experts
    pos_base = tl.zeros([1], dtype=tl.int32)
    for i in tl.range(0, n, BLOCK, num_stages=NSTAGE):
        offs = i + tl.arange(0, BLOCK)
        mask = offs < n
        e = tl.load(topk_ids_ptr + offs, mask=mask, other=-2)
        if is_expert_cta:
            match = e == pid
            m32 = match.to(tl.int32)
            rank = tl.cumsum(m32, axis=0) - 1
            pos = pos_base + rank
            dst = pid * m_max + pos
            tl.store(src2dst_ptr + offs, dst, mask=mask & match)
            pos_base += tl.sum(m32)
        else:
            match = (e < 0) | (e >= num_experts)
            zeros = tl.zeros([BLOCK], dtype=tl.int32)
            tl.store(src2dst_ptr + offs, zeros, mask=mask & match)
    if is_expert_cta:
        tl.store(masked_m_ptr + pid + tl.arange(0, 1), pos_base)


@triton.jit
def _dispatch_segmented_kernel(
    topk_ids_ptr,
    masked_m_ptr,
    src2dst_ptr,
    ws_ptr,
    n,
    m_max,
    num_experts,
    elems_per_cta,
    NE_POW2: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """One launch, no scratch round trip: segmented private-rank + atomic
    range allocator.

    CTA ``pid`` owns ``[pid * epc, min(n, (pid + 1) * epc))`` and row
    ``pid`` of the workspace. Sweep: zero the private row (plain store +
    CTA barrier, so the zero is ordered before every lane's first rank
    atomic), then each element takes its bucket rank via an uncontended
    ``atomic_add`` on the CTA's own row. The rank, the id and the valid bit
    stay in registers.

    After a CTA barrier (orders every rank atomic before the read-back), the
    CTA-local counts are recovered with a *plain* ``tl.load`` — the barrier
    already orders the row's atomics before the read, so a read-modify-write
    ``atomic_xchg`` would buy nothing — and one
    ``atomic_add(masked_m + e, cnt[e])`` per expert grants the CTA a
    *disjoint* destination range per bucket — the return value is the range
    start. Ranges partition ``[0, masked_m[e])`` regardless of arrival
    order, so ``base + rank`` is a bijection into the bucket and
    ``masked_m`` ends up holding the exact totals. A final plain store
    re-zeroes the private row so the workspace stays self-cleaning across
    launches (the store retires after every lane's rank atomic, which is all
    the row's next-launch zero needs). No cross-CTA wait, no packed scratch,
    no prefix scan: each ``src2dst`` slot is written exactly once from
    registers.
    """
    pid = tl.program_id(0)
    cols = tl.arange(0, NE_POW2)
    my_row = ws_ptr + pid.to(tl.int64) * num_experts
    tl.store(
        my_row + cols,
        tl.zeros([NE_POW2], dtype=tl.int32),
        mask=cols < num_experts,
    )
    tl.debug_barrier()
    end = tl.minimum(n, (pid + 1) * elems_per_cta)
    offs = pid * elems_per_cta + tl.arange(0, BLOCK)
    mask = offs < end
    e = tl.load(topk_ids_ptr + offs, mask=mask, other=-1)
    valid = (e >= 0) & (e < num_experts)
    e_safe = tl.where(valid, e, 0)
    rank = tl.atomic_add(my_row + e_safe, 1, mask=valid)
    tl.debug_barrier()
    cnt = tl.load(my_row + cols, mask=cols < num_experts, other=0)
    base = tl.atomic_add(masked_m_ptr + cols, cnt, mask=cols < num_experts)
    b = tl.gather(base, e_safe, axis=0)
    dst = tl.where(valid, e_safe * m_max + b + rank, 0)
    tl.store(src2dst_ptr + offs, dst, mask=mask)
    tl.store(
        my_row + cols,
        tl.zeros([NE_POW2], dtype=tl.int32),
        mask=cols < num_experts,
    )


@triton.jit
def _masked_m_fill_kernel(
    masked_m_ptr,
    num_experts,
    BLOCK: tl.constexpr,
):
    """Zero the allocator counters. Replaces a ``torch.zeros`` memset launch
    with a ~1.8 us Triton launch that shares the stream with the dispatch
    kernel (the allocator needs counters at zero before any CTA's range
    grant, so a separate ordered launch is the simplest portable way)."""
    offs = tl.arange(0, BLOCK)
    tl.store(
        masked_m_ptr + offs,
        tl.zeros([BLOCK], dtype=tl.int32),
        mask=offs < num_experts,
    )


def _launch_fallback(flat, masked_m, src2dst, n, m_max):
    if n <= 128:
        block, num_warps = 128, 1
    elif n <= 2048:
        block, num_warps = 512, 4
    else:
        block, num_warps = 1024, 4
    grid = (triton.cdiv(n, block),)
    _dispatch_atomic_fallback_kernel[grid](
        flat, masked_m, src2dst, n, m_max, BLOCK=block, num_warps=num_warps
    )


def fused_moe_dispatch_index(topk_ids, num_local_experts, m_max):
    flat = topk_ids.reshape(-1)
    n = flat.numel()
    ne = num_local_experts
    device = flat.device

    masked_m = torch.empty(ne, dtype=torch.int32, device=device)
    src2dst = torch.empty(n, dtype=torch.int32, device=device)

    if n == 0:
        masked_m.zero_()
        return masked_m, src2dst

    if n <= _SINGLE_CTA_MAX:
        ne_pow2 = triton.next_power_of_2(ne)
        block = max(triton.next_power_of_2(n), 16)
        # 8 warps beats 4/1 from block=512 up; tiny blocks stay on 1 warp to
        # avoid idle lanes.
        num_warps = 1 if block <= 128 else 8
        _dispatch_single_cta_kernel[(1,)](
            flat,
            masked_m,
            src2dst,
            n,
            m_max,
            ne,
            NE_POW2=ne_pow2,
            BLOCK=block,
            num_warps=num_warps,
        )
        return masked_m, src2dst

    if n <= _SCAN_MAX_N:
        # Single launch, no atomics: one CTA per expert plus one for invalid
        # ids. No workspace, no memset — every output slot is written exactly
        # once by its owning CTA.
        _dispatch_expert_scan_kernel[(ne + 1,)](
            flat,
            masked_m,
            src2dst,
            n,
            m_max,
            ne,
            BLOCK=4096,
            NSTAGE=2,
            num_warps=8,
        )
        return masked_m, src2dst

    ne_pow2 = triton.next_power_of_2(ne)
    if ne_pow2 > _MAX_PRIVATE_NE_POW2:
        # Private counter rows too wide: single-pass atomic kernel, which
        # needs zeroed counters.
        masked_m.zero_()
        _launch_fallback(flat, masked_m, src2dst, n, m_max)
        return masked_m, src2dst

    # Single-launch segmented path: private rank rows + atomic range
    # allocator. All scratch lives in one allocation (src2dst | workspace
    # rows | allocator counters); the counters are zeroed by the tiny fill
    # kernel below, so no ``torch.zeros`` memset launch is enqueued.
    num_ctas = triton.cdiv(n, _SEG_ELEMS_PER_CTA)
    buf = torch.empty(n + num_ctas * ne + ne, dtype=torch.int32, device=device)
    src2dst = buf[:n]
    ws = buf[n : n + num_ctas * ne]
    masked_m = buf[n + num_ctas * ne :]
    _masked_m_fill_kernel[(1,)](masked_m, ne, BLOCK=ne_pow2, num_warps=1)
    _dispatch_segmented_kernel[(num_ctas,)](
        flat,
        masked_m,
        src2dst,
        ws,
        n,
        m_max,
        ne,
        _SEG_ELEMS_PER_CTA,
        NE_POW2=ne_pow2,
        BLOCK=_SEG_ELEMS_PER_CTA,
        num_warps=8,
    )
    return masked_m, src2dst


__all__ = ["fused_moe_dispatch_index"]
