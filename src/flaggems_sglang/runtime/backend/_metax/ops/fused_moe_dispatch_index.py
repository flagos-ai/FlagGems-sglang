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

"""moe/fused_moe_dispatch_index: contention-free Triton dispatch index builder.

Profiling the previous single-pass version showed the whole cost of large
shapes sits in ``BLOCK_SIZE * grid`` atomics funnelling onto the same 32
``masked_m`` counters, and the cost of small shapes sits in the
``torch.zeros`` memset launch. Both are removed:

- small inputs run as one CTA in one launch: the kernel zeroes the counters
  itself (``atomic_xchg`` + CTA barrier, so no separate memset kernel) and
  then dispatches with plain atomics;
- large inputs split into two contention-free launches: K1 gives every CTA
  a *private* counter row in a scratch workspace and histograms its shard
  without any cross-CTA traffic; K2 re-derives each CTA's exclusive
  per-expert base from those rows, pre-loads the base into the CTA's
  private row (so the atomic rank already includes it -- no per-element
  gather), and scatters. ``masked_m`` falls out of K2's row totals, so
  every buffer is a plain ``torch.empty``.

Expert counts too wide for the private-row layout (register budget) keep
the previous single-pass atomic-cursor kernel.
"""

import torch
import triton
import triton.language as tl

# Inputs up to this size are handled by a single CTA in a single launch.
_SINGLE_CTA_MAX = 2048
# Elements per CTA (and per inner loop iteration) once the input is split.
_MULTI_BLOCK = 512
_ELEMS_PER_CTA = 1024
# Private counter rows live in registers/smem sized NE_POW2; wider expert
# counts fall back to the single-pass atomic kernel.
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
def _dispatch_hist_kernel(
    topk_ids_ptr,
    ws_ptr,
    n,
    num_experts,
    elems_per_cta,
    NE_POW2: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Per-CTA expert histogram into a private row of the workspace.

    CTA ``pid`` exclusively owns row ``pid``, so it can zero that row
    itself and then bump the counters with atomics that never contend with
    another CTA. Workspace layout: ``[hist: num_ctas * num_experts]``.
    """
    pid = tl.program_id(0)
    cols = tl.arange(0, NE_POW2)
    col_mask = cols < num_experts
    my_row = ws_ptr + pid * num_experts
    tl.atomic_xchg(my_row + cols, 0, mask=col_mask)
    tl.debug_barrier()
    start = pid * elems_per_cta
    for i in range(0, elems_per_cta, BLOCK):
        offs = start + i + tl.arange(0, BLOCK)
        mask = offs < n
        e = tl.load(topk_ids_ptr + offs, mask=mask, other=-1)
        valid = (e >= 0) & (e < num_experts)
        e_safe = tl.where(valid, e, 0)
        tl.atomic_add(my_row + e_safe, 1, mask=valid)


@triton.jit
def _dispatch_scatter_kernel(
    topk_ids_ptr,
    ws_ptr,
    masked_m_ptr,
    src2dst_ptr,
    n,
    m_max,
    num_experts,
    num_ctas,
    elems_per_cta,
    NE_POW2: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Places every element at ``expert * m_max + base + local rank``.

    Each CTA sums the histogram rows before it into its exclusive base
    (and, on CTA 0, the full totals into ``masked_m``), pre-loads the base
    into its own scratch row, and recovers each element's bucket offset
    with private atomics -- the returned rank already includes the base.
    Workspace layout: ``[hist: nc*ne][scratch: nc*ne]``; the scratch row is
    initialized here, so it needs no zeroing pass. Padding ids (<0) write
    dst 0.
    """
    pid = tl.program_id(0)
    cols = tl.arange(0, NE_POW2)
    col_mask = cols < num_experts
    base = tl.zeros([NE_POW2], dtype=tl.int32)
    total = tl.zeros([NE_POW2], dtype=tl.int32)
    for r in range(0, num_ctas):
        row = tl.load(ws_ptr + r * num_experts + cols, mask=col_mask, other=0)
        base += tl.where(r < pid, row, 0)
        total += row
    if pid == 0:
        tl.store(masked_m_ptr + cols, total, mask=col_mask)
    my_row = ws_ptr + (num_ctas + pid) * num_experts
    tl.atomic_xchg(my_row + cols, base, mask=col_mask)
    tl.debug_barrier()
    start = pid * elems_per_cta
    for i in range(0, elems_per_cta, BLOCK):
        offs = start + i + tl.arange(0, BLOCK)
        mask = offs < n
        e = tl.load(topk_ids_ptr + offs, mask=mask, other=-1)
        valid = (e >= 0) & (e < num_experts)
        e_safe = tl.where(valid, e, 0)
        rank = tl.atomic_add(my_row + e_safe, 1, mask=valid)
        dst = tl.where(valid, e_safe * m_max + rank, 0)
        tl.store(src2dst_ptr + offs, dst, mask=mask)


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
        return masked_m, src2dst

    ne_pow2 = triton.next_power_of_2(ne)
    if ne_pow2 > _MAX_PRIVATE_NE_POW2:
        # Private counter rows too wide: single-pass atomic kernel, which
        # needs zeroed counters.
        masked_m.zero_()
        _launch_fallback(flat, masked_m, src2dst, n, m_max)
        return masked_m, src2dst

    if n <= _SINGLE_CTA_MAX:
        block = max(triton.next_power_of_2(n), 16)
        # Sweep (BI-V150): 8 warps beats 4/1 from block=512 up; tiny blocks
        # stay on 1 warp to avoid idle lanes.
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

    num_ctas = triton.cdiv(n, _ELEMS_PER_CTA)
    ws = torch.empty(2 * num_ctas * ne, dtype=torch.int32, device=device)
    _dispatch_hist_kernel[(num_ctas,)](
        flat,
        ws,
        n,
        ne,
        _ELEMS_PER_CTA,
        NE_POW2=ne_pow2,
        BLOCK=_MULTI_BLOCK,
        num_warps=8,
    )
    _dispatch_scatter_kernel[(num_ctas,)](
        flat,
        ws,
        masked_m,
        src2dst,
        n,
        m_max,
        ne,
        num_ctas,
        _ELEMS_PER_CTA,
        NE_POW2=ne_pow2,
        BLOCK=_MULTI_BLOCK,
        num_warps=8,
    )
    return masked_m, src2dst


__all__ = ["fused_moe_dispatch_index"]
