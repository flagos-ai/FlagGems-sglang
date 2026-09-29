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

"""Operator: moe/fused_moe_dispatch_index

Builds the permuted grouped-GEMM destination indices for masked/DeepGEMM MoE
"""

import torch
import triton
import triton.language as tl

# Flat length below which the (expert, chunk) scan runs with a narrow-tile
# single-chunk launch (module-level int constant — immutable).
_EC_SMALL = 1 << 12
# Flat length above which the EC pre-pass's redundant chunk re-reads lose to
# the one-shot one-hot/atomic kernel (module-level int constant — immutable).
_EC_ELEMS_MAX = 1 << 16


@triton.jit
def _moe_dispatch_ec_kernel(
    ids_ptr,
    out_ptr,
    n_elements,
    m_max,
    e_off,
    n_chunks,
    BLOCK: tl.constexpr,
    E: tl.constexpr,
    EVEN: tl.constexpr,
):
    pid = tl.program_id(0)
    e = pid % E
    c = pid // E
    # Exact count of expert e over the full chunks [0, c): the bucket rank
    # offset of every hit in chunk c. Only full chunks are visited, so the
    # loads need no mask. No atomics anywhere — chunk c's rank base is a pure
    # function of the input.
    base = tl.zeros((1,), dtype=tl.int32)
    for s in range(0, c):
        poffs = s * BLOCK + tl.arange(0, BLOCK)
        pids = tl.load(ids_ptr + poffs)
        base += tl.sum((pids == e).to(tl.int32), axis=0)
    offs = c * BLOCK + tl.arange(0, BLOCK)
    if EVEN:
        ids = tl.load(ids_ptr + offs)
        hit = ids == e
    else:
        in_bounds = offs < n_elements
        ids = tl.load(ids_ptr + offs, mask=in_bounds, other=-1)
        hit = in_bounds & (ids == e)
    h = hit.to(tl.int32)
    # Rank of each hit inside expert e's bucket = pre-chunk count + prefix.
    rank = tl.cumsum(h, axis=0)
    dst = e * m_max + base + rank - 1
    tl.store(out_ptr + e_off + offs, dst, mask=hit)
    # The last chunk's program for expert e knows the full bucket size.
    if c == n_chunks - 1:
        tl.store(out_ptr + e, tl.sum(base, axis=0) + tl.sum(h, axis=0))
    # Program e == 0 owns the padding lanes so every dst row is written and a
    # plain torch.empty allocation is enough for the combined output buffer.
    if e == 0:
        if EVEN:
            pad = ids < 0
        else:
            pad = in_bounds & (ids < 0)
        tl.store(
            out_ptr + e_off + offs,
            tl.zeros((BLOCK,), dtype=tl.int32),
            mask=pad,
        )


@triton.jit
def _fused_moe_dispatch_index_kernel(
    ids_ptr,
    masked_m_ptr,
    dst_ptr,
    n_elements,
    m_max,
    BLOCK: tl.constexpr,
    E: tl.constexpr,
    EXACT_E: tl.constexpr,
    EVEN: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    if EVEN:
        ids = tl.load(ids_ptr + offs)
    else:
        in_bounds = offs < n_elements
        ids = tl.load(ids_ptr + offs, mask=in_bounds, other=-1)
    valid = ids >= 0
    e_range = tl.arange(0, E)
    # One-hot selector: [BLOCK, E], sel[lane, e] == 1 iff this lane is routed
    # to expert e. Padding (-1) and out-of-bounds lanes are all-zero rows.
    # With EXACT_E every column maps to a real expert, so the column guard
    # compiles away.
    if EXACT_E:
        sel2d = valid[:, None] & (ids[:, None] == e_range[None, :])
    else:
        sel2d = (
            valid[:, None]
            & (ids[:, None] == e_range[None, :])
            & (e_range < n_elements)[None, :]
        )
    seli = sel2d.to(tl.int32)
    # Inclusive prefix sum down the lanes: rank of each selected lane inside
    # this block's slice of expert e's bucket, for every expert at once.
    prefix = tl.cumsum(seli, axis=0)
    # Per-expert count claimed from the global cursor in one vectorized atomic.
    counts = tl.sum(seli, axis=0)
    if EXACT_E:
        base = tl.atomic_add(masked_m_ptr + e_range, counts)
    else:
        base = tl.atomic_add(
            masked_m_ptr + e_range, counts, mask=e_range < n_elements
        )
    # Fused per-lane extraction: only the column matching the lane's id
    # contributes, yielding ids*m_max + base[id] + rank - 1 in one reduction.
    # Padding (-1) and out-of-bounds lanes fall through and are stored as 0.
    dst = ids * m_max - 1 + tl.sum(seli * (base[None, :] + prefix), axis=1)
    dst = tl.where(valid, dst, 0)
    if EVEN:
        tl.store(dst_ptr + offs, dst)
    else:
        tl.store(dst_ptr + offs, dst, mask=offs < n_elements)


def fused_moe_dispatch_index(topk_ids, num_local_experts, m_max):
    n = topk_ids.numel()
    device = topk_ids.device
    if n == 0:
        return (
            torch.zeros(num_local_experts, dtype=torch.int32, device=device),
            torch.empty(0, dtype=torch.int32, device=device),
        )
    flat = topk_ids if topk_ids.is_contiguous() else topk_ids.reshape(-1)
    if num_local_experts <= 0:
        return (
            torch.empty(0, dtype=torch.int32, device=device),
            torch.zeros(n, dtype=torch.int32, device=device),
        )
    if n <= _EC_ELEMS_MAX:
        # One empty allocation backs both outputs (masked_m at offset 0, dst
        # right after); the kernel writes every word itself, so no pre-zeroing
        # fill launch is needed. BLOCK/warps picked by per-size do_bench sweep
        # on the eval target: small routings are launch-latency-bound, larger
        # ones want a wide tile so the chunk count (and with it the EC
        # pre-pass depth) stays at single digits.
        if n <= _EC_SMALL:
            block, warps = 4096, 4
        else:
            block, warps = 8192, 8
        n_chunks = (n + block - 1) // block
        _moe_dispatch_ec_kernel[(num_local_experts * n_chunks,)](
            flat,
            buf := torch.empty(
                num_local_experts + n, dtype=torch.int32, device=device
            ),
            n,
            m_max,
            num_local_experts,
            n_chunks,
            BLOCK=block,
            E=num_local_experts,
            EVEN=(n % block == 0),
            num_warps=warps,
            num_stages=1,
        )
        return buf[:num_local_experts], buf[num_local_experts:]
    # Very large routing fallback: one fused one-hot/atomic kernel over a
    # single pre-zeroed buffer (masked_m must start zeroed for the cursor
    # atomics).
    buf = torch.zeros(num_local_experts + n, dtype=torch.int32, device=device)
    block = 32
    epow2 = (
        1 << (num_local_experts - 1).bit_length()
        if num_local_experts > 1
        else 1
    )
    _fused_moe_dispatch_index_kernel[((n + block - 1) // block,)](
        flat,
        buf,  # masked_m lives at offset 0
        buf[num_local_experts:],  # dst region starts right after it
        n,
        m_max,
        BLOCK=block,
        E=epow2,
        EXACT_E=(num_local_experts == epow2),
        EVEN=(n % block == 0),
        num_warps=2,
        num_stages=1,
    )
    return buf[:num_local_experts], buf[num_local_experts:]


__all__ = ["fused_moe_dispatch_index"]
