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

"""Triton kernel for moe/fused_moe_dispatch_index.

Builds the permutation grouped-GEMM target indices for masked/DeepGEMM MoE
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

_BLOCK = 128
_NUM_WARPS = 4


@triton.jit(do_not_specialize=["N"])
def _dispatch_hist_kernel(
    ids_ptr,  # [N] int32 expert ids (-1 = padding)
    hist_ptr,  # [P, BLOCK_E] int32 per-block counts
    N,  # total number of (token, slot) pairs
    BLOCK: tl.constexpr,  # elements per program
    BLOCK_E: tl.constexpr,  # padded power-of-2 expert count
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    ids = tl.load(ids_ptr + offs, mask=mask, other=-1)
    valid = mask & (ids >= 0)

    e_range = tl.arange(0, BLOCK_E)
    e_mask = (
        e_range < BLOCK_E
    )  # hist rows are allocated padded; all lanes valid
    # One-hot match matrix: M[i, e] = 1 iff element i routes to expert e.
    match = (valid[:, None] & (ids[:, None] == e_range[None, :])).to(tl.int32)
    cnts = tl.sum(match, axis=0)
    tl.store(hist_ptr + pid * BLOCK_E + e_range, cnts, mask=e_mask)


@triton.jit
def _dispatch_scan_kernel(
    hist_ptr,  # [P, BLOCK_E] int32 per-block counts
    masked_m_ptr,  # [num_experts] int32 output counts
    scan_ptr,  # [P, BLOCK_E] int32 exclusive block prefixes
    P,
    num_experts,
    BLOCK_E: tl.constexpr,  # padded power-of-2 expert count
    BLOCK_P: tl.constexpr,  # padded power-of-2 block count
):
    p_range = tl.arange(0, BLOCK_P)
    e_range = tl.arange(0, BLOCK_E)
    e_mask = e_range < num_experts
    pmask = p_range < P

    hist = tl.load(
        hist_ptr + p_range[:, None] * BLOCK_E + e_range[None, :],
        mask=pmask[:, None] & e_mask[None, :],
        other=0,
    )
    tl.store(masked_m_ptr + e_range, tl.sum(hist, axis=0), mask=e_mask)
    scan = tl.cumsum(hist, axis=0) - hist  # exclusive prefix along blocks
    tl.store(
        scan_ptr + p_range[:, None] * BLOCK_E + e_range[None, :],
        scan,
        mask=pmask[:, None] & e_mask[None, :],
    )


@triton.jit(do_not_specialize=["N", "m_max"])
def _dispatch_fill_kernel(
    ids_ptr,  # [N] int32 expert ids (-1 = padding)
    scan_ptr,  # [P, BLOCK_E] int32 exclusive block prefixes
    dst_ptr,  # [N] int32 output destination rows
    N,  # total number of (token, slot) pairs
    m_max,  # bucket capacity per expert
    BLOCK: tl.constexpr,  # elements per program
    BLOCK_E: tl.constexpr,  # padded power-of-2 expert count
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    ids = tl.load(ids_ptr + offs, mask=mask, other=-1)
    valid = mask & (ids >= 0)

    e_range = tl.arange(0, BLOCK_E)
    # One-hot match matrix for this tile: M[i, e] = 1 iff element i routes
    # to expert e. The running per-expert count gives the exclusive
    # bucket-local rank via the one-hot rows.
    oh = ((ids[:, None] == e_range[None, :]) & valid[:, None]).to(tl.int32)
    incl = tl.cumsum(oh, axis=0)
    rank = tl.sum(incl * oh, axis=1) - 1
    # Base row: this block's bucket offsets = counts accumulated by every
    # earlier block (scan[pid] is the exclusive prefix, so block 0 reads
    # zeros). Gathered per element directly by expert id — cheaper and more
    # portable than broadcasting the row into a [BLOCK, BLOCK_E] product.
    base = tl.load(scan_ptr + pid * BLOCK_E + ids, mask=valid, other=0)
    tl.store(dst_ptr + offs, ids * m_max + base + rank, mask=valid)
    # padding / unmatched elements keep the 0 default written below.


def fused_moe_dispatch_index(topk_ids, num_local_experts, m_max):
    flat = topk_ids.reshape(-1)
    n = flat.numel()
    device = flat.device
    if n == 0 or num_local_experts == 0:
        return (
            torch.zeros(num_local_experts, dtype=torch.int32, device=device),
            torch.zeros(n, dtype=torch.int32, device=device),
        )

    block_e = triton.next_power_of_2(num_local_experts)
    grid_p = triton.cdiv(n, _BLOCK)
    block_p = triton.next_power_of_2(grid_p)

    masked_m = torch.zeros(num_local_experts, dtype=torch.int32, device=device)
    src2dst = torch.zeros(n, dtype=torch.int32, device=device)
    hist = torch.zeros(grid_p * block_e, dtype=torch.int32, device=device)
    scan = torch.zeros(grid_p * block_e, dtype=torch.int32, device=device)

    _dispatch_hist_kernel[(grid_p,)](
        flat,
        hist,
        n,
        BLOCK=_BLOCK,
        BLOCK_E=block_e,
        num_warps=_NUM_WARPS,
    )
    _dispatch_scan_kernel[(1,)](
        hist,
        masked_m,
        scan,
        grid_p,
        num_local_experts,
        BLOCK_E=block_e,
        BLOCK_P=block_p,
        num_warps=_NUM_WARPS,
    )
    _dispatch_fill_kernel[(grid_p,)](
        flat,
        scan,
        src2dst,
        n,
        m_max,
        BLOCK=_BLOCK,
        BLOCK_E=block_e,
        num_warps=_NUM_WARPS,
    )
    return masked_m, src2dst


__all__ = ["fused_moe_dispatch_index"]
