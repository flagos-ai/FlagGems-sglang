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

import torch
import triton
import triton.language as tl


@triton.jit
def _zero_experts_identity_kernel(
    idx_ptr,  # [num_tokens, top_k] int32
    scale_ptr,  # [num_tokens, top_k] fp32
    hidden_ptr,  # [num_tokens, hidden_dim], contiguous
    out_ptr,  # [num_tokens, hidden_dim], contiguous
    hidden_dim,
    num_experts,
    NBLK: tl.constexpr,  # hidden chunks per row = cdiv(hidden_dim, BLOCK_H)
    TOPK: tl.constexpr,  # pow2 >= top_k
    BLOCK_H: tl.constexpr,
    EVEN: tl.constexpr,  # hidden_dim % BLOCK_H == 0 -> no bounds mask needed
):
    pid = tl.program_id(0)
    tok = pid // NBLK
    blk = pid % NBLK

    offs_h = blk * BLOCK_H + tl.arange(0, BLOCK_H)
    base = tok * hidden_dim

    # Fold this token's routing slots into one fp32 scale total: identity
    # slots (idx >= num_experts) keep their scale, every other slot is 0.
    offs_k = tl.arange(0, TOPK)
    idx = tl.load(idx_ptr + tok * TOPK + offs_k)
    scale = tl.load(scale_ptr + tok * TOPK + offs_k)
    total = tl.sum(tl.where(idx >= num_experts, scale, 0.0), axis=0)

    if EVEN:
        h = tl.load(
            hidden_ptr + base + offs_h, eviction_policy="evict_last"
        ).to(tl.float32)
        tl.store(
            out_ptr + base + offs_h, (h * total).to(out_ptr.dtype.element_ty)
        )
    else:
        mask_h = offs_h < hidden_dim
        h = tl.load(
            hidden_ptr + base + offs_h,
            mask=mask_h,
            other=0.0,
            eviction_policy="evict_last",
        ).to(tl.float32)
        tl.store(
            out_ptr + base + offs_h,
            (h * total).to(out_ptr.dtype.element_ty),
            mask=mask_h,
        )


# Narrow-block launch: the sweep (1/8/64/512/4096 tokens) picked these for
# every regime. Kept as plain locals so no module-level mutable state exists.
_BLOCK_H = 1024
_NUM_WARPS = 2


def zero_experts_identity(
    expert_indices, expert_scales, num_experts, zero_expert_type, hidden_states
):
    num_tokens, top_k = expert_indices.shape
    hidden_dim = hidden_states.shape[1]

    if not (expert_indices.is_contiguous() and expert_scales.is_contiguous()):
        expert_indices = expert_indices.contiguous()
        expert_scales = expert_scales.contiguous()
    if not hidden_states.is_contiguous():
        hidden_states = hidden_states.contiguous()

    out = torch.empty_like(hidden_states)

    if num_tokens == 0 or hidden_dim == 0:
        return out

    block_h = _BLOCK_H
    nblk = (hidden_dim + block_h - 1) // block_h
    even = (hidden_dim % block_h) == 0

    _zero_experts_identity_kernel[(num_tokens * nblk,)](
        expert_indices,
        expert_scales,
        hidden_states,
        out,
        hidden_dim,
        num_experts,
        NBLK=nblk,
        TOPK=triton.next_power_of_2(top_k),
        BLOCK_H=block_h,
        EVEN=even,
        num_warps=_NUM_WARPS,
    )
    return out


__all__ = ["zero_experts_identity"]
