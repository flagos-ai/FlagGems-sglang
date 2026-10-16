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
def _zero_experts_bulk_kernel(
    idx_ptr,
    scale_ptr,
    h_ptr,
    out_ptr,
    num_experts,
    TPB: tl.constexpr,
    BLOCK_D: tl.constexpr,
    K: tl.constexpr,
    D: tl.constexpr,
):
    t0 = tl.program_id(0) * TPB
    db = tl.program_id(1)
    offs_t = t0 + tl.arange(0, TPB)
    offs_k = tl.arange(0, K)
    ii = tl.load(idx_ptr + offs_t[:, None] * K + offs_k[None, :])
    ss = tl.load(scale_ptr + offs_t[:, None] * K + offs_k[None, :])
    zero_sc = tl.where(ii >= num_experts, ss, 0.0)
    total = tl.sum(zero_sc, axis=1)  # [TPB] fp32, exact like the reference
    offs_d = db * BLOCK_D + tl.arange(0, BLOCK_D)
    ptrs = offs_t[:, None] * D + offs_d[None, :]
    x = tl.load(h_ptr + ptrs)
    y = x.to(tl.float32) * total[:, None]
    tl.store(out_ptr + ptrs, y.to(out_ptr.dtype.element_ty))


@triton.jit
def _zero_experts_token_kernel(
    idx_ptr,
    scale_ptr,
    h_ptr,
    out_ptr,
    num_experts,
    K,
    BLOCK_D: tl.constexpr,
    BLOCK_K: tl.constexpr,
    EVEN_K: tl.constexpr,
    D: tl.constexpr,
):
    t = tl.program_id(0)
    db = tl.program_id(1)
    offs_k = tl.arange(0, BLOCK_K)
    if EVEN_K:
        ii = tl.load(idx_ptr + t * K + offs_k)
        ss = tl.load(scale_ptr + t * K + offs_k)
    else:
        kmask = offs_k < K
        ii = tl.load(idx_ptr + t * K + offs_k, mask=kmask, other=0)
        ss = tl.load(scale_ptr + t * K + offs_k, mask=kmask, other=0.0)
    zero_sc = tl.where(ii >= num_experts, ss, 0.0)
    total = tl.sum(zero_sc, axis=0)
    offs_d = db * BLOCK_D + tl.arange(0, BLOCK_D)
    x = tl.load(h_ptr + t * D + offs_d)
    y = x.to(tl.float32) * total
    tl.store(out_ptr + t * D + offs_d, y.to(out_ptr.dtype.element_ty))


def zero_experts_identity(
    expert_indices, expert_scales, num_experts, zero_expert_type, hidden_states
):
    T, K = expert_indices.shape
    D = hidden_states.shape[-1]
    out = torch.empty_like(hidden_states)

    TPB = 64
    if K == triton.next_power_of_2(K) and T >= TPB:
        # Hot path: 2D [TPB, BLOCK_D] tiles, everything maskless.
        BLOCK_D = 1024 if D % 1024 == 0 else 256
        nbulk = (T // TPB) * TPB
        _zero_experts_bulk_kernel[(nbulk // TPB, D // BLOCK_D)](
            expert_indices,
            expert_scales,
            hidden_states,
            out,
            num_experts,
            TPB=TPB,
            BLOCK_D=BLOCK_D,
            K=K,
            D=D,
        )
        if T > nbulk:
            _zero_experts_token_kernel[(T - nbulk, D // BLOCK_D)](
                expert_indices[nbulk:],
                expert_scales[nbulk:],
                hidden_states[nbulk:],
                out[nbulk:],
                num_experts,
                K,
                BLOCK_D=BLOCK_D,
                BLOCK_K=K,
                EVEN_K=True,
                D=D,
            )
    else:
        # Cold path: small T or non-power-of-two top_k (2D masked k-loads
        # are unreliable on this backend). One program per token.
        BLOCK_K = triton.next_power_of_2(K)
        BLOCK_D = 1024 if D % 1024 == 0 else 256
        _zero_experts_token_kernel[(T, D // BLOCK_D)](
            expert_indices,
            expert_scales,
            hidden_states,
            out,
            num_experts,
            K,
            BLOCK_D=BLOCK_D,
            BLOCK_K=BLOCK_K,
            EVEN_K=(K == BLOCK_K),
            D=D,
        )
    return out


__all__ = ["zero_experts_identity"]
