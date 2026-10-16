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
    idx_ptr,
    scale_ptr,
    h_ptr,
    out_ptr,
    num_experts,
    K,
    D,
    KP: tl.constexpr,
    BLOCK_D: tl.constexpr,
    ND: tl.constexpr,
):
    pid_t = tl.program_id(0)

    offs_k = tl.arange(0, KP)
    r_mask = offs_k < K
    idx = tl.load(idx_ptr + pid_t * K + offs_k, mask=r_mask, other=0)
    sc = tl.load(scale_ptr + pid_t * K + offs_k, mask=r_mask, other=0.0).to(
        tl.float32
    )
    total = tl.sum(tl.where(idx >= num_experts, sc, 0.0), axis=0)

    for i in tl.static_range(ND):
        offs_d = i * BLOCK_D + tl.arange(0, BLOCK_D)
        offs_d = tl.max_contiguous(tl.multiple_of(offs_d, BLOCK_D), BLOCK_D)
        d_mask = offs_d < D
        h = tl.load(h_ptr + pid_t * D + offs_d, mask=d_mask, other=0.0).to(
            tl.float32
        )
        res = h * total
        tl.store(
            out_ptr + pid_t * D + offs_d,
            res.to(out_ptr.dtype.element_ty),
            mask=d_mask,
        )


def zero_experts_identity(
    expert_indices, expert_scales, num_experts, zero_expert_type, hidden_states
):
    T, K = expert_indices.shape
    D = hidden_states.shape[-1]

    if not expert_indices.is_contiguous():
        expert_indices = expert_indices.contiguous()
    if not expert_scales.is_contiguous():
        expert_scales = expert_scales.contiguous()
    if not hidden_states.is_contiguous():
        hidden_states = hidden_states.contiguous()

    out = torch.empty_like(hidden_states)
    if T == 0:
        return out

    KP = triton.next_power_of_2(max(K, 1))

    BLOCK_D = min(4096, triton.next_power_of_2(max(D, 1)))
    _zero_experts_identity_kernel[(T,)](
        expert_indices,
        expert_scales,
        hidden_states,
        out,
        num_experts,
        K,
        D,
        KP=KP,
        BLOCK_D=BLOCK_D,
        ND=triton.cdiv(D, BLOCK_D),
        num_warps=max(4, min(32, BLOCK_D // 128)),
    )
    return out


__all__ = ["zero_experts_identity"]
