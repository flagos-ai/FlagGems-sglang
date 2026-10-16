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
    hidden_ptr,
    idx_ptr,
    scale_ptr,
    out_ptr,
    num_experts,
    TOP_K: tl.constexpr,
    HIDDEN_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_t = tl.program_id(0)
    pid_d = tl.program_id(1)

    offs_k = tl.arange(0, TOP_K)
    idx = tl.load(idx_ptr + pid_t * TOP_K + offs_k)
    scale = tl.load(scale_ptr + pid_t * TOP_K + offs_k)
    total = tl.sum(tl.where(idx >= num_experts, scale.to(tl.float32), 0.0))

    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    offs_h = pid_t * HIDDEN_DIM + offs_d
    if BLOCK_D <= HIDDEN_DIM and HIDDEN_DIM % BLOCK_D == 0:
        h = tl.load(hidden_ptr + offs_h)
        out = (h.to(tl.float32) * total).to(out_ptr.dtype.element_ty)
        tl.store(out_ptr + offs_h, out)
    else:
        mask_d = offs_d < HIDDEN_DIM
        h = tl.load(hidden_ptr + offs_h, mask=mask_d, other=0.0)
        out = (h.to(tl.float32) * total).to(out_ptr.dtype.element_ty)
        tl.store(out_ptr + offs_h, out, mask=mask_d)


@triton.jit
def _zero_experts_identity_general_kernel(
    hidden_ptr,
    idx_ptr,
    scale_ptr,
    out_ptr,
    hidden_stride,
    idx_stride,
    scale_stride,
    top_k,
    hidden_dim,
    num_experts,
    TOP_K_PADDED: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_t = tl.program_id(0)
    pid_d = tl.program_id(1)

    offs_k = tl.arange(0, TOP_K_PADDED)
    mask_k = offs_k < top_k
    idx = tl.load(idx_ptr + pid_t * idx_stride + offs_k, mask=mask_k, other=0)
    scale = tl.load(
        scale_ptr + pid_t * scale_stride + offs_k, mask=mask_k, other=0.0
    )
    total = tl.sum(tl.where(idx >= num_experts, scale.to(tl.float32), 0.0))

    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    mask_d = offs_d < hidden_dim
    offs_h = pid_t * hidden_stride + offs_d
    h = tl.load(hidden_ptr + offs_h, mask=mask_d, other=0.0)
    out = (h.to(tl.float32) * total).to(out_ptr.dtype.element_ty)
    tl.store(out_ptr + offs_h, out, mask=mask_d)


def zero_experts_identity(
    expert_indices, expert_scales, num_experts, zero_expert_type, hidden_states
):
    num_tokens, top_k = expert_scales.shape
    hidden_dim = hidden_states.shape[1]
    out = torch.empty_like(hidden_states)

    if num_tokens == 0:
        return out

    if num_tokens <= 64:
        block_d, num_warps = 1024, 4
    else:
        block_d, num_warps = 8192, 8

    top_k_padded = triton.next_power_of_2(max(top_k, 2))
    contiguous = (
        hidden_states.stride(1) == 1
        and hidden_states.stride(0) == hidden_dim
        and out.stride(1) == 1
        and out.stride(0) == hidden_dim
        and expert_scales.stride(1) == 1
        and expert_scales.stride(0) == top_k
        and expert_indices.stride(1) == 1
        and expert_indices.stride(0) == top_k
        and top_k == top_k_padded
    )
    if contiguous:
        _zero_experts_identity_kernel[
            (num_tokens, triton.cdiv(hidden_dim, block_d))
        ](
            hidden_states,
            expert_indices,
            expert_scales,
            out,
            num_experts,
            TOP_K=top_k,
            HIDDEN_DIM=hidden_dim,
            BLOCK_D=block_d,
            num_warps=num_warps,
            num_stages=2,
        )
    else:
        _zero_experts_identity_general_kernel[
            (num_tokens, triton.cdiv(hidden_dim, block_d))
        ](
            hidden_states,
            expert_indices,
            expert_scales,
            out,
            hidden_states.stride(0),
            expert_indices.stride(0),
            expert_scales.stride(0),
            top_k,
            hidden_dim,
            num_experts,
            TOP_K_PADDED=top_k_padded,
            BLOCK_D=block_d,
            num_warps=num_warps,
            num_stages=2,
        )
    return out


__all__ = ["zero_experts_identity"]
