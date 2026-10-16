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


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_H": 2048}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_H": 2048}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_H": 4096}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_H": 4096}, num_warps=8, num_stages=2),
    ],
    key=["H", "T", "top_k"],
)
@triton.jit
def _zero_experts_identity_kernel(
    idx_ptr,
    scale_ptr,
    h_ptr,
    out_ptr,
    num_experts,
    stride_it,
    stride_st,
    stride_ht,
    stride_ot,
    top_k,
    H,
    T,
    BLOCK_H: tl.constexpr,
    TOP_K: tl.constexpr,
):
    pid_c = tl.program_id(0)
    pid_t = tl.program_id(1)

    offs_k = tl.arange(0, TOP_K)
    idx = tl.load(idx_ptr + pid_t * stride_it + offs_k)
    sc = tl.load(scale_ptr + pid_t * stride_st + offs_k)
    total = tl.sum(tl.where(idx >= num_experts, sc, 0.0), axis=0)

    offs_h = pid_c * BLOCK_H + tl.arange(0, BLOCK_H)
    offs_h = tl.max_contiguous(tl.multiple_of(offs_h, BLOCK_H), BLOCK_H)
    full = (pid_c + 1) * BLOCK_H <= H
    if total == 0.0:
        z = tl.zeros([BLOCK_H], dtype=out_ptr.dtype.element_ty)
        if full:
            tl.store(out_ptr + pid_t * stride_ot + offs_h, z)
        else:
            tl.store(out_ptr + pid_t * stride_ot + offs_h, z, mask=offs_h < H)
    else:
        if full:
            h = tl.load(h_ptr + pid_t * stride_ht + offs_h)
            out = h.to(tl.float32) * total
            tl.store(
                out_ptr + pid_t * stride_ot + offs_h,
                out.to(out_ptr.dtype.element_ty),
            )
        else:
            h = tl.load(
                h_ptr + pid_t * stride_ht + offs_h,
                mask=offs_h < H,
                other=0.0,
            )
            out = h.to(tl.float32) * total
            tl.store(
                out_ptr + pid_t * stride_ot + offs_h,
                out.to(out_ptr.dtype.element_ty),
                mask=offs_h < H,
            )


def zero_experts_identity(
    expert_indices, expert_scales, num_experts, zero_expert_type, hidden_states
):
    num_tokens, top_k = expert_scales.shape
    H = hidden_states.shape[-1]

    out = torch.empty_like(hidden_states)

    TOP_K = triton.next_power_of_2(top_k)

    def grid(meta):
        return (triton.cdiv(H, meta["BLOCK_H"]), num_tokens)

    _zero_experts_identity_kernel[grid](
        expert_indices,
        expert_scales,
        hidden_states,
        out,
        num_experts,
        expert_indices.stride(0),
        expert_scales.stride(0),
        hidden_states.stride(0),
        out.stride(0),
        top_k,
        H,
        num_tokens,
        TOP_K=TOP_K,
    )
    return out


__all__ = ["zero_experts_identity"]
