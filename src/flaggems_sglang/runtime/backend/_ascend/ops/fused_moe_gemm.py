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
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import triton
import triton.language as tl


@triton.jit
def _persistent_fused_moe_gemm_kernel(
    A_ptr,
    B_ptr,
    topk_weights_ptr,
    topk_ids_ptr,
    Out_ptr,
    num_routes,
    K,
    stride_am,
    stride_ak,
    stride_be,
    stride_bn,
    stride_bk,
    stride_out_t,
    stride_out_j,
    stride_out_n,
    top_k: tl.constexpr,
    N: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    program_id = tl.program_id(0)
    num_programs = tl.num_programs(0)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_tasks = num_routes * num_pid_n

    for task in tl.range(program_id, num_tasks, num_programs):
        route = task // num_pid_n
        pid_n = task % num_pid_n
        token = route // top_k
        slot = route % top_k
        expert_id = tl.load(topk_ids_ptr + route).to(tl.int64)
        routing_weight = tl.load(topk_weights_ptr + route).to(tl.float32)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        n_mask = offs_n < N
        accumulator = tl.zeros((BLOCK_N,), dtype=tl.float32)

        for k_start in tl.range(0, K, BLOCK_K):
            offs_k = k_start + tl.arange(0, BLOCK_K)
            k_mask = offs_k < K
            a = tl.load(
                A_ptr
                + token.to(tl.int64) * stride_am
                + offs_k.to(tl.int64) * stride_ak,
                mask=k_mask,
                other=0.0,
            ).to(tl.float32)
            b = tl.load(
                B_ptr
                + expert_id * stride_be
                + offs_n[None, :].to(tl.int64) * stride_bn
                + offs_k[:, None].to(tl.int64) * stride_bk,
                mask=k_mask[:, None] & n_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            accumulator += tl.sum(a[:, None] * b, axis=0)

        tl.store(
            Out_ptr
            + token.to(tl.int64) * stride_out_t
            + slot.to(tl.int64) * stride_out_j
            + offs_n.to(tl.int64) * stride_out_n,
            accumulator * routing_weight,
            mask=n_mask,
        )


def fused_moe_gemm(
    A: torch.Tensor,
    B: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    top_k: int,
) -> torch.Tensor:
    T, K = A.shape
    _, N, K_b = B.shape
    assert K == K_b

    topk_weights_work = topk_weights.contiguous().reshape(-1)
    topk_ids_work = topk_ids.to(torch.int32).contiguous().reshape(-1)
    output = torch.empty(
        (T, top_k, N),
        dtype=A.dtype,
        device=A.device,
    )

    block_n = 64
    block_k = 32
    num_tasks = T * top_k * triton.cdiv(N, block_n)
    num_programs = min(32, num_tasks)
    _persistent_fused_moe_gemm_kernel[(num_programs, 1, 1)](
        A,
        B,
        topk_weights_work,
        topk_ids_work,
        output,
        T * top_k,
        K,
        A.stride(0),
        A.stride(1),
        B.stride(0),
        B.stride(1),
        B.stride(2),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        top_k=top_k,
        N=N,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_warps=4,
        num_stages=2,
    )
    return output


__all__ = ["fused_moe_gemm"]
