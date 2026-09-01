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

"""Kunlunxin MoE GEMM using FlagTree's XPU matmul kernel shape."""

import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["M"])
def _xpu_matmul_kernel(
    A_ptr,
    B_ptr,
    C_ptr,
    M,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    EVEN_K: tl.constexpr,
):
    program_id = tl.program_id(0)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    width = GROUP_M * grid_n
    group_id = program_id // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (program_id % group_size)
    pid_n = (program_id % width) // group_size

    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    columns = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    wrapped_rows = tl.max_contiguous(
        tl.multiple_of(rows % M, BLOCK_M),
        BLOCK_M,
    )
    wrapped_columns = tl.max_contiguous(
        tl.multiple_of(columns % N, BLOCK_N),
        BLOCK_N,
    )
    k_offsets = tl.arange(0, BLOCK_K)

    a_ptrs = (
        A_ptr
        + wrapped_rows[:, None] * stride_am
        + k_offsets[None, :] * stride_ak
    )
    b_ptrs = (
        B_ptr
        + k_offsets[:, None] * stride_bk
        + wrapped_columns[None, :] * stride_bn
    )
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_block in range(0, tl.cdiv(K, BLOCK_K)):
        if EVEN_K:
            a = tl.load(a_ptrs)
            b = tl.load(b_ptrs)
        else:
            remaining = K - k_block * BLOCK_K
            a = tl.load(
                a_ptrs,
                mask=k_offsets[None, :] < remaining,
                other=0.0,
            )
            b = tl.load(
                b_ptrs,
                mask=k_offsets[:, None] < remaining,
                other=0.0,
            )
        accumulator = tl.dot(a, b, accumulator, out_dtype=tl.float32)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    columns = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = C_ptr + rows[:, None] * stride_cm + columns[None, :] * stride_cn
    tl.store(
        c_ptrs,
        accumulator,
        mask=(rows < M)[:, None] & (columns < N)[None, :],
    )


def _cpu_route_plan(topk_ids, num_experts, top_k):
    flat_ids_cpu = (
        topk_ids.detach()
        .reshape(-1)
        .to(
            device="cpu",
            dtype=torch.int64,
        )
    )
    order_cpu = torch.argsort(flat_ids_cpu, stable=False)
    counts = torch.bincount(
        flat_ids_cpu,
        minlength=num_experts,
    ).tolist()
    route_tokens_cpu = torch.div(
        order_cpu,
        top_k,
        rounding_mode="floor",
    )
    inverse_cpu = torch.empty_like(order_cpu)
    inverse_cpu[order_cpu] = torch.arange(
        order_cpu.numel(),
        dtype=order_cpu.dtype,
    )
    return route_tokens_cpu, order_cpu, inverse_cpu, counts


def fused_moe_gemm(A, B, topk_weights, topk_ids, top_k):
    T, K = A.shape
    E, N, K_b = B.shape
    assert K == K_b

    route_tokens_cpu, order_cpu, inverse_cpu, counts = _cpu_route_plan(
        topk_ids,
        E,
        top_k,
    )
    device = A.device
    route_tokens = route_tokens_cpu.to(device=device)
    order = order_cpu.to(device=device)
    inverse = inverse_cpu.to(device=device)

    A_sorted = A.contiguous().index_select(0, route_tokens).contiguous()
    B_work = B.contiguous()
    weights_sorted = (
        topk_weights.contiguous().reshape(-1).index_select(0, order).float()
    )
    sorted_fp32 = torch.empty(
        (T * top_k, N),
        dtype=torch.float32,
        device=device,
    )

    block_m = 64
    block_n = 128
    block_k = 32
    route_start = 0
    for expert, count in enumerate(counts):
        if count:
            A_expert = A_sorted.narrow(0, route_start, count)
            B_expert = B_work[expert]
            C_expert = sorted_fp32.narrow(0, route_start, count)
            grid = (
                triton.cdiv(count, block_m) * triton.cdiv(N, block_n),
                1,
            )
            _xpu_matmul_kernel[grid](
                A_expert,
                B_expert,
                C_expert,
                count,
                A_expert.stride(0),
                A_expert.stride(1),
                B_expert.stride(1),
                B_expert.stride(0),
                C_expert.stride(0),
                C_expert.stride(1),
                N=N,
                K=K,
                BLOCK_M=block_m,
                BLOCK_N=block_n,
                BLOCK_K=block_k,
                GROUP_M=8,
                EVEN_K=K % block_k == 0,
                num_warps=4,
                num_stages=4,
            )
        route_start += count

    sorted_fp32.mul_(weights_sorted[:, None])
    sorted_output = sorted_fp32.to(dtype=A.dtype)
    output = sorted_output.index_select(0, inverse)
    return output.reshape(T, top_k, N)


__all__ = ["fused_moe_gemm"]
