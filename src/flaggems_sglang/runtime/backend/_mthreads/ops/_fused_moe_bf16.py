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

# Adapted from flagos-ai/sglang-plugin-FL PR #125 at 68b414959bbf6ce89501c149293294d83d742e91.
"""BF16 specialization of SGLang's canonical routed-expert GEMM.

Original ancestry: vLLM a6221a144af772fd1a68fe7e627935dc53e81738.
Quantization, TMA, expert filtering, atomic combine and bias are excluded
by the public wrapper. BF16 materialization and dot order are preserved.
"""

import triton
import triton.language as tl


@triton.jit
def fused_moe_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    topk_weights_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    N,
    K,
    EM,
    num_valid_tokens,
    stride_am,
    stride_ak,
    stride_be,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    top_k: tl.constexpr,
    compute_type: tl.constexpr,
    even_Ks: tl.constexpr,
    FUSE_SWIGLU: tl.constexpr = False,
):
    """
    Implements the fused computation for a Mixture of Experts (MOE) using
    token and expert matrices.

    Key Parameters:
    - A: The input tensor representing tokens with shape (*, K), where '*' can
        be any shape representing batches and K is the feature dimension of
        each token.
    - B: The stacked MOE weight tensor with shape (E, N, K), where E is
        the number of experts, K is the input feature dimension, and N is
        the output feature dimension.
    - C: The output cache tensor with shape (M, topk, N), where M is the
        total number of tokens post padding, topk is the number of times
        each token is repeated, and N is the output feature dimension.
    - sorted_token_ids: A tensor containing the sorted indices of tokens,
        repeated topk times and arranged by the expert index they are
        assigned to.
    - expert_ids: A tensor containing the indices of the expert for each
        block. It determines which expert matrix from B should be used for
        each block in A.

    This kernel performs the multiplication of a token by its corresponding
    expert matrix as determined by `expert_ids`. The sorting of
    `sorted_token_ids` by expert index and padding ensures divisibility by
    BLOCK_SIZE_M, which is necessary to maintain consistency in block matrix
    multiplication across different blocks processed by the same expert.
    """
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N // 2 if FUSE_SWIGLU else N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + pid % num_pid_in_group % group_size_m
    pid_n = pid % num_pid_in_group // group_size_m
    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
        return
    offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(
        tl.int64
    )
    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id)
    offs_token = offs_token.to(tl.int64)
    token_mask = offs_token < num_valid_tokens
    off_experts_i32 = tl.load(expert_ids_ptr + pid_m)
    off_experts = off_experts_i32.to(tl.int64)
    offs_bn = (
        pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)
    ) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (
        offs_token[:, None] // top_k * stride_am + offs_k[None, :] * stride_ak
    )
    if FUSE_SWIGLU:
        gate_b_ptrs = (
            b_ptr
            + off_experts * stride_be
            + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
        )
        up_b_ptrs = (
            b_ptr
            + off_experts * stride_be
            + (
                offs_k[:, None] * stride_bk
                + (offs_bn[None, :] + N // 2) * stride_bn
            )
        )
    else:
        b_ptrs = (
            b_ptr
            + off_experts * stride_be
            + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
        )
    if FUSE_SWIGLU:
        gate_accumulator = tl.zeros(
            (BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32
        )
        up_accumulator = tl.zeros(
            (BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32
        )
    else:
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k_start in range(0, K, BLOCK_SIZE_K):
        if even_Ks:
            a = tl.load(a_ptrs, mask=token_mask[:, None], other=0.0)
        else:
            a = tl.load(
                a_ptrs,
                mask=token_mask[:, None] & (offs_k[None, :] < K - k_start),
                other=0.0,
            )
        if FUSE_SWIGLU:
            if even_Ks:
                gate_b = tl.load(gate_b_ptrs)
                up_b = tl.load(up_b_ptrs)
            else:
                gate_b = tl.load(
                    gate_b_ptrs, mask=offs_k[:, None] < K - k_start, other=0.0
                )
                up_b = tl.load(
                    up_b_ptrs, mask=offs_k[:, None] < K - k_start, other=0.0
                )
        elif even_Ks:
            b = tl.load(b_ptrs)
        else:
            b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k_start, other=0.0)
        if FUSE_SWIGLU:
            gate_accumulator += tl.dot(a, gate_b)
            up_accumulator += tl.dot(a, up_b)
        else:
            accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        if not FUSE_SWIGLU:
            b_ptrs += BLOCK_SIZE_K * stride_bk
        if FUSE_SWIGLU:
            gate_b_ptrs += BLOCK_SIZE_K * stride_bk
            up_b_ptrs += BLOCK_SIZE_K * stride_bk
    if FUSE_SWIGLU:
        gate_bf16 = gate_accumulator.to(compute_type)
        up_bf16 = up_accumulator.to(compute_type)
        gate_f = gate_bf16.to(tl.float32)
        up_f = up_bf16.to(tl.float32)
        out_act = (gate_f * tl.sigmoid(gate_f) * up_f).to(compute_type)
    else:
        if MUL_ROUTED_WEIGHT:
            moe_weight = tl.load(
                topk_weights_ptr + offs_token, mask=token_mask, other=0
            )
            accumulator *= moe_weight[:, None]
        accumulator = accumulator.to(compute_type)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    if FUSE_SWIGLU:
        offs_half = offs_cn
        c_ptrs = (
            c_ptr
            + stride_cm * offs_token[:, None]
            + stride_cn * offs_half[None, :]
        )
        c_mask = token_mask[:, None] & (offs_half[None, :] < N // 2)
        tl.store(c_ptrs, out_act, mask=c_mask)
    else:
        c_ptrs = (
            c_ptr
            + stride_cm * offs_token[:, None]
            + stride_cn * offs_cn[None, :]
        )
        c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
        tl.store(c_ptrs, accumulator, mask=c_mask)
