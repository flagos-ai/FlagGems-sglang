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


@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_N": 64, "BLOCK_K": 32}, num_warps=4, num_stages=2
        ),
        triton.Config(
            {"BLOCK_N": 128, "BLOCK_K": 32}, num_warps=4, num_stages=2
        ),
        triton.Config(
            {"BLOCK_N": 64, "BLOCK_K": 64}, num_warps=4, num_stages=2
        ),
        triton.Config(
            {"BLOCK_N": 128, "BLOCK_K": 64}, num_warps=8, num_stages=3
        ),
        triton.Config(
            {"BLOCK_N": 256, "BLOCK_K": 32}, num_warps=8, num_stages=3
        ),
        triton.Config(
            {"BLOCK_N": 256, "BLOCK_K": 64}, num_warps=8, num_stages=3
        ),
    ],
    key=["N", "K"],
)
@triton.jit
def _fused_moe_gemm_single_token_kernel(
    A_ptr,
    B_ptr,
    topk_weights_ptr,
    topk_ids_ptr,
    Out_ptr,
    T,
    E,
    N,
    K,
    top_k,
    stride_am,
    stride_ak,
    stride_be,
    stride_bn,
    stride_bk,
    stride_out_t,
    stride_out_j,
    stride_out_n,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_tok = tl.program_id(0)
    pid_slot = tl.program_id(1)
    pid_n = tl.program_id(2)

    if pid_tok >= T or pid_slot >= top_k:
        return

    expert_id = tl.load(topk_ids_ptr + pid_tok * top_k + pid_slot).to(tl.int64)
    routing_weight = tl.load(topk_weights_ptr + pid_tok * top_k + pid_slot).to(
        tl.float32
    )

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = offs_n < N

    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        k_mask = offs_k < K

        a_ptrs = (
            A_ptr
            + pid_tok.to(tl.int64) * stride_am
            + offs_k.to(tl.int64) * stride_ak
        )
        a = tl.load(a_ptrs, mask=k_mask, other=0.0).to(tl.float32)

        b_ptrs = (
            B_ptr
            + expert_id * stride_be
            + offs_n[None, :].to(tl.int64) * stride_bn
            + offs_k[:, None].to(tl.int64) * stride_bk
        )
        b = tl.load(
            b_ptrs,
            mask=k_mask[:, None] & n_mask[None, :],
            other=0.0,
        ).to(tl.float32)

        acc += tl.sum(a[:, None] * b, axis=0)

    scaled = acc * routing_weight

    out_ptrs = (
        Out_ptr
        + pid_tok.to(tl.int64) * stride_out_t
        + pid_slot.to(tl.int64) * stride_out_j
        + offs_n.to(tl.int64) * stride_out_n
    )
    tl.store(out_ptrs, scaled, mask=n_mask)


def fused_moe_gemm(
    A: torch.Tensor,
    B: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    top_k: int,
) -> torch.Tensor:
    assert A.ndim == 2, "A must be (T, K)"
    assert B.ndim == 3, "B must be (E, N, K)"
    assert topk_weights.ndim == 2, "topk_weights must be (T, top_k)"
    assert topk_ids.ndim == 2, "topk_ids must be (T, top_k)"

    T, K = A.shape
    E, N, K_b = B.shape
    assert K == K_b, f"Hidden dimension mismatch: A has K={K}, B has K={K_b}"
    assert topk_weights.shape == (T, top_k), "topk_weights shape mismatch"
    assert topk_ids.shape == (T, top_k), "topk_ids shape mismatch"

    # A/B 原 dtype 直传：kernel 内 load 后即升 fp32 累加，且 A/B 走显式 strides，
    # 无需 contiguous；topk_* 在 kernel 里是扁平索引，须连续（张量很小，代价可忽略）。
    topk_weights_c = topk_weights.contiguous()
    topk_ids_c = topk_ids.to(torch.int32).contiguous()

    # grid 覆盖每个 (token, slot, n-block) 且各写各的位置：empty 安全；
    # tl.store 落到 A.dtype 指针时自动降精度，与参考实现的舍入点一致。
    Out = torch.empty((T, top_k, N), dtype=A.dtype, device=A.device)

    grid = lambda meta: (T, top_k, triton.cdiv(N, meta["BLOCK_N"]))

    _fused_moe_gemm_single_token_kernel[grid](
        A,
        B,
        topk_weights_c,
        topk_ids_c,
        Out,
        T,
        E,
        N,
        K,
        top_k,
        A.stride(0),
        A.stride(1),
        B.stride(0),
        B.stride(1),
        B.stride(2),
        Out.stride(0),
        Out.stride(1),
        Out.stride(2),
    )

    return Out


__all__ = ["fused_moe_gemm"]
