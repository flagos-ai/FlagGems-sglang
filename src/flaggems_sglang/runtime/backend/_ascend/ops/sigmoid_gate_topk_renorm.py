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

import torch
import triton
import triton.language as tl


@triton.jit
def _sigmoid_gate_topk_renorm_kernel(
    logits,
    bias,
    routed_w,
    indices,
    shared_w,
    logits_stride_m,
    logits_stride_g,
    bias_stride,
    route_scale,
    global_scale,
    route_scale_stride_m,
    global_scale_stride_m,
    N: tl.constexpr,
    S: tl.constexpr,
    K: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_S: tl.constexpr,
    ROUTE_SCALE_IS_TENSOR: tl.constexpr,
    ROUTE_SCALE_IS_ROW: tl.constexpr,
    GLOBAL_SCALE_IS_TENSOR: tl.constexpr,
    GLOBAL_SCALE_IS_ROW: tl.constexpr,
):
    row = tl.program_id(0)

    offs_n = tl.arange(0, BLOCK_N)
    mask_n = offs_n < N

    routed_logits = tl.load(
        logits + row * logits_stride_m + offs_n * logits_stride_g,
        mask=mask_n,
        other=-float("inf"),
    ).to(tl.float32)
    bias_f = tl.load(
        bias + offs_n * bias_stride,
        mask=mask_n,
        other=-float("inf"),
    ).to(tl.float32)

    scores = tl.sigmoid(routed_logits) + bias_f
    scores = tl.where(mask_n, scores, -float("inf"))

    offs_k = tl.arange(0, BLOCK_K)
    mask_k = offs_k < K
    topk_probs = tl.full((BLOCK_K,), 0.0, tl.float32)
    topk_idx = tl.full((BLOCK_K,), 0, tl.int32)

    # Deterministic descending Top-k. Equal scores pick the smaller index; this
    # may differ from torch.topk only on exact ties.
    for slot in tl.static_range(0, K):
        best_score = tl.max(scores, axis=0)
        candidate_idx = tl.where((scores == best_score) & mask_n, offs_n, N)
        best_idx = tl.min(candidate_idx, axis=0)

        selected_logit = tl.max(
            tl.where(offs_n == best_idx, routed_logits, -float("inf")),
            axis=0,
        )
        selected_prob = tl.sigmoid(selected_logit)

        topk_probs = tl.where(offs_k == slot, selected_prob, topk_probs)
        topk_idx = tl.where(offs_k == slot, best_idx.to(tl.int32), topk_idx)
        scores = tl.where(offs_n == best_idx, -float("inf"), scores)

    offs_s = tl.arange(0, BLOCK_S)
    mask_s = offs_s < S
    shared_logits = tl.load(
        logits + row * logits_stride_m + (N + offs_s) * logits_stride_g,
        mask=mask_s,
        other=-float("inf"),
    ).to(tl.float32)
    shared_probs = tl.sigmoid(shared_logits)
    shared_probs = tl.where(mask_s, shared_probs, 0.0)

    denom = tl.sum(topk_probs, axis=0) + tl.sum(shared_probs, axis=0)

    if ROUTE_SCALE_IS_TENSOR:
        if ROUTE_SCALE_IS_ROW:
            route_scale_f = tl.load(
                route_scale + row * route_scale_stride_m
            ).to(tl.float32)
        else:
            route_scale_f = tl.load(route_scale).to(tl.float32)
    else:
        route_scale_f = route_scale

    if GLOBAL_SCALE_IS_TENSOR:
        if GLOBAL_SCALE_IS_ROW:
            global_scale_f = tl.load(
                global_scale + row * global_scale_stride_m
            ).to(tl.float32)
        else:
            global_scale_f = tl.load(global_scale).to(tl.float32)
    else:
        global_scale_f = global_scale

    scale = route_scale_f * global_scale_f
    routed_out = topk_probs / denom * scale
    shared_out = shared_probs / denom * scale

    tl.store(routed_w + row * K + offs_k, routed_out, mask=mask_k)
    tl.store(indices + row * K + offs_k, topk_idx, mask=mask_k)
    tl.store(shared_w + row * S + offs_s, shared_out, mask=mask_s)


def _pow2_at_least_1(x):
    x = int(x)
    if x <= 1:
        return 1
    return triton.next_power_of_2(x)


def _num_warps(block_size):
    if block_size >= 2048:
        return 8
    if block_size >= 512:
        return 4
    return 1


def _row_stride_or_zero(x):
    if torch.is_tensor(x) and x.dim() > 0:
        return x.stride(0)
    return 0


def _is_row_tensor(x):
    return torch.is_tensor(x) and x.numel() > 1


def sigmoid_gate_topk_renorm(
    logits, k, n_shared_experts, route_scale, global_scale, bias
):
    M, G = logits.shape
    K = int(k)
    S = int(n_shared_experts)
    N = G - S

    routed_w = torch.empty((M, K), device=logits.device, dtype=logits.dtype)
    indices = torch.empty((M, K), device=logits.device, dtype=torch.int32)
    shared_w = torch.empty((M, S), device=logits.device, dtype=logits.dtype)

    if M == 0:
        return routed_w, indices, shared_w

    block_n = _pow2_at_least_1(N)
    block_k = _pow2_at_least_1(K)
    block_s = _pow2_at_least_1(S)
    block_max = max(block_n, block_k, block_s)

    _sigmoid_gate_topk_renorm_kernel[(M,)](
        logits,
        bias,
        routed_w,
        indices,
        shared_w,
        logits.stride(0),
        logits.stride(1),
        bias.stride(0),
        route_scale,
        global_scale,
        _row_stride_or_zero(route_scale),
        _row_stride_or_zero(global_scale),
        N,
        S,
        K,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        BLOCK_S=block_s,
        ROUTE_SCALE_IS_TENSOR=torch.is_tensor(route_scale),
        ROUTE_SCALE_IS_ROW=_is_row_tensor(route_scale),
        GLOBAL_SCALE_IS_TENSOR=torch.is_tensor(global_scale),
        GLOBAL_SCALE_IS_ROW=_is_row_tensor(global_scale),
        num_warps=_num_warps(block_max),
    )

    return routed_w, indices, shared_w


reference = sigmoid_gate_topk_renorm


__all__ = ["sigmoid_gate_topk_renorm"]
