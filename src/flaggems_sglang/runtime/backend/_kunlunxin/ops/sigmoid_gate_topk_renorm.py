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
def _sigmoid_f32(x):
    return 1.0 / (1.0 + tl.exp(0.0 - x))


@triton.jit
def _kunlun_tile_topk_kernel(
    logits,
    bias,
    cand_scores,
    cand_logits,
    cand_indices,
    logits_stride_m,
    logits_stride_g,
    bias_stride,
    cand_stride_m,
    cand_stride_t,
    cand_stride_k,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    row = tl.program_id(0)
    tile = tl.program_id(1)
    neg_large = -3.4028234663852886e38

    offs_n = tile * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < N
    routed_logits = tl.load(
        logits + row * logits_stride_m + offs_n * logits_stride_g,
        mask=mask_n,
        other=neg_large,
    ).to(tl.float32)
    bias_f = tl.load(
        bias + offs_n * bias_stride,
        mask=mask_n,
        other=neg_large,
    ).to(tl.float32)
    scores = _sigmoid_f32(routed_logits) + bias_f
    scores = tl.where(mask_n, scores, neg_large)

    offs_k = tl.arange(0, BLOCK_K)
    top_scores = tl.full((BLOCK_K,), neg_large, tl.float32)
    top_logits = tl.full((BLOCK_K,), neg_large, tl.float32)
    top_indices = tl.full((BLOCK_K,), N, tl.int32)

    for slot in tl.static_range(0, K):
        best_score = tl.max(scores, axis=0)
        candidate_idx = tl.where((scores == best_score) & mask_n, offs_n, N)
        best_idx = tl.min(candidate_idx, axis=0)
        selected_logit = tl.max(
            tl.where(offs_n == best_idx, routed_logits, neg_large),
            axis=0,
        )

        top_scores = tl.where(offs_k == slot, best_score, top_scores)
        top_logits = tl.where(offs_k == slot, selected_logit, top_logits)
        top_indices = tl.where(
            offs_k == slot, best_idx.to(tl.int32), top_indices
        )
        scores = tl.where(offs_n == best_idx, neg_large, scores)

    mask_k = offs_k < K
    base = row * cand_stride_m + tile * cand_stride_t + offs_k * cand_stride_k
    tl.store(cand_scores + base, top_scores, mask=mask_k)
    tl.store(cand_logits + base, top_logits, mask=mask_k)
    tl.store(cand_indices + base, top_indices, mask=mask_k)


@triton.jit
def _kunlun_merge_topk_renorm_kernel(
    logits,
    cand_scores,
    cand_logits,
    cand_indices,
    routed_w,
    indices,
    shared_w,
    logits_stride_m,
    logits_stride_g,
    cand_stride_m,
    cand_stride_t,
    cand_stride_k,
    routed_w_stride_m,
    routed_w_stride_k,
    indices_stride_m,
    indices_stride_k,
    shared_w_stride_m,
    shared_w_stride_s,
    route_scale,
    global_scale,
    route_scale_stride_m,
    global_scale_stride_m,
    N: tl.constexpr,
    S: tl.constexpr,
    K: tl.constexpr,
    NUM_TILES: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_K: tl.constexpr,
    ROUTE_SCALE_IS_TENSOR: tl.constexpr,
    ROUTE_SCALE_IS_ROW: tl.constexpr,
    GLOBAL_SCALE_IS_TENSOR: tl.constexpr,
    GLOBAL_SCALE_IS_ROW: tl.constexpr,
):
    row = tl.program_id(0)
    neg_large = -3.4028234663852886e38

    offs_c = tl.arange(0, BLOCK_C)
    tile_ids = offs_c // K
    slot_ids = offs_c - tile_ids * K
    mask_c = offs_c < (NUM_TILES * K)
    cand_offsets = (
        row * cand_stride_m
        + tile_ids * cand_stride_t
        + slot_ids * cand_stride_k
    )

    scores = tl.load(
        cand_scores + cand_offsets, mask=mask_c, other=neg_large
    ).to(tl.float32)
    logits_v = tl.load(
        cand_logits + cand_offsets, mask=mask_c, other=neg_large
    ).to(tl.float32)
    idx_v = tl.load(cand_indices + cand_offsets, mask=mask_c, other=N).to(
        tl.int32
    )
    scores = tl.where(mask_c & (idx_v < N), scores, neg_large)

    offs_k = tl.arange(0, BLOCK_K)
    top_logits = tl.full((BLOCK_K,), neg_large, tl.float32)
    top_indices = tl.full((BLOCK_K,), N, tl.int32)
    denom = 0.0

    for slot in tl.static_range(0, K):
        best_score = tl.max(scores, axis=0)
        candidate_idx = tl.where((scores == best_score) & mask_c, idx_v, N)
        best_idx = tl.min(candidate_idx, axis=0)
        selected_logit = tl.max(
            tl.where(
                (idx_v == best_idx) & (scores == best_score),
                logits_v,
                neg_large,
            ),
            axis=0,
        )
        selected_prob = _sigmoid_f32(selected_logit)

        top_logits = tl.where(offs_k == slot, selected_logit, top_logits)
        top_indices = tl.where(
            offs_k == slot, best_idx.to(tl.int32), top_indices
        )
        denom += selected_prob
        scores = tl.where(
            (idx_v == best_idx) & (scores == best_score), neg_large, scores
        )

    for shared_idx in tl.static_range(0, S):
        shared_logit = tl.load(
            logits + row * logits_stride_m + (N + shared_idx) * logits_stride_g
        ).to(tl.float32)
        denom += _sigmoid_f32(shared_logit)

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

    for slot in tl.static_range(0, K):
        selected_logit = tl.max(
            tl.where(offs_k == slot, top_logits, neg_large),
            axis=0,
        )
        selected_idx = tl.min(
            tl.where(offs_k == slot, top_indices, N),
            axis=0,
        )
        routed_out = _sigmoid_f32(selected_logit) / denom * scale
        tl.store(
            routed_w + row * routed_w_stride_m + slot * routed_w_stride_k,
            routed_out,
        )
        tl.store(
            indices + row * indices_stride_m + slot * indices_stride_k,
            selected_idx.to(tl.int32),
        )

    for shared_idx in tl.static_range(0, S):
        shared_logit = tl.load(
            logits + row * logits_stride_m + (N + shared_idx) * logits_stride_g
        ).to(tl.float32)
        shared_out = _sigmoid_f32(shared_logit) / denom * scale
        tl.store(
            shared_w
            + row * shared_w_stride_m
            + shared_idx * shared_w_stride_s,
            shared_out,
        )


def _pow2_at_least_1(x):
    x = int(x)
    if x <= 1:
        return 1
    return triton.next_power_of_2(x)


def _row_stride_or_zero(x):
    if torch.is_tensor(x) and x.dim() > 0:
        return x.stride(0)
    return 0


def _is_row_tensor(x):
    return torch.is_tensor(x) and x.numel() > 1


def _num_warps_for_block(block_size):
    return 1


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

    tile_n = min(max(N, 1), 128)
    block_n = _pow2_at_least_1(tile_n)
    block_k = _pow2_at_least_1(K)
    num_tiles = triton.cdiv(N, block_n)
    cand_count = num_tiles * max(K, 1)
    block_c = _pow2_at_least_1(cand_count)

    cand_scores = torch.empty(
        (M, num_tiles, K), device=logits.device, dtype=torch.float32
    )
    cand_logits = torch.empty(
        (M, num_tiles, K), device=logits.device, dtype=torch.float32
    )
    cand_indices = torch.empty(
        (M, num_tiles, K), device=logits.device, dtype=torch.int32
    )

    _kunlun_tile_topk_kernel[(M, num_tiles)](
        logits,
        bias,
        cand_scores,
        cand_logits,
        cand_indices,
        logits.stride(0),
        logits.stride(1),
        bias.stride(0),
        cand_scores.stride(0),
        cand_scores.stride(1),
        cand_scores.stride(2),
        N,
        K,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_warps=2 if block_n >= 128 else 1,
        num_stages=1,
    )

    _kunlun_merge_topk_renorm_kernel[(M,)](
        logits,
        cand_scores,
        cand_logits,
        cand_indices,
        routed_w,
        indices,
        shared_w,
        logits.stride(0),
        logits.stride(1),
        cand_scores.stride(0),
        cand_scores.stride(1),
        cand_scores.stride(2),
        routed_w.stride(0),
        routed_w.stride(1),
        indices.stride(0),
        indices.stride(1),
        shared_w.stride(0),
        shared_w.stride(1),
        route_scale,
        global_scale,
        _row_stride_or_zero(route_scale),
        _row_stride_or_zero(global_scale),
        N,
        S,
        K,
        num_tiles,
        BLOCK_C=block_c,
        BLOCK_K=block_k,
        ROUTE_SCALE_IS_TENSOR=torch.is_tensor(route_scale),
        ROUTE_SCALE_IS_ROW=_is_row_tensor(route_scale),
        GLOBAL_SCALE_IS_TENSOR=torch.is_tensor(global_scale),
        GLOBAL_SCALE_IS_ROW=_is_row_tensor(global_scale),
        num_warps=_num_warps_for_block(block_c),
        num_stages=1,
    )

    return routed_w, indices, shared_w


reference = sigmoid_gate_topk_renorm


__all__ = ["sigmoid_gate_topk_renorm"]
