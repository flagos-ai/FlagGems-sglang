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
def _moe_fused_gate_rows_kernel(
    scores_ptr,
    bias_ptr,
    weights_ptr,
    indices_ptr,
    M,
    stride_sm,
    stride_sn,
    N: tl.constexpr,
    K: tl.constexpr,
    K_ROUTED: tl.constexpr,
    SCORING_FUNC: tl.constexpr,
    HAS_SOFTCAP: tl.constexpr,
    SOFTCAP: tl.constexpr,
    RENORMALIZE: tl.constexpr,
    APPLY_SCALE: tl.constexpr,
    ROUTED_SCALE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    row_offsets = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    expert_offsets = tl.arange(0, BLOCK_N)
    valid_rows = row_offsets < M
    valid_experts = expert_offsets < N
    valid = valid_rows[:, None] & valid_experts[None, :]

    bias = tl.load(
        bias_ptr + expert_offsets,
        mask=valid_experts,
        other=0.0,
    ).to(tl.float32)
    scores = tl.load(
        scores_ptr
        + row_offsets[:, None] * stride_sm
        + expert_offsets[None, :] * stride_sn,
        mask=valid,
        other=0.0,
    ).to(tl.float32)

    if SCORING_FUNC == 0:
        activated = tl.fdiv(1.0, 1.0 + tl.exp(-scores))
        selection = activated + bias[None, :]
    elif SCORING_FUNC == 1:
        softplus = tl.where(
            scores > 20.0,
            scores,
            tl.log(1.0 + tl.exp(scores)),
        )
        activated = tl.sqrt(softplus)
        selection = activated + bias[None, :]
    else:
        logits = scores
        if HAS_SOFTCAP:
            scaled = logits * (1.0 / SOFTCAP)
            logits = SOFTCAP * (2.0 * tl.sigmoid(2.0 * scaled) - 1.0)
        selection = tl.where(valid, logits + bias[None, :], -float("inf"))
        row_max = tl.max(selection, axis=1)[:, None]
        exponentials = tl.where(valid, tl.exp(selection - row_max), 0.0)
        denominator = tl.sum(exponentials, axis=1)[:, None]
        activated = exponentials / denominator

    selection = tl.where(valid, selection, -float("inf"))
    selection = tl.where(selection == selection, selection, -1.0e30)

    rank_offsets = tl.arange(0, BLOCK_K)
    selected_weights = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)
    selected_indices = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.int32)
    routed_sum = tl.zeros((BLOCK_M, 1), dtype=tl.float32)
    remaining = selection
    for rank in tl.static_range(0, K_ROUTED):
        best_value = tl.max(remaining, axis=1)[:, None]
        best_expert = tl.max(
            tl.where(
                remaining == best_value,
                expert_offsets[None, :],
                -1,
            ),
            axis=1,
        )[:, None]
        weight = tl.sum(
            tl.where(
                expert_offsets[None, :] == best_expert,
                activated,
                0.0,
            ),
            axis=1,
        )[:, None]
        slot = rank_offsets[None, :] == rank
        selected_weights = tl.where(slot, weight, selected_weights)
        selected_indices = tl.where(slot, best_expert, selected_indices)
        routed_sum += weight
        remaining = tl.where(
            expert_offsets[None, :] == best_expert,
            -float("inf"),
            remaining,
        )

    rank_mask = rank_offsets < K
    shared_mask = (rank_offsets[None, :] >= K_ROUTED) & rank_mask[None, :]
    values = tl.where(
        shared_mask,
        routed_sum / ROUTED_SCALE,
        selected_weights,
    )
    output_indices = tl.where(
        shared_mask,
        N + rank_offsets[None, :] - K_ROUTED,
        selected_indices,
    )
    if RENORMALIZE:
        denominator = tl.where(routed_sum > 0.0, routed_sum, 1.0)
        values /= denominator
    if APPLY_SCALE:
        values *= ROUTED_SCALE

    output_offsets = row_offsets[:, None] * K + rank_offsets[None, :]
    output_mask = valid_rows[:, None] & rank_mask[None, :]
    tl.store(weights_ptr + output_offsets, values, mask=output_mask)
    tl.store(indices_ptr + output_offsets, output_indices, mask=output_mask)


@triton.jit
def _moe_fused_gate_kernel(
    scores_ptr,
    bias_ptr,
    weights_ptr,
    indices_ptr,
    stride_sm,
    stride_sn,
    N: tl.constexpr,
    K: tl.constexpr,
    K_ROUTED: tl.constexpr,
    SCORING_FUNC: tl.constexpr,
    HAS_SOFTCAP: tl.constexpr,
    SOFTCAP: tl.constexpr,
    USE_OUTPUT_SCRATCH: tl.constexpr,
    RENORMALIZE: tl.constexpr,
    APPLY_SCALE: tl.constexpr,
    ROUTED_SCALE: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    TOPK_GROUP: tl.constexpr,
    EXPERTS_PER_GROUP: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    offsets = tl.arange(0, BLOCK_N)
    valid = offsets < N
    bias = tl.load(bias_ptr + offsets, mask=valid, other=0.0).to(tl.float32)

    row = tl.program_id(0)
    scores = tl.load(
        scores_ptr + row * stride_sm + offsets * stride_sn,
        mask=valid,
        other=0.0,
    ).to(tl.float32)

    if SCORING_FUNC == 0:
        activated = tl.fdiv(1.0, 1.0 + tl.exp(-scores))
        selection = activated + bias
    elif SCORING_FUNC == 1:
        softplus = tl.where(
            scores > 20.0,
            scores,
            tl.log(1.0 + tl.exp(scores)),
        )
        activated = tl.sqrt(softplus)
        selection = activated + bias
    else:
        logits = scores
        if HAS_SOFTCAP:
            scaled = logits * (1.0 / SOFTCAP)
            logits = SOFTCAP * (2.0 * tl.sigmoid(2.0 * scaled) - 1.0)
        selection = tl.where(valid, logits + bias, -float("inf"))
        row_max = tl.max(selection, axis=0)
        exponentials = tl.where(valid, tl.exp(selection - row_max), 0.0)
        denominator = tl.sum(exponentials, axis=0)
        activated = exponentials / denominator

    selection = tl.where(valid, selection, -float("inf"))
    selection = tl.where(selection == selection, selection, -1.0e30)

    if NUM_GROUPS > 1:
        group_of_expert = offsets // EXPERTS_PER_GROUP
        if N == BLOCK_N:
            group_values = tl.reshape(
                selection,
                (NUM_GROUPS, EXPERTS_PER_GROUP),
            )
            local_offsets = tl.arange(0, EXPERTS_PER_GROUP)
            first_values, first_lanes = tl.max(
                group_values,
                axis=1,
                return_indices=True,
                return_indices_tie_break_left=False,
            )
            second_values = tl.where(
                local_offsets != tl.expand_dims(first_lanes, 1),
                group_values,
                -float("inf"),
            )
            group_scores = first_values + tl.max(second_values, axis=1)
            group_offsets = tl.arange(0, NUM_GROUPS)
        else:
            group_scores = tl.full((BLOCK_N,), -float("inf"), tl.float32)
            for group in tl.static_range(0, NUM_GROUPS):
                in_group = valid & (group_of_expert == group)
                group_values = tl.where(in_group, selection, -float("inf"))
                first_value, first_lane = tl.max(
                    group_values,
                    axis=0,
                    return_indices=True,
                    return_indices_tie_break_left=False,
                )
                second_values = tl.where(
                    in_group & (offsets != first_lane),
                    selection,
                    -float("inf"),
                )
                group_score = first_value + tl.max(second_values, axis=0)
                group_scores = tl.where(
                    offsets == group, group_score, group_scores
                )
            group_offsets = offsets

        kept_experts = tl.zeros((BLOCK_N,), dtype=tl.int1)
        remaining_groups = group_scores
        for _ in tl.static_range(0, TOPK_GROUP):
            best_group = tl.argmax(
                remaining_groups,
                axis=0,
                tie_break_left=False,
            )
            kept_experts = kept_experts | (group_of_expert == best_group)
            remaining_groups = tl.where(
                group_offsets == best_group,
                -float("inf"),
                remaining_groups,
            )
        selection = tl.where(kept_experts & valid, selection, -float("inf"))

    rank_offsets = tl.arange(0, BLOCK_K)
    selected_weights = tl.zeros((BLOCK_K,), dtype=tl.float32)
    selected_indices = tl.zeros((BLOCK_K,), dtype=tl.int32)
    routed_sum = 0.0
    remaining = selection
    output_start = row * K
    for rank in tl.static_range(0, K_ROUTED):
        best_value = tl.max(remaining, axis=0)
        best_expert = tl.max(
            tl.where(remaining == best_value, offsets, -1),
            axis=0,
        )
        weight = tl.sum(
            tl.where(offsets == best_expert, activated, 0.0),
            axis=0,
        )
        if USE_OUTPUT_SCRATCH:
            tl.store(weights_ptr + output_start + rank, weight)
            tl.store(indices_ptr + output_start + rank, best_expert)
        else:
            selected_weights = tl.where(
                rank_offsets == rank,
                weight,
                selected_weights,
            )
            selected_indices = tl.where(
                rank_offsets == rank,
                best_expert,
                selected_indices,
            )
        routed_sum += weight
        remaining = tl.where(
            offsets == best_expert,
            -float("inf"),
            remaining,
        )

    rank_mask = rank_offsets < K
    if USE_OUTPUT_SCRATCH:
        for shared_rank in tl.static_range(K_ROUTED, K):
            tl.store(
                weights_ptr + output_start + shared_rank,
                routed_sum / ROUTED_SCALE,
            )
            tl.store(
                indices_ptr + output_start + shared_rank,
                N + shared_rank - K_ROUTED,
            )
        if RENORMALIZE or APPLY_SCALE:
            tl.debug_barrier()
            values = tl.load(
                weights_ptr + output_start + rank_offsets,
                mask=rank_mask,
                other=0.0,
            )
            if RENORMALIZE:
                denominator = tl.where(routed_sum > 0.0, routed_sum, 1.0)
                values /= denominator
            if APPLY_SCALE:
                values *= ROUTED_SCALE
            tl.store(
                weights_ptr + output_start + rank_offsets,
                values,
                mask=rank_mask,
            )
    else:
        shared_mask = (rank_offsets >= K_ROUTED) & rank_mask
        values = tl.where(
            shared_mask,
            routed_sum / ROUTED_SCALE,
            selected_weights,
        )
        output_indices = tl.where(
            shared_mask,
            N + rank_offsets - K_ROUTED,
            selected_indices,
        )
        if RENORMALIZE:
            denominator = tl.where(routed_sum > 0.0, routed_sum, 1.0)
            values /= denominator
        if APPLY_SCALE:
            values *= ROUTED_SCALE
        tl.store(
            weights_ptr + output_start + rank_offsets,
            values,
            mask=rank_mask,
        )
        tl.store(
            indices_ptr + output_start + rank_offsets,
            output_indices,
            mask=rank_mask,
        )


def moe_fused_gate(
    scores,
    bias,
    topk,
    scoring_func="sigmoid",
    num_fused_shared_experts=0,
    renormalize=True,
    routed_scaling_factor=1.0,
    apply_routed_scaling_factor_on_output=False,
    moe_softcapping=0.0,
    num_expert_group=1,
    topk_group=1,
):
    if scoring_func == "sigmoid":
        scoring_func_id = 0
    elif scoring_func == "sqrtsoftplus":
        scoring_func_id = 1
    elif scoring_func == "softmax":
        scoring_func_id = 2
    else:
        raise ValueError("unsupported scoring_func")

    if routed_scaling_factor is None:
        routed_scaling_factor = 1.0

    M, N = scores.shape
    K = int(topk)
    K_routed = K - int(num_fused_shared_experts)
    if K_routed <= 0:
        raise ValueError("topk must be greater than num_fused_shared_experts")
    if num_expert_group > 1:
        if N % num_expert_group != 0:
            raise ValueError(
                "num_experts must be divisible by num_expert_group"
            )
        if topk_group < 1 or topk_group > num_expert_group:
            raise ValueError("invalid topk_group")

    weights = torch.empty((M, K), dtype=torch.float32, device=scores.device)
    indices = torch.empty((M, K), dtype=torch.int32, device=scores.device)
    if M == 0:
        return weights, indices

    block_n = max(16, triton.next_power_of_2(N))
    if block_n <= 512:
        num_warps = 1
    else:
        num_warps = 4
    block_k = max(16, triton.next_power_of_2(K))

    max_grid_x = 32768
    if M <= max_grid_x:
        if (
            num_expert_group == 1
            and block_n == 128
            and scores.dtype == torch.float32
            and M >= 2
        ):
            block_m = 2
            _moe_fused_gate_rows_kernel[(triton.cdiv(M, block_m),)](
                scores,
                bias,
                weights,
                indices,
                M,
                scores.stride(0),
                scores.stride(1),
                N=N,
                K=K,
                K_ROUTED=K_routed,
                SCORING_FUNC=scoring_func_id,
                HAS_SOFTCAP=bool(moe_softcapping != 0.0),
                SOFTCAP=float(moe_softcapping),
                RENORMALIZE=bool(renormalize),
                APPLY_SCALE=bool(apply_routed_scaling_factor_on_output),
                ROUTED_SCALE=float(routed_scaling_factor),
                BLOCK_M=block_m,
                BLOCK_N=block_n,
                BLOCK_K=block_k,
                num_warps=num_warps,
                num_stages=1,
            )
        else:
            _moe_fused_gate_kernel[(M,)](
                scores,
                bias,
                weights,
                indices,
                scores.stride(0),
                scores.stride(1),
                N=N,
                K=K,
                K_ROUTED=K_routed,
                SCORING_FUNC=scoring_func_id,
                HAS_SOFTCAP=bool(moe_softcapping != 0.0),
                SOFTCAP=float(moe_softcapping),
                USE_OUTPUT_SCRATCH=False,
                RENORMALIZE=bool(renormalize),
                APPLY_SCALE=bool(apply_routed_scaling_factor_on_output),
                ROUTED_SCALE=float(routed_scaling_factor),
                NUM_GROUPS=int(num_expert_group),
                TOPK_GROUP=int(topk_group),
                EXPERTS_PER_GROUP=N // int(num_expert_group),
                BLOCK_N=block_n,
                BLOCK_K=block_k,
                num_warps=num_warps,
                num_stages=1,
            )
        return weights, indices

    for row_start in range(0, M, max_grid_x):
        grid_rows = min(max_grid_x, M - row_start)
        row_stop = row_start + grid_rows
        score_chunk = scores[row_start:row_stop]
        weight_chunk = weights[row_start:row_stop]
        index_chunk = indices[row_start:row_stop]
        if (
            num_expert_group == 1
            and block_n == 128
            and scores.dtype == torch.float32
            and grid_rows >= 2
        ):
            block_m = 2
            _moe_fused_gate_rows_kernel[(triton.cdiv(grid_rows, block_m),)](
                score_chunk,
                bias,
                weight_chunk,
                index_chunk,
                grid_rows,
                scores.stride(0),
                scores.stride(1),
                N=N,
                K=K,
                K_ROUTED=K_routed,
                SCORING_FUNC=scoring_func_id,
                HAS_SOFTCAP=bool(moe_softcapping != 0.0),
                SOFTCAP=float(moe_softcapping),
                RENORMALIZE=bool(renormalize),
                APPLY_SCALE=bool(apply_routed_scaling_factor_on_output),
                ROUTED_SCALE=float(routed_scaling_factor),
                BLOCK_M=block_m,
                BLOCK_N=block_n,
                BLOCK_K=block_k,
                num_warps=num_warps,
                num_stages=1,
            )
            continue
        _moe_fused_gate_kernel[(grid_rows,)](
            score_chunk,
            bias,
            weight_chunk,
            index_chunk,
            scores.stride(0),
            scores.stride(1),
            N=N,
            K=K,
            K_ROUTED=K_routed,
            SCORING_FUNC=scoring_func_id,
            HAS_SOFTCAP=bool(moe_softcapping != 0.0),
            SOFTCAP=float(moe_softcapping),
            USE_OUTPUT_SCRATCH=False,
            RENORMALIZE=bool(renormalize),
            APPLY_SCALE=bool(apply_routed_scaling_factor_on_output),
            ROUTED_SCALE=float(routed_scaling_factor),
            NUM_GROUPS=int(num_expert_group),
            TOPK_GROUP=int(topk_group),
            EXPERTS_PER_GROUP=N // int(num_expert_group),
            BLOCK_N=block_n,
            BLOCK_K=block_k,
            num_warps=num_warps,
            num_stages=1,
        )
    return weights, indices


__all__ = ["moe_fused_gate"]
