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
            first_values = tl.max(group_values, axis=1)
            first_lanes = tl.max(
                tl.where(
                    group_values == tl.expand_dims(first_values, 1),
                    local_offsets,
                    -1,
                ),
                axis=1,
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
                first_value = tl.max(group_values, axis=0)
                first_lane = tl.max(
                    tl.where(
                        group_values == first_value,
                        offsets,
                        -1,
                    ),
                    axis=0,
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
            best_group_score = tl.max(remaining_groups, axis=0)
            best_group = tl.max(
                tl.where(
                    remaining_groups == best_group_score,
                    group_offsets,
                    -1,
                ),
                axis=0,
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
    tl.store(weights_ptr + output_start + rank_offsets, values, mask=rank_mask)
    tl.store(indices_ptr + output_start + rank_offsets, output_indices, mask=rank_mask)


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
    NUM_GROUPS: tl.constexpr,
    TOPK_GROUP: tl.constexpr,
    EXPERTS_PER_GROUP: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GATHER_BACK: tl.constexpr,
):
    # Anchored tile origin: the last program slides back to cover [M-BLOCK_M, M),
    # so every program reads/writes full rows and no row mask is needed. Rows
    # covered by two programs are recomputed identically (idempotent stores).
    row0 = tl.minimum(tl.program_id(0) * BLOCK_M, M - BLOCK_M)
    rows = row0 + tl.arange(0, BLOCK_M)
    cols = tl.arange(0, BLOCK_N)

    bias = tl.load(bias_ptr + cols).to(tl.float32)
    scores = tl.load(
        scores_ptr + rows[:, None] * stride_sm + cols[None, :] * stride_sn
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
        selection = logits + bias[None, :]
        row_max = tl.max(selection, axis=1)[:, None]
        exponentials = tl.exp(selection - row_max)
        denominator = tl.sum(exponentials, axis=1)[:, None]
        activated = exponentials / denominator

    selection = tl.where(selection == selection, selection, -1.0e30)

    if NUM_GROUPS > 1:
        # Per-group top-2 sums via a contiguous row-split view (N == BLOCK_N).
        sel_g = tl.reshape(selection, (BLOCK_M * NUM_GROUPS, EXPERTS_PER_GROUP))
        local_offsets = tl.arange(0, EXPERTS_PER_GROUP)
        first_values = tl.max(sel_g, axis=1)
        first_lanes = tl.max(
            tl.where(
                sel_g == first_values[:, None],
                local_offsets[None, :],
                -1,
            ),
            axis=1,
        )
        second_values = tl.where(
            local_offsets[None, :] != first_lanes[:, None],
            sel_g,
            -float("inf"),
        )
        group_scores = first_values + tl.max(second_values, axis=1)

        gs = tl.reshape(group_scores, (BLOCK_M, NUM_GROUPS))
        gcols = tl.arange(0, NUM_GROUPS)
        kept = tl.zeros((BLOCK_M, NUM_GROUPS), dtype=tl.int1)
        remaining_g = gs
        for _ in tl.static_range(0, TOPK_GROUP):
            best_g = tl.max(remaining_g, axis=1)
            best_gid = tl.max(
                tl.where(
                    remaining_g == best_g[:, None],
                    gcols[None, :],
                    -1,
                ),
                axis=1,
            )
            kept = kept | (gcols[None, :] == best_gid[:, None])
            remaining_g = tl.where(
                gcols[None, :] == best_gid[:, None],
                -float("inf"),
                remaining_g,
            )
        kept_flat = tl.reshape(kept, (BLOCK_M * NUM_GROUPS,))
        sel_g = tl.where(kept_flat[:, None], sel_g, -float("inf"))
        selection = tl.reshape(sel_g, (BLOCK_M, BLOCK_N))

    rank_offsets = tl.arange(0, BLOCK_K)
    rank_mask = rank_offsets < K
    shared_mask = (rank_offsets >= K_ROUTED) & rank_mask
    selected_indices = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.int32)
    routed_sum = tl.zeros((BLOCK_M,), dtype=tl.float32)
    if not GATHER_BACK:
        selected_weights = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)
    remaining = selection
    for rank in tl.static_range(0, K_ROUTED):
        best_value = tl.max(remaining, axis=1)
        best_expert = tl.max(
            tl.where(
                remaining == best_value[:, None],
                cols[None, :],
                -1,
            ),
            axis=1,
        )
        if not GATHER_BACK:
            weight = tl.sum(
                tl.where(
                    cols[None, :] == best_expert[:, None],
                    activated,
                    0.0,
                ),
                axis=1,
            )
            selected_weights = tl.where(
                rank_offsets[None, :] == rank,
                weight[:, None],
                selected_weights,
            )
            routed_sum += weight
        selected_indices = tl.where(
            rank_offsets[None, :] == rank,
            best_expert[:, None],
            selected_indices,
        )
        remaining = tl.where(
            cols[None, :] == best_expert[:, None],
            -float("inf"),
            remaining,
        )

    if GATHER_BACK:
        # Recompute activations only for the K_ROUTED selected experts by
        # gathering their scores (same expression as above => bitwise-equal
        # values), replacing the per-round onehot extraction tree.
        gmask = rank_offsets[None, :] < K_ROUTED
        g_scores = tl.load(
            scores_ptr
            + rows[:, None] * stride_sm
            + selected_indices * stride_sn,
            mask=gmask,
            other=0.0,
        ).to(tl.float32)
        if SCORING_FUNC == 0:
            g_act = tl.fdiv(1.0, 1.0 + tl.exp(-g_scores))
        elif SCORING_FUNC == 1:
            g_softplus = tl.where(
                g_scores > 20.0,
                g_scores,
                tl.log(1.0 + tl.exp(g_scores)),
            )
            g_act = tl.sqrt(g_softplus)
        else:
            g_act = activated * 0.0
        g_act = tl.where(gmask, g_act, 0.0)
        routed_sum = tl.sum(g_act, axis=1)
        values = tl.where(
            shared_mask,
            routed_sum[:, None] / ROUTED_SCALE,
            g_act,
        )
    else:
        values = tl.where(
            shared_mask,
            routed_sum[:, None] / ROUTED_SCALE,
            selected_weights,
        )
    output_indices = tl.where(
        shared_mask,
        N + rank_offsets[None, :] - K_ROUTED,
        selected_indices,
    )
    if RENORMALIZE:
        denominator = tl.where(routed_sum > 0.0, routed_sum, 1.0)
        values = values / denominator[:, None]
    if APPLY_SCALE:
        values = values * ROUTED_SCALE
    out_offsets = rows[:, None] * K + rank_offsets[None, :]
    tl.store(weights_ptr + out_offsets, values, mask=rank_mask[None, :])
    tl.store(indices_ptr + out_offsets, output_indices, mask=rank_mask[None, :])


def _prev_pow2(n):
    p = 1
    while p * 2 <= n:
        p *= 2
    return p


def _rows_kernel_eligible(n, num_expert_group):
    # The rows kernel loads full [BLOCK_M, BLOCK_N] tiles unmasked, so it
    # requires N to be a power of two >= 16. The group section uses a
    # row-split reshape plus tl.arange(0, EXPERTS_PER_GROUP), which requires
    # power-of-two group boundaries (and a width-2 minimum for the local
    # offset vector).
    if n < 16 or n != triton.next_power_of_2(n):
        return False
    if num_expert_group > 1:
        if num_expert_group != triton.next_power_of_2(num_expert_group):
            return False
        if n % num_expert_group != 0:
            return False
        experts_per_group = n // num_expert_group
        if experts_per_group < 2 or experts_per_group != triton.next_power_of_2(
            experts_per_group
        ):
            return False
    return True


# Per-process first-call validation state for the rows kernel:
# key -> True (validated, use directly) / False (fall back to the row kernel).
_ROWS_KERNEL_STATE = {}


def _launch_rows_kernel(
    score_chunk,
    bias,
    weight_chunk,
    index_chunk,
    rows,
    n,
    k,
    k_routed,
    scoring_func_id,
    has_softcap,
    softcap,
    renormalize,
    apply_scale,
    routed_scale,
    num_expert_group,
    topk_group,
    block_m,
    gather_back,
    stride_sm,
    stride_sn,
    block_k,
):
    key = (
        rows,
        n,
        k,
        k_routed,
        scoring_func_id,
        has_softcap,
        softcap,
        renormalize,
        apply_scale,
        routed_scale,
        num_expert_group,
        topk_group,
        block_m,
        gather_back,
        score_chunk.dtype,
    )
    state = _ROWS_KERNEL_STATE.get(key)
    if state is True:
        _moe_fused_gate_rows_kernel[(triton.cdiv(rows, block_m),)](
            score_chunk,
            bias,
            weight_chunk,
            index_chunk,
            rows,
            stride_sm,
            stride_sn,
            N=n,
            K=k,
            K_ROUTED=k_routed,
            SCORING_FUNC=scoring_func_id,
            HAS_SOFTCAP=has_softcap,
            SOFTCAP=softcap,
            RENORMALIZE=renormalize,
            APPLY_SCALE=apply_scale,
            ROUTED_SCALE=routed_scale,
            NUM_GROUPS=num_expert_group,
            TOPK_GROUP=topk_group,
            EXPERTS_PER_GROUP=n // num_expert_group,
            BLOCK_M=block_m,
            BLOCK_N=n,
            BLOCK_K=block_k,
            GATHER_BACK=gather_back,
            num_warps=1,
            num_stages=1,
        )
        return True
    if state is False:
        return False

    # First call for this configuration: compile+run the rows kernel into
    # scratch buffers and cross-check it against the (generic-semantics) row
    # kernel before adopting it. A compile failure (e.g. an environment-level
    # backend regression) or a value mismatch permanently disables the rows
    # path for this configuration in this process.
    try:
        w_ref = torch.empty_like(weight_chunk)
        i_ref = torch.empty_like(index_chunk)
        _moe_fused_gate_rows_kernel[(triton.cdiv(rows, block_m),)](
            score_chunk,
            bias,
            w_ref,
            i_ref,
            rows,
            stride_sm,
            stride_sn,
            N=n,
            K=k,
            K_ROUTED=k_routed,
            SCORING_FUNC=scoring_func_id,
            HAS_SOFTCAP=has_softcap,
            SOFTCAP=softcap,
            RENORMALIZE=renormalize,
            APPLY_SCALE=apply_scale,
            ROUTED_SCALE=routed_scale,
            NUM_GROUPS=num_expert_group,
            TOPK_GROUP=topk_group,
            EXPERTS_PER_GROUP=n // num_expert_group,
            BLOCK_M=block_m,
            BLOCK_N=n,
            BLOCK_K=block_k,
            GATHER_BACK=gather_back,
            num_warps=1,
            num_stages=1,
        )
    except Exception:
        _ROWS_KERNEL_STATE[key] = False
        return False
    _moe_fused_gate_kernel[(rows,)](
        score_chunk,
        bias,
        weight_chunk,
        index_chunk,
        stride_sm,
        stride_sn,
        N=n,
        K=k,
        K_ROUTED=k_routed,
        SCORING_FUNC=scoring_func_id,
        HAS_SOFTCAP=has_softcap,
        SOFTCAP=softcap,
        RENORMALIZE=renormalize,
        APPLY_SCALE=apply_scale,
        ROUTED_SCALE=routed_scale,
        NUM_GROUPS=num_expert_group,
        TOPK_GROUP=topk_group,
        EXPERTS_PER_GROUP=n // num_expert_group,
        BLOCK_N=max(16, triton.next_power_of_2(n)),
        BLOCK_K=block_k,
        num_warps=1 if n <= 512 else 4,
        num_stages=1,
    )
    ok = torch.equal(i_ref, index_chunk) and torch.allclose(
        w_ref, weight_chunk, atol=1e-4, rtol=1e-4
    )
    _ROWS_KERNEL_STATE[key] = bool(ok)
    if ok:
        weight_chunk.copy_(w_ref)
        index_chunk.copy_(i_ref)
        return True
    return False


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

    num_expert_group = int(num_expert_group)
    topk_group = int(topk_group)
    block_k = max(16, triton.next_power_of_2(K))
    rows_eligible = _rows_kernel_eligible(N, num_expert_group)
    # gather-back re-computes activations from gathered scores; only the
    # sigmoid/sqrtsoftplus activations are reproducible from a single score.
    gather_back = scoring_func_id != 2

    max_grid_x = 32768
    for row_start in range(0, M, max_grid_x):
        rows = min(max_grid_x, M - row_start)
        row_stop = row_start + rows
        score_chunk = scores[row_start:row_stop]
        weight_chunk = weights[row_start:row_stop]
        index_chunk = indices[row_start:row_stop]

        block_m = 0
        if rows_eligible and rows >= 192:
            # Measured on L600 (48 SP): one program per row batch, num_warps=1.
            # 192 <= rows <= 768 keeps cdiv(rows, 16) <= 48 (single wave with
            # BLOCK_M=16); larger batches halve the wave count with BLOCK_M=32.
            block_m = 16 if rows <= 768 else 32
            block_m = min(block_m, _prev_pow2(rows))
            if block_m < 2:
                block_m = 0
        if block_m >= 2:
            launched = _launch_rows_kernel(
                score_chunk,
                bias,
                weight_chunk,
                index_chunk,
                rows,
                N,
                K,
                K_routed,
                scoring_func_id,
                bool(moe_softcapping != 0.0),
                float(moe_softcapping),
                bool(renormalize),
                bool(apply_routed_scaling_factor_on_output),
                float(routed_scaling_factor),
                num_expert_group,
                topk_group,
                block_m,
                gather_back,
                scores.stride(0),
                scores.stride(1),
                block_k,
            )
            if launched:
                continue

        _moe_fused_gate_kernel[(rows,)](
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
            RENORMALIZE=bool(renormalize),
            APPLY_SCALE=bool(apply_routed_scaling_factor_on_output),
            ROUTED_SCALE=float(routed_scaling_factor),
            NUM_GROUPS=num_expert_group,
            TOPK_GROUP=topk_group,
            EXPERTS_PER_GROUP=N // num_expert_group,
            BLOCK_N=max(16, triton.next_power_of_2(N)),
            BLOCK_K=block_k,
            num_warps=1 if N <= 512 else 4,
            num_stages=1,
        )
    return weights, indices


__all__ = ["moe_fused_gate"]
