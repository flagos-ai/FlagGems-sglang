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
import torch.nn.functional as F


def reference(
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
    scores = scores.float()
    bias = bias.float()
    M, N = scores.shape
    K = topk
    K_routed = topk - num_fused_shared_experts
    if routed_scaling_factor is None:
        routed_scaling_factor = 1.0

    if scoring_func == "sigmoid":
        activated = torch.sigmoid(scores)
        biased = activated + bias[None, :]
    elif scoring_func == "sqrtsoftplus":
        activated = torch.sqrt(F.softplus(scores))
        biased = activated + bias[None, :]
    else:
        logit = scores
        if moe_softcapping != 0.0:
            logit = moe_softcapping * torch.tanh(logit / moe_softcapping)
        biased = logit + bias[None, :]
        activated = torch.softmax(biased, dim=-1)

    if num_expert_group > 1:
        experts_per_group = N // num_expert_group
        biased_g = biased.view(M, num_expert_group, experts_per_group)
        top2 = torch.topk(biased_g, 2, dim=-1).values
        group_score = top2.sum(dim=-1)
        keep_idx = torch.topk(group_score, topk_group, dim=-1).indices
        keep_mask_g = torch.zeros(
            M, num_expert_group, dtype=torch.bool, device=scores.device
        )
        keep_mask_g.scatter_(1, keep_idx, True)
        keep_mask = (
            keep_mask_g.unsqueeze(-1)
            .expand(M, num_expert_group, experts_per_group)
            .reshape(M, N)
        )
        biased = torch.where(
            keep_mask, biased, torch.full_like(biased, -float("inf"))
        )

    _, top_idx = torch.topk(biased, K_routed, dim=-1)
    selected_vals = torch.gather(activated, 1, top_idx)
    routed_sum = selected_vals.sum(dim=-1, keepdim=True)

    weights = torch.zeros(M, K, dtype=torch.float32, device=scores.device)
    indices = torch.zeros(M, K, dtype=torch.int32, device=scores.device)
    weights[:, :K_routed] = selected_vals
    indices[:, :K_routed] = top_idx.to(torch.int32)

    num_shared = K - K_routed
    if num_shared > 0:
        shared_weight = routed_sum / routed_scaling_factor
        shared_idx = N + torch.arange(
            num_shared, device=scores.device, dtype=torch.int32
        )
        weights[:, K_routed:] = shared_weight.expand(M, num_shared)
        indices[:, K_routed:] = shared_idx[None, :].expand(M, num_shared)

    if renormalize:
        norm = torch.where(
            routed_sum > 0, routed_sum, torch.ones_like(routed_sum)
        )
        weights = weights / norm
    if apply_routed_scaling_factor_on_output:
        weights = weights * routed_scaling_factor

    return weights, indices
