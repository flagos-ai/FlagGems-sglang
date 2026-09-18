# moe_fused_gate (moe/moe_fused_gate)

## 任务描述

Triton 融合 MoE 路由门控：对已计算好的门控 logits 执行评分、加偏置、可选分组 top-k 选择、可选重归一化及输出缩放，输出每个 token 的 top-k 专家权重和索引。

## 接口签名

```python
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
)
```

> 选手实现的函数签名需与上述 `reference(...)` 完全一致。

## 计算定义

- 输入 `scores`: `[M, N]`，fp32/fp16/bf16；`bias`: `[N]`，float32
- 输出：`(weights: [M, topk] float32, indices: [M, topk] int32)`
- 评分函数（`sel` 用于选择，`activated` 用于输出权重，二者仅在 sigmoid/sqrtsoftplus 时不同）：
  - `sigmoid`: `activated = sigmoid(scores)`；`sel = activated + bias`
  - `sqrtsoftplus`: `activated = sqrt(softplus(scores))`；`sel = activated + bias`
  - `softmax`: 若 `moe_softcapping != 0` 先对 scores 做 `tanh` 软截断，再 `sel = activated = softmax(logit + bias)`
- 若 `num_expert_group > 1`（DeepSeek-V3 分组路由）：将 N 个专家等分为 `num_expert_group` 组，每组得分 = 组内 top-2 的 `sel` 之和，保留得分最高的 `topk_group` 组，其余专家的 `sel` 置为 `-inf`
- 按 `sel` 取前 `K_routed = topk - num_fused_shared_experts` 个专家，其输出权重取对应的 `activated`（无 bias）
- 若 `num_fused_shared_experts > 0`：剩余 `topk` 槽的权重 = `routed_sum / routed_scaling_factor`，索引 = `N + slot_offset`
- 若 `renormalize`：所有权重除以 `routed_sum`（防止除零）
- 若 `apply_routed_scaling_factor_on_output`：所有权重乘以 `routed_scaling_factor`

## 正确性判别标准

Per-dtype tolerance:
- float32: `atol=1e-4, rtol=1e-4`
- bfloat16: `atol=1.5e-2, rtol=1.5e-2`
- float16: `atol=1e-2, rtol=1e-2`

## 参考实现

```python
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
        keep_mask_g = torch.zeros(M, num_expert_group, dtype=torch.bool, device=scores.device)
        keep_mask_g.scatter_(1, keep_idx, True)
        keep_mask = (
            keep_mask_g.unsqueeze(-1)
            .expand(M, num_expert_group, experts_per_group)
            .reshape(M, N)
        )
        biased = torch.where(keep_mask, biased, torch.full_like(biased, -float("inf")))

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
        shared_idx = N + torch.arange(num_shared, device=scores.device, dtype=torch.int32)
        weights[:, K_routed:] = shared_weight.expand(M, num_shared)
        indices[:, K_routed:] = shared_idx[None, :].expand(M, num_shared)

    if renormalize:
        norm = torch.where(routed_sum > 0, routed_sum, torch.ones_like(routed_sum))
        weights = weights / norm
    if apply_routed_scaling_factor_on_output:
        weights = weights * routed_scaling_factor

    return weights, indices
```
