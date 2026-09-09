# moe_fused_mul_sum (moe/moe_fused_mul_sum)

## 任务描述

融合加权求和合并操作：将每个 token 的 top-k 个专家输出按路由权重加权求和，可选地对专家并行（EP）部署中无效的专家分配进行掩码处理。

## 接口签名

```python
def reference(inputs, topk_weights, topk_ids=None, expert_map=None, routed_scaling_factor=None, is_ep=False)
```

> 选手实现的函数签名需与上述 `reference(...)` 完全一致。

## 计算定义

- 输入 `inputs`: `[T, top_k, D]`；`topk_weights`: `[T, top_k]`；`topk_ids`: `[T, top_k]` int32 或 None；`expert_map`: `[num_experts]` int32 或 None；`routed_scaling_factor`: float 或 None；`is_ep`: bool
- 输出：`[T, D]`，与 `inputs` 同 dtype
- 令 `w = topk_weights * (routed_scaling_factor if routed_scaling_factor is not None else 1.0)`
- 若提供 `expert_map`：`w *= (expert_map[topk_ids] >= 0)`（丢弃路由到本 EP rank 不拥有的专家的 token）
- 否则若 `is_ep=True`：`w *= (topk_ids >= 0)`（丢弃已标记为无效的槽位）
- 最终输出：

  $$\text{out}[t, d] = \sum_{k} \text{inputs}[t, k, d] \times w[t, k]$$

## 正确性判别标准

Per-dtype tolerance:
- float32: `atol=1e-4, rtol=1e-4`
- bfloat16: `atol=1.5e-2, rtol=1.5e-2`
- float16: `atol=1e-2, rtol=1e-2`

## 参考实现

```python
import torch


def reference(inputs, topk_weights, topk_ids=None, expert_map=None, routed_scaling_factor=None, is_ep=False):
    scale = 1.0 if routed_scaling_factor is None else routed_scaling_factor
    w = topk_weights.float() * scale

    if expert_map is not None:
        valid = expert_map[topk_ids.long()] >= 0
        w = w * valid.to(w.dtype)
    elif is_ep:
        valid = topk_ids >= 0
        w = w * valid.to(w.dtype)

    weighted = inputs.float() * w.unsqueeze(-1)
    out = weighted.sum(dim=1)
    return out.to(inputs.dtype)
```
