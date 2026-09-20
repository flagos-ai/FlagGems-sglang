# sigmoid_gate_topk_renorm (moe/sigmoid_gate_topk_renorm)

## 任务描述

Sigmoid 门控 Top-k 归一化路由：使用 sigmoid 激活加偏置进行专家打分，选取 top-k 路由专家，再将路由专家与共享专家的 sigmoid 权重联合归一化后按比例缩放，输出各专家权重及路由专家索引。

## 接口签名

```python
def reference(logits, k, n_shared_experts, route_scale, global_scale, bias)
```

> 选手实现的函数签名需与上述 `reference(...)` 完全一致。

## 计算定义

输入：
- `logits`: `[T, N+S]` — T 个 token，N 个路由专家 + S 个共享专家的原始 logit（最后 S 列为共享专家）
- `k`: int — 每个 token 选取的路由专家数
- `n_shared_experts` (S): int — 共享专家数
- `route_scale`: float — 路由缩放系数
- `global_scale`: `[1]` float32 — 全局缩放系数
- `bias`: `[N]` float32 — 路由专家打分偏置

计算步骤（均在 float32 精度下执行）：

1. 分离路由与共享 logit：
   - 路由 logit：`routed_logits = logits[:, :N]`，形状 `[T, N]`
   - 共享 logit：`shared_logits = logits[:, N:N+S]`，形状 `[T, S]`

2. 带偏置的 sigmoid 打分（用于选择，不用于最终权重）：
   `sel = sigmoid(routed_logits) + bias`，形状 `[T, N]`

3. Top-k 选择：`indices = argsort(sel, descending=True)[:, :k]`，形状 `[T, k]`，int32

4. 聚合激活 logit（使用原始未 sigmoid 的 logit）：
   - 聚合路由：`routed_vals = gather(routed_logits, dim=1, index=indices)`，形状 `[T, k]`
   - 拼接共享：`active = cat([routed_vals, shared_logits], dim=-1)`，形状 `[T, k+S]`

5. Sigmoid 归一化权重：
   `probs = sigmoid(active)`
   `weights = probs / probs.sum(dim=-1, keepdim=True)`
   — 路由与共享专家联合归一化

6. 缩放：`weights = weights * route_scale * global_scale`

7. 分割输出：
   - `routed_weights = weights[:, :k]`，转换为输入 dtype，形状 `[T, k]`
   - `shared_weights = weights[:, k:]`，转换为输入 dtype，形状 `[T, S]`

输出：`(routed_weights, indices, shared_weights)`

## 正确性判别标准

Per-dtype tolerance:
- float32: `atol=1e-4, rtol=1e-4`
- bfloat16: `atol=1.5e-2, rtol=1.5e-2`
- float16: `atol=1e-2, rtol=1e-2`

`indices` 需与参考实现精确匹配（exact equality）。

## 参考实现

```python
import torch


def reference(logits, k, n_shared_experts, route_scale, global_scale, bias):
    M, G = logits.shape
    N = G - n_shared_experts
    S = n_shared_experts

    logits_f = logits.float()
    routed_logits = logits_f[:, :N]
    sel = torch.sigmoid(routed_logits) + bias.float()[None, :]

    _, idx = torch.topk(sel, k, dim=-1)
    routed_vals = torch.gather(routed_logits, 1, idx)
    shared_vals = logits_f[:, N:N + S]

    active = torch.cat([routed_vals, shared_vals], dim=-1)
    probs = torch.sigmoid(active)
    weights = probs / probs.sum(dim=-1, keepdim=True)
    weights = weights * route_scale * global_scale.float()

    routed_w = weights[:, :k].to(logits.dtype)
    shared_w = weights[:, k:].to(logits.dtype)
    indices = idx.to(torch.int32)
    return routed_w, indices, shared_w
```
