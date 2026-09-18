# fused_moe_router_tensorcore (moe/fused_moe_router_tensorcore)

## 任务描述

融合 MoE 路由器（Tensor Core 实现）：与 `fused_moe_router_cudacore` 计算语义完全相同，通过矩阵乘法计算路由 logit，可选 logit 软封顶和专家修正偏置，再以全局 softmax 权重进行 top-k 专家选择。区别在于底层 kernel 使用 `tl.dot`（Tensor Core）加速 GEMM，且限制 `topk <= 2`。

## 接口签名

```python
def reference(x, router_weight, topk, moe_softcapping, correction_bias=None)
```

> 选手实现的函数签名需与上述 `reference(...)` 完全一致。

## 计算定义

输入：
- `x`: `[B, H]` — token 隐藏状态，H 须为 64 的倍数（与 baseline BLOCK_SIZE_K 对齐）
- `router_weight`: `[E, H]` — 路由权重矩阵，E 为专家数
- `topk`: int，满足 `topk <= 2`
- `moe_softcapping`: float — logit 软封顶系数（为 0 时不启用）
- `correction_bias`: `[E]` float32 或 None — 专家修正偏置

计算步骤（与 cudacore 变体数学等价）：

1. 路由 logit：`logits = x.float() @ router_weight.float().T`，形状 `[B, E]`

2. 软封顶（当 `moe_softcapping != 0`）：
   `logits = tanh(logits / cap) * cap`，将 logit 压缩至 `(-cap, cap)` 区间

3. 加修正偏置（当 `correction_bias` 不为 None）：
   `logits = logits + correction_bias`

4. 全局 softmax：`probs = softmax(logits, dim=-1)`，形状 `[B, E]`

5. Top-k 选择（`topk <= 2`）：`topk_ids = argsort(logits, descending=True)[:, :topk]`，形状 `[B, topk]`，为 int32

6. 聚合权重：`topk_weights = gather(probs, dim=-1, index=topk_ids)`，形状 `[B, topk]`，float32
   — 权重来自全 E 专家的 softmax，不对 topk 子集重新归一化

输出：`(topk_weights, topk_ids)`

## 正确性判别标准

Per-dtype tolerance:
- float32: `atol=1e-4, rtol=1e-4`
- bfloat16: `atol=1.5e-2, rtol=1.5e-2`
- float16: `atol=1e-2, rtol=1e-2`

`topk_ids` 需与参考实现精确匹配（exact equality）。

## 参考实现

```python
import torch


def reference(x, router_weight, topk, moe_softcapping, correction_bias=None):
    logits = x.float() @ router_weight.float().t()
    if moe_softcapping != 0:
        logits = torch.tanh(logits / moe_softcapping) * moe_softcapping
    if correction_bias is not None:
        logits = logits + correction_bias.float()

    probs = torch.softmax(logits, dim=-1)
    topk_logits, topk_ids = torch.topk(logits, topk, dim=-1)
    topk_weights = torch.gather(probs, -1, topk_ids)

    return topk_weights, topk_ids.to(torch.int32)
```
