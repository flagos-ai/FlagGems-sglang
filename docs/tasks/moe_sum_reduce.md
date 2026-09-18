# moe_sum_reduce (moe/moe_sum_reduce)

## 任务描述

MoE sum reduction: reduces expert outputs by summing across the top-k dimension with routing weights applied.

## 接口签名

```python
def reference(input, routed_scaling_factor)
```

> 选手实现的函数签名需与上述 `reference(...)` 完全一致。

## 计算定义

- `input`: `[num_tokens, top_k, hidden_dim]`
- `routed_scaling_factor`: scalar
- `output = input.sum(dim=1) * routed_scaling_factor`
- float32 累加，输出 cast 回输入 dtype

## 正确性判别标准

Per-dtype tolerance:
- float32: `atol=1e-4, rtol=1e-4`
- bfloat16: `atol=1.5e-2, rtol=1.5e-2`
- float16: `atol=1e-2, rtol=1e-2`


## 参考实现

```python
import torch


def reference(input, routed_scaling_factor):
    return input.float().sum(dim=1).mul(routed_scaling_factor).to(input.dtype)
```
