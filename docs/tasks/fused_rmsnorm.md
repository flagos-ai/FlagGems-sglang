# fused_rmsnorm (activation_norm/fused_rmsnorm)

## 任务描述

Fused RMS normalization: `x * rsqrt(mean(x^2) + eps) * weight`, computed in fp32 and cast back.

## 接口签名

```python
def reference(x, weight, eps)
```

> 选手实现的函数签名需与上述 `reference(...)` 完全一致。

## 计算定义

- `rms = sqrt(mean(x^2, dim=-1) + eps)`
- `out = (x / rms) * weight`
- 中间计算使用 float32，输出 cast 回输入 dtype

## 正确性判别标准

Per-dtype tolerance:
- float32: `atol=1e-4, rtol=1e-4`
- bfloat16: `atol=1.5e-2, rtol=1.5e-2`
- float16: `atol=1e-2, rtol=1e-2`


## 参考实现

```python
import torch


def reference(x, weight, eps):
    x32 = x.float()
    rms = torch.sqrt((x32 * x32).mean(dim=-1, keepdim=True) + eps)
    out = (x32 / rms) * weight.float()
    return out.to(x.dtype)
```
