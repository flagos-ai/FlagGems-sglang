# softcap_out (activation_norm/softcap_out)

## 任务描述

Softcap activation (out-of-place): `tanh(x / cap) * cap`, bounding values to [-cap, cap].

## 接口签名

```python
def reference(x, softcap_const)
```

> 选手实现的函数签名需与上述 `reference(...)` 完全一致。

## 计算定义

- `output = tanh(x / softcap_const) * softcap_const`
- 将值限制在 `[-softcap_const, softcap_const]` 范围内
- float32 计算，输出保持 float32

## 正确性判别标准

Per-dtype tolerance:
- float32: `atol=1e-4, rtol=1e-4`
- bfloat16: `atol=1.5e-2, rtol=1.5e-2`
- float16: `atol=1e-2, rtol=1e-2`


## 参考实现

```python
import torch


def reference(x, softcap_const):
    return torch.tanh(x.to(torch.float32) / softcap_const) * softcap_const
```
