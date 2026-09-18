# gelu_and_mul (activation_norm/gelu_and_mul)

## 任务描述

门控 GELU 激活：将输入在最后一维对半分为 gate 和 up 两部分，对 gate 施加精确（erf-based）GELU 激活后与 up 逐元素相乘，输出维度为输入的一半。

## 接口签名

```python
def reference(hidden_states)
```

> 选手实现的函数签名需与上述 `reference(...)` 完全一致。

## 计算定义

- 输入 `hidden_states`: `[bs, 2*d]`，任意浮点 dtype
- 输出：`[bs, d]`，与输入同 dtype
- 令 `d = hidden_states.shape[-1] // 2`：
  - `x1 = hidden_states[..., :d]`（gate 部分）
  - `x3 = hidden_states[..., d:]`（up 部分）
  - `out = gelu(x1.float(), approximate="none") * x3.float()`，转回输入 dtype
- GELU 使用精确 erf 公式：`gelu(x) = x * Φ(x) = x * (1 + erf(x / sqrt(2))) / 2`，不使用 tanh 近似

## 正确性判别标准

Per-dtype tolerance:
- float32: `atol=1e-4, rtol=1e-4`
- bfloat16: `atol=1.5e-2, rtol=1.5e-2`
- float16: `atol=1e-2, rtol=1e-2`

## 参考实现

```python
import torch.nn.functional as F


def reference(hidden_states):
    d = hidden_states.shape[-1] // 2
    x1, x3 = hidden_states[..., :d], hidden_states[..., d:]
    out = F.gelu(x1.float(), approximate="none") * x3.float()
    return out.to(hidden_states.dtype)
```
