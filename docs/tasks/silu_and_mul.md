# silu_and_mul (activation_norm/silu_and_mul)

## 任务描述

Gated SiLU activation: `out = silu(x[..., :d]) * x[..., d:]`.

## 接口签名

```python
def reference(hidden_states):
```

> 选手实现的函数签名需与上述 `reference(...)` 完全一致。

## 计算定义

- `hidden_states` any floating dtype, last dim even.
- `d = hidden_states.shape[-1] // 2`. Computed in fp32, cast back to input dtype.
- Only the unquantized path is in scope (the kernel's `quantize=`/`scales=`
  fusion is a separate feature, out of scope).

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
    out = F.silu(x1.float()) * x3.float()
    return out.to(hidden_states.dtype)
```
