# per_token_group_quant_int8 (quantization/per_token_group_quant_int8)

## 任务描述

逐 token 分组 INT8 量化：将输入张量最后一维按 `group_size` 切分为若干组，每组独立计算基于绝对最大值的缩放因子并量化为 int8 整数；输出量化整数张量和每组一个的 float32 缩放因子张量。

## 接口签名

```python
def reference(x, group_size, dtype=torch.int8)
```

> 选手实现的函数签名需与上述 `reference(...)` 完全一致。

## 计算定义

- 输入 `x`: `[..., K]`，任意浮点 dtype，连续存储；`K` 必须能被 `group_size` 整除
- 输出：`(x_q: [..., K] int8, x_s: [..., K // group_size] float32)`
- 将 `x` 在最后一维以 `group_size` 为单位分组，对每组 `g`：
  - `scale = max(|x[g]|, ε) / 127`，其中 `ε = 1e-10`
  - `x_q[g] = clamp(trunc(x[g] / scale), -128, 127)`（向零截断后转 int8）
  - `x_s[g] = scale`（float32）
- 缩放因子 shape 为 `x.shape[:-1] + (K // group_size,)`

## 正确性判别标准

Per-dtype tolerance:
- float32: `atol=1e-4, rtol=1e-4`
- bfloat16: `atol=1.5e-2, rtol=1.5e-2`
- float16: `atol=1e-2, rtol=1e-2`

## 参考实现

```python
import torch

_EPS = 1e-10


def reference(x, group_size, dtype=torch.int8):
    iinfo = torch.iinfo(dtype)
    int8_min, int8_max = iinfo.min, iinfo.max

    x_ = x.reshape(x.numel() // group_size, group_size)
    amax = x_.abs().max(dim=-1, keepdim=True)[0].clamp(min=_EPS).to(torch.float32)
    x_s = amax / int8_max
    x_q = (x_ / x_s).clamp(min=int8_min, max=int8_max).to(dtype)
    x_q = x_q.reshape(x.shape)
    x_s = x_s.reshape(x.shape[:-1] + (x.shape[-1] // group_size,))
    return x_q, x_s
```
