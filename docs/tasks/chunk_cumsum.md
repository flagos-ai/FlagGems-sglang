# chunk_cumsum (mamba/chunk_cumsum)

## 任务描述

Chunk-wise cumulative sum for Mamba SSM: computes `dt * A` cumsum within each chunk, returning processed dt and cumsum.

## 接口签名

```python
def reference(dt, A, chunk_size, dt_bias=None, dt_softplus=False)
```

> 选手实现的函数签名需与上述 `reference(...)` 完全一致。

## 计算定义

- `dt`: `[batch, seqlen, nheads]` — 时间步长
- `A`: `[nheads]` — 衰减系数（负值）
- 可选 `dt_bias` 加到 dt 上，可选 `dt_softplus` 对 dt 做 softplus
- dt 经 clamp(min=0) 后，reshape 为 `[batch, nheads, nchunks, chunk_size]`
- `dA = dt * A`，在 chunk_size 维上做 cumsum
- 输出: `(dt_out, dA_cumsum)`，shape 均为 `[batch, nheads, nchunks, chunk_size]`

## 正确性判别标准

Per-dtype tolerance:
- float32: `atol=1e-4, rtol=1e-4`
- bfloat16: `atol=1.5e-2, rtol=1.5e-2`
- float16: `atol=1e-2, rtol=1e-2`


## 参考实现

```python
import math

import torch
import torch.nn.functional as F


def reference(dt, A, chunk_size, dt_bias=None, dt_softplus=False):
    batch, seqlen, nheads = dt.shape
    nchunks = math.ceil(seqlen / chunk_size)

    dt_f = dt.float()
    if dt_bias is not None:
        dt_f = dt_f + dt_bias.float()
    if dt_softplus:
        dt_f = torch.where(dt_f <= 20.0, F.softplus(dt_f), dt_f)
    dt_f = dt_f.clamp(min=0.0)

    dt_out = dt_f.reshape(batch, nchunks, chunk_size, nheads).permute(0, 3, 1, 2).contiguous()
    dA = dt_out * A.float().view(1, nheads, 1, 1)
    dA_cumsum = dA.cumsum(dim=-1)
    return dt_out, dA_cumsum
```
