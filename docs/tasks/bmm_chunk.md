# bmm_chunk (mamba/bmm_chunk)

## 任务描述

Batched matrix multiply within chunks: reshape input into chunks and perform per-group batched inner product.输入 shape 为 `[batch, seqlen, ngroups, k]`，按 `chunk_size` 切块后做 einsum `bcigk,bcjgk->bcgij`（即每个 group 内的 chunk-local K*K^T）。

注意：`causal` 参数当前未使用（保留用于未来扩展），实现时可忽略。

## 接口签名

```python
def reference(a, b, chunk_size, causal=False)
```

> 选手实现的函数签名需与上述 `reference(...)` 完全一致。

## 计算定义

- 输入 `a`, `b`: `[batch, seqlen, ngroups, k]`
- 按 `chunk_size` 切块: `a_c = a.reshape(batch, nchunks, chunk_size, ngroups, k)`
- 计算 per-group chunk-local 内积: `out = einsum("bcigk,bcjgk->bcgij", a_c, b_c)`
- 输出: `[batch, nchunks, ngroups, chunk_size, chunk_size]`
- 全程使用 float32 计算

## 正确性判别标准

Per-dtype tolerance:
- float32: `atol=1e-4, rtol=1e-4`
- bfloat16: `atol=1.5e-2, rtol=1.5e-2`
- float16: `atol=1e-2, rtol=1e-2`


## 参考实现

```python
import math

import torch


def reference(a, b, chunk_size, causal=False):
    batch, seqlen, ngroups, k = a.shape
    nchunks = math.ceil(seqlen / chunk_size)

    a_c = a.reshape(batch, nchunks, chunk_size, ngroups, k).float()
    b_c = b.reshape(batch, nchunks, chunk_size, ngroups, k).float()
    out = torch.einsum("bcigk,bcjgk->bcgij", a_c, b_c)
    return out
```
