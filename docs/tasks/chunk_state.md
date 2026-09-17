# chunk_state (mamba/chunk_state)

## 任务描述

Mamba SSM chunk-state computation: computes per-chunk hidden states by accumulating `x * B * dt * exp(decay)` via einsum.

## 接口签名

```python
def reference(B, x, dt, dA_cumsum)
```

> 选手实现的函数签名需与上述 `reference(...)` 完全一致。
> 注意：参数 `B` 是 SSM 的状态投影矩阵（shape `[batch, seqlen, ngroups, dstate]`），不是 batch size。

## 计算定义

- `B`（SSM矩阵）: `[batch, seqlen, ngroups, dstate]`
- `x`: `[batch, seqlen, nheads, headdim]`
- `dt`: `[batch, nheads, nchunks, chunk_size]`
- `dA_cumsum`: `[batch, nheads, nchunks, chunk_size]`
- 计算 decay: `exp(dA_cumsum[..., -1:] - dA_cumsum)`
- scale = decay * dt
- 将 x 和 B 切块后做 einsum: `states = einsum("bcthp,bcthn->bchpn", x_c, B_scaled)`
- 输出: `[batch, nchunks, nheads, headdim, dstate]`

## 正确性判别标准

`atol=3e-2, rtol=3e-2`.


## 参考实现

```python
import torch


def reference(B, x, dt, dA_cumsum):
    batch, seqlen, nheads, headdim = x.shape
    _, _, nchunks, chunk_size = dt.shape
    _, _, ngroups, dstate = B.shape
    ratio = nheads // ngroups

    x_c = x.reshape(batch, nchunks, chunk_size, nheads, headdim).float()
    B_c = B.reshape(batch, nchunks, chunk_size, ngroups, dstate).float()
    B_c = B_c.repeat_interleave(ratio, dim=3)

    dA_last = dA_cumsum[..., -1:].float()
    decay = torch.exp(dA_last - dA_cumsum.float())
    scale = (decay * dt.float()).permute(0, 2, 3, 1)

    Bs = B_c * scale.unsqueeze(-1)
    states = torch.einsum("bcthp,bcthn->bchpn", x_c, Bs)
    return states
```
