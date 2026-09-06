# fused_moe_gemm (moe/fused_moe_gemm)

## 任务描述

The core fused-MoE GEMM: each token's row is multiplied against the weight
matrix of whichever expert it was routed to (one call per top-k slot), with
the routing weight applied.

## 接口签名

```python
def reference(A, B, topk_weights, topk_ids, top_k):
```

> 选手实现的函数签名需与上述 `reference(...)` 完全一致。

## 计算定义

- `out[t, j] = (A[t] @ B[topk_ids[t,j]].T) * topk_weights[t, j]`.

The baseline's actual kernel call additionally needs `sorted_token_ids` /
`expert_ids` / `num_tokens_post_padded` (from `moe_align_block_size`) to
group tokens by assigned expert for the grid — that's scheduling metadata
that doesn't affect the output layout (the kernel writes each result back to
its own token's row via `offs_token`, not to its sorted/padded position), so
`baseline.py` computes it internally; it isn't part of the logical contract.

## 正确性判别标准

`atol=0.5, rtol=1e-2`.


## 参考实现

```python
import torch


def reference(A, B, topk_weights, topk_ids, top_k):
    T, K = A.shape
    E, N, _ = B.shape
    A32 = A.float()
    B32 = B.float()

    out = torch.empty(T, top_k, N, dtype=A.dtype, device=A.device)
    for t in range(T):
        for j in range(top_k):
            e = int(topk_ids[t, j].item())
            row = A32[t] @ B32[e].t()
            out[t, j] = (row * topk_weights[t, j].float()).to(A.dtype)
    return out
```
