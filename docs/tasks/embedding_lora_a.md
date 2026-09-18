# embedding_lora_a (lora/embedding_lora_a)

## 任务描述

LoRA-A embedding lookup: gathers embedding rows per-segment according to LoRA adapter routing, producing the first-stage LoRA output.

## 接口签名

```python
def reference(input_ids, weights, batch_info, vocab_size, extra_embeddings=None)
```

> 选手实现的函数签名需与上述 `reference(...)` 完全一致。

## 计算定义

- 对每个 segment（由 batch_info 指定的连续 token 区间）:
  1. 获取 adapter weight index 和 rank
  2. 用 `input_ids` 做 embedding lookup: `out[rows, :r] = weights[w_idx, :r, tokens].T`
  3. 若有 `extra_embeddings` 且 token >= vocab_size，则用 extra embedding 替换
- 输出: `[S, rank]`，与 weights 同 dtype

## 正确性判别标准

Per-dtype tolerance:
- float32: `atol=1e-4, rtol=1e-4`
- bfloat16: `atol=1.5e-2, rtol=1.5e-2`
- float16: `atol=1e-2, rtol=1e-2`


## 参考实现

```python
import torch


def reference(input_ids, weights, batch_info, vocab_size, extra_embeddings=None):
    S = input_ids.shape[0]
    rank = weights.shape[1]
    out = torch.zeros(S, rank, dtype=weights.dtype, device=weights.device)

    seg_indptr = batch_info.seg_indptr
    weight_indices = batch_info.weight_indices
    lora_ranks = batch_info.lora_ranks

    for b in range(batch_info.bs):
        start = int(seg_indptr[b].item())
        end = int(seg_indptr[b + 1].item())
        if start == end:
            continue
        w_idx = int(weight_indices[b].item())
        r = int(lora_ranks[w_idx].item())
        if r == 0:
            continue

        tokens = input_ids[start:end].long()
        is_extra = tokens >= vocab_size
        clamped = tokens.clamp(max=vocab_size - 1)
        out[start:end, :r] = weights[w_idx, :r, clamped].t()

        if extra_embeddings is not None and bool(is_extra.any()):
            extra_idx = (tokens - vocab_size).clamp(min=0)
            extra_vals = extra_embeddings[w_idx, extra_idx, :r]
            out[start:end, :r] = torch.where(
                is_extra.unsqueeze(-1), extra_vals, out[start:end, :r]
            )

    return out
```
