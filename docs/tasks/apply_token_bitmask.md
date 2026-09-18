# apply_token_bitmask (sampling_grammar/apply_token_bitmask)

## 任务描述

Apply a token bitmask to logits: for each token, set logits to `-inf` where the corresponding bit is unset in the bitmask (used for constrained decoding / grammar sampling).

## 接口签名

```python
def reference(logits, bitmask)
```

> 选手实现的函数签名需与上述 `reference(...)` 完全一致。

## 计算定义

- `logits`: `[B, V]` float tensor
- `bitmask`: `[B, ceil(V/32)]` int32 tensor，每 bit 对应一个 token
- 对于第 `v` 个 token：`word_idx = v // 32`, `bit_idx = v % 32`
- 若 `(bitmask[:, word_idx] >> bit_idx) & 1 == 0`，则置 `logits[:, v] = -inf`
- 输出与 `logits` 同 shape 同 dtype

## 正确性判别标准

Per-dtype tolerance:
- float32: `atol=1e-4, rtol=1e-4`
- bfloat16: `atol=1.5e-2, rtol=1.5e-2`
- float16: `atol=1e-2, rtol=1e-2`


## 参考实现

```python
import torch


def reference(logits, bitmask):
    B, V = logits.shape
    v_idx = torch.arange(V, device=logits.device)
    word_idx = v_idx // 32
    bit_idx = v_idx % 32
    bits = (bitmask[:, word_idx] >> bit_idx) & 1
    return torch.where(bits == 0, torch.full_like(logits, float("-inf")), logits)
```
