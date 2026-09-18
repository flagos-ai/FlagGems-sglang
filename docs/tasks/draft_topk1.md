# draft_topk1 (speculative/draft_topk1)

## 任务描述

投机解码 Top-1 草稿生成：从下一个 token 的 logit 中选取概率最高的 token 作为草稿候选，返回其位置概率（恒为 1）、token 索引、更新后的位置编号，以及可选的草稿 token 缓冲区。

## 接口签名

```python
def reference(next_token_logits, positions, draft_tokens=None, draft_token_column=0)
```

> 选手实现的函数签名需与上述 `reference(...)` 完全一致。

## 计算定义

输入：
- `next_token_logits`: `[B, V]` — B 个序列各自下一个位置的 logit，V 为词表大小
- `positions`: `[B]` int64 — 各序列当前位置编号
- `draft_tokens`: `[B, D]` int 或 None — 草稿 token 缓冲区，D 为草稿步长（可选）
- `draft_token_column`: int — 将本步草稿写入 `draft_tokens` 的列索引（默认 0）

计算步骤：

1. Argmax 选择：`topk_index = argmax(next_token_logits, dim=-1, keepdim=True)`，形状 `[B, 1]`，int64
   — 贪心选取 logit 最大的 token，不经 softmax，直接取最大值下标

2. Top-1 概率（恒为 1）：`topk_p = ones([B, 1], dtype=float32)`
   — 贪心策略下 top-1 的验收概率为 1

3. 位置更新：`out_positions = positions + 1`，形状 `[B]`
   — 当前位置向前推进一步

4. 草稿缓冲区更新（当 `draft_tokens` 不为 None）：
   - `out_draft_tokens = draft_tokens.clone()`
   - `out_draft_tokens[:, draft_token_column] = topk_index.squeeze(-1)`
   — 将所选 token 写入草稿缓冲区的指定列

输出：`(topk_p, topk_index, out_positions, out_draft_tokens)`
- 当 `draft_tokens` 为 None 时，`out_draft_tokens` 为 None

## 正确性判别标准

Per-dtype tolerance:
- float32: `atol=1e-4, rtol=1e-4`
- bfloat16: `atol=1.5e-2, rtol=1.5e-2`
- float16: `atol=1e-2, rtol=1e-2`

`topk_index` 和 `out_draft_tokens` 需与参考实现精确匹配（exact equality）。

## 参考实现

```python
import torch


def reference(next_token_logits, positions, draft_tokens=None, draft_token_column=0):
    bs = next_token_logits.shape[0]
    topk_index = next_token_logits.argmax(dim=-1, keepdim=True).to(torch.int64)
    topk_p = torch.ones(bs, 1, dtype=torch.float32, device=next_token_logits.device)

    out_positions = positions + 1
    out_draft_tokens = None
    if draft_tokens is not None:
        out_draft_tokens = draft_tokens.clone()
        out_draft_tokens[:, draft_token_column] = topk_index.squeeze(-1)

    return topk_p, topk_index, out_positions, out_draft_tokens
```
