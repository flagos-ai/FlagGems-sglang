# merge_state (attention/merge_state)

## 任务描述

Combine two partial (prefix/suffix) split-KV attention states — each an
output plus its log-sum-exp — into one, using the standard online-softmax
merge rule.

## 接口签名

```python
def reference(prefix_output, prefix_lse, suffix_output, suffix_lse):
```

> 选手实现的函数签名需与上述 `reference(...)` 完全一致。

## 计算定义

- `max_lse = max(prefix_lse, suffix_lse)` (elementwise; `inf` treated as `-inf`).
- `p = exp(prefix_lse - max_lse)`, `s = exp(suffix_lse - max_lse)`.
- `output = (prefix_output * p + suffix_output * s) / (p + s)`.
- `output_lse = log(p + s) + max_lse`.

## 正确性判别标准

Per-dtype tolerance:
- float32: `atol=1e-4, rtol=1e-4`
- bfloat16: `atol=1.5e-2, rtol=1.5e-2`
- float16: `atol=1e-2, rtol=1e-2`


## 参考实现

```python
import torch


def reference(prefix_output, prefix_lse, suffix_output, suffix_lse):
    p_lse = torch.where(
        prefix_lse == float("inf"), torch.full_like(prefix_lse, float("-inf")), prefix_lse
    ).float()
    s_lse = torch.where(
        suffix_lse == float("inf"), torch.full_like(suffix_lse, float("-inf")), suffix_lse
    ).float()

    max_lse = torch.maximum(p_lse, s_lse)
    p_lse = p_lse - max_lse
    s_lse = s_lse - max_lse
    p_se = torch.exp(p_lse)
    s_se = torch.exp(s_lse)
    out_se = p_se + s_se

    output_lse = (torch.log(out_se) + max_lse).to(prefix_lse.dtype)

    p_scale = (p_se / out_se).unsqueeze(-1)
    s_scale = (s_se / out_se).unsqueeze(-1)
    output = (
        prefix_output.float() * p_scale + suffix_output.float() * s_scale
    ).to(prefix_output.dtype)
    return output, output_lse
```
