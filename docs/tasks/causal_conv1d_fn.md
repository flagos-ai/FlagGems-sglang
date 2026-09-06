# causal_conv1d_fn (mamba/causal_conv1d_fn)

## 任务描述

Depthwise causal 1D convolution over variable-length, concatenated
("continuous batching") sequences, fused with an optional activation.

## 接口签名

```python
def reference(x, weight, bias, query_start_loc, seq_lens_cpu, activation="silu"):
```

> 选手实现的函数签名需与上述 `reference(...)` 完全一致。

## 计算定义

- Sequence `i` occupies columns `query_start_loc[i]:query_start_loc[i+1]`
  of `x` (concatenated left-to-right, no cross-sequence bleed).
- Per sequence, per channel `d`, per position `t` (0-indexed within the
  sequence): `conv[d,t] = sum_{k=0}^{width-1} weight[d,k] * x_pad[d, t+k]`,
  where `x_pad` is that sequence's channel row left-padded with
  `width-1` zeros (causal — no future or cross-sequence lookback).
- `out = conv + bias` (if given), then `out = silu(out)` if
  `activation in {"silu", "swish"}`.
- Scope: fresh prefill only (no cache, no `conv_states`, no `cache_indices`).

## 正确性判别标准

Per-dtype tolerance:
- float32: `atol=1e-4, rtol=1e-4`
- bfloat16: `atol=1.5e-2, rtol=1.5e-2`
- float16: `atol=1e-2, rtol=1e-2`


## 参考实现

```python
import torch
import torch.nn.functional as F


def reference(x, weight, bias, query_start_loc, seq_lens_cpu, activation="silu"):
    dim, _ = x.shape
    width = weight.shape[1]
    out = torch.zeros_like(x)

    for i in range(len(seq_lens_cpu)):
        start = int(query_start_loc[i].item())
        end = int(query_start_loc[i + 1].item())
        seg = x[:, start:end].float()
        seg_len = seg.shape[1]
        padded = F.pad(seg, (width - 1, 0))

        conv = torch.zeros(dim, seg_len, device=x.device)
        for k in range(width):
            conv += weight[:, k : k + 1].float() * padded[:, k : k + seg_len]
        if bias is not None:
            conv += bias.float().unsqueeze(-1)
        if activation in ("silu", "swish"):
            conv = conv * torch.sigmoid(conv)

        out[:, start:end] = conv.to(x.dtype)

    return out
```
