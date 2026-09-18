# sgemm_lora_b (lora/sgemm_lora_b)

## 任务描述

LoRA-B single GEMM: applies LoRA-B weight matrix per-segment with scaling, producing the final LoRA output.

## 接口签名

```python
def reference(x, weights, batch_info, base_output)
```

> 选手实现的函数签名需与上述 `reference(...)` 完全一致。

## 计算定义

- 对每个 segment:
  1. 取 x 的对应行（支持 permutation 重排）
  2. `out[rows] += scaling * (x_seg @ W.T)`
- `base_output` 为已有的 dense 输出，LoRA 做增量叠加
- float32 计算，输出 cast 回 base_output 的 dtype

## 正确性判别标准

Per-dtype tolerance:
- float32: `atol=1e-4, rtol=1e-4`
- bfloat16: `atol=1.5e-2, rtol=1.5e-2`
- float16: `atol=1e-2, rtol=1e-2`


## 参考实现

```python
import torch


def reference(x, weights, batch_info, base_output):
    out = base_output.clone().float()

    seg_indptr = batch_info.seg_indptr
    weight_indices = batch_info.weight_indices
    lora_ranks = batch_info.lora_ranks
    scalings = batch_info.scalings
    permutation = batch_info.permutation

    for b in range(batch_info.bs):
        start = int(seg_indptr[b].item())
        end = int(seg_indptr[b + 1].item())
        if start == end:
            continue
        w_idx = int(weight_indices[b].item())
        if int(lora_ranks[w_idx].item()) == 0:
            continue
        scaling = float(scalings[w_idx].item())
        if permutation is not None:
            rows = permutation[start:end].long()
        else:
            rows = torch.arange(start, end, device=x.device)

        x_seg = x[rows].float()
        w = weights[w_idx].float()
        out[rows] += scaling * (x_seg @ w.t())

    return out.to(base_output.dtype)
```
