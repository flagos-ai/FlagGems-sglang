# qkv_lora_b (lora/qkv_lora_b)

## 任务描述

LoRA-B for QKV projection: applies the second-stage LoRA matrix (B) to produce Q, K, V outputs per-segment.

## 接口签名

```python
def reference(x, qkv_lora_b, batch_info, output_offset, max_qkv_out_dim, base_output)
```

> 选手实现的函数签名需与上述 `reference(...)` 完全一致。

## 计算定义

- 对每个 segment:
  1. 取 x 的对应行（支持 permutation 重排）
  2. 按 `output_offset` 分 n_slices（对应 Q/K/V 三个投影）
  3. 每个 slice: `out[rows, o_start:o_end] += scaling * (x_slice @ W_slice.T)`
- `base_output` 为已有的 dense 输出，LoRA 做增量叠加
- 输出与 `base_output` 同 shape 同 dtype

## 正确性判别标准

Per-dtype tolerance:
- float32: `atol=1e-4, rtol=1e-4`
- bfloat16: `atol=1.5e-2, rtol=1.5e-2`
- float16: `atol=1e-2, rtol=1e-2`


## 参考实现

```python
import torch


def reference(x, qkv_lora_b, batch_info, output_offset, max_qkv_out_dim, base_output):
    out = base_output.clone().float()
    n_slices = output_offset.numel() - 1
    r = qkv_lora_b.shape[-1]

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
        for i in range(n_slices):
            o_start = int(output_offset[i].item())
            o_end = int(output_offset[i + 1].item())
            x_slice = x_seg[:, i * r : (i + 1) * r]
            w_slice = qkv_lora_b[w_idx, o_start:o_end, :].float()
            out[rows, o_start:o_end] += scaling * (x_slice @ w_slice.t())

    return out.to(base_output.dtype)
```
