# sgemm_lora_a (lora/sgemm_lora_a)

## 任务描述

LoRA "A"（降维投影）矩阵的分段批量 GEMM：将输入按 segment 分组，每个 segment（属于同一请求的连续 token 行）与其对应 adapter 的权重切片相乘，输出各 segment 的投影结果。

## 接口签名

```python
def reference(x, weights, batch_info, stack_num=1)
```

> 选手实现的函数签名需与上述 `reference(...)` 完全一致。

## 计算定义

- 输入 `x`: `[S, K]`；`weights`: `[num_lora, stack_num*r, K]`；`batch_info` 包含每个 segment `b` 的行范围、adapter 索引及可选的 permutation
- 输出：`[S, stack_num*r]`，与 `x` 同 dtype
- 对每个 segment `b`（行范围 `seg_indptr[b]:seg_indptr[b+1]`，adapter 索引 `w = weight_indices[b]`）：
  - 若存在 `permutation`：`rows = permutation[start:end]`，否则 `rows = arange(start, end)`
  - `out[rows] = x[rows].float() @ weights[w].float().T`，结果转回 `x` 的 dtype
- 每个 adapter 使用固定 rank `r`，输出宽度 `stack_num * r` 即权重矩阵的完整第一维

## 正确性判别标准

Per-dtype tolerance:
- float32: `atol=1e-4, rtol=1e-4`
- bfloat16: `atol=1.5e-2, rtol=1.5e-2`
- float16: `atol=1e-2, rtol=1e-2`

## 参考实现

```python
import torch


def reference(x, weights, batch_info, stack_num=1):
    S, K = x.shape
    R = weights.shape[1]
    out = torch.zeros(S, R, dtype=x.dtype, device=x.device)

    seg_indptr = batch_info.seg_indptr
    weight_indices = batch_info.weight_indices
    permutation = batch_info.permutation

    for b in range(batch_info.bs):
        start = int(seg_indptr[b].item())
        end = int(seg_indptr[b + 1].item())
        if start == end:
            continue
        w_idx = int(weight_indices[b].item())
        if permutation is not None:
            rows = permutation[start:end].long()
        else:
            rows = torch.arange(start, end, device=x.device)

        x_seg = x[rows].float()
        w = weights[w_idx].float()
        val = x_seg @ w.t()
        out[rows] = val.to(x.dtype)

    return out
```
