# gate_up_lora_b (lora/gate_up_lora_b)

## 任务描述

LoRA "B"（升维投影）矩阵的 gate/up 双切片分段批量 GEMM：将两个独立的分段 GEMM 融合为一次 kernel 调用，gate 切片和 up 切片各自读取 `x` 中对应的 `r` 宽列，与 adapter 权重相乘后加到 `base_output` 的对应列区间上，输出与 `base_output` 同 shape。

## 接口签名

```python
def reference(x, gate_up_lora_b, batch_info, output_dim, base_output)
```

> 选手实现的函数签名需与上述 `reference(...)` 完全一致。

## 计算定义

- 输入 `x`: `[S, 2r]`；`gate_up_lora_b`: `[num_lora, 2*output_dim, r]`；`base_output`: `[S, 2*output_dim]`
- 输出：`[S, 2*output_dim]`，与 `base_output` 同 dtype；初始值为 `base_output` 的克隆（float32）
- 对每个 segment `b`（行范围 `seg_indptr[b]:seg_indptr[b+1]`，adapter 索引 `w = weight_indices[b]`，scaling `scalings[w]`），以及切片 `i ∈ {0 (gate), 1 (up)}`：
  - `o_start = i * output_dim`，`o_end = o_start + output_dim`
  - `x_slice = x[rows, i*r : (i+1)*r]`（float32）
  - `w_slice = gate_up_lora_b[w, o_start:o_end, :]`（float32）
  - `out[rows, o_start:o_end] += scalings[w] * (x_slice @ w_slice.T)`
- 计算以 float32 精度完成，结果转回 `base_output` 的 dtype

## 正确性判别标准

Per-dtype tolerance:
- float32: `atol=1e-4, rtol=1e-4`
- bfloat16: `atol=1.5e-2, rtol=1.5e-2`
- float16: `atol=1e-2, rtol=1e-2`

## 参考实现

```python
import torch


def reference(x, gate_up_lora_b, batch_info, output_dim, base_output):
    out = base_output.clone().float()
    r = gate_up_lora_b.shape[-1]

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
        for i in range(2):
            o_start = i * output_dim
            o_end = o_start + output_dim
            x_slice = x_seg[:, i * r : (i + 1) * r]
            w_slice = gate_up_lora_b[w_idx, o_start:o_end, :].float()
            out[rows, o_start:o_end] += scaling * (x_slice @ w_slice.t())

    return out.to(base_output.dtype)
```
