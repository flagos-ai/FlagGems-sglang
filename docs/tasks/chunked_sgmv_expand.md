# chunked_sgmv_expand (lora/chunked_sgmv_expand)

## 任务描述

分块 SGMV Expand：对批次中的每条请求，将 LoRA 低秩激活向量 `x`（形如 `[S, n_slices * r]`）通过对应适配器的 B 矩阵展开到输出维度，乘以缩放系数后累加到基础输出上，实现多切片批量低秩适配前向（expand 方向）计算。

## 接口签名

```python
def chunked_sgmv_expand(x, weights, batch_info, slice_offsets, max_slice_size, base_output)
```

> 选手实现的函数签名需与上述 `chunked_sgmv_expand(...)` 完全一致。

## 计算定义

输入说明：
- `x`: `[S, n_slices * r]` float 张量，每行为一个 token 在所有切片上的低秩激活向量拼接，`r` 为权重的 rank 维度（`weights.shape[-1]`）
- `weights`: `[num_lora, out_features, r]` float 张量，所有适配器的 B 矩阵（第二维为输出特征，第三维为 rank）
- `batch_info`: 批次元信息对象，包含：
  - `seg_indptr`: `[B+1]` int 张量，第 `b` 条请求对应 `permutation[seg_indptr[b] : seg_indptr[b+1]]`
  - `weight_indices`: `[B]` int 张量，第 `b` 条请求使用的 LoRA 权重索引 `w_idx`
  - `lora_ranks`: `[num_lora]` int 张量，每个适配器的有效 rank
  - `scalings`: `[num_lora]` float 张量，每个适配器的缩放系数
  - `permutation`: `[S]` int 张量，将分段索引映射回全局 token 行号
  - `bs`: int，批次中的请求数 `B`
- `slice_offsets`: `[n_slices + 1]` int 张量，各切片在输出特征维度上的起止偏移
- `max_slice_size`: int，单个切片的最大输出特征数（仅供参考）
- `base_output`: `[S, total_out_features]` float 张量，基础输出（LoRA 增量将累加其上）

输出：`[S, total_out_features]`，dtype 与 `base_output` 一致。

逐请求计算流程（对第 `b` 条请求，共 `n_slices = len(slice_offsets) - 1` 个切片）：

1. 取分段范围和权重索引，若为空或 `lora_ranks[w_idx] == 0` 则跳过。
2. 取全局行号 `rows = permutation[start:end]`，缩放系数 `scaling = scalings[w_idx]`。
3. 对第 `i` 个切片，输出列范围为 `[o_start, o_end) = [slice_offsets[i], slice_offsets[i+1])`，输入列范围为 `[i*r, (i+1)*r)`：

$$\text{out}[\text{rows},\, o\_start : o\_end] \mathrel{+}= \text{scaling} \cdot x[\text{rows},\, i \cdot r : (i+1) \cdot r] \times W[w\_idx,\, o\_start : o\_end,\, :]^{\top}$$

其中 $W[w\_idx, o\_start:o\_end, :]$ 的形状为 `[o_end - o_start, r]`，以 float32 精度计算后写回，最终转换为 `base_output.dtype`。

## 正确性判别标准

Per-dtype tolerance:
- float32: `atol=1e-4, rtol=1e-4`
- bfloat16: `atol=1.5e-2, rtol=1.5e-2`
- float16: `atol=1e-2, rtol=1e-2`

## 参考实现

```python
import torch


def reference(x, weights, batch_info, slice_offsets, max_slice_size, base_output):
    out = base_output.clone().float()
    n_slices = slice_offsets.numel() - 1
    r = weights.shape[-1]

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
        rows = permutation[start:end].long()

        x_seg = x[rows].float()
        for i in range(n_slices):
            o_start = int(slice_offsets[i].item())
            o_end = int(slice_offsets[i + 1].item())
            x_slice = x_seg[:, i * r : (i + 1) * r]
            w_slice = weights[w_idx, o_start:o_end, :].float()
            out[rows, o_start:o_end] += scaling * (x_slice @ w_slice.t())

    return out.to(base_output.dtype)
```
