# chunked_sgmv_shrink (lora/chunked_sgmv_shrink)

## 任务描述

分块 SGMV Shrink：对批次中的每条请求，将输入激活向量 `x`（形如 `[S, K]`）通过对应适配器的 A 矩阵压缩到低秩空间，实现分段批量低秩适配前向（shrink 方向）计算，输出结果写入 `[S, N]` 张量对应行。

## 接口签名

```python
def chunked_sgmv_shrink(x, weights, batch_info, num_slices=1)
```

> 选手实现的函数签名需与上述 `chunked_sgmv_shrink(...)` 完全一致。

## 计算定义

输入说明：
- `x`: `[S, K]` float 张量，`S` 为总 token 数，`K` 为输入特征维度
- `weights`: `[num_lora, N, K]` float 张量，所有适配器的 A 矩阵，`N` 为输出（低秩）维度
- `batch_info`: 批次元信息对象，包含：
  - `seg_indptr`: `[B+1]` int 张量，第 `b` 条请求对应 `permutation[seg_indptr[b] : seg_indptr[b+1]]`
  - `weight_indices`: `[B]` int 张量，第 `b` 条请求使用的 LoRA 权重索引 `w_idx`
  - `permutation`: `[S]` int 张量，将分段索引映射回全局 token 行号
  - `bs`: int，批次中的请求数 `B`
- `num_slices`: int，切片数量（参数保留，当前实现固定为单切片）

输出：`[S, N]` 张量，与 `x` 同 dtype，未被任何请求覆盖的行保持为 0。

逐请求计算流程（对第 `b` 条请求）：

1. 取分段范围 `[start, end) = [seg_indptr[b], seg_indptr[b+1])`；若为空则跳过。
2. 取权重索引 `w_idx = weight_indices[b]`，对应权重矩阵 $W = \text{weights}[w\_idx]$，形状 `[N, K]`。
3. 取全局行号 `rows = permutation[start:end]`，激活子矩阵 $X_{seg} = x[\text{rows}]$，形状 `[len, K]`。
4. 以 float32 精度执行矩阵乘法后写回原 dtype：

$$\text{out}[\text{rows}] = \left(X_{seg}^{(\text{fp32})} \times W^{(\text{fp32})\top}\right).to(\text{x.dtype})$$

即：

$$\text{out}[\text{rows},\, j] = \sum_{k=0}^{K-1} x[\text{rows},\, k] \cdot W[w\_idx,\, j,\, k], \quad j \in [0, N)$$

## 正确性判别标准

Per-dtype tolerance:
- float32: `atol=1e-4, rtol=1e-4`
- bfloat16: `atol=1.5e-2, rtol=1.5e-2`
- float16: `atol=1e-2, rtol=1e-2`

## 参考实现

```python
def reference(x, weights, batch_info, num_slices=1):
    S, K = x.shape
    N = weights.shape[1]
    out = x.new_zeros(S, N)

    seg_indptr = batch_info.seg_indptr
    weight_indices = batch_info.weight_indices
    permutation = batch_info.permutation

    for b in range(batch_info.bs):
        start = int(seg_indptr[b].item())
        end = int(seg_indptr[b + 1].item())
        if start == end:
            continue
        w_idx = int(weight_indices[b].item())
        rows = permutation[start:end].long()

        x_seg = x[rows].float()
        w = weights[w_idx].float()
        out[rows] = (x_seg @ w.t()).to(x.dtype)

    return out
```
