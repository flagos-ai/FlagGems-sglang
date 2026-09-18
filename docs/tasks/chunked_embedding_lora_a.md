# chunked_embedding_lora_a (lora/chunked_embedding_lora_a)

## 任务描述

分块 LoRA-A 嵌入查找：对批次中的每条请求，根据对应的 LoRA 权重索引和激活 rank，从嵌入权重矩阵中按 token id 查表，将结果写入输出张量的对应行，实现分段批量的低秩适配嵌入前向计算。

## 接口签名

```python
def chunked_embedding_lora_a(input_ids, weights, batch_info, vocab_size)
```

> 选手实现的函数签名需与上述 `chunked_embedding_lora_a(...)` 完全一致。

## 计算定义

输入说明：
- `input_ids`: `[S]` int 张量，`S` 为当前批次的总 token 数
- `weights`: `[num_lora, max_rank, vocab_size]` float 张量，所有 LoRA 适配器的 A 矩阵（嵌入形式，第二维为 rank，第三维为词表）
- `batch_info`: 批次元信息对象，包含：
  - `seg_indptr`: `[B+1]` int 张量，第 `b` 条请求对应 `permutation[seg_indptr[b] : seg_indptr[b+1]]`
  - `weight_indices`: `[B]` int 张量，第 `b` 条请求使用的 LoRA 权重索引 `w_idx`
  - `lora_ranks`: `[num_lora]` int 张量，每个适配器的有效 rank `r`
  - `permutation`: `[S]` int 张量，将分段索引映射回全局 token 行号
  - `bs`: int，批次中的请求数 `B`
- `vocab_size`: int，词表大小（仅供参考，实际由 `weights` 决定）

输出：`[S, max_rank]` float 张量，未被任何请求覆盖的行保持为 0。

逐请求计算流程（对第 `b` 条请求）：

1. 取分段范围 `[start, end) = [seg_indptr[b], seg_indptr[b+1])`；若为空则跳过。
2. 取权重索引 `w_idx = weight_indices[b]`，有效 rank `r = lora_ranks[w_idx]`；若 `r == 0` 则跳过。
3. 取全局行号 `rows = permutation[start:end]`，对应 token id `tokens = input_ids[rows]`。
4. 嵌入查表（转置形式）：

$$\text{out}[\text{rows},\, :r] = \text{weights}[w\_idx,\, :r,\, \text{tokens}]^{\top}$$

即对每个位置 $i \in [\text{start}, \text{end})$，有：

$$\text{out}[\text{rows}[i],\, :r] = \text{weights}[w\_idx,\, :r,\, \text{tokens}[i]]$$

## 正确性判别标准

Per-dtype tolerance:
- float32: `atol=1e-4, rtol=1e-4`
- bfloat16: `atol=1.5e-2, rtol=1.5e-2`
- float16: `atol=1e-2, rtol=1e-2`

## 参考实现

```python
def reference(input_ids, weights, batch_info, vocab_size):
    S = input_ids.shape[0]
    rank = weights.shape[1]
    out = weights.new_zeros(S, rank)

    seg_indptr = batch_info.seg_indptr
    weight_indices = batch_info.weight_indices
    lora_ranks = batch_info.lora_ranks
    permutation = batch_info.permutation

    for b in range(batch_info.bs):
        start = int(seg_indptr[b].item())
        end = int(seg_indptr[b + 1].item())
        if start == end:
            continue
        w_idx = int(weight_indices[b].item())
        r = int(lora_ranks[w_idx].item())
        if r == 0:
            continue

        rows = permutation[start:end].long()
        tokens = input_ids[rows].long()
        out[rows, :r] = weights[w_idx, :r, tokens].t()

    return out
```
