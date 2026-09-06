# per_group_transpose (quantization/per_group_transpose)

## 任务描述

Per-expert-group transpose: given a row-major `[M, K]` tensor whose rows are
partitioned into contiguous groups by `expert_offsets` (a grouped-GEMM
row layout), transpose each group's `[n_e, K]` block to `[K, n_e]`
independently, writing each transposed block back at the same flat byte
offset its input block occupied.

## 接口签名

```python
def reference(a, expert_offsets, m_alignment=1):
```

> 选手实现的函数签名需与上述 `reference(...)` 完全一致。

## 计算定义

- `expert_offsets` is a cumulative row-count array (`expert_offsets[0] == 0`,
  `expert_offsets[E] == M`); group `e` owns rows
  `expert_offsets[e]:expert_offsets[e+1]`.
- For each group: `out.view(-1)[start*K : start*K + n_e*K] == a[start:end].T.contiguous().view(-1)`,
  where `start, end = expert_offsets[e], expert_offsets[e+1]`, `n_e = end - start`.
- Output has the same total shape/size as `a` — it is not a global transpose,
  only a per-group one, laid out group-by-group.
- `m_alignment` is a compiler alignment hint on group boundaries in the
  baseline kernel; it has no effect on the output values.

## 正确性判别标准

精确相等（`atol=0, rtol=0`）。本算子为整数/字节搬运操作，无浮点计算，结果必须 bit-exact。


## 参考实现

```python
import torch


def reference(a, expert_offsets, m_alignment=1):
    m, k = a.shape
    out = torch.empty_like(a)
    flat_in = a.reshape(-1)
    flat_out = out.reshape(-1)
    num_experts = expert_offsets.numel() - 1
    for e in range(num_experts):
        start = int(expert_offsets[e].item())
        end = int(expert_offsets[e + 1].item())
        n = end - start
        if n <= 0:
            continue
        seg = a[start:end].t().contiguous().reshape(-1)
        flat_out[start * k : start * k + n * k] = seg
    return out
```
