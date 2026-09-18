# interleaved_rope (rope/interleaved_rope)

## 任务描述

多模态交错 RoPE 流选择：给定三路并行的 cos/sin 已作用 RoPE 流（时间/高度/宽度），按维度下标 mod 3 将它们交错合并为一路输出，用于 Qwen2-VL 等多模态模型的位置编码合并。

## 接口签名

```python
def reference(x, mrope_section)
```

> 选手实现的函数签名需与上述 `reference(...)` 完全一致。

## 计算定义

输入：
- `x`: `[3, S, D]` — 三路 RoPE 流，S 为序列长度，D 为特征维度
  - `x[0]`：时间流（temporal）
  - `x[1]`：高度流（height）
  - `x[2]`：宽度流（width）
- `mrope_section`: `[s0, s1, s2]` int 列表，满足 `s0 + s1 + s2 = D // 3`，指定各模态维度段的长度

按维度下标 `d`（0 到 D-1）选择来源：

1. 若 `d % 3 == 1` 且 `d < mrope_section[1] * 3`：取高度流，`out[:, d] = x[1][:, d]`
2. 若 `d % 3 == 2` 且 `d < mrope_section[2] * 3`：取宽度流，`out[:, d] = x[2][:, d]`
3. 否则（`d % 3 == 0`，或超出段边界的维度）：取时间流，`out[:, d] = x[0][:, d]`

此操作为纯选择（gather/select），不含浮点运算，输出 dtype 与输入 `x` 相同，形状为 `[S, D]`。

## 正确性判别标准

Per-dtype tolerance:
- float32: `atol=1e-4, rtol=1e-4`
- bfloat16: `atol=1.5e-2, rtol=1.5e-2`
- float16: `atol=1e-2, rtol=1e-2`

## 参考实现

```python
import torch


def reference(x, mrope_section):
    _, S, D = x.shape
    d = torch.arange(D, device=x.device)
    cond_a = (d % 3 == 1) & (d < mrope_section[1] * 3)
    cond_b = (d % 3 == 2) & (d < mrope_section[2] * 3)

    out = x[0].clone()
    out[:, cond_a] = x[1][:, cond_a]
    out[:, cond_b] = x[2][:, cond_b]
    return out
```
