# per_token_quant_int8 (quantization/per_token_quant_int8)

## 任务描述

逐 token（整行）INT8 量化：对输入矩阵的每一行计算一个基于绝对最大值的缩放因子，将该行量化为 int8 整数；输出量化后的整数矩阵和每行一个的 float32 缩放因子向量。

## 接口签名

```python
def reference(x)
```

> 选手实现的函数签名需与上述 `reference(...)` 完全一致。

## 计算定义

- 输入 `x`: `[M, N]`，任意浮点 dtype，连续存储
- 输出：`(x_q: [M, N] int8, x_s: [M, 1] float32)`
- 对每行 `row`：
  - `scale = max(|x[row]|, 1e-10) / 127`
  - `x_q[row] = clamp(round(x[row] / scale), -128, 127)`
- 等价于以整行作为一个 group 调用 per-token-group 量化（`group_size = N`）

## 正确性判别标准

Per-dtype tolerance:
- float32: `atol=1e-4, rtol=1e-4`
- bfloat16: `atol=1.5e-2, rtol=1.5e-2`
- float16: `atol=1e-2, rtol=1e-2`

## 参考实现

```python
from flaggems_reference.per_token_group_quant_int8 import reference as _group_reference


def reference(x):
    return _group_reference(x, group_size=x.shape[-1])
```
