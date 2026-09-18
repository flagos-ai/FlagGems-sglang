# log_scaling_tau (attention/log_scaling_tau)

## 任务描述

对输入张量 `x` 的每一行乘以对应的标量 `tau[t]`，在 float32 下计算后 cast 回 `x.dtype`。该操作用于在一次 launch 中将对数空间的注意力缩放因子 `tau` 应用到融合 QKV 投影输出的 query 切片，替代独立的广播乘法算子。SGLang baseline 对应 `sglang.kernels.ops.attention.log_scaling_tau.apply_log_scaling_tau`。

## 接口签名

```python
def log_scaling_tau(x, tau):
```

> 选手实现的函数签名需与上述 `log_scaling_tau(...)` 完全一致。

## 计算定义

对每一行 `t` 独立缩放，`tau[t]` 广播到该行的所有元素：

$$\text{out}[t, \ldots] = \left(x[t, \ldots]_{\text{float32}} \times \tau[t]\right) \to \text{x.dtype}$$

等价的逐行向量形式：

$$\text{out} = \mathbf{x}_{\text{float32}} \odot \boldsymbol{\tau}_{\text{reshape}} \to \text{x.dtype}$$

其中 $\boldsymbol{\tau}_{\text{reshape}} \in \mathbb{R}^{T \times 1 \times \cdots \times 1}$，广播覆盖所有尾部维度。

## 输入输出规格

| 参数 | 形状 | dtype | 说明 |
|------|------|-------|------|
| `x` | `[T, ...]` | float16 / bfloat16 / float32 | 主输入张量，支持任意尾部维度 |
| `tau` | `[T]` | float32 | 每行的缩放标量 |
| **output** | `[T, ...]` | 同 `x` | 缩放后的结果，形状与 `x` 相同 |

## 测试用例

| T | 尾部形状 | dtype | 说明 |
|---|---------|-------|------|
| 4 | `[64]` | float16 | 基本一维尾部 |
| 8 | `[128]` | float16 | 典型 query 头维度 |
| 16 | `[64, 128]` | float16 | 二维尾部 |
| 32 | `[256]` | float16 | 较大行尺寸 |
| 4 | `[64]` | bfloat16 | bf16 快速路径覆盖（8 对齐） |

正确性判别：fp16 default，`atol=1e-2, rtol=1e-2`。

## 参考实现

```python
import torch


def reference(x, tau):
    rows = x.shape[0]
    tau_r = tau.reshape(rows).float()
    shape = [rows] + [1] * (x.dim() - 1)
    return (x.float() * tau_r.view(shape)).to(x.dtype)
```

## 注意事项

- 计算须在 float32 精度下进行（`x.float()`），最终 cast 回 `x.dtype`。
- `tau` 需 reshape 为 `[T, 1, ..., 1]` 后才能正确广播到所有尾部维度。
- 实际实现中存在一条 bf16 快速路径（`row_scale_bf16`），要求 16 字节对齐且 `inner % 8 == 0`，与标量 Triton kernel 输出位完全相同；测试用例使用 float16 以覆盖通用路径，两条路径实现相同的数学公式。
