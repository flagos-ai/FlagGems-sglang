# fused_gdn_gating (fla/fused_gdn_gating)

## 任务描述

为 Gated DeltaNet 计算融合门控：给定每头的对数衰减参数 `A_log`、原始门控 logits `a` 和 `b`，以及偏置 `dt_bias`，在一个核中同时计算对数衰减门 `g` 和输出门 `beta_output`。该操作仅在 decode 阶段调用（每行一个 token），因此输出带有一个大小为 1 的前置 seq_len 维度。SGLang baseline 对应 `sglang.kernels.ops.attention.fla.fused_gdn_gating.fused_gdn_gating`。

## 接口签名

```python
def fused_gdn_gating(A_log, a, b, dt_bias, beta=1.0, threshold=20.0)
```

> 选手实现的函数签名需与上述 `fused_gdn_gating(...)` 完全一致。

## 计算定义

- `A_log`: `[H]` float32，每头的对数衰减参数
- `a`, `b`: `[B, H]` float32，原始门控 logits（分别用于计算 `g` 和 `beta_output`）
- `dt_bias`: `[H]` float32，时间步偏置，广播到 batch 维
- `beta`, `threshold`: 标量 softplus 参数，默认 `1.0` / `20.0`

中间量 `x`：

$$x = a + \text{dt\_bias}$$

数值稳定的 beta-softplus（高于阈值时线性通过）：

$$\text{softplus\_x} = \begin{cases} x & \text{if } \beta \cdot x > \text{threshold} \\ \frac{1}{\beta}\ln(1 + e^{\beta x}) & \text{otherwise} \end{cases}$$

对数衰减门：

$$g = -\exp(A\_\text{log}) \cdot \text{softplus\_x}$$

输出门：

$$\beta\_\text{output} = \sigma(b) = \frac{1}{1 + e^{-b}}$$

两个输出均在 float32 精度下计算，并在最前面增加一个大小为 1 的维度：

$$g \in \mathbb{R}^{1 \times B \times H}, \quad \beta\_\text{output} \in \mathbb{R}^{1 \times B \times H}$$

函数返回顺序为 `(g, beta_output)`。

## 正确性判别标准

float32 default
- float32: `atol=1e-4, rtol=1e-4`

## 参考实现

```python
import torch
import torch.nn.functional as F


def reference(A_log, a, b, dt_bias, beta=1.0, threshold=20.0):
    x = a.float() + dt_bias.float()
    softplus_x = torch.where(beta * x <= threshold, F.softplus(x, beta=beta), x)
    g = -torch.exp(A_log.float()) * softplus_x
    beta_output = torch.sigmoid(b.float())
    return g.unsqueeze(0).to(torch.float32), beta_output.unsqueeze(0).to(torch.float32)
```
