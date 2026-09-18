# l2norm (activation_norm/l2norm)

## 任务描述

沿最后一个维度做 L2 归一化：将输入除以其 L2 范数，不含可学习权重。该操作在 FLA 风格的线性注意力中用于 Q/K 归一化。计算在 float32 精度下进行，结果 cast 回输入的原始 dtype。SGLang baseline 对应 `sglang.kernels.ops.attention.fla.l2norm.l2norm_fwd`。

## 接口签名

```python
def l2norm(x, eps=1e-6):
```

> 选手实现的函数签名需与上述 `l2norm(...)` 完全一致。

## 计算定义

$$\text{out} = \frac{\mathbf{x}}{\sqrt{\sum_{i} x_i^2 + \epsilon}}$$

其中求和沿最后一个维度（`dim=-1`）进行，结果保持维度（`keepdim=True`），计算在 float32 精度下完成，最终 cast 回 `x.dtype`：

$$\text{out} = \left(\mathbf{x}_\text{float32} \cdot \frac{1}{\sqrt{\|\mathbf{x}_\text{float32}\|_2^2 + \epsilon}}\right) \to \text{x.dtype}$$

## 输入输出规格

| 参数 | 形状 | dtype | 说明 |
|------|------|-------|------|
| `x` | `[..., D]` | float16 / bfloat16 / float32 | 任意前置维度，归一化沿最后维 |
| `eps` | 标量 | float | 数值稳定项，默认 1e-6 |
| **output** | `[..., D]` | 同 `x` | L2 归一化后的结果，形状与 `x` 相同 |

## 测试用例

| 形状 | dtype | eps |
|------|-------|-----|
| `[1, 64]` | bfloat16 | 1e-6 |
| `[4, 128]` | bfloat16 | 1e-6 |
| `[16, 256]` | bfloat16 | 1e-6 |
| `[8, 32, 128]` | bfloat16 | 1e-6 |
| `[4, 64]` | float32 | 1e-6 |

正确性判别：bfloat16 default，`atol=1.5e-2, rtol=1.5e-2`。

## 参考实现

```python
def reference(x, eps=1e-6):
    xf = x.float()
    rstd = (xf.pow(2).sum(dim=-1, keepdim=True) + eps).rsqrt()
    return (xf * rstd).to(x.dtype)
```

## 注意事项

- 计算须在 float32 精度下进行（`x.float()`），最终结果 cast 回 `x.dtype`，以保证数值稳定。
- 归一化沿 `dim=-1`，支持任意形状的输入（batch 维度任意）。
- 与 RMSNorm 不同，L2Norm 无可学习权重，也不除以维度数量，是纯粹的单位向量归一化。
