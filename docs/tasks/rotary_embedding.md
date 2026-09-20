# rotary_embedding (diffusion/rotary_embedding)

## 任务描述

旋转位置编码（RoPE）应用：将预计算的余弦/正弦位置编码应用到输入 token 特征上，以交错配对方式对相邻维度对执行二维旋转变换，用于扩散模型的位置编码。

## 接口签名

```python
def reference(x, cos, sin, interleaved)
```

> 选手实现的函数签名需与上述 `reference(...)` 完全一致。

## 计算定义

输入：
- `x`: `[T, H, D]` — T 个 token，H 个注意力头，每头维度 D，bfloat16
- `cos`: `[T, D//2]` — 预计算的余弦值
- `sin`: `[T, D//2]` — 预计算的正弦值
- `interleaved`: bool — 本问题使用 False（半宽非交错形式）

计算步骤（float32 精度下执行）：

1. 拆分偶数/奇数维度：
   - `x1 = x[..., 0::2]`，形状 `[T, H, D//2]`（偶数下标维度）
   - `x2 = x[..., 1::2]`，形状 `[T, H, D//2]`（奇数下标维度）

2. 广播 cos/sin 到 head 维度：
   - `c = cos.reshape(T, 1, D//2)`
   - `s = sin.reshape(T, 1, D//2)`

3. 二维旋转（对每个维度对 `(x1[i], x2[i])` 应用旋转矩阵）：
   ```
   o1 = x1 * c - x2 * s
   o2 = x1 * s + x2 * c
   ```
   对应旋转矩阵：`[[cos, -sin], [sin, cos]]`

4. 交错重组：`out = stack([o1, o2], dim=-1).reshape(T, H, D)`
   — 将 `(o1[..., i], o2[..., i])` 还原为连续的偶奇维度对

输出转换为输入 dtype（bfloat16），形状与 `x` 相同 `[T, H, D]`。

## 正确性判别标准

Per-dtype tolerance:
- float32: `atol=1e-4, rtol=1e-4`
- bfloat16: `atol=1.5e-2, rtol=1.5e-2`
- float16: `atol=1e-2, rtol=1e-2`

## 参考实现

```python
import torch


def reference(x, cos, sin, interleaved):
    xf = x.float()
    x1 = xf[..., 0::2]
    x2 = xf[..., 1::2]
    c = cos.float().reshape(cos.shape[0], 1, -1)
    s = sin.float().reshape(sin.shape[0], 1, -1)
    o1 = x1 * c - x2 * s
    o2 = x1 * s + x2 * c
    out = torch.stack([o1, o2], dim=-1).reshape(xf.shape)
    return out.to(x.dtype)
```
