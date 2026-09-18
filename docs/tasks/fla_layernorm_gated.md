# fla_layernorm_gated (fla/layernorm_gated)

## 任务描述

FLA 系列融合门控归一化算子：先对输入 `x` 做 RMSNorm 或 LayerNorm，再乘以可学习的 weight/bias，最后用 gate 张量 `g` 施加门控激活。与 Mamba/SSD 组的 `layernorm_gated` 不同，本算子无通道分组，gate 始终在归一化之后施加，且门控激活函数可在 `swish`/`silu`/`sigmoid` 中选择。不涉及 residual 融合。

## 接口签名

```python
def fla_layernorm_gated(x, g, weight, bias, activation="swish", eps=1e-5, is_rms_norm=True):
```

> 选手实现的函数签名需与上述 `fla_layernorm_gated(...)` 完全一致。

## 计算定义

- `x`: `[T, D]` 输入张量；`g`: `[T, D]` 门控张量；`weight`: `[D]` 或 None；`bias`: `[D]` 或 None
- 归一化（在 float32 精度下）：
  - `is_rms_norm=True`（RMSNorm）：`x_hat = x / sqrt(mean(x²) + eps)`
  - `is_rms_norm=False`（LayerNorm）：先减均值，再 `x_hat = (x - mean) / sqrt(var + eps)`
- 仿射：`y = x_hat * weight + bias`（weight/bias 为 None 时跳过）
- 门控：
  - `activation in {"swish", "silu"}`：`y = y * g * sigmoid(g)`（swish gate）
  - `activation == "sigmoid"`：`y = y * sigmoid(g)`
- 输出 cast 回输入 dtype，shape 为 `[T, D]`

## 正确性判别标准

Per-dtype tolerance:
- float32: `atol=1e-4, rtol=1e-4`
- bfloat16: `atol=1.5e-2, rtol=1.5e-2`
- float16: `atol=1e-2, rtol=1e-2`


## 参考实现

```python
def reference(x, g, weight, bias, activation="swish", eps=1e-5, is_rms_norm=True):
    out_dtype = x.dtype
    xf = x.float()

    if is_rms_norm:
        var = (xf**2).mean(dim=-1, keepdim=True)
        x_hat = xf * (var + eps).rsqrt()
    else:
        mean = xf.mean(dim=-1, keepdim=True)
        var = ((xf - mean) ** 2).mean(dim=-1, keepdim=True)
        x_hat = (xf - mean) * (var + eps).rsqrt()

    y = x_hat
    if weight is not None:
        y = y * weight.float()
    if bias is not None:
        y = y + bias.float()

    gf = g.float()
    if activation in ("swish", "silu"):
        y = y * gf * gf.sigmoid()
    elif activation == "sigmoid":
        y = y * gf.sigmoid()

    return y.to(out_dtype)
```
