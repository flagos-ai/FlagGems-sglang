# mamba_layernorm_gated (mamba/layernorm_gated)

## 任务描述

Gated LayerNorm for Mamba: applies layer normalization then multiplies by a gated activation `silu(z) * norm(x)`.

## 接口签名

```python
def reference(x, weight, bias, eps, z=None, group_size=None, norm_before_gate=True, is_rms_norm=True)
```

> 选手实现的函数签名需与上述 `reference(...)` 完全一致。

## 计算定义

- 将输入按 `group_size` 分组: `x.view(M, ngroups, group_size)`
- 若 `norm_before_gate=False`: 先门控 `x = x * z * sigmoid(z)`
- RMS norm（`is_rms_norm=True`）或 LayerNorm:
  - RMS: `x_hat = x * rsqrt(mean(x^2) + eps)`
  - LN: `x_hat = (x - mean) * rsqrt(var + eps)`
- 应用 weight（和可选 bias）
- 若 `norm_before_gate=True`: 后门控 `y = y * z * sigmoid(z)`
- 输出: `[M, N]`，cast 回输入 dtype

## 正确性判别标准

Per-dtype tolerance:
- float32: `atol=1e-4, rtol=1e-4`
- bfloat16: `atol=1.5e-2, rtol=1.5e-2`
- float16: `atol=1e-2, rtol=1e-2`


## 参考实现

```python
import torch


def reference(x, weight, bias, eps, z=None, group_size=None, norm_before_gate=True, is_rms_norm=True):
    M, N = x.shape
    if group_size is None:
        group_size = N
    ngroups = N // group_size
    out_dtype = x.dtype

    xf = x.float().view(M, ngroups, group_size)
    zf = z.float().view(M, ngroups, group_size) if z is not None else None

    if zf is not None and not norm_before_gate:
        xf = xf * zf * torch.sigmoid(zf)

    if is_rms_norm:
        var = (xf**2).mean(dim=-1, keepdim=True)
    else:
        mean = xf.mean(dim=-1, keepdim=True)
        xf = xf - mean
        var = (xf**2).mean(dim=-1, keepdim=True)

    rstd = torch.rsqrt(var + eps)
    x_hat = xf * rstd

    w = weight.float().view(ngroups, group_size)
    y = x_hat * w
    if bias is not None:
        b = bias.float().view(ngroups, group_size)
        y = y + b

    if zf is not None and norm_before_gate:
        y = y * zf * torch.sigmoid(zf)

    return y.reshape(M, N).to(out_dtype)
```
