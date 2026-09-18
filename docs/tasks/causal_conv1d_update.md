# causal_conv1d_update (mamba/causal_conv1d_update)

## 任务描述

深度因果卷积的解码步更新算子：给定一批新 token（每行一个或少量几个），利用滚动 `conv_state` 缓存作为历史，计算卷积输出并将缓存前移。本题仅覆盖基础路径（无循环缓冲区、无 `conv_state_indices` 插槽间接寻址）。

## 接口签名

```python
def causal_conv1d_update(x, conv_state, weight, bias=None, activation="silu"):
```

> 选手实现的函数签名需与上述 `causal_conv1d_update(...)` 完全一致。

## 计算定义

- `x`: `[batch, dim]` 或 `[batch, dim, seqlen]`；若为 2D 则视 `seqlen=1`
- `conv_state`: `[batch, dim, state_len]` 卷积历史缓存
- `weight`: `[dim, width]` 深度卷积核
- `x_cat = concat(conv_state, x, dim=-1)`，shape 为 `[batch, dim, state_len + seqlen]`
- 对每个新位置 `t ∈ [0, seqlen)`：`out[:, :, t] = sum_{k=0}^{width-1} weight[:, k] * x_cat[:, :, t + (state_len + 1 - width) + k]`，即取以 `t` 结尾的 `width` 窗口做加权求和
- 加 `bias`（若不为 None），再对结果施加 `silu(x) = x * sigmoid(x)`（当 `activation in {"silu", "swish"}`）
- 更新后的 `conv_state` 为 `x_cat[:, :, -state_len:]`（窗口前移 `seqlen`）；参考实现先 clone 再更新，不修改调用方传入的原张量
- 返回 `(out, conv_state)`，`out` 与 `x` 同 shape 同 dtype

## 正确性判别标准

Per-dtype tolerance（对 out 与更新后的 conv_state 均适用）：
- float32: `atol=1e-4, rtol=1e-4`
- bfloat16: `atol=1.5e-2, rtol=1.5e-2`
- float16: `atol=1e-2, rtol=1e-2`


## 参考实现

```python
import torch


def reference(x, conv_state, weight, bias=None, activation="silu"):
    unsqueeze = x.dim() == 2
    if unsqueeze:
        x = x.unsqueeze(-1)
    batch, dim, seqlen = x.shape
    width = weight.shape[1]
    state_len = conv_state.shape[-1]
    conv_state = conv_state.clone()

    x_cat = torch.cat([conv_state.float(), x.float()], dim=-1)
    out = torch.zeros_like(x, dtype=torch.float32)

    for t in range(seqlen):
        window = x_cat[:, :, t : t + state_len + 1][:, :, -width:]
        val = (window * weight.float().unsqueeze(0)).sum(-1)
        if bias is not None:
            val = val + bias.float()
        if activation in ("silu", "swish"):
            val = val * torch.sigmoid(val)
        out[:, :, t] = val

    new_conv_state = x_cat[:, :, -state_len:]
    conv_state.copy_(new_conv_state.to(conv_state.dtype))

    out = out.to(x.dtype)
    if unsqueeze:
        out = out.squeeze(-1)
    return out, conv_state
```
