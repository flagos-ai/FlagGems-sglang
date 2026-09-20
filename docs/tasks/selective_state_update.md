# selective_state_update (mamba/selective_state_update)

## 任务描述

Mamba 选择性状态空间模型（SSM）单步解码递推：给定当前各头的 SSM 状态、一个新 token 的输入/门控/时间步，以及离散化参数，更新状态并输出该 token 的输出向量。支持可选的时间步偏置、softplus 激活、D 跳跃连接和 SiLU 门控。

## 接口签名

```python
def reference(state, x, dt, A, B, C, D=None, z=None, dt_bias=None, dt_softplus=False)
```

> 选手实现的函数签名需与上述 `reference(...)` 完全一致。

## 计算定义

输入维度：
- `state`: `[B, nheads, dim, dstate]` — 当前 SSM 状态
- `x`: `[B, nheads, dim]` — 输入 token 特征
- `dt`: `[B, nheads, dim]` — 时间步参数 Δ
- `A`: `[nheads, dstate]` — 状态矩阵（负值，用于衰减）
- `B`: `[B, ngroups, dstate]` — 输入投影矩阵
- `C`: `[B, ngroups, dstate]` — 输出投影矩阵
- `D`: `[nheads, dim]` 或 None — 跳跃连接权重
- `z`: `[B, nheads, dim]` 或 None — SiLU 门控输入
- `dt_bias`: `[nheads, dim]` 或 None — 时间步偏置
- `dt_softplus`: bool — 是否对 dt 应用 softplus

计算步骤（全程在 float32 精度下执行）：

1. 时间步处理：
   - `dt' = dt + dt_bias`（当 `dt_bias` 不为 None）
   - `dt' = softplus(dt')`（当 `dt_softplus=True`），即 `log(1 + exp(dt'))`

2. 离散化衰减因子：
   `dA[b, h, p, n] = exp(dt'[b, h, p] * A[h, n])`，形状 `[B, nheads, dim, dstate]`

3. 输入矩阵广播（将 ngroups 维广播至 nheads，`ratio = nheads // ngroups`）：
   `B_bcast = B.repeat_interleave(ratio, dim=1)`，形状 `[B, nheads, dstate]`
   `dB[b, h, p, n] = dt'[b, h, p] * B_bcast[b, h, n]`，形状 `[B, nheads, dim, dstate]`

4. 状态更新（in-place 递推）：
   `state_new[b, h, p, n] = state[b, h, p, n] * dA[b, h, p, n] + dB[b, h, p, n] * x[b, h, p]`

5. 输出投影：
   `C_bcast = C.repeat_interleave(ratio, dim=1)`，形状 `[B, nheads, dstate]`
   `y[b, h, p] = sum_n(state_new[b, h, p, n] * C_bcast[b, h, n])`，即 `einsum('bhpn,bhn->bhp')`

6. 可选 D 跳跃连接（当 `D` 不为 None）：
   `y = y + D * x`

7. 可选 SiLU 门控（当 `z` 不为 None）：
   `y = y * (z * sigmoid(z))`，即 `y * silu(z)`

输出：`(y.to(x.dtype), state_new.to(state.dtype))`

## 正确性判别标准

Per-dtype tolerance:
- float32: `atol=1e-4, rtol=1e-4`
- bfloat16: `atol=1.5e-2, rtol=1.5e-2`
- float16: `atol=1e-2, rtol=1e-2`

## 参考实现

```python
import torch
import torch.nn.functional as F


def reference(state, x, dt, A, B, C, D=None, z=None, dt_bias=None, dt_softplus=False):
    state = state.clone()
    batch, nheads, dim, dstate = state.shape
    ngroups = B.shape[1]
    ratio = nheads // ngroups

    dt_f = dt.float()
    if dt_bias is not None:
        dt_f = dt_f + dt_bias.float()
    if dt_softplus:
        dt_f = F.softplus(dt_f)

    dA = torch.exp(dt_f.unsqueeze(-1) * A.float().unsqueeze(0))
    B_exp = B.float().repeat_interleave(ratio, dim=1)
    dB = dt_f.unsqueeze(-1) * B_exp.unsqueeze(2)

    new_state = state.float() * dA + dB * x.float().unsqueeze(-1)
    state = new_state.to(state.dtype)

    C_exp = C.float().repeat_interleave(ratio, dim=1)
    y = torch.einsum("bhpn,bhn->bhp", new_state, C_exp)

    if D is not None:
        y = y + D.float() * x.float()
    if z is not None:
        y = y * (z.float() * torch.sigmoid(z.float()))

    return y.to(x.dtype), state
```
