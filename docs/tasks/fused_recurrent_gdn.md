# fused_recurrent_gdn (fla/fused_recurrent_gdn)

## 任务描述

Fused recurrent Gated Delta Network: sequential recurrence where state is updated per timestep via gated decay + delta correction, with output read out via einsum.

## 接口签名

```python
def reference(
    q, k, v, g, beta, scale, initial_state, output_final_state, use_qk_l2norm_in_kernel=False
)
```

> 选手实现的函数签名需与上述 `reference(...)` 完全一致。

## 计算定义

- 逐时间步 `t=0..T-1` 的循环递推:
  1. `state = state * exp(gate)`（指数衰减）
  2. `pred = einsum("bhvk,bhk->bhv", state, k_t)`（读出预测）
  3. `correction = (v_t - pred) * beta`（delta 修正）
  4. `state = state + correction[:,:,None] * k_t[:,None,:]`（外积更新）
  5. `output_t = einsum("bhvk,bhk->bhv", state, q_t)`（读出输出）
- 可选 L2 norm on q/k, 可选 initial_state
- 输出: `(output [B,T,HV,V], final_state or None)`

## 正确性判别标准

`atol=1e-2, rtol=1e-2`.


## 参考实现

```python
import torch


def reference(
    q, k, v, g, beta, scale, initial_state, output_final_state, use_qk_l2norm_in_kernel=False
):
    B, T, H, K = q.shape
    HV = v.shape[2]
    V = v.shape[-1]
    ratio = HV // H
    beta_headwise = beta.dim() == v.dim()

    if initial_state is not None:
        state = initial_state.float().clone()
    else:
        state = q.new_zeros(B, HV, V, K, dtype=torch.float32)

    o = q.new_zeros(B, T, HV, V, dtype=torch.float32)

    for t in range(T):
        qt = q[:, t].float()
        kt = k[:, t].float()
        vt = v[:, t].float()
        gt = g[:, t].float()

        if use_qk_l2norm_in_kernel:
            qt = qt / (qt.pow(2).sum(-1, keepdim=True) + 1e-6).sqrt()
            kt = kt / (kt.pow(2).sum(-1, keepdim=True) + 1e-6).sqrt()
        qt = qt * scale

        kt_e = kt.repeat_interleave(ratio, dim=1) if ratio > 1 else kt  # (B, HV, K)
        qt_e = qt.repeat_interleave(ratio, dim=1) if ratio > 1 else qt  # (B, HV, K)

        state = state * gt.exp()[:, :, None, None]

        pred = torch.einsum("bhvk,bhk->bhv", state, kt_e)
        vt_corr = vt - pred

        if beta_headwise:
            bt = beta[:, t].float()
        else:
            bt = beta[:, t].float().unsqueeze(-1)
        vt_corr = vt_corr * bt

        state = state + vt_corr.unsqueeze(-1) * kt_e.unsqueeze(-2)

        o[:, t] = torch.einsum("bhvk,bhk->bhv", state, qt_e)

    final_state = state if output_final_state else None
    return o.to(v.dtype), final_state
```
