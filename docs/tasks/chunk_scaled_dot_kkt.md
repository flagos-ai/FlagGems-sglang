# chunk_scaled_dot_kkt (fla/chunk_scaled_dot_kkt)

## 任务描述

Delta-rule 内 chunk 修正矩阵的构建算子：对每个 chunk 逐 head 计算 `beta * K @ K.T`，施加严格下三角掩码，并可选地按 `g_cumsum` 进行衰减缩放。输出用于后续 `solve_tril` 求解。本题仅覆盖固定 batch 路径（`cu_seqlens=None`，`T` 整除 `chunk_size`）；`H`（beta 的 head 数）可为 `Hg`（k 的 head 数）的整数倍（GQA 共享）。

## 接口签名

```python
def chunk_scaled_dot_kkt(k, beta, g_cumsum=None, chunk_size=64):
```

> 选手实现的函数签名需与上述 `chunk_scaled_dot_kkt(...)` 完全一致。

## 计算定义

- `k`: `[B, T, Hg, K]`；`beta`: `[B, T, H]`；`g_cumsum`: `[B, T, H]` 或 None；`chunk_size = BT`，`NT = T // BT`
- 将 `k` 在 head 维上 repeat `ratio = H // Hg` 倍，reshape 为 `[B, NT, H, BT, K]`
- 对每个 chunk：`A[b,n,h,i,j] = k[b,n,h,i,:] · k[b,n,h,j,:]`（点积）
- 若 `g_cumsum` 不为 None：`A[i,j] *= exp(g_cumsum[i] - g_cumsum[j])` 当指数 `<= 0`，否则置 `0`（safe-exp 保护）
- 乘以 `beta`：`A[b,n,h,i,j] *= beta[b,n,h,i]`
- 严格下三角掩码：`i > j` 保留，`i <= j` 置 `0`
- 输出 reshape 为 `[B, T, H, BT]`，dtype 为 float32

## 正确性判别标准

Per-dtype tolerance:
- float32: `atol=1e-4, rtol=1e-4`
- bfloat16: `atol=1.5e-2, rtol=1.5e-2`
- float16: `atol=1e-2, rtol=1e-2`


## 参考实现

```python
import torch


def reference(k, beta, g_cumsum=None, chunk_size=64):
    B, T, Hg, K = k.shape
    H = beta.shape[-1]
    ratio = H // Hg
    BT = chunk_size
    NT = T // BT

    k_c = k.float().view(B, NT, BT, Hg, K)
    k_c = k_c.repeat_interleave(ratio, dim=3)  # (B, NT, BT, H, K)
    k_c = k_c.permute(0, 1, 3, 2, 4)  # (B, NT, H, BT, K)

    A = torch.einsum("bnhik,bnhjk->bnhij", k_c, k_c)

    if g_cumsum is not None:
        g_c = g_cumsum.float().view(B, NT, BT, H).permute(0, 1, 3, 2)  # (B, NT, H, BT)
        g_diff = g_c.unsqueeze(-1) - g_c.unsqueeze(-2)
        A = A * torch.where(g_diff <= 0, torch.exp(g_diff), torch.zeros_like(g_diff))

    beta_c = beta.float().view(B, NT, BT, H).permute(0, 1, 3, 2)  # (B, NT, H, BT)
    A = A * beta_c.unsqueeze(-1)

    causal = torch.tril(torch.ones(BT, BT, dtype=torch.bool, device=k.device), diagonal=-1)
    A = torch.where(causal, A, torch.zeros_like(A))

    # (B, NT, H, BT_i, BT_j) -> (B, NT, BT_i, H, BT_j) -> (B, T, H, BT)
    out = A.permute(0, 1, 3, 2, 4).reshape(B, T, H, BT)
    return out
```
