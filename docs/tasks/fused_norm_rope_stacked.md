# fused_norm_rope_stacked (speculative/fused_norm_rope_stacked)

## 任务描述

融合 RMSNorm 与旋转位置编码（RoPE）的堆叠 KV 计算：对多层堆叠的 KV 张量，先对 K 部分逐层进行 RMSNorm 归一化，再对归一化后的 K 施加 RoPE 旋转编码，最后将 K 和 V 分别从 `[T, L, H, D]` 转置为 `[L, T, H, D]` 格式输出。

## 接口签名

```python
def fused_norm_rope_stacked(kv, k_norm_weight, eps, cos_sin_cache, positions, num_kv_heads, head_dim, rotary_dim)
```

> 选手实现的函数签名需与上述 `fused_norm_rope_stacked(...)` 完全一致。

## 计算定义

输入说明：
- `kv`: `[T, L, H*D*2]` 张量，`T` 为序列长度，`L` 为层数，最后一维前半为 K、后半为 V（各占 `H*D`）
- `k_norm_weight`: `[L, D]` 或可广播为 `[1, L, 1, D]` 的 float 张量，RMSNorm 可学习权重
- `eps`: `[L]` 或可广播为 `[1, L, 1, 1]` 的 float 张量，RMSNorm 各层的稳定项 ε
- `cos_sin_cache`: `[max_pos, rotary_dim]` float 张量，前半列为 cos 值，后半列为 sin 值
- `positions`: `[T]` int 张量，每个 token 的位置索引
- `num_kv_heads`: int，KV 头数 `H`
- `head_dim`: int，每头维度 `D`
- `rotary_dim`: int，参与旋转的维度数（`rotary_dim <= D`，取 K 的前 `rotary_dim` 维做旋转）

输出：
- `k_out`: `[L, T, H, D]` 张量，经 RMSNorm + RoPE 后的 K，dtype 与 `kv` 一致
- `v_out`: `[L, T, H, D]` 张量，直接转置后的 V，dtype 与 `kv` 一致

计算步骤（以 float32 精度进行）：

**步骤 1：分离 K 和 V**

$$K = kv[\ldots,\; :H \cdot D].reshape(T, L, H, D)$$
$$V = kv[\ldots,\; H \cdot D:].reshape(T, L, H, D)$$

**步骤 2：对 K 做逐层 RMSNorm**

对每层 $l$，每头 $h$，每个 token $t$：

$$\text{inv\_rms}_{t,l,h} = \left(\frac{1}{D}\sum_{d=0}^{D-1} K_{t,l,h,d}^2 + \varepsilon_l\right)^{-1/2}$$

$$K^{\text{norm}}_{t,l,h,d} = K_{t,l,h,d} \cdot \text{inv\_rms}_{t,l,h} \cdot w_{l,d}$$

其中 $w_{l,d}$ 为 `k_norm_weight[l, d]`。

**步骤 3：对归一化后的 K 施加 RoPE**

设 $r = \text{rotary\_dim} / 2$，对每个 token $t$ 取位置 $p = \text{positions}[t]$：

$$\cos_d = \text{cos\_sin\_cache}[p,\; d], \quad d \in [0, r)$$
$$\sin_d = \text{cos\_sin\_cache}[p,\; r + d], \quad d \in [0, r)$$

旋转（仅作用于每头前 `rotary_dim` 维）：

$$k^{\text{rot}}_{t,l,h,d} = K^{\text{norm}}_{t,l,h,d} \cdot \cos_d - K^{\text{norm}}_{t,l,h,r+d} \cdot \sin_d, \quad d \in [0, r)$$
$$k^{\text{rot}}_{t,l,h,r+d} = K^{\text{norm}}_{t,l,h,r+d} \cdot \cos_d + K^{\text{norm}}_{t,l,h,d} \cdot \sin_d, \quad d \in [0, r)$$

`rotary_dim` 之后的维度保持不变。

**步骤 4：转置输出**

$$k\_out = K^{\text{rot}}.permute(1, 0, 2, 3).contiguous().to(\text{kv.dtype})$$
$$v\_out = V.permute(1, 0, 2, 3).contiguous()$$

## 正确性判别标准

Per-dtype tolerance:
- float32: `atol=1e-4, rtol=1e-4`
- bfloat16: `atol=1.5e-2, rtol=1.5e-2`
- float16: `atol=1e-2, rtol=1e-2`

## 参考实现

```python
def reference(kv, k_norm_weight, eps, cos_sin_cache, positions, num_kv_heads, head_dim, rotary_dim):
    T, L, _ = kv.shape
    H, D = num_kv_heads, head_dim
    kv_size = H * D
    half_rd = rotary_dim // 2

    k_all = kv[..., :kv_size].float().view(T, L, H, D)
    v_all = kv[..., kv_size:].view(T, L, H, D)

    w = k_norm_weight.float().view(1, L, 1, D)
    eps_l = eps.float().view(1, L, 1, 1)
    inv_rms = (k_all.pow(2).mean(dim=-1, keepdim=True) + eps_l).rsqrt()
    k_normed = k_all * inv_rms * w

    pos = positions.long()
    cos = cos_sin_cache[pos, :half_rd].float().view(T, 1, 1, half_rd)
    sin = cos_sin_cache[pos, half_rd:rotary_dim].float().view(T, 1, 1, half_rd)

    k1 = k_normed[..., :half_rd]
    k2 = k_normed[..., half_rd:rotary_dim]
    rot1 = k1 * cos - k2 * sin
    rot2 = k2 * cos + k1 * sin

    k_out = k_normed.clone()
    k_out[..., :half_rd] = rot1
    k_out[..., half_rd:rotary_dim] = rot2

    k_out = k_out.permute(1, 0, 2, 3).contiguous().to(kv.dtype)
    v_out = v_all.permute(1, 0, 2, 3).contiguous()
    return k_out, v_out
```
