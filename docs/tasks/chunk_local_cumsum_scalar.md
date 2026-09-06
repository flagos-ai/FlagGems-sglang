# chunk_local_cumsum_scalar (fla/chunk_local_cumsum_scalar)

## 任务描述

Per-chunk (not global) cumulative sum of a scalar per-token, per-head gate
value — the FLA-family building block that turns raw log-decay values into
within-chunk cumulative decays for gated linear attention.

## 接口签名

```python
def reference(g, chunk_size, reverse=False, scale=None):
```

> 选手实现的函数签名需与上述 `reference(...)` 完全一致。

## 计算定义

- Reshape `g` to `(B, NT, BT, H)` (`NT = T // BT` chunks of width `BT =
  chunk_size`); cumsum along the within-chunk axis (reversed first if
  `reverse=True`, then flipped back), scaled by `scale` if given.
- Scope: `head_first=False` layout, `cu_seqlens=None` (fixed-size batching
  only); `T` is always an exact multiple of `chunk_size` (power of 2).

## 正确性判别标准

Per-dtype tolerance:
- float32: `atol=1e-4, rtol=1e-4`
- bfloat16: `atol=1.5e-2, rtol=1.5e-2`
- float16: `atol=1e-2, rtol=1e-2`


## 参考实现

```python
def reference(g, chunk_size, reverse=False, scale=None):
    B, T, H = g.shape
    BT = chunk_size
    NT = T // BT

    g_c = g.float().view(B, NT, BT, H)
    if reverse:
        g_c = g_c.flip(2)
    out = g_c.cumsum(dim=2)
    if scale is not None:
        out = out * scale
    if reverse:
        out = out.flip(2)

    return out.reshape(B, T, H)
```
