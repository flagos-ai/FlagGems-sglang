# ernie45_rope_fused (rope/ernie45_rope_fused)

## 任务描述

Ernie4.5 多模态融合 RoPE 算子：h/w 交替通道布局的就地旋转位置编码。与 `mrope_fused` 的连续 t/h/w 块不同，本算子在 `section_h + section_w` 范围内按通道奇偶交替使用 h 或 w 位置，其余通道使用 t 位置。本题仅覆盖 `is_neox_style=True`（neox 风格 rotate-half）。

## 接口签名

```python
def ernie45_rope_fused(q, k, cos_sin_cache, positions, mrope_section, head_size, rotary_dim):
```

> 选手实现的函数签名需与上述 `ernie45_rope_fused(...)` 完全一致。

## 计算定义

- `q`: `[T, n_qh * head_size]`，`k`: `[T, n_kh * head_size]`
- `cos_sin_cache`: `[max_pos, rotary_dim]`；`positions`: `[3, T]` int64，行为 `[t_pos, h_pos, w_pos]`
- `mrope_section = [section_h, section_w, section_t]`，要求 `section_h == section_w`，且 `section_h + section_w + section_t == rotary_dim // 2`
- 对通道索引 `r ∈ [0, rotary_dim // 2)`：
  - 若 `r < section_h + section_w`：`r` 为偶数用 `h_pos`，`r` 为奇数用 `w_pos`
  - 否则（`r >= section_h + section_w`）：用 `t_pos`
- `cos[t, r] = cos_sin_cache[pos[t, r], r]`，`sin[t, r] = cos_sin_cache[pos[t, r], r + rotary_dim // 2]`
- 对每个 head 的前 `rotary_dim` 个通道做 rotate-half：`[x1, x2] -> [x1*cos - x2*sin, x2*cos + x1*sin]`；`rotary_dim:head_size` 通道直通
- 输出 `(q_out, k_out)`，shape 与输入相同

## 正确性判别标准

Per-dtype tolerance:
- float32: `atol=1e-4, rtol=1e-4`
- bfloat16: `atol=1.5e-2, rtol=1.5e-2`
- float16: `atol=1e-2, rtol=1e-2`


## 参考实现

```python
import torch


def _apply_rope(x, n_h, head_size, rotary_dim, cos, sin):
    num_tokens = x.shape[0]
    half_rd = rotary_dim // 2
    x = x.view(num_tokens, n_h, head_size).clone()
    x1 = x[..., :half_rd].float()
    x2 = x[..., half_rd:rotary_dim].float()
    cos_e = cos.unsqueeze(1)
    sin_e = sin.unsqueeze(1)
    new1 = x1 * cos_e - x2 * sin_e
    new2 = x2 * cos_e + x1 * sin_e
    out = torch.cat([new1.to(x.dtype), new2.to(x.dtype), x[..., rotary_dim:]], dim=-1)
    return out.view(num_tokens, n_h * head_size)


def reference(q, k, cos_sin_cache, positions, mrope_section, head_size, rotary_dim):
    num_tokens, n_q_dim = q.shape
    n_k_dim = k.shape[1]
    n_qh = n_q_dim // head_size
    n_kh = n_k_dim // head_size
    half_rd = rotary_dim // 2

    section_h, section_w, section_t = mrope_section
    assert section_h == section_w, "Ernie4.5 layout assumes section_h == section_w"
    section_hw = section_h + section_w

    tpos = positions[0].long()
    hpos = positions[1].long()
    wpos = positions[2].long()

    ridx = torch.arange(half_rd, device=q.device)
    use_hw = (ridx < section_hw).unsqueeze(0)
    use_h = ((ridx % 2) == 0).unsqueeze(0)

    pos_hw = torch.where(use_h, hpos.unsqueeze(1), wpos.unsqueeze(1))
    pos = torch.where(use_hw, pos_hw, tpos.unsqueeze(1))  # (T, half_rd)

    col = ridx.unsqueeze(0).expand(num_tokens, half_rd)
    cos = cos_sin_cache[pos, col].float()
    sin = cos_sin_cache[pos, col + half_rd].float()

    q_out = _apply_rope(q, n_qh, head_size, rotary_dim, cos, sin)
    k_out = _apply_rope(k, n_kh, head_size, rotary_dim, cos, sin)
    return q_out, k_out
```
