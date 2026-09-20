# softcap_inplace_logits (activation_norm/softcap_inplace_logits)

## 任务描述

对 logits 进行原地软截断（soft-capping）：对输入张量最后一行连续维度的每个元素应用 `tanh(x / cap) * cap`，输出与输入同 shape 同 dtype。实际算子会原地修改输入缓冲区，此处参考实现先克隆输入以保持纯函数式签名。

## 接口签名

```python
def reference(full_logits, final_logit_softcapping)
```

> 选手实现的函数签名需与上述 `reference(...)` 完全一致。

## 计算定义

- 输入 `full_logits`: `[..., N]`，任意浮点 dtype；`final_logit_softcapping`: 标量 float
- 输出：与输入同 shape 同 dtype
- 逐元素计算：

  $$\text{out}[i] = \tanh\!\left(\frac{\text{full\_logits}[i]}{\text{final\_logit\_softcapping}}\right) \times \text{final\_logit\_softcapping}$$

- 该变换将输出值软限制在 `(-final_logit_softcapping, +final_logit_softcapping)` 区间内

## 正确性判别标准

Per-dtype tolerance:
- float32: `atol=1e-4, rtol=1e-4`
- bfloat16: `atol=1.5e-2, rtol=1.5e-2`
- float16: `atol=1e-2, rtol=1e-2`

## 参考实现

```python
def reference(full_logits, final_logit_softcapping):
    return (full_logits / final_logit_softcapping).tanh() * final_logit_softcapping
```
