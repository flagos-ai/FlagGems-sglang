# act_and_mul (moe/act_and_mul)

## 任务描述

MoE 路径下的门控激活算子：对 `gateup_output` 拆成 gate 和 up 两半，对 gate 施加激活函数后与 up 逐元素相乘，输出到独立缓冲区。支持 SiLU 和 GELU（tanh 近似），以及可选的 SwiGLU 输入裁剪。本题仅涵盖无专家过滤路径（`topk_ids=None, expert_ids=None`），每行均参与计算。

## 接口签名

```python
def act_and_mul(gateup_output, activation="silu", swiglu_limit=None):
```

> 选手实现的函数签名需与上述 `act_and_mul(...)` 完全一致。

## 计算定义

- `gateup_output`: `[M, 2H]` 张量，`gate = gateup_output[:, :H]`，`up = gateup_output[:, H:]`
- 若 `swiglu_limit` 不为 None：`gate = clamp(gate, max=swiglu_limit)`，`up = clamp(up, -swiglu_limit, swiglu_limit)`
- 激活函数：`silu(x) = x * sigmoid(x)`；`gelu(x)` 使用 tanh 近似
- `out = activation(gate) * up`，其中激活（及可选 clamp）在 float32 精度下计算，随后 gate 和 up 均 cast 回输入 dtype，逐元素相乘在输入 dtype 下完成
- 输出 shape：`[M, H]`

## 正确性判别标准

Per-dtype tolerance:
- float32: `atol=1e-4, rtol=1e-4`
- bfloat16: `atol=1.5e-2, rtol=1.5e-2`
- float16: `atol=1e-2, rtol=1e-2`


## 参考实现

```python
import torch.nn.functional as F


def reference(gateup_output, activation="silu", swiglu_limit=None):
    hidden_size = gateup_output.shape[1]
    half = hidden_size // 2
    gate = gateup_output[:, :half].float()
    up = gateup_output[:, half:].float()

    if swiglu_limit is not None:
        gate = gate.clamp(max=swiglu_limit)
        up = up.clamp(min=-swiglu_limit, max=swiglu_limit)

    if activation == "silu":
        act = F.silu(gate)
    elif activation == "gelu":
        act = F.gelu(gate, approximate="tanh")
    else:
        raise ValueError(f"Unsupported activation: {activation}")

    out = (act.to(gateup_output.dtype) * up.to(gateup_output.dtype)).to(gateup_output.dtype)
    return out
```
