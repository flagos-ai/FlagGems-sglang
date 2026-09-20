# state_passing (mamba/state_passing)

## 任务描述

Mamba2 SSD 分块扫描的状态传递阶段：对多个时间块执行跨块的顺序 SSM 状态递推，将每块末尾的衰减因子和状态增量依次累积，输出每块的初始状态快照和最终状态。

## 接口签名

```python
def reference(states, dA_cumsum, initial_states=None)
```

> 选手实现的函数签名需与上述 `reference(...)` 完全一致。

## 计算定义

输入：
- `states`: `[B, nchunks, nheads, dim]` — 每块的 SSM 状态增量
- `dA_cumsum`: `[B, nheads, nchunks, L]` — 每块的对数衰减因子累积和，L 为块内长度
- `initial_states`: `[B, nheads, dim]` float32 或 None — 序列起始状态（默认为零）

计算步骤（float32 精度）：

1. 初始化运行状态：
   - 若 `initial_states` 为 None：`cur = zeros([B, nheads, dim])`
   - 否则：`cur = initial_states.float().clone()`

2. 提取各块末尾衰减因子（取每块最后一个时间步的累积对数衰减）：
   `dA_last[b, c, h] = dA_cumsum[b, h, c, -1]`，形状调整为 `[B, nchunks, nheads]`

3. 顺序块间递推（对 c = 0, 1, ..., nchunks-1）：
   - 记录当前状态：`out[:, c] = cur.to(states.dtype)`
   - 计算块衰减：`decay[b, h] = exp(dA_last[b, c, h])`，形状 `[B, nheads, 1]`
   - 更新运行状态：`cur = cur * decay + states[:, c].float()`
   — 即 `cur_{c+1} = cur_c * exp(dA[c, -1]) + states[c]`，为跨块的离散 SSM 递推

4. 最终状态：`final_states = cur`（float32）

输出：`(out, final_states)`
- `out`: `[B, nchunks, nheads, dim]`，与 `states` 同 dtype，为每块的输入状态（非更新后状态）
- `final_states`: `[B, nheads, dim]`，float32

## 正确性判别标准

Per-dtype tolerance:
- float32: `atol=1e-4, rtol=1e-4`
- bfloat16: `atol=1.5e-2, rtol=1.5e-2`
- float16: `atol=1e-2, rtol=1e-2`

## 参考实现

```python
import torch


def reference(states, dA_cumsum, initial_states=None):
    batch, nchunks, nheads, dim = states.shape

    if initial_states is None:
        cur = states.new_zeros(batch, nheads, dim, dtype=torch.float32)
    else:
        cur = initial_states.float().clone()

    out = torch.empty(batch, nchunks, nheads, dim, device=states.device, dtype=states.dtype)
    states_f = states.float()
    dA_last = dA_cumsum[..., -1].float().permute(0, 2, 1)

    for c in range(nchunks):
        out[:, c] = cur.to(states.dtype)
        decay = torch.exp(dA_last[:, c]).unsqueeze(-1)
        cur = cur * decay + states_f[:, c]

    final_states = cur
    return out, final_states
```
