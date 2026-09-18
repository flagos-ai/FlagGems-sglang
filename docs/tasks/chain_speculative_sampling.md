# chain_speculative_sampling (sampling_grammar/chain_speculative_sampling)

## 任务描述

链式推测采样（Chain Speculative Sampling）：对每个批次请求，逐步验证 draft 模型生成的候选 token，根据 target 与 draft 概率之比接受或拒绝每个位置，最终用修正后的分布采样一个额外 token，输出接受的 token 序列、接受位置索引及每条请求的接受数量。

## 接口签名

```python
def chain_speculative_sampling(
    candidates,
    retrive_index,
    uniform_samples,
    uniform_samples_for_final_sampling,
    target_probs,
    draft_probs,
    num_slots,
)
```

> 选手实现的函数签名需与上述 `chain_speculative_sampling(...)` 完全一致。

## 计算定义

输入张量说明：
- `candidates`: `[B, S]` int 张量，每行为一条请求的候选 token 序列（含根节点）
- `retrive_index`: `[B, S]` int 张量，每个位置对应全局 slot 索引
- `uniform_samples`: `[B, S-1]` float 张量，用于逐步接受/拒绝的均匀随机数
- `uniform_samples_for_final_sampling`: `[B]` float 张量，用于最终 token 采样的随机数
- `target_probs`: `[B, S, V]` float 张量，target 模型在每个位置上的全词表概率分布
- `draft_probs`: `[B, S-1, V]` float 张量，draft 模型在每个位置上的全词表概率分布
- `num_slots`: int，全局 slot 总数

输出：
- `predicts`: `[num_slots]` int 张量，写入被接受 token 及最终采样 token
- `accept_index`: `[B, S]` int 张量，各请求被接受位置的 slot 索引（未使用位置为 -1）
- `accept_token_num`: `[B]` int32 张量，各请求实际接受的 draft token 数量（不含最终 token）

逐步接受/拒绝规则（对每条请求 `b`，从第 1 步到第 S-1 步）：

设第 `step` 位置的 draft token 为 `t`，则：

$$\text{接受条件}: \quad u \cdot q(t) < p(t)$$

其中 $u$ 为 `uniform_samples[b, step-1]`，$p(t)$ 为 `target_probs[b, cur\_row, t]`，$q(t)$ 为 `draft_probs[b, cur\_row, t]`。

若接受，则继续验证下一步；若拒绝，则终止。

最终 token 采样：
- 若全部接受（`all_accepted=True`），修正分布为 $\text{val} = p$
- 否则，修正分布为 $\text{val} = \max(p - q, 0)$（逐元素）

对修正分布归一化后，用 `uniform_samples_for_final_sampling[b]` 通过逆 CDF（累积求和后取首个超过阈值 $u \cdot \sum \text{val}$ 的索引）采样最终 token，写入 `predicts[last_slot]`。

## 正确性判别标准

输出为整数 token id / 索引，要求精确匹配（确定性输出）：
- float32: `atol=0`
- bfloat16: `atol=0`
- float16: `atol=0`

## 参考实现

```python
import torch


def reference(
    candidates,
    retrive_index,
    uniform_samples,
    uniform_samples_for_final_sampling,
    target_probs,
    draft_probs,
    num_slots,
):
    B, S = candidates.shape
    V = target_probs.shape[-1]

    predicts = torch.zeros(num_slots, dtype=candidates.dtype, device=candidates.device)
    accept_index = torch.full(
        (B, S), -1, dtype=retrive_index.dtype, device=candidates.device
    )
    accept_token_num = torch.zeros(B, dtype=torch.int32, device=candidates.device)

    for b in range(B):
        root = int(retrive_index[b, 0].item())
        accept_index[b, 0] = root
        last_slot = root
        cur_row = 0
        num_accept = 0
        step = 1
        all_accepted = True

        while step < S:
            draft_token = int(candidates[b, step].item())
            p = target_probs[b, cur_row, draft_token]
            q = draft_probs[b, cur_row, draft_token]
            coin = uniform_samples[b, step - 1]
            if coin * q < p:
                num_accept += 1
                predicts[last_slot] = draft_token
                cur_row = step
                curr_slot = int(retrive_index[b, step].item())
                accept_index[b, num_accept] = curr_slot
                last_slot = curr_slot
                step += 1
            else:
                all_accepted = False
                break
        accept_token_num[b] = num_accept

        coin_final = uniform_samples_for_final_sampling[b]
        p_row = target_probs[b, cur_row]
        if all_accepted:
            val = p_row.clone()
        else:
            q_row = torch.nan_to_num(draft_probs[b, cur_row], nan=0.0)
            val = (p_row - q_row).clamp(min=0.0)

        norm_sum = val.sum()
        target_u = coin_final * norm_sum
        cumsum = torch.cumsum(val, dim=0)
        match = cumsum > target_u
        final_token = int(match.float().argmax().item()) if match.any() else V - 1
        predicts[last_slot] = final_token

    return predicts, accept_index, accept_token_num
```
