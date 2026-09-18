# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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

    predicts = torch.zeros(
        num_slots, dtype=candidates.dtype, device=candidates.device
    )
    accept_index = torch.full(
        (B, S), -1, dtype=retrive_index.dtype, device=candidates.device
    )
    accept_token_num = torch.zeros(
        B, dtype=torch.int32, device=candidates.device
    )

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
        final_token = (
            int(match.float().argmax().item()) if match.any() else V - 1
        )
        predicts[last_slot] = final_token

    return predicts, accept_index, accept_token_num
