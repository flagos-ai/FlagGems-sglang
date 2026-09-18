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
import torch.nn.functional as F


def reference(
    q_extend,
    k_extend,
    v_extend,
    k_buffer,
    v_buffer,
    qo_indptr,
    kv_indptr,
    kv_indices,
    max_len_extend,
):
    B = qo_indptr.size(0) - 1
    _, H_Q, D = q_extend.shape
    _, H_KV, _ = k_extend.shape
    group_size = H_Q // H_KV
    scale = 1.0 / D**0.5

    o = torch.empty_like(q_extend, dtype=torch.float32)
    for i in range(B):
        q_start, q_end = int(qo_indptr[i].item()), int(qo_indptr[i + 1].item())
        kv_start, kv_end = int(kv_indptr[i].item()), int(
            kv_indptr[i + 1].item()
        )

        prefix_indices = kv_indices[kv_start:kv_end]
        k_prefix = k_buffer[prefix_indices]
        v_prefix = v_buffer[prefix_indices]

        k_ext = k_extend[q_start:q_end]
        v_ext = v_extend[q_start:q_end]
        q_ext = q_extend[q_start:q_end]

        k_full = torch.cat([k_prefix, k_ext], dim=0).float()
        v_full = torch.cat([v_prefix, v_ext], dim=0).float()
        if group_size != 1:
            k_full = k_full.repeat_interleave(group_size, dim=1)
            v_full = v_full.repeat_interleave(group_size, dim=1)

        prefix_len = k_prefix.size(0)
        extend_len = k_ext.size(0)
        total_len = prefix_len + extend_len

        pos_keys = torch.arange(total_len, device=q_extend.device)
        t = prefix_len + torch.arange(extend_len, device=q_extend.device)
        causal_mask = pos_keys.unsqueeze(0) <= t.unsqueeze(1)

        attn_scores = (
            torch.einsum("qhd,khd->qhk", q_ext.float(), k_full) * scale
        )
        attn_scores = attn_scores.masked_fill(
            ~causal_mask.unsqueeze(1), float("-inf")
        )
        attn_weights = F.softmax(attn_scores, dim=-1)
        o[q_start:q_end] = torch.einsum("qhk,khd->qhd", attn_weights, v_full)

    return o
