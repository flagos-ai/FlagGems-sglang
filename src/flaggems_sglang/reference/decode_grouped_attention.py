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


def reference(q, k_buffer, v_buffer, kv_indptr, kv_indices, sm_scale):
    B = kv_indptr.size(0) - 1
    _, H_Q, D = q.shape
    _, H_KV, _ = k_buffer.shape
    group_size = H_Q // H_KV

    o_ref = torch.empty(
        (B, H_Q, v_buffer.shape[-1]), dtype=torch.float32, device=q.device
    )
    for b in range(B):
        start = int(kv_indptr[b].item())
        end = int(kv_indptr[b + 1].item())
        idx = kv_indices[start:end]

        k_seq = k_buffer.index_select(0, idx)
        v_seq = v_buffer.index_select(0, idx)
        if H_KV != H_Q:
            k_seq = k_seq.repeat_interleave(group_size, dim=1)
            v_seq = v_seq.repeat_interleave(group_size, dim=1)

        q_f32 = q[b].to(torch.float32)
        k_f32 = k_seq.to(torch.float32)
        v_f32 = v_seq.to(torch.float32)

        logits = torch.einsum("hd,lhd->hl", q_f32, k_f32) * float(sm_scale)
        logits = logits - logits.max(dim=-1, keepdim=True).values
        p = torch.softmax(logits, dim=-1)
        o_ref[b] = torch.einsum("hl,lhd->hd", p, v_f32)

    return o_ref
