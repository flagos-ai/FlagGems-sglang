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

"""Reference implementation for lora/step_b_v."""

import torch


def reference(
    attn_lora_a,
    B_buf,
    batch_info,
    base_output,
    qk_nope_head_dim,
    v_head_dim,
):
    h = attn_lora_a.shape[1]
    rank = B_buf.shape[-1]
    out = base_output.clone().float()

    seg_indptr = batch_info.seg_indptr
    weight_indices = batch_info.weight_indices
    lora_ranks = batch_info.lora_ranks
    scalings = batch_info.scalings
    permutation = batch_info.permutation

    # The same B_buf step_a_q reads, offset to the V half of each head block.
    full_k = qk_nope_head_dim + v_head_dim
    weights = B_buf.reshape(B_buf.shape[0], h, full_k, rank)

    for b in range(batch_info.bs):
        start = int(seg_indptr[b].item())
        end = int(seg_indptr[b + 1].item())
        if start == end:
            continue
        w_idx = int(weight_indices[b].item())
        if int(lora_ranks[w_idx].item()) == 0:
            continue
        scaling = float(scalings[w_idx].item())
        if permutation is not None:
            rows = permutation[start:end].long()
        else:
            rows = torch.arange(start, end, device=attn_lora_a.device)

        x_seg = attn_lora_a[rows].float()
        for head in range(h):
            w = weights[w_idx, head, qk_nope_head_dim:].float()
            out[rows, head] += scaling * (x_seg[:, head] @ w.t())

    return out.to(base_output.dtype)
