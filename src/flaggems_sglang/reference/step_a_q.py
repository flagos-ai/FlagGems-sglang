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

"""Reference implementation for lora/step_a_q."""

import torch


def reference(q_nope, B_buf, batch_info, full_K_per_head):
    s, h, qk_nope = q_nope.shape
    rank = B_buf.shape[-1]
    out = q_nope.new_zeros((s, h, rank))

    seg_indptr = batch_info.seg_indptr
    weight_indices = batch_info.weight_indices
    lora_ranks = batch_info.lora_ranks
    permutation = batch_info.permutation

    # B_buf stacks every head's rows; each head block is full_K_per_head rows
    # of which only the leading qk_nope (the K half) are read here.
    weights = B_buf.reshape(B_buf.shape[0], h, full_K_per_head, rank)

    for b in range(batch_info.bs):
        start = int(seg_indptr[b].item())
        end = int(seg_indptr[b + 1].item())
        if start == end:
            continue
        w_idx = int(weight_indices[b].item())
        if int(lora_ranks[w_idx].item()) == 0:
            continue
        if permutation is not None:
            rows = permutation[start:end].long()
        else:
            rows = torch.arange(start, end, device=q_nope.device)

        x_seg = q_nope[rows].float()
        for head in range(h):
            w = weights[w_idx, head, :qk_nope].float()
            out[rows, head] = (x_seg[:, head] @ w).to(out.dtype)

    return out
