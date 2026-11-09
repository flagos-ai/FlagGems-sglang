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

"""Reference implementation for lora/step_a_v."""

import torch


def reference(attn_output, A_buf, batch_info):
    s, h, _ = attn_output.shape
    rank = A_buf.shape[1]
    out = attn_output.new_zeros((s, h, rank))

    seg_indptr = batch_info.seg_indptr
    weight_indices = batch_info.weight_indices
    lora_ranks = batch_info.lora_ranks
    permutation = batch_info.permutation

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
            rows = torch.arange(start, end, device=attn_output.device)

        x_seg = attn_output[rows].float()
        # Same A_buf as step_b_q, but contracted along kv_lora_rank (the
        # transposed direction) rather than along rank. A is shared across
        # heads, so one weight serves every head.
        w = A_buf[w_idx].float().t()
        for head in range(h):
            out[rows, head] = (x_seg[:, head] @ w).to(out.dtype)

    return out
