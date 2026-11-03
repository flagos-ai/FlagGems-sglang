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

"""Reference implementation for moe/ep_scatter."""

import torch


def reference(
    recv_x,
    recv_x_scale,
    recv_topk,
    num_recv_tokens_per_expert,
    num_valid_tokens_per_expert,
    expert_start_loc,
    output_tensor,
    output_tensor_scale,
    m_indices,
    output_index,
):
    num_experts = num_recv_tokens_per_expert.shape[0]
    t, topk = recv_topk.shape

    # Pass 1: padded-per-expert start offsets, and the m_indices map (expert id
    # for the valid rows of each expert's slab, -1 for its padding tail).
    counts = num_recv_tokens_per_expert.to(torch.int64)
    starts = torch.cumsum(counts, 0) - counts
    m_out = m_indices.clone()
    for e in range(num_experts):
        s = int(starts[e])
        pad = int(counts[e])
        valid = int(num_valid_tokens_per_expert[e])
        idx = torch.arange(pad, device=m_indices.device)
        m_out[s : s + pad] = torch.where(
            idx < valid,
            torch.full_like(idx, e, dtype=m_out.dtype),
            torch.full_like(idx, -1, dtype=m_out.dtype),
        )

    # Pass 2: every (token, slot) with a non-negative expert claims the next
    # free row of that expert's slab. The kernel claims via atomic_add, so the
    # order inside an expert is not deterministic; this reference fills in
    # token-major order and the check compares each expert's range as a
    # multiset (same convention as moe/moe_align_block_size).
    cursor = starts.clone()
    out = output_tensor.clone()
    out_s = output_tensor_scale.clone()
    out_idx = output_index.clone()
    ids = recv_topk.to(torch.int64)
    for ti in range(t):
        for j in range(topk):
            e = int(ids[ti, j])
            if e < 0:
                continue
            dest = int(cursor[e])
            cursor[e] += 1
            out[dest] = recv_x[ti]
            if recv_x_scale is not None:
                out_s[dest] = recv_x_scale[ti]
            out_idx[ti, j] = dest

    return cursor.to(expert_start_loc.dtype), out, out_s, m_out, out_idx
