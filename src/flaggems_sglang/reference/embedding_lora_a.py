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
    input_ids, weights, batch_info, vocab_size, extra_embeddings=None
):
    S = input_ids.shape[0]
    rank = weights.shape[1]
    out = torch.zeros(S, rank, dtype=weights.dtype, device=weights.device)

    seg_indptr = batch_info.seg_indptr
    weight_indices = batch_info.weight_indices
    lora_ranks = batch_info.lora_ranks

    for b in range(batch_info.bs):
        start = int(seg_indptr[b].item())
        end = int(seg_indptr[b + 1].item())
        if start == end:
            continue
        w_idx = int(weight_indices[b].item())
        r = int(lora_ranks[w_idx].item())
        if r == 0:
            continue

        tokens = input_ids[start:end].long()
        is_extra = tokens >= vocab_size
        clamped = tokens.clamp(max=vocab_size - 1)
        out[start:end, :r] = weights[w_idx, :r, clamped].t()

        if extra_embeddings is not None and bool(is_extra.any()):
            extra_idx = (tokens - vocab_size).clamp(min=0)
            extra_vals = extra_embeddings[w_idx, extra_idx, :r]
            out[start:end, :r] = torch.where(
                is_extra.unsqueeze(-1), extra_vals, out[start:end, :r]
            )

    return out
