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


def reference(input_ids, weights, batch_info, vocab_size):
    S = input_ids.shape[0]
    rank = weights.shape[1]
    out = weights.new_zeros(S, rank)

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
        r = int(lora_ranks[w_idx].item())
        if r == 0:
            continue

        rows = permutation[start:end].long()
        tokens = input_ids[rows].long()
        out[rows, :r] = weights[w_idx, :r, tokens].t()

    return out
