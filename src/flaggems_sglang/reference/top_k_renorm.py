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

"""Reference implementation for sampling_grammar/top_k_renorm."""

import torch


def reference(probs, top_k):
    values = probs.float().contiguous()
    if isinstance(top_k, torch.Tensor):
        ks = top_k.to(device=values.device, dtype=torch.int64).reshape(-1)
        ks = ks.expand(values.shape[0])
    else:
        ks = torch.full(
            (values.shape[0],), top_k, device=values.device, dtype=torch.int64
        )
    # The threshold is the k-th largest value, not a cumulative probability.
    sorted_values = values.sort(dim=-1, descending=True).values
    pivot = sorted_values.gather(1, (ks - 1).unsqueeze(1))
    # Every value at or above the pivot is kept, so ties can leave more than
    # k non-zeros.
    kept = torch.where(values >= pivot, values, torch.zeros_like(values))
    return kept / kept.sum(dim=-1, keepdim=True)
