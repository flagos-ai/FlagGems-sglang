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


def reference(probs, top_p):
    probs = probs.float().contiguous()
    batch, vocab = probs.shape
    if isinstance(top_p, torch.Tensor):
        top_ps = top_p.to(device=probs.device, dtype=torch.float32).reshape(-1)
        if top_ps.numel() == 1:
            top_ps = top_ps.expand(batch)
    else:
        top_ps = torch.full(
            (batch,), float(top_p), device=probs.device, dtype=torch.float32
        )

    # FlashInfer's threshold semantics: sort ascending, drop the prefix whose
    # cumulative mass is below 1 - p, keep every tie at the pivot.
    sorted_probs = torch.sort(probs, dim=-1).values
    cdf = torch.cumsum(sorted_probs, dim=-1)
    cutoff = torch.searchsorted(
        cdf, (1.0 - top_ps).unsqueeze(1), right=False
    ).squeeze(1)
    cutoff = cutoff.clamp(max=vocab - 1)
    pivots = sorted_probs.gather(1, cutoff.unsqueeze(1))

    kept = torch.where(probs >= pivots, probs, torch.zeros_like(probs))
    return kept / kept.sum(dim=-1, keepdim=True)
