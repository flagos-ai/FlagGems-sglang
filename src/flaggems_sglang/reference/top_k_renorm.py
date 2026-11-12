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

"""Reference for sampling/top_k_renorm.

The cutoff is the k-th largest value of the row, and the mask is ``probs >=
cutoff`` rather than a rank comparison, so a row with ties at the cutoff keeps
more than k entries. This mirrors sglang's ``top_k_renorm_probs``, whose mask
is ``torch.where(probs >= pivot, probs, 0)`` after taking ``pivot`` from a
descending sort at index ``k - 1``.
"""

import torch


def reference(probs, top_k):
    probs_fp32 = probs.float()
    batch, vocab = probs_fp32.shape
    if batch == 0 or vocab == 0:
        return probs_fp32

    if isinstance(top_k, torch.Tensor):
        ks = top_k.to(device=probs.device, dtype=torch.int64).reshape(-1)
        if ks.numel() == 1:
            ks = ks.expand(batch)
    else:
        ks = torch.full(
            (batch,), int(top_k), device=probs.device, dtype=torch.int64
        )

    sorted_probs = torch.sort(probs_fp32, dim=-1, descending=True).values
    cutoff = (ks - 1).clamp(min=0, max=vocab - 1)
    pivots = sorted_probs.gather(1, cutoff.unsqueeze(1)).squeeze(1)

    kept = torch.where(
        probs_fp32 >= pivots.unsqueeze(1),
        probs_fp32,
        torch.zeros_like(probs_fp32),
    )
    sums = kept.sum(dim=-1, keepdim=True)
    # A row with no retained mass would divide zero by zero; leaving it at zero
    # is what the kernel does too, and it keeps the two definitions aligned.
    safe = torch.where(sums > 0, sums, torch.ones_like(sums))
    return kept / safe
