# Copyright 2026, The FlagOS Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied. See the License for the specific language governing
# permissions and limitations under the License.

"""Reference for kv_cache/reduce_head_scores.

Kept as the plain Torch expression on purpose: on CUDA, ``sum(dim=1)`` over
32 rows of bf16 goes through ``Reduce.cuh`` with ``vt0 = 4``, which is the
exact order the kernel reproduces. Writing the reduction out by hand here
would only introduce a second order to disagree with.

The relu is applied to the scores, not to the product, so a negative score
still contributes nothing regardless of its weight sign.
"""

import torch


def reference(scores: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    return (scores.relu() * weights.unsqueeze(-1)).sum(dim=1).float()
