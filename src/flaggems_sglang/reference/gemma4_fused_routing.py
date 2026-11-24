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

"""Reference implementation for moe/gemma4_fused_routing."""

import torch


def reference(gating_output, per_expert_scale, topk):
    logits = gating_output.to(torch.float32)
    # Descending by logit, ties broken toward the lower expert id: a stable
    # sort keeps the original (ascending) id order among equal logits.
    order = torch.argsort(-logits, dim=-1, stable=True)
    top_ids = order[:, :topk]
    top_logits = torch.gather(logits, 1, top_ids)

    # The softmax denominator covers the kept logits only, not all E.
    weights = torch.softmax(top_logits, dim=-1)
    weights = weights * per_expert_scale.to(torch.float32)[top_ids]
    return weights.to(torch.float32), top_ids.to(torch.int32)
