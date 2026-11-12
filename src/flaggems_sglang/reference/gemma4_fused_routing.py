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

"""Reference implementation for sampling/gemma4_fused_routing."""

import torch


def reference(gating_output, per_expert_scale, topk):
    logits = gating_output.to(torch.float32)
    scale = per_expert_scale.to(torch.float32)
    # Descending order, ties preferring the smaller expert id: a stable
    # descending sort gives exactly that.
    vals, ids = torch.sort(logits, dim=-1, descending=True, stable=True)
    vals = vals[:, :topk]
    ids = ids[:, :topk]
    weights = torch.softmax(vals, dim=-1) * scale[ids]
    return weights.to(torch.float32), ids.to(torch.int32)
