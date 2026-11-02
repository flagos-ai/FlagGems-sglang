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
    topk_weights, topk_ids, gating_output, renormalize, routed_scaling_factor
):
    k = topk_weights.shape[1]
    scores = torch.sigmoid(gating_output.float())
    w, ids = torch.topk(scores, k, dim=-1)
    if renormalize:
        w = w * routed_scaling_factor / (w.sum(dim=-1, keepdim=True) + 1e-20)
    return w.contiguous(), ids.to(torch.int32).contiguous()
