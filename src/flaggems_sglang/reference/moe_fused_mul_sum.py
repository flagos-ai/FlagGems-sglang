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
    inputs,
    topk_weights,
    topk_ids=None,
    expert_map=None,
    routed_scaling_factor=None,
    is_ep=False,
):
    scale = 1.0 if routed_scaling_factor is None else routed_scaling_factor
    w = topk_weights.float() * scale

    if expert_map is not None:
        valid = expert_map[topk_ids.long()] >= 0
        w = w * valid.to(w.dtype)
    elif is_ep:
        valid = topk_ids >= 0
        w = w * valid.to(w.dtype)

    weighted = inputs.float() * w.unsqueeze(-1)
    out = weighted.sum(dim=1)
    return out.to(inputs.dtype)
