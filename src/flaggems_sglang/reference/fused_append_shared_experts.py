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

"""Reference implementation for moe/fused_append_shared_experts."""

import torch


def reference(
    topk_ids, topk_weights, num_fused_shared_experts, scale_factor, N
):
    m, k = topk_ids.shape
    s = int(num_fused_shared_experts)
    if s <= 0:
        return topk_ids, topk_weights

    shared_ids = (
        torch.arange(N, N + s, device=topk_ids.device, dtype=topk_ids.dtype)
        .reshape(1, s)
        .expand(m, s)
    )
    shared_w = torch.full(
        (m, s),
        scale_factor,
        dtype=topk_weights.dtype,
        device=topk_weights.device,
    )
    return (
        torch.cat([topk_ids, shared_ids], dim=1).contiguous(),
        torch.cat([topk_weights, shared_w], dim=1).contiguous(),
    )
