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

"""Pure-torch reference implementation of moe/sigmoid_gate_topk_renorm."""

import torch


def reference(logits, k, n_shared_experts, route_scale, global_scale, bias):
    M, G = logits.shape
    N = G - n_shared_experts
    S = n_shared_experts

    logits_f = logits.float()
    routed_logits = logits_f[:, :N]
    sel = torch.sigmoid(routed_logits) + bias.float()[None, :]

    _, idx = torch.topk(sel, k, dim=-1)
    routed_vals = torch.gather(routed_logits, 1, idx)
    shared_vals = logits_f[:, N : N + S]

    active = torch.cat([routed_vals, shared_vals], dim=-1)
    probs = torch.sigmoid(active)
    weights = probs / probs.sum(dim=-1, keepdim=True)
    weights = weights * route_scale * global_scale.float()

    routed_w = weights[:, :k].to(logits.dtype)
    shared_w = weights[:, k:].to(logits.dtype)
    indices = idx.to(torch.int32)
    return routed_w, indices, shared_w
