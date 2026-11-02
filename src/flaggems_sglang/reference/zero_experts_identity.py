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
    expert_indices, expert_scales, num_experts, zero_expert_type, hidden_states
):
    # Any routed slot whose expert id is >= num_experts is an identity expert:
    # its contribution is the token's own hidden state times the route weight.
    zero_scales = torch.where(
        expert_indices >= num_experts,
        expert_scales,
        torch.zeros_like(expert_scales),
    )
    total = zero_scales.float().sum(dim=-1, keepdim=True)
    return (hidden_states.float() * total).to(hidden_states.dtype)
