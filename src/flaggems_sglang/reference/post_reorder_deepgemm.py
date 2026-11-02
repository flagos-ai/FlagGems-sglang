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
    down_output,
    output,
    src2dst,
    topk_ids,
    topk_weights,
    topk,
    num_tokens,
    hidden_size,
    routed_scaling_factor,
):
    # The validity gate is expert_id >= 0, not != num_local_experts: the
    # DeepGEMM path marks padding with -1 and puts the fused shared expert at
    # num_experts, which the CUTLASS gate would silently drop.
    acc = torch.zeros(
        num_tokens, hidden_size, dtype=torch.float32, device=output.device
    )
    for i in range(topk):
        valid = (topk_ids[:, i] >= 0).float()
        rows = down_output[src2dst[:, i].clamp(min=0).long()].float()
        acc += rows * (topk_weights[:, i].float() * valid)[:, None]
    return (acc * routed_scaling_factor).to(output.dtype)
