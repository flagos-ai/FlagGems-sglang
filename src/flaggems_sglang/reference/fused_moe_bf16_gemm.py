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

"""Independent per-expert matmul oracle for padded SGLang routing."""

import torch


def reference(
    a,
    b,
    out,
    topk_weights,
    sorted_token_ids,
    expert_ids,
    num_tokens_post_padded,
    config,
    *,
    top_k,
    mul_routed_weight=False,
    fuse_swiglu=False,
    tuning_profile=None,
):
    result = torch.empty_like(out)
    flat = result.view(-1, result.shape[-1])
    count = int(num_tokens_post_padded.cpu().item())
    indices = sorted_token_ids[:count].cpu().tolist()
    experts = expert_ids.cpu().tolist()
    for block, start in enumerate(range(0, count, config["BLOCK_SIZE_M"])):
        ids = [
            i
            for i in indices[start : start + config["BLOCK_SIZE_M"]]
            if 0 <= i < topk_weights.numel()
        ]
        if not ids:
            continue
        route = torch.tensor(ids, device=a.device, dtype=torch.long)
        values = a[route // top_k].float() @ b[experts[block]].float().T
        if fuse_swiglu:
            gate, up = values.to(torch.bfloat16).chunk(2, dim=-1)
            values = gate.float() * torch.sigmoid(gate.float()) * up.float()
        elif mul_routed_weight:
            values = values * topk_weights.flatten()[route, None]
        flat[route] = values.to(out.dtype)
    return result
