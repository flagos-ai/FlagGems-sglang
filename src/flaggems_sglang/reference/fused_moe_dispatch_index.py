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


def reference(topk_ids, num_local_experts, m_max):
    flat = topk_ids.reshape(-1)
    device = flat.device
    masked_m = torch.zeros(num_local_experts, dtype=torch.int32, device=device)
    src2dst = torch.empty(flat.numel(), dtype=torch.int32, device=device)
    counts = [0] * num_local_experts
    flat_cpu = flat.tolist()
    dst = []
    for e in flat_cpu:
        if e < 0:
            dst.append(0)
            continue
        dst.append(e * m_max + counts[e])
        counts[e] += 1
    src2dst.copy_(torch.tensor(dst, dtype=torch.int32, device=device))
    masked_m.copy_(torch.tensor(counts, dtype=torch.int32, device=device))
    return masked_m, src2dst
