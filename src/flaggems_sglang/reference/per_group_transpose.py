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


def reference(a, expert_offsets, m_alignment=1):
    m, k = a.shape
    out = torch.empty_like(a)
    flat_out = out.reshape(-1)
    num_experts = expert_offsets.numel() - 1
    for e in range(num_experts):
        start = int(expert_offsets[e].item())
        end = int(expert_offsets[e + 1].item())
        n = end - start
        if n <= 0:
            continue
        seg = a[start:end].t().contiguous().reshape(-1)
        flat_out[start * k : start * k + n * k] = seg
    return out
