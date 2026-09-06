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


def reference(A, B, topk_weights, topk_ids, top_k):
    T, K = A.shape
    E, N, _ = B.shape
    A32 = A.float()
    B32 = B.float()

    out = torch.empty(T, top_k, N, dtype=A.dtype, device=A.device)
    for t in range(T):
        for j in range(top_k):
            e = int(topk_ids[t, j].item())
            row = A32[t] @ B32[e].t()
            out[t, j] = (row * topk_weights[t, j].float()).to(A.dtype)
    return out
