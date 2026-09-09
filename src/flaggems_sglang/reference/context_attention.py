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
import torch.nn.functional as F


def reference(q, k, v, b_start_loc, b_seq_len, max_input_len, is_causal):
    o = torch.empty_like(q, dtype=torch.float32)
    B = b_seq_len.shape[0]
    for i in range(B):
        start = int(b_start_loc[i].item())
        length = int(b_seq_len[i].item())
        end = start + length
        o[start:end] = F.scaled_dot_product_attention(
            q[start:end].permute(1, 0, 2).float(),
            k[start:end].permute(1, 0, 2).float(),
            v[start:end].permute(1, 0, 2).float(),
            is_causal=is_causal,
        ).permute(1, 0, 2)
    return o
