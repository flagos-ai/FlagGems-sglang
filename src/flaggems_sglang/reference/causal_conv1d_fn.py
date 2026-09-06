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


def reference(
    x, weight, bias, query_start_loc, seq_lens_cpu, activation="silu"
):
    dim, _ = x.shape
    width = weight.shape[1]
    out = torch.zeros_like(x)

    for i in range(len(seq_lens_cpu)):
        start = int(query_start_loc[i].item())
        end = int(query_start_loc[i + 1].item())
        seg = x[:, start:end].float()
        seg_len = seg.shape[1]
        padded = F.pad(seg, (width - 1, 0))

        conv = torch.zeros(dim, seg_len, device=x.device)
        for k in range(width):
            conv += weight[:, k : k + 1].float() * padded[:, k : k + seg_len]
        if bias is not None:
            conv += bias.float().unsqueeze(-1)
        if activation in ("silu", "swish"):
            conv = conv * torch.sigmoid(conv)

        out[:, start:end] = conv.to(x.dtype)

    return out
