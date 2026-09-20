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

"""Pure-torch reference implementation of moe/silu_and_mul_masked."""

import torch


def reference(input, masked_m):
    E, T, H = input.shape
    half = H // 2
    out = torch.zeros(E, T, half, dtype=torch.bfloat16, device=input.device)
    for e in range(E):
        n = int(masked_m[e].item())
        if n <= 0:
            continue
        gate = input[e, :n, :half].float()
        up = input[e, :n, half:].float()
        val = gate * torch.sigmoid(gate) * up
        out[e, :n] = val.to(torch.bfloat16)
    return out
