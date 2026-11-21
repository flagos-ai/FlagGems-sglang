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

"""Reference for moe/silu_and_mul_masked_post_per_tensor_quant."""

import torch

_FP8_DTYPE = torch.float8_e4m3fn


def reference(input, masked_m, scale):
    num_experts, max_tokens, hidden2 = input.shape
    hidden = hidden2 // 2
    fp8_max = torch.finfo(_FP8_DTYPE).max
    fp8_min = -fp8_max
    # The baseline always reads element 0: one scalar for the whole call.
    s = scale.reshape(()).float()

    out = torch.zeros(
        num_experts, max_tokens, hidden, dtype=_FP8_DTYPE, device=input.device
    )
    for e in range(num_experts):
        n = int(masked_m[e].item())
        if n <= 0:
            continue
        gate = input[e, :n, :hidden].float()
        up = input[e, :n, hidden:].float()
        val = gate * torch.sigmoid(gate) * up
        out[e, :n] = (val / s).clamp(fp8_min, fp8_max).to(_FP8_DTYPE)

    return out
