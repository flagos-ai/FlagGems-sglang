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

"""Reference for moe/silu_and_mul_masked_post_quant."""

import torch

_FP8_DTYPE = torch.float8_e4m3fn


def reference(input, quant_group_size, masked_m):
    num_experts, max_tokens, hidden2 = input.shape
    hidden = hidden2 // 2
    groups = hidden // quant_group_size
    fp8_max = torch.finfo(_FP8_DTYPE).max
    fp8_min = -fp8_max

    out_q = torch.zeros(
        num_experts, max_tokens, hidden, dtype=_FP8_DTYPE, device=input.device
    )
    out_s = torch.zeros(
        num_experts,
        max_tokens,
        groups,
        dtype=torch.float32,
        device=input.device,
    )

    for e in range(num_experts):
        # Only rows below this expert's masked_m are defined.
        n = int(masked_m[e].item())
        if n <= 0:
            continue
        gate = input[e, :n, :hidden].float()
        up = input[e, :n, hidden:].float()
        val = gate * torch.sigmoid(gate) * up
        val_g = val.view(n, groups, quant_group_size)

        absmax = val_g.abs().amax(dim=-1).clamp(min=1e-10)
        scale = absmax / fp8_max
        q = (
            (val_g / scale.unsqueeze(-1))
            .clamp(fp8_min, fp8_max)
            .to(_FP8_DTYPE)
        )

        out_q[e, :n] = q.view(n, hidden)
        out_s[e, :n] = scale

    return out_q, out_s
