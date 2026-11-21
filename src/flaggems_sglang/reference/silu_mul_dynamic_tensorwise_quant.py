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

"""Reference for moe/silu_mul_dynamic_tensorwise_quant."""

import torch


def reference(
    input,
    output,
    scale,
    num_tokens_tensor,
    expected_num_tokens,
    intermediate_size,
):
    n = int(num_tokens_tensor.item())
    fp8_max = float(torch.finfo(torch.float8_e4m3fn).max)

    rows = input[:n].to(torch.float32)
    gate, up = rows[:, :intermediate_size], rows[:, intermediate_size:]
    act = gate * torch.sigmoid(gate) * up

    # Pass 1: one scale for the whole live region.
    absmax = torch.clamp(act.abs().amax(), min=1e-10)
    new_scale = (absmax / fp8_max).reshape(scale.shape)

    # Pass 2: quantize by 1/scale. Rows past num_tokens keep their old value;
    # the kernel's loop bound is num_tokens * intermediate_size.
    out = output.clone()
    out[:n] = (act / new_scale).to(output.dtype)
    return out, new_scale.to(scale.dtype)
