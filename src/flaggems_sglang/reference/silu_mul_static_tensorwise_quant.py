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

"""Reference for moe/silu_mul_static_tensorwise_quant."""

import torch
import torch.nn.functional as F


def reference(
    input,
    output,
    scale,
    num_tokens_tensor,
    expected_num_tokens,
    intermediate_size,
):
    # The live token count lives on device, which is what keeps the kernel
    # CUDA-graph safe; the grid is sized from expected_num_tokens.
    out = output.clone()
    n = int(num_tokens_tensor)
    gate = input[:n, :intermediate_size].float()
    up = input[:n, intermediate_size:].float()
    val = F.silu(gate) * up / float(scale)
    out[:n] = val.to(torch.float8_e4m3fn)
    return out
