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

"""Reference for silu_mul_static_tensorwise_quant."""

import torch


def reference(
    input,
    output,
    scale,
    num_tokens_tensor,
    expected_num_tokens,
    intermediate_size,
):
    intermediate_size = int(intermediate_size)
    rows = int(expected_num_tokens)

    if rows == 0:
        return output.clone()

    gate = input[:rows, :intermediate_size].float()
    up = input[:rows, intermediate_size:].float()
    # Same expression order as the kernel: silu as one precise division, then
    # the multiply, then the division by the static scale. A reciprocal
    # multiply anywhere in this chain costs an ulp and can move an element
    # across an FP8 rounding boundary.
    activation = gate / (1.0 + torch.exp(-gate)) * up
    scaled = (activation / scale.float().reshape(())).to(output.dtype)

    # The active-prefix selection is done on the device: reading the token count
    # back to the host would serialise the whole call behind a copy and make any
    # timing of this reference meaningless.
    active = (
        torch.arange(rows, device=input.device) < num_tokens_tensor.reshape(())
    ).reshape(rows, 1)
    quantized = output.clone()
    # Byte-level select, because `where` does not accept the FP8 dtypes.
    selected = torch.where(
        active,
        scaled.view(torch.uint8),
        quantized.view(torch.uint8),
    )
    return selected.view(output.dtype)
