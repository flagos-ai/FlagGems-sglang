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

"""Reference for silu_mul_dynamic_tensorwise_quant."""

import torch

_EPS = 1e-10
_FP8_MAX = 448.0

# The divisors are made once per device and reused. Building one per call costs
# a host-to-device copy of a pageable scalar, which drains the stream and makes
# any timing of this reference meaningless.
_DIVISOR = {}
_EPS_TENSOR = {}


def _divisor(device):
    key = str(device)
    if key not in _DIVISOR:
        _DIVISOR[key] = torch.tensor(
            [_FP8_MAX], device=device, dtype=torch.float32
        )
        _EPS_TENSOR[key] = torch.tensor(
            [_EPS], device=device, dtype=torch.float32
        )
    return _DIVISOR[key]


def _eps(device):
    _divisor(device)
    return _EPS_TENSOR[str(device)]


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
    device = input.device

    floor = _eps(device) / _divisor(device)
    if rows == 0 or intermediate_size == 0:
        return output.clone(), floor.clone()

    gate = input[:rows, :intermediate_size].float()
    up = input[:rows, intermediate_size:].float()
    # The same expression order as the kernel, and a precise division for silu:
    # a reciprocal multiply here would cost an ulp and could move an element
    # across an FP8 rounding boundary.
    activation = gate / (1.0 + torch.exp(-gate)) * up

    # Both the prefix selection and the absmax stay on the device: reading the
    # token count back to the host would serialise the call behind a copy and
    # make any timing of this reference meaningless.
    active = (
        torch.arange(rows, device=device) < num_tokens_tensor.reshape(())
    ).reshape(rows, 1)
    counted = torch.where(active, activation, torch.zeros_like(activation))
    absmax = counted.abs().amax().clamp_min(_EPS)
    # The divisor has to be a tensor, not a Python float: torch's tensor /
    # scalar path computes through a reciprocal and lands one ulp away from
    # IEEE division, which is enough to change the quantisation grid.
    scale_out = (absmax / _divisor(device)).reshape(1)

    scaled = (activation / scale_out.reshape(())).to(output.dtype)
    quantized = output.clone()
    # Byte-level select, because `where` does not accept the FP8 dtypes.
    selected = torch.where(
        active,
        scaled.view(torch.uint8),
        quantized.view(torch.uint8),
    )
    return selected.view(output.dtype), scale_out
