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

import torch.nn.functional as F


def reference(gateup_output, activation="silu", swiglu_limit=None):
    hidden_size = gateup_output.shape[1]
    half = hidden_size // 2
    gate = gateup_output[:, :half].float()
    up = gateup_output[:, half:].float()

    if swiglu_limit is not None:
        gate = gate.clamp(max=swiglu_limit)
        up = up.clamp(min=-swiglu_limit, max=swiglu_limit)

    if activation == "silu":
        act = F.silu(gate)
    elif activation == "gelu":
        act = F.gelu(gate, approximate="tanh")
    else:
        raise ValueError(f"Unsupported activation: {activation}")

    out = (act.to(gateup_output.dtype) * up.to(gateup_output.dtype)).to(
        gateup_output.dtype
    )
    return out
