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

"""Reference implementation for moe/swiglu_oai_split."""

import torch


def reference(gate_up, alpha, beta, limit, out_dtype):
    two_i = gate_up.shape[-1]
    n = two_i // 2
    x = gate_up.reshape(-1, two_i).to(torch.float32)
    gate, up = x[:, :n], x[:, n:]
    if limit is not None:
        # Note the asymmetry: gate is clamped above only, up on both sides.
        gate = gate.clamp(max=limit)
        up = up.clamp(min=-limit, max=limit)
    out = gate * torch.sigmoid(alpha * gate) * (up + beta)
    return out.to(out_dtype if out_dtype is not None else gate_up.dtype)
