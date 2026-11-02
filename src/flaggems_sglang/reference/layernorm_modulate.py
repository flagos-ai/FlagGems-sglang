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

"""Reference implementation for diffusion/layernorm_modulate."""

import torch


def reference(x, scale, shift, eps):
    # Keep LayerNorm and modulation in fp32; round only the final output.
    ln = torch.nn.functional.layer_norm(
        x.to(torch.float32), (x.shape[-1],), eps=eps
    )
    out = ln * (1.0 + scale.to(torch.float32).unsqueeze(1))
    out = out + shift.to(torch.float32).unsqueeze(1)
    return out.to(x.dtype)
