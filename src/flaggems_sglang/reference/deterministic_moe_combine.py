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

"""Independent ordered FP32 accumulation with a single BF16 output."""

import torch


def reference(routed, gate_logits, shared, out=None, *, launch_options=None):
    total = torch.zeros_like(shared, dtype=torch.float32)
    for k in range(routed.shape[1]):
        total = total + routed[:, k].float()
    result = (total + torch.sigmoid(gate_logits.float()) * shared.float()).to(
        shared.dtype
    )
    if out is not None:
        out.copy_(result)
        return out
    return result
