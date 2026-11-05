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

"""Reference implementation for quantization/hadamard_transform."""

import torch


def _hadamard(n, device, dtype=torch.float32):
    h = torch.ones(1, 1, device=device, dtype=dtype)
    while h.shape[0] < n:
        h = torch.cat(
            [torch.cat([h, h], dim=1), torch.cat([h, -h], dim=1)], dim=0
        )
    return h


def reference(x, scale):
    n = x.shape[-1]
    h = _hadamard(n, x.device)
    out = x.float().reshape(-1, n) @ h.t()
    return (out * scale).reshape(x.shape).to(x.dtype)
