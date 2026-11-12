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

"""Reference implementation for activation_norm/grouped_gemma_rmsnorm."""

import torch


def reference(x, weight, group_size, eps=1e-6):
    x32 = x.to(torch.float32)
    lead, n = x32.shape[:-1], x32.shape[-1]
    ng = n // group_size
    xg = x32.reshape(-1, ng, group_size)
    var = (xg * xg).mean(dim=-1, keepdim=True)
    wg = weight.to(torch.float32).reshape(ng, group_size)
    out = xg * torch.rsqrt(var + eps) * (1.0 + wg)
    return out.reshape(*lead, n).to(x.dtype)
