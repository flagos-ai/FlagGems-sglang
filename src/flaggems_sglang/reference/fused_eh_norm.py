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

import torch


def _rmsnorm(x, weight, eps):
    xf = x.float()
    return (
        xf
        * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
        * weight.float()
    )


def reference(inputs_embeds, previous_hidden, enorm_weight, hnorm_weight, eps):
    e = _rmsnorm(inputs_embeds, enorm_weight, eps)
    h = _rmsnorm(previous_hidden, hnorm_weight, eps)
    return torch.cat([e, h], dim=-1).to(inputs_embeds.dtype)
