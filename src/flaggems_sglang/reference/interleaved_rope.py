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


def reference(x, mrope_section):
    _, S, D = x.shape
    d = torch.arange(D, device=x.device)
    cond_a = (d % 3 == 1) & (d < mrope_section[1] * 3)
    cond_b = (d % 3 == 2) & (d < mrope_section[2] * 3)

    out = x[0].clone()
    out[:, cond_a] = x[1][:, cond_a]
    out[:, cond_b] = x[2][:, cond_b]
    return out
