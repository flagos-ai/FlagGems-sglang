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
import torch.nn.functional as F


def reference(A_log, a, b, dt_bias, beta=1.0, threshold=20.0):
    x = a.float() + dt_bias.float()
    softplus_x = torch.where(
        beta * x <= threshold, F.softplus(x, beta=beta), x
    )
    g = -torch.exp(A_log.float()) * softplus_x
    beta_output = torch.sigmoid(b.float())
    return g.unsqueeze(0).to(torch.float32), beta_output.unsqueeze(0).to(
        torch.float32
    )
