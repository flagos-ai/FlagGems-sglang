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

"""Independent native BF16 gate materialization reference."""

import torch
import torch.nn.functional as F


def reference(
    hidden,
    gate_weight,
    shared_output,
    out=None,
    *,
    gate_logits_out=None,
    gate_prob_out=None,
    launch_options=None,
):
    logits = F.linear(hidden, gate_weight).to(torch.bfloat16)
    prob = torch.sigmoid(logits).to(torch.bfloat16)
    result = (shared_output * prob).to(torch.bfloat16)
    if gate_logits_out is not None:
        gate_logits_out.copy_(logits)
    if gate_prob_out is not None:
        gate_prob_out.copy_(prob)
    if out is not None:
        out.copy_(result)
        return out
    return result
