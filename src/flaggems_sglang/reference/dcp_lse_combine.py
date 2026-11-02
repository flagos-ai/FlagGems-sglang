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


def reference(recv_output, recv_lse, is_lse_base_on_e, return_lse):
    lse = recv_lse.to(torch.float32)
    # NaN and +inf shards are treated as "no contribution".
    lse = torch.where(
        torch.isnan(lse) | (lse == float("inf")), float("-inf"), lse
    )

    lse_max = lse.amax(dim=0)
    lse_max = torch.where(
        lse_max == float("-inf"), torch.zeros_like(lse_max), lse_max
    )

    centered = lse - lse_max[None]
    w = torch.exp(centered) if is_lse_base_on_e else torch.exp2(centered)
    weight_sum = w.sum(dim=0)

    acc = (recv_output.to(torch.float32) * w[..., None]).sum(
        dim=0
    ) / weight_sum[..., None]
    out = acc.to(recv_output.dtype)

    if not return_lse:
        return out, None
    log = torch.log if is_lse_base_on_e else torch.log2
    return out, (log(weight_sum) + lse_max).to(recv_lse.dtype)
