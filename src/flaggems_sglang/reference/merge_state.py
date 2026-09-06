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


def reference(prefix_output, prefix_lse, suffix_output, suffix_lse):
    p_lse = torch.where(
        prefix_lse == float("inf"),
        torch.full_like(prefix_lse, float("-inf")),
        prefix_lse,
    ).float()
    s_lse = torch.where(
        suffix_lse == float("inf"),
        torch.full_like(suffix_lse, float("-inf")),
        suffix_lse,
    ).float()

    max_lse = torch.maximum(p_lse, s_lse)
    p_lse = p_lse - max_lse
    s_lse = s_lse - max_lse
    p_se = torch.exp(p_lse)
    s_se = torch.exp(s_lse)
    out_se = p_se + s_se

    output_lse = (torch.log(out_se) + max_lse).to(prefix_lse.dtype)

    p_scale = (p_se / out_se).unsqueeze(-1)
    s_scale = (s_se / out_se).unsqueeze(-1)
    output = (
        prefix_output.float() * p_scale + suffix_output.float() * s_scale
    ).to(prefix_output.dtype)
    return output, output_lse
